"""Reconcile two existing backtest reports onto one additive hierarchy.

A hierarchy is a single-rooted tree.  Every node names one series, carries a
positive rational weight, and has at most one parent.  A forecast group is
identified by ``(origin, target_date, method)``; the group must contain exactly
one forecast fact for every hierarchy node and a single horizon.  Groups from
different origins are never merged even when their target dates overlap, and
``actual`` never enters the reconciliation.

For each group we solve the exact convex program

    minimize  sum_nodes weight_i * (x_i - forecast_i)^2
    subject to x_parent = sum of its direct children,   x_leaf >= 0.

The optimum is computed jointly by :mod:`demand_backtest.tree_qp`; we never clip
negative values and then patch parents, and we never enumerate all active sets.
Output values, the objective and the leaf-gradient KKT certificate are canonical
Fraction strings.
"""
import hashlib
import json
import re
from datetime import date
from fractions import Fraction

from .common import ident, number, require, rows
from .tree_qp import objective, reconcile

_MAX_NODES = 200
_MAX_FACTS = 200000
_DAY = re.compile(r'\d{4}-\d{2}-\d{2}')
_FRACTION = re.compile(r'-?(?:0|[1-9]\d*)(?:/[1-9]\d*)?')
_WEIGHT = re.compile(r'(?:0|[1-9]\d*)(?:/[1-9]\d*)?')
_SHA = re.compile(r'[0-9a-f]{64}')
_FACT_FIELDS = ('series origin target_date horizon method forecast')
# Fields produced by the two backtest modes; `actual` and its bookkeeping are
# carried by the reports but deliberately ignored by reconciliation.
_EXTRA_FIELDS = {'actual', 'scale', 'error_forecast_minus_actual',
                 'actual_published_on'}


def _bounded(value, field):
    if isinstance(value, str) and any(len(part) > 18
                                      for part in value.split('/')):
        raise ValueError(field + ' is out of bounds')


def _fraction(value, field, positive=False):
    pattern = _WEIGHT if positive else _FRACTION
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(field + ' must be a canonical exact Fraction string')
    _bounded(value, field)
    result = Fraction(value)
    if str(result) != value:
        raise ValueError(field + ' must be a canonical exact Fraction string')
    if positive and result <= 0:
        raise ValueError(field + ' must be a positive Fraction')
    return result


def _parse_day(value, field):
    if not isinstance(value, str) or not _DAY.fullmatch(value):
        raise ValueError(field + ' must be YYYY-MM-DD')
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(field + ' must be a valid calendar day') from error


def _parse_hierarchy(data):
    require(data, 'nodes')
    items = rows(data['nodes'], _MAX_NODES)
    if not items:
        raise ValueError('hierarchy needs at least one node')
    series, parents, weights = [], [], []
    for entry in items:
        require(entry, 'series parent weight')
        sid = ident(entry['series'])
        if sid in series:
            raise ValueError('duplicate hierarchy series')
        parent = entry['parent']
        if parent is not None:
            parent = ident(parent)
            if parent == sid:
                raise ValueError('a node cannot be its own parent')
        series.append(sid)
        parents.append(parent)
        weights.append(_fraction(entry['weight'], 'weight', positive=True))
    known = set(series)
    roots = [i for i, parent in enumerate(parents) if parent is None]
    if len(roots) != 1:
        raise ValueError('hierarchy must have exactly one root')
    for parent in parents:
        if parent is not None and parent not in known:
            raise ValueError('hierarchy parent reference is unknown')
    index = {sid: i for i, sid in enumerate(series)}
    children = [[] for _ in series]
    for i, parent in enumerate(parents):
        if parent is not None:
            children[index[parent]].append(i)
    # A valid root plus valid parent references can still hide a detached cycle
    # that is unreachable from the root, so verify full reachability.
    visited, stack = 0, [roots[0]]
    while stack:
        node = stack.pop()
        visited += 1
        stack.extend(children[node])
    if visited != len(series):
        raise ValueError('hierarchy must be one tree reachable from its root')
    leaves = [i for i, kids in enumerate(children) if not kids]
    return {'series': series, 'parents': parents, 'weights': weights,
            'children': children, 'leaves': leaves, 'root': roots[0]}


def _kind(result):
    if not isinstance(result, dict):
        raise ValueError('report result must be an object')
    if result.get('mode') == 'revision_vintage':
        return 'vintage'
    if 'mode' not in result and {'origins', 'forecasts', 'metrics',
                                 'evaluation_unit'} <= set(result):
        return 'calendar'
    raise ValueError('report is neither a calendar nor a revision vintage backtest')


def _report(data):
    require(data, 'schema_version source_sha256 result')
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError('unsupported report schema_version')
    token = data['source_sha256']
    if not isinstance(token, str) or not _SHA.fullmatch(token):
        raise ValueError('report source_sha256 must be lowercase SHA256')
    kind = _kind(data['result'])
    return kind, data['source_sha256'], data['result']['forecasts']


def _groups(sources, hierarchy):
    """Pool both reports' facts by (origin, target_date, method).

    A group is one forecast decision.  Every group must carry exactly one fact
    per hierarchy node, all at one horizon, and a repeated fact is an error even
    when it comes from the other report.  Different origins stay separate even
    if their target dates coincide.
    """
    known = set(hierarchy['series'])
    required = set(_FACT_FIELDS.split())
    groups = {}
    for kind, facts in sources:
        for fact in rows(facts, _MAX_FACTS):
            if not isinstance(fact, dict):
                raise ValueError('forecast facts must be objects')
            keys = set(fact)
            if not required <= keys or not keys <= required | _EXTRA_FIELDS:
                raise ValueError('forecast fact has undefined or missing fields')
            sid = ident(fact['series'])
            origin = fact['origin']
            _parse_day(origin, 'origin')
            _parse_day(fact['target_date'], 'target_date')
            step = number(fact['horizon'], 1, 100)
            method = ident(fact['method'])
            forecast = _fraction(fact['forecast'], 'forecast')
            key = (origin, fact['target_date'], method)
            group = groups.get(key)
            if group is None:
                group = {'horizon': step, 'facts': {}, 'inputs': set()}
                groups[key] = group
            elif group['horizon'] != step:
                raise ValueError('forecasts in one group must share one horizon')
            if sid in group['facts']:
                raise ValueError('duplicate forecast fact in group '
                                 + '/'.join(key) + ' for series ' + sid)
            group['facts'][sid] = forecast
            group['inputs'].add(kind)
    full = set(known)
    for key, group in groups.items():
        present = set(group['facts'])
        if present != full:
            missing = sorted(full - present)
            extra = sorted(present - full)
            raise ValueError(
                'group ' + '/'.join(key) + ' must contain exactly the '
                'hierarchy series'
                + ('' if not missing else '; missing ' + ','.join(missing))
                + ('' if not extra else '; unknown ' + ','.join(extra)))
    return groups


def _text(value):
    return (str(value.numerator) if value.denominator == 1
            else f'{value.numerator}/{value.denominator}')


def _emit_group(key, group, hierarchy):
    series = hierarchy['series']
    # The solver works directly on the hierarchy indices; output is sorted by
    # series so the report never depends on the input file's node order.
    forecasts = [group['facts'][sid] for sid in series]
    weights = hierarchy['weights']
    kids = hierarchy['children']
    leaves = hierarchy['leaves']
    leaf_set = set(leaves)

    values, gradients, _, _ = reconcile(
        kids, hierarchy['root'], weights, forecasts, leaves)
    total = objective(weights, forecasts, values)

    nodes, certificate = [], []
    for position in sorted(range(len(series)), key=lambda i: series[i]):
        sid = series[position]
        leaf = position in leaf_set
        # A leaf is reported as zero from its exact value, so a free leaf that
        # lands exactly on the boundary is treated consistently with a leaf the
        # active set pinned there.
        nodes.append({'series': sid,
                      'parent': hierarchy['parents'][position],
                      'weight': _text(weights[position]),
                      'forecast': _text(forecasts[position]),
                      'reconciled': _text(values[position]),
                      'leaf': leaf,
                      'zero': leaf and values[position] == 0})
        if leaf:
            certificate.append({'series': sid,
                                'value': _text(values[position]),
                                'gradient': _text(gradients[position])})
    zero_leaves = sorted(entry['series'] for entry in certificate
                         if entry['value'] == '0')
    return {'inputs': sorted(group['inputs']), 'origin': key[0],
            'target_date': key[1], 'method': key[2],
            'horizon': group['horizon'],
            'nodes': nodes, 'objective': _text(total),
            'zero_leaves': zero_leaves, 'leaf_kkt': certificate}


def _canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def analyze_reconcile(reports, hierarchy, report_sha256=None,
                      hierarchy_sha256=None):
    """Reconcile exactly one calendar and one revision vintage report.

    ``reports`` is the list of the two parsed report objects; their order does
    not matter and the output does not depend on it.  ``report_sha256`` binds
    the exact input bytes when supplied (the CLI passes file digests) and falls
    back to a canonical digest of the parsed object otherwise.
    """
    if not isinstance(reports, list) or len(reports) != 2:
        raise ValueError('exactly two reports are required')
    parsed = [_report(report) for report in reports]
    if [kind for kind, _, _ in parsed].count('calendar') != 1 or \
            [kind for kind, _, _ in parsed].count('vintage') != 1:
        raise ValueError('one calendar and one revision vintage report required')

    tree = _parse_hierarchy(hierarchy)
    if report_sha256 is None:
        report_sha256 = [_canonical_digest(report) for report in reports]
    if not isinstance(report_sha256, list) or len(report_sha256) != 2 or \
            not all(isinstance(d, str) and _SHA.fullmatch(d)
                    for d in report_sha256):
        raise ValueError('report digests must be two lowercase SHA256 strings')
    if hierarchy_sha256 is None:
        hierarchy_sha256 = _canonical_digest(hierarchy)
    if not isinstance(hierarchy_sha256, str) or \
            not _SHA.fullmatch(hierarchy_sha256):
        raise ValueError('hierarchy digest must be a lowercase SHA256 string')

    total = sum(len(facts) for _, _, facts in parsed)
    if total > _MAX_FACTS:
        raise ValueError('reconciliation exceeds 200000 forecast facts')

    reports_out, groups_out = [], []
    # Sort by kind so the report, and the group sequence derived from it, does
    # not depend on the order the caller listed the two inputs in.
    for source_index, (kind, source_sha, facts) in sorted(
            enumerate(parsed), key=lambda pair: pair[1][0]):
        reports_out.append({'kind': kind, 'source_sha256': source_sha,
                            'report_sha256': report_sha256[source_index]})
    for key, group in sorted(_groups([(kind, facts)
                                      for kind, _, facts in parsed],
                                     tree).items()):
        groups_out.append(_emit_group(key, group, tree))

    bound = {report['kind']: report['report_sha256']
             for report in reports_out}
    return {'reports': reports_out,
            'hierarchy': {'nodes': len(tree['series']),
                          'root': tree['series'][tree['root']],
                          'leaves': sorted(tree['series'][i]
                                           for i in tree['leaves']),
                          'sha256': hierarchy_sha256},
            'groups': groups_out,
            'fact_count': total,
            'group_count': len(groups_out),
            'input_sha256': {'calendar': bound['calendar'],
                             'vintage': bound['vintage'],
                             'hierarchy': hierarchy_sha256},
            'aggregation': ('every reconciled parent equals the exact sum of '
                            'its direct reconciled children; zero leaves are '
                            'exact zeros'),
            'evaluation_unit': ('origin-target-method group; overlapping '
                                'targets from different origins stay separate')}
