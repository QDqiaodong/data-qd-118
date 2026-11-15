"""Minimum-cost replenishment planning from a forecast fact store snapshot.

The planner consumes one read-only cross-section of the SQLite fact store
(``origin`` + ``method``), rounds every forecast up to an integer demand, and
solves one joint integer min-cost flow over all series.  It never uses a
per-series greedy rule and never expands the problem unit by unit: each
shortest-path augmentation moves the bottleneck quantity in one step, and
shortest paths are recomputed over the whole residual graph so nodes that
become reachable only after an augmentation keep valid potentials.

Timeline (day offsets are calendar days from ``origin``, offset 0):

* order days are ``origin .. last target date``; an order placed on day ``t``
  for series ``i`` arrives on ``t + lead_i``;
* its shelf life starts on the arrival day, so the ordered batch is usable on
  target day ``k`` iff ``arrival <= k < arrival + shelf_i`` (half-open);
* an initial batch is usable on ``k`` iff ``available <= k < expires``;
* holding fee is ``holding_i`` per unit per day from the batch's available day
  (arrival day for orders) to the day the unit is consumed; discarding stock is
  free, while every demand must be met exactly on its day;
* all series share one per-day ordering capacity.

Network: source -> shared day-capacity nodes -> per-(series, order day)
purchase nodes -> per-(series, target day) demand nodes -> sink; initial
batches hang directly off the source.  All capacities and costs are exact
integers, so the min-cost flow is integral.  On success the output carries the
full residual network and the node potentials (reduced costs are nonnegative on
every residual arc).  When no flow can saturate the demand edges, the residual
reachable set and its forward cut edges form a max-flow/min-cut certificate and
the unmet quantity is reported as the shortfall.
"""
import sqlite3
import urllib.parse
from datetime import timedelta
from fractions import Fraction
from pathlib import Path

from .common import ident, number, read, require, rows
from .reconcile import _FRACTION, _parse_day

_MAX_SERIES = 20
_MAX_BATCHES = 600
_MAX_TARGET_DAYS = 30
_MAX_LEAD = 100_000
_MAX_SHELF = 100_000
_MAX_COST = 10 ** 9
_MAX_QTY = 10 ** 9
_SERIES_FIELDS = ('series lead_time_days shelf_life_days unit_cost '
                  'holding_cost')
_CAP_FIELDS = 'date capacity'
_BATCH_FIELDS = 'series available expires quantity'
_CONFIG_FIELDS = ('origin method series daily_capacity initial_batches')


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #

def _day(value, field):
    return _parse_day(value, field)


def load_config(data):
    """Validate the planner config and return a normalized, sorted structure."""
    require(data, _CONFIG_FIELDS)
    origin = _day(data['origin'], 'origin')
    method = ident(data['method'])
    entries = rows(data['series'], _MAX_SERIES)
    if not entries:
        raise ValueError('at least one series required')
    series = []
    names = set()
    for entry in entries:
        require(entry, _SERIES_FIELDS)
        sid = ident(entry['series'])
        if sid in names:
            raise ValueError('duplicate config series: ' + sid)
        names.add(sid)
        lead = number(entry['lead_time_days'], 0, _MAX_LEAD)
        shelf = number(entry['shelf_life_days'], 1, _MAX_SHELF)
        unit_cost = number(entry['unit_cost'], 0, _MAX_COST)
        holding_cost = number(entry['holding_cost'], 0, _MAX_COST)
        series.append({'series': sid, 'lead': lead, 'shelf': shelf,
                       'unit_cost': unit_cost, 'holding': holding_cost})
    series.sort(key=lambda item: item['series'])

    capacities = []
    cap_days = set()
    for entry in rows(data['daily_capacity'], _MAX_TARGET_DAYS + 1):
        require(entry, _CAP_FIELDS)
        day = _day(entry['date'], 'daily_capacity.date')
        cap = number(entry['capacity'], 0, _MAX_QTY)
        if day in cap_days:
            raise ValueError('duplicate daily_capacity date: '
                             + entry['date'])
        cap_days.add(day)
        capacities.append({'date': day, 'capacity': cap})
    if not capacities:
        raise ValueError('daily_capacity must cover every order day')
    capacities.sort(key=lambda item: item['date'])

    batches = []
    for entry in rows(data['initial_batches'], _MAX_BATCHES):
        require(entry, _BATCH_FIELDS)
        sid = ident(entry['series'])
        if sid not in names:
            raise ValueError('initial batch references unknown series: ' + sid)
        available = _day(entry['available'], 'initial_batches.available')
        expires = _day(entry['expires'], 'initial_batches.expires')
        quantity = number(entry['quantity'], 0, _MAX_QTY)
        if available < origin:
            raise ValueError('initial batch available day precedes origin')
        if expires <= available:
            raise ValueError('initial batch interval must be nonempty: '
                             'available < expires')
        batches.append({'series': sid, 'available': available,
                        'expires': expires, 'quantity': quantity,
                        'input_index': len(batches)})
    batches.sort(key=lambda item: (item['series'], item['available'],
                                   item['expires'], item['input_index']))
    for index, batch in enumerate(batches, 1):
        batch['batch'] = index
    return {'origin': origin, 'method': method, 'series': series,
            'capacities': capacities, 'batches': batches}


# --------------------------------------------------------------------------- #
# Read-only fact store cross-section
# --------------------------------------------------------------------------- #

def _connect_read_only(db_path):
    path = Path(db_path)
    if path.is_symlink():
        raise ValueError('fact store path cannot be a symlink')
    if not path.exists() or not path.is_file():
        raise ValueError('fact store does not exist')
    uri = 'file:' + urllib.parse.quote(str(path.resolve()), safe='/') \
        + '?mode=ro'
    try:
        # mode=ro guarantees the planner cannot modify the source store,
        # including its WAL or journal metadata.
        connection = sqlite3.connect(uri, uri=True, timeout=30.0)
    except sqlite3.Error as error:
        raise ValueError('cannot open fact store: ' + str(error)) from error
    connection.row_factory = sqlite3.Row
    return connection


def read_cross_section(db_path, origin, method, expected_series):
    """Read one (origin, method) snapshot without writing to the store."""
    connection = _connect_read_only(db_path)
    try:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if {'meta', 'facts'} - tables:
            raise ValueError('not a supported forecast fact store')
        schema = connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if schema is None or schema[0] != '1':
            raise ValueError('unsupported store schema_version')
        generation_row = connection.execute(
            "SELECT value FROM meta WHERE key='generation'").fetchone()
        if generation_row is None:
            raise ValueError('store metadata is incomplete')
        generation = number(int(generation_row[0]), 0, 2 ** 63 - 1)
        cursor = connection.execute(
            'SELECT series, target_date, forecast FROM facts'
            ' WHERE origin=? AND method=?'
            ' ORDER BY series, target_date', (origin.isoformat(), method))
        raw = [(row['series'], row['target_date'], row['forecast'])
               for row in cursor]
    finally:
        connection.close()
    if not raw:
        raise ValueError('cross-section is empty for the given origin/method')

    by_series = {}
    for sid, target_text, forecast_text in raw:
        target = _day(target_text, 'target_date')
        if not isinstance(forecast_text, str) or \
                not _FRACTION.fullmatch(forecast_text):
            raise ValueError('stored forecast must be a canonical Fraction')
        try:
            forecast = Fraction(forecast_text)
        except ZeroDivisionError as error:
            raise ValueError('stored forecast has a zero denominator') from error
        if forecast < 0:
            raise ValueError('negative forecast is rejected for series '
                             + sid + ' on ' + target_text)
        demand = -(-forecast.numerator // forecast.denominator)  # exact ceil
        if demand > _MAX_QTY:
            raise ValueError('rounded demand exceeds 10^9 for ' + sid)
        by_series.setdefault(sid, {})[target] = demand

    found = set(by_series)
    expected = set(expected_series)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(
            'cross-section series must match the config exactly'
            + ('' if not missing else '; missing ' + ','.join(missing))
            + ('' if not extra else '; unexpected ' + ','.join(extra)))

    reference = None
    for sid in sorted(by_series):
        days = sorted(by_series[sid])
        offsets = [(day - origin).days for day in days]
        if any(offset <= 0 or offset > _MAX_TARGET_DAYS for offset in offsets):
            raise ValueError('target dates must be within 30 days after origin')
        if any(b - a != 1 for a, b in zip(offsets, offsets[1:])):
            raise ValueError('each series target dates must be consecutive '
                             'calendar days')
        if reference is None:
            reference = days
        elif days != reference:
            raise ValueError('every series must share the same target dates')
    target_days = reference
    demands = {(sid, day): by_series[sid][day]
               for sid in by_series for day in by_series[sid]}
    return {'generation': generation, 'target_days': target_days,
            'demands': demands, 'fact_count': len(raw)}


# --------------------------------------------------------------------------- #
# Min-cost flow network
# --------------------------------------------------------------------------- #

class _Network:
    """Residual-capacity graph with stable integer node and edge ids."""

    def __init__(self):
        self.adj = []
        self.nodes = []
        self.edges = []

    def add_node(self, kind, **ref):
        node_id = len(self.nodes)
        self.nodes.append({'id': node_id, 'kind': kind, 'ref': ref})
        self.adj.append([])
        return node_id

    def add_edge(self, u, v, capacity, cost, kind):
        if capacity < 0 or cost < 0:
            raise ValueError('network capacities and costs must be nonnegative')
        edge_id = len(self.edges)
        self.edges.append({'id': edge_id, 'from': u, 'to': v,
                           'capacity': capacity, 'flow': 0, 'cost': cost,
                           'kind': kind})
        # Arc layout: [to, reverse arc index, residual capacity, cost, edge id]
        # A reverse arc carries ~edge_id so flow bookkeeping finds the edge.
        self.adj[u].append([v, len(self.adj[v]), capacity, cost, edge_id])
        self.adj[v].append([u, len(self.adj[u]) - 1, 0, -cost, ~edge_id])
        return edge_id


def build_network(config, snapshot):
    """Construct the deterministic integer min-cost flow network."""
    origin = config['origin']
    target_days = snapshot['target_days']
    last_offset = (target_days[-1] - origin).days
    order_days = [origin + timedelta(days=t) for t in range(last_offset + 1)]

    cap_days = {entry['date'] for entry in config['capacities']}
    if cap_days != set(order_days):
        raise ValueError('daily_capacity dates must be exactly the order days '
                         'origin through the last target date')
    capacity_by_day = {entry['date']: entry['capacity']
                       for entry in config['capacities']}

    network = _Network()
    source = network.add_node('source')
    sink = network.add_node('sink')
    params = {item['series']: item for item in config['series']}
    series_names = [item['series'] for item in config['series']]

    day_nodes = {}
    for day in order_days:
        node = network.add_node('capacity_day', date=day.isoformat())
        day_nodes[day] = node
        network.add_edge(source, node, capacity_by_day[day], 0, 'capacity')

    # Purchase nodes: one per (series, order day) that can reach a target.
    purchase = {}
    for sid in series_names:
        item = params[sid]
        for t, day in enumerate(order_days):
            arrival = t + item['lead']
            expiry = arrival + item['shelf']
            if not any(arrival <= k < expiry
                       for k in range(1, last_offset + 1)):
                continue
            node = network.add_node(
                'order', series=sid, order_date=day.isoformat(),
                arrival_date=(origin + timedelta(days=arrival)).isoformat(),
                expires_date=(origin + timedelta(days=expiry)).isoformat())
            purchase[(sid, t)] = (node, arrival, expiry)
            network.add_edge(day_nodes[day], node, _infinite(snapshot),
                             item['unit_cost'], 'purchase')

    batch_nodes = {}
    for batch in config['batches']:
        if batch['quantity'] == 0:
            batch_nodes[batch['batch']] = None
            continue
        avail_offset = (batch['available'] - origin).days
        expires_offset = (batch['expires'] - origin).days
        node = network.add_node(
            'initial_batch', batch=batch['batch'], series=batch['series'],
            available=batch['available'].isoformat(),
            expires=batch['expires'].isoformat())
        batch_nodes[batch['batch']] = (node, avail_offset, expires_offset)
        network.add_edge(source, node, batch['quantity'], 0, 'batch_cap')

    total_demand = 0
    for sid in series_names:
        item = params[sid]
        for day in target_days:
            k = (day - origin).days
            demand = snapshot['demands'][(sid, day)]
            total_demand += demand
            node = network.add_node('demand', series=sid,
                                    target_date=day.isoformat())
            if demand:
                network.add_edge(node, sink, demand, 0, 'demand')
            for t in range(last_offset + 1):
                spot = purchase.get((sid, t))
                if spot is None:
                    continue
                p_node, arrival, expiry = spot
                if arrival <= k < expiry:
                    holding = item['holding'] * (k - arrival)
                    network.add_edge(p_node, node, _infinite(snapshot),
                                     holding, 'order_supply')
            for batch in config['batches']:
                spot = batch_nodes.get(batch['batch'])
                if spot is None or batch['series'] != sid:
                    continue
                b_node, avail_offset, expires_offset = spot
                if avail_offset <= k < expires_offset:
                    holding = item['holding'] * (k - avail_offset)
                    network.add_edge(b_node, node, batch['quantity'],
                                     holding, 'batch_supply')
    return network, source, sink, total_demand


def _infinite(snapshot):
    """An edge capacity that can never bind before all demand is satisfied."""
    return sum(snapshot['demands'].values()) + 1


# --------------------------------------------------------------------------- #
# Successive shortest paths with full-graph shortest-path potentials
# --------------------------------------------------------------------------- #

def min_cost_flow(network, source, sink, required):
    """Augment bulk bottleneck quantities until saturation or a max flow.

    Each iteration computes shortest-path distances over the *original*
    residual costs with Bellman-Ford and takes those distances as the node
    potentials.  Using shortest-path distances directly (rather than reduced
    cost increments) keeps potentials valid on every node, including nodes
    that become reachable only after an earlier augmentation (for example
    those behind a zero-capacity order day): the shortest-path triangle
    inequality is exactly the condition that every residual arc has a
    nonnegative reduced cost.

    Returns ``(satisfied, potentials, reachable)`` where ``reachable`` is the
    residual source side (a min cut) when saturation is impossible.  All edge
    flows are written back into ``network.edges``.
    """
    count = len(network.nodes)
    potentials = [0 if value is None else value
                  for value in _shortest_paths(network, source)[0]]
    satisfied = 0
    reachable = [source]
    while satisfied < required:
        distances, previous = _shortest_paths(network, source)
        if distances[sink] is None:
            # No augmenting path remains: the finite-distance nodes form the
            # residual min cut.  Unreachable nodes keep zero potential: no
            # positive residual arc connects the reachable component to them,
            # so they cannot violate the reduced-cost certificate.
            reachable = _residual_closure(network, source)
            break
        potentials = [0 if value is None else value for value in distances]
        augment = required - satisfied
        v = sink
        while v != source:
            u, arc_index = previous[v]
            augment = min(augment, network.adj[u][arc_index][2])
            v = u
        if augment <= 0:
            raise ValueError('zero-capacity augmenting path')
        v = sink
        while v != source:
            u, arc_index = previous[v]
            arc = network.adj[u][arc_index]
            reverse = network.adj[arc[0]][arc[1]]
            arc[2] -= augment
            reverse[2] += augment
            edge_id = arc[4]
            if edge_id >= 0:
                network.edges[edge_id]['flow'] += augment
            else:
                network.edges[~edge_id]['flow'] -= augment
            v = u
        satisfied += augment
    return satisfied, potentials, reachable


def _shortest_paths(network, source):
    """Shortest residual paths from the source over original residual costs.

    Returns ``(distances, previous)`` with ``None`` distances on unreachable
    nodes.  Residual graphs from successive shortest paths have no negative
    cycle, so Bellman-Ford relaxation terminates with exact distances.
    """
    count = len(network.nodes)
    distances = [None] * count
    previous = [None] * count
    distances[source] = 0
    for _ in range(count - 1):
        changed = False
        for u in range(count):
            if distances[u] is None:
                continue
            for arc_index, arc in enumerate(network.adj[u]):
                v, _, capacity, cost, _ = arc
                if capacity <= 0:
                    continue
                candidate = distances[u] + cost
                if distances[v] is None or candidate < distances[v]:
                    distances[v] = candidate
                    previous[v] = (u, arc_index)
                    changed = True
        if not changed:
            break
    for u in range(count):
        if distances[u] is None:
            continue
        for arc in network.adj[u]:
            v, _, capacity, cost, _ = arc
            if capacity > 0 and distances[u] + cost < distances[v]:
                raise ValueError('residual graph has a negative cycle')
    return distances, previous


def _residual_closure(network, source):
    seen = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for arc in network.adj[u]:
            v, _, capacity, _, _ = arc
            if capacity > 0 and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #

def _node_ref(network, node_id):
    return network.nodes[node_id]['ref']


def assemble_report(config, snapshot, network, source, sink, total_demand,
                    satisfied, potentials, reachable, config_sha):
    """Turn the solved network into the deterministic output contract."""
    feasible = satisfied == total_demand
    edges = network.edges
    purchase_cost = sum(edge['flow'] * edge['cost']
                        for edge in edges if edge['kind'] == 'purchase')
    holding_cost = sum(edge['flow'] * edge['cost'] for edge in edges
                       if edge['kind'] in ('order_supply', 'batch_supply'))

    orders = []
    for edge in edges:
        if edge['kind'] != 'purchase' or edge['flow'] == 0:
            continue
        ref = _node_ref(network, edge['to'])
        orders.append({'series': ref['series'],
                       'order_date': ref['order_date'],
                       'arrival_date': ref['arrival_date'],
                       'expires_date': ref['expires_date'],
                       'quantity': edge['flow']})
    orders.sort(key=lambda item: (item['series'], item['order_date']))

    allocations = []
    for edge in edges:
        if edge['flow'] == 0 or edge['kind'] not in ('order_supply',
                                                      'batch_supply'):
            continue
        source_ref = _node_ref(network, edge['from'])
        demand_ref = _node_ref(network, edge['to'])
        if edge['kind'] == 'order_supply':
            kind, ref = 'order', source_ref['order_date']
        else:
            kind, ref = 'initial', source_ref['batch']
        allocations.append({'series': demand_ref['series'],
                            'target_date': demand_ref['target_date'],
                            'source_kind': kind, 'source_ref': ref,
                            'quantity': edge['flow']})
    allocations.sort(key=lambda item: (item['series'], item['target_date'],
                                       item['source_kind'],
                                       str(item['source_ref'])))

    batch_usage = {}
    for edge in edges:
        if edge['kind'] == 'batch_cap':
            ref = _node_ref(network, edge['to'])
            batch_usage[ref['batch']] = edge['flow']
    initial_batches = []
    for batch in config['batches']:
        initial_batches.append({
            'batch': batch['batch'], 'series': batch['series'],
            'available': batch['available'].isoformat(),
            'expires': batch['expires'].isoformat(),
            'quantity': batch['quantity'],
            'used': batch_usage.get(batch['batch'], 0)})

    demand_rows = [{'series': sid, 'target_date': day.isoformat(),
                    'demand': snapshot['demands'][(sid, day)]}
                   for sid in sorted({key[0] for key in snapshot['demands']})
                   for day in snapshot['target_days']]

    certificate = {
        'kind': 'potentials' if feasible else 'min_cut',
        'source_id': source, 'sink_id': sink,
        'nodes': [{'id': node['id'], 'kind': node['kind'], 'ref': node['ref']}
                  for node in network.nodes],
        'edges': [{'id': edge['id'], 'from': edge['from'], 'to': edge['to'],
                   'capacity': edge['capacity'], 'flow': edge['flow'],
                   'cost': edge['cost'], 'kind': edge['kind']}
                  for edge in edges],
        'potentials': {str(node_id): potentials[node_id]
                       for node_id in range(len(network.nodes))},
        'reduced_cost_rule': ('for every residual arc u->v with residual '
                              'capacity > 0 (reverse arcs included): '
                              'cost + potential[u] - potential[v] >= 0')}
    if not feasible:
        cut_edges = [{'id': edge['id'], 'from': edge['from'],
                      'to': edge['to'], 'capacity': edge['capacity'],
                      'flow': edge['flow'], 'kind': edge['kind']}
                     for edge in edges
                     if edge['from'] in reachable and edge['to'] not in reachable]
        cut_capacity = sum(edge['capacity'] for edge in cut_edges)
        certificate.update({
            'reachable': sorted(reachable),
            'cut_edges': cut_edges,
            'cut_capacity': cut_capacity,
            'max_flow': satisfied,
            'min_cut_rule': ('reachable is closed under positive residual '
                             'arcs and contains the source but not the sink; '
                             'cut_capacity equals max_flow')})

    unmet = []
    for edge in edges:
        if edge['kind'] != 'demand':
            continue
        ref = _node_ref(network, edge['from'])
        gap = edge['capacity'] - edge['flow']
        if gap:
            unmet.append({'series': ref['series'],
                          'target_date': ref['target_date'], 'unmet': gap})
    unmet.sort(key=lambda item: (item['series'], item['target_date']))

    return {
        'schema_version': 1, 'mode': 'demand_plan',
        'config_sha256': config_sha,
        'source': {'kind': 'forecast_fact_store',
                   'generation': snapshot['generation'],
                   'cross_section': {
                       'origin': config['origin'].isoformat(),
                       'method': config['method'],
                       'fact_count': snapshot['fact_count']}},
        'result': {
            'status': 'feasible' if feasible else 'infeasible',
            'origin': config['origin'].isoformat(),
            'method': config['method'],
            'generation': snapshot['generation'],
            'cross_section_fact_count': snapshot['fact_count'],
            'target_dates': [day.isoformat() for day in snapshot['target_days']],
            'series_parameters': [
                {'series': item['series'],
                 'lead_time_days': item['lead'],
                 'shelf_life_days': item['shelf'],
                 'unit_cost': item['unit_cost'],
                 'holding_cost': item['holding']}
                for item in config['series']],
            'daily_capacity': [
                {'date': item['date'].isoformat(),
                 'capacity': item['capacity']}
                for item in config['capacities']],
            'initial_batches': initial_batches,
            'demand': demand_rows,
            'orders': orders,
            'allocations': allocations,
            'purchase_cost': purchase_cost,
            'holding_cost': holding_cost,
            'total_cost': purchase_cost + holding_cost,
            'flow': {'total_demand': total_demand, 'satisfied': satisfied,
                     'shortfall': total_demand - satisfied},
            'demand_shortfall': unmet,
            'certificate': certificate}}


# --------------------------------------------------------------------------- #
# Entry point used by the CLI
# --------------------------------------------------------------------------- #

def plan(db_path, config_path):
    data, config_sha = read(config_path)
    config = load_config(data)
    snapshot = read_cross_section(
        db_path, config['origin'], config['method'],
        [item['series'] for item in config['series']])
    network, source, sink, total_demand = build_network(config, snapshot)
    satisfied, potentials, reachable = min_cost_flow(
        network, source, sink, total_demand)
    return assemble_report(config, snapshot, network, source, sink,
                           total_demand, satisfied, potentials, reachable,
                           config_sha)
