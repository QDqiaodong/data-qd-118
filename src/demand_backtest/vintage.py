"""Vintage backtests: each origin only sees revisions published by that origin."""
from bisect import bisect_right
from collections import defaultdict
from datetime import date, timedelta
from fractions import Fraction
import re

from .common import ident, number, require, rows
from .forecast import metrics, predict, ratio

_METHODS = ('moving_mean', 'naive', 'seasonal_naive')
_MAX_FACTS = 200000
_DAY = re.compile(r'\d{4}-\d{2}-\d{2}')


def _parse_day(value, field):
    if not isinstance(value, str) or not _DAY.fullmatch(value):
        raise ValueError(field + ' must be YYYY-MM-DD')
    return date.fromisoformat(value)


def _published_at(published, cutoff):
    """Latest publication date at or before cutoff, or None."""
    index = bisect_right(published, cutoff) - 1
    return published[index] if index >= 0 else None


def _first_gap(start, days):
    for offset, day in enumerate(days):
        expected = start + timedelta(days=offset)
        if day != expected:
            return expected
    return start + timedelta(days=len(days))


def _skip(skipped, sid, origin, reason, detail=None):
    skipped.append({'series': sid, 'origin': origin.isoformat(),
                    'reason': reason,
                    'detail': detail.isoformat() if detail is not None else None})


def analyze_vintage(data):
    require(data, 'revisions origins score_as_of initial_train horizon mean_window season')
    initial = number(data['initial_train'], 2, 10000)
    horizon = number(data['horizon'], 1, 100)
    window = number(data['mean_window'], 1, initial)
    season = number(data['season'], 1, initial)
    score_as_of = _parse_day(data['score_as_of'], 'score_as_of')

    # series -> business day -> {published_on: quantity}. Same key and same
    # publication date must agree; identical repeats collapse into one version.
    versions = defaultdict(dict)
    for revision in rows(data['revisions'], 20000):
        require(revision, 'series date published_on quantity')
        sid = ident(revision['series'])
        day = _parse_day(revision['date'], 'date')
        published_on = _parse_day(revision['published_on'], 'published_on')
        if day > published_on:
            raise ValueError('date must not be after published_on')
        quantity = number(revision['quantity'])
        by_publication = versions[sid].setdefault(day, {})
        if published_on in by_publication:
            if by_publication[published_on] != quantity:
                raise ValueError('conflicting revision for series date published_on')
        else:
            by_publication[published_on] = quantity
    if not versions:
        raise ValueError('at least one series required')

    dates = {sid: sorted(values) for sid, values in versions.items()}
    published = {sid: {day: sorted(versions[sid][day]) for day in days}
                 for sid, days in dates.items()}
    index_of = {sid: {day: i for i, day in enumerate(days)} for sid, days in dates.items()}

    # Origins are the caller's forecast decisions; identical rows collapse and
    # the canonical (series, date) order makes every output order deterministic.
    requested = []
    skipped = []
    seen = set()
    for row in rows(data['origins'], 20000):
        require(row, 'series date')
        sid = ident(row['series'])
        day = _parse_day(row['date'], 'origin date')
        if (sid, day) in seen:
            continue
        seen.add((sid, day))
        if sid not in versions:
            _skip(skipped, sid, day, 'unknown_series')
            continue
        requested.append((sid, day))
    requested.sort()
    if len(requested) * horizon * len(_METHODS) > _MAX_FACTS:
        raise ValueError('evaluation exceeds 200000 forecast points')

    # Version lineage: every version of every series/date, with its publication
    # date, whether it is the final revision, and whether scoring uses it.
    lineage = []
    for sid in sorted(versions):
        for day in dates[sid]:
            by_publication = versions[sid][day]
            ordered = published[sid][day]
            scored_on = _published_at(ordered, score_as_of)
            lineage.append({'series': sid, 'date': day.isoformat(), 'versions': [
                {'published_on': on.isoformat(), 'quantity': by_publication[on],
                 'latest': on == ordered[-1], 'scored': on == scored_on}
                for on in ordered]})

    points = []
    groups = defaultdict(list)
    evaluated = []
    for sid, origin in requested:
        days = dates[sid]
        first = days[0]

        # The prefix is every consecutive calendar day from the first business
        # day through the origin; each day carries the newest revision already
        # published by the origin. A gap or a missing origin day is a skip.
        if origin < first:
            _skip(skipped, sid, origin, 'insufficient_history')
            continue
        index = index_of[sid].get(origin)
        if index is None:
            gap = _first_gap(first, [d for d in days if d <= origin])
            _skip(skipped, sid, origin, 'missing_calendar_date', min(gap, origin))
            continue
        prefix = days[:index + 1]
        gap = _first_gap(first, prefix)
        if gap <= origin:
            _skip(skipped, sid, origin, 'missing_calendar_date', gap)
            continue
        if len(prefix) < initial:
            _skip(skipped, sid, origin, 'insufficient_history')
            continue

        train = []
        missing = None
        for day in prefix:
            used = _published_at(published[sid][day], origin)
            if used is None:
                missing = day
                break
            train.append(versions[sid][day][used])
        if missing is not None:
            _skip(skipped, sid, origin, 'missing_training_vintage', missing)
            continue

        # Targets are scored from the newest revision published by score_as_of.
        targets = []
        missing = None
        for step in range(1, horizon + 1):
            day = origin + timedelta(days=step)
            by_publication = published[sid].get(day)
            used = _published_at(by_publication or [], score_as_of)
            if used is None:
                missing = (day, 'missing_target' if not by_publication
                           else 'missing_target_vintage')
                break
            targets.append((day, versions[sid][day][used], used))
        if missing is not None:
            _skip(skipped, sid, origin, missing[1], missing[0])
            continue

        scale = Fraction(sum(abs(b - a) for a, b in zip(train, train[1:])),
                         len(prefix) - 1)
        predictions = predict(train, horizon, window, season)
        evaluated.append({'series': sid, 'origin': origin.isoformat(),
                          'train_start': first.isoformat(),
                          'training_count': len(prefix), 'mase_scale': ratio(scale)})
        for method, forecasts in sorted(predictions.items()):
            for step, forecast in enumerate(forecasts, 1):
                day, actual, actual_on = targets[step - 1]
                point = {'series': sid, 'origin': origin.isoformat(),
                         'target_date': day.isoformat(), 'horizon': step,
                         'method': method, 'forecast': forecast, 'actual': actual,
                         'actual_published_on': actual_on.isoformat(),
                         'scale': scale}
                points.append(point)
                groups[sid, method].append(point)
    if len(points) > _MAX_FACTS:
        raise ValueError('evaluation exceeds 200000 forecast points')

    summaries = []
    for sid in sorted(versions):
        for method in _METHODS:
            summaries.append(dict(series=sid, method=method,
                                  **metrics(groups[sid, method])))
    serial = [dict(p, forecast=ratio(p['forecast']), scale=ratio(p['scale']),
                   error_forecast_minus_actual=ratio(p['forecast'] - p['actual']))
              for p in points]
    skipped.sort(key=lambda item: (item['series'], item['origin'], item['reason']))
    return {'mode': 'revision_vintage', 'score_as_of': score_as_of.isoformat(),
            'origins': evaluated, 'lineage': lineage, 'forecasts': serial,
            'metrics': summaries, 'skipped': skipped,
            'evaluation_unit': 'origin-target forecast; overlapping targets are intentionally counted separately'}
