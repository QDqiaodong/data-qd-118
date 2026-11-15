"""Contract, optimality and certificate tests for demand-plan.

The planner's own network machinery is never trusted here:

* :func:`verify_plan` re-reads the published JSON and re-checks every flow,
  capacity, demand, eligibility window, cost and the residual potential /
  min-cut certificate from scratch;
* :func:`brute_force_optimum` enumerates every bounded integer allocation on
  small instances, so optimality is checked by exhaustive search rather than
  by another flow implementation;
* :func:`series_greedy` is the explicitly forbidden per-series greedy policy
  (walk target days forward, use the currently cheapest eligible source).  The
  shared-capacity counterexample shows it losing against the joint optimum.
"""
import json
import tempfile
import unittest
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path

from demand_backtest import plan, store

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'fixtures'
ORIGIN = date(2026, 1, 1)


def canonical_fraction(value):
    return str(Fraction(value))


_STORE_SEQ = 0


def build_store(directory, rows, origin=ORIGIN, method='naive'):
    """Create a fresh one-report fact store from target-offset rows."""
    global _STORE_SEQ
    _STORE_SEQ += 1
    stamp = _STORE_SEQ
    facts = [{'series': sid, 'origin': origin.isoformat(),
              'target_date': (origin + timedelta(days=offset)).isoformat(),
              'horizon': offset, 'method': method,
              'forecast': canonical_fraction(value), 'actual': 0,
              'scale': '1'}
             for sid, offset, value in rows]
    report = {'schema_version': 1,
              'source_sha256': f'{stamp:064x}',
              'result': {'origins': [], 'forecasts': facts, 'metrics': [],
                         'evaluation_unit': 'x'}}
    report_path = directory / f'report-{stamp}.json'
    report_path.write_text(json.dumps(report))
    manifest = directory / f'manifest-{stamp}.json'
    manifest.write_text(json.dumps(
        {'txid': f'seed-{stamp}', 'expected_generation': 0,
         'puts': [{'name': f'report-{stamp}',
                   'path': str(report_path)}], 'drops': []}))
    db_path = directory / f'facts-{stamp}.db'
    store.apply_facts(db_path, manifest)
    return db_path


def make_config(series_params, capacities, batches, origin=ORIGIN,
                method='naive', last_day=None):
    """capacities/batches are expressed in day offsets from origin.

    Capacity offsets run from 0 (origin) through the last order day.
    """
    if last_day is None:
        last_day = max(t for t, _ in capacities)
    full = {t: 0 for t in range(last_day + 1)}
    full.update(dict(capacities))
    return {
        'origin': origin.isoformat(), 'method': method,
        'series': [{'series': sid, 'lead_time_days': lead,
                    'shelf_life_days': shelf, 'unit_cost': unit,
                    'holding_cost': hold}
                   for sid, lead, shelf, unit, hold in series_params],
        'daily_capacity': [
            {'date': (origin + timedelta(days=t)).isoformat(),
             'capacity': cap} for t, cap in sorted(full.items())],
        'initial_batches': [
            {'series': sid,
             'available': (origin + timedelta(days=avail)).isoformat(),
             'expires': (origin + timedelta(days=expires)).isoformat(),
             'quantity': qty}
            for sid, avail, expires, qty in batches]}


# --------------------------------------------------------------------------- #
# Independent exhaustive oracle
# --------------------------------------------------------------------------- #

def brute_force_optimum(config, demands):
    """Enumerate every feasible integer allocation on a small instance.

    Returns ``(feasible, minimum_cost)`` where minimum cost is None when no
    allocation satisfies every demand.  Demand rows are
    ``(series, target_offset, quantity)``.
    """
    origin = date.fromisoformat(config['origin'])
    params = {item['series']: item for item in config['series']}
    last_offset = max(
        (date.fromisoformat(item['date']) - origin).days
        for item in config['daily_capacity'])
    cap = {(date.fromisoformat(item['date']) - origin).days:
           item['capacity'] for item in config['daily_capacity']}
    cells = sorted((sid, k, qty) for (sid, k, qty) in demands if qty)

    # Every eligible source for a demand cell, with its per-unit cost.
    options = []
    for sid, k, _ in cells:
        local = []
        item = params[sid]
        for t in range(last_offset + 1):
            arrival = t + item['lead_time_days']
            expires = arrival + item['shelf_life_days']
            if arrival <= k < expires:
                local.append(('order', t,
                              item['unit_cost']
                              + item['holding_cost'] * (k - arrival)))
        for index, batch in enumerate(config['initial_batches']):
            if batch['series'] != sid:
                continue
            avail = (date.fromisoformat(batch['available']) - origin).days
            expires = (date.fromisoformat(batch['expires']) - origin).days
            if avail <= k < expires:
                local.append(('batch', index,
                              item['holding_cost'] * (k - avail)))
        options.append(local)
    if any(not local for local in options):
        return False, None

    remaining = [0] * (len(cells) + 1)
    for index in range(len(cells) - 1, -1, -1):
        remaining[index] = remaining[index + 1] + cells[index][2]

    best = [None]

    def recurse(index, day_used, batch_used, cost):
        if best[0] is not None and cost >= best[0]:
            return
        if index == len(cells):
            best[0] = cost
            return
        sid, k, qty = cells[index]
        choices = options[index]

        def split(option_index, left, running_cost):
            if left == 0:
                recurse(index + 1, day_used, batch_used, running_cost)
                return
            if option_index == len(choices):
                return
            kind, ref, unit_cost = choices[option_index]
            for amount in range(left + 1):
                if amount:
                    if kind == 'order':
                        if day_used.get(ref, 0) + amount > cap[ref]:
                            continue
                        day_used[ref] = day_used.get(ref, 0) + amount
                    else:
                        if batch_used.get(ref, 0) + amount > \
                                config['initial_batches'][ref]['quantity']:
                            continue
                        batch_used[ref] = batch_used.get(ref, 0) + amount
                split(option_index + 1, left - amount,
                      running_cost + amount * unit_cost)
                if amount:
                    if kind == 'order':
                        day_used[ref] -= amount
                    else:
                        batch_used[ref] -= amount

        split(0, qty, cost)

    recurse(0, {}, {}, 0)
    return (best[0] is not None), best[0]


def series_greedy(config, demands):
    """The forbidden per-series greedy policy, kept only as a counterexample.

    Each series is handled in isolation, target days forward, always buying the
    currently cheapest eligible source; shared day capacity is consumed on a
    first-come basis.  Returns ``(feasible, cost)``.
    """
    origin = date.fromisoformat(config['origin'])
    params = {item['series']: item for item in config['series']}
    last_offset = max(
        (date.fromisoformat(item['date']) - origin).days
        for item in config['daily_capacity'])
    cap = {(date.fromisoformat(item['date']) - origin).days:
           item['capacity'] for item in config['daily_capacity']}
    stock = {}
    for index, batch in enumerate(config['initial_batches']):
        stock[index] = batch['quantity']
    used = {t: 0 for t in cap}
    cost = 0
    for sid in sorted(params):
        item = params[sid]
        for k in sorted(k for s, k, q in demands if s == sid and q):
            qty = next(q for s, kk, q in demands if s == sid and kk == k)
            sources = []
            for index, batch in enumerate(config['initial_batches']):
                if batch['series'] != sid:
                    continue
                avail = (date.fromisoformat(batch['available']) - origin).days
                expires = (date.fromisoformat(batch['expires']) - origin).days
                if avail <= k < expires:
                    sources.append(('batch', index,
                                    item['holding_cost'] * (k - avail),
                                    stock[index]))
            for t in range(last_offset + 1):
                arrival = t + item['lead_time_days']
                expires = arrival + item['shelf_life_days']
                if arrival <= k < expires:
                    sources.append(('order', t,
                                    item['unit_cost']
                                    + item['holding_cost'] * (k - arrival),
                                    cap[t] - used[t]))
            sources.sort(key=lambda source: (source[2], source[0], source[1]))
            left = qty
            for kind, ref, unit_cost, available in sources:
                take = min(left, available)
                if take <= 0:
                    continue
                if kind == 'batch':
                    stock[ref] -= take
                else:
                    used[ref] += take
                cost += take * unit_cost
                left -= take
            if left:
                return False, None
    return True, cost


# --------------------------------------------------------------------------- #
# Independent report verifier
# --------------------------------------------------------------------------- #

def verify_plan(report, config):
    """Re-derive every invariant from the published report, no solver imports."""
    assert report['schema_version'] == 1
    assert report['mode'] == 'demand_plan'
    assert len(report['config_sha256']) == 64
    result = report['result']
    origin = date.fromisoformat(result['origin'])
    target_days = [date.fromisoformat(text) for text in result['target_dates']]
    params = {item['series']: item for item in result['series_parameters']}
    demand = {(row['series'], date.fromisoformat(row['target_date'])):
              row['demand'] for row in result['demand']}
    capacities = {(date.fromisoformat(row['date'])): row['capacity']
                  for row in result['daily_capacity']}

    # Orders respect the shared day capacity.
    ordered_by_day = {}
    for order in result['orders']:
        assert order['quantity'] > 0
        day = date.fromisoformat(order['order_date'])
        ordered_by_day[day] = ordered_by_day.get(day, 0) + order['quantity']
        arrival = date.fromisoformat(order['arrival_date'])
        expires = date.fromisoformat(order['expires_date'])
        item = params[order['series']]
        assert (arrival - day).days == item['lead_time_days']
        assert (expires - arrival).days == item['shelf_life_days']
    for day, qty in ordered_by_day.items():
        assert qty <= capacities[day]

    # Allocations partition every demand and only use valid eligibility windows.
    served = {}
    for allocation in result['allocations']:
        assert allocation['quantity'] > 0
        key = (allocation['series'],
               date.fromisoformat(allocation['target_date']))
        served[key] = served.get(key, 0) + allocation['quantity']
        item = params[allocation['series']]
        target = date.fromisoformat(allocation['target_date'])
        if allocation['source_kind'] == 'order':
            order = next(o for o in result['orders']
                         if o['series'] == allocation['series']
                         and o['order_date'] == allocation['source_ref'])
            arrival = date.fromisoformat(order['arrival_date'])
            expires = date.fromisoformat(order['expires_date'])
            assert arrival <= target < expires
        else:
            batch = next(b for b in result['initial_batches']
                         if b['batch'] == allocation['source_ref']
                         and b['series'] == allocation['series'])
            assert batch['used'] <= batch['quantity']
            available = date.fromisoformat(batch['available'])
            expires = date.fromisoformat(batch['expires'])
            assert available <= target < expires
    unmet = {(row['series'], date.fromisoformat(row['target_date'])):
             row['unmet'] for row in result['demand_shortfall']}
    for key, qty in demand.items():
        assert served.get(key, 0) + unmet.get(key, 0) == qty

    # Initial batches are never over-used and may be discarded.
    for batch in result['initial_batches']:
        assert 0 <= batch['used'] <= batch['quantity']

    # Recompute the cost independently from the allocations.
    purchase = sum(o['quantity'] * params[o['series']]['unit_cost']
                   for o in result['orders'])
    holding = 0
    for row in result['allocations']:
        item = params[row['series']]
        target = date.fromisoformat(row['target_date'])
        if row['source_kind'] == 'order':
            order = next(o for o in result['orders']
                         if o['series'] == row['series']
                         and o['order_date'] == row['source_ref'])
            arrival = date.fromisoformat(order['arrival_date'])
        else:
            batch = next(b for b in result['initial_batches']
                         if b['batch'] == row['source_ref'])
            arrival = date.fromisoformat(batch['available'])
        holding += row['quantity'] * item['holding_cost'] \
            * (target - arrival).days
    assert purchase == result['purchase_cost']
    assert holding == result['holding_cost']
    assert purchase + holding == result['total_cost']

    cert = result['certificate']
    edges = cert['edges']
    nodes = {node['id'] for node in cert['nodes']}
    potentials = {int(k): v for k, v in cert['potentials'].items()}
    assert set(potentials) == nodes
    for edge in edges:
        assert 0 <= edge['flow'] <= edge['capacity']
    # Reconstruct residual arcs and check the certificate appropriate to the
    # reported status.  Potentials certify only nodes reachable from the
    # source on residual arcs; an infeasible run is certified by its min cut.
    residual = []
    for edge in edges:
        if edge['capacity'] - edge['flow'] > 0:
            residual.append((edge['from'], edge['to'], edge['cost']))
        if edge['flow'] > 0:
            residual.append((edge['to'], edge['from'], -edge['cost']))
    source_id = cert['source_id']
    reachable_from_source = {source_id}
    stack = [source_id]
    while stack:
        u = stack.pop()
        for x, v, _ in residual:
            if x == u and v not in reachable_from_source:
                reachable_from_source.add(v)
                stack.append(v)
    if result['status'] == 'feasible':
        assert cert['kind'] == 'potentials'
        assert result['flow']['shortfall'] == 0
        for u, v, cost in residual:
            if u in reachable_from_source:
                assert cost + potentials[u] - potentials[v] >= 0
        # Every demand edge is saturated.
        for edge in edges:
            if edge['kind'] == 'demand':
                assert edge['flow'] == edge['capacity']
    else:
        assert cert['kind'] == 'min_cut'
        reachable = set(cert['reachable'])
        assert cert['source_id'] in reachable
        assert cert['sink_id'] not in reachable
        # Reachable set is closed under positive residual arcs.
        for u, v, _ in residual:
            assert not (u in reachable and v not in reachable)
        # Potentials are finite integers on the reachable side.
        for node_id in reachable:
            assert isinstance(potentials[node_id], int)
        listed = {edge['id'] for edge in cert['cut_edges']}
        actual = {edge['id'] for edge in edges
                  if edge['from'] in reachable and edge['to'] not in reachable}
        assert listed == actual
        assert cert['cut_capacity'] == cert['max_flow'] == \
            result['flow']['satisfied']
        assert sum(edge['capacity'] for edge in cert['cut_edges']) == \
            cert['cut_capacity']
        assert result['flow']['shortfall'] == sum(
            row['unmet'] for row in result['demand_shortfall'])
    # Unreachable nodes may carry any finite potential value.
    for node_id in nodes:
        assert isinstance(potentials[node_id], int)
    return result['status'], result['total_cost']


def run_planner(directory, config, rows, origin=ORIGIN):
    db_path = build_store(directory, rows, origin=origin,
                          method=config['method'])
    config_path = directory / 'config.json'
    config_path.write_text(json.dumps(config))
    return plan.plan(db_path, config_path)


def config_for(demands, series_params, capacities, batches, origin=ORIGIN,
               method='naive'):
    """Build a config whose capacity grid spans every order day."""
    last_day = max(k for _, k, _ in demands)
    return make_config(series_params, capacities, batches, origin=origin,
                       method=method, last_day=last_day)


class ExhaustiveOptimalityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _check(self, config, demands):
        report = run_planner(self.directory, config, demands)
        feasible, optimum = brute_force_optimum(config, demands)
        status, cost = verify_plan(report, config)
        if feasible:
            self.assertEqual(status, 'feasible')
            self.assertEqual(cost, optimum)
        else:
            self.assertEqual(status, 'infeasible')
            self.assertIsNone(optimum)

    def test_small_instances_match_exhaustive_enumeration(self):
        demand_sets = [
            [('a', 1, 2), ('a', 2, 1), ('b', 1, 1), ('b', 2, 2)],
            [('a', 1, 1), ('a', 2, 2), ('a', 3, 1)],
            [('a', 2, 1), ('a', 3, 1), ('b', 2, 2), ('b', 3, 1)],
        ]
        configs = [
            config_for(demand_sets[0],
                       [('a', 0, 3, 10, 1), ('b', 1, 2, 4, 2)],
                       [(0, 2), (1, 3), (2, 3)], []),
            config_for(demand_sets[1],
                       [('a', 1, 2, 5, 3)],
                       [(0, 1), (1, 4), (2, 2)],
                       [('a', 1, 2, 1)]),
            config_for(demand_sets[2],
                       [('a', 0, 2, 7, 0), ('b', 1, 3, 6, 5)],
                       [(0, 0), (1, 2), (2, 5)],
                       [('b', 2, 4, 2)]),
        ]
        for config, demands in zip(configs, demand_sets):
            with self.subTest(config=config):
                self._check(config, demands)

    def test_forecast_ceiling_and_fractional_demand(self):
        # Forecasts 5/2 -> 3, 1/4 -> 1, 0 -> 0 after the exact ceiling.
        demands = [('a', 1, Fraction(5, 2)), ('a', 2, Fraction(1, 4)),
                   ('a', 3, 0)]
        config = config_for(demands, [('a', 0, 3, 10, 1)],
                            [(0, 5), (1, 5), (2, 5)], [])
        report = run_planner(self.directory, config, demands)
        verify_plan(report, config)
        rows = report['result']['demand']
        self.assertEqual([row['demand'] for row in rows], [3, 1, 0])
        integer_demands = [(s, k, -(-f.numerator // f.denominator))
                           for s, k, f in demands]
        self.assertEqual(
            report['result']['total_cost'],
            brute_force_optimum(config, integer_demands)[1])

    def test_expiring_initial_batch_is_discarded_then_replaced(self):
        # Batch [1,2) expires before the day-2 demand and must be discarded;
        # only a fresh order can cover that demand.
        demands = [('a', 1, 2), ('a', 2, 3)]
        config = config_for(demands, [('a', 0, 3, 10, 1)],
                            [(0, 5), (1, 5), (2, 5)],
                            [('a', 1, 2, 4)])
        self._check(config, demands)
        report = run_planner(self.directory, config, demands)
        batch = report['result']['initial_batches'][0]
        self.assertEqual(batch['quantity'], 4)
        self.assertEqual(batch['used'], 2)

    def test_long_lead_time_misses_early_targets(self):
        # Lead time 3 means the first possible arrival is day 3; day 1/2 demand
        # is infeasible even though plenty of capacity exists.
        demands = [('a', 1, 1), ('a', 2, 1), ('a', 3, 1)]
        config = config_for(demands, [('a', 3, 2, 10, 1)],
                            [(0, 9), (1, 9), (2, 9), (3, 9)], [])
        report = run_planner(self.directory, config, demands)
        self.assertEqual(report['result']['status'], 'infeasible')
        verify_plan(report, config)
        self.assertEqual(
            {row['target_date'] for row in
             report['result']['demand_shortfall']},
            {'2026-01-02', '2026-01-03'})

    def test_half_open_shelf_window_excludes_expiry_day(self):
        # Shelf life 2 from a day-2 arrival gives the half-open window
        # [day 2, day 4): usable on day 2 and day 3, never on day 4.  A day-4
        # demand therefore needs its own later order, and the verifier checks
        # every allocation against that boundary directly.
        demands = [('a', 2, 1), ('a', 3, 1), ('a', 4, 1)]
        config = config_for(demands, [('a', 1, 2, 10, 0)],
                            [(0, 5), (1, 5), (2, 5), (3, 5)], [])
        self._check(config, demands)
        report = run_planner(self.directory, config, demands)
        used_windows = set()
        for order in report['result']['orders']:
            used_windows.add((order['arrival_date'], order['expires_date']))
        # The day-4 demand is covered by an order arriving on day 3 or 4,
        # never by the day-2-arrival batch that expires at the start of day 4.
        for allocation in report['result']['allocations']:
            if allocation['target_date'] == '2026-01-05':
                self.assertNotEqual(allocation['source_ref'], '2026-01-01')

    def test_shared_capacity_greedy_counterexample(self):
        # Two series share one ordering capacity per day.  Series b has lead
        # time 1, so its day-1 demand can only be ordered on day 0.  The
        # per-series greedy processes a first and spends the single day-0 slot
        # on a's day-1 demand (an early order is never more expensive), leaving
        # b with no feasible source.  The joint min-cost flow reserves day 0
        # for b and buys a's unit on day 1, satisfying every demand.
        demands = [('a', 1, 1), ('a', 2, 1), ('b', 1, 1), ('b', 2, 1)]
        config = config_for(
            demands,
            [('a', 0, 3, 10, 0), ('b', 1, 2, 10, 0)],
            [(0, 1), (1, 9), (2, 9)], [])
        feasible_greedy, _ = series_greedy(config, demands)
        self.assertFalse(feasible_greedy)
        report = run_planner(self.directory, config, demands)
        status, cost = verify_plan(report, config)
        self.assertEqual(status, 'feasible')
        self.assertEqual(cost, brute_force_optimum(config, demands)[1])

    def test_long_lead_early_purchase_greedy_failure(self):
        # Positive holding makes the greedy prefer the latest eligible order.
        # The day-2 order slot is frozen at zero, so the day-2 demand can only
        # be covered by an order placed on day 0; greedy instead spends day 0
        # on the day-1 demand and has nothing left for day 2.  The joint flow
        # buys both units on day 0 and pays the holding fee.
        demands = [('a', 1, 1), ('a', 2, 1)]
        config = config_for(
            demands, [('a', 0, 2, 10, 5)],
            [(0, 1), (1, 1), (2, 0)], [])
        feasible_greedy, _ = series_greedy(config, demands)
        self.assertFalse(feasible_greedy)
        report = run_planner(self.directory, config, demands)
        status, cost = verify_plan(report, config)
        self.assertEqual(status, 'feasible')
        optimum = brute_force_optimum(config, demands)[1]
        self.assertEqual(cost, optimum)
        # One unit is held for a day before the day-2 consumption.
        self.assertEqual(cost, 30)


    def test_deterministic_output_across_reruns(self):
        demands = [('a', 1, 2), ('a', 2, 1), ('b', 1, 1), ('b', 2, 2)]
        config = config_for(
            demands,
            [('a', 0, 3, 10, 1), ('b', 1, 2, 4, 2)],
            [(0, 2), (1, 3), (2, 3)],
            [('b', 1, 3, 1)])
        first = run_planner(self.directory, config, demands)
        second = run_planner(self.directory, config, demands)
        self.assertEqual(first, second)
        self.assertEqual(first['source']['kind'], 'forecast_fact_store')
        self.assertEqual(first['source']['generation'],
                         first['result']['generation'])
        self.assertEqual(first['source']['cross_section']['fact_count'],
                         len(demands))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def base_config(self):
        return make_config([('a', 0, 2, 10, 1)], [(0, 5), (1, 5)], [])

    def test_negative_forecast_is_rejected(self):
        config = self.base_config()
        with self.assertRaises(ValueError):
            run_planner(self.directory, config, [('a', 1, -1)])

    def test_cross_section_must_be_nonempty(self):
        config = self.base_config()
        with self.assertRaises(ValueError):
            run_planner(self.directory, config, [])

    def test_series_sets_must_match(self):
        config = make_config(
            [('a', 0, 2, 10, 1), ('b', 0, 2, 10, 1)],
            [(0, 5), (1, 5)], [])
        with self.assertRaises(ValueError):
            run_planner(self.directory, config, [('a', 1, 1)])

    def test_targets_must_be_consecutive_and_within_30_days(self):
        config = self.base_config()
        with self.assertRaises(ValueError):
            run_planner(self.directory, config,
                        [('a', 1, 1), ('a', 3, 1)])
        with self.assertRaises(ValueError):
            run_planner(self.directory, config, [('a', 31, 1)])

    def test_shared_target_dates_are_required(self):
        config = make_config(
            [('a', 0, 2, 10, 1), ('b', 0, 2, 10, 1)],
            [(0, 5), (1, 5)], [])
        db_path = build_store(
            self.directory,
            [('a', 1, 1), ('a', 2, 1), ('b', 1, 1)])
        config_path = self.directory / 'config.json'
        config_path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            plan.plan(db_path, config_path)

    def test_capacity_dates_must_cover_order_days(self):
        config = self.base_config()
        config['daily_capacity'] = config['daily_capacity'][:1]
        with self.assertRaises(ValueError):
            run_planner(self.directory, config, [('a', 1, 1)])

    def test_bad_config_shapes_exit_value_error(self):
        good = self.base_config()
        bad_copies = []
        broken = json.loads(json.dumps(good))
        broken['series'][0]['shelf_life_days'] = 0
        bad_copies.append(broken)
        broken = json.loads(json.dumps(good))
        broken['initial_batches'] = [
            {'series': 'a', 'available': ORIGIN.isoformat(),
             'expires': ORIGIN.isoformat(), 'quantity': 1}]
        bad_copies.append(broken)
        broken = json.loads(json.dumps(good))
        broken['series'][0]['unit_cost'] = 10 ** 9 + 1
        bad_copies.append(broken)
        broken = json.loads(json.dumps(good))
        broken['extra'] = 1
        bad_copies.append(broken)
        rows = [('a', 1, 1)]
        db_path = build_store(self.directory, rows)
        for broken in bad_copies:
            config_path = self.directory / 'broken.json'
            config_path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                plan.plan(db_path, config_path)

    def test_zero_capacity_is_infeasible_not_an_error(self):
        # Every order day is frozen at zero, so nothing can be purchased.
        demands = [('a', 1, 2)]
        config = config_for(demands, [('a', 0, 2, 10, 1)],
                            [(0, 0), (1, 0)], [])
        report = run_planner(self.directory, config, demands)
        status, _ = verify_plan(report, config)
        self.assertEqual(status, 'infeasible')
        self.assertEqual(report['result']['flow']['shortfall'], 2)

    def test_limits_are_enforced(self):
        too_many = make_config(
            [(f's{i}', 0, 2, 1, 0) for i in range(21)],
            [(0, 1), (1, 1)], [])
        with self.assertRaises(ValueError):
            plan.load_config(too_many)
        config = self.base_config()
        config['initial_batches'] = [
            {'series': 'a', 'available': ORIGIN.isoformat(),
             'expires': (ORIGIN + timedelta(days=2)).isoformat(),
             'quantity': 1} for _ in range(601)]
        with self.assertRaises(ValueError):
            plan.load_config(config)


class ScaleAndFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_20_series_and_600_batches_scale(self):
        series_params = [(f's{i:02d}', i % 3, 5, 1 + i, i % 4)
                         for i in range(20)]
        batches = []
        for i in range(20):
            for j in range(30):
                batches.append((f's{i:02d}', 0, 4 + (j % 3), 1))
        self.assertEqual(len(batches), 600)
        config = make_config(series_params,
                             [(t, 10 ** 9) for t in range(31)], batches)
        demands = [(f's{i:02d}', k, 1)
                   for i in range(20) for k in range(1, 31)]
        report = run_planner(self.directory, config, demands)
        verify_plan(report, config)
        self.assertEqual(report['result']['status'], 'feasible')
        self.assertEqual(report['result']['flow']['satisfied'],
                         20 * 30)

    def test_fixture_config_against_real_backtest_store(self):
        report_path = self.directory / 'backtest.json'
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-m', 'demand_backtest.cli',
             '--source', str(FIXTURES / 'input.json'),
             '--output', str(report_path)],
            capture_output=True, text=True,
            env={'PYTHONPATH': str(ROOT / 'src'), 'PATH': '/usr/bin:/bin'})
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = self.directory / 'manifest.json'
        manifest.write_text(json.dumps({
            'txid': 'seed', 'expected_generation': 0,
            'puts': [{'name': 'backtest', 'path': str(report_path)}],
            'drops': []}))
        db_path = self.directory / 'facts.db'
        store.apply_facts(db_path, manifest)
        config = json.loads((FIXTURES / 'plan_config.json').read_text())
        config_path = self.directory / 'plan_config.json'
        config_path.write_text(json.dumps(config))
        report = plan.plan(db_path, config_path)
        verify_plan(report, config)
        result_block = report['result']
        self.assertEqual(result_block['status'], 'feasible')
        # Naive forecasts are 6 on both targets; the 6-unit batch covers day 1
        # for free, and 6 fresh units cover day 2 at unit cost 5.
        self.assertEqual(result_block['total_cost'], 30)
        self.assertEqual(result_block['purchase_cost'], 30)
        self.assertEqual(result_block['holding_cost'], 0)
        self.assertEqual(result_block['initial_batches'][0]['used'], 6)


if __name__ == '__main__':
    unittest.main()
