"""Contract and end-to-end tests for hierarchy reconciliation.

The small-tree checks use the exponential oracle from test_tree_qp as an
independent reference; the wide-tree check would time out if the implementation
enumerated active sets.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from fractions import Fraction as F
from pathlib import Path

from demand_backtest.model import analyze
from demand_backtest.reconcile import analyze_reconcile
from demand_backtest.tree_qp import objective
from demand_backtest.vintage import analyze_vintage

from test_tree_qp import enumerate_optimum

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'fixtures'


def report(kind, result, token=None):
    token = token or hashlib.sha256(kind.encode()).hexdigest()
    return {'schema_version': 1, 'source_sha256': token, 'result': result}


def fact(series, origin, target, method, horizon, forecast):
    return {'series': series, 'origin': origin, 'target_date': target,
            'horizon': horizon, 'method': method, 'forecast': forecast}


def calendar_report(facts):
    return report('calendar', {'origins': [], 'forecasts': facts,
                               'metrics': [], 'evaluation_unit': 'x'})


def vintage_report(facts):
    return report('vintage', {'mode': 'revision_vintage', 'score_as_of': '2026-01-02',
                              'origins': [], 'lineage': [], 'forecasts': facts,
                              'metrics': [], 'skipped': [],
                              'evaluation_unit': 'x'})


def hierarchy(nodes):
    return {'nodes': [{'series': s, 'parent': p, 'weight': w}
                      for s, p, w in nodes]}


class FixtureReconciliationTests(unittest.TestCase):
    """Real reports produced by the two existing backtest modes."""

    def setUp(self):
        self.calendar = report('calendar',
                               analyze(json.loads((FIXTURES /
                                                    'reconcile_calendar_input.json').read_text())))
        self.vintage = report('vintage', analyze_vintage(json.loads(
            (FIXTURES / 'reconcile_vintage_input.json').read_text())))
        self.tree = json.loads((FIXTURES / 'hierarchy.json').read_text())

    def test_groups_are_unique_sorted_and_complete(self):
        result = analyze_reconcile([self.calendar, self.vintage], self.tree)
        self.assertEqual(result['fact_count'], 60)
        self.assertEqual(result['group_count'], 12)
        keys = [(g['origin'], g['target_date'], g['method'])
                for g in result['groups']]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(result['hierarchy']['root'], 'all')
        self.assertEqual(result['hierarchy']['leaves'],
                         ['bolts', 'nuts', 'steady'])
        for group in result['groups']:
            self.assertEqual([n['series'] for n in group['nodes']],
                             ['all', 'bolts', 'nuts', 'steady', 'total'])
            self.assertEqual(len(group['nodes']), 5)

    def test_parent_equals_sum_of_direct_children_exactly(self):
        result = analyze_reconcile([self.calendar, self.vintage], self.tree)
        for group in result['groups']:
            for node in group['nodes']:
                children = [n for n in group['nodes']
                            if n['parent'] == node['series']]
                if children:
                    total = sum((F(c['reconciled']) for c in children), F(0))
                    self.assertEqual(F(node['reconciled']), total)
                else:
                    self.assertGreaterEqual(F(node['reconciled']), 0)

    def test_optimum_reaches_zero_leaf_and_matches_oracle(self):
        result = analyze_reconcile([self.calendar, self.vintage], self.tree)
        any_zero = any(group['zero_leaves'] for group in result['groups'])
        self.assertTrue(any_zero)
        # Use the smallest group and compare against the exponential oracle.
        group = result['groups'][1]  # origin 01-04, target 01-05, naive
        series = [n['series'] for n in group['nodes']]
        parents = [n['parent'] for n in group['nodes']]
        weights = [F(n['weight']) for n in group['nodes']]
        forecasts = [F(n['forecast']) for n in group['nodes']]
        values = {n['series']: F(n['reconciled']) for n in group['nodes']}
        self.assertEqual(F(group['objective']),
                         enumerate_optimum(series, parents, weights, forecasts))
        # Zero-then-recompute-parents is feasible but not optimal here: the
        # strict solver objective is strictly smaller, proving we did not take
        # that shortcut.
        clipped = dict(values)
        for node in group['nodes']:
            if node['leaf']:
                clipped[node['series']] = max(forecasts[series.index(
                    node['series'])], F(0))
        for _ in range(len(series)):
            for node in group['nodes']:
                kids = [n for n in group['nodes']
                        if n['parent'] == node['series']]
                if kids:
                    clipped[node['series']] = sum(
                        (clipped[k['series']] for k in kids), F(0))
        self.assertLess(F(group['objective']),
                        objective(weights, forecasts, [clipped[s]
                                                       for s in series]))

    def test_kkt_certificate_is_exact(self):
        result = analyze_reconcile([self.calendar, self.vintage], self.tree)
        for group in result['groups']:
            certificate = {e['series']: e for e in group['leaf_kkt']}
            self.assertEqual(sorted(certificate), ['bolts', 'nuts', 'steady'])
            zero = []
            for node in group['nodes']:
                if not node['leaf']:
                    continue
                entry = certificate[node['series']]
                self.assertEqual(F(entry['value']), F(node['reconciled']))
                if node['zero']:
                    zero.append(node['series'])
                    self.assertEqual(F(node['reconciled']), 0)
                    self.assertGreaterEqual(F(entry['gradient']), 0)
                else:
                    self.assertGreater(F(node['reconciled']), 0)
                    self.assertEqual(F(entry['gradient']), 0)
            self.assertEqual(zero, group['zero_leaves'])

    def test_objective_matches_recomputed_sum(self):
        result = analyze_reconcile([self.calendar, self.vintage], self.tree)
        for group in result['groups']:
            weights = [F(n['weight']) for n in group['nodes']]
            forecasts = [F(n['forecast']) for n in group['nodes']]
            values = [F(n['reconciled']) for n in group['nodes']]
            self.assertEqual(F(group['objective']),
                             objective(weights, forecasts, values))

    def test_report_order_and_fact_order_do_not_change_output(self):
        baseline = analyze_reconcile([self.calendar, self.vintage], self.tree)
        swapped = analyze_reconcile([self.vintage, self.calendar], self.tree)
        self.assertEqual(baseline, swapped)
        # A reordered hierarchy file has its own bytes/SHA, but the reconciled
        # groups must not depend on the node order or on fact list order.
        shuffled_tree = {'nodes': list(reversed(self.tree['nodes']))}
        shuffled_reports = [dict(self.calendar), dict(self.vintage)]
        shuffled_reports[0]['result'] = dict(shuffled_reports[0]['result'],
                                             forecasts=list(reversed(
                                                 shuffled_reports[0]['result']['forecasts'])))
        digests = [hashlib.sha256(b'c').hexdigest(),
                   hashlib.sha256(b'v').hexdigest()]
        tree_digest = hashlib.sha256(b't').hexdigest()
        reordered = analyze_reconcile(shuffled_reports, shuffled_tree,
                                      report_sha256=digests,
                                      hierarchy_sha256=tree_digest)
        fixed = analyze_reconcile([self.calendar, self.vintage], self.tree,
                                  report_sha256=digests,
                                  hierarchy_sha256=tree_digest)
        self.assertEqual(fixed['groups'], reordered['groups'])

    def test_fraction_text_is_canonical(self):
        result = analyze_reconcile([self.calendar, self.vintage], self.tree)
        for group in result['groups']:
            for text in [group['objective']] + [
                    n['reconciled'] for n in group['nodes']] + [
                    n['forecast'] for n in group['nodes']]:
                self.assertEqual(str(F(text)), text)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tree = hierarchy([('all', None, '1'), ('a', 'all', '1'),
                               ('b', 'all', '1')])

    def test_different_origins_with_same_target_are_not_merged(self):
        # Both groups predict the same target date 2026-01-02, but they are two
        # separate forecast decisions because their origins differ.  Each group
        # is completed from the other report (disjoint series, as the two
        # backtest modes cover different series here).
        first_calendar = [fact('all', '2026-01-01', '2026-01-02', 'naive', 1, '4'),
                          fact('a', '2026-01-01', '2026-01-02', 'naive', 1, '1')]
        first_vintage = [fact('b', '2026-01-01', '2026-01-02', 'naive', 1, '3')]
        second_calendar = [fact('all', '2026-01-02', '2026-01-02', 'naive', 1, '7'),
                           fact('a', '2026-01-02', '2026-01-02', 'naive', 1, '2')]
        second_vintage = [fact('b', '2026-01-02', '2026-01-02', 'naive', 1, '5')]
        result = analyze_reconcile(
            [calendar_report(first_calendar + second_calendar),
             vintage_report(first_vintage + second_vintage)], self.tree)
        self.assertEqual([(g['origin'], g['target_date'])
                          for g in result['groups']],
                         [('2026-01-01', '2026-01-02'),
                          ('2026-01-02', '2026-01-02')])
        self.assertEqual([g['inputs'] for g in result['groups']],
                         [['calendar', 'vintage'], ['calendar', 'vintage']])

    def test_actual_and_extra_fields_are_ignored(self):
        facts = [dict(fact('all', '2026-01-01', '2026-01-02', 'naive', 1, '4'),
                      actual=999, scale='0', error_forecast_minus_actual='x'),
                 fact('a', '2026-01-01', '2026-01-02', 'naive', 1, '1'),
                 fact('b', '2026-01-01', '2026-01-02', 'naive', 1, '3')]
        digests = [hashlib.sha256(b'c').hexdigest(),
                   hashlib.sha256(b'v').hexdigest()]
        first = analyze_reconcile([calendar_report(facts), vintage_report([])],
                                  self.tree, report_sha256=digests)
        changed = [dict(f, actual=-7) if 'actual' in f else f for f in facts]
        second = analyze_reconcile([calendar_report(changed),
                                    vintage_report([])], self.tree,
                                   report_sha256=digests)
        self.assertEqual(first, second)
        self.assertNotIn('actual', json.dumps(first))

    def test_horizon_mismatch_and_duplicate_and_ragged_groups_rejected(self):
        base = [fact('all', '2026-01-01', '2026-01-02', 'naive', 1, '4'),
                fact('a', '2026-01-01', '2026-01-02', 'naive', 1, '1'),
                fact('b', '2026-01-01', '2026-01-02', 'naive', 1, '3')]
        both = [calendar_report(base), vintage_report([])]

        mixed = [dict(base[0], horizon=2)] + base[1:]
        with self.assertRaisesRegex(ValueError, 'horizon'):
            analyze_reconcile([calendar_report(mixed), vintage_report([])],
                              self.tree)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            analyze_reconcile([calendar_report(base),
                               vintage_report([dict(base[0])])], self.tree)
        with self.assertRaisesRegex(ValueError, 'must contain exactly'):
            analyze_reconcile([calendar_report(base[:2]), vintage_report([])],
                              self.tree)
        with self.assertRaisesRegex(ValueError, 'must contain exactly'):
            analyze_reconcile([calendar_report(base + [
                fact('ghost', '2026-01-01', '2026-01-02', 'naive', 1, '9')]),
                vintage_report([])], self.tree)
        self.assertTrue(analyze_reconcile(both, self.tree)['groups'])

    def test_empty_pool_is_valid(self):
        result = analyze_reconcile([calendar_report([]),
                                    vintage_report([])], self.tree)
        self.assertEqual(result['groups'], [])
        self.assertEqual(result['group_count'], 0)

    def test_input_digests_are_bound_and_independent_of_order(self):
        first = calendar_report([])
        second = vintage_report([])
        digests = [hashlib.sha256(b'cal').hexdigest(),
                   hashlib.sha256(b'vin').hexdigest()]
        bound = analyze_reconcile([first, second], self.tree,
                                  report_sha256=digests)
        self.assertEqual(bound['reports'][0]['report_sha256'], digests[0])
        self.assertEqual(bound['reports'][1]['report_sha256'], digests[1])
        swapped = analyze_reconcile([second, first], self.tree,
                                    report_sha256=list(reversed(digests)))
        self.assertEqual(bound, swapped)
        self.assertEqual(bound['input_sha256']['calendar'], digests[0])


class RejectedInputTests(unittest.TestCase):
    def setUp(self):
        self.tree = hierarchy([('all', None, '1'), ('a', 'all', '1')])

    def test_bad_hierarchies_rejected(self):
        def fail(tree):
            with self.assertRaises(ValueError):
                analyze_reconcile([calendar_report([]),
                                   vintage_report([])], tree)
        fail(hierarchy([('all', None, '1'), ('a', None, '1')]))          # two roots
        fail(hierarchy([('all', 'a', '1'), ('a', 'all', '1')]))          # no root
        fail(hierarchy([('all', None, '1'), ('a', 'ghost', '1')]))       # unknown parent
        fail(hierarchy([('all', None, '1'), ('a', 'all', '1'),
                        ('b', 'b', '1')]))                                # self parent
        fail(hierarchy([('all', None, '1'), ('a', 'all', '1'),
                        ('a', 'all', '2')]))                               # duplicate series
        fail(hierarchy([('all', None, '0')]))                             # zero weight
        fail(hierarchy([('all', None, '-1')]))                            # negative weight
        fail(hierarchy([('all', None, '0/1')]))                           # zero fraction
        fail(hierarchy([('all', None, '1.5')]))                           # decimal
        fail(hierarchy([('all', None, '1/0')]))                           # zero denominator
        fail({'nodes': []})                                               # no nodes
        fail({'nodes': [{'series': 'all'}]})                              # missing field
        fail({'nodes': [{'series': 'all', 'parent': None, 'weight': '1',
                         'extra': 1}]})                                   # extra field

    def test_detached_cycle_is_rejected(self):
        tree = hierarchy([('root', None, '1'), ('a', 'b', '1'),
                          ('b', 'a', '1')])
        with self.assertRaises(ValueError):
            analyze_reconcile([calendar_report([]), vintage_report([])], tree)

    def test_tree_larger_than_200_nodes_rejected(self):
        nodes = [{'series': f'n{i}', 'parent': None if i == 0 else 'n0',
                  'weight': '1'} for i in range(201)]
        with self.assertRaisesRegex(ValueError, 'bounded'):
            analyze_reconcile([calendar_report([]), vintage_report([])],
                              {'nodes': nodes})

    def test_noncanonical_forecast_fraction_rejected(self):
        facts = [fact('all', '2026-01-01', '2026-01-02', 'naive', 1, '2/4'),
                 fact('a', '2026-01-01', '2026-01-02', 'naive', 1, '1')]
        with self.assertRaisesRegex(ValueError, 'Fraction'):
            analyze_reconcile([calendar_report(facts), vintage_report([])],
                              self.tree)
        facts[0]['forecast'] = 1.5
        with self.assertRaisesRegex(ValueError, 'Fraction'):
            analyze_reconcile([calendar_report(facts), vintage_report([])],
                              self.tree)
        facts[0]['forecast'] = '1/-2'
        with self.assertRaisesRegex(ValueError, 'Fraction'):
            analyze_reconcile([calendar_report(facts), vintage_report([])],
                              self.tree)

    def test_report_shape_and_kind_rejected(self):
        with self.assertRaises(ValueError):
            analyze_reconcile([calendar_report([])], self.tree)
        with self.assertRaises(ValueError):
            analyze_reconcile([calendar_report([]), calendar_report([])],
                              self.tree)
        with self.assertRaises(ValueError):
            analyze_reconcile([vintage_report([]), vintage_report([])],
                              self.tree)
        broken = calendar_report([])
        broken['schema_version'] = 2
        with self.assertRaisesRegex(ValueError, 'schema_version'):
            analyze_reconcile([broken, vintage_report([])], self.tree)
        broken = calendar_report([])
        broken['source_sha256'] = 'not-a-sha'
        with self.assertRaisesRegex(ValueError, 'source_sha256'):
            analyze_reconcile([broken, vintage_report([])], self.tree)


class WideTreeTests(unittest.TestCase):
    def test_200_node_wide_groups_reconcile_without_enumeration(self):
        # 200 nodes, 199 leaves, 1,000 groups = exactly 200,000 facts.  The
        # active-set space has 2^199 elements per group, so the exact solve
        # completing at this size proves no active-set enumeration happens.
        size = 200
        nodes = [{'series': 'n0', 'parent': None, 'weight': '1000'}]
        nodes += [{'series': f'n{i}', 'parent': 'n0', 'weight': str(i % 7 + 1)}
                  for i in range(1, size)]
        tree = {'nodes': nodes}
        facts = []
        for group in range(1000):
            origin = (date(2026, 1, 1) + timedelta(days=group)).isoformat()
            target = '2026-12-01'
            for i in range(size):
                value = (i * 13 + group * 7) % 41 - 20
                facts.append(fact(f'n{i}', origin, target, 'naive', 1,
                                  str(value)))
        result = analyze_reconcile([calendar_report(facts),
                                    vintage_report([])], tree)
        self.assertEqual(result['group_count'], 1000)
        self.assertEqual(result['fact_count'], 200000)
        for group in result['groups'][:5]:
            weights = [F(n['weight']) for n in group['nodes']]
            forecasts = [F(n['forecast']) for n in group['nodes']]
            values = [F(n['reconciled']) for n in group['nodes']]
            self.assertEqual(F(group['objective']),
                             objective(weights, forecasts, values))
            for node in group['nodes']:
                if node['leaf']:
                    self.assertGreaterEqual(F(node['reconciled']), 0)

    def test_more_than_200000_facts_rejected(self):
        tree = {'nodes': [{'series': 'all', 'parent': None, 'weight': '1'}]}
        facts = [fact('all', f'2026-{i // 331 + 1:02d}-{i % 331 + 1:02d}',
                      '2026-07-01', 'naive', 1, '1') for i in range(200001)]
        with self.assertRaisesRegex(ValueError, '200000'):
            analyze_reconcile([calendar_report(facts), vintage_report([])],
                              tree)


class CommandLineTests(unittest.TestCase):
    def run_reconcile(self, calendar, vintage, tree, output):
        with tempfile.TemporaryDirectory() as directory:
            names = {}
            for name, payload in (('calendar', calendar), ('vintage', vintage),
                                  ('hierarchy', tree)):
                path = Path(directory) / f'{name}.json'
                path.write_text(json.dumps(payload) if not isinstance(
                    payload, str) else payload)
                names[name] = str(path)
            return subprocess.run(
                [sys.executable, '-m', 'demand_backtest.reconcile_cli',
                 '--report', names['calendar'], '--report', names['vintage'],
                 '--hierarchy', names['hierarchy'], '--output', str(output)],
                capture_output=True)

    def test_cli_binds_both_report_bytes(self):
        calendar = json.loads((FIXTURES /
                               'reconcile_calendar_input.json').read_text())
        vintage = json.loads((FIXTURES /
                              'reconcile_vintage_input.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            calendar_path = directory / 'c.json'
            vintage_path = directory / 'v.json'
            subprocess.run([sys.executable, '-m', 'demand_backtest.cli',
                            '--source', str(FIXTURES /
                                           'reconcile_calendar_input.json'),
                            '--output', str(calendar_path)], check=True,
                           capture_output=True)
            subprocess.run([sys.executable, '-m', 'demand_backtest.cli',
                            '--vintage-source', str(FIXTURES /
                                                   'reconcile_vintage_input.json'),
                            '--output', str(vintage_path)], check=True,
                           capture_output=True)
            output = directory / 'out.json'
            result = subprocess.run(
                [sys.executable, '-m', 'demand_backtest.reconcile_cli',
                 '--report', str(calendar_path), '--report', str(vintage_path),
                 '--hierarchy', str(FIXTURES / 'hierarchy.json'),
                 '--output', str(output)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text())
            self.assertEqual(payload['schema_version'], 1)
            self.assertEqual(payload['mode'], 'hierarchy_reconcile')
            self.assertEqual(payload['result']['group_count'], 12)
            self.assertEqual(payload['result']['input_sha256']['calendar'],
                             hashlib.sha256(calendar_path.read_bytes()).hexdigest())
            self.assertEqual(payload['result']['input_sha256']['vintage'],
                             hashlib.sha256(vintage_path.read_bytes()).hexdigest())
            self.assertEqual(payload['result']['input_sha256']['hierarchy'],
                             hashlib.sha256((FIXTURES / 'hierarchy.json')
                                            .read_bytes()).hexdigest())
            before = output.read_bytes()
            again = subprocess.run(
                [sys.executable, '-m', 'demand_backtest.reconcile_cli',
                 '--report', str(calendar_path), '--report', str(vintage_path),
                 '--hierarchy', str(FIXTURES / 'hierarchy.json'),
                 '--output', str(output)], capture_output=True)
            self.assertEqual(again.returncode, 0)
            self.assertEqual(before, output.read_bytes())

    def test_bad_inputs_exit_2_and_preserve_old_output(self):
        good_tree = json.loads((FIXTURES / 'hierarchy.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out.json'
            output.write_text('prior')
            cases = [
                ('{}', json.dumps(vintage_report([])), good_tree),
                (json.dumps(calendar_report([])),
                 json.dumps({'bogus': True}), good_tree),
                (json.dumps(calendar_report([])),
                 json.dumps(vintage_report([])), {'nodes': []}),
                (json.dumps(calendar_report([])),
                 json.dumps(vintage_report([])),
                 json.dumps(hierarchy([('a', None, '1'), ('b', None, '1')]))),
                (json.dumps(calendar_report([fact('all', '2026-01-01',
                                                  '2026-01-02', 'naive', 1,
                                                  '1')])),
                 json.dumps(vintage_report([])),
                 json.dumps(hierarchy([('all', None, '1'),
                                       ('a', 'all', '1')]))),
            ]
            for calendar, vintage, tree in cases:
                result = self.run_reconcile(calendar, vintage, tree, output)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertEqual(output.read_text(), 'prior')

    def test_output_must_differ_from_every_input(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            calendar = directory / 'c.json'
            vintage = directory / 'v.json'
            tree = directory / 'h.json'
            calendar.write_text(json.dumps(calendar_report([])))
            vintage.write_text(json.dumps(vintage_report([])))
            tree.write_text(json.dumps(hierarchy([('all', None, '1')])))
            for target in (calendar, vintage, tree):
                result = subprocess.run(
                    [sys.executable, '-m', 'demand_backtest.reconcile_cli',
                     '--report', str(calendar), '--report', str(vintage),
                     '--hierarchy', str(tree), '--output', str(target)],
                    capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertTrue(target.exists())
            self.assertEqual(json.loads(calendar.read_text())['schema_version'], 1)
            self.assertEqual(json.loads(vintage.read_text())['schema_version'], 1)

    def test_two_reports_required(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / 'r.json'
            report_path.write_text(json.dumps(calendar_report([])))
            tree_path = Path(directory) / 'h.json'
            tree_path.write_text(json.dumps(hierarchy([('all', None, '1')])))
            output = Path(directory) / 'out.json'
            for reports in ([report_path], [report_path, report_path, report_path]):
                command = [sys.executable, '-m',
                           'demand_backtest.reconcile_cli', '--hierarchy',
                           str(tree_path), '--output', str(output)]
                for path in reports:
                    command += ['--report', str(path)]
                result = subprocess.run(command, capture_output=True)
                self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
