"""End-to-end CLI tests for the forecast fact store.

Every test here drives the real ``demand-store`` command as a separate process
and uses a tiny standard-library verifier (below) that reads the SQLite file
independently of the implementation's own helpers.
"""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'fixtures'
_CRASH_BEFORE = 97
_CRASH_AFTER = 98


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)


def calendar_input(series, days):
    return {'observations': [
        {'series': series, 'quantity': (i * 3 + 1) % 5,
         'date': (date(2026, 1, 1) + timedelta(days=i)).isoformat()}
        for i in range(days)], 'initial_train': 3, 'horizon': 2,
        'mean_window': 2, 'season': 2}


def vintage_input(series, days, origin_count):
    return {'revisions': [
        {'series': series, 'quantity': (i * 3 + 1) % 5,
         'date': (date(2026, 1, 1) + timedelta(days=i)).isoformat(),
         'published_on': (date(2026, 1, 1) + timedelta(days=i)).isoformat()}
        for i in range(days)],
        'origins': [{'series': series,
                     'date': (date(2026, 1, 3) + timedelta(days=i)).isoformat()}
                    for i in range(origin_count)],
        'score_as_of': (date(2026, 1, 1) + timedelta(days=days - 1)).isoformat(),
        'initial_train': 3, 'horizon': 2, 'mean_window': 2, 'season': 2}


def demand_store(*arguments, env=None):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    if env:
        environment.update(env)
    return subprocess.run(
        [sys.executable, '-m', 'demand_backtest.store_cli', *arguments],
        capture_output=True, text=True, env=environment)


def backtest(mode, source, output):
    result = subprocess.run(
        [sys.executable, '-m', 'demand_backtest.cli', mode, str(source),
         '--output', str(output)], capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')))
    if result.returncode:
        raise AssertionError(result.stderr)
    return output


def independent_state(db):
    """Read facts, owners, reports and ledger straight from SQLite."""
    connection = sqlite3.connect(str(db))
    connection.row_factory = sqlite3.Row
    try:
        generation = connection.execute(
            "SELECT value FROM meta WHERE key='generation'").fetchone()[0]
        facts = {}
        for row in connection.execute(
                'SELECT series, origin, target_date, method, horizon,'
                ' forecast, actual, scale FROM facts'):
            facts[(row['series'], row['origin'], row['target_date'],
                   row['method'])] = (row['horizon'], row['forecast'],
                                      row['actual'], row['scale'])
        owners = {}
        for row in connection.execute(
                'SELECT series, origin, target_date, method, name'
                ' FROM fact_reports'):
            owners.setdefault(
                (row['series'], row['origin'], row['target_date'],
                 row['method']), set()).add(row['name'])
        reports = {row['name']: (row['kind'], row['source_sha256'],
                                 row['report_sha256'], row['fact_count'])
                   for row in connection.execute('SELECT * FROM reports')}
        ledger = [(row['txid'], row['generation'], row['content_sha256'],
                   row['receipt'], row['receipt_sha256'])
                  for row in connection.execute(
                      'SELECT * FROM ledger ORDER BY generation')]
        return {'generation': int(generation), 'facts': facts,
                'owners': owners, 'reports': reports, 'ledger': ledger}
    finally:
        connection.close()


class StoreCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'facts.db'

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, name, txid, generation, puts=None, drops=None):
        path = self.directory / name
        path.write_text(json.dumps({'txid': txid,
                                    'expected_generation': generation,
                                    'puts': puts or [], 'drops': drops or []}))
        return path

    def report(self, name, result, mode='calendar'):
        payload = {'schema_version': 1,
                   'source_sha256': hashlib.sha256(name.encode()).hexdigest(),
                   'result': result}
        path = self.directory / name
        path.write_text(json.dumps(payload))
        return path

    def calendar_report(self, name, series, days):
        input_path = self.directory / (name + '.input.json')
        input_path.write_text(json.dumps(calendar_input(series, days)))
        return backtest('--source', input_path, self.directory / name)

    def test_shared_retraction_and_ledger_from_real_reports(self):
        calendar = self.calendar_report('calendar.json', 's', 7)
        vintage_input_path = self.directory / 'vintage.input.json'
        vintage_input_path.write_text(
            json.dumps(vintage_input('s', 7, 3)))
        vintage = backtest('--vintage-source', vintage_input_path,
                           self.directory / 'vintage.json')
        seed = self.manifest('seed.json', 'seed', 0, [
            {'name': 'cal', 'path': str(calendar)},
            {'name': 'vin', 'path': str(vintage)}])
        applied = demand_store('apply', '--db', str(self.db),
                               '--manifest', str(seed))
        self.assertEqual(applied.returncode, 0, applied.stderr)
        receipt = json.loads(applied.stdout)
        self.assertEqual(receipt['generation'], 1)
        state = independent_state(self.db)
        self.assertTrue(all(owner == {'cal', 'vin'}
                            for owner in state['owners'].values()))
        self.assertEqual(state['reports']['cal'][2],
                         hashlib.sha256(calendar.read_bytes()).hexdigest())
        # Drop one owner: facts remain, owned now by the survivor only.
        drop_one = self.manifest('drop.json', 'drop-one', 1, [],
                                 [{'name': 'cal'}])
        dropped = demand_store('apply', '--db', str(self.db),
                               '--manifest', str(drop_one))
        self.assertEqual(dropped.returncode, 0, dropped.stderr)
        state = independent_state(self.db)
        self.assertTrue(all(owner == {'vin'}
                            for owner in state['owners'].values()))
        self.assertNotIn('cal', state['reports'])
        query = demand_store('query', '--db', str(self.db))
        self.assertEqual(query.returncode, 0, query.stderr)
        snapshot = json.loads(query.stdout)
        self.assertEqual(snapshot['generation'], state['generation'])
        for fact in snapshot['facts']:
            self.assertEqual(fact['reports'], ['vin'])
        # The ledger stores canonical JSON whose digest matches its column.
        for txid, generation, content, text, digest in state['ledger']:
            self.assertEqual(canonical(json.loads(text)), text)
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                             digest)
            self.assertEqual(len(content), 64)

    def test_all_reports_corrected_together_but_single_side_conflicts(self):
        base = [{'series': 's', 'origin': '2026-01-03',
                 'target_date': '2026-01-04', 'horizon': 1, 'method': 'naive',
                 'forecast': '3', 'actual': 5, 'scale': '4'}]
        first = self.report('a.json', None)
        payload = json.loads(first.read_text())
        # Use a real calendar result shape so the report parser accepts it.
        payload['result'] = {'origins': [], 'forecasts': base, 'metrics': [],
                             'evaluation_unit': 'x'}
        first.write_text(json.dumps(payload))
        second = self.directory / 'b.json'
        second.write_bytes(first.read_bytes())
        seed = self.manifest('seed.json', 'seed', 0, [
            {'name': 'a', 'path': str(first)},
            {'name': 'b', 'path': str(second)}])
        self.assertEqual(demand_store('apply', '--db', str(self.db),
                                      '--manifest', str(seed)).returncode, 0)
        before = independent_state(self.db)
        corrected = [dict(base[0], actual=7)]
        changed = self.report('changed.json', {'origins': [],
                                               'forecasts': corrected,
                                               'metrics': [],
                                               'evaluation_unit': 'x'})
        one_sided = self.manifest('onesided.json', 'onesided', 1, [
            {'name': 'c', 'path': str(changed)}], [{'name': 'a'}])
        refused = demand_store('apply', '--db', str(self.db),
                              '--manifest', str(one_sided))
        self.assertEqual(refused.returncode, 2)
        self.assertIn('conflicts', refused.stderr)
        self.assertEqual(independent_state(self.db), before)
        together = self.manifest('together.json', 'together', 1, [
            {'name': 'c', 'path': str(changed)}],
            [{'name': 'a'}, {'name': 'b'}])
        accepted = demand_store('apply', '--db', str(self.db),
                                '--manifest', str(together))
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        state = independent_state(self.db)
        self.assertEqual([values[2] for values in state['facts'].values()],
                         ['7'])
        self.assertEqual(list(state['owners'].values())[0], {'c'})

    def test_two_processes_competing_for_generation(self):
        series_a = self.calendar_report('a.json', 'a', 7)
        series_b = self.calendar_report('b.json', 'b', 7)
        self.assertEqual(demand_store(
            'apply', '--db', str(self.db), '--manifest', str(self.manifest(
                'seed.json', 'seed', 0,
                [{'name': 'a', 'path': str(series_a)}]))).returncode, 0)
        manifests = [self.manifest('race1.json', 'race1', 1,
                                   [{'name': 'x', 'path': str(series_b)}]),
                     self.manifest('race2.json', 'race2', 1,
                                   [{'name': 'y', 'path': str(series_b)}])]
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        processes = [subprocess.Popen(
            [sys.executable, '-m', 'demand_backtest.store_cli', 'apply',
             '--db', str(self.db), '--manifest', str(manifest)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=environment) for manifest in manifests]
        for process in processes:
            process.communicate()
        codes = [process.returncode for process in processes]
        self.assertEqual(sorted(codes), [0, 2])
        state = independent_state(self.db)
        self.assertEqual(state['generation'], 2)
        self.assertEqual(len(state['ledger']), 2)
        # Exactly one raced report is bound; the loser changed nothing.
        self.assertEqual(len(state['reports']), 2)

    def test_crash_before_commit_leaves_logical_store_unchanged(self):
        seed = self.calendar_report('seed.json', 's', 7)
        self.assertEqual(demand_store(
            'apply', '--db', str(self.db), '--manifest', str(self.manifest(
                'm0.json', 'm0', 0, [{'name': 'a', 'path': str(seed)}]))
        ).returncode, 0)
        before = independent_state(self.db)
        other = self.calendar_report('other.json', 't', 7)
        manifest = self.manifest('next.json', 'next', 1,
                                 [{'name': 'b', 'path': str(other)}])
        crashed = demand_store('apply', '--db', str(self.db),
                               '--manifest', str(manifest),
                               env={'DEMAND_STORE_FAULT': 'before_commit'})
        self.assertEqual(crashed.returncode, _CRASH_BEFORE)
        self.assertEqual(independent_state(self.db), before)
        retried = demand_store('apply', '--db', str(self.db),
                               '--manifest', str(manifest))
        self.assertEqual(retried.returncode, 0, retried.stderr)
        self.assertEqual(json.loads(retried.stdout)['generation'], 2)

    def test_crash_after_commit_retry_replays_original_receipt(self):
        seed = self.calendar_report('seed.json', 's', 7)
        self.assertEqual(demand_store(
            'apply', '--db', str(self.db), '--manifest', str(self.manifest(
                'm0.json', 'm0', 0, [{'name': 'a', 'path': str(seed)}]))
        ).returncode, 0)
        other = self.calendar_report('other.json', 't', 7)
        manifest = self.manifest('next.json', 'next', 1,
                                 [{'name': 'b', 'path': str(other)}])
        crashed = demand_store('apply', '--db', str(self.db),
                               '--manifest', str(manifest),
                               env={'DEMAND_STORE_FAULT': 'after_commit'})
        self.assertEqual(crashed.returncode, _CRASH_AFTER)
        committed = independent_state(self.db)
        self.assertEqual(committed['generation'], 2)
        first_receipt = json.loads(committed['ledger'][-1][3])
        # Path and expected generation are stale after the crash, yet the
        # identical report bytes replay the original receipt.
        moved = self.directory / 'moved.json'
        moved.write_bytes(other.read_bytes())
        retry = self.manifest('retry.json', 'next', 0,
                              [{'name': 'b', 'path': str(moved)}])
        replayed = demand_store('apply', '--db', str(self.db),
                                '--manifest', str(retry))
        self.assertEqual(replayed.returncode, 0, replayed.stderr)
        self.assertEqual(json.loads(replayed.stdout), first_receipt)
        self.assertEqual(independent_state(self.db), committed)

    def test_query_returns_a_single_generation_while_applies_interleave(self):
        seed = self.calendar_report('seed.json', 's', 7)
        self.assertEqual(demand_store(
            'apply', '--db', str(self.db), '--manifest', str(self.manifest(
                'm0.json', 'm0', 0, [{'name': 'a', 'path': str(seed)}]))
        ).returncode, 0)
        other = self.calendar_report('other.json', 't', 7)
        manifest = self.manifest('next.json', 'next', 1,
                                 [{'name': 'b', 'path': str(other)}])
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        reader = subprocess.Popen(
            [sys.executable, '-m', 'demand_backtest.store_cli', 'query',
             '--db', str(self.db)], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=environment)
        writer = subprocess.Popen(
            [sys.executable, '-m', 'demand_backtest.store_cli', 'apply',
             '--db', str(self.db), '--manifest', str(manifest)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=environment)
        snapshot_text, query_error = reader.communicate()
        writer.communicate()
        self.assertEqual(reader.returncode, 0, query_error)
        snapshot = json.loads(snapshot_text)
        state = independent_state(self.db)
        # Whatever generation the snapshot names, its counts agree with it:
        # generation 1 has one report and one report's facts, never a mixture.
        self.assertIn(snapshot['generation'], (1, state['generation']))
        if snapshot['generation'] == 1:
            self.assertEqual(len(snapshot['reports']), 1)
            self.assertEqual(snapshot['fact_count'],
                             snapshot['reports'][0]['fact_count'])
        else:
            self.assertEqual(len(snapshot['reports']),
                             len(state['reports']))
        self.assertLessEqual(snapshot['generation'], state['generation'])


    def test_bad_manifests_and_bad_reports_exit_2(self):
        good = self.calendar_report('good.json', 's', 7)
        payload = json.loads(good.read_text())
        broken = self.directory / 'broken.json'
        broken.write_text(json.dumps({'schema_version': 1,
                                      'source_sha256': payload[
                                          'source_sha256'],
                                      'result': {'forecasts': []}}))
        cases = [
            ('{"txid": "x", "txid": "x", "expected_generation": 0,'
             ' "puts": [], "drops": []}'),
            (json.dumps({'txid': 'x', 'expected_generation': 0,
                         'puts': [{'name': 'a', 'path': str(good)}],
                         'drops': [], 'extra': 1})),
            (json.dumps({'txid': 'x', 'expected_generation': 0,
                         'puts': [{'name': 'a', 'path': str(broken)}],
                         'drops': []})),
            (json.dumps({'txid': 'x', 'expected_generation': 0,
                         'puts': [{'name': 'a', 'path': str(good)},
                                  {'name': 'a', 'path': str(good)}],
                         'drops': []})),
            (json.dumps({'txid': 'x', 'expected_generation': 0, 'puts': [],
                         'drops': [{'name': 'gone'}]})),
            # The store path may not equal any source report.
            (json.dumps({'txid': 'x', 'expected_generation': 0,
                         'puts': [{'name': 'a', 'path': str(self.db)}],
                         'drops': []})),
        ]
        for number, payload in enumerate(cases):
            manifest = self.directory / f'bad{number}.json'
            manifest.write_text(payload)
            result = demand_store('apply', '--db', str(self.db),
                                  '--manifest', str(manifest))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(self.db.exists())

    def test_help_and_usage_are_well_formed(self):
        self.assertEqual(demand_store('--help').returncode, 0)
        self.assertEqual(demand_store('apply', '--help').returncode, 0)
        self.assertEqual(demand_store('query', '--help').returncode, 0)
        self.assertEqual(demand_store('apply', '--db', str(self.db))
                         .returncode, 2)
        self.assertEqual(demand_store('query', '--db', str(self.db))
                         .returncode, 2)


if __name__ == '__main__':
    unittest.main()
