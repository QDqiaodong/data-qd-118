"""Contract and CLI tests for the SQLite forecast fact store.

The store consumes reports already produced by ``demand-backtest``; these tests
build such reports with the existing analyzers and never rerun a model inside
the store.  Crash tests kill a real subprocess mid-transaction through
``DEMAND_STORE_FAULT`` so recovery is exercised against genuine SQLite
journaling, not an in-process mock.
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

from fractions import Fraction

from demand_backtest.model import analyze
from demand_backtest.vintage import analyze_vintage
from demand_backtest.store import apply_facts, query_facts

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'fixtures'
_FAILURE = 97
_CRASHED_AFTER_COMMIT = 98


def build_calendar_report(facts):
    """Wrap raw facts in the calendar report shape produced by the CLI."""
    return {'schema_version': 1,
            'source_sha256': hashlib.sha256(
                json.dumps(facts, sort_keys=True).encode()).hexdigest(),
            'result': {'origins': [], 'forecasts': facts, 'metrics': [],
                       'evaluation_unit': 'x'}}


def build_vintage_report(facts):
    return {'schema_version': 1,
            'source_sha256': hashlib.sha256(
                json.dumps(facts, sort_keys=True).encode()).hexdigest(),
            'result': {'mode': 'revision_vintage', 'score_as_of': '2026-01-07',
                       'origins': [], 'lineage': [], 'forecasts': facts,
                       'metrics': [], 'skipped': [], 'evaluation_unit': 'x'}}


def raw_fact(series, origin, target, method, horizon, forecast, actual, scale):
    return {'series': series, 'origin': origin, 'target_date': target,
            'horizon': horizon, 'method': method, 'forecast': forecast,
            'actual': actual, 'scale': scale,
            'error_forecast_minus_actual': 'ignored'}


def real_reports(directory):
    """Run the real backtest CLI over the two reconciliation fixtures.

    These are exactly the reports an operator would hand to the store: the
    calendar and revision-vintage modes over disjoint series produce 60 facts.
    """
    calendar = Path(directory) / 'calendar.json'
    vintage = Path(directory) / 'vintage.json'
    for mode, source, output in (
            ('--source', FIXTURES / 'reconcile_calendar_input.json', calendar),
            ('--vintage-source', FIXTURES / 'reconcile_vintage_input.json',
             vintage)):
        result = subprocess.run(
            [sys.executable, '-m', 'demand_backtest.cli', mode, str(source),
             '--output', str(output)], capture_output=True)
        if result.returncode:
            raise AssertionError(result.stderr.decode())
    return calendar, vintage


def matching_reports(directories):
    """Calendar and vintage reports that describe the same fact keys.

    Both analyzers evaluate the same series with the same oracles, so their
    facts agree on horizon/forecast/actual/scale and legitimate sharing occurs.
    """
    observations = [{'series': 's',
                     'date': (date(2026, 1, 1) + timedelta(days=i)).isoformat(),
                     'quantity': (i * 3 + 1) % 5} for i in range(7)]
    calendar_input = {'observations': observations, 'initial_train': 3,
                      'horizon': 2, 'mean_window': 2, 'season': 2}
    vintage_input = {
        'revisions': [{'series': row['series'], 'date': row['date'],
                       'published_on': row['date'],
                       'quantity': row['quantity']} for row in observations],
        'origins': [{'series': 's',
                     'date': (date(2026, 1, 3) + timedelta(days=i)).isoformat()}
                    for i in range(3)],
        'score_as_of': '2026-01-07', 'initial_train': 3, 'horizon': 2,
        'mean_window': 2, 'season': 2}
    paths = []
    for name, payload, kind in (
            ('matching_calendar.json', analyze(calendar_input), 'calendar'),
            ('matching_vintage.json', analyze_vintage(vintage_input), 'vintage')):
        path = Path(directories) / name
        path.write_text(json.dumps({'schema_version': 1,
                                    'source_sha256': hashlib.sha256(
                                        name.encode()).hexdigest(),
                                    'result': payload}))
        paths.append(path)
    return paths


def write_manifest(directory, name, txid, generation, puts=None, drops=None):
    path = Path(directory) / name
    path.write_text(json.dumps({'txid': txid,
                                'expected_generation': generation,
                                'puts': puts or [], 'drops': drops or []}))
    return path


def run_store(*arguments):
    return subprocess.run(
        [sys.executable, '-m', 'demand_backtest.store_cli', *arguments],
        capture_output=True, text=True)


class RealReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'store.db'
        self.calendar, self.vintage = real_reports(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, name, txid, generation, puts, drops=None):
        return write_manifest(self.directory, name, txid, generation, puts,
                              drops)

    def test_apply_query_and_shared_provenance(self):
        calendar, vintage = matching_reports(self.directory)
        batch = self.manifest('batch.json', 'seed', 0,
                              [{'name': 'c', 'path': str(calendar)},
                               {'name': 'v', 'path': str(vintage)}])
        receipt = apply_facts(self.db, batch)
        self.assertEqual(receipt['generation'], 1)
        self.assertEqual(receipt['facts_added'], 18)
        self.assertEqual(receipt['facts_shared'], 0)
        self.assertEqual(receipt['fact_count'], 18)
        self.assertEqual([entry['name'] for entry in receipt['reports']],
                         ['c', 'v'])
        snapshot = query_facts(self.db)
        self.assertEqual(snapshot['generation'], 1)
        self.assertEqual(snapshot['fact_count'], 18)
        self.assertEqual([entry['name'] for entry in snapshot['reports']],
                         ['c', 'v'])
        for fact in snapshot['facts']:
            self.assertEqual(fact['reports'], ['c', 'v'])
            for field in ('forecast', 'actual', 'scale'):
                self.assertEqual(str(Fraction(fact[field])), fact[field])
        keys = [(fact['series'], fact['origin'], fact['target_date'],
                 fact['method']) for fact in snapshot['facts']]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), len(keys))

    def test_fixture_reports_are_consumed_without_rerunning_models(self):
        batch = self.manifest('batch.json', 'seed', 0,
                              [{'name': 'cal', 'path': str(self.calendar)},
                               {'name': 'vin', 'path': str(self.vintage)}])
        receipt = apply_facts(self.db, batch)
        self.assertEqual(receipt['fact_count'], 60)
        # Provenance is byte-bound: the digest is the report file's SHA256,
        # not a regenerated model output.
        self.assertEqual(
            receipt['reports'][0]['report_sha256'],
            hashlib.sha256(self.calendar.read_bytes()).hexdigest())
        report = json.loads(self.calendar.read_text())
        self.assertEqual(
            len(query_facts(self.db)['facts']),
            len(report['result']['forecasts']) + len(json.loads(
                self.vintage.read_text())['result']['forecasts']))

    def test_shared_retraction_keeps_fact_until_last_owner_leaves(self):
        calendar, vintage = matching_reports(self.directory)
        apply_facts(self.db, self.manifest('seed.json', 'seed', 0, [
            {'name': 'c', 'path': str(calendar)},
            {'name': 'v', 'path': str(vintage)}]))
        receipt = apply_facts(self.db, self.manifest(
            'drop_c.json', 'drop-c', 1, [], [{'name': 'c'}]))
        self.assertEqual(receipt['facts_removed'], 0)
        self.assertEqual(receipt['fact_count'], 18)
        snapshot = query_facts(self.db)
        self.assertEqual(snapshot['reports'], [
            {'name': 'v', 'kind': 'vintage',
             'source_sha256': json.loads(vintage.read_text())['source_sha256'],
             'report_sha256': hashlib.sha256(vintage.read_bytes()).hexdigest(),
             'fact_count': 18}])
        self.assertTrue(all(fact['reports'] == ['v']
                            for fact in snapshot['facts']))
        receipt = apply_facts(self.db, self.manifest(
            'drop_v.json', 'drop-v', 2, [], [{'name': 'v'}]))
        self.assertEqual(receipt['facts_removed'], 18)
        self.assertEqual(receipt['fact_count'], 0)
        snapshot = query_facts(self.db)
        self.assertEqual(snapshot['facts'], [])
        self.assertEqual(snapshot['reports'], [])

    def test_dropping_unknown_name_is_rejected_and_store_unchanged(self):
        apply_facts(self.db, self.manifest('seed.json', 'seed', 0, [
            {'name': 'cal', 'path': str(self.calendar)}]))
        before = query_facts(self.db)
        with self.assertRaisesRegex(ValueError, 'unknown report'):
            apply_facts(self.db, self.manifest(
                'bad.json', 'bad', 1, [], [{'name': 'ghost'}]))
        self.assertEqual(query_facts(self.db), before)

    def test_batch_names_are_mutually_exclusive(self):
        name = {'name': 'same', 'path': str(self.calendar)}
        with self.assertRaisesRegex(ValueError, 'more than once'):
            manifest = self.manifest('dup_put.json', 'dup', 0, [name, name])
            apply_facts(self.db, manifest)
        with self.assertRaisesRegex(ValueError, 'more than once'):
            manifest = self.manifest('dup_drop.json', 'dup', 0,
                                     [{'name': 'x', 'path': str(self.calendar)}],
                                     [{'name': 'x'}])
            apply_facts(self.db, manifest)


class ConflictTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'store.db'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, kind, facts):
        path = self.directory / name
        report = (build_calendar_report(facts) if kind == 'calendar'
                  else build_vintage_report(facts))
        path.write_text(json.dumps(report))
        return path

    def apply(self, name, txid, generation, puts, drops=None):
        manifest = write_manifest(self.directory, name, txid, generation,
                                  puts, drops)
        return apply_facts(self.db, manifest)

    def test_all_reports_corrected_together_is_accepted(self):
        base = [raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3',
                         5, '4')]
        first = self.write('a.json', 'calendar', base)
        second = self.write('b.json', 'calendar', base)
        self.apply('seed.json', 'seed', 0, [
            {'name': 'a', 'path': str(first)},
            {'name': 'b', 'path': str(second)}])
        corrected = [dict(base[0], actual=7)]
        replacement = self.write('corrected.json', 'calendar', corrected)
        receipt = self.apply(
            'repair.json', 'repair', 1,
            [{'name': 'fresh', 'path': str(replacement)}],
            [{'name': 'a'}, {'name': 'b'}])
        self.assertEqual(receipt['fact_count'], 1)
        snapshot = query_facts(self.db)
        self.assertEqual(snapshot['facts'][0]['actual'], '7')
        self.assertEqual(snapshot['facts'][0]['reports'], ['fresh'])

    def test_single_sided_correction_is_rejected(self):
        base = [raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3',
                         5, '4')]
        first = self.write('a.json', 'calendar', base)
        second = self.write('b.json', 'calendar', base)
        self.apply('seed.json', 'seed', 0, [
            {'name': 'a', 'path': str(first)},
            {'name': 'b', 'path': str(second)}])
        corrected = [dict(base[0], actual=7)]
        replacement = self.write('corrected.json', 'calendar', corrected)
        before = query_facts(self.db)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.apply('one_sided.json', 'one-sided', 1,
                       [{'name': 'c', 'path': str(replacement)}],
                       [{'name': 'a'}])
        self.assertEqual(query_facts(self.db), before)

    def test_whole_batch_refused_when_any_fact_differs(self):
        first = self.write('a.json', 'calendar', [
            raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                     '4')])
        second = self.write('b.json', 'calendar', [
            raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '4', 5,
                     '4')])
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.apply('seed.json', 'seed', 0, [
                {'name': 'a', 'path': str(first)},
                {'name': 'b', 'path': str(second)}])
        self.assertFalse(self.db.exists())

    def test_single_report_repeating_a_fact_is_rejected(self):
        fact = raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                        '4')
        report = self.write('repeat.json', 'calendar', [fact, dict(fact)])
        with self.assertRaisesRegex(ValueError, 'repeats'):
            self.apply('seed.json', 'seed', 0,
                       [{'name': 'r', 'path': str(report)}])
        self.assertFalse(self.db.exists())

    def test_noncanonical_values_are_rejected(self):
        for forecast in ('2/4', '1.5', '1/-2'):
            report = self.write('bad.json', 'calendar', [
                raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1,
                         forecast, 5, '4')])
            with self.assertRaisesRegex(ValueError, 'Fraction'):
                self.apply('seed.json', 'seed', 0,
                           [{'name': 'r', 'path': str(report)}])
            self.assertFalse(self.db.exists())

    def test_negative_actual_or_scale_is_rejected(self):
        for field in ('actual', 'scale'):
            report = self.write('bad.json', 'calendar', [
                raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3',
                         5, '4')])
            payload = json.loads(report.read_text())
            payload['result']['forecasts'][0][field] = '-1'
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                self.apply('seed.json', 'seed', 0,
                           [{'name': 'r', 'path': str(report)}])
            self.assertFalse(self.db.exists())


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'store.db'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, facts):
        path = self.directory / name
        path.write_text(json.dumps(build_calendar_report(facts)))
        return path

    def apply(self, name, txid, generation, puts, drops=None):
        manifest = write_manifest(self.directory, name, txid, generation,
                                  puts, drops)
        return apply_facts(self.db, manifest)

    def test_generation_must_match_exactly(self):
        report = self.write('a.json', [
            raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                     '4')])
        self.apply('seed.json', 'seed', 0,
                   [{'name': 'a', 'path': str(report)}])
        before = query_facts(self.db)
        with self.assertRaisesRegex(ValueError, 'generation 0 but store is '
                                                'at generation 1'):
            self.apply('stale.json', 'stale', 0,
                       [{'name': 'b', 'path': str(report)}])
        with self.assertRaisesRegex(ValueError, 'generation'):
            self.apply('future.json', 'future', 2,
                       [{'name': 'b', 'path': str(report)}])
        self.assertEqual(query_facts(self.db), before)
        self.assertEqual(before['generation'], 1)

    def test_retry_same_txid_same_content_returns_original_receipt(self):
        report = self.write('a.json', [
            raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                     '4')])
        first = self.apply('seed.json', 'seed', 0,
                           [{'name': 'a', 'path': str(report)}])
        # Same report bytes under a different path and a stale expected
        # generation: fingerprints bind bytes, so this replays the receipt.
        copy = self.directory / 'copy.json'
        copy.write_bytes(report.read_bytes())
        replay = self.apply('retry.json', 'seed', 0,
                            [{'name': 'a', 'path': str(copy)}])
        self.assertEqual(replay, first)
        self.assertEqual(query_facts(self.db)['generation'], 1)
        # The CLI also replays.
        manifest = write_manifest(self.directory, 'retry2.json', 'seed',
                                  generation=7,
                                  puts=[{'name': 'a', 'path': str(copy)}])
        result = run_store('apply', '--db', str(self.db),
                           '--manifest', str(manifest))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), first)

    def test_retry_same_txid_different_content_is_rejected(self):
        first = self.write('a.json', [
            raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                     '4')])
        second = self.write('b.json', [
            raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                     '4'),
            raw_fact('s', '2026-01-04', '2026-01-05', 'naive', 1, '4', 0,
                     '4')])
        self.apply('seed.json', 'seed', 0,
                   [{'name': 'a', 'path': str(first)}])
        before = query_facts(self.db)
        with self.assertRaisesRegex(ValueError, 'different content'):
            self.apply('retry.json', 'seed', 1,
                       [{'name': 'a', 'path': str(second)}])
        self.assertEqual(query_facts(self.db), before)
        result = run_store(
            'apply', '--db', str(self.db),
            '--manifest', str(write_manifest(
                self.directory, 'retry2.json', 'seed', 1,
                [{'name': 'a', 'path': str(second)}])))
        self.assertEqual(result.returncode, 2)

    def test_retry_same_txid_ignores_manifest_order(self):
        first = self.write('a.json', [
            raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                     '4')])
        second = self.write('b.json', [
            raw_fact('t', '2026-01-03', '2026-01-04', 'naive', 1, '9', 0,
                     '7')])
        receipt = self.apply('seed.json', 'seed', 0, [
            {'name': 'a', 'path': str(first)},
            {'name': 'b', 'path': str(second)}])
        replay = self.apply('retry.json', 'seed', 0, [
            {'name': 'b', 'path': str(second)},
            {'name': 'a', 'path': str(first)}])
        self.assertEqual(replay, receipt)
        self.assertEqual(query_facts(self.db)['generation'], 1)

    def test_same_report_bytes_under_two_names_share_facts(self):
        base = [raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                         '4')]
        original = self.write('original.json', base)
        copy = self.directory / 'copy.json'
        copy.write_bytes(original.read_bytes())
        receipt = self.apply('seed.json', 'seed', 0, [
            {'name': 'one', 'path': str(original)},
            {'name': 'two', 'path': str(copy)}])
        self.assertEqual(receipt['facts_added'], 1)
        self.assertEqual(receipt['facts_shared'], 0)
        snapshot = query_facts(self.db)
        self.assertEqual(snapshot['facts'][0]['reports'], ['one', 'two'])
        self.assertEqual(snapshot['reports'][0]['report_sha256'],
                         snapshot['reports'][1]['report_sha256'])


class CrashAndConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'store.db'

    def tearDown(self):
        self.temp.cleanup()

    def setUpStore(self):
        report = self.directory / 'a.json'
        report.write_text(json.dumps(build_calendar_report([
            raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                     '4')])))
        manifest = write_manifest(self.directory, 'seed.json', 'seed', 0,
                                  [{'name': 'a', 'path': str(report)}])
        apply_facts(self.db, manifest)
        return report

    def crash_apply(self, manifest, fault):
        environment = dict(os.environ, DEMAND_STORE_FAULT=fault,
                           PYTHONPATH=str(ROOT / 'src'))
        return subprocess.run(
            [sys.executable, '-m', 'demand_backtest.store_cli', 'apply',
             '--db', str(self.db), '--manifest', str(manifest)],
            capture_output=True, env=environment)

    def test_crash_before_commit_rolls_back_all_changes(self):
        self.setUpStore()
        before = query_facts(self.db)
        report = self.directory / 'b.json'
        report.write_text(json.dumps(build_calendar_report([
            raw_fact('t', '2026-01-03', '2026-01-04', 'naive', 1, '9', 0,
                     '7')])))
        manifest = write_manifest(self.directory, 'next.json', 'next', 1,
                                  [{'name': 'b', 'path': str(report)}])
        crashed = self.crash_apply(manifest, 'before_commit')
        self.assertEqual(crashed.returncode, _FAILURE)
        self.assertEqual(query_facts(self.db), before)
        self.assertEqual(query_facts(self.db)['generation'], 1)
        # A retry with the fresh expected generation applies normally.
        receipt = apply_facts(self.db, manifest)
        self.assertEqual(receipt['generation'], 2)
        self.assertEqual(receipt['fact_count'], 2)

    def test_crash_after_commit_recovers_by_idempotent_retry(self):
        self.setUpStore()
        report = self.directory / 'b.json'
        report.write_text(json.dumps(build_calendar_report([
            raw_fact('t', '2026-01-03', '2026-01-04', 'naive', 1, '9', 0,
                     '7')])))
        manifest = write_manifest(self.directory, 'next.json', 'next', 1,
                                  [{'name': 'b', 'path': str(report)}])
        crashed = self.crash_apply(manifest, 'after_commit')
        self.assertEqual(crashed.returncode, _CRASHED_AFTER_COMMIT)
        self.assertEqual(query_facts(self.db)['generation'], 2)
        self.assertEqual(query_facts(self.db)['fact_count'], 2)
        # The receipt was never seen, so the operator retries the same batch;
        # its expected generation is now stale, yet the ledger replays it.
        receipt = apply_facts(self.db, manifest)
        self.assertEqual(receipt['generation'], 2)
        self.assertEqual(query_facts(self.db)['generation'], 2)

    def test_two_processes_racing_the_same_generation(self):
        self.setUpStore()
        first = self.directory / 'b.json'
        second = self.directory / 'c.json'
        first.write_text(json.dumps(build_calendar_report([
            raw_fact('t', '2026-01-03', '2026-01-04', 'naive', 1, '9', 0,
                     '7')])))
        second.write_text(json.dumps(build_calendar_report([
            raw_fact('u', '2026-01-03', '2026-01-04', 'naive', 1, '8', 0,
                     '6')])))
        manifests = [write_manifest(self.directory, f'race{i}.json', f'race{i}',
                                    1, [{'name': name, 'path': str(path)}])
                     for i, (name, path) in
                     enumerate((('b', first), ('c', second)))]
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        processes = [subprocess.Popen(
            [sys.executable, '-m', 'demand_backtest.store_cli', 'apply',
             '--db', str(self.db), '--manifest', str(manifest)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=environment) for manifest in manifests]
        codes = []
        for process in processes:
            process.communicate()
            codes.append(process.returncode)
        self.assertEqual(sorted(codes), [0, 2])
        snapshot = query_facts(self.db)
        self.assertEqual(snapshot['generation'], 2)
        self.assertEqual(snapshot['fact_count'], 2)
        self.assertEqual(len(snapshot['reports']), 2)

    def test_read_write_interleaving_snapshot(self):
        self.setUpStore()
        # The reader holds an open read transaction.  A concurrent apply may
        # commit new facts, but every later read in this transaction still sees
        # the single generation it started on -- never a half-mixed state.
        reader = sqlite3.connect(str(self.db), isolation_level=None,
                                 timeout=30)
        reader.row_factory = sqlite3.Row
        reader.execute('PRAGMA query_only=ON')
        reader.execute('BEGIN')
        first = reader.execute(
            "SELECT value AS generation,"
            ' (SELECT COUNT(*) FROM facts) AS facts,'
            ' (SELECT COUNT(*) FROM reports) AS reports FROM meta'
            " WHERE key='generation'").fetchone()
        self.assertEqual({'generation': first['generation'],
                          'facts': first['facts'],
                          'reports': first['reports']},
                         {'generation': '1', 'facts': 1, 'reports': 1})
        report = self.directory / 'b.json'
        report.write_text(json.dumps(build_calendar_report([
            raw_fact('t', '2026-01-03', '2026-01-04', 'naive', 1, '9', 0,
                     '7')])))
        manifest = write_manifest(self.directory, 'next.json', 'next', 1,
                                  [{'name': 'b', 'path': str(report)}])
        apply_facts(self.db, manifest)
        second = reader.execute(
            "SELECT value AS generation,"
            ' (SELECT COUNT(*) FROM facts) AS facts,'
            ' (SELECT COUNT(*) FROM reports) AS reports FROM meta'
            " WHERE key='generation'").fetchone()
        self.assertEqual({'generation': second['generation'],
                          'facts': second['facts'],
                          'reports': second['reports']},
                         {'generation': '1', 'facts': 1, 'reports': 1})
        reader.execute('COMMIT')
        reader.close()
        # A fresh reader opened after the commit sees the complete new state.
        snapshot = query_facts(self.db)
        self.assertEqual(snapshot['generation'], 2)
        self.assertEqual(snapshot['fact_count'], 2)
        self.assertEqual(len(snapshot['reports']), 2)

    def test_two_read_snapshots_never_mix_generations(self):
        self.setUpStore()
        before = query_facts(self.db)
        report = self.directory / 'b.json'
        report.write_text(json.dumps(build_calendar_report([
            raw_fact('t', '2026-01-03', '2026-01-04', 'naive', 1, '9', 0,
                     '7')])))
        apply_facts(self.db, write_manifest(
            self.directory, 'next.json', 'next', 1,
            [{'name': 'b', 'path': str(report)}]))
        after = query_facts(self.db)
        self.assertEqual(before['generation'], 1)
        self.assertEqual(len(before['reports']), 1)
        self.assertEqual(len(before['facts']), 1)
        self.assertEqual(after['generation'], 2)
        self.assertEqual(len(after['reports']), 2)
        self.assertEqual(len(after['facts']), 2)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'store.db'

    def tearDown(self):
        self.temp.cleanup()

    def test_single_report_with_200000_facts_applies_and_queries(self):
        # One report at the documented per-report ceiling: 1,000 origins x
        # 200 series.  Apply and query must both complete exactly.
        facts = []
        start = date(2020, 1, 1)
        for group in range(1000):
            origin = (start + timedelta(days=group)).isoformat()
            for index in range(200):
                facts.append(raw_fact(
                    f's{index:03d}', origin, '2030-12-01', 'naive', 1,
                    str((index + group) % 17), (index * 7 + group) % 11, '4'))
        self.assertEqual(len(facts), 200000)
        report = self.directory / 'large.json'
        report.write_text(json.dumps(build_calendar_report(facts)))
        source_bytes = report.read_bytes()
        manifest = write_manifest(self.directory, 'large.json.manifest',
                                  'large', 0,
                                  [{'name': 'large', 'path': str(report)}])
        receipt = apply_facts(self.db, manifest)
        self.assertEqual(receipt['facts_added'], 200000)
        self.assertEqual(receipt['fact_count'], 200000)
        self.assertEqual(report.read_bytes(), source_bytes)
        snapshot = query_facts(self.db)
        self.assertEqual(snapshot['fact_count'], 200000)
        self.assertEqual(snapshot['generation'], 1)
        keys = [(f['series'], f['origin'], f['target_date'], f['method'])
                for f in snapshot['facts']]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), 200000)
        # Sharing the same report again adds no facts but keeps both owners.
        media = write_manifest(self.directory, 'share.json', 'share', 1,
                               [{'name': 'copy', 'path': str(report)}])
        shared = apply_facts(self.db, media)
        self.assertEqual(shared['facts_added'], 0)
        self.assertEqual(shared['facts_shared'], 200000)
        # Dropping both owners empties the store without touching the report.
        gone = write_manifest(self.directory, 'gone.json', 'gone', 2, [],
                              [{'name': 'large'}, {'name': 'copy'}])
        emptied = apply_facts(self.db, gone)
        self.assertEqual(emptied['facts_removed'], 200000)
        self.assertEqual(emptied['fact_count'], 0)
        self.assertEqual(report.read_bytes(), source_bytes)

    def test_over_200000_facts_in_one_report_is_rejected(self):
        facts = [raw_fact('all', f'2026-{i // 331 + 1:02d}-{i % 331 + 1:02d}',
                          '2027-12-01', 'naive', 1, '1', 0, '1')
                 for i in range(200001)]
        report = self.directory / 'huge.json'
        report.write_text(json.dumps(build_calendar_report(facts)))
        manifest = write_manifest(self.directory, 'huge.json.manifest', 'huge',
                                  0, [{'name': 'h', 'path': str(report)}])
        with self.assertRaisesRegex(ValueError, '200000'):
            apply_facts(self.db, manifest)
        self.assertFalse(self.db.exists())


class PathAndCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'store.db'

    def tearDown(self):
        self.temp.cleanup()

    def test_store_must_not_overwrite_sources_or_manifest(self):
        report = self.directory / 'a.json'
        report.write_text(json.dumps(build_calendar_report([
            raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                     '4')])))
        manifest = write_manifest(self.directory, 'm.json', 'seed', 0,
                                  [{'name': 'a', 'path': str(report)}])
        with self.assertRaisesRegex(ValueError, 'must not overwrite'):
            apply_facts(report, manifest)
        with self.assertRaisesRegex(ValueError, 'must differ'):
            apply_facts(manifest, manifest)
        self.assertEqual(json.loads(report.read_text())['schema_version'], 1)
        result = run_store('apply', '--db', str(report),
                           '--manifest', str(manifest))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(report.read_text())['schema_version'], 1)

    def test_symlinked_store_is_rejected(self):
        target = self.directory / 'real.db'
        target.write_text('')
        link = self.directory / 'link.db'
        os.symlink(target, link)
        report = self.directory / 'a.json'
        report.write_text(json.dumps(build_calendar_report([])))
        manifest = write_manifest(self.directory, 'm.json', 'seed', 0,
                                  [{'name': 'a', 'path': str(report)}])
        with self.assertRaisesRegex(ValueError, 'symlink'):
            apply_facts(link, manifest)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            query_facts(link)

    def test_bad_manifests_exit_2_without_creating_store(self):
        report = self.directory / 'a.json'
        report.write_text(json.dumps(build_calendar_report([
            raw_fact('s', '2026-01-03', '2026-01-04', 'naive', 1, '3', 5,
                     '4')])))
        bad_manifests = [
            {'expected_generation': 0, 'puts': [], 'drops': []},
            {'txid': '', 'expected_generation': 0, 'puts': [], 'drops': []},
            {'txid': 'x', 'expected_generation': -1, 'puts': [], 'drops': []},
            {'txid': 'x', 'expected_generation': 0, 'puts': {}, 'drops': []},
            {'txid': 'x', 'expected_generation': 0, 'puts': [],
             'drops': [{'name': 'd', 'path': str(report)}]},
            {'txid': 'x', 'expected_generation': 0, 'puts': [
                {'name': 'a', 'path': str(report), 'extra': 1}], 'drops': []},
        ]
        for number, payload in enumerate(bad_manifests):
            path = self.directory / f'bad{number}.json'
            path.write_text(json.dumps(payload))
            result = run_store('apply', '--db', str(self.db),
                               '--manifest', str(path))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(self.db.exists())

    def test_query_of_missing_or_foreign_store_exits_2(self):
        self.assertEqual(run_store('query', '--db', str(self.db)).returncode, 2)
        foreign = self.directory / 'foreign.db'
        connection = sqlite3.connect(str(foreign))
        connection.execute('CREATE TABLE other (x)')
        connection.commit()
        connection.close()
        result = run_store('query', '--db', str(foreign))
        self.assertEqual(result.returncode, 2)

    def test_cli_apply_then_query_outputs_one_snapshot(self):
        calendar, vintage = real_reports(self.directory)
        manifest = write_manifest(self.directory, 'm.json', 'seed', 0,
                                  [{'name': 'cal', 'path': str(calendar)},
                                   {'name': 'vin', 'path': str(vintage)}])
        applied = run_store('apply', '--db', str(self.db),
                            '--manifest', str(manifest))
        self.assertEqual(applied.returncode, 0, applied.stderr)
        receipt = json.loads(applied.stdout)
        self.assertEqual(receipt['fact_count'], 60)
        queried = run_store('query', '--db', str(self.db))
        self.assertEqual(queried.returncode, 0, queried.stderr)
        snapshot = json.loads(queried.stdout)
        self.assertEqual(snapshot['generation'], receipt['generation'])
        self.assertEqual(snapshot['fact_count'], receipt['fact_count'])
        self.assertEqual([entry['name'] for entry in snapshot['reports']],
                         ['cal', 'vin'])
        self.assertEqual(
            hashlib.sha256(calendar.read_bytes()).hexdigest(),
            snapshot['reports'][0]['report_sha256'])


if __name__ == '__main__':
    unittest.main()
