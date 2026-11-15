"""End-to-end CLI tests for the demand-plan replenishment command.

Every test drives the real ``demand-plan`` command as a separate process and
treats the fact store as an existing artifact produced through the real
``demand-store`` / ``demand-backtest`` commands.  A small verifier imported
from :mod:`test_plan` re-checks the published JSON independently.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'fixtures'

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plan import config_for, make_config, verify_plan  # noqa: E402

from demand_backtest import store  # noqa: E402

ORIGIN = '2026-01-01'


def demand_plan(*arguments, env=None):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    if env:
        environment.update(env)
    return subprocess.run(
        [sys.executable, '-m', 'demand_backtest.plan_cli', *arguments],
        capture_output=True, text=True, env=environment)


def demand_store(*arguments):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    return subprocess.run(
        [sys.executable, '-m', 'demand_backtest.store_cli', *arguments],
        capture_output=True, text=True, env=environment)


class PlanCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'facts.db'
        self._build_store()
        self.config = config_for(
            [('a', 1, 1), ('a', 2, 1)],
            [('a', 0, 3, 10, 1)],
            [(0, 5), (1, 5), (2, 5)],
            [('a', 1, 2, 1)])
        self.config_path = self.directory / 'config.json'
        self.config_path.write_text(json.dumps(self.config))
        self.output = self.directory / 'plan.json'

    def tearDown(self):
        self.temp.cleanup()

    def _build_store(self):
        facts = [
            {'series': 'a', 'origin': ORIGIN, 'target_date': '2026-01-02',
             'horizon': 1, 'method': 'naive', 'forecast': '2', 'actual': 0,
             'scale': '1'},
            {'series': 'a', 'origin': ORIGIN, 'target_date': '2026-01-03',
             'horizon': 2, 'method': 'naive', 'forecast': '5/2', 'actual': 0,
             'scale': '1'}]
        report = {'schema_version': 1,
                  'source_sha256': 'a' * 64,
                  'result': {'origins': [], 'forecasts': facts,
                             'metrics': [], 'evaluation_unit': 'x'}}
        report_path = self.directory / 'report.json'
        report_path.write_text(json.dumps(report))
        manifest = self.directory / 'manifest.json'
        manifest.write_text(json.dumps({
            'txid': 'seed', 'expected_generation': 0,
            'puts': [{'name': 'report', 'path': str(report_path)}],
            'drops': []}))
        store.apply_facts(self.db, manifest)

    def test_success_writes_atomic_report_and_prints_it(self):
        result = demand_plan('--db', str(self.db),
                             '--config', str(self.config_path),
                             '--output', str(self.output))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['schema_version'], 1)
        self.assertEqual(report['mode'], 'demand_plan')
        on_disk = json.loads(self.output.read_text())
        self.assertEqual(on_disk, report)
        # The output binds the exact config bytes and the store generation.
        self.assertEqual(report['config_sha256'],
                         hashlib.sha256(
                             self.config_path.read_bytes()).hexdigest())
        self.assertEqual(report['result']['generation'], 1)
        verify_plan(report, self.config)
        # Forecast 5/2 is rounded up to 3, so total demand is 5.
        self.assertEqual(report['result']['flow']['total_demand'], 5)

    def test_bad_input_exits_2_and_preserves_old_output(self):
        good = demand_plan('--db', str(self.db),
                           '--config', str(self.config_path),
                           '--output', str(self.output))
        self.assertEqual(good.returncode, 0, good.stderr)
        before = self.output.read_bytes()
        broken = self.directory / 'broken.json'
        broken.write_text(json.dumps({'origin': ORIGIN, 'method': 'naive'}))
        failed = demand_plan('--db', str(self.db),
                             '--config', str(broken),
                             '--output', str(self.directory / 'new.json'))
        self.assertEqual(failed.returncode, 2)
        self.assertIn('expected object fields', failed.stderr)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertFalse((self.directory / 'new.json').exists())

    def test_output_collision_and_symlink_are_rejected(self):
        for target in (self.db, self.config_path):
            result = demand_plan('--db', str(self.db),
                                 '--config', str(self.config_path),
                                 '--output', str(target))
            self.assertEqual(result.returncode, 2)
        link = self.directory / 'link.json'
        link.symlink_to(self.output)
        result = demand_plan('--db', str(self.db),
                             '--config', str(self.config_path),
                             '--output', str(link))
        self.assertEqual(result.returncode, 2)

    def test_infeasible_plan_exits_0_with_min_cut_certificate(self):
        tight = config_for(
            [('a', 1, 1), ('a', 2, 1)],
            [('a', 0, 2, 10, 1)],
            [(0, 0), (1, 0), (2, 0)], [])
        tight_path = self.directory / 'tight.json'
        tight_path.write_text(json.dumps(tight))
        result = demand_plan('--db', str(self.db),
                             '--config', str(tight_path),
                             '--output', str(self.output))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['result']['status'], 'infeasible')
        verify_plan(report, tight)
        self.assertTrue(report['result']['certificate']['cut_edges'])

    def test_source_store_is_never_modified(self):
        before = self.db.read_bytes()
        query = demand_store('query', '--db', str(self.db))
        self.assertEqual(query.returncode, 0, query.stderr)
        snapshot = json.loads(query.stdout)
        result = demand_plan('--db', str(self.db),
                             '--config', str(self.config_path),
                             '--output', str(self.output))
        self.assertEqual(result.returncode, 0, result.stderr)
        # The read-only planner cannot change the store file or its WAL.
        self.assertEqual(self.db.read_bytes(), before)
        wal = Path(str(self.db) + '-wal')
        wal_before = wal.read_bytes() if wal.exists() else None
        result = demand_plan('--db', str(self.db),
                             '--config', str(self.config_path),
                             '--output', str(self.directory / 'again.json'))
        self.assertEqual(result.returncode, 0, result.stderr)
        if wal_before is None:
            self.assertFalse(wal.exists())
        else:
            self.assertEqual(wal.read_bytes(), wal_before)
        again = demand_store('query', '--db', str(self.db))
        self.assertEqual(json.loads(again.stdout), snapshot)

    def test_missing_store_and_help(self):
        missing = demand_plan('--db', str(self.directory / 'nope.db'),
                              '--config', str(self.config_path),
                              '--output', str(self.output))
        self.assertEqual(missing.returncode, 2)
        self.assertEqual(demand_plan('--help').returncode, 0)
        no_args = demand_plan()
        self.assertEqual(no_args.returncode, 2)

    def test_fixture_backtest_flows_through_real_commands(self):
        backtest = self.directory / 'backtest.json'
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        run = subprocess.run(
            [sys.executable, '-m', 'demand_backtest.cli',
             '--source', str(FIXTURES / 'input.json'),
             '--output', str(backtest)],
            capture_output=True, text=True, env=env)
        self.assertEqual(run.returncode, 0, run.stderr)
        manifest = self.directory / 'fixture-manifest.json'
        manifest.write_text(json.dumps({
            'txid': 'fixture', 'expected_generation': 1,
            'puts': [{'name': 'backtest', 'path': str(backtest)}],
            'drops': []}))
        applied = demand_store('apply', '--db', str(self.db),
                               '--manifest', str(manifest))
        self.assertEqual(applied.returncode, 0, applied.stderr)
        config = json.loads((FIXTURES / 'plan_config.json').read_text())
        config_path = self.directory / 'fixture-config.json'
        config_path.write_text(json.dumps(config))
        result = demand_plan('--db', str(self.db),
                             '--config', str(config_path),
                             '--output', str(self.output))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        verify_plan(report, config)
        self.assertEqual(report['result']['total_cost'], 30)


if __name__ == '__main__':
    unittest.main()
