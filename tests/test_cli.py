import unittest, tempfile, subprocess, sys, json, hashlib
from pathlib import Path
from demand_backtest.model import analyze
from demand_backtest.vintage import analyze_vintage
ROOT=Path(__file__).resolve().parents[1]
class CLITests(unittest.TestCase):
    def test_installed_cli_and_exact_source_hash(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'report.json';source=ROOT/'fixtures/input.json'
            command=[sys.executable,'-m','demand_backtest.cli','--source',str(source),'--output',str(out)]
            result=subprocess.run(command,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(out.read_text());self.assertEqual(report['result'],analyze(json.loads(source.read_text())))
            self.assertEqual(report['source_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
            before=out.read_bytes();self.assertEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(before,out.read_bytes())
    def test_vintage_source_hash_report_and_rerun(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'report.json';source=ROOT/'fixtures/vintage.json'
            command=[sys.executable,'-m','demand_backtest.cli','--vintage-source',str(source),'--output',str(out)]
            result=subprocess.run(command,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(out.read_text())
            self.assertEqual(report['result'],analyze_vintage(json.loads(source.read_text())))
            self.assertEqual(report['source_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(report['schema_version'],1)
            before=out.read_bytes();self.assertEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(before,out.read_bytes())
    def test_sources_are_mutually_exclusive(self):
        for flags in (['--source',str(ROOT/'fixtures/input.json'),'--vintage-source',str(ROOT/'fixtures/vintage.json')],[]):
            result=subprocess.run([sys.executable,'-m','demand_backtest.cli',*flags,'--output','/tmp/unused.json'],capture_output=True)
            self.assertEqual(result.returncode,2)
    def test_failed_vintage_update_retains_prior_output(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);source=p/'bad.json';out=p/'report.json';out.write_text('prior')
            for invalid in ['{}','{"a":1,"a":2}','NaN']:
                source.write_text(invalid)
                result=subprocess.run([sys.executable,'-m','demand_backtest.cli','--vintage-source',str(source),'--output',str(out)],capture_output=True)
                self.assertEqual(result.returncode,2);self.assertEqual(out.read_text(),'prior')
            source.write_text(json.dumps({'revisions':[{'series':'s','date':'2026-01-02','published_on':'2026-01-01','quantity':1}],'origins':[],'score_as_of':'2026-01-02','initial_train':2,'horizon':1,'mean_window':1,'season':1}))
            result=subprocess.run([sys.executable,'-m','demand_backtest.cli','--vintage-source',str(source),'--output',str(out)],capture_output=True)
            self.assertEqual(result.returncode,2);self.assertEqual(out.read_text(),'prior')
    def test_failed_update_retains_prior_output_and_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);source=p/'bad.json';out=p/'report.json';out.write_text('prior')
            for invalid in ['{}','{"a":1,"a":2}','NaN']:
                source.write_text(invalid)
                result=subprocess.run([sys.executable,'-m','demand_backtest.cli','--source',str(source),'--output',str(out)],capture_output=True)
                self.assertEqual(result.returncode,2);self.assertEqual(out.read_text(),'prior')
            before=source.read_bytes()
            result=subprocess.run([sys.executable,'-m','demand_backtest.cli','--source',str(source),'--output',str(source)],capture_output=True)
            self.assertEqual(result.returncode,2);self.assertEqual(source.read_bytes(),before)
if __name__=='__main__':unittest.main()
