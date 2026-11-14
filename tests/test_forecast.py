import unittest,json
from pathlib import Path
from demand_backtest.model import analyze
from demand_backtest.forecast import predict
ROOT=Path(__file__).resolve().parents[1]
class ForecastTests(unittest.TestCase):
    def setUp(self):self.d=json.loads((ROOT/'fixtures/input.json').read_text())
    def test_known_multistep_errors(self):
        r=analyze(self.d);self.assertEqual(len(r['origins']),2);self.assertEqual(len(r['forecasts']),12)
        self.assertEqual({x['method']:x['mae'] for x in r['metrics']},{'naive':'3','moving_mean':'4','seasonal_naive':'4'})
    def test_no_future_leakage_in_first_origin(self):
        before=analyze(self.d);self.d['observations'][3]['quantity']=10000;after=analyze(self.d)
        first=lambda r:[x['forecast'] for x in r['forecasts'] if x['origin']=='2026-01-03']
        self.assertEqual(first(before),first(after))
    def test_horizon_longer_than_season_does_not_read_future(self):
        self.assertEqual(predict([2,4,6],5,2,2)['seasonal_naive'],[4,6,4,6,4])
    def test_zero_demand_has_undefined_wape_and_mase(self):
        for x in self.d['observations']:x['quantity']=0
        r=analyze(self.d)
        self.assertTrue(all(x['wape'] is None and x['mase'] is None and x['mase_count']==0 for x in r['metrics']))
    def test_date_gap_duplicate_and_insufficient_history_rejected(self):
        self.d['observations'].pop(2)
        with self.assertRaises(ValueError):analyze(self.d)
        self.setUp();self.d['observations'].append(self.d['observations'][0])
        with self.assertRaises(ValueError):analyze(self.d)
    def test_order_does_not_change_result(self):
        r=analyze(self.d);self.d['observations'].reverse();self.assertEqual(r,analyze(self.d))
    def test_oversized_backtest_rejected_before_expansion(self):
        from datetime import date,timedelta
        self.d.update(initial_train=2,horizon=100,mean_window=2,season=2)
        self.d['observations']=[{'series':'X','date':(date(2026,1,1)+timedelta(days=i)).isoformat(),'quantity':i} for i in range(1000)]
        with self.assertRaisesRegex(ValueError,'200000'):analyze(self.d)
    def test_window_too_large_or_negative_quantity_rejected(self):
        self.d['mean_window']=4
        with self.assertRaises(ValueError):analyze(self.d)
        self.setUp();self.d['observations'][0]['quantity']=-1
        with self.assertRaises(ValueError):analyze(self.d)
if __name__=='__main__':unittest.main()
