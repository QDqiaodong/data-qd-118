import unittest, json
from datetime import date, timedelta
from pathlib import Path
from demand_backtest.vintage import analyze_vintage

ROOT = Path(__file__).resolve().parents[1]


def day(n):
    return (date(2026, 1, 1) + timedelta(days=n - 1)).isoformat()


def revision(sid, n, published, quantity):
    return {'series': sid, 'date': day(n), 'published_on': day(published), 'quantity': quantity}


class VintageFixtureTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'fixtures/vintage.json').read_text())

    def test_hand_computed_origins_and_facts(self):
        result = analyze_vintage(self.data)
        self.assertEqual(result['mode'], 'revision_vintage')
        self.assertEqual([(o['series'], o['origin'], o['training_count'], o['mase_scale'])
                          for o in result['origins']],
                         [('bolts', '2026-01-03', 3, '2'), ('bolts', '2026-01-04', 4, '2'),
                          ('bolts', '2026-01-05', 5, '2'), ('bolts', '2026-01-06', 6, '112/5'),
                          ('steady', '2026-01-03', 3, '0')])
        # bolts training prefixes at each origin, using only revisions published
        # by that origin: [2,4,6], [2,4,6,8], [2,4,6,9,10], [2,4,60,9,10,12].
        by_origin = {(f['origin'], f['method'], f['horizon']): (f['forecast'], f['actual'])
                     for f in result['forecasts'] if f['series'] == 'bolts'}
        self.assertEqual(by_origin['2026-01-03', 'moving_mean', 1], ('5', 9))
        self.assertEqual(by_origin['2026-01-04', 'moving_mean', 1], ('7', 10))
        self.assertEqual(by_origin['2026-01-05', 'moving_mean', 1], ('19/2', 12))
        self.assertEqual(by_origin['2026-01-06', 'moving_mean', 1], ('11', 14))
        self.assertEqual(by_origin['2026-01-03', 'naive', 2], ('6', 10))
        self.assertEqual(by_origin['2026-01-06', 'naive', 2], ('12', 16))
        self.assertEqual(by_origin['2026-01-04', 'seasonal_naive', 1], ('6', 10))
        self.assertEqual(len(result['forecasts']), 4 * 2 * 3 + 2 * 3)

    def test_hand_computed_metrics_and_zero_scale(self):
        result = analyze_vintage(self.data)
        metrics = {(m['series'], m['method']): m for m in result['metrics']}
        naive = metrics['bolts', 'naive']
        self.assertEqual(naive['count'], 8)
        self.assertEqual(naive['mae'], '25/8')
        self.assertEqual(naive['mse'], '85/8')
        self.assertEqual(naive['bias_forecast_minus_actual'], '-25/8')
        self.assertEqual(naive['wape'], '25/97')
        moving = metrics['bolts', 'moving_mean']
        self.assertEqual(moving['mae'], '4')
        self.assertEqual(moving['mse'], '271/16')
        self.assertEqual(moving['mase'], '173/112')
        # steady is flat: every origin scale is 0, so MASE stays undefined.
        for method in ('moving_mean', 'naive', 'seasonal_naive'):
            flat = metrics['steady', method]
            self.assertEqual((flat['count'], flat['mae'], flat['mse']), (2, '0', '0'))
            self.assertIsNone(flat['mase'])
            self.assertEqual(flat['mase_count'], 0)
        self.assertEqual(metrics['steady', 'naive']['wape'], '0')

    def test_skips_are_explicit_and_incomplete_origins_are_not_scored(self):
        result = analyze_vintage(self.data)
        self.assertEqual(result['skipped'], [
            {'series': 'bolts', 'origin': '2026-01-02',
             'reason': 'insufficient_history', 'detail': None},
            {'series': 'bolts', 'origin': '2026-01-07',
             'reason': 'missing_target', 'detail': '2026-01-09'}])

    def test_lineage_marks_every_version_and_scored_revision(self):
        result = analyze_vintage(self.data)
        lineage = {(e['series'], e['date']): e['versions'] for e in result['lineage']}
        self.assertEqual(lineage['bolts', '2026-01-03'], [
            {'published_on': '2026-01-03', 'quantity': 6, 'latest': False, 'scored': False},
            {'published_on': '2026-01-06', 'quantity': 60, 'latest': True, 'scored': True}])
        self.assertEqual(lineage['bolts', '2026-01-04'], [
            {'published_on': '2026-01-04', 'quantity': 8, 'latest': False, 'scored': False},
            {'published_on': '2026-01-05', 'quantity': 9, 'latest': True, 'scored': True}])
        self.assertEqual(lineage['bolts', '2026-01-01'], [
            {'published_on': '2026-01-01', 'quantity': 2, 'latest': True, 'scored': True}])
        self.assertEqual(result['score_as_of'], '2026-01-08')

    def test_late_publication_never_rewrites_an_earlier_origin(self):
        before = analyze_vintage(self.data)
        early = [f for f in before['forecasts'] if f['origin'] <= '2026-01-05']
        # The 2026-01-06 revision of 2026-01-03 and the 2026-01-05 revision of
        # 2026-01-04 arrive after those origins trained; adding a still later
        # revision must not change any forecast made before it was published.
        self.data['revisions'].append(revision('bolts', 5, 9, 9999))
        after = analyze_vintage(self.data)
        late = [f for f in after['forecasts'] if f['origin'] <= '2026-01-05']
        self.assertEqual(early, late)
        self.assertNotIn(9999, [f['actual'] for f in late])

    def test_later_published_revision_changes_later_origins_only(self):
        origin = '2026-01-06'

        def fact(result, method='moving_mean'):
            return [f for f in result['forecasts']
                    if f['origin'] == origin and f['method'] == method and f['horizon'] == 1][0]

        base = analyze_vintage(self.data)
        self.assertEqual(fact(base)['forecast'], '11')
        # A revision of 01-07 published after the origin cannot enter its prefix.
        self.data['revisions'].append(revision('bolts', 7, 10, 100))
        self.assertEqual(fact(analyze_vintage(self.data))['forecast'], '11')
        # A revision of 01-05 published on the origin date does rewrite training:
        # the moving mean over [01-05, 01-06] becomes (100 + 12)/2.
        self.data['revisions'].append(revision('bolts', 5, 6, 100))
        rewritten = fact(analyze_vintage(self.data))
        self.assertEqual(rewritten['forecast'], '56')
        self.assertEqual(rewritten['actual'], 14)

    def test_score_as_of_caps_the_scored_actual_version(self):
        self.data['score_as_of'] = '2026-01-05'
        capped = analyze_vintage(self.data)
        # With scoring capped at 01-05 the 01-06 revision of 01-03 is invisible.
        self.assertIn({'series': 'bolts', 'origin': '2026-01-06',
                       'reason': 'missing_target_vintage', 'detail': '2026-01-07'},
                      capped['skipped'])
        actual = {(f['origin'], f['target_date'], f['method']): f['actual']
                  for f in capped['forecasts'] if f['series'] == 'bolts'}
        # On 01-04 the 01-05 revision (quantity 9) is already visible.
        self.assertEqual(actual['2026-01-03', '2026-01-04', 'naive'], 9)
        self.assertEqual(capped['forecasts'][0]['actual_published_on'], '2026-01-05')


class VintageContractTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'fixtures/vintage.json').read_text())

    def test_conflicting_same_key_and_publication_rejected_but_repeats_collapse(self):
        self.data['revisions'].append(revision('bolts', 1, 1, 2))
        analyze_vintage(self.data)
        self.data['revisions'].append(revision('bolts', 1, 1, 3))
        with self.assertRaisesRegex(ValueError, 'conflict'):
            analyze_vintage(self.data)

    def test_publication_before_business_day_rejected(self):
        self.data['revisions'].append(revision('bolts', 4, 3, 1))
        with self.assertRaisesRegex(ValueError, 'published_on'):
            analyze_vintage(self.data)

    def test_unknown_fields_duplicate_origins_and_order_collapse(self):
        with self.assertRaises(ValueError):
            analyze_vintage(dict(self.data, extra=1))
        base = analyze_vintage(self.data)
        duplicated = dict(self.data, origins=self.data['origins'] + [
            {'series': 'bolts', 'date': '2026-01-04'},
            {'series': 'steady', 'date': '2026-01-03'}])
        duplicated['revisions'] = list(reversed(self.data['revisions']))
        self.assertEqual(base, analyze_vintage(duplicated))

    def test_origin_on_missing_calendar_day_is_skipped(self):
        self.data['revisions'] = [r for r in self.data['revisions']
                                  if not (r['series'] == 'bolts' and r['date'] == '2026-01-03')]
        self.data['origins'] = [{'series': 'bolts', 'date': '2026-01-04'}]
        result = analyze_vintage(self.data)
        self.assertEqual(result['origins'], [])
        self.assertEqual(result['skipped'], [
            {'series': 'bolts', 'origin': '2026-01-04',
             'reason': 'missing_calendar_date', 'detail': '2026-01-03'}])

    def test_missing_training_vintage_is_skipped(self):
        # A day exists in the series but its first revision was published after
        # the evaluated origin, so that origin cannot use it.
        for r in self.data['revisions']:
            if r['series'] == 'bolts' and r['date'] == '2026-01-02':
                r['published_on'] = '2026-01-06'
        self.data['origins'] = [{'series': 'bolts', 'date': '2026-01-05'}]
        result = analyze_vintage(self.data)
        self.assertEqual(result['origins'], [])
        self.assertEqual(result['skipped'], [
            {'series': 'bolts', 'origin': '2026-01-05',
             'reason': 'missing_training_vintage', 'detail': '2026-01-02'}])

    def test_unknown_origin_series_skipped_and_out_of_order_shuffled_input(self):
        self.data['origins'].append({'series': 'ghost', 'date': '2026-01-03'})
        base = analyze_vintage(self.data)
        self.assertIn({'series': 'ghost', 'origin': '2026-01-03',
                       'reason': 'unknown_series', 'detail': None}, base['skipped'])
        shuffled = dict(self.data, revisions=list(reversed(self.data['revisions'])),
                        origins=list(reversed(self.data['origins'])))
        self.assertEqual(base, analyze_vintage(shuffled))

    def test_oversized_expansion_rejected_before_scoring(self):
        self.data.update(initial_train=2, horizon=100, mean_window=2, season=2)
        self.data['revisions'] = [revision('X', i, i, i) for i in range(1, 1001)]
        self.data['origins'] = [{'series': 'X', 'date': day(i)} for i in range(2, 1001)]
        with self.assertRaisesRegex(ValueError, '200000'):
            analyze_vintage(self.data)

    def test_missing_series_or_parameter_rejected(self):
        with self.assertRaises(ValueError):
            analyze_vintage(dict(self.data, revisions=[]))
        for key in ('score_as_of', 'origins', 'revisions'):
            broken = dict(self.data)
            del broken[key]
            with self.assertRaises(ValueError):
                analyze_vintage(broken)
        with self.assertRaises(ValueError):
            analyze_vintage(dict(self.data, score_as_of='2026-1-8'))
        with self.assertRaises(ValueError):
            analyze_vintage(dict(self.data, mean_window=4))


if __name__ == '__main__':
    unittest.main()
