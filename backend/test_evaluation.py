import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import (
    bootstrap_ci, brier_3way, calibration_table, log_loss_3way, rps, save_metrics,
)
from features import count_range, fit_dispersion, holdout_split, season_folds


class CountRangeTests(unittest.TestCase):
    def test_no_extra_spread_measured_on_poisson_counts(self):
        rng = np.random.default_rng(0)
        mu = rng.uniform(8, 18, 50_000)

        self.assertAlmostEqual(fit_dispersion(rng.poisson(mu), mu), 0.0, delta=0.005)

    def test_recovers_the_extra_spread_of_negative_binomial_counts(self):
        rng = np.random.default_rng(1)
        mu, alpha = rng.uniform(8, 18, 50_000), 0.05
        y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))  # mean mu, variance mu + alpha·mu²

        self.assertAlmostEqual(fit_dispersion(y, mu), alpha, delta=0.005)

    def test_fitted_spread_covers_as_often_as_the_true_spread(self):
        rng = np.random.default_rng(2)
        mu = rng.uniform(8, 18, 50_000)
        y = rng.negative_binomial(1 / 0.05, 1 / (1 + 0.05 * mu))
        inside = lambda bounds: np.mean((y >= bounds[0]) & (y <= bounds[1]))

        fitted = inside(count_range(mu, fit_dispersion(y, mu), (0.1, 0.9)))

        # Whole-number counts with both ends included cover a little over 80%, never under
        self.assertAlmostEqual(fitted, inside(count_range(mu, 0.05, (0.1, 0.9))), delta=0.01)
        self.assertGreaterEqual(fitted, 0.8)
        self.assertLess(inside(count_range(mu, 0.0, (0.1, 0.9))), 0.8)  # Poisson range is too narrow

    def test_zero_spread_gives_the_poisson_range(self):
        low, high = count_range(np.array([12.0]), 0.0, (0.1, 0.9))

        self.assertEqual((low[0], high[0]), (8, 17))  # Poisson(12): 10th and 90th percentiles


class RpsTests(unittest.TestCase):
    def test_certain_and_right_scores_zero(self):
        self.assertAlmostEqual(rps([[1, 0, 0]], [0]), 0.0)

    def test_certain_home_win_but_away_win_scores_one(self):
        # cumulative forecast [1, 1] vs outcome [0, 0] -> (1 + 1) / 2
        self.assertAlmostEqual(rps([[1, 0, 0]], [2]), 1.0)

    def test_worked_example_for_a_draw(self):
        # cumulative forecast [0.5, 0.8] vs outcome [0, 1] -> (0.25 + 0.04) / 2
        self.assertAlmostEqual(rps([[0.5, 0.3, 0.2]], [1]), 0.145)

    def test_averages_over_matches(self):
        self.assertAlmostEqual(rps([[1, 0, 0], [1, 0, 0]], [0, 2]), 0.5)


class ThreeWayScoreTests(unittest.TestCase):
    def test_log_loss_is_minus_log_of_chance_given_to_what_happened(self):
        self.assertAlmostEqual(log_loss_3way([[0.5, 0.3, 0.2]], [2]), -np.log(0.2))

    def test_brier_sums_squared_errors_over_the_three_outcomes(self):
        # (0.5 - 0)² + (0.3 - 1)² + (0.2 - 0)² = 0.25 + 0.49 + 0.04
        self.assertAlmostEqual(brier_3way([[0.5, 0.3, 0.2]], [1]), 0.78)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.df = pd.DataFrame({"match_id": np.repeat(np.arange(200), 5),
                                "value": rng.normal(size=1000)})

    def test_interval_surrounds_the_estimate(self):
        ci = bootstrap_ci(lambda d: d.value.mean(), self.df, "match_id", n=200)

        self.assertAlmostEqual(ci["estimate"], self.df.value.mean())
        self.assertLess(ci["low"], ci["estimate"])
        self.assertGreater(ci["high"], ci["estimate"])

    def test_same_seed_gives_same_interval(self):
        a = bootstrap_ci(lambda d: d.value.mean(), self.df, "match_id", n=100, seed=7)
        b = bootstrap_ci(lambda d: d.value.mean(), self.df, "match_id", n=100, seed=7)

        self.assertEqual(a, b)

    def test_constant_metric_has_zero_width(self):
        ci = bootstrap_ci(lambda d: 3.0, self.df, "match_id", n=50)

        self.assertEqual((ci["low"], ci["high"]), (3.0, 3.0))


class CalibrationTests(unittest.TestCase):
    def test_well_calibrated_forecasts_match_outcomes_per_bin(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 200_000)
        y = rng.uniform(0, 1, p.size) < p  # each event happens with exactly chance p

        table = calibration_table(y, p, bins=5)

        self.assertEqual(len(table), 5)
        for row in table:
            self.assertAlmostEqual(row["predicted"], row["actual"], delta=0.01)
        self.assertEqual(sum(r["count"] for r in table), p.size)


class SaveMetricsTests(unittest.TestCase):
    def test_adds_a_section_without_losing_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            save_metrics("xg", {"brier": 0.08}, path)
            save_metrics("shots", {"mae": 3.7}, path)

            self.assertEqual(json.loads(path.read_text()),
                             {"xg": {"brier": 0.08}, "shots": {"mae": 3.7}})


class SeasonSplitTests(unittest.TestCase):
    def test_holds_out_the_newest_seasons(self):
        dev, holdout = holdout_split([2016, 2014, 2015, 2017, 2018], n_holdout=2)

        self.assertEqual(dev, [2014, 2015, 2016])
        self.assertEqual(holdout, [2017, 2018])

    def test_each_fold_trains_only_on_earlier_seasons(self):
        folds = season_folds([2014, 2015, 2016, 2017], n_folds=2)

        self.assertEqual(folds, [([2014, 2015], 2016), ([2014, 2015, 2016], 2017)])

    def test_rejects_too_few_seasons(self):
        with self.assertRaises(ValueError):
            season_folds([2014, 2015], n_folds=2)


if __name__ == "__main__":
    unittest.main()
