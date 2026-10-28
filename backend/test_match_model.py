import unittest

import numpy as np
import pandas as pd

from match_model import fill_unseen, fit_strengths, time_weights


class TimeWeightTests(unittest.TestCase):
    def test_a_match_one_half_life_ago_counts_half(self):
        dates = pd.to_datetime(["2020-01-01", "2020-07-01"])

        w = time_weights(dates, as_of=pd.Timestamp("2020-07-01"), half_life_days=182)

        self.assertAlmostEqual(w[0], 0.5, delta=0.01)
        self.assertAlmostEqual(w[1], 1.0)


class FillUnseenTests(unittest.TestCase):
    def test_promoted_club_starts_at_the_level_of_the_weakest_clubs(self):
        bundle = {"teams": np.array(["A", "B", "C", "D", "New"]),
                  "attack": np.array([[0.4, 0.2, -0.2, -0.4, 9.0]]),
                  "defence": np.array([[0.3, 0.1, -0.1, -0.3, 9.0]]),
                  "mu": np.array([0.3]), "home_adv": np.array([0.2])}

        filled = fill_unseen(bundle, ["New"], n_weakest=2)

        self.assertAlmostEqual(filled["attack"][0, 4], -0.3)   # mean of -0.2 and -0.4
        self.assertAlmostEqual(filled["defence"][0, 4], -0.2)  # mean of -0.1 and -0.3
        self.assertEqual(bundle["attack"][0, 4], 9.0)  # the original is left alone


class FitTests(unittest.TestCase):
    def test_finds_the_team_that_scores_the_most(self):
        rng = np.random.default_rng(0)
        teams = ["Strong", "Mid", "Weak", "Other"]
        rows = []
        for day in range(120):
            home, away = rng.choice(teams, 2, replace=False)
            rate = {"Strong": 2.5, "Mid": 1.3, "Weak": 0.6, "Other": 1.2}
            rows.append({"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                         "home": home, "away": away,
                         "home_goals": rng.poisson(rate[home]), "away_goals": rng.poisson(rate[away])})
        matches = pd.DataFrame(rows)

        b = fit_strengths(matches, teams, target="goals", as_of=pd.Timestamp("2020-05-01"),
                          half_life_days=10_000, draws=300, tune=300)

        attack = dict(zip(b["teams"], b["attack"].mean(axis=0)))
        self.assertEqual(max(attack, key=attack.get), "Strong")
        self.assertEqual(min(attack, key=attack.get), "Weak")
        self.assertEqual(b["attack"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
