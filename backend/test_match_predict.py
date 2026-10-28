import unittest

import numpy as np
from scipy.stats import poisson

from match_predict import outcome_probabilities, predict_match


def bundle(attack, defence, home_adv=0.0, mu=0.3, draws=4):
    """A fake fitted model: every posterior draw has the same strengths."""
    teams = list(attack)
    return {
        "teams": np.array(teams),
        "mu": np.full(draws, mu),
        "home_adv": np.full(draws, home_adv),
        "attack": np.tile([attack[t] for t in teams], (draws, 1)),
        "defence": np.tile([defence[t] for t in teams], (draws, 1)),
    }


class OutcomeProbabilityTests(unittest.TestCase):
    def test_three_outcomes_add_up_to_one(self):
        p, _ = outcome_probabilities(np.array([1.6, 1.2]), np.array([0.9, 1.4]))

        self.assertAlmostEqual(sum(p), 1.0, places=6)

    def test_matches_poisson_maths_for_one_draw(self):
        # A 0-0 needs both sides to score nothing: e^-1.5 * e^-1.0
        _, grid = outcome_probabilities(np.array([1.5]), np.array([1.0]))

        self.assertAlmostEqual(grid[0, 0], poisson.pmf(0, 1.5) * poisson.pmf(0, 1.0))

    def test_equal_rates_give_equal_win_chances(self):
        (home, draw, away), _ = outcome_probabilities(np.array([1.3]), np.array([1.3]))

        self.assertAlmostEqual(home, away)
        self.assertGreater(draw, 0.2)


class PredictMatchTests(unittest.TestCase):
    def test_home_advantage_favours_the_home_side_between_equal_teams(self):
        b = bundle({"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0}, home_adv=0.25)

        d = predict_match(b, "A", "B")

        self.assertGreater(d["p_home"], d["p_away"])

    def test_stronger_attack_is_favoured_away_from_home(self):
        b = bundle({"Strong": 0.5, "Weak": -0.5}, {"Strong": 0.0, "Weak": 0.0})

        d = predict_match(b, "Weak", "Strong")

        self.assertGreater(d["p_away"], d["p_home"])
        self.assertGreater(d["expected_goals_away"], d["expected_goals_home"])

    def test_returns_three_most_likely_scorelines_in_order(self):
        b = bundle({"A": 0.2, "B": 0.0}, {"A": 0.0, "B": 0.0})

        scores = predict_match(b, "A", "B")["top_scorelines"]

        self.assertEqual(len(scores), 3)
        chances = [s["chance"] for s in scores]
        self.assertEqual(chances, sorted(chances, reverse=True))

    def test_unknown_team_raises(self):
        b = bundle({"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0})

        with self.assertRaises(KeyError):
            predict_match(b, "A", "Nowhere FC")


if __name__ == "__main__":
    unittest.main()
