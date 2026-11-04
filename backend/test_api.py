import math
import os
import unittest

os.environ["GEMINI_API_KEY"] = ""  # never call Gemini from tests. Set before main loads .env

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


class TeamsTests(unittest.TestCase):
    def test_lists_every_team_sorted(self):
        teams = client.get("/api/teams").json()

        # Every team in the recent seasons, so 20 for one season and more once relegation mixes in
        self.assertEqual(len(teams), len(main.league))
        self.assertGreaterEqual(len(teams), 20)
        self.assertEqual(teams, sorted(teams))


class TeamProfileTests(unittest.TestCase):
    def test_profile_includes_numbers_the_explanations_quote(self):
        d = client.get("/api/team/Arsenal").json()

        self.assertEqual(d["form_matches"], 10)
        self.assertEqual(d["quality_matches"], 20)
        self.assertNotIn("opponent", d)

    def test_matchup_expected_xg_is_shots_times_average_quality(self):
        d = client.get("/api/team/Arsenal", params={"opponent": "Chelsea", "venue": "away"}).json()

        quality = (d["team_xg_per_shot_against_recently"] + d["opponent_xg_per_shot_recently"]) / 2
        self.assertAlmostEqual(d["expected_xg_against_next_match"],
                               d["expected_shots_against_next_match"] * quality, delta=0.05)
        # Whichever candidate won training, the page names the other one as the one it beat
        model = d["shots_model"]
        self.assertEqual(model["chosen_over"], main.OTHER_SHOTS_CANDIDATE[model["name"]])
        self.assertNotEqual(model["chosen_over"], model["name"])

    def test_falls_back_to_short_summary_without_gemini_key(self):
        d = client.get("/api/team/Arsenal").json()

        self.assertIn("did not load", d["report"])

    def test_unknown_team_is_404(self):
        self.assertEqual(client.get("/api/team/Nowhere FC").status_code, 404)

    def test_team_cannot_play_itself(self):
        res = client.get("/api/team/Arsenal", params={"opponent": "Arsenal"})

        self.assertEqual(res.status_code, 400)


class MatchPredictionTests(unittest.TestCase):
    def strongest_and_weakest(self):
        """The best and worst rated clubs that the app also lists."""
        b = main.MATCH_MODEL
        rating = dict(zip(b["teams"], (b["attack"] + b["defence"]).mean(axis=0)))
        listed = [t for t in rating if t in main.league.index]
        return max(listed, key=rating.get), min(listed, key=rating.get)

    def test_matchup_includes_outcome_chances_that_add_up(self):
        d = client.get("/api/team/Arsenal", params={"opponent": "Chelsea"}).json()["match_prediction"]

        self.assertAlmostEqual(d["p_team_win"] + d["p_draw"] + d["p_team_loss"], 1.0, delta=0.002)
        self.assertEqual(len(d["top_scorelines"]), 3)

    def test_strongest_club_at_home_is_favoured_over_weakest(self):
        strong, weak = self.strongest_and_weakest()

        d = client.get(f"/api/team/{strong}", params={"opponent": weak, "venue": "home"}).json()

        self.assertGreater(d["match_prediction"]["p_team_win"], d["match_prediction"]["p_team_loss"])

    def test_swapping_venue_swaps_who_is_at_home(self):
        home = client.get("/api/team/Arsenal", params={"opponent": "Chelsea", "venue": "home"}).json()
        away = client.get("/api/team/Arsenal", params={"opponent": "Chelsea", "venue": "away"}).json()

        self.assertEqual(home["match_prediction"]["home_team"], "Arsenal")
        self.assertEqual(away["match_prediction"]["home_team"], "Chelsea")

    def test_shots_range_surrounds_the_expected_shots(self):
        d = client.get("/api/team/Arsenal", params={"opponent": "Chelsea"}).json()

        low, high = d["expected_shots_range_80"]
        self.assertLessEqual(low, d["expected_shots_against_next_match"])
        self.assertGreaterEqual(high, d["expected_shots_against_next_match"])


class ModelCardTests(unittest.TestCase):
    def test_returns_the_saved_training_results(self):
        d = client.get("/api/model-card").json()

        self.assertIn("match_model", d)


class ExplainTests(unittest.TestCase):
    def test_returns_xg_and_model_info(self):
        d = client.get("/api/explain", params={"x": 108, "y": 40}).json()

        self.assertTrue(0 < d["xg"] < 1)
        self.assertGreater(d["xg_model_info"]["training_shots"], 0)
        self.assertEqual(d["xg_model_info"], main.xg_model_info())
        self.assertRegex(d["xg_model_info"]["training_seasons"], r"^\d{4}/\d{2}")

    def test_steps_add_up_to_final_xg(self):
        d = client.get("/api/explain", params={"x": 100, "y": 30, "shot_type": "Head"}).json()

        total = sum(s["points"] for s in d["steps"])
        self.assertAlmostEqual(total, (d["xg"] - d["average_shot"]) * 100, delta=0.5)

    def test_formula_numbers_rebuild_the_final_xg(self):
        d = client.get("/api/explain", params={"x": 100, "y": 30, "shot_type": "Head"}).json()

        z = d["start_log_odds"] + sum(s["log_odds"] for s in d["steps"])
        self.assertAlmostEqual(d["log_odds"], z, delta=0.01)
        self.assertAlmostEqual(d["xg"], 1 / (1 + math.exp(-z)), delta=0.001)

    def test_reports_distance_and_angle_of_the_shot(self):
        # Dead centre, 10.5 m out: angle = atan(7.32 * 10.5 / (10.5² - 3.66²)) = 38.4°
        d = client.get("/api/explain", params={"x": 108, "y": 40}).json()

        self.assertAlmostEqual(d["distance_m"], 10.5, delta=0.05)
        self.assertAlmostEqual(d["angle_deg"], 38.4, delta=0.1)

    def test_rejects_shots_off_the_attacking_half(self):
        res = client.get("/api/explain", params={"x": 10, "y": 40})

        self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
