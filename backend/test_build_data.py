import unittest

import pandas as pd

from build_data import transform_shots


class TransformShotsTests(unittest.TestCase):
    def test_maps_understat_rows_to_app_schema(self):
        raw = pd.DataFrame([
            {
                "match_id": 1, "h_a": "h", "h_team": "Home", "a_team": "Away",
                "X": 0.9, "Y": 0.5, "xG": 0.25, "result": "Goal", "minute": 12,
            },
            {
                "match_id": 1, "h_a": "a", "h_team": "Home", "a_team": "Away",
                "X": 0.8, "Y": 0.25, "xG": 0.1, "result": "SavedShot", "minute": 34,
            },
        ])

        shots = transform_shots(raw)

        self.assertEqual(
            list(shots.columns),
            ["match_id", "team", "against", "x", "y", "xg", "goal", "minute"],
        )
        self.assertEqual(shots["team"].tolist(), ["Home", "Away"])
        self.assertEqual(shots["against"].tolist(), ["Away", "Home"])
        self.assertEqual(shots["x"].tolist(), [114.0, 108.0])
        self.assertEqual(shots["y"].tolist(), [40.0, 20.0])
        self.assertEqual(shots["goal"].tolist(), [True, False])

    def test_rejects_out_of_range_coordinates(self):
        raw = pd.DataFrame([
            {
                "match_id": 1, "h_a": "h", "h_team": "Home", "a_team": "Away",
                "X": 1.1, "Y": 0.5, "xG": 0.25, "result": "MissedShots", "minute": 12,
            },
        ])

        with self.assertRaisesRegex(ValueError, "coordinates must be between 0 and 1"):
            transform_shots(raw)


if __name__ == "__main__":
    unittest.main()
