import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from download_odds import fair_probabilities, season_code, tidy_odds
from team_names import attach_odds, odds_name

DATA = Path(__file__).parent / "data"


class FairProbabilityTests(unittest.TestCase):
    def test_removes_the_bookmaker_margin(self):
        # 1/1.8 + 1/3.6 + 1/4.5 = 1.056: the extra 5.6% is the bookmaker's margin
        p = fair_probabilities(np.array([1.8]), np.array([3.6]), np.array([4.5]))

        self.assertAlmostEqual(p.sum(), 1.0)
        # Shares keep their ratio to each other: 1/1.8 : 1/3.6 : 1/4.5 = 10 : 5 : 4
        np.testing.assert_allclose(p[0], [10 / 19, 5 / 19, 4 / 19])


class SeasonCodeTests(unittest.TestCase):
    def test_uses_two_digit_years_of_both_halves(self):
        self.assertEqual(season_code(2014), "1415")
        self.assertEqual(season_code(2025), "2526")


class TidyOddsTests(unittest.TestCase):
    def raw(self, **odds):
        return pd.DataFrame({"HomeTeam": ["Man United"], "AwayTeam": ["Wolves"], "FTR": ["H"], **odds})

    def test_prefers_pinnacle_closing_odds(self):
        out = tidy_odds(self.raw(PSCH=[2.0], PSCD=[4.0], PSCA=[4.0], B365H=[1.5], B365D=[4.0], B365A=[6.0]), 2020)

        self.assertEqual(out.source[0], "Pinnacle closing")
        self.assertAlmostEqual(out.p_home[0], 0.5)

    def test_falls_back_when_pinnacle_is_missing(self):
        out = tidy_odds(self.raw(PSCH=[np.nan], PSCD=[np.nan], PSCA=[np.nan],
                                 B365H=[2.0], B365D=[4.0], B365A=[4.0]), 2020)

        self.assertEqual(out.source[0], "Bet365 opening")
        self.assertAlmostEqual(out.p_home[0], 0.5)

    def test_keeps_teams_season_and_result(self):
        out = tidy_odds(self.raw(PSCH=[2.0], PSCD=[4.0], PSCA=[4.0]), 2020)

        self.assertEqual(list(out.loc[0, ["season", "home", "away", "result"]]),
                         [2020, "Man United", "Wolves", "H"])


class TeamNameTests(unittest.TestCase):
    def test_maps_long_understat_names_to_bookmaker_names(self):
        self.assertEqual(odds_name("Manchester United"), "Man United")
        self.assertEqual(odds_name("Wolverhampton Wanderers"), "Wolves")

    def test_leaves_names_that_already_match(self):
        self.assertEqual(odds_name("Arsenal"), "Arsenal")


@unittest.skipUnless(list(DATA.glob("matches_*.csv")) and list(DATA.glob("odds_*.csv")),
                     "needs downloaded matches_*.csv and odds_*.csv")
class JoinTests(unittest.TestCase):
    def test_every_downloaded_match_finds_its_odds(self):
        matches = pd.concat([pd.read_csv(f) for f in DATA.glob("matches_*.csv")])
        odds = pd.concat([pd.read_csv(f) for f in DATA.glob("odds_*.csv")])
        odds = odds[odds.season.isin(matches.season.unique())]

        joined = attach_odds(matches, odds)

        missing = joined[joined.p_home.isna()]
        self.assertEqual(len(missing), 0, f"No odds for:\n{missing[['season', 'home', 'away']]}")
        self.assertEqual(len(joined), len(matches))


if __name__ == "__main__":
    unittest.main()
