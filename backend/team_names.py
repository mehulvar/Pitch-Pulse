"""Understat and football-data.co.uk spell some clubs differently. This lines them up."""
import pandas as pd

# Understat name -> football-data.co.uk name. Clubs spelled the same in both are left out
UNDERSTAT_TO_ODDS = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
}


def odds_name(understat_name):
    return UNDERSTAT_TO_ODDS.get(understat_name, understat_name)


def attach_odds(matches, odds):
    """Add the bookmaker's home/draw/away chances to each Understat match.
    Each pair of clubs meets once at each ground per season, so season + home + away is unique."""
    keyed = matches.assign(home_odds_name=matches.home.map(odds_name),
                           away_odds_name=matches.away.map(odds_name))
    cols = ["season", "home", "away", "p_home", "p_draw", "p_away", "source"]
    return keyed.merge(
        odds[cols].rename(columns={"home": "home_odds_name", "away": "away_odds_name"}),
        on=["season", "home_odds_name", "away_odds_name"], how="left",
    ).drop(columns=["home_odds_name", "away_odds_name"])
