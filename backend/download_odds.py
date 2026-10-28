"""Download bookmaker odds for every season, the benchmark the match model is compared against.
Source: football-data.co.uk, free historical results and odds for English football."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from download_epl import SEASONS

OUT = Path(__file__).parent / "data"
URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"  # E0 is the Premier League

# Best odds first. Closing odds (just before kickoff) are the sharpest forecast the market makes,
# and Pinnacle has the lowest margin, so it is the standard benchmark in betting research
ODDS_SOURCES = [
    ("Pinnacle closing", "PSCH", "PSCD", "PSCA"),
    ("Market average closing", "AvgCH", "AvgCD", "AvgCA"),
    ("Bet365 closing", "B365CH", "B365CD", "B365CA"),
    ("Bet365 opening", "B365H", "B365D", "B365A"),
]


def season_code(season):
    """football-data names seasons by both years: 2014 (2014/15) becomes "1415"."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def fair_probabilities(home, draw, away):
    """Turn decimal odds into chances that add up to 1.
    1 / odds overshoots 1 by the bookmaker's margin, so each share is scaled down evenly."""
    raw = np.column_stack([1 / home, 1 / draw, 1 / away])
    return raw / raw.sum(axis=1, keepdims=True)


def tidy_odds(raw, season):
    """One row per match: teams, result and fair home/draw/away chances from the best odds available."""
    probs = np.full((len(raw), 3), np.nan)
    source = np.full(len(raw), None, dtype=object)
    for name, *cols in ODDS_SOURCES:
        if not set(cols) <= set(raw.columns):
            continue
        odds = raw[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        usable = np.isnan(probs[:, 0]) & np.all(odds > 1, axis=1)  # still missing, and real odds
        probs[usable] = fair_probabilities(*odds[usable].T)
        source[usable] = name
    return pd.DataFrame({
        "season": season,
        "home": raw.HomeTeam.to_numpy(),
        "away": raw.AwayTeam.to_numpy(),
        "result": raw.FTR.to_numpy(),
        "p_home": probs[:, 0], "p_draw": probs[:, 1], "p_away": probs[:, 2],
        "source": source,
    })


def download_odds(season):
    # requests brings its own trusted certificates, which this Python's built-in urllib lacks
    response = requests.get(URL.format(code=season_code(season)), timeout=30)
    response.raise_for_status()
    raw = pd.read_csv(io.StringIO(response.content.decode("latin-1")))
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])  # some files end with empty rows
    odds = tidy_odds(raw, season)
    odds.to_csv(OUT / f"odds_{season}.csv", index=False)
    print(f"Season {season}: {len(odds)} matches, {odds.p_home.notna().sum()} with odds "
          f"({odds.source.value_counts().to_dict()})")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for season in SEASONS:
        download_odds(season)
