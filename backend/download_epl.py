import random
import time
from pathlib import Path

import pandas as pd
import requests
from understatapi import UnderstatClient

# Every finished Premier League season Understat has. 2014 means 2014/15, 2025 means 2025/26
SEASONS = range(2014, 2026)

OUT = Path(__file__).parent / "data"  # next to this script, wherever it is run from
OUT.mkdir(exist_ok=True)


def save_matches(matches, season):
    """Save one row per played match: teams, final score, xG and Understat's outcome chances.
    The match model needs real scores (the shot files leave out own goals). Understat's chances
    are worked out after the match from its own xG, so they are a hindsight ceiling, not a forecast."""
    played = [m for m in matches if m["isResult"]]
    pd.DataFrame({
        "match_id": [int(m["id"]) for m in played],
        "date": [m["datetime"] for m in played],
        "season": season,
        "home": [m["h"]["title"] for m in played],
        "away": [m["a"]["title"] for m in played],
        "home_goals": [int(m["goals"]["h"]) for m in played],
        "away_goals": [int(m["goals"]["a"]) for m in played],
        "home_xg_understat": [float(m["xG"]["h"]) for m in played],
        "away_xg_understat": [float(m["xG"]["a"]) for m in played],
        "understat_p_home": [float(m["forecast"]["w"]) for m in played],
        "understat_p_draw": [float(m["forecast"]["d"]) for m in played],
        "understat_p_away": [float(m["forecast"]["l"]) for m in played],
    }).to_csv(OUT / f"matches_{season}.csv", index=False)


def fetch_shots(match_id, tries=5):
    """Fetch one match's shots. If the connection drops, wait and retry on a fresh session."""
    for attempt in range(tries):
        try:
            # A new client per request, so one dropped connection can't poison the next
            with UnderstatClient() as understat:
                return understat.match(match=match_id).get_shot_data()
        except requests.exceptions.RequestException as e:
            wait = 10 * (attempt + 1)  # wait longer after each failure: 10s, 20s, 30s...
            print(f"Match {match_id} failed ({type(e).__name__}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Match {match_id} failed {tries} times. Run the script again later")


def download_season(season):
    """Download every shot of one season into a single CSV."""
    out_file = OUT / f"shots_{season}.csv"
    with UnderstatClient() as understat:
        matches = understat.league(league="EPL").get_match_data(season=str(season))
    save_matches(matches, season)
    played = [m for m in matches if m["isResult"]]  # skip matches not yet played

    # If the script stopped before, remember which matches are already saved
    done = set()
    if out_file.exists():
        done = set(pd.read_csv(out_file).match_id)

    for i, m in enumerate(played):
        if int(m["id"]) in done:
            continue
        shots = fetch_shots(m["id"])
        rows = pd.DataFrame(shots["h"] + shots["a"])  # home and away shots together
        if not rows.empty:
            # Append to the file. Write the column names only when the file is new
            rows.to_csv(out_file, mode="a", header=not out_file.exists(), index=False)
        # Slower, slightly random pause so we look less like a bot hammering the site
        time.sleep(2 + random.random())
        if i % 20 == 0:
            print(f"{season}: {i}/{len(played)} matches")

    total = pd.read_csv(out_file)
    print(f"Season {season}: {len(total)} shots from {total.match_id.nunique()} matches")


if __name__ == "__main__":  # so other scripts can import SEASONS without starting a download
    for season in SEASONS:
        download_season(season)