import random
import time
from pathlib import Path

import pandas as pd
import requests
from understatapi import UnderstatClient

# Start with one season. Once it works we add the rest
SEASONS = [2016]

OUT = Path("data")
OUT.mkdir(exist_ok=True)


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


for season in SEASONS:
    download_season(season)