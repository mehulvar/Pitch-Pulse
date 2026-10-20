import json
import os
from pathlib import Path


import anthropic
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv
from google import genai

load_dotenv()  # loads GEMINI_API_KEY from the .env file into the environment

BASE = Path(__file__).parent

# Model name lives in one place so it is easy to change
GEMINI_MODEL = "gemini-3.5-flash"

# One row per shot, with our model's xG, created by build_data.py
shots = pd.read_csv(BASE / "shots.csv")

app = FastAPI(title="Pitch Intel")


def matches_played(team):
    """A team played a match if it either took a shot or was shot at."""
    ids = set(shots[shots.team == team].match_id) | set(shots[shots.against == team].match_id)
    return len(ids)


def team_profile(team, opponent=None):
    """All the numbers about how a team defends. Claude only sees these."""
    n = matches_played(team)
    if n == 0:
        raise HTTPException(404, f"Unknown team: {team}")

    # Every shot taken against this team
    conceded = shots[shots.against == team]
    conceded_per_match = len(conceded) / n

    # Split conceded xG into three lanes, from the defending team's view.
    # StatsBomb y runs 0 to 80 and low y is the attacker's left, so it is the defender's right.
    lanes = pd.cut(
        conceded.y, [0, 80 / 3, 160 / 3, 80],
        labels=["right flank", "center", "left flank"], include_lowest=True,
    )
    lane_xg = conceded.groupby(lanes, observed=False).xg.sum()
    total = lane_xg.sum()
    lane_share = {k: round(float(v / total), 2) for k, v in lane_xg.items()} if total else {}

    profile = {
        "team": team,
        "matches": n,
        "shots_against_per_match": round(conceded_per_match, 1),
        "xg_against_per_match": round(conceded.xg.sum() / n, 2),
        "avg_xg_per_shot_conceded": round(conceded.xg.mean(), 3),
        "share_of_xg_conceded_by_lane": lane_share,
        # Raw shots for the map on the frontend
        "shots": conceded[["x", "y", "xg", "goal"]].round(3).to_dict("records"),
    }

    if opponent:
        m = matches_played(opponent)
        if m == 0:
            raise HTTPException(404, f"Unknown team: {opponent}")
        opponent_shots_per_match = len(shots[shots.team == opponent]) / m
        # Simple baseline: average what this team usually concedes
        # with what the opponent usually shoots. A real model replaces this later.
        profile["opponent"] = opponent
        profile["expected_shots_against_next_match"] = round(
            (conceded_per_match + opponent_shots_per_match) / 2, 1
        )
    return profile


def write_report(profile):
    """Gemini turns the numbers into a short scouting report."""
    facts = {k: v for k, v in profile.items() if k != "shots"}  # skip the raw shot list
    fallback = (
        f"{facts['team']} concede {facts['shots_against_per_match']} shots per match "
        f"worth {facts['xg_against_per_match']} expected goals. "
        "(Set GEMINI_API_KEY to get the full report.)"
    )
    if not os.getenv("GEMINI_API_KEY"):
        return fallback

    # The rules for the report. They go in the same input as the numbers
    instructions = (
        "You write scouting reports for football fans and analysts. "
        "Use only the numbers in the data below and never add outside facts. "
        "Write three short sections with these exact labels. "
        "What to expect: how many shots this team concedes and how dangerous they are, "
        "plus the expected shots in the next match if it is in the data. "
        "Where they are weak: use the lane shares and name the flank or the center. "
        "What to do: one or two concrete tactical suggestions for a team attacking them, "
        "based on those weak spots. "
        "Briefly explain xG in plain words the first time you use it. "
        "Keep the whole report under 130 words."
    )
    try:
        client = genai.Client()  # reads GEMINI_API_KEY from the environment
        interaction = client.interactions.create(
            model=GEMINI_MODEL,
            input=instructions + "\n\nData:\n" + json.dumps(facts),
        )
        return interaction.output_text
    except Exception as e:
        # Show the real error in the terminal so it is easy to debug
        print("Gemini call failed:", e)
        return fallback


@app.get("/api/teams")
def list_teams():
    return sorted(set(shots.team) | set(shots.against))


@app.get("/api/team/{team}")
def get_team(team: str, opponent: str | None = None):
    profile = team_profile(team, opponent)
    profile["report"] = write_report(profile)
    return profile


# Serve the frontend from the same server. Mounted last so it doesn't hide the /api routes
app.mount("/", StaticFiles(directory=BASE.parent / "frontend", html=True))