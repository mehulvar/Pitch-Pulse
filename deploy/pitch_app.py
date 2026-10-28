import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from google.genai import types
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from google import genai
import time
from concurrent.futures import ThreadPoolExecutor
from features import (
    FEATURES, FORM_WINDOW, QUALITY_WINDOW, SET_PIECE_SITUATIONS, add_form, factor_of,
    latest_form, latest_quality, load_shots, score_shots, team_match_table, xg_features,
)

load_dotenv()
BASE = Path(__file__).parent
GEMINI_MODEL = "gemini-3.5-flash-lite"
POOL = ThreadPoolExecutor(max_workers=2)  # lets us stop waiting on Gemini after a time limit

# ---- Runs once when the server starts ----
xg_model = joblib.load(BASE / "models" / "xg_model.joblib")           # chance a shot scores
shots_bundle = joblib.load(BASE / "models" / "shots_model.joblib")    # shots a team concedes

shots = load_shots()
shots["our_xg"] = score_shots(shots, xg_model)  # our xG model scores every shot
tm = add_form(team_match_table(shots, xg_col="our_xg"))  # one row per team per match

# The numbers on the page describe the latest 3 seasons in the data
first_season = int(shots.season.max()) - 2
recent_shots = shots[shots.season >= first_season]
recent_tm = tm[tm.season >= first_season]


def season_label(s):
    return f"{int(s)}/{(int(s) + 1) % 100:02d}"


def seasons_window(df):
    """Readable span of the seasons in a table, like 2016/17 or 2014/15 to 2016/17."""
    seasons = sorted(df.season.unique())
    if len(seasons) == 1:
        return season_label(seasons[0])
    return f"{season_label(seasons[0])} to {season_label(seasons[-1])}"


window = seasons_window(recent_shots)

# What the xG model learned from, so the page can say so
XG_MODEL_INFO = {"training_shots": len(shots), "training_seasons": seasons_window(shots)}

# League table: one row per team. It gives every number something to be compared against
by_team = recent_tm.groupby("team")
league = pd.DataFrame({
    "matches": by_team.size(),
    "shots_against": by_team.shots_against.mean(),
    "xg_against": by_team.xg_against.mean(),
    "xg_per_shot_against": by_team.xg_against.sum() / by_team.shots_against.sum(),
    "shots_for": by_team.shots_for.mean(),
    "xg_per_shot_for": by_team.xg_for.sum() / by_team.shots_for.sum(),
})


def shares(df):
    """Where the xG of a group of shots came from. Each value is a share of the total, 0 to 1."""
    total = df.our_xg.sum()
    if total == 0:
        return {}

    def share(mask):
        return round(float(df.our_xg[mask].sum() / total), 2)

    # Coordinates are fractions of the pitch. Wide vs central avoids guessing left from right
    
    inside_box = (df.X > 0.843) & df.Y.between(0.204, 0.796)
    return {
        "inside_penalty_box": share(inside_box),
        "set_pieces_and_penalties": share(df.situation.isin(SET_PIECE_SITUATIONS)),
        "headers": share(df.shotType == "Head"),
    }


LEAGUE_SHARES = shares(recent_shots)  # computed once for comparison

# The two candidates in train_shots.py. Whichever won, the other is the one it beat
OTHER_SHOTS_CANDIDATE = {
    "Poisson regression": "LightGBM (Poisson)",
    "LightGBM (Poisson)": "Poisson regression",
}


def team_profile(team, opponent=None, venue="home"):
    """All the numbers about how a team defends. The report writer only sees these."""
    if team not in league.index:
        raise HTTPException(404, f"Unknown team: {team}")
    row = league.loc[team]
    faced = recent_shots[recent_shots.against == team]  # every shot taken against this team

    profile = {
        "team": team,
        "seasons_covered": window,
        "matches": int(row.matches),
        "shots_against_per_match": round(float(row.shots_against), 1),
        "xg_against_per_match": round(float(row.xg_against), 2),
        "xg_per_shot_conceded": round(float(row.xg_per_shot_against), 3),
        "where_xg_conceded_comes_from": shares(faced),
        "league_shares": LEAGUE_SHARES,
        "league_avg_shots_against": round(float(league.shots_against.mean()), 1),
        "league_avg_xg_against": round(float(league.xg_against.mean()), 2),
        "teams_in_comparison": len(league),
        # How many past matches count as "recent" for the matchup numbers
        "form_matches": FORM_WINDOW,
        "quality_matches": QUALITY_WINDOW,
        # Rank 1 means fewest conceded, so a low rank number is a strong defense
        "rank_shots_against": int(league.shots_against.rank().loc[team]),
        "rank_xg_against": int(league.xg_against.rank().loc[team]),
        # Shots for the map, converted to the 120 x 80 pitch the page draws
        "shots": pd.DataFrame({
            "x": (faced.X * 120).round(1), "y": (faced.Y * 80).round(1),
            "xg": faced.our_xg.round(3), "goal": faced.goal,
        }).to_dict("records"),
    }

    if opponent:
        if opponent == team or opponent not in league.index:
            raise HTTPException(400, f"Pick a different opponent: {opponent}")
        t, o = latest_form(tm, team), latest_form(tm, opponent)
        tq, oq = latest_quality(tm, team), latest_quality(tm, opponent)

        # Inputs for the shots model: both teams' recent form and who plays at home
        model_input = pd.DataFrame([{
            "t_shots_against_form": t.shots_against, "t_xg_against_form": t.xg_against,
            "t_shots_for_form": t.shots_for,
            "o_shots_for_form": o.shots_for, "o_xg_for_form": o.xg_for,
            "o_shots_against_form": o.shots_against,
            "o_is_home": 1 if venue == "away" else 0,
        }])[FEATURES]
        exp_shots = max(float(shots_bundle["model"].predict(model_input)[0]), 0.0)

        # Danger per shot: blend the chances this team gives up with the chances the opponent creates
        exp_quality = (tq["xg_per_shot_against"] + oq["xg_per_shot_for"]) / 2
        profile.update({
            "opponent": opponent,
            "venue": f"{team} plays {venue}",
            "opponent_shots_per_match_recently": round(float(o.shots_for), 1),
            "opponent_xg_per_shot_recently": round(float(oq["xg_per_shot_for"]), 3),
            "team_xg_per_shot_against_recently": round(float(tq["xg_per_shot_against"]), 3),
            "expected_shots_against_next_match": round(exp_shots, 1),
            # Expected goals = expected shots times the typical danger of each shot
            "expected_xg_against_next_match": round(exp_shots * exp_quality, 2),
            "shots_model": {
                "name": shots_bundle["name"],
                "avg_miss": round(float(shots_bundle["mae"]), 1),
                "simple_average_miss": round(float(shots_bundle["baseline_mae"]), 1),
                # train_shots.py compares two candidates and keeps the better one
                "chosen_over": OTHER_SHOTS_CANDIDATE[shots_bundle["name"]],
            },
        })
    return profile


INSTRUCTIONS = (
    "You write scouting reports for football fans who may be new to analytics. "
    "Use only the numbers in the data below and never add outside facts. "
    "Say which seasons the numbers cover. "
    "Every time you give a number, compare it with the league average or rank so it means something. "
    "The first time you use xG, define it in one plain sentence: the chance a shot scores, "
    "from 0 to 1, based on where and how it was taken. "
    "Write four short sections with these exact labels. "
    "How they defend: shots against and xG against per match with league average and rank, "
    "and what that says (many shots, dangerous shots, or both). "
    "Where they get hurt: use where_xg_conceded_comes_from compared with league_shares. "
    "A gap smaller than 0.03 means in line with the league, so do not call it higher or lower. "
    "Name the one or two biggest gaps of 0.03 or more, such as set pieces, headers, or the penalty box. "
    "Next match: only if an opponent is in the data. Give the expected shots and expected goals, "
    "say whether the team plays at home or away, and compare the opponent's recent shots per match "
    "and xG per shot to the league. Say these are model estimates. "
    "What to do: two concrete tactical suggestions for attacking this team, tied to the numbers above. "
    "Aim for 170 to 220 words. Never invent numbers."
)



REPORT_CACHE = {}  # remembers finished reports so repeat clicks don't use the free daily quota


def short_summary(profile):
    """Plain one-sentence summary, used when Gemini is unavailable."""
    return (
        f"{profile['team']} concede {profile['shots_against_per_match']} shots per match "
        f"worth {profile['xg_against_per_match']} expected goals. "
        "(The full report did not load. Check the terminal for the reason.)"
    )


def write_report(profile):
    """Gemini turns the numbers into a scouting report. Returns None if it can't."""
    if not os.getenv("GEMINI_API_KEY"):
        return None
    facts = {k: v for k, v in profile.items() if k != "shots"}  # skip the raw shot list
    try:
        start = time.time()
        client = genai.Client()
        # Run the call in a background thread so we can give up after 40 seconds
        future = POOL.submit(lambda: client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Data:\n" + json.dumps(facts),  # the numbers
            config=types.GenerateContentConfig(system_instruction=INSTRUCTIONS),  # the rules
        ))
        response = future.result(timeout=40)
        print(f"Gemini answered in {time.time() - start:.1f}s")
        return response.text
    except TimeoutError:
        print("Gemini took longer than 40 seconds, showing the short summary instead")
    except Exception as e:
        print("Gemini call failed:", e)
    return None

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


app = FastAPI(title="Pitch Intel")


@app.get("/api/teams")
def list_teams():
    return sorted(league.index)


@app.get("/api/team/{team}")
def get_team(team: str, opponent: str | None = None, venue: str = "home"):
    profile = team_profile(team, opponent, venue)
    key = (team, opponent, venue)
    report = REPORT_CACHE.get(key) or write_report(profile)
    if report:
        REPORT_CACHE[key] = report  # only real reports get remembered, never the fallback
    profile["report"] = report or short_summary(profile)
    return profile


@app.get("/api/explain")
def explain(x: float = Query(ge=60, le=120), y: float = Query(ge=0, le=80),
            shot_type: str = "RightFoot",
            situation: str = "OpenPlay", last_action: str = "Pass"):
    """Show what pushes the xG of one shot up or down."""
    # The page sends map coordinates (0 to 120 and 0 to 80). The model wants fractions
    shot = pd.DataFrame([{"X": x / 120, "Y": y / 80, "shotType": shot_type,
                          "situation": situation, "lastAction": last_action}])
    feats = xg_features(shot)

    # The model reports how much each column pushed this shot's score, plus a starting point
    contrib = xg_model.booster_.predict(feats, pred_contrib=True)[0]
    start, per_column = contrib[-1], contrib[:-1]

    # Add up columns that belong to one factor, like all the body part columns
    factors = {}
    for col, c in zip(feats.columns, per_column):
        factors[factor_of(col)] = factors.get(factor_of(col), 0) + c

    # Walk from a typical shot to this shot one factor at a time, biggest push first
    running, steps = start, []
    for name, c in sorted(factors.items(), key=lambda kv: -abs(kv[1])):
        before = sigmoid(running)
        running += c
        steps.append({"factor": name, "points": round(float((sigmoid(running) - before) * 100), 1),
                      "log_odds": round(float(c), 3)})  # the push itself, for the formula on the page
    return {
        "xg": round(float(sigmoid(running)), 3),
        "average_shot": round(float(sigmoid(start)), 3),
        # The score z before it becomes a chance: xG = 1 / (1 + e^-z)
        "start_log_odds": round(float(start), 3),
        "log_odds": round(float(running), 3),
        # The two measurements the model works out from where the shot was taken
        "distance_m": round(float(feats.distance[0]), 1),
        "angle_deg": round(float(np.degrees(feats.angle[0])), 1),
        "steps": steps,
        "xg_model_info": XG_MODEL_INFO,
    }


# Serve the page from the same server. Mounted last so it doesn't hide the /api routes
