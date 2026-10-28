"""Shared helpers: load the data and turn it into numbers for the models.
The training scripts and the server both import this file so they always agree."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# Understat gives shot locations as fractions of the pitch, so we convert to meters
PITCH_LENGTH, PITCH_WIDTH, GOAL_WIDTH = 105, 68, 7.32

# Categories the xG model knows about. Anything else is counted as "Other"
SHOT_TYPES = ["RightFoot", "LeftFoot", "Head", "OtherBodyPart"]
SITUATIONS = ["OpenPlay", "FromCorner", "SetPiece", "DirectFreekick", "Penalty"]
LAST_ACTIONS = ["Pass", "Cross", "TakeOn", "Rebound", "Throughball", "Standard",
                "BallRecovery", "Aerial", "Chipped", "HeadPass"]
SET_PIECE_SITUATIONS = ["FromCorner", "SetPiece", "DirectFreekick", "Penalty"]

FORM_WINDOW = 10     # how many past matches count as a team's "recent form"
QUALITY_WINDOW = 20  # how many past matches we use to measure typical shot quality
FORM_COLS = ["shots_for", "shots_against", "xg_for", "xg_against"]

# Inputs of the shots-against model. t_ is the defending team, o_ is the attacking opponent
FEATURES = [
    "t_shots_against_form", "t_xg_against_form", "t_shots_for_form",
    "o_shots_for_form", "o_xg_for_form", "o_shots_against_form",
    "o_is_home",
]


def load_shots():
    """Read every downloaded season into one table with one row per shot."""
    files = sorted(DATA_DIR.glob("shots_*.csv"))
    if not files:
        raise SystemExit("No files in data/. Run download_epl.py first")
    shots = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    shots = shots.drop_duplicates(subset="id")
    shots["date"] = pd.to_datetime(shots["date"])
    shots = shots[shots.result != "OwnGoal"].reset_index(drop=True)  # not a real attempt by the attacker

    # Understat marks each shot as home or away. From that we get who shot and who was shot at
    is_home = shots.h_a == "h"
    shots["team"] = np.where(is_home, shots.h_team, shots.a_team)
    shots["against"] = np.where(is_home, shots.a_team, shots.h_team)
    shots["goal"] = (shots.result == "Goal").astype(int)
    return shots


def xg_features(df):
    """Turn shots into the numbers the xG model learns from."""
    # X runs from 0 to 1 toward the goal being attacked. Y runs across the pitch
    dx = (1 - df["X"].astype(float)) * PITCH_LENGTH   # meters from the goal line
    dy = (0.5 - df["Y"].astype(float)) * PITCH_WIDTH  # meters off the center line
    out = pd.DataFrame({
        "distance": np.hypot(dx, dy),
        # The angle the goal mouth takes up from where the shot is taken. Wider means easier
        "angle": np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2),
        "meters_from_goal_line": dx,
        "meters_off_center": dy.abs(),
    })
    # One yes/no column per category
    groups = [("shot", SHOT_TYPES, "shotType"),
              ("situation", SITUATIONS, "situation"),
              ("last", LAST_ACTIONS, "lastAction")]
    for prefix, values, col in groups:
        for v in values:
            out[f"{prefix}_{v}"] = (df[col] == v).astype(int)
        out[f"{prefix}_Other"] = (~df[col].isin(values)).astype(int)
    return out


def factor_of(column):
    """Group the model's columns into a few readable factors for the explainer."""
    if column in ("distance", "meters_from_goal_line"):
        return "Distance to goal"
    if column == "angle":
        return "Angle to goal"
    if column == "meters_off_center":
        return "How wide of center"
    if column.startswith("shot_"):
        return "Body part"
    if column.startswith("situation_"):
        return "Type of chance"
    return "Previous action"


def score_shots(shots, model):
    """The chance each shot scores, according to our xG model."""
    return model.predict_proba(xg_features(shots))[:, 1]


def team_match_table(shots, xg_col):
    """One row per team per match, with shots and xG created (for) and conceded (against)."""
    # The two teams in each match come from the shot rows, so a team with zero shots still appears
    matches = shots.groupby("match_id").agg(
        date=("date", "first"), season=("season", "first"),
        home=("h_team", "first"), away=("a_team", "first"),
    ).reset_index()

    # What each team created in each match
    created = shots.groupby(["match_id", "team"]).agg(
        shots_for=(xg_col, "size"), xg_for=(xg_col, "sum")
    ).reset_index()

    # Every match becomes two rows, one from each team's point of view
    sides = []
    for side, other, is_home in [("home", "away", 1), ("away", "home", 0)]:
        part = matches[["match_id", "date", "season", side, other]].copy()
        part.columns = ["match_id", "date", "season", "team", "opp"]
        part["is_home"] = is_home
        sides.append(part)
    tm = pd.concat(sides, ignore_index=True)
    tm = tm.merge(created, on=["match_id", "team"], how="left")

    # What the opponent created in the same match is what this team conceded
    conceded = created.rename(columns={"team": "opp", "shots_for": "shots_against",
                                       "xg_for": "xg_against"})
    tm = tm.merge(conceded, on=["match_id", "opp"], how="left")
    tm[FORM_COLS] = tm[FORM_COLS].fillna(0)  # no row means the team had no shots
    return tm.sort_values(["date", "match_id"]).reset_index(drop=True)


def add_form(tm):
    """Add each team's average over its previous matches, never including the match itself."""
    tm = tm.copy()
    by_team = tm.groupby("team")
    for col in FORM_COLS:
        # shift(1) skips the current match so the model can't peek at the answer
        tm[f"{col}_form"] = by_team[col].transform(
            lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=3).mean()
        )
    return tm


def matchup_table(tm):
    """One row per defending team and attacking opponent, with both teams' recent form."""
    form = [f"{c}_form" for c in FORM_COLS]
    defender = tm.rename(columns={c: f"t_{c}" for c in form})
    attacker = tm[["match_id", "team", "is_home"] + form].rename(
        columns={"team": "opp", "is_home": "o_is_home", **{c: f"o_{c}" for c in form}}
    )
    return defender.merge(attacker, on=["match_id", "opp"])


def latest_form(tm, team):
    """A team's averages over its most recent matches. Used to predict its next match."""
    return tm[tm.team == team].tail(FORM_WINDOW)[FORM_COLS].mean()


def latest_quality(tm, team):
    """Average xG per shot a team creates and concedes over its recent matches."""
    last = tm[tm.team == team].tail(QUALITY_WINDOW)
    return {
        "xg_per_shot_for": last.xg_for.sum() / max(last.shots_for.sum(), 1),
        "xg_per_shot_against": last.xg_against.sum() / max(last.shots_against.sum(), 1),
    }

def holdout_split(seasons, n_holdout=2):
    """Split seasons into ones for building the model and the newest ones, kept back for the final test.
    The held-out seasons are scored once at the very end, never used to choose settings."""
    ordered = sorted(set(int(s) for s in seasons))
    return ordered[:-n_holdout], ordered[-n_holdout:]


def season_folds(seasons, n_folds=3):
    """Practice rounds for choosing settings: train on every earlier season, test on the next one.
    Returns [(train_seasons, test_season), ...] for the last n_folds seasons."""
    ordered = sorted(set(int(s) for s in seasons))
    if len(ordered) <= n_folds:
        raise ValueError(f"Need more than {n_folds} seasons for {n_folds} folds, got {len(ordered)}")
    return [(ordered[:i], ordered[i]) for i in range(len(ordered) - n_folds, len(ordered))]