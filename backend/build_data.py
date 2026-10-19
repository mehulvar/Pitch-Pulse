import numpy as np
import pandas as pd
import lightgbm as lgb
from statsbombpy import sb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss

# StatsBomb free data: FIFA World Cup 2018 (small, fast to download)
COMPETITION_ID, SEASON_ID = 43, 3


def load_shots():
    """Download every match and keep only the shot events."""
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    frames = []
    for _, m in matches.iterrows():
        events = sb.events(match_id=m.match_id)
        shots = events[events.type == "Shot"].copy()
        shots["match_id"] = m.match_id
        # The team being shot at is whichever team in the match did not shoot
        shots["against"] = np.where(shots.team == m.home_team, m.away_team, m.home_team)
        frames.append(shots)
    return pd.concat(frames, ignore_index=True)


def make_features(df):
    """Turn raw shot rows into numbers the model can learn from."""
    # StatsBomb pitch is 120 x 80 and every team attacks toward x = 120
    x = df.location.apply(lambda loc: loc[0])
    y = df.location.apply(lambda loc: loc[1])
    dx, dy = 120 - x, 40 - y  # gap to the center of the goal
    return pd.DataFrame({
        "x": x,
        "y": y,
        "distance": np.hypot(dx, dy),
        # Angle the goal mouth (8 units wide) takes up from the shot location
        "angle": np.arctan2(8 * dx, dx**2 + dy**2 - 16),
        "is_header": (df.shot_body_part == "Head").astype(int),
        "is_penalty": (df.shot_type == "Penalty").astype(int),
        "is_free_kick": (df.shot_type == "Free Kick").astype(int),
        "under_pressure": df.under_pressure.fillna(False).astype(bool).astype(int),
    })


shots = load_shots()
shots = shots[shots.period != 5].reset_index(drop=True)  # drop penalty shootouts

X = make_features(shots)
y = (shots.shot_outcome == "Goal").astype(int)  # 1 if the shot scored

# Split by match so no match shows up in both train and test
train_idx, test_idx = next(
    GroupShuffleSplit(test_size=0.2, random_state=0).split(X, y, groups=shots.match_id)
)

# Small model on purpose: there are only about 1,500 shots
model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=20, verbose=-1
)
model.fit(X.iloc[train_idx], y.iloc[train_idx])

# Brier score: lower is better. We compare against StatsBomb's own xG as a yardstick
pred = model.predict_proba(X.iloc[test_idx])[:, 1]
print("Our model Brier:     ", round(brier_score_loss(y.iloc[test_idx], pred), 4))
print("StatsBomb xG Brier:  ", round(brier_score_loss(y.iloc[test_idx], shots.shot_statsbomb_xg.iloc[test_idx]), 4))

# Score every shot with our model and save the table the API will read
out = pd.DataFrame({
    "match_id": shots.match_id,
    "team": shots.team,
    "against": shots.against,
    "x": X.x,
    "y": X.y,
    "xg": model.predict_proba(X)[:, 1].round(3),
    "goal": y,
    "minute": shots.minute,
})
out.to_csv("shots.csv", index=False)
print("Saved", len(out), "shots to shots.csv")