"""Train the model that predicts how many shots a team will concede in a match."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, add_form, load_shots, matchup_table, score_shots, team_match_table, train_mask
MODEL_DIR = Path(__file__).parent / "models"
TEST_FROM = 2024
TARGET = "shots_against"  # shots the opponent took in that match

# Score every shot with the xG model from train_xg.py so team xG numbers come from our model
xg_model = joblib.load(MODEL_DIR / "xg_model.joblib")
shots = load_shots()
shots["our_xg"] = score_shots(shots, xg_model)
tm = add_form(team_match_table(shots, xg_col="our_xg"))

# Drop matches where a team has fewer than 3 earlier matches, since there is no form yet
data = matchup_table(tm).dropna(subset=FEATURES)
is_train = train_mask(data.date)
train, test = data[is_train], data[~is_train]


# The two models we compare. Both predict a count, which is what Poisson models are built for
CANDIDATES = {
    "Poisson regression": lambda: make_pipeline(
        StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000)),
    "LightGBM (Poisson)": lambda: lgb.LGBMRegressor(
        objective="poisson", n_estimators=300, learning_rate=0.03, num_leaves=8,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        verbose=-1),
}
BASELINE = "Simple average (current app)"

scores = {}


def grade(name, pred):
    pred = np.clip(pred, 0.1, None)  # Poisson deviance needs positive predictions
    scores[name] = (mean_absolute_error(test[TARGET], pred),
                    mean_poisson_deviance(test[TARGET], pred))
    print(f"{name:30}{scores[name][0]:14.3f}{scores[name][1]:12.3f}")


print(f"Train matchups: {len(train)}   Test matchups: {len(test)}")
print(f"{'':30}{'Avg miss (shots)':>14}{'Deviance':>12}   (lower is better)")

# What the app does today: average the defense's shots conceded and the attack's shots taken
grade(BASELINE, (test.t_shots_against_form + test.o_shots_for_form) / 2)
for name, make in CANDIDATES.items():
    grade(name, make().fit(train[FEATURES], train[TARGET]).predict(test[FEATURES]))

# Keep whichever candidate predicts best, then refit it on all the data for the app
winner = min(CANDIDATES, key=lambda n: scores[n][1])
print("\nWinner:", winner)
final = CANDIDATES[winner]().fit(data[FEATURES], data[TARGET])

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump({
    "name": winner,
    "model": final,
    "mae": scores[winner][0],
    "baseline_mae": scores[BASELINE][0],
}, MODEL_DIR / "shots_model.joblib")
print("Saved models/shots_model.joblib")