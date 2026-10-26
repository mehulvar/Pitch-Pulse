"""Train our own xG model on Premier League shots and compare it with Understat's xG."""
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from features import load_shots, train_mask, xg_features

MODEL_DIR = Path(__file__).parent / "models"
TEST_FROM = 2024  # learn from seasons before this, grade on this season and newer

shots = load_shots()
X = xg_features(shots)
y = shots.goal

train = train_mask(shots.date)
test = ~train
if train.sum() == 0 or test.sum() == 0:
    raise SystemExit(f"Need seasons before and after {TEST_FROM} in data/. Let the download run longer")


def new_model():
    # Small and cautious: goals are rare, so a big model would just memorize shots
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
    )


# Honest test: train on the past, grade on seasons the model has never seen
model = new_model().fit(X[train], y[train])
ours = model.predict_proba(X[test])[:, 1]
theirs = shots.xG[test].clip(0.001, 0.999)  # Understat's own xG, our yardstick

print(f"Test shots: {test.sum()}   Goals scored: {y[test].sum()}")
print(f"{'':16}{'Brier':>8}{'Log loss':>10}{'Total xG':>10}   (lower loss is better)")
for name, p in [("Our model", ours), ("Understat xG", theirs)]:
    print(f"{name:16}{brier_score_loss(y[test], p):8.4f}{log_loss(y[test], p):10.4f}{p.sum():10.1f}")

# Calibration: when we say a shot has a 10% chance, do about 10% go in?
buckets = pd.qcut(ours, 5, duplicates="drop")
calib = pd.DataFrame({"predicted": ours, "scored": y[test].values, "bucket": buckets})
print("\nCalibration (predicted should be close to scored):")
print(calib.groupby("bucket", observed=True)[["predicted", "scored"]].mean().round(3))

# For the app, refit on every season so it also learns from the newest matches
final = new_model().fit(X, y)
importance = pd.Series(final.booster_.feature_importance("gain"), index=X.columns)
print("\nWhat the model relies on most:")
print((importance / importance.sum()).sort_values(ascending=False).head(6).round(3))

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(final, MODEL_DIR / "xg_model.joblib")
print("\nSaved models/xg_model.joblib")