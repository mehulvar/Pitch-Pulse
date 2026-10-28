"""Train our own xG model on Premier League shots and compare it with Understat's xG.

1. Tune the settings with Optuna, scored on practice rounds inside the development seasons.
2. Try the same search with monotonic constraints (further away or a tighter angle never helps).
3. Grade the chosen model once on the held-out newest seasons, with confidence intervals.
4. Refit on every season for the app."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import brier_score_loss, log_loss

from evaluation import bootstrap_ci, calibration_table, save_metrics
from features import holdout_split, load_shots, season_folds, xg_features

MODEL_DIR = Path(__file__).parent / "models"
N_TRIALS = 40
# Keep the constrained model unless it is worse than this on the practice rounds.
# It's about 0.1% of a typical log loss, so only a real loss of accuracy counts
MONOTONE_TOLERANCE = 0.0003
# +1: xG can only go up as this grows, -1: only down. Other inputs are free
MONOTONE = {"distance": -1, "meters_from_goal_line": -1, "angle": 1}


def model_with(params, columns, monotone):
    constraints = [MONOTONE.get(c, 0) for c in columns] if monotone else None
    return lgb.LGBMClassifier(**params, subsample_freq=1, monotone_constraints=constraints,
                              monotone_constraints_method="advanced", verbose=-1)


def suggest(trial):
    """The settings Optuna searches over. Goals are rare, so it leans toward small, cautious trees."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 48),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def cv_log_loss(params, X, y, seasons, folds, monotone):
    """Average log loss over the practice rounds: train on earlier seasons, test on the next one."""
    losses = []
    for train_seasons, test_season in folds:
        tr, te = seasons.isin(train_seasons), seasons == test_season
        model = model_with(params, X.columns, monotone).fit(X[tr], y[tr])
        losses.append(log_loss(y[te], model.predict_proba(X[te])[:, 1]))
    return float(np.mean(losses))


def search(X, y, seasons, folds, monotone):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda t: cv_log_loss(suggest(t), X, y, seasons, folds, monotone), n_trials=N_TRIALS)
    print(f"  {'with' if monotone else 'without'} constraints: practice log loss {study.best_value:.5f}")
    return study.best_params, study.best_value


def holdout_report(shots, ours):
    """Our xG against Understat's on the held-out shots, with 95% intervals from resampling matches."""
    df = shots.assign(ours=ours, theirs=shots.xG.clip(0.001, 0.999))
    report = {"shots": int(len(df)), "goals": int(df.goal.sum()),
              "total_xg_ours": round(float(df.ours.sum()), 1), "total_xg_understat": round(float(df.theirs.sum()), 1)}
    for name, metric in [("brier", brier_score_loss), ("log_loss", log_loss)]:
        for who, label in [("ours", "ours"), ("theirs", "understat")]:
            ci = bootstrap_ci(lambda d, m=metric, w=who: m(d.goal, d[w]), df, "match_id")
            report[f"{name}_{label}"] = {k: round(v, 5) for k, v in ci.items()}
        gap = bootstrap_ci(lambda d, m=metric: m(d.goal, d.ours) - m(d.goal, d.theirs), df, "match_id")
        report[f"{name}_gap_to_understat"] = {k: round(v, 5) for k, v in gap.items()}
    report["calibration"] = calibration_table(df.goal, df.ours, bins=10)
    return report


def print_report(r):
    print(f"\nHeld-out seasons: {r['shots']} shots, {r['goals']} goals "
          f"(our total xG {r['total_xg_ours']}, Understat {r['total_xg_understat']})")
    for name in ("brier", "log_loss"):
        for who in ("ours", "understat", "gap_to_understat"):
            ci = r[f"{name}_{who}"]
            print(f"  {name:8} {who:17} {ci['estimate']:.5f}  [{ci['low']:.5f}, {ci['high']:.5f}]")
    print("Calibration (predicted vs actual goal rate):")
    for row in r["calibration"]:
        print(f"  {row['predicted']:.3f} -> {row['actual']:.3f}  ({row['count']} shots)")


def main():
    shots = load_shots()
    X, y, seasons = xg_features(shots), shots.goal, shots.season
    dev, holdout = holdout_split(seasons.unique(), n_holdout=2)
    folds = season_folds(dev, n_folds=3)
    in_dev = seasons.isin(dev)
    print(f"Tuning on {dev[0]}-{dev[-1]} ({in_dev.sum()} shots), testing on {holdout}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    free_params, free_loss = search(X[in_dev], y[in_dev], seasons[in_dev], folds, monotone=False)
    mono_params, mono_loss = search(X[in_dev], y[in_dev], seasons[in_dev], folds, monotone=True)
    monotone = mono_loss <= free_loss + MONOTONE_TOLERANCE
    params = mono_params if monotone else free_params
    print(f"Chosen: {'with' if monotone else 'without'} monotonic constraints")

    tested = model_with(params, X.columns, monotone).fit(X[in_dev], y[in_dev])
    report = holdout_report(shots[~in_dev], tested.predict_proba(X[~in_dev])[:, 1])
    print_report(report)

    final = model_with(params, X.columns, monotone).fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "xg_model.joblib")
    # train_shots.py grades itself with this one, so its test seasons stay unseen there too
    joblib.dump(tested, MODEL_DIR / "xg_model_dev.joblib")
    save_metrics("xg_model", {
        "params": params, "monotone_constraints": bool(monotone),
        "practice_log_loss": {"with_constraints": round(mono_loss, 5), "without_constraints": round(free_loss, 5)},
        "dev_seasons": [int(s) for s in dev], "holdout_seasons": [int(s) for s in holdout],
        "holdout": report, "training_shots": int(len(shots)),
    })
    print("\nSaved models/xg_model.joblib and the scores in metrics.json")


if __name__ == "__main__":
    main()
