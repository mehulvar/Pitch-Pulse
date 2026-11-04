"""Train the model that predicts how many shots a team will concede in a match.

Same method as train_xg.py: choose settings on practice rounds inside the development seasons,
grade once on the held-out newest seasons, then refit on everything for the app."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import bootstrap_ci, save_metrics
from features import (
    FEATURES, add_form, count_range, fit_dispersion, holdout_split, load_shots, matchup_table,
    score_shots, season_folds, team_match_table,
)

MODEL_DIR = Path(__file__).parent / "models"
TARGET = "shots_against"  # shots the opponent took in that match
N_TRIALS = 30
POISSON_ALPHAS = [0.01, 0.1, 1.0, 10.0]  # how strongly the regression's weights are shrunk
INTERVAL = (0.1, 0.9)  # the 80% range shown on the page
BASELINE = "Simple average"


def matchups(xg_model):
    """One row per team per match with both teams' recent form, scored with the given xG model."""
    shots = load_shots()
    shots["our_xg"] = score_shots(shots, xg_model)
    tm = add_form(team_match_table(shots, xg_col="our_xg"))
    # Drop matches where a team has fewer than 3 earlier matches, since there is no form yet
    return matchup_table(tm).dropna(subset=FEATURES).reset_index(drop=True)


def poisson_regression(alpha):
    return make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=1000))


def lightgbm_poisson(params):
    return lgb.LGBMRegressor(objective="poisson", subsample_freq=1, verbose=-1, **params)


def cv_deviance(make, data, folds):
    """Average Poisson deviance over the practice rounds. Lower means better predicted counts."""
    scores = []
    for train_seasons, test_season in folds:
        tr, te = data[data.season.isin(train_seasons)], data[data.season == test_season]
        pred = np.clip(make().fit(tr[FEATURES], tr[TARGET]).predict(te[FEATURES]), 0.1, None)
        scores.append(mean_poisson_deviance(te[TARGET], pred))
    return float(np.mean(scores))


def tune_candidates(data, folds):
    """Best settings for each candidate, chosen on practice rounds only."""
    alpha = min(POISSON_ALPHAS, key=lambda a: cv_deviance(lambda: poisson_regression(a), data, folds))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 32),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        return cv_deviance(lambda: lightgbm_poisson(params), data, folds)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=N_TRIALS)
    return {
        "Poisson regression": (lambda: poisson_regression(alpha), {"alpha": alpha},
                               cv_deviance(lambda: poisson_regression(alpha), data, folds)),
        "LightGBM (Poisson)": (lambda: lightgbm_poisson(study.best_params), study.best_params, study.best_value),
    }


def grade(test, pred, alpha):
    """Average miss and deviance with 95% intervals, plus how often the real count fell in the 80% range:
    once with a plain Poisson range, once with the negative binomial using the spread measured in training."""
    df = test.assign(pred=np.clip(pred, 0.1, None))
    out = {name: {k: round(v, 3) for k, v in bootstrap_ci(
        lambda d, m=metric: m(d[TARGET], d.pred), df, "match_id").items()}
        for name, metric in [("avg_miss", mean_absolute_error), ("deviance", mean_poisson_deviance)]}
    for label, spread in [("poisson", 0.0), ("negative_binomial", alpha)]:
        low, high = count_range(df.pred, spread, INTERVAL)
        out[f"interval_80_coverage_{label}"] = round(float(((df[TARGET] >= low) & (df[TARGET] <= high)).mean()), 3)
    out["interval_80_coverage"] = out["interval_80_coverage_negative_binomial"]  # the range the app shows
    out["dispersion"] = round(alpha, 4)
    return out


def fit_and_grade(make, train, test):
    """Fit on the development seasons, measure the extra spread there, grade on the held-out seasons."""
    model = make().fit(train[FEATURES], train[TARGET])
    alpha = fit_dispersion(train[TARGET], np.clip(model.predict(train[FEATURES]), 0.1, None))
    return grade(test, model.predict(test[FEATURES]), alpha)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Graded with the xG model that never saw the held-out seasons, so nothing leaks into the test
    data = matchups(joblib.load(MODEL_DIR / "xg_model_dev.joblib"))
    dev, holdout = holdout_split(data.season.unique(), n_holdout=2)
    train, test = data[data.season.isin(dev)], data[data.season.isin(holdout)]
    candidates = tune_candidates(train, season_folds(dev, n_folds=3))
    winner = min(candidates, key=lambda n: candidates[n][2])
    print(f"Practice deviance: { {n: round(c[2], 4) for n, c in candidates.items()} } -> {winner}")

    baseline = lambda df: (df.t_shots_against_form + df.o_shots_for_form) / 2
    results = {BASELINE: grade(test, baseline(test), fit_dispersion(train[TARGET], baseline(train)))}
    for name, (make, _, _) in candidates.items():
        results[name] = fit_and_grade(make, train, test)
    print(f"\nHeld-out seasons {holdout}, {len(test)} team-matches (95% intervals)")
    for name, r in results.items():
        miss = r["avg_miss"]
        print(f"  {name:22} avg miss {miss['estimate']:.2f} shots [{miss['low']:.2f}, {miss['high']:.2f}]"
              f"   real count inside the 80% range: Poisson {r['interval_80_coverage_poisson']:.0%}, "
              f"negative binomial {r['interval_80_coverage_negative_binomial']:.0%}")

    # The app's model: the winner refitted on every season, scored with the full xG model
    everything = matchups(joblib.load(MODEL_DIR / "xg_model.joblib"))
    final = candidates[winner][0]().fit(everything[FEATURES], everything[TARGET])
    joblib.dump({
        "name": winner, "model": final,
        "mae": results[winner]["avg_miss"]["estimate"],
        "baseline_mae": results[BASELINE]["avg_miss"]["estimate"],
        "interval_80_coverage": results[winner]["interval_80_coverage"],
        "dispersion": fit_dispersion(everything[TARGET], np.clip(final.predict(everything[FEATURES]), 0.1, None)),
    }, MODEL_DIR / "shots_model.joblib")
    save_metrics("shots_model", {
        "winner": winner, "settings": {n: c[1] for n, c in candidates.items()},
        "practice_deviance": {n: round(c[2], 4) for n, c in candidates.items()},
        "holdout_seasons": [int(s) for s in holdout], "holdout_team_matches": int(len(test)),
        "holdout": results,
    })
    print("\nSaved models/shots_model.joblib and the scores in metrics.json")


if __name__ == "__main__":
    main()
