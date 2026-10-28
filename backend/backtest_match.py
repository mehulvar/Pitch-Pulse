"""Grade the match model the honest way: replay the newest seasons week by week, refitting on only
the matches played before each week, and compare its forecasts with the betting market.

1. Tune on the last development season: half-life of the time decay, and goals vs xG as the input.
2. Backtest the chosen setup on the held-out seasons, which nothing above has looked at.
3. Fit once on everything and save the draws the app uses."""
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import bootstrap_ci, brier_3way, log_loss_3way, rps, save_metrics
from features import holdout_split
from match_model import TARGETS, WINDOW_DAYS, fit_strengths, load_matches, save_bundle
from match_predict import MODEL_FILE, predict_match
from team_names import attach_odds

DATA_DIR = Path(__file__).parent / "data"
PREDICTIONS_FILE = Path(__file__).parent / "models" / "backtest_predictions.csv"
HALF_LIVES = [180, 365, 730, 1460]  # days. Shorter reacts faster to form, longer is steadier
TUNING_REFIT_WEEKS = 4        # refit monthly while tuning, to keep it quick
FORECASTS = {                 # column prefix -> name in the results table
    "model": "Our match model",
    "market": "Bookmaker closing odds",
    "base": "Base rates (home/draw/away %)",
    # Not a forecast: Understat works it out after the match from that match's own xG.
    # It shows how predictable results are even knowing every chance, a ceiling, not a rival
    "understat": "Hindsight: the match's own xG",
}


def load_backtest_data():
    matches = load_matches()
    odds = pd.concat([pd.read_csv(f) for f in sorted(DATA_DIR.glob("odds_*.csv"))], ignore_index=True)
    joined = attach_odds(matches, odds)
    return joined.rename(columns={"p_home": "market_home", "p_draw": "market_draw", "p_away": "market_away",
                                  "understat_p_home": "understat_home", "understat_p_draw": "understat_draw",
                                  "understat_p_away": "understat_away"})


def forecast_season(matches, season, target, half_life, refit_weeks=1):
    """Our forecast for every match of one season, refitting every refit_weeks on earlier matches only."""
    games = matches[matches.season == season]
    blocks = (games.date - games.date.min()).dt.days // (7 * refit_weeks)
    rows = []
    for _, block in games.groupby(blocks):
        as_of = block.date.min().normalize()
        recent = matches[(matches.date < as_of) & (matches.date >= as_of - pd.Timedelta(days=WINDOW_DAYS))]
        teams = sorted(set(recent.home) | set(recent.away) | set(block.home) | set(block.away))
        bundle = fit_strengths(matches, teams, target, as_of, half_life)
        for idx, m in block.iterrows():
            p = predict_match(bundle, m.home, m.away)
            rows.append({"index": idx, "model_home": p["p_home"], "model_draw": p["p_draw"],
                         "model_away": p["p_away"]})
    return pd.DataFrame(rows).set_index("index")


def with_base_rates(matches, season):
    """The simplest forecast: how often home wins, draws and away wins happened before this season."""
    before = matches[matches.season < season].outcome.value_counts(normalize=True)
    games = matches[matches.season == season]
    return games.assign(base_home=before.get(0, 0), base_draw=before.get(1, 0), base_away=before.get(2, 0))


def probs(df, prefix):
    return df[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]].to_numpy()


def tune(matches, season):
    """Pick the input (goals or xG) and half-life with the best RPS on one development season."""
    results = []
    for target in TARGETS:
        for half_life in HALF_LIVES:
            f = forecast_season(matches, season, target, half_life, refit_weeks=TUNING_REFIT_WEEKS)
            score = rps(probs(f, "model"), matches.loc[f.index, "outcome"])
            results.append({"target": target, "half_life_days": half_life, "rps": round(score, 4)})
            print(f"  tuning {season}: {target:5} half-life {half_life:3} days -> RPS {score:.4f}")
    return min(results, key=lambda r: r["rps"]), results


def score_table(df):
    """Each forecast's RPS, log loss and Brier with 95% intervals, plus its RPS gap to the market."""
    table = {}
    for prefix, name in FORECASTS.items():
        scored = df.dropna(subset=[f"{prefix}_home"])
        row = {}
        for metric_name, metric in [("rps", rps), ("log_loss", log_loss_3way), ("brier", brier_3way)]:
            ci = bootstrap_ci(lambda d, m=metric: m(probs(d, prefix), d.outcome), scored, "match_id")
            row[metric_name] = {k: round(v, 4) for k, v in ci.items()}
        gap = bootstrap_ci(lambda d: rps(probs(d, prefix), d.outcome) - rps(probs(d, "market"), d.outcome),
                           scored, "match_id")
        row["rps_minus_market"] = {k: round(v, 4) for k, v in gap.items()}
        table[name] = row
    return table


def print_table(table, n_matches):
    print(f"\nHeld-out backtest, {n_matches} matches (lower is better, 95% intervals)")
    print(f"{'':32}{'RPS':>24}{'Log loss':>24}{'vs market (RPS)':>26}")
    for name, row in table.items():
        cells = [f"{row[m]['estimate']:.4f} [{row[m]['low']:.4f}, {row[m]['high']:.4f}]"
                 for m in ("rps", "log_loss", "rps_minus_market")]
        print(f"{name:32}{cells[0]:>24}{cells[1]:>24}{cells[2]:>26}")


def main():
    matches = load_backtest_data()
    dev, holdout = holdout_split(matches.season.unique(), n_holdout=2)
    print(f"Tuning on {dev[-1]}, testing on {holdout} (never used for any choice)")
    best, tuning = tune(matches, dev[-1])
    print(f"Chosen: {best}")

    forecasts = []
    for season in holdout:
        print(f"Backtesting {season}, refitting every week...")
        f = forecast_season(matches, season, best["target"], best["half_life_days"])
        forecasts.append(with_base_rates(matches, season).join(f))
    backtest = pd.concat(forecasts)
    PREDICTIONS_FILE.parent.mkdir(exist_ok=True)
    backtest.to_csv(PREDICTIONS_FILE, index=False)

    table = score_table(backtest)
    print_table(table, len(backtest))

    # The model the app uses: same setup, fitted on every match up to the latest one
    as_of = matches.date.max() + pd.Timedelta(days=1)
    recent = matches[matches.date >= as_of - pd.Timedelta(days=WINDOW_DAYS)]
    teams = sorted(set(recent.home) | set(recent.away))
    save_bundle(fit_strengths(matches, teams, best["target"], as_of, best["half_life_days"], draws=1000),
                MODEL_FILE)
    save_metrics("match_model", {
        "chosen": best, "tuning": tuning, "tuning_season": int(dev[-1]),
        "holdout_seasons": [int(s) for s in holdout], "holdout_matches": int(len(backtest)),
        "scores": table,
        "market_source": backtest.source.value_counts().to_dict(),
        "fitted_through": str(matches.date.max().date()),
    })
    print(f"\nSaved {MODEL_FILE.name} and the scores in metrics.json")


if __name__ == "__main__":
    main()
