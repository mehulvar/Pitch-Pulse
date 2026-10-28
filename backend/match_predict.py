"""Turn the fitted match model into win/draw/loss chances and likely scores.
Numpy and scipy only, so the app can use it without installing PyMC."""
from pathlib import Path

import numpy as np
from scipy.stats import poisson

MODEL_FILE = Path(__file__).parent / "models" / "match_model.npz"
MAX_GOALS = 10  # scores above 10 are so rare they don't change the chances
TOP_SCORELINES = 3


def load_bundle(path=MODEL_FILE):
    """The saved posterior draws: one row per draw, one column per team."""
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def scoring_rates(bundle, home, away):
    """Expected goals for each side, once per posterior draw.
    log rate = overall level + home advantage + attack of the side - defence of the other side."""
    teams = list(bundle["teams"])
    if home not in teams or away not in teams:
        missing = home if home not in teams else away
        raise KeyError(f"The match model has no rating for {missing}")
    h, a = teams.index(home), teams.index(away)
    rate_home = np.exp(bundle["mu"] + bundle["home_adv"] + bundle["attack"][:, h] - bundle["defence"][:, a])
    rate_away = np.exp(bundle["mu"] + bundle["attack"][:, a] - bundle["defence"][:, h])
    return rate_home, rate_away


def outcome_probabilities(rate_home, rate_away):
    """Chance of a home win, draw and away win, plus the chance of every scoreline up to 10-10.
    Each side's goals are Poisson. Averaging over the draws carries the model's uncertainty through."""
    goals = np.arange(MAX_GOALS + 1)
    home_pmf = poisson.pmf(goals[None, :], rate_home[:, None])  # draws x goals
    away_pmf = poisson.pmf(goals[None, :], rate_away[:, None])
    grid = np.einsum("di,dj->ij", home_pmf, away_pmf)  # home goals x away goals
    grid = grid / grid.sum()  # averages the draws and puts back the sliver lost above 10 goals
    return (float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())), grid


def predict_match(bundle, home, away):
    rate_home, rate_away = scoring_rates(bundle, home, away)
    (p_home, p_draw, p_away), grid = outcome_probabilities(rate_home, rate_away)
    best = np.argsort(grid, axis=None)[::-1][:TOP_SCORELINES]
    return {
        "p_home": round(p_home, 3), "p_draw": round(p_draw, 3), "p_away": round(p_away, 3),
        "expected_goals_home": round(float(rate_home.mean()), 2),
        "expected_goals_away": round(float(rate_away.mean()), 2),
        "top_scorelines": [{"home": int(i // grid.shape[1]), "away": int(i % grid.shape[1]),
                            "chance": round(float(grid.flat[i]), 3)} for i in best],
    }
