"""Bayesian match model: every club gets an attack and a defence rating, learned from past matches.

    log(expected goals for the home side) = level + home advantage + home attack - away defence
    log(expected goals for the away side) = level                  + away attack - home defence

Ratings are pulled toward the league average (partial pooling), so a club with a few lucky
results isn't rated as a superpower. Recent matches count more than old ones.
Training only: the app reads the saved draws through match_predict.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

DATA_DIR = Path(__file__).parent / "data"
WINDOW_DAYS = 4 * 365  # older matches barely count after decay, so leave them out to fit faster
MIN_XG = 0.05          # a side with no shots has 0 xG, which the Gamma likelihood can't take
N_WEAKEST = 3          # promoted clubs start at the average of the 3 weakest rated clubs
TARGETS = ("goals", "xg")


def load_matches(data_dir=DATA_DIR):
    """Every downloaded match, oldest first, with the result as 0 (home win), 1 (draw), 2 (away win)."""
    files = sorted(Path(data_dir).glob("matches_*.csv"))
    if not files:
        raise SystemExit("No matches_*.csv in data/. Run download_epl.py first")
    matches = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    matches["date"] = pd.to_datetime(matches.date)
    diff = matches.home_goals - matches.away_goals
    matches["outcome"] = np.select([diff > 0, diff == 0], [0, 1], 2)
    return matches.sort_values("date").reset_index(drop=True)


def time_weights(dates, as_of, half_life_days):
    """How much each match counts: 1 for today, 0.5 one half-life ago, 0.25 two half-lives ago."""
    age_days = np.asarray((as_of - pd.to_datetime(dates)) / pd.Timedelta(days=1), dtype=float)
    return 0.5 ** (age_days / half_life_days)


def _observed(target, matches):
    """What the model learns from: goals scored, or xG created (less luck, more signal)."""
    if target == "goals":
        return matches.home_goals.to_numpy(), matches.away_goals.to_numpy()
    if target == "xg":
        return (np.maximum(matches.home_xg_understat.to_numpy(), MIN_XG),
                np.maximum(matches.away_xg_understat.to_numpy(), MIN_XG))
    raise ValueError(f"target must be one of {TARGETS}, got {target}")


def fit_strengths(matches, teams, target, as_of, half_life_days, draws=500, tune=500, seed=0):
    """Fit the model on matches before as_of. Returns posterior draws for every club in teams."""
    past = matches[(matches.date < as_of) & (matches.date >= as_of - pd.Timedelta(days=WINDOW_DAYS))]
    index = {t: i for i, t in enumerate(teams)}
    home, away = past.home.map(index).to_numpy(), past.away.map(index).to_numpy()
    obs_home, obs_away = _observed(target, past)
    weight = time_weights(past.date, as_of, half_life_days)

    with pm.Model():
        level = pm.Normal("mu", 0.3, 0.5)
        home_adv = pm.Normal("home_adv", 0.2, 0.2)
        # Sum to zero, so a rating means "better or worse than the average club"
        attack = pm.ZeroSumNormal("attack", sigma=pm.HalfNormal("sigma_attack", 0.5), shape=len(teams))
        defence = pm.ZeroSumNormal("defence", sigma=pm.HalfNormal("sigma_defence", 0.5), shape=len(teams))
        rate_home = pm.math.exp(level + home_adv + attack[home] - defence[away])
        rate_away = pm.math.exp(level + attack[away] - defence[home])

        if target == "goals":
            dist_home, dist_away = pm.Poisson.dist(mu=rate_home), pm.Poisson.dist(mu=rate_away)
        else:
            shape = pm.HalfNormal("xg_shape", 5.0)  # how spread out a side's xG is around its rate
            dist_home = pm.Gamma.dist(alpha=shape, beta=shape / rate_home)
            dist_away = pm.Gamma.dist(alpha=shape, beta=shape / rate_away)
        # Weighted likelihood: each match's evidence counts in proportion to how recent it is
        pm.Potential("weighted_fit", (weight * (pm.logp(dist_home, obs_home) + pm.logp(dist_away, obs_away))).sum())

        trace = pm.sample(draws=draws, tune=tune, chains=2, nuts_sampler="nutpie",
                          random_seed=seed, progressbar=False)

    post = trace["posterior"].to_dataset().stack(sample=("chain", "draw"))  # one row per draw
    bundle = {
        "teams": np.array(teams),
        "mu": post["mu"].to_numpy(),
        "home_adv": post["home_adv"].to_numpy(),
        "attack": post["attack"].to_numpy().T,    # draws x teams
        "defence": post["defence"].to_numpy().T,
    }
    unseen = sorted(set(teams) - set(past.home) - set(past.away))
    return fill_unseen(bundle, unseen) if unseen else bundle


def fill_unseen(bundle, unseen, n_weakest=N_WEAKEST):
    """Clubs with no matches in the window (newly promoted) get the average rating of the weakest
    clubs, draw by draw. Promoted sides usually replace the weakest ones, so that's a fair start."""
    teams = list(bundle["teams"])
    unseen_idx = [teams.index(t) for t in unseen]
    seen_idx = [i for i in range(len(teams)) if i not in unseen_idx]
    total = bundle["attack"][:, seen_idx] + bundle["defence"][:, seen_idx]
    weakest = np.take_along_axis(np.array(seen_idx)[None, :].repeat(len(total), 0),
                                 np.argsort(total, axis=1)[:, :n_weakest], axis=1)
    filled = {**bundle, "attack": bundle["attack"].copy(), "defence": bundle["defence"].copy()}
    for key in ("attack", "defence"):
        start = np.take_along_axis(bundle[key], weakest, axis=1).mean(axis=1)
        filled[key][:, unseen_idx] = start[:, None]
    return filled


def save_bundle(bundle, path):
    np.savez_compressed(path, **bundle)
