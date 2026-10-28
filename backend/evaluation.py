"""Scores and checks shared by the training scripts, so every model is graded the same way."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS_FILE = Path(__file__).parent / "models" / "metrics.json"  # what the model card shows
BOOTSTRAP_ROUNDS = 1000
CONFIDENCE = 0.95


def _as_arrays(probs, outcome):
    """Forecasts as an (n, 3) array of home/draw/away chances, outcomes as 0, 1 or 2."""
    return np.asarray(probs, dtype=float), np.asarray(outcome, dtype=int)


def rps(probs, outcome):
    """Ranked probability score for home/draw/away forecasts. 0 is perfect, lower is better.
    It compares running totals, so calling a home win a draw is less wrong than an away win."""
    p, o = _as_arrays(probs, outcome)
    actual = np.eye(3)[o]
    gaps = np.cumsum(p, axis=1)[:, :2] - np.cumsum(actual, axis=1)[:, :2]
    return float(np.mean(np.sum(gaps**2, axis=1) / 2))


def log_loss_3way(probs, outcome, eps=1e-12):
    """Average of -log(chance given to what actually happened). Lower is better."""
    p, o = _as_arrays(probs, outcome)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(o)), o], eps, 1))))


def brier_3way(probs, outcome):
    """Squared error summed over the three outcomes, averaged over matches. Lower is better."""
    p, o = _as_arrays(probs, outcome)
    return float(np.mean(np.sum((p - np.eye(3)[o]) ** 2, axis=1)))


def bootstrap_ci(metric, df, group_col, n=BOOTSTRAP_ROUNDS, seed=0):
    """A metric with a 95% confidence interval.
    Whole groups (matches) are resampled, not single rows, because shots in one match are related."""
    rows_by_group = list(df.groupby(group_col).indices.values())
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        picked = rng.integers(0, len(rows_by_group), len(rows_by_group))
        scores.append(metric(df.iloc[np.concatenate([rows_by_group[i] for i in picked])]))
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(scores, [tail, 100 - tail])
    return {"estimate": float(metric(df)), "low": float(low), "high": float(high)}


def calibration_table(y, p, bins=10):
    """For each group of similar forecasts: the average forecast and how often it really happened.
    A well calibrated model has the two close together in every row."""
    df = pd.DataFrame({"p": np.asarray(p, dtype=float), "y": np.asarray(y, dtype=float)})
    df["bin"] = pd.qcut(df.p, bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True)
    return [{"predicted": float(g.p.mean()), "actual": float(g.y.mean()), "count": int(len(g))}
            for _, g in grouped]


def save_metrics(section, values, path=METRICS_FILE):
    """Write one model's results into metrics.json, keeping the other models' results."""
    path = Path(path)
    current = json.loads(path.read_text()) if path.exists() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**current, section: values}, indent=2))
