from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
INPUT_DIR = BASE / "data"
OUTPUT_FILE = INPUT_DIR / "shots.csv"

REQUIRED_COLUMNS = {
    "match_id", "h_a", "h_team", "a_team", "X", "Y", "xG", "result", "minute",
}


def transform_shots(raw):
    """Convert Understat shot rows to the columns used by the API and frontend."""
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Understat data is missing required columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError("Understat data contains no shots")

    invalid_sides = set(raw["h_a"].dropna().unique()) - {"h", "a"}
    if invalid_sides or raw["h_a"].isna().any():
        raise ValueError(f"Understat data contains invalid home/away values: {invalid_sides}")

    coordinates = raw[["X", "Y", "xG"]].apply(pd.to_numeric, errors="raise")
    if coordinates.isna().any().any():
        raise ValueError("Understat coordinates and xG must not be missing")
    if not coordinates["X"].between(0, 1).all() or not coordinates["Y"].between(0, 1).all():
        raise ValueError("Understat shot coordinates must be between 0 and 1")
    if not coordinates["xG"].between(0, 1).all():
        raise ValueError("Understat xG values must be between 0 and 1")

    is_home = raw["h_a"].eq("h")
    return pd.DataFrame({
        "match_id": raw["match_id"],
        "team": raw["h_team"].where(is_home, raw["a_team"]),
        "against": raw["a_team"].where(is_home, raw["h_team"]),
        # Map normalized Understat coordinates to the attacking half shown by the app.
        "x": 60 + coordinates["X"] * 60,
        "y": coordinates["Y"] * 80,
        "xg": coordinates["xG"],
        "goal": raw["result"].eq("Goal"),
        "minute": pd.to_numeric(raw["minute"], errors="raise"),
    })


def build_data():
    """Build the app's shot table from all downloaded Understat season files."""
    source_files = sorted(INPUT_DIR.glob("shots_*.csv"))
    if not source_files:
        raise FileNotFoundError(
            f"No downloaded seasons found in {INPUT_DIR}. Run download_epl.py first."
        )

    raw = pd.concat((pd.read_csv(path) for path in source_files), ignore_index=True)
    shots = transform_shots(raw)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    shots.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(shots)} EPL shots from {len(source_files)} season file(s) to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_data()
