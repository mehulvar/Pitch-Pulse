#!/usr/bin/env bash
# Rebuilds deploy/ (the folder Vercel serves) from backend/ and frontend/.
# Run it after changing the app or retraining, then commit deploy/ with the change.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
DEPLOY="$ROOT/deploy"
RECENT_SEASONS=3  # main.py shows the latest 3 seasons, so only those ship

# Vercel serves public/ itself, so drop the line where main.py serves the frontend
MOUNT_LINE='app.mount("/", StaticFiles(directory=BASE.parent / "frontend", html=True))'
grep -qF "$MOUNT_LINE" "$BACKEND/main.py" || { echo "Frontend mount line not found in main.py" >&2; exit 1; }
grep -vF "$MOUNT_LINE" "$BACKEND/main.py" | grep -v '^from fastapi.staticfiles import StaticFiles$' > "$DEPLOY/pitch_app.py"

cp "$BACKEND/features.py" "$BACKEND/match_predict.py" "$DEPLOY/"

mkdir -p "$DEPLOY/models" "$DEPLOY/public" "$DEPLOY/data"
cp "$BACKEND/models/xg_model.joblib" "$BACKEND/models/shots_model.joblib" \
   "$BACKEND/models/match_model.npz" "$BACKEND/models/metrics.json" "$DEPLOY/models/"
cp "$ROOT/frontend/index.html" "$ROOT/frontend/model-card.js" "$DEPLOY/public/"

rm -f "$DEPLOY"/data/shots_*.csv
ls "$BACKEND"/data/shots_[0-9]*.csv | sort | tail -n "$RECENT_SEASONS" | xargs -I{} cp {} "$DEPLOY/data/"

echo "deploy/ rebuilt with: $(ls "$DEPLOY/data" | tr '\n' ' ')"
