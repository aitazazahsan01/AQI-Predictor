#!/usr/bin/env bash
# Convenience wrapper: activates the venv, loads .env, and runs a pipeline.
#
#   ./run.sh test              run the unit tests
#   ./run.sh hourly            Module 1 - ingest the current hour
#   ./run.sh daily             Module 2 - build today's engineered features
#   ./run.sh backfill-dry      Module 3 - preview the historical fetch (no writes)
#   ./run.sh backfill          Module 3 - load ~4 years of history
#   ./run.sh train-offline     Module 4 - train on data fetched straight from the API
#   ./run.sh train             Module 4 - train from the feature store and register models
#
# Extra arguments are passed through, e.g.  ./run.sh daily --all

set -euo pipefail

VENV="${AQI_VENV:-$HOME/venvs/aqi-predictor}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$VENV/bin/activate" ]; then
  echo "No virtualenv at $VENV" >&2
  echo "Create one with:  python3 -m venv $VENV && $VENV/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJECT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "Warning: no .env file found - pipelines needing credentials will fail." >&2
fi

command="${1:-}"
shift || true

case "$command" in
  test)          exec python -m pytest tests/ "$@" ;;
  hourly)        exec python scripts/run_feature_pipeline.py "$@" ;;
  daily)         exec python scripts/run_daily_aggregation.py "$@" ;;
  backfill-dry)  exec python scripts/backfill_historical.py --dry-run "$@" ;;
  backfill)      exec python scripts/backfill_historical.py "$@" ;;
  train-offline) exec python scripts/run_training_pipeline.py --offline "$@" ;;
  train)         exec python scripts/run_training_pipeline.py "$@" ;;
  *)
    echo "Usage: ./run.sh {test|hourly|daily|backfill-dry|backfill|train-offline|train} [args...]" >&2
    exit 1
    ;;
esac
