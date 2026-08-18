#!/usr/bin/env sh
set -eu
SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../.." && pwd)
test -f "$PROJECT_DIRECTORY/.env" || { echo "Missing task environment: .env" >&2; exit 1; }
export RUN_OPENROUTER_SMOKE=1
cd "$PROJECT_DIRECTORY"
uv run --extra test pytest -m provider_smoke tests/smoke
