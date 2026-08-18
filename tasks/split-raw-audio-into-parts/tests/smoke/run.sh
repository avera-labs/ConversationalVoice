#!/usr/bin/env sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../.." && pwd)
OSS_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../../../.." && pwd)

if [ ! -f "$OSS_DIRECTORY/.env.test" ]; then
    echo "Missing OSS model smoke configuration: .env.test" >&2
    exit 1
fi

export RUN_MODEL_SMOKE_TEST=1

cd "$PROJECT_DIRECTORY"
uv run pytest -m model_smoke tests/smoke
