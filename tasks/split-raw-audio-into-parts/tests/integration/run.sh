#!/usr/bin/env sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../.." && pwd)
OSS_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../../../.." && pwd)
ENV_FILE="$OSS_DIRECTORY/.env.test"

if [ ! -f "$ENV_FILE" ]; then
    echo "Missing OSS integration configuration: .env.test" >&2
    exit 1
fi

export RUN_INTEGRATION_TESTS=1

cd "$PROJECT_DIRECTORY"
uv run pytest -m integration tests/integration
