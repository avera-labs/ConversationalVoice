#!/usr/bin/env sh
set -eu
SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../.." && pwd)
OSS_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../../../.." && pwd)
test -f "$OSS_DIRECTORY/.env.test" || { echo "Missing OSS integration configuration: .env.test" >&2; exit 1; }
export RUN_INTEGRATION_TESTS=1
cd "$PROJECT_DIRECTORY"
uv run --extra test pytest -m integration tests/integration
