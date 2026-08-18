#!/usr/bin/env sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../.." && pwd)

export RUN_CAPACITY_TESTS=1

cd "$PROJECT_DIRECTORY"
uv run pytest -m capacity tests/capacity
