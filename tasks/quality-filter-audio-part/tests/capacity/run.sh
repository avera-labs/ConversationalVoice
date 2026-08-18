#!/usr/bin/env sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../.." && pwd)

: "${CAPACITY_SOURCE_WAV:?Set CAPACITY_SOURCE_WAV to a normalized WAV}"
: "${CAPACITY_REPORT_PATH:?Set CAPACITY_REPORT_PATH to an output JSON path}"

export RUN_CAPACITY_TESTS=1
cd "$PROJECT_DIRECTORY"
uv run pytest -m capacity tests/capacity
