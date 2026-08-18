#!/usr/bin/env sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../.." && pwd)

: "${CERTIFICATION_TARGET:?Set CERTIFICATION_TARGET to a100 or gh200}"
: "${CAPACITY_REPORT_PATH:?Set CAPACITY_REPORT_PATH to an output JSON path}"
: "${CAPACITY_SOURCE_WAV:?Set CAPACITY_SOURCE_WAV to a normalized speech WAV}"

export RUN_CAPACITY_TESTS=1
cd "$PROJECT_DIRECTORY"
uv run pytest -m capacity tests/capacity
