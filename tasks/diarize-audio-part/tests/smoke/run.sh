#!/usr/bin/env sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIRECTORY=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../.." && pwd)

: "${CERTIFICATION_TARGET:?Set CERTIFICATION_TARGET to a100 or gh200}"
: "${COMPATIBILITY_REPORT_PATH:?Set COMPATIBILITY_REPORT_PATH to an output JSON path}"
: "${MODEL_SMOKE_WAV:?Set MODEL_SMOKE_WAV to a normalized speech WAV}"

export RUN_MODEL_SMOKE_TEST=1
cd "$PROJECT_DIRECTORY"
uv run pytest -m model_smoke tests/smoke
