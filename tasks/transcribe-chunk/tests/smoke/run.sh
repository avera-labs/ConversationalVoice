#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${PROJECT_DIR}"
RUN_MODEL_SMOKE_TEST=1 uv run --extra test pytest -q tests/smoke/test_real_model.py
