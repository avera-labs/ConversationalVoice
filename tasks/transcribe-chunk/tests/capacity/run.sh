#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${PROJECT_DIR}"
RUN_MODEL_CAPACITY_TEST=1 uv run --extra test pytest -q tests/capacity/test_120_second_slice.py
