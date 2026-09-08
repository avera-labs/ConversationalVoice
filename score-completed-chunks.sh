#!/bin/sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

exec uv run \
  --project "${SCRIPT_DIRECTORY}/tools/score-completed-chunks" \
  score-completed-chunks "$@"
