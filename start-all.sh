#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_ENV="${PROJECT_ROOT}/.env"
INGEST_API_DIR="${PROJECT_ROOT}/services/ingest-api"
SPLIT_TASK_DIR="${PROJECT_ROOT}/tasks/split-raw-audio-into-parts"
DIARIZATION_TASK_DIR="${PROJECT_ROOT}/tasks/diarize-audio-part"
QUALITY_FILTER_TASK_DIR="${PROJECT_ROOT}/tasks/quality-filter-audio-part"
SEPARATE_CHUNK_TASK_DIR="${PROJECT_ROOT}/tasks/separate-chunk"
TRANSCRIBE_CHUNK_TASK_DIR="${PROJECT_ROOT}/tasks/transcribe-chunk"
PERSONA_CHUNK_TASK_DIR="${PROJECT_ROOT}/tasks/persona-chunk"
EXTEND_CHUNK_TASK_DIR="${PROJECT_ROOT}/tasks/extend-chunk"

HTTP_HOST="${HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${HTTP_PORT:-8000}"
CELERY_LOG_LEVEL="${CELERY_LOG_LEVEL:-INFO}"

CHILD_PIDS=()
CHILD_NAMES=()

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./start-all.sh

Starts the ingest HTTP server and every Celery task worker in this OSS tree.
Before starting processes, synchronizes every project from its lock file and
reinstalls its local shared packages so source changes cannot remain cached.

Optional environment variables:
  HTTP_HOST         Uvicorn bind address (default: 0.0.0.0)
  HTTP_PORT         Uvicorn bind port (default: 8000)
  CELERY_LOG_LEVEL  Celery log level (default: INFO)
EOF
}

ensure_known_task_directories() {
  local task_dir
  local task_name

  for task_dir in "${PROJECT_ROOT}"/tasks/*; do
    [[ -d "${task_dir}" ]] || continue
    task_name="${task_dir##*/}"
    case "${task_name}" in
      split-raw-audio-into-parts | diarize-audio-part | quality-filter-audio-part | separate-chunk | transcribe-chunk | persona-chunk | extend-chunk) ;;
      *) die "No start command is registered for tasks/${task_name}." ;;
    esac
  done
}

ensure_env_link() {
  local workdir="$1"
  local target="${workdir}/.env"

  if [[ -e "${target}" || -L "${target}" ]]; then
    printf 'Environment already exists, skipping: %s\n' "${target}"
    return
  fi

  ln -s "../../.env" "${target}"
  printf 'Linked environment: %s -> ../../.env\n' "${target}"
}

sync_local_packages() {
  local workdir="$1"
  shift
  local args=(sync --locked --inexact)
  local package

  for package in "$@"; do
    args+=(--reinstall-package "${package}")
  done

  printf 'Synchronizing local packages: %s\n' "${workdir}"
  (
    cd -- "${workdir}"
    uv "${args[@]}"
  )
}

verify_task_registry() {
  printf 'Verifying shared task registry\n'
  (
    cd -- "${INGEST_API_DIR}"
    uv run --no-sync python - <<'PY'
from voice_pipeline_task_contracts import ALL_TASKS

expected = {
    "split_raw_audio_into_parts",
    "diarize_audio_part",
    "quality_filter_audio_part",
    "separate_chunk",
    "transcribe_chunk",
    "persona_chunk",
    "extend_chunk",
}
actual = {contract.name for contract in ALL_TASKS}
if actual != expected:
    missing = ", ".join(sorted(expected - actual)) or "none"
    unexpected = ", ".join(sorted(actual - expected)) or "none"
    raise SystemExit(
        f"Task registry mismatch: missing={missing}; unexpected={unexpected}"
    )
print("Task registry verified: " + ", ".join(sorted(actual)))
PY
  )
}

start_process() {
  local name="$1"
  local workdir="$2"
  shift 2

  printf 'Starting %s\n' "${name}"
  (
    cd -- "${workdir}"
    exec uv run "$@"
  ) &
  CHILD_PIDS+=("$!")
  CHILD_NAMES+=("${name}")
}

cleanup() {
  local exit_code=$?

  trap - EXIT INT TERM
  if ((${#CHILD_PIDS[@]} > 0)); then
    printf '\nStopping all services...\n'
    kill -TERM "${CHILD_PIDS[@]}" 2>/dev/null || true
    wait "${CHILD_PIDS[@]}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}

if (($# > 0)); then
  if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
  fi
  usage >&2
  exit 2
fi

command -v uv >/dev/null 2>&1 || die "uv is not installed or is not on PATH."
[[ -f "${ROOT_ENV}" ]] || die "Missing OSS root environment file: ${ROOT_ENV}"

for workdir in "${INGEST_API_DIR}" "${SPLIT_TASK_DIR}" "${DIARIZATION_TASK_DIR}" "${QUALITY_FILTER_TASK_DIR}" "${SEPARATE_CHUNK_TASK_DIR}" "${TRANSCRIBE_CHUNK_TASK_DIR}" "${PERSONA_CHUNK_TASK_DIR}" "${EXTEND_CHUNK_TASK_DIR}"; do
  [[ -d "${workdir}" ]] || die "Missing service directory: ${workdir}"
done
ensure_known_task_directories

ensure_env_link "${INGEST_API_DIR}"
ensure_env_link "${SPLIT_TASK_DIR}"
ensure_env_link "${DIARIZATION_TASK_DIR}"
ensure_env_link "${QUALITY_FILTER_TASK_DIR}"
ensure_env_link "${SEPARATE_CHUNK_TASK_DIR}"
ensure_env_link "${TRANSCRIBE_CHUNK_TASK_DIR}"
ensure_env_link "${PERSONA_CHUNK_TASK_DIR}"
ensure_env_link "${EXTEND_CHUNK_TASK_DIR}"

sync_local_packages \
  "${INGEST_API_DIR}" \
  voice-pipeline-models \
  voice-pipeline-task-client \
  voice-pipeline-task-contracts
sync_local_packages \
  "${SPLIT_TASK_DIR}" \
  voice-pipeline-models \
  voice-pipeline-task-client \
  voice-pipeline-task-contracts
sync_local_packages \
  "${DIARIZATION_TASK_DIR}" \
  voice-pipeline-models \
  voice-pipeline-diarization-artifact \
  voice-pipeline-task-client \
  voice-pipeline-task-contracts
sync_local_packages \
  "${QUALITY_FILTER_TASK_DIR}" \
  voice-pipeline-diarization-artifact \
  voice-pipeline-models \
  voice-pipeline-task-client \
  voice-pipeline-task-contracts
sync_local_packages \
  "${SEPARATE_CHUNK_TASK_DIR}" \
  voice-pipeline-chunk-contracts \
  voice-pipeline-diarization-artifact \
  voice-pipeline-models \
  voice-pipeline-task-client \
  voice-pipeline-task-contracts
sync_local_packages \
  "${TRANSCRIBE_CHUNK_TASK_DIR}" \
  voice-pipeline-chunk-contracts \
  voice-pipeline-models \
  voice-pipeline-task-client \
  voice-pipeline-task-contracts
sync_local_packages \
  "${PERSONA_CHUNK_TASK_DIR}" \
  voice-pipeline-chunk-contracts \
  voice-pipeline-models \
  voice-pipeline-task-client \
  voice-pipeline-task-contracts
sync_local_packages \
  "${EXTEND_CHUNK_TASK_DIR}" \
  voice-pipeline-chunk-contracts \
  voice-pipeline-models \
  voice-pipeline-task-contracts
verify_task_registry

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_process \
  "ingest HTTP server (${HTTP_HOST}:${HTTP_PORT})" \
  "${INGEST_API_DIR}" \
  uvicorn voice_pipeline_ingest_api.app:create_app \
  --factory \
  --host "${HTTP_HOST}" \
  --port "${HTTP_PORT}"

# Solo workers cannot consume gossip events while inference is blocking. Peer
# discovery is not used here, so disable it to avoid stale-heartbeat warnings.
start_process \
  "split_raw_audio_into_parts worker" \
  "${SPLIT_TASK_DIR}" \
  celery \
  -A voice_pipeline_split_raw_audio_into_parts.worker:app \
  worker \
  --loglevel="${CELERY_LOG_LEVEL}" \
  --pool=solo \
  --concurrency=1 \
  --without-gossip \
  --without-mingle \
  --queues=split_raw_audio_into_parts \
  --hostname=split-raw-audio-into-parts@%h

start_process \
  "diarize_audio_part worker" \
  "${DIARIZATION_TASK_DIR}" \
  celery \
  -A voice_pipeline_diarize_audio_part.worker:app \
  worker \
  --loglevel="${CELERY_LOG_LEVEL}" \
  --pool=solo \
  --concurrency=1 \
  --without-gossip \
  --without-mingle \
  --queues=diarize_audio_part \
  --hostname=diarize-audio-part@%h

start_process \
  "quality_filter_audio_part worker" \
  "${QUALITY_FILTER_TASK_DIR}" \
  celery \
  -A voice_pipeline_quality_filter_audio_part.worker:app \
  worker \
  --loglevel="${CELERY_LOG_LEVEL}" \
  --pool=solo \
  --concurrency=1 \
  --without-gossip \
  --without-mingle \
  --queues=quality_filter_audio_part \
  --hostname=quality-filter-audio-part@%h

start_process \
  "separate_chunk worker" \
  "${SEPARATE_CHUNK_TASK_DIR}" \
  celery \
  -A voice_pipeline_separate_chunk.worker:app \
  worker \
  --loglevel="${CELERY_LOG_LEVEL}" \
  --pool=solo \
  --concurrency=1 \
  --without-gossip \
  --without-mingle \
  --queues=separate_chunk \
  --hostname=separate-chunk@%h

start_process \
  "transcribe_chunk worker" \
  "${TRANSCRIBE_CHUNK_TASK_DIR}" \
  celery \
  -A voice_pipeline_transcribe_chunk.worker:app \
  worker \
  --loglevel="${CELERY_LOG_LEVEL}" \
  --pool=solo \
  --concurrency=1 \
  --without-gossip \
  --without-mingle \
  --queues=transcribe_chunk \
  --hostname=transcribe-chunk@%h

start_process \
  "persona_chunk worker" \
  "${PERSONA_CHUNK_TASK_DIR}" \
  celery \
  -A voice_pipeline_persona_chunk.worker:app \
  worker \
  --loglevel="${CELERY_LOG_LEVEL}" \
  --pool=solo \
  --concurrency=1 \
  --without-gossip \
  --without-mingle \
  --queues=persona_chunk \
  --hostname=persona-chunk@%h

start_process \
  "extend_chunk worker" \
  "${EXTEND_CHUNK_TASK_DIR}" \
  celery \
  -A voice_pipeline_extend_chunk.worker:app \
  worker \
  --loglevel="${CELERY_LOG_LEVEL}" \
  --pool=solo \
  --concurrency=1 \
  --without-gossip \
  --without-mingle \
  --queues=extend_chunk \
  --hostname=extend-chunk@%h

printf 'All services are running. Press Ctrl-C to stop them.\n'

set +e
FINISHED_PID=""
wait -n -p FINISHED_PID "${CHILD_PIDS[@]}"
EXIT_CODE=$?
set -e

FINISHED_NAME="unknown service"
for index in "${!CHILD_PIDS[@]}"; do
  if [[ "${CHILD_PIDS[${index}]}" == "${FINISHED_PID}" ]]; then
    FINISHED_NAME="${CHILD_NAMES[${index}]}"
    break
  fi
done

printf 'Service exited: %s (status %s)\n' "${FINISHED_NAME}" "${EXIT_CODE}" >&2
if ((EXIT_CODE == 0)); then
  EXIT_CODE=1
fi
exit "${EXIT_CODE}"
