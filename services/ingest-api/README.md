# Voice Pipeline Ingest API

This service accepts podcast audio, normalizes it to a 16 kHz mono 16-bit PCM
WAV, stores the WAV, creates a `raw_audios` row, and publishes the row UUID to
the `split_raw_audio_into_parts` Celery task. It also provides read-only status
queries.

## Requirements

- Python 3.11 or newer
- FFmpeg installed before the service starts, with both `ffmpeg` and `ffprobe`
  available on `PATH`
- PostgreSQL
- Redis
- S3 or an S3-compatible object store

FFmpeg is a required system dependency for decoding MP3, AAC, OGG, and other
compressed input formats through pydub. Install it with the operating system or
container package manager before starting the service. A deployment should
verify both commands during image build or host provisioning:

```bash
ffmpeg -version
ffprobe -version
```

## Configuration

Deployment connections are supplied through environment variables:

```text
DATABASE_URL
CELERY_BROKER_URL
S3_BUCKET
S3_REGION
S3_ENDPOINT_URL
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
INGEST_CONFIG_FILE
```

`S3_ENDPOINT_URL`, the two AWS credential variables, and `INGEST_CONFIG_FILE`
are optional. Leave `S3_ENDPOINT_URL` unset when using AWS S3. Static AWS
credentials can be supplied through `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`; they are consumed directly by the standard AWS SDK
credential chain and are not copied into the service configuration model.
IAM roles, web identity, and other standard SDK credential providers continue
to work without these two variables.

The service always loads its packaged `resources/default.toml`. An optional
TOML file named by `INGEST_CONFIG_FILE` overrides only the keys it contains.
Unknown keys, invalid types, and invalid values prevent startup. Policy
settings cannot be overridden by environment variables.

The bundled defaults are:

```toml
[ingest]
max_upload_bytes = 314572800
max_concurrent_requests = 10
```

The concurrency limit applies independently to each HTTP worker process.
Additional `POST /v1/raw-audios` requests wait for a slot. Health, readiness,
and read-only routes bypass the ingest queue.

When the service is deployed behind NGINX, limit the complete multipart request
before it reaches FastAPI. The value includes 5 MiB for multipart metadata and
framing in addition to the 300 MiB audio limit:

```nginx
location = /v1/raw-audios {
    client_max_body_size 305m;
}
```

NGINX limits the complete request body. The service still enforces
`max_upload_bytes` against the exact original audio bytes.

## HTTP API

`GET /` serves the interactive Swagger API documentation.

`POST /v1/raw-audios` accepts `multipart/form-data` with these fields:

- `audio`: required, non-empty audio file
- `title`: optional text
- `source_url`: optional text stored as metadata; it is never fetched
- `lang`: optional text, default `en`
- `meta`: optional JSON object encoded as a form string

A new upload returns `202 Accepted` with the raw audio UUID, `pending` status,
the SHA-1 of the original upload bytes, and the Celery task ID. Existing
content returns `200 OK` with `deduplicated: true` and no task ID. A duplicate
is never normalized, uploaded, inserted, or republished, including when the
existing row has `failed` status.

If initial split-task publication fails, ingest changes the new row from
`pending` to `failed` with a conditional update. It never overwrites
`splitting` or `split_completed` when broker acknowledgement is ambiguous and
the split task has already advanced the row.

`GET /v1/raw-audios/{raw_audio_id}` returns the stored row without changing its
state or publishing a task. Invalid UUIDs return `422`, and missing rows return
`404`.

`POST /v1/tasks/trigger` publishes any task from the shared task registry. The
JSON request contains only the registered task name and the UUID passed as its
single positional argument:

```json
{
  "task_name": "split_raw_audio_into_parts",
  "id": "12345678-1234-5678-1234-567812345678"
}
```

A successful request returns `202 Accepted` with the same fields and the broker
`task_id`. Unknown task names, invalid UUIDs, and extra fields return `422`.
Broker publication failures return `503` without exposing broker details. The
endpoint never accepts arbitrary Celery task names or queue names.

The normalized object key is always:

```text
raw_audios/<raw_audio_id>/audio.wav
```

The database row is committed before Celery publication. A final publication
failure changes the committed row to `failed` with a safe stage-specific error
and returns `503` with the row UUID. The Celery message uses the shared task
registry, targets the `split_raw_audio_into_parts` queue explicitly, and contains
one positional argument only: the string form of `raw_audio_id`.

## Audio normalization

Original upload bytes are copied to a temporary file in fixed-size chunks, and
`content_sha1` is calculated from those original bytes. The temporary original
is never an S3 artifact.

pydub does not provide streaming decode and normalization. It decodes one
complete input inside an isolated child process, with channel and sample-rate
reductions applied before later conversions to limit peak memory where
possible. The child process has a fixed 30-second execution timeout. On a
timeout, its process group is terminated so that the decoder started by pydub
cannot remain behind, and any incomplete WAV is removed.

The service does not construct or execute an ffmpeg command directly. pydub
selects and invokes its configured decoder internally when the input format
requires one. Every completed output is validated as a non-empty 16 kHz mono
16-bit PCM WAV before it can be uploaded.

## Development

```bash
uv sync
uv run pytest
uv run uvicorn voice_pipeline_ingest_api.app:create_app --factory
```

At startup, the service loads `.env` from its current working directory when
the file exists. Variables already present in the process environment take
precedence and are never overwritten by `.env`. The file is intended for local
development and is excluded from Git; production deployments should inject
environment variables through the deployment platform.

`DATABASE_URL` may use `postgresql://`, `postgres://`, or the explicit
`postgresql+psycopg://` SQLAlchemy form. Generic PostgreSQL URLs are mapped to
the installed psycopg 3 driver automatically.

External integration tests are enabled when all of these variables are set:

```text
TEST_DATABASE_URL
TEST_CELERY_BROKER_URL
TEST_S3_BUCKET
TEST_S3_REGION
TEST_S3_ENDPOINT_URL
TEST_AWS_ACCESS_KEY_ID
TEST_AWS_SECRET_ACCESS_KEY
```

The integration suite receives these variables from the process environment.
Load the service `.env` through uv and select the integration marker explicitly:

```bash
uv run --env-file .env pytest -m integration tests/integration
```

Test S3 credentials are passed directly to test-only clients and are not copied
into the standard AWS environment variables. Integration tests are excluded
from the default `uv run pytest` command.

The target database, Redis database, and object-storage bucket must all be
dedicated to tests. The schema is initialized when `raw_audios` does not exist;
existing pipeline rows, queued Redis messages, and stored test objects are
removed between tests.

## Probes

- `GET /health` checks that the process is running.
- `GET /ready` checks PostgreSQL, S3, and the Celery broker.

Probe failures return a generic error and do not expose connection strings,
credentials, endpoints, or provider error messages.
