# Split Raw Audio Into Parts

This independent Celery task project turns one normalized raw-audio WAV into
deterministic conversation windows. It uses the shared task registry for its
name, queue, and positional UUID argument contract.

## Input contract

The input artifact is a non-empty 16 kHz, mono, 16-bit PCM WAV. Ingest owns
that normalization and validation boundary. This task reads PCM sample frames
directly and does not invoke FFmpeg or pydub.

## Configuration

Deployment connections use these environment variables:

```text
DATABASE_URL
CELERY_BROKER_URL
S3_BUCKET
S3_REGION
S3_ENDPOINT_URL
HF_TOKEN
```

`S3_ENDPOINT_URL` is optional. AWS credentials are resolved through the
standard SDK credential chain and are not copied into application settings.
The model and windowing policy is loaded from the packaged
`resources/default.toml`. A reviewed TOML override can be supplied directly to
the settings loader; individual policy values are not environment variables.

`max_window_ms` is a hard limit after padding. When both requested padding
values cannot fit, the available frames are balanced between the two sides;
capacity blocked by an audio boundary is reassigned to the other side.

The bundled defaults are:

```toml
[vad]
model = "pyannote/segmentation-3.0"
device = "cpu"

[windowing]
gap_threshold_ms = 15000
min_window_ms = 20000
max_window_ms = 900000
pad_before_ms = 250
pad_after_ms = 250
```

## Task composition

`create_celery_app` configures JSON messages, late acknowledgement, worker-loss
rejection, one-message prefetch, and the registered dedicated queue.
`register_split_task` binds the shared Celery contract to a callable handler.
`TaskRuntime` composes the repository, storage client, lazy VAD wrapper,
downstream publisher, handler, and registered task. The downstream publisher is a
thin adapter that selects `DIARIZE_AUDIO_PART` and delegates broker behavior to
`voice-pipeline-task-client`. The `worker` module loads configuration at
startup and releases process-local clients on worker shutdown.

For a claimed row, the handler runs these operations in order:

1. Download the normalized WAV.
2. Run `pyannote/segmentation-3.0` and upload `vad_segments.json`.
3. Build deterministic windows, cut each PCM range, and upload every part.
4. Insert missing `audio_parts` and set `raw_audios.status` to
   `split_completed` in one database transaction.
5. After commit, publish one registered diarization task for each pending part.

The atomic status claim is the first persistence operation after UUID parsing.
A delivery that observes `splitting` returns `already_processing` without file,
model, storage, or broker work. A delivery that observes `split_completed`
skips model and artifact processing and publishes only for existing pending
parts.

The pure `vad_artifact` module serializes normalized model output into the
stable `vad_segments.json` contract before a storage adapter uploads it. Empty
VAD output still produces an artifact with an empty `segments` array.

The `vad` adapter imports PyTorch and pyannote only when inference is first
requested, then reuses the pipeline from a process-local cache. The bundled
policy uses `device = "cpu"` as the portable default. With an explicitly
reviewed `device = "auto"` override, selection order is CUDA, MPS, then CPU.
An explicit `cuda` or `mps` setting fails instead of silently falling back
when that device is unavailable. Run the real-model smoke test inside every
GPU deployment image before selecting `auto` or `cuda`.

The adapter reads the normalized 16 kHz mono 16-bit PCM frames into an
in-memory waveform payload, so pyannote does not invoke a separate media
decoder. It converts model timestamps to deterministic sample frames, clamps
and merges speech spans, and writes the same normalized result to
`vad_segments.json`.

The infrastructure adapters keep transactions short and do not use database
locks. Task ownership is claimed with one conditional
`pending|failed -> splitting` update. A concurrent delivery that observes
`splitting` returns without processing, while `split_completed` is available
for pending downstream-dispatch recovery. Because the schema has no processing
lease, a stale `splitting` row must be changed to `failed` by a separately
reviewed operational recovery process before redelivery.

All expected part objects are uploaded before persistence. The repository then
inserts each `(raw_audio_id, part_index)` with `ON CONFLICT DO NOTHING`, reuses
the existing row identifier for duplicates, and changes the raw audio to
`split_completed` in the same transaction. The storage adapter accepts only
canonical `s3://` URIs in the configured bucket and never deletes deterministic
objects on a failed attempt.

Downstream publishing references the shared `diarize_audio_part` task name and
queue and sends one positional `audio_part_id`. It uses bounded publish retries
and supplies a confirmation timeout to transports that support publisher
confirms. Redis acknowledges its enqueue command but does not provide AMQP-style
publisher confirms. External database, storage, and broker exception text is
retained only as an exception cause; persisted task errors are selected from
bounded, reviewed messages.

The worker must run with a solo pool and concurrency one. Audio, model output,
and local file paths are never sent through Redis.

## Running the worker

The module fails fast when required environment or TOML configuration is
invalid. Model weights remain unloaded until a claimed task reaches inference.

```bash
uv run celery \
  -A voice_pipeline_split_raw_audio_into_parts.worker:app \
  worker \
  --loglevel=INFO \
  --pool=solo \
  --concurrency=1 \
  --queues=split_raw_audio_into_parts
```

## Development

The default suite is self-contained and skips external integration, capacity,
and real-model smoke tests:

```bash
uv sync
uv run pytest
```

The integration suite uses the existing OSS-root `.env.test` file and expects
these variable names:

```text
TEST_DATABASE_URL
TEST_CELERY_BROKER_URL
TEST_S3_BUCKET
TEST_S3_REGION
TEST_S3_ENDPOINT_URL
TEST_AWS_ACCESS_KEY_ID
TEST_AWS_SECRET_ACCESS_KEY
TEST_HF_TOKEN
```

The configured database must already contain the authoritative OSS schema, the
bucket must already exist, and the dedicated diarization queue must initially
be empty. Tests use random raw-audio UUIDs and remove only the rows, object
prefixes, and messages that those UUIDs own. They do not create, drop, or
truncate shared infrastructure.

```bash
tests/integration/run.sh
```

The real-model smoke test uses CPU by default. A GPU deployment image must pass
the same test with its intended device before its policy selects that device:

```bash
tests/smoke/run.sh
TEST_VAD_DEVICE=cuda tests/smoke/run.sh
```

The capacity check creates a 901-second local PCM WAV, streams a 900-second
part, and verifies bounded Python memory use:

```bash
tests/capacity/run.sh
```
