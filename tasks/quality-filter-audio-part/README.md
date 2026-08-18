# Audio Part Quality-Filter Worker

This standalone Celery worker claims one diarized audio part, evaluates speech
quality, plans strict two-speaker dialogue windows, uploads deterministic chunk
WAV files, and commits all chunk rows together with the completed part state.

See [FLOW.md](FLOW.md) for the end-to-end quality and window-planning flow.

## Runtime contract

- Task and queue: `quality_filter_audio_part`
- Argument: one canonical `audio_part_id` UUID string
- Inputs: persisted absolute `s3://` audio and diarization URIs
- Output key: `chunks/<chunk_index>/audio.wav` below the part directory
- Worker pool: `solo`, concurrency `1`, prefetch multiplier `1`
- Claim: one compare-and-set update from `diarized` or `failed` to `filtering`

The worker derives speech intervals from the union of diarization turns. Music
detection runs once over the whole part. WADA SNR and music overlap are then
evaluated per speech interval. Quality groups produce good regions, and each
region is independently processed by the Sidon raw-window planner. Greedy merge
never crosses a good-region boundary and has no final duration cap; it continues
while the union remains a strict two-speaker span without an overlong effective
monologue.

An accepted window is written as 16 kHz mono 16-bit PCM WAV. All persisted time
values are integer milliseconds, and duration is always `end_ms - start_ms`.

## Configuration

Required environment variables:

```text
DATABASE_URL
CELERY_BROKER_URL
S3_BUCKET
S3_REGION
```

Optional environment variables:

```text
S3_ENDPOINT_URL
QUALITY_FILTER_CONFIG_FILE
MUSIC_MODEL_CACHE_DIR
```

Policy defaults live in
`src/voice_pipeline_quality_filter_audio_part/resources/default.toml`.
Deployment connection values remain environment-only. The optional policy file
may override model artifact SHA-256 values without changing source code.

The pinned Keras model, mean array, and standard-deviation array are bundled in
the Python package next to the music detector. `MUSIC_MODEL_CACHE_DIR` may point
to an external artifact directory when deployments need to override the bundled
files. The worker verifies all three SHA-256 checksums before loading the model.

The process automatically detects `x86_64` or `aarch64` and whether TensorFlow
has an available GPU. Unsupported architectures and invalid model artifacts fail
before the worker consumes messages.

## Run

```bash
uv sync --extra test --locked
uv run quality-filter-audio-part-worker
```

## Tests

```bash
uv run pytest
```

The default suite uses deterministic fake adapters and does not need model
weights or external services. Integration, real-model smoke, and capacity tests
are separately marked and have dedicated `run.sh` entry points under their test
directories. The integration runner uses the OSS `.env.test` service contract.
The smoke and capacity runners require a checksummed policy override and an
external model cache.

Production deployment must keep `--pool=solo`, concurrency `1`, and prefetch
multiplier `1`. Set the process-level task timeout only after recording the
900-second capacity report on each supported architecture/device combination;
the worker does not embed an unverified universal timeout.

## Operational limitations

A process lost after claim can leave a row in `filtering`. This worker never
infers staleness from timestamps. Recovery must be handled by a separately
reviewed operation.

After the completion transaction commits, the handler publishes one registered
`separate_chunk(chunk_id)` task per new chunk in `chunk_index` order. Publication
is best effort: an exception stops the remaining fan-out and is re-raised, but
does not roll back completed rows. A completed redelivery returns immediately
and does not enumerate or republish chunks.
