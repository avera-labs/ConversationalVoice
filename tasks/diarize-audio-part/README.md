# Audio Part Diarization Worker

This standalone Celery worker claims one `audio_parts` row, downloads its normalized WAV,
runs the configured DiariZen speaker-diarization model, uploads the canonical diarization
artifact and clean per-speaker reference audio, commits the `diarized` state, and publishes
`quality_filter_audio_part` only after the commit.

## Runtime contract

- Task and queue: `diarize_audio_part`
- Argument: one canonical `audio_part_id` UUID string
- Input: the absolute `s3://` URI stored in `audio_parts.audio_uri`
- Output keys: sibling `diarization.json` plus deterministic objects below
  `speaker-references/`
- Worker pool: `solo`, concurrency `1`, prefetch multiplier `1`
- Device policy: `auto` selects CUDA when available and otherwise allows CPU execution

Claims use one conditional update from `pending` or `failed` to `diarizing`. Duplicate
delivery decisions use only `audio_parts.status`: `diarized` recovers downstream publication,
`diarizing` and `filtering` are no-ops, and `completed` is already complete. A failed retry
runs the full inference again and overwrites the deterministic object.
For example, an input at
`s3://<bucket>/raw_audios/<raw_audio_id>/audio_parts/<part_index>/audio.wav`
produces
`s3://<bucket>/raw_audios/<raw_audio_id>/audio_parts/<part_index>/diarization.json` and
`s3://<bucket>/raw_audios/<raw_audio_id>/audio_parts/<part_index>/speaker-references/references.json`.

## Speaker reference contract

The worker derives speaker references from the same normalized turns written to
`diarization.json`; it never changes that artifact to carry reference metadata. A sweep-line
pass finds maximal intervals during which exactly one diarization speaker is active. Each
interval is trimmed by 500 milliseconds at both edges and retained only when more than 4,000
milliseconds remain.

Candidates are grouped by diarization speaker ID and sorted by effective duration descending,
then by start and end time. The worker collects from the longest candidate first and truncates
the final candidate when needed so each speaker contributes at most 30,000 milliseconds of
effective speech. Speakers with less than 4,000 milliseconds are omitted. Multiple selected
segments are assembled in selection order with exactly 500 milliseconds of digital silence
between them and no leading or trailing silence.

Reference WAVs are 16 kHz mono 16-bit PCM and preserve the source sample amplitudes. The
versioned `references.json` manifest records the diarization speaker ID, deterministic WAV URI,
byte size, SHA-256, selected audio-part-relative millisecond ranges, effective speech duration,
and total WAV duration including inserted silence. It contains `"speakers": []` when no speaker
qualifies. An empty manifest is a successful result and does not block quality filtering.

All reference artifacts exist only in S3. No reference URI or manifest content is stored in the
database. Consumers discover the manifest at the deterministic path above. Reference WAVs are
uploaded before `references.json`, and database completion happens only after the manifest is
uploaded. On retries, deterministic keys are overwritten; the manifest is the authoritative
list, so unlisted objects must be ignored.

The numeric suffix in `speaker-<id>.wav` is the DiariZen speaker ID within one audio part. It is
not a Sidon output slot and is not a stable identity across audio parts.

Each attempt that reaches Python cleanup emits exactly one `diarize_audio_part.finished`
structured log after temporary-workspace cleanup. It contains the canonical identifier,
effective model, actual device, diarization and reference speaker counts, and monotonic stage timings. It
does not contain artifact URIs, local paths, endpoints, credentials, or raw exceptions.

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
HF_TOKEN
DIARIZATION_CONFIG_FILE
```

The non-sensitive defaults are in
`src/voice_pipeline_diarize_audio_part/resources/default.toml`. They define the Hugging Face
model, device policy, and speaker-reference thresholds. The worker loads the
model through `DiariZenPipeline.from_pretrained(model)`, allowing the model repository to
provide its own configuration and embedding dependency. DiariZen and its pyannote fork are
both pinned to Git revision `844f5555b0a98acd0931511fc641a8c5b8ba92c7`.

Task workspaces are `TemporaryDirectory` instances under the operating-system temporary
directory with a dedicated prefix. Normal completion and exceptions clean them immediately.
Before the worker starts consuming messages, it removes only orphan directories with that
prefix and never follows symbolic links. Deploy the temporary directory on container-local
ephemeral storage or tmpfs so container replacement also removes crash remnants.

## Install and run

```bash
uv sync --extra test --locked
uv run diarize-audio-part-worker
```

The console entry point enforces a solo worker with concurrency one. The adapter resolves
`auto` before construction and makes DiariZen initialize directly on that device; it does not
move an initialized pipeline between devices. Model weights are fetched
through the standard Hugging Face cache and credential flow and are not included in this
repository or Python distribution.

## Tests

```bash
uv run pytest
tests/integration/run.sh
CERTIFICATION_TARGET=a100 MODEL_SMOKE_WAV=/path/to/short.wav \
  COMPATIBILITY_REPORT_PATH=/path/to/a100-smoke.json tests/smoke/run.sh
CERTIFICATION_TARGET=a100 CAPACITY_SOURCE_WAV=/path/to/speech.wav \
  CAPACITY_REPORT_PATH=/path/to/a100-capacity.json tests/capacity/run.sh
```

The default suite does not download model weights or require services. Integration tests use
a deterministic fake model with PostgreSQL, Redis, and S3-compatible storage. Real-model and
900-second capacity tests are opt-in and write a machine-readable compatibility report.

Release certification requires both of these CUDA targets:

| Target | Host architecture | Required validation |
| --- | --- | --- |
| NVIDIA A100 | `x86_64` | import, CUDA detection, model load, short WAV, 900-second capacity |
| NVIDIA GH200 | `aarch64` | CUDA-enabled ARM wheel, import, CUDA detection, model load, short WAV, 900-second capacity |

A local development probe on an NVIDIA A10G (`x86_64`, torch and torchaudio 2.6.0,
CUDA 12.4 runtime) completed model inference through `DiariZenPipeline.from_pretrained` on a 27.804-second
speech WAV with 10 turns in 6.001 seconds and 3,025,580,544 peak allocated VRAM bytes. This is diagnostic evidence,
not A100 or GH200 certification. The publication gate remains open until both required target
reports are produced.

Other CUDA GPUs and CPU execution are allowed but are not release-certification targets. The
lock routes `aarch64` PyTorch and torchaudio 2.6.0 to the official CUDA 12.6 wheel index; x86_64
uses the default package index.

## Operational limitation

A worker lost after claiming a row can leave it in `diarizing`. This task does not infer
staleness from timestamps and does not reclaim that state. An independently reviewed recovery
operation must move the row to `failed` before redelivery. A `SIGKILL` or OOM kill also cannot
emit the in-process terminal log; Celery worker-lost monitoring covers that path.
