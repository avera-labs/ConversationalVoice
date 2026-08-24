# Transcribe Chunk ZH Worker

This single-process Celery worker transcribes separated Chinese (`zh`) speaker
tracks with the pinned ModelScope Paraformer model, restores punctuation with
CT-PUNC, writes per-character `word_alignment.json`, derives
`transcript.json` from that alignment, commits `transcribed`, and publishes
`persona_chunk`.

## Contract

- Task and queue: `transcribe_chunk_zh`
- Argument: one chunk UUID string
- Accepted language: `zh`
- Output backend: `paraformer_zh`
- Timebase: chunk-relative integer milliseconds

Every Chinese character is one `words[]` entry. Inserted punctuation is
attached to the preceding character and never receives an independent
timestamp. Both output documents otherwise use the same schema, speaker
mapping, deterministic S3 keys, and canonical JSON encoding as the English
worker.

The worker captures the selected token posterior from the pinned FunASR
decoder and stores it as the character confidence. Missing decoder scores or
any text/timestamp/confidence count mismatch fails the task instead of writing
fabricated confidence values or truncated output.
Punctuation inference is split into bounded character batches using the
reviewed `punctuation.max_chars` policy so large chunks do not create an
unbounded CT-PUNC request.

## Configuration

Required environment variables are `DATABASE_URL`, `CELERY_BROKER_URL`,
`S3_BUCKET`, and `S3_REGION`. Optional variables are `S3_ENDPOINT_URL`,
`PARAFORMER_ZH_CONFIG_FILE`, `PARAFORMER_MODEL_DIR`, and `CTPUNC_MODEL_DIR`.
Local model directories are recommended for production so workers never need
network downloads while serving tasks.

The reviewed defaults resolve to the ModelScope repositories declared in
`resources/default.toml`. The dependency lock pins FunASR and Transformers;
update the adapter tests whenever either runtime is deliberately upgraded.

## Run

```bash
uv sync --extra test
uv run pytest
uv run transcribe-chunk-zh-worker
```
