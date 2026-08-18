# Transcribe Chunk Worker

This single-process Celery worker transcribes the two separated tracks of an English chunk with Parakeet TDT. It consumes the persisted chunk-relative diarization snapshot and the separation result, uploads deterministic transcript artifacts, atomically advances the chunk to `transcribed`, and then publishes `persona_chunk` on a best-effort basis.

## Task contract

- Task and queue: `transcribe_chunk`
- Argument: one canonical chunk UUID string
- Accepted language: `en`
- Claim transition: `separated` or an ASR-owned `failed` row to `transcribing`
- Completion transition: `transcribing` to `transcribed`
- Failure transition: `transcribing` to `failed`, followed by re-raising the exception

The task does not process or dispatch Chinese chunks. A non-English direct invocation fails before audio download or model loading.

## Input and speaker identity

The worker downloads the fixed `speaker-0.wav` and `speaker-1.wav` objects recorded in `chunks.final_results.separation`. These suffixes are chunk-local output slots, not diarization speaker IDs and not identities that persist across chunks.

Each `speaker_audio` entry carries both `output_slot` and `diarization_speaker_id`. That persisted bijection is the only source of speaker identity. The worker preserves it in `transcript.json`, `word_alignment.json`, and `chunks.final_results.transcription`; it never renames or reorders the WAV objects based on the mapping.

The persisted diarization segments for the mapped speaker define the ASR slices. Segments separated by at most 2,000 ms are merged, and each merged interval is padded by 500 ms within chunk bounds. Both values are validated TOML settings. No additional VAD model is loaded.

An empty slice plan or a model response with no words produces canonical empty arrays and may still complete successfully. The worker does not perform a cross-track duplicate-text quality audit.

## Outputs

For a chunk whose audio object is `.../chunks/<index>/audio.wav`, the worker writes:

```text
.../chunks/<index>/results/transcript.json
.../chunks/<index>/results/word_alignment.json
```

Both objects use canonical UTF-8 JSON, chunk-relative integer millisecond timestamps, the output-slot-to-diarization-ID mapping, and pinned model provenance. The durable `final_results.transcription` namespace stores artifact URI, size, and SHA-256 metadata without duplicating transcript text.

Retries overwrite the same deterministic S3 keys. Uploaded objects are not deleted if the final database commit fails.

After the first successful database commit, the worker publishes one
`persona_chunk(chunk_id)` message. Publication failure is re-raised but does
not roll the committed `transcribed` state back. An `already_transcribed`
redelivery does not automatically republish the downstream task. A valid
transcription remains `already_transcribed` after the chunk advances to
`persona_generating`, `persona_generated`, `extending`, `completed`, or an
extension-owned `rejected` state.

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
HF_HOME
PARAKEET_CONFIG_FILE
```

`PARAKEET_CONFIG_FILE` defaults to the bundled `resources/default.toml`. The configuration pins the Hugging Face revision, NeMo artifact size and SHA-256, local-attention context, slicing thresholds, and utterance segmentation thresholds.

## Pinned runtime

- Parakeet revision: `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`
- NeMo artifact SHA-256: `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d`
- NeMo: `2.1.x`
- Torch and torchaudio: `2.9.x` from the CUDA 12.6 index
- Encoder attention: `rel_pos_local_attn` with context `[256, 256]`

The required x86_64 CUDA target was certified on an NVIDIA A10G. A cold model
load followed by an eight-second speech sample completed in 38.30 seconds and
returned finite word timestamps and confidence values. This is a functional
smoke result, not a throughput guarantee. The uv lock resolves Linux aarch64,
but aarch64 model execution remains best-effort until the same smoke test runs
on that architecture.

The same A10G completed the CUDA capacity test for a 120-second slice in 33.76
seconds including cold model load, with local attention active. This
representative input is not a chunk-duration limit.

## Run

From this directory:

```bash
uv sync --extra test
uv run pytest
uv run transcribe-chunk-worker
```

The repository-level `start-all.sh` starts this worker together with the other pipeline services.
