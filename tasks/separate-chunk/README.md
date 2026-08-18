# Separate chunk worker

This single-process Celery worker separates one existing chunk WAV into two
fixed output slots with DialogueSidon. It consumes only a canonical `chunk_id`.

## Output track identity

The numeric suffix in `speaker-0.wav` and `speaker-1.wav` identifies the
canonical, chunk-local separation output slot. It is not a diarization speaker
ID and does not identify the same person across chunks.

After the consistency audit passes, each `speaker_audio` entry records the
required `diarization_speaker_id` for its `output_slot`. The two entries form a
complete bijection with the IDs in `chunks.diarizations`. Consumers must use
this mapping; it never renames or reorders storage objects.

After the first successful separation commit, an English chunk is published to
the registered `transcribe_chunk` queue. Other languages remain separated. A
publication exception is re-raised without rolling back the committed
separation result.

An invocation that finds an existing separation result validates the complete
diarization snapshot, exact result schema, canonical output URIs, and persisted
slot-to-speaker mapping before returning `already_separated`, including when the
chunk has advanced to `persona_generating`, `persona_generated`, `extending`,
`completed`, or an extension-owned `rejected` state. This validation does
not download audio or load either model. A missing or partial result under a
downstream status is a contract failure, not a successful no-op.

## Runtime

Required environment variables are `DATABASE_URL`, `CELERY_BROKER_URL`,
`S3_BUCKET`, and `S3_REGION`. `S3_ENDPOINT_URL`, `HF_TOKEN`, `HF_HOME`, and
`SIDON_CONFIG_FILE` are optional. Start with `uv run separate-chunk-worker`.

The supported release target is CUDA on x86_64. The dependency set avoids
architecture-specific application code and is expected to resolve on aarch64,
but aarch64 remains best-effort until it passes the same model smoke test.

## Pinned runtime

- DialogueSidon revision: `d43d7478402a5527136c6733c3f4359c37b312ab`.
- WavLM alignment revision: `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`.
- Torch and torchaudio: `2.9.x` from the CUDA 12.6 index.
- Transformers: `4.57.6`; Diffusers: `0.37.x`.

The loader checks the byte size and SHA-256 of every DialogueSidon artifact
before `torch.export` loading. Alignment uses the source-compatible WavLM
similarity and margin thresholds (`0.86` and `0.01`) with an RMS voice-presence
gate. A successful audit requires usable evidence for both diarization speakers.
The default maximum window, overlap, and crossfade are the source-compatible
120 seconds, 10 seconds, and 5 seconds; all are validated TOML policy values.

The x86_64 certification run used an NVIDIA A10G. A 20-second input with 100
diffusion steps produced two finite 24 kHz tracks in 16.82 seconds. This is a
functional smoke result, not a throughput guarantee.

Run `tests/capacity/run.sh` to execute the pinned model at the maximum
120-second inference-window size on CUDA. The NVIDIA A10G certification run
completed this case in 37.46 seconds and produced two finite 24 kHz tracks.

Worker loss may leave a row in `separating`. Redelivery is a no-op in that
state; recovery requires confirming the old worker has stopped, changing the
row to `failed`, and explicitly republishing the task.
