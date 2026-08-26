# Reconstruct Chunk Worker

This single-process Celery worker performs source-faithful reconstruction of an
a two-speaker chunk carrying any canonical language identifier before dialogue extension.

## Contract

- Task and queue: `reconstruct_chunk`
- Argument: canonical chunk UUID string
- Claim: `persona_generated -> reconstructing`
- Completion: `reconstructing -> reconstructed`, then publish `extend_chunk`
  without a pipeline-wide language allowlist
- Durable result: `chunks.final_results.reconstruction`

For every utterance in the canonical chunk transcript, the worker slices the
mapped separated track and builds an audio reference from a fixed clean speaker
sample, exactly one second of PCM silence, and that separated utterance. It does
not call ASR. The existing transcript supplies immutable source text. MiMo
inserts approved tags at their audible positions in `text_with_audio_tags` and
produces one actor-facing `instruction`; removing the tags must reproduce the
source text exactly.

The default audio-tag model is `xiaomi/mimo-v2.5`; the default voice-cloning
model is `fish-audio/s2.1-pro`. These and the forced aligner are configured in
`resources/default.toml`.

Immediately after each isolated TTS segment is generated, the worker aligns it
against immutable plain `text` with the pinned
`Qwen/Qwen3-ForcedAligner-0.6B` model. This happens before source gaps,
overlaps, and track silence are restored, preventing those silent regions from
shifting model timestamps. Once the reconstruction schedule is known, each
segment-relative word timestamp is shifted by the utterance start. Inline tags
are restored from `text_with_audio_tags` as ordered zero-duration `audio_tag`
items at their exact plain-text offsets.

The request includes `response_format=json_schema` and appends the same JSON
Schema and complete tag allowlist to the system prompt. Invalid provider or
semantic output is retried at most twice with the failure reason. Audio-tag requests use
`provider.require_parameters=true` without a provider allowlist, so OpenRouter
automatically selects only endpoints that support the requested parameters,
including structured output. Provider fallbacks remain enabled. Runtime
validation remains authoritative as a final contract check.

MiMo reasoning is disabled by default for this bounded extraction request so
the completion budget is reserved for the JSON result. Invalid-response errors
report only categorical response shape, finish reason, and whether reasoning
was present; response content is never included.

Outputs are deterministic:

```text
.../results/reconstruction/
├── manifest.json
├── transcript.json
├── speaker-0.wav
└── speaker-1.wav
```

The reconstructed transcript records both source and generated timing. Source
gaps are retained, overlap onsets are adapted relative to generated anchor
durations, and a speaker is never scheduled over itself. Both output WAVs are
equal-duration 44.1 kHz mono PCM16 tracks. Every utterance also contains a
`word_alignment` array with assembled-track word timestamps and zero-duration
audio-tag positions.

## Code structure

- `task.py` owns Celery state transitions and workflow orchestration.
- `inputs.py` validates and loads upstream artifacts and speaker references.
- `reconstruction.py` slices utterances, builds references, synthesizes audio,
  and schedules generated segments.
- `outputs.py` builds and validates the durable manifest and task result.
- `openrouter.py` implements structured position-aware audio annotation.
- `fish_audio.py` implements reference-conditioned speech synthesis.
- `providers.py` preserves the original provider import surface for callers.

## Run

```bash
uv sync --extra test
uv run pytest
uv run reconstruct-chunk-worker
```

The default forced-aligner policy uses CUDA bfloat16 inference. Its model
revision, device, and dtype are configured in `resources/default.toml`.
