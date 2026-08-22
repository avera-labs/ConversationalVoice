# Reconstruct Chunk Worker

This single-process Celery worker performs source-faithful reconstruction of a
two-speaker chunk before dialogue extension.

## Contract

- Task and queue: `reconstruct_chunk`
- Argument: canonical chunk UUID string
- Claim: `persona_generated -> reconstructing`
- Completion: `reconstructing -> reconstructed`, then publish `extend_chunk`
- Durable result: `chunks.final_results.reconstruction`

For every utterance in the canonical chunk transcript, the worker slices the
mapped separated track and builds an audio reference from a fixed clean speaker
sample, exactly one second of PCM silence, and that separated utterance. It does
not call ASR. The existing transcript supplies all synthesis text.

The default audio-tag model is `xiaomi/mimo-v2.5`; the default voice-cloning
model is `fish-audio/s2.1-pro`. Both are configured in
`resources/default.toml`.

The request includes `response_format=json_schema` and appends the same JSON
Schema to the system prompt. Audio-tag requests use
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
equal-duration 44.1 kHz mono PCM16 tracks.

## Code structure

- `task.py` owns Celery state transitions and workflow orchestration.
- `inputs.py` validates and loads upstream artifacts and speaker references.
- `reconstruction.py` slices utterances, builds references, synthesizes audio,
  and schedules generated segments.
- `outputs.py` builds and validates the durable manifest and task result.
- `openrouter.py` implements structured audio-tag extraction.
- `fish_audio.py` implements reference-conditioned speech synthesis.
- `providers.py` preserves the original provider import surface for callers.

## Run

```bash
uv sync --extra test
uv run pytest
uv run reconstruct-chunk-worker
```
