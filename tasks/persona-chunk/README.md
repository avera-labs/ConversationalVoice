# Persona Chunk Worker

This single-process Celery worker generates a structured vocal-persona document for an English chunk. It validates the completed separation and transcription contracts, converts the original mixed chunk WAV to a compact MP3, and sends the MP3 plus a deterministic speaker-labelled SRT transcript to OpenRouter.

## Task contract

- Task and queue: `persona_chunk`
- Argument: one canonical chunk UUID string
- Claim transition: `transcribed`, or a persona-owned `failed` row, to `persona_generating`
- Completion transition: `persona_generating` to `persona_generated`, followed by publishing `extend_chunk`
- Failure transition: `persona_generating` to `failed`, followed by re-raising
- A successor-publication failure transitions `persona_generated` to `failed` while retaining the durable persona
- `persona_generating` is an in-progress no-op; `extending`, `completed`, and an extension-owned `rejected` state validate the durable persona and return without republishing

Stale in-progress rows require explicit operator recovery. A retry with an already durable persona validates it and republishes `extend_chunk` without rerunning OpenRouter.

## Inputs

The worker consumes the original mixed `chunks.audio_uri` WAV and the canonical `results/transcript.json` artifact. It does not download `word_alignment.json`, separated tracks, or rerun any upstream model. Before provider I/O, it validates the chunk-relative diarization snapshot, the complete separation and transcription result namespaces, input byte identities, WAV format, language, and speaker mapping.

The original mixed audio preserves scene context, overlap, tone, and interaction. Transcript cues identify speakers as `[Speaker <diarization_speaker_id>]`; output-slot filenames and array positions are never treated as speaker identity.

## OpenRouter

OpenRouter is the only provider. The default model is `xiaomi/mimo-v2.5`, configured in `resources/default.toml`. The request contains MP3 `input_audio`, transcript text, and a strict `response_format=json_schema` structured-output contract. The exact JSON Schema is also embedded in the system prompt together with the field semantics.

Transport timeouts, network failures, HTTP 408/409/429/5xx, and malformed JSON receive the bounded retry policy from TOML. Deterministic 4xx responses are not retried. The worker trusts OpenRouter's strict structured-output enforcement and does not run a second schema validator over the model response. Artifact identity metadata is validated before database completion, while an `already_completed` invocation validates the stored durable document without contacting OpenRouter. The application never changes the configured model or backend; OpenRouter may route among eligible endpoints serving the same model ID.

Logs and persisted errors exclude credentials, authorization headers, audio base64, transcript text, prompts, raw responses, and persona text.

## Outputs and compatibility

The worker uploads canonical UTF-8 JSON to:

```text
.../chunks/<index>/results/persona.json
```

The exact same document is stored in `chunks.persona`. It preserves the established top-level `scene`, `speakers`, and `usage` objects and all of their fields. Each persona speaker retains a string `speaker_id`, and the list is sorted lexicographically by that string. New metadata is confined to additional top-level fields: `schema_version`, `backend`, `config_version`, `language`, and `speaker_mapping`.

`speaker_mapping` contains two entries in output-slot order and is the complete mapping between fixed separation slots and integer diarization speaker IDs. Persisting it never renames or reorders the S3 separation objects.

`chunks.final_results.persona` stores the OpenRouter model ID and the URI, byte size, and SHA-256 identities of the input WAV, input transcript, and persona artifact. Completion writes the persona column, result namespace, `persona_generated` status, and cleared error in one transaction before successor publication.

## Configuration

Required environment variables:

```text
DATABASE_URL
CELERY_BROKER_URL
S3_BUCKET
S3_REGION
OPENROUTER_API_KEY
```

Optional variables are `S3_ENDPOINT_URL` and `PERSONA_CONFIG_FILE`. The provider endpoint is fixed to the official OpenRouter API and cannot be redirected through configuration. The model is configured only through TOML.

The host must provide `ffmpeg`. The worker requires no GPU. Both Linux x86_64 and aarch64 dependency environments are locked; actual ffmpeg and network availability remain host responsibilities.

## Run

```bash
uv sync --extra test
uv run pytest
uv run persona-chunk-worker
```

The repository-level `start-all.sh` starts this worker with the rest of the pipeline.
