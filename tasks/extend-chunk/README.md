# Extend Chunk Worker

This single-process Celery worker generates and synthesizes an English (`en`) or
Chinese (`zh`) dialogue continuation for a completed chunk persona. It creates
extension-only artifacts; it does not copy or concatenate the original chunk
audio.

## Task contract

- Task and queue: `extend_chunk`
- Argument: one canonical chunk UUID string
- Claim transition: `reconstructed`, or an extension-owned `failed` row with a durable reconstruction result, to `extending`
- Completion transition: `extending` to `completed`
- Known quality rejection: `extending` to `rejected` when a mapped speaker has neither a DiariZen reference nor a usable non-overlapped interval in the current chunk
- Technical failure: `extending` to `failed`, followed by re-raising
- `extending`, `rejected`, and `completed` direct invocations are no-ops after their applicable durable contracts are checked

Stale in-progress rows require explicit operator recovery. `rejected` and `completed` are terminal.

## Inputs and identity

The worker validates the chunk-relative diarization snapshot and the complete separation, transcription, persona, and reconstruction result namespaces. It downloads both the canonical chunk transcript and the paired reconstruction transcript. A lightweight timing-only pass derives reconstruction turns/min, backchannels/min, and overlap-events/min and converts them into feasible continuation counts. Cross-speaker transcript overlaps use the same 60 ms minimum as the evaluation protocol. The overlap target is scaled to the complete planned continuation duration and expressed to the dialogue model both as an exact feasible event count and as average temporal and turn spacing. The worker also downloads the audio-part speaker-reference manifest. For each mapped speaker, it prefers the reference WAV listed in that manifest. If the manifest has no entry for that speaker, the worker downloads the corresponding Sidon separation track and finds the longest interval in which only that diarization speaker is active. It trims 500 milliseconds from both edges, resolves equal-duration candidates by earliest start time, and caps the retained interval at 30 seconds. The fallback slice is temporary and is not uploaded as another object.

Extension `speaker_id` values are fixed output slots `0` and `1`. They retain the exact separation mapping to `diarization_speaker_id`; neither LLM generation nor synthesis may relabel the speakers. This makes future concatenation with the original separated tracks possible without making concatenation part of this task.

## Dialogue generation

OpenRouter is the only dialogue provider. The default model is `google/gemini-3.7-flash`. The request contains the current persona, canonical transcript, and reconstruction-derived interaction targets and asks only for the continuation. It uses strict structured output with Gemini-compatible JSON Schema keywords, embeds the schema and complete audio-tag allowlist in the system prompt, and enables OpenRouter response healing for non-streaming JSON repair. The model produces `instruction` and position-preserving `text_with_audio_tags`, but never `text`; the worker validates the inline tags and derives plain `text` before synthesis. The prompt asks the model to match the duration-scaled interaction targets and distribute overlap events across the continuation. Invalid provider or semantic output is retried with a safe reason code and correction requirement.

The configured target duration defaults to 120 seconds. It is a generation target rather than a hard audio limit because final duration depends on TTS prosody. Each utterance is classified as one of:

- `dialogue`: ordinary spoken dialogue; a short cooperative completion, correction, or interruption may overlap a substantive anchor;
- `backchannel`: a brief acknowledgement that always overlaps and is normalized locally to a conventional evaluator-recognized cue;
- `paralinguistic`: laughter, crying, breathing, coughing, or another approved non-lexical event, optionally overlapping the previous speaker.

Audio tags are embedded at their audible positions in `text_with_audio_tags`. One or two adjacent tags are allowed at a position; three are rejected. The durable plain `text` is derived by the worker, while `instruction` stores one concise actor-facing performance direction.

## Voice cloning and timeline

OpenRouter is the only external API used for speech. The worker transcribes each selected reference with `fish-audio/transcribe-1` through OpenRouter's transcription endpoint, then invokes the configured `fish-audio/s2.1-pro` model through OpenRouter's speech endpoint. The reference audio and transcript are supplied as stateless `input_references` for every TTS request associated with that output slot. No remote voice resource is created or persisted.

The task schedules sequential utterances with the configured turn gap. An overlap may follow only a sequential, substantive dialogue anchor and cannot occur consecutively. Its start is constrained to a late, semantically safe point in the anchor, and the assembled tracks must retain measurable overlap. Backchannels are fitted wholly inside the anchor tail when possible, so the listener yields the floor; other brief interruptions use the configured overlap window. It renders two equal-duration, 44.1 kHz mono PCM16 WAVs, with silence everywhere that the corresponding speaker is inactive. Transcript timestamps are relative to the start of the extension, so a future concatenation step can resample to its target format and apply its own offset and gap policy.

Before scheduling or adding silence, every isolated TTS WAV is forced-aligned
against its worker-derived plain `text` with the pinned
`Qwen/Qwen3-ForcedAligner-0.6B` model. The segment-relative word timestamps are
shifted by the final utterance start when tracks are assembled. Audio tags are
reinserted from `text_with_audio_tags` at their exact plain-text offsets as
`audio_tag` alignment items with `start_ms == end_ms`, so tags preserve order
without consuming timeline duration.

## Outputs

All output keys are deterministic:

```text
.../chunks/<index>/results/dialogue-extension/
├── script.json
├── transcript.json
├── speaker-0.wav
└── speaker-1.wav
```

`script.json` preserves the model-produced continuation, utterance type, placement, `instruction`, and `text_with_audio_tags`, plus worker-derived `text`. `transcript.json` adds actual synthesized start and end times and a `word_alignment` array to every utterance. Each array contains ordered `word` and zero-duration `audio_tag` items with global audio timestamps and plain-text character offsets. The two WAV filenames remain fixed output slots and never use diarization speaker IDs.

`chunks.final_results.dialogue_extension` stores the model identities, target and actual duration, input artifact identities, reference-audio mapping, and URI/size/SHA-256 identities for all four outputs. Each speaker reference records whether it came from the DiariZen reference set or a separated-track slice, the source object identity, the source timebase and selected intervals, the exact temporary audio identity sent to Fish Audio, and the reference transcript hash. Completion commits this namespace and `status = 'completed'` atomically after every artifact is uploaded and validated.

## Configuration

Required environment variables:

```text
DATABASE_URL
CELERY_BROKER_URL
S3_BUCKET
S3_REGION
OPENROUTER_API_KEY
```

Optional variables are `S3_ENDPOINT_URL` and `EXTEND_CHUNK_CONFIG_FILE`. OpenRouter endpoints are fixed to the official API and cannot be redirected through configuration. The dialogue, reference-transcription, TTS, and forced-alignment models, duration target, synthesis controls, retry limits, and timeline gaps are configured in `resources/default.toml`.

The default forced-alignment policy requires a CUDA GPU and uses bfloat16. Both Linux x86_64 and aarch64 dependency environments are locked.

## Run

```bash
uv sync --extra test
uv run pytest
uv run extend-chunk-worker
```

The repository-level `start-all.sh` starts this worker with the rest of the pipeline.
