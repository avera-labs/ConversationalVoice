# Voice Pipeline OSS Project Specification

## Goal

Build a clean, reproducible, deployment-friendly open-source pipeline for
turning podcast audio into high-quality, structured two-speaker dialogue
artifacts.

This project is self-contained and independently understandable, buildable,
testable, and deployable.

## System boundary

The complete data lineage is:

```text
raw_audios (one ingested podcast)
  -> audio_parts (VAD-selected conversation windows)
    -> chunks (accepted clean two-speaker segments)
```

`schema/schema.sql` is the sole persistence contract.

## Processing graph

```text
POST audio + metadata
  -> ingest
       transcode to 16 kHz mono WAV
       upload WAV
       create raw_audios
  -> split-raw-audio-into-parts
       detect conversation windows with voice activity detection
       upload window audio
       create audio_parts
  -> diarization
       run speaker diarization per audio part
       upload speaker-label JSON
       upload clean per-speaker reference WAVs and their S3-only manifest
  -> quality-filter
       filter speech by SNR and music overlap
       plan strict two-speaker spans
       create one chunks row per accepted span
  -> processing (per chunk)
       two-speaker source separation and speaker-mapping audit
       -> speech transcription
       -> LLM persona extraction
       -> LLM dialogue continuation
       -> voice-cloning speech synthesis into two speaker tracks
       -> upload artifacts and write chunks.final_results
```

Workers explicitly publish registered successor tasks through Celery with
Redis as broker. Split and quality filtering fan out; later chunk stages are
one-to-one.

## Celery task registry

`packages/task-contracts/src/voice_pipeline_task_contracts/tasks.py` is the
single registry for task names, dedicated queues, and UUID arguments. Every
current queue has the same name as its task.

| Task | UUID argument | Successor |
| --- | --- | --- |
| `split_raw_audio_into_parts` | `raw_audio_id` | `diarize_audio_part` per part |
| `diarize_audio_part` | `audio_part_id` | `quality_filter_audio_part` |
| `quality_filter_audio_part` | `audio_part_id` | `separate_chunk` per chunk |
| `separate_chunk` | `chunk_id` | `transcribe_chunk` |
| `transcribe_chunk` | `chunk_id` | `persona_chunk` |
| `persona_chunk` | `chunk_id` | `extend_chunk` |
| `extend_chunk` | `chunk_id` | terminal |

## Persistence ownership

| Step | Primary row | Required persistence effect |
| --- | --- | --- |
| ingest | `raw_audios` | Create the source record with normalized audio URI and source metadata. |
| split | `raw_audios`, then new `audio_parts` | Persist VAD output and create indexed window rows with source-relative timing. |
| diarization | `audio_parts` | Persist `diarization_uri`; speaker-reference WAVs and their manifest remain S3-only. |
| quality-filter | `audio_parts`, then new `chunks` | Track filtering and create one indexed chunk row for every accepted span. |
| separation | `chunks` | Persist the chunk diarization snapshot, fixed speaker mapping, and `final_results.separation`. |
| transcription | `chunks` | Persist transcript artifact identities in `final_results.transcription`. |
| persona | `chunks` | Persist the persona document and `final_results.persona`. |
| extension | `chunks` | Persist `final_results.dialogue_extension` and complete the chunk. |

Workers claim eligible rows transactionally before expensive work. Technical
errors use `failed`; separation and extension may instead use terminal
`rejected` quality outcomes. Stale in-progress rows require explicit operator
recovery.

Tasks must be idempotent and retry-safe. Database uniqueness constraints,
content identities, and deterministic object keys should make repeated
delivery converge on the same state.

## Dialogue extension stage

`EXTEND_CHUNK` is the final per-chunk task. It uses the completed transcript,
persona, and speaker references to generate a dialogue continuation and
synthesize two speaker tracks. Output speaker slots preserve the separation
mapping. A missing diarization reference falls back to a clean interval from
the matching separated track; no usable reference from either path is a
quality rejection.

It writes deterministic `script.json`, `transcript.json`, `speaker-0.wav`, and
`speaker-1.wav` extension artifacts without concatenating the source chunk.
Their identities and generation metadata are committed in
`chunks.final_results.dialogue_extension` with status `completed`.

## Project layout principles

- One independent Celery task equals one independent uv project.
- Every task project owns a `pyproject.toml` and `uv.lock`.
- Each task is deployable as a single-process worker with its own dependency
  set and queue.
- Model-heavy dependencies stay isolated in the task that needs them.
- Shared packages are limited to SQLAlchemy models, task routing and publishing,
  and stable cross-task artifact contracts. Runtime integrations remain local
  to each service or task.
- Celery messages carry IDs, small metadata, and URIs—not audio bytes, model
  outputs, or other large payloads.

## Security and publication gate

No OSS file may contain:

- credentials, tokens, or secret values;
- internal domains or endpoints;
- hard-coded bucket names;
- company identifiers in new source/configuration/package content;
- repository-external project history, source material, or operational
  assumptions;
- non-English documentation, comments, docstrings, examples, configuration
  comments, logs, errors, fixtures, or user-visible text.

Use environment variables and harmless placeholders in example configuration.
Treat local `.env` files as sensitive and never use them as source material
for documentation.

## Review gates

Implementation proceeds one user-reviewed step at a time. Each review includes:

1. exact scope and contract;
2. files added or changed;
3. tests and verification results;
4. schema/status implications;
5. security/publication scan;
6. explicit unresolved decisions.

No subsequent step begins before the current step is approved.

## Status and retry contract

The schema constrains raw-audio and audio-part status values:

```text
raw_audios: pending/failed -> splitting -> split_completed/failed
audio_parts: pending/failed -> diarizing -> diarized/failed
             diarized/failed -> filtering -> completed/failed
```

Chunk status values are intentionally enforced by task repositories rather
than a database check constraint:

```text
pending/failed           -> separating          -> separated/rejected/failed
separated/failed         -> transcribing        -> transcribed/failed
transcribed/failed       -> persona_generating  -> persona_generated/failed
persona_generated/failed -> extending           -> completed/rejected/failed
```

For a failed chunk, durable `final_results` namespaces determine the retrying
stage. `completed` and `rejected` are terminal, while stale in-progress states
require operator recovery. Successor publication occurs after durable stage
completion; publication failures do not remove committed artifacts. Depending
on the producing stage, they may transition `pending`, `split_completed`,
`diarized`, or `persona_generated` to `failed`.

The dependency-free `packages/task-contracts` package owns routing contracts.
`packages/task-client` owns producer-side UUID serialization, bounded retry,
publish confirmation, readiness checks, and safe publication errors.
