# Voice Pipeline Models

Shared SQLAlchemy 2.x models for the Voice Pipeline PostgreSQL database.

## Contract

The package maps the tables defined by
[`schema/schema.sql`](../../schema/schema.sql):

- `raw_audios`
- `audio_parts`
- `chunks`

The SQL schema is authoritative. This package mirrors its columns, database
defaults, constraints, indexes, foreign keys, and delete behavior. Schema
changes must be made in `schema/schema.sql` first and then reflected here.

The package intentionally does not create database engines, sessions, or
application configuration. Consumers remain responsible for connection and
transaction lifecycle management.

## Table relationships

```text
raw_audios 1 ──< audio_parts 1 ──< chunks
```

- A `raw_audios` row represents one ingested podcast audio artifact.
- An `audio_parts` row represents one VAD-selected conversation window. Its
  `raw_audio_id` references `raw_audios.id`, and `(raw_audio_id, part_index)`
  is unique.
- A `chunks` row represents one accepted two-speaker segment. Its
  `audio_part_id` references `audio_parts.id`, and
  `(audio_part_id, chunk_index)` is unique.
- Deleting a `raw_audios` row cascades to all of its `audio_parts` and
  `chunks`. Deleting an `audio_parts` row cascades to its `chunks`.
- `audio_parts.relative_start_ms` and `relative_end_ms` are relative to the
  raw audio. The equivalent fields on `chunks` are relative to the containing
  audio part.

## Raw audio status contract

`raw_audios.status` accepts these values:

| Status | Meaning |
| --- | --- |
| `pending` | Ingest completed and split processing may be claimed. |
| `splitting` | One split worker atomically claimed the row. |
| `split_completed` | VAD artifacts and all expected audio-part rows were persisted. |
| `failed` | Ingest, split processing, or downstream task publication failed. |

The split task owns these transitions:

```text
pending         -> splitting
failed          -> splitting
splitting       -> split_completed
splitting       -> failed
split_completed -> failed  # downstream publication failure only
```

Ingest may transition `pending -> failed` when split-task publication fails.
A split worker must claim a row with one conditional database update; reading
the status and updating it in separate operations is not a concurrency guard.
The split task creates every `audio_parts` row with `pending` status.

## Audio part status contract

`audio_parts.status` accepts these values:

| Status | Meaning |
| --- | --- |
| `pending` | The part is ready for diarization. |
| `diarizing` | A diarization worker atomically claimed the part. |
| `diarized` | The durable diarization artifact and URI were committed. |
| `filtering` | A quality-filter worker atomically claimed the part. |
| `completed` | Quality filtering completed successfully. |
| `failed` | The current stage or downstream task publication failed. |

The approved transitions are:

```text
pending    -> diarizing
failed     -> diarizing or filtering
diarizing  -> diarized or failed
diarized   -> filtering or failed
filtering  -> completed or failed
```

Retry behavior is selected only from `status`; `diarization_uri` does not
determine the retry branch. A diarization retry claims `failed`, clears any old
URI, and reruns the full deterministic operation.

Chunks use `pending`, `separating`, `separated`, `transcribing`, `transcribed`,
`persona_generating`, `persona_generated`, `extending`, `completed`, `rejected`,
and `failed`. A known separation or extension quality rejection is terminal. A
failed chunk without a separation namespace may retry separation; a failed
chunk with valid separation and no transcription namespace may retry
transcription; a failed chunk with valid separation and transcription
namespaces but no persona may retry persona; a failed chunk with a valid
persona and no dialogue-extension namespace may retry extension. Stale
in-progress rows require explicit operator recovery. The database intentionally
does not constrain chunk status values; task repositories own transition
validation, while `schema/schema.sql` documents the shared vocabulary.

## S3 object layout

Ingest and split artifacts remain grouped by `raw_audio_id`:

```text
s3://<bucket>/raw_audios/<raw_audio_id>/
├── audio.wav
├── vad_segments.json
└── audio_parts/
    └── <part_index>/
        ├── audio.wav
        ├── diarization.json
        ├── speaker-references/
        │   ├── references.json
        │   └── speaker-<diarization-speaker-id>.wav
        └── chunks/
            └── <chunk_index>/
                ├── audio.wav
                └── results/
                    ├── separated/
                    │   ├── speaker-0.wav
                    │   └── speaker-1.wav
                    ├── transcript.json
                    ├── word_alignment.json
                    ├── persona.json
                    └── dialogue-extension/
                        ├── script.json
                        ├── transcript.json
                        ├── speaker-0.wav
                        └── speaker-1.wav
```

The `speaker-0.wav` and `speaker-1.wav` filenames identify fixed, chunk-local separation output slots. Their numeric suffixes are not diarization speaker IDs and do not identify the same person across chunks.

After the track-consistency audit succeeds, the two entries in `chunks.final_results.separation.speaker_audio` persist the complete mapping through `output_slot` and `diarization_speaker_id`. Consumers must read this mapping rather than infer identity from an object key. Persisting the mapping never renames or reorders the two S3 objects.

The diarization task derives its deterministic object key from the persisted
`audio_parts.audio_uri`. It replaces the input object's filename with
`diarization.json`, preserving the configured bucket and parent directory.

The same task derives `speaker-references/references.json` and its listed WAVs
from the audio-part parent directory. These reference objects are not stored in
the database. The manifest is always written, including an empty speaker list,
and is the authoritative inventory for the reference WAVs. A reference filename
uses an audio-part-local DiariZen speaker ID; it is not a chunk-local separation
slot and does not establish identity across audio parts.

- `vad_segments.json` contains normalized speech intervals produced by
  `pyannote/segmentation-3.0`.
- `transcript.json` contains the utterance-level transcription.
- `word_alignment.json` contains the word-level transcription.
- `<bucket>` is configuration, never a hard-coded value.
- `<raw_audio_id>` and `<audio_part_id>` are canonical UUID strings.
- `<part_index>` and `<chunk_index>` are zero-based decimal integers matching
  their database columns.

Every URI stored in the three tables is an absolute `s3://` URI.
`audio_parts.diarization_uri` stores the approved sibling diarization URI.
The quality-filter task owns `chunks/<chunk_index>/audio.wav` and persists its
URI together with the accepted chunk row.
Result artifact URIs are stored in `chunks.final_results`; structured
diarization and persona data may also be stored in `chunks.diarizations` and
`chunks.persona`. Object keys are deterministic so retries converge on the
same object instead of creating duplicates.

`chunks.final_results.transcription` preserves the same `output_slot` to
`diarization_speaker_id` mapping as separation and binds both JSON artifacts by
canonical URI, byte size, and SHA-256. Transcript timestamps are relative to
the chunk.

`persona.json` and `chunks.persona` preserve the public `scene`, `speakers`,
and `usage` structure. Persona speakers retain their string `speaker_id` and
are sorted by that string. The additional top-level `speaker_mapping` records
the complete chunk-local `output_slot` to integer `diarization_speaker_id`
bijection; consumers must not infer it from array order or filenames.

`chunks.final_results.persona` binds the persona artifact to the original
chunk WAV and `transcript.json` by canonical URI, byte size, and SHA-256.

The `dialogue-extension` directory contains only generated continuation audio;
it never includes or concatenates the original chunk. Its `speaker-0.wav` and
`speaker-1.wav` are equal-duration tracks on an extension-relative timebase.
Their fixed slot IDs preserve the exact separation mapping to diarization
speaker IDs. `script.json` contains the continuation lines, type, delivery
tone, placement, and audio tags; the sibling `transcript.json` adds actual TTS
timings. `chunks.final_results.dialogue_extension` binds all inputs, mapped
speaker references, model identities, and output artifact identities.
For each output slot, dialogue extension prefers the audio-part reference
listed for its mapped diarization speaker. If that entry is absent, it uses the
longest chunk-relative interval where only that speaker is active and slices
the corresponding separation track after trimming 500 milliseconds from both
interval edges. The result records the selected source, source identity,
timebase, interval, and exact reference-audio identity; the temporary fallback
slice is not stored as another object.
Both reference transcription and Fish Audio synthesis are requested through
OpenRouter with `fish-audio/transcribe-1` and `fish-audio/s2.1-pro`; the worker
does not require a provider-specific Fish Audio credential.

## Development

```bash
uv sync
uv run pytest
```

The package requires PostgreSQL-compatible SQLAlchemy types, but its contract
tests do not require a running database.
