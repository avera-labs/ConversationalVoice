<img src="assets/averalabs-logo.png" alt="AveraLabs" height="28" />

# Conversational Voice

**Full-Duplex Speech Data from Real Conversations**

Conversational Voice converts a broad range of real-world recordings into
curated training data for full-duplex speech models. The pipeline identifies
two-speaker exchanges, separates the speakers into individual tracks, validates
speaker consistency, annotates timing and speaker attributes, and uses each
exchange as the basis for new dialogue while preserving its turn-taking, pauses,
overlap, and paralinguistic cues.

**[Explore the project page, pipeline, and method comparisons →][project-page]**

## What it produces

Each conversation that passes screening and quality control yields two datasets:
validated speaker-separated tracks from the source recording and new dialogue
derived from the same exchange.

- **Speaker-separated source data:** a clean, time-aligned track for each
  speaker, extracted from the source recording and validated for speaker
  consistency.
- **Conversation-grounded expansion data:** controllable new dialogue that
  retains the context and interaction patterns of the source exchange.

Large payloads stay in object storage. Celery messages contain only stable UUIDs,
and PostgreSQL records the lineage and state of every source, audio part, and
accepted chunk.

## Architecture

[![Conversational Voice Pipeline architecture](assets/pipeline-architecture.png)][project-page]

Every stage is an independently installable worker with its own locked
dependencies and dedicated queue. A worker claims eligible database state,
performs its work, writes deterministic artifacts, commits the result, and only
then publishes the registered successor task. Split and quality-filter stages
fan out; later chunk stages are one-to-one.

The persistent data lineage is:

```text
raw_audios (one normalized source recording)
  └── audio_parts (VAD-selected conversation windows)
        └── chunks (accepted clean two-speaker dialogue segments)
```

## Requirements

- [`uv`](https://docs.astral.sh/uv/)
- FFmpeg and FFprobe on `PATH`
- PostgreSQL
- Redis
- AWS S3 or an S3-compatible object store
- A Hugging Face token with access to the configured VAD, diarization,
  separation, and transcription models
- An OpenRouter API key for speaker-profile extraction, dialogue expansion,
  reference transcription, and speech synthesis
- An NVIDIA CUDA host for the supported separation and transcription runtime;

## Basic usage

### 1. Clone the repository

```bash
git clone https://github.com/avera-labs/ConversationalVoice.git
cd ConversationalVoice
```

### 2. Prepare infrastructure

Create a PostgreSQL database, a Redis database, and an object-storage bucket.
Apply the authoritative schema with a PostgreSQL-compatible connection URI:

```bash
psql "postgresql://pipeline:password@localhost:5432/voice_pipeline" \
  -f schema/schema.sql
```

This repository intentionally does not prescribe a database, broker, or object
store deployment. The configured resources must exist before the services
start.

### 3. Configure the environment

Create `.env` in the repository root. The startup script links this file into
each independent service and task project.

```dotenv
DATABASE_URL=postgresql+psycopg://pipeline:password@localhost:5432/voice_pipeline
CELERY_BROKER_URL=redis://localhost:6379/0

S3_BUCKET=voice-pipeline
S3_REGION=us-east-1
# Set this only for an S3-compatible service such as MinIO.
# S3_ENDPOINT_URL=http://localhost:9000

# Optional when the standard AWS SDK credential chain already provides access.
# AWS_ACCESS_KEY_ID=replace-me
# AWS_SECRET_ACCESS_KEY=replace-me

HF_TOKEN=replace-me
OPENROUTER_API_KEY=replace-me
```

Do not commit `.env`. Policy and model settings live in each project's packaged
`resources/default.toml`; the task-specific READMEs document reviewed override
files when customization is required.

### 4. Start the API and every worker

```bash
./start-all.sh
```

The script verifies the task registry, synchronizes every uv project from its
lock file, starts the ingest API on `http://localhost:8000`, and starts one solo
worker for every dedicated queue. Press `Ctrl-C` to stop the complete local
process group.

Optional startup settings:

```bash
HTTP_HOST=127.0.0.1 HTTP_PORT=8080 CELERY_LOG_LEVEL=DEBUG ./start-all.sh
```

Check the service and its dependencies:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/ready
```

Interactive API documentation is available at
[`http://localhost:8000/`](http://localhost:8000/).

### 5. Submit audio

```bash
curl --request POST http://localhost:8000/v1/raw-audios \
  --form "audio=@./conversation.mp3" \
  --form "title=Example conversation" \
  --form "source_url=https://example.com/source" \
  --form "lang=en"
```

The API returns `202 Accepted` for a new upload and includes the source UUID and
initial Celery task ID. Uploads are deduplicated by the SHA-1 of the original
bytes; an existing upload returns `200 OK` with `deduplicated: true`.

Use the returned UUID to inspect the source row:

```bash
curl --fail http://localhost:8000/v1/raw-audios/<raw_audio_id>
```

This endpoint reports the `raw_audios` state. Downstream audio-part and chunk
states are stored in PostgreSQL, while their audio and JSON artifacts are stored
under deterministic keys in the configured bucket.

## Processing stages

| Stage | Input | Primary result |
| --- | --- | --- |
| Ingest API | Uploaded audio and metadata | Normalized WAV and `raw_audios` row |
| VAD split | Raw-audio UUID | Conversation-window WAVs and `audio_parts` rows |
| Diarization | Audio-part UUID | Speaker turns and clean reference WAVs |
| Quality filter | Diarized audio part | Clean two-speaker `chunks` rows |
| Separation | Chunk UUID | Two fixed speaker tracks and audited speaker mapping |
| Transcription | Separated English chunk | Transcript and word-alignment artifacts |
| Speaker profile (`persona_chunk`) | Transcribed chunk | Structured scene and speaker-profile document |
| Dialogue expansion (`extend_chunk`) | Profile-complete chunk | New dialogue script, transcript, and two synthesized tracks |

Task names, queue names, UUID arguments, and successor relationships are
defined centrally in
[`packages/task-contracts`](packages/task-contracts/README.md).

## Repository layout

```text
services/ingest-api/             HTTP ingestion and read-only status API
tasks/                           One independently deployable Celery worker per stage
packages/models/                 Shared SQLAlchemy persistence models
packages/task-contracts/         Stable task names, queues, and UUID contracts
packages/task-client/            Confirmed, bounded-retry task publication
packages/diarization-artifact/   Shared diarization artifact contract
packages/chunk-contracts/        Shared chunk artifact and speaker identity contracts
schema/schema.sql                Authoritative PostgreSQL schema
assets/                          Project-page audio and comparison artifacts
index.html                       Static project page
start-all.sh                     Local all-services launcher
```

Each service and task contains a focused README with its exact runtime contract,
configuration, model policy, output schema, and test commands.

## Development and tests

The repository is a collection of independent uv projects. Run tests from the
project you are changing:

```bash
cd services/ingest-api  # or tasks/<task-name>, packages/<package-name>
uv sync
uv run pytest
```

Default test suites are self-contained. Integration, model smoke, and capacity
tests are opt-in and document their external prerequisites in the corresponding
project README.

## License

The project-authored software is released under the
[Business Source License 1.1](LICENSE). Non-production use is permitted, and
the Additional Use Grant permits non-commercial academic research, education,
evaluation, testing, and benchmarking in production. Other production use
requires a separate commercial license from AveraLabs. Each version converts
to the MIT License four years after its first public distribution under the
Business Source License 1.1.

See [Third-Party Notices](THIRD_PARTY_NOTICES.md) for third-party licenses and
attributions. The project license does not replace or override those terms.

[project-page]: //avera-labs.github.io/ConversationalVoice/
