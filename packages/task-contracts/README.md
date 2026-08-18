# Voice Pipeline Task Contracts

This package is the single registry for Celery task names, queue names, and
single-UUID argument contracts used across independently deployed services and
workers. It has no Celery dependency and does not import task implementations.

## Registered tasks

| Constant | Task name | Queue | UUID argument |
| --- | --- | --- | --- |
| `SPLIT_RAW_AUDIO_INTO_PARTS` | `split_raw_audio_into_parts` | `split_raw_audio_into_parts` | `raw_audio_id`: UUID string |
| `DIARIZE_AUDIO_PART` | `diarize_audio_part` | `diarize_audio_part` | `audio_part_id`: UUID string |
| `QUALITY_FILTER_AUDIO_PART` | `quality_filter_audio_part` | `quality_filter_audio_part` | `audio_part_id`: UUID string |
| `SEPARATE_CHUNK` | `separate_chunk` | `separate_chunk` | `chunk_id`: UUID string |
| `TRANSCRIBE_CHUNK` | `transcribe_chunk` | `transcribe_chunk` | `chunk_id`: UUID string |
| `PERSONA_CHUNK` | `persona_chunk` | `persona_chunk` | `chunk_id`: UUID string |
| `EXTEND_CHUNK` | `extend_chunk` | `extend_chunk` | `chunk_id`: UUID string |

The diarization task publishes `QUALITY_FILTER_AUDIO_PART` only after its
durable artifact URI and `diarized` status commit successfully.
The quality-filter task publishes `SEPARATE_CHUNK` after committing accepted
chunks. Separation publishes `TRANSCRIBE_CHUNK` for English chunks after its
durable completion commit.
Transcription publishes `PERSONA_CHUNK` after its durable completion commit.
Persona publishes `EXTEND_CHUNK` after its durable completion commit.

Publishers use `voice-pipeline-task-client`, which applies the registered
queue, UUID serialization, bounded retry policy, publish confirmation, and safe
errors consistently:

```python
from voice_pipeline_task_client import TaskPublisher
from voice_pipeline_task_contracts import SPLIT_RAW_AUDIO_INTO_PARTS

publisher.publish(SPLIT_RAW_AUDIO_INTO_PARTS, raw_audio_id)
```

Dynamic callers use `TaskPublisher.publish_registered(task_name, identifier)`.
An unknown name is rejected before Celery is contacted.

Workers use the same object when registering their implementation:

```python
from voice_pipeline_task_contracts import SPLIT_RAW_AUDIO_INTO_PARTS

@celery_app.task(
    name=SPLIT_RAW_AUDIO_INTO_PARTS.name,
    queue=SPLIT_RAW_AUDIO_INTO_PARTS.queue,
)
def split_raw_audio_into_parts(raw_audio_id: str) -> None:
    ...
```

## Development

```bash
uv sync
uv run pytest
```
