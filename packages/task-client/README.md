# Voice Pipeline Task Client

This package provides the shared Celery producer used by HTTP services and
workers. It publishes only contracts from `voice-pipeline-task-contracts`, and
every message contains one UUID string positional argument.

The package owns producer serialization, retry policy, publish confirmation,
readiness checks, safe errors, and client lifecycle. Application projects
choose the contract they publish and keep their domain-specific sequencing.

```python
from voice_pipeline_task_client import TaskPublisher
from voice_pipeline_task_contracts import DIARIZE_AUDIO_PART

publisher = TaskPublisher.create(
    client_name="voice-pipeline-example",
    broker_url=broker_url,
)
task_id = publisher.publish(DIARIZE_AUDIO_PART, audio_part_id)
```

## Development

```bash
uv sync
uv run pytest
```
