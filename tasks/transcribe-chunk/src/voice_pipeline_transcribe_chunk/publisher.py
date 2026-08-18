from uuid import UUID

from voice_pipeline_task_client import TaskPublisher
from voice_pipeline_task_contracts import PERSONA_CHUNK


class PersonaChunkPublisher:
    def __init__(self, publisher):
        self._publisher = publisher

    @classmethod
    def create(cls, environment):
        return cls(
            TaskPublisher.create(
                client_name="voice-pipeline-transcribe-chunk-publisher",
                broker_url=environment.celery_broker_url,
            )
        )

    def publish(self, chunk_id: UUID) -> str:
        return self._publisher.publish(PERSONA_CHUNK, chunk_id)

    def close(self):
        self._publisher.close()
