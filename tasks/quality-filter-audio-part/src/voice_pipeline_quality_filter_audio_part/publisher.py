from uuid import UUID

from voice_pipeline_task_client import TaskPublisher
from voice_pipeline_task_contracts import SEPARATE_CHUNK


class SeparateChunkPublisher:
    def __init__(self, publisher):
        self._publisher = publisher

    @classmethod
    def create(cls, settings):
        return cls(
            TaskPublisher.create(
                client_name="voice-pipeline-quality-filter-publisher",
                broker_url=settings.celery_broker_url,
            )
        )

    def publish(self, chunk_id: UUID) -> str:
        return self._publisher.publish(SEPARATE_CHUNK, chunk_id)

    def close(self):
        self._publisher.close()
