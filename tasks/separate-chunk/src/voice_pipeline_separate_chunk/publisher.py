from uuid import UUID

from voice_pipeline_task_client import TaskPublisher
from voice_pipeline_task_contracts import TRANSCRIBE_CHUNK, TRANSCRIBE_CHUNK_ZH


class TranscribeChunkPublisher:
    def __init__(self, publisher):
        self._publisher = publisher

    @classmethod
    def create(cls, environment):
        return cls(
            TaskPublisher.create(
                client_name="voice-pipeline-separate-chunk-publisher",
                broker_url=environment.celery_broker_url,
            )
        )

    def publish(self, chunk_id: UUID, language: str) -> str:
        contract = TRANSCRIBE_CHUNK_ZH if language == "zh" else TRANSCRIBE_CHUNK
        return self._publisher.publish(contract, chunk_id)

    def close(self):
        self._publisher.close()
