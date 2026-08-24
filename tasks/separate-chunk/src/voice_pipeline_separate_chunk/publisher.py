from uuid import UUID

from voice_pipeline_task_client import TaskPublisher
from voice_pipeline_task_contracts import TRANSCRIBE_CHUNK, TRANSCRIBE_CHUNK_ZH


TASK_BY_LANGUAGE = {
    "en": TRANSCRIBE_CHUNK,
    "zh": TRANSCRIBE_CHUNK_ZH,
}


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
        try:
            contract = TASK_BY_LANGUAGE[language]
        except KeyError as exc:
            raise ValueError("unsupported chunk language") from exc
        return self._publisher.publish(contract, chunk_id)

    def close(self):
        self._publisher.close()
