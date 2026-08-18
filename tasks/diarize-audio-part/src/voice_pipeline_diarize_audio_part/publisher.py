"""Quality-filter task publication adapter."""

from __future__ import annotations

from uuid import UUID

from voice_pipeline_task_client import TaskPublisher
from voice_pipeline_task_contracts import QUALITY_FILTER_AUDIO_PART

from .config import EnvironmentSettings


class QualityFilterPublisher:
    def __init__(self, task_publisher: TaskPublisher) -> None:
        self._task_publisher = task_publisher

    @classmethod
    def create(cls, settings: EnvironmentSettings) -> QualityFilterPublisher:
        return cls(
            TaskPublisher.create(
                client_name="voice-pipeline-diarize-audio-part-publisher",
                broker_url=settings.celery_broker_url,
            )
        )

    def check_readiness(self) -> None:
        self._task_publisher.check_readiness()

    def publish(self, audio_part_id: UUID) -> str:
        return self._task_publisher.publish(QUALITY_FILTER_AUDIO_PART, audio_part_id)

    def close(self) -> None:
        self._task_publisher.close()
