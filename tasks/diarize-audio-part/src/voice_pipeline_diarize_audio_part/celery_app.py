"""Celery application factory."""

from celery import Celery
from voice_pipeline_task_contracts import DIARIZE_AUDIO_PART

from .config import EnvironmentSettings


def create_celery_app(settings: EnvironmentSettings) -> Celery:
    app = Celery("voice-pipeline-diarize-audio-part", broker=settings.celery_broker_url)
    app.conf.update(
        accept_content=["json"],
        result_serializer="json",
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_routes={DIARIZE_AUDIO_PART.name: {"queue": DIARIZE_AUDIO_PART.queue}},
        task_serializer="json",
        worker_prefetch_multiplier=1,
    )
    return app
