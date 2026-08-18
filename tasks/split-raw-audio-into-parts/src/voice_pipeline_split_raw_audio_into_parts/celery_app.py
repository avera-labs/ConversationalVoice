from celery import Celery
from voice_pipeline_task_contracts import SPLIT_RAW_AUDIO_INTO_PARTS

from .config import EnvironmentSettings


def create_celery_app(settings: EnvironmentSettings) -> Celery:
    """Create the Celery app without importing model or persistence adapters."""

    app = Celery(
        "voice-pipeline-split-raw-audio-into-parts",
        broker=settings.celery_broker_url,
    )
    app.conf.update(
        accept_content=["json"],
        result_serializer="json",
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_routes={
            SPLIT_RAW_AUDIO_INTO_PARTS.name: {
                "queue": SPLIT_RAW_AUDIO_INTO_PARTS.queue,
            }
        },
        task_serializer="json",
        worker_prefetch_multiplier=1,
    )
    return app
