from celery import Celery
from voice_pipeline_task_contracts import TRANSCRIBE_CHUNK_ZH


def create_app(environment):
    app = Celery(
        "voice-pipeline-transcribe-chunk-zh", broker=environment.celery_broker_url
    )
    app.conf.update(
        accept_content=["json"],
        task_serializer="json",
        result_serializer="json",
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_routes={TRANSCRIBE_CHUNK_ZH.name: {"queue": TRANSCRIBE_CHUNK_ZH.queue}},
    )
    return app
