from voice_pipeline_task_contracts import TRANSCRIBE_CHUNK

from voice_pipeline_transcribe_chunk.celery_app import create_app
from voice_pipeline_transcribe_chunk.config import load_settings


def test_celery_route_is_exact():
    environment = load_settings(
        {
            "DATABASE_URL": "postgresql://example/db",
            "CELERY_BROKER_URL": "redis://example/0",
            "S3_BUCKET": "bucket",
            "S3_REGION": "us-east-1",
        }
    ).environment
    app = create_app(environment)
    try:
        assert app.conf.task_routes == {
            TRANSCRIBE_CHUNK.name: {"queue": TRANSCRIBE_CHUNK.queue}
        }
        assert app.conf.worker_prefetch_multiplier == 1
        assert app.conf.task_serializer == "json"
    finally:
        app.close()
