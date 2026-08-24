from voice_pipeline_task_contracts import TRANSCRIBE_CHUNK_ZH
from voice_pipeline_transcribe_chunk_zh.celery_app import create_app
from voice_pipeline_transcribe_chunk_zh.config import load_settings
from voice_pipeline_transcribe_chunk_zh.task import register


def test_routes_only_chinese_transcription_task():
    settings = load_settings(
        {
            "DATABASE_URL": "postgresql://example/db",
            "CELERY_BROKER_URL": "redis://example/0",
            "S3_BUCKET": "bucket",
            "S3_REGION": "us-east-1",
        }
    )
    app = create_app(settings.environment)
    try:
        assert app.conf.task_routes == {
            TRANSCRIBE_CHUNK_ZH.name: {"queue": TRANSCRIBE_CHUNK_ZH.queue}
        }
        assert app.conf.worker_prefetch_multiplier == 1
        assert app.conf.task_serializer == "json"
    finally:
        app.close()


def test_registered_celery_task_dispatches_uuid_argument():
    settings = load_settings(
        {
            "DATABASE_URL": "postgresql://example/db",
            "CELERY_BROKER_URL": "redis://example/0",
            "S3_BUCKET": "bucket",
            "S3_REGION": "us-east-1",
        }
    )
    app = create_app(settings.environment)
    calls = []
    try:
        task = register(app, lambda value: calls.append(value) or "complete")
        assert task.name == TRANSCRIBE_CHUNK_ZH.name
        assert task.run("11111111-1111-1111-1111-111111111111") == "complete"
        assert calls == ["11111111-1111-1111-1111-111111111111"]
    finally:
        app.close()
