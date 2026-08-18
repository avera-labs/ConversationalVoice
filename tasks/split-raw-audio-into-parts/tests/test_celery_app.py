from voice_pipeline_task_contracts import SPLIT_RAW_AUDIO_INTO_PARTS

from voice_pipeline_split_raw_audio_into_parts.celery_app import create_celery_app
from voice_pipeline_split_raw_audio_into_parts.config import EnvironmentSettings
from voice_pipeline_split_raw_audio_into_parts.task import register_split_task


def _environment() -> EnvironmentSettings:
    return EnvironmentSettings(
        database_url="postgresql://example.test/database",
        celery_broker_url="redis://example.test:6379/0",
        s3_bucket="test-bucket",
        s3_region="test-region-1",
        hf_token="<hf-token>",
    )


def test_celery_app_uses_registered_route_and_worker_safety_options() -> None:
    app = create_celery_app(_environment())

    assert app.conf.task_routes == {
        SPLIT_RAW_AUDIO_INTO_PARTS.name: {
            "queue": SPLIT_RAW_AUDIO_INTO_PARTS.queue,
        }
    }
    assert app.conf.task_acks_late is True
    assert app.conf.task_reject_on_worker_lost is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.task_serializer == "json"
    assert app.conf.accept_content == ["json"]


def test_registered_task_delegates_only_the_uuid_string() -> None:
    app = create_celery_app(_environment())
    received: list[str] = []

    def handler(raw_audio_id: str) -> dict[str, object]:
        received.append(raw_audio_id)
        return {"raw_audio_id": raw_audio_id, "status": "accepted"}

    task = register_split_task(app, handler)
    result = task.apply(args=["12345678-1234-5678-1234-567812345678"]).get()

    assert task.name == SPLIT_RAW_AUDIO_INTO_PARTS.name
    assert task.queue == SPLIT_RAW_AUDIO_INTO_PARTS.queue
    assert task.acks_late is True
    assert task.reject_on_worker_lost is True
    assert received == ["12345678-1234-5678-1234-567812345678"]
    assert result == {
        "raw_audio_id": "12345678-1234-5678-1234-567812345678",
        "status": "accepted",
    }
