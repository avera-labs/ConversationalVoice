from voice_pipeline_task_contracts import DIARIZE_AUDIO_PART

from voice_pipeline_diarize_audio_part.celery_app import create_celery_app
from voice_pipeline_diarize_audio_part.config import EnvironmentSettings
from voice_pipeline_diarize_audio_part.task import register_diarization_task


def environment() -> EnvironmentSettings:
    return EnvironmentSettings(
        database_url="postgresql://db.example/test",
        celery_broker_url="redis://broker.example/0",
        s3_bucket="bucket",
        s3_region="region",
    )


def test_celery_safety_configuration_and_registry_route() -> None:
    app = create_celery_app(environment())
    assert app.conf.task_acks_late is True
    assert app.conf.task_reject_on_worker_lost is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert (
        app.conf.task_routes[DIARIZE_AUDIO_PART.name]["queue"]
        == DIARIZE_AUDIO_PART.queue
    )


def test_task_registration_uses_shared_contract() -> None:
    app = create_celery_app(environment())
    received: list[str] = []

    class Handler:
        def __call__(self, identifier: str) -> dict[str, str]:
            received.append(identifier)
            return {"id": identifier}

    task = register_diarization_task(app, Handler())
    assert task.name == DIARIZE_AUDIO_PART.name
    assert task.ignore_result is True
    assert task.run("12345678-1234-5678-1234-567812345678") == {
        "id": "12345678-1234-5678-1234-567812345678"
    }
    assert received == ["12345678-1234-5678-1234-567812345678"]
