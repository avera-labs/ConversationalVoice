from voice_pipeline_task_contracts import QUALITY_FILTER_AUDIO_PART
from voice_pipeline_quality_filter_audio_part.celery_app import create_celery_app
from voice_pipeline_quality_filter_audio_part.config import EnvironmentSettings
from voice_pipeline_quality_filter_audio_part.task import register_quality_filter_task


def settings(tmp_path):
    return EnvironmentSettings(
        database_url="postgresql://db/example",
        celery_broker_url="redis://broker/0",
        s3_bucket="bucket",
        s3_region="us-east-1",
        music_model_cache_dir=tmp_path,
    )


def test_json_only_solo_safe_celery_configuration(tmp_path) -> None:
    app = create_celery_app(settings(tmp_path))
    assert app.conf.task_serializer == "json"
    assert app.conf.accept_content == ["json"]
    assert app.conf.worker_prefetch_multiplier == 1
    task = register_quality_filter_task(app, lambda identifier: {"audio_part_id": identifier})
    assert task.name == QUALITY_FILTER_AUDIO_PART.name
    assert task.queue == QUALITY_FILTER_AUDIO_PART.queue
    assert task.acks_late is True
    assert task.reject_on_worker_lost is True
    app.close()
