from voice_pipeline_reconstruct_chunk.celery_app import create_app
from voice_pipeline_task_contracts import RECONSTRUCT_CHUNK


def test_task_route(settings):
    app = create_app(settings.environment)
    assert app.conf.task_routes == {
        RECONSTRUCT_CHUNK.name: {"queue": RECONSTRUCT_CHUNK.queue}
    }
    app.close()
