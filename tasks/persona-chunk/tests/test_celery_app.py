from voice_pipeline_task_contracts import PERSONA_CHUNK

from voice_pipeline_persona_chunk.celery_app import create_app


class Environment:
    celery_broker_url = "redis://example/0"


def test_celery_routes_persona_contract():
    app = create_app(Environment())
    try:
        assert app.conf.task_routes == {
            PERSONA_CHUNK.name: {"queue": PERSONA_CHUNK.queue}
        }
        assert app.conf.worker_prefetch_multiplier == 1
    finally:
        app.close()
