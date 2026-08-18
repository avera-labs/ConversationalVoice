from voice_pipeline_task_contracts import SEPARATE_CHUNK

from voice_pipeline_separate_chunk.celery_app import create_app


class Env:
    celery_broker_url = "redis://example/0"


def test_route_uses_shared_contract():
    app = create_app(Env())
    assert app.conf.task_routes[SEPARATE_CHUNK.name]["queue"] == SEPARATE_CHUNK.queue
    app.close()
