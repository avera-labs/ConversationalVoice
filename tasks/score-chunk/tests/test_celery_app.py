from types import SimpleNamespace

from voice_pipeline_score_chunk.celery_app import create_app
from voice_pipeline_task_contracts import SCORE_CHUNK


def test_app_routes_only_score_queue() -> None:
    app = create_app(SimpleNamespace(celery_broker_url="redis://localhost/0"))
    assert app.conf.task_routes == {SCORE_CHUNK.name: {"queue": SCORE_CHUNK.queue}}
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.task_acks_late is True
