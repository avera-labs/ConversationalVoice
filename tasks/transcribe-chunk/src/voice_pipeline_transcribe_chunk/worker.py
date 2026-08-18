from typing import Any

from celery.signals import worker_shutdown
from voice_pipeline_task_contracts import TRANSCRIBE_CHUNK

from .config import load_settings
from .runtime import Runtime
from .workspace import cleanup_orphaned_workspaces

settings = load_settings()
cleanup_orphaned_workspaces(prefix=settings.policy.task.workspace_prefix)
runtime = Runtime.create(settings)
app = runtime.app


@worker_shutdown.connect(weak=False)
def close_runtime(**_kwargs: Any):
    runtime.close()


def main():
    app.worker_main(
        [
            "worker",
            "--loglevel=INFO",
            "--pool=solo",
            "--concurrency=1",
            f"--queues={TRANSCRIBE_CHUNK.queue}",
        ]
    )
