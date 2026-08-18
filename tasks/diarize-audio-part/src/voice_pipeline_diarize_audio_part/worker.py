"""Single-process diarization worker entry point."""

from __future__ import annotations

from typing import Any

from celery.signals import worker_shutdown
from voice_pipeline_task_contracts import DIARIZE_AUDIO_PART

from .config import load_settings
from .runtime import TaskRuntime
from .workspace import cleanup_orphaned_workspaces

settings = load_settings()
cleanup_orphaned_workspaces(prefix=settings.policy.task.workspace_prefix)
runtime = TaskRuntime.create(settings)
app = runtime.app


@worker_shutdown.connect(weak=False)
def close_runtime(**_kwargs: Any) -> None:
    runtime.close()


def main() -> None:
    app.worker_main(
        [
            "worker",
            "--loglevel=INFO",
            "--pool=solo",
            "--concurrency=1",
            f"--queues={DIARIZE_AUDIO_PART.queue}",
        ]
    )
