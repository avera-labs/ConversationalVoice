from __future__ import annotations

from typing import Any

from celery.signals import worker_shutdown

from .config import load_application_settings
from .runtime import TaskRuntime


runtime = TaskRuntime.create(load_application_settings())
app = runtime.app


@worker_shutdown.connect(weak=False)
def close_runtime(**_kwargs: Any) -> None:
    """Release clients and model state when the worker process exits."""

    runtime.close()
