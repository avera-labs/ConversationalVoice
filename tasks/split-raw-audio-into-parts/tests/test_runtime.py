from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline_split_raw_audio_into_parts.config import (
    ApplicationSettings,
    EnvironmentSettings,
    PolicySettings,
    VadPolicy,
    WindowingPolicy,
)
from voice_pipeline_split_raw_audio_into_parts import runtime as runtime_module


class Closable:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def close(self) -> None:
        self.events.append(f"close:{self.name}")


def _settings() -> ApplicationSettings:
    return ApplicationSettings(
        environment=EnvironmentSettings(
            database_url="postgresql://example.test/database",
            celery_broker_url="redis://example.test:6379/0",
            s3_bucket="test-bucket",
            s3_region="test-region-1",
            hf_token="<hf-token>",
        ),
        policy=PolicySettings(
            vad=VadPolicy(
                model="pyannote/segmentation-3.0",
                device="cpu",
            ),
            windowing=WindowingPolicy(
                gap_threshold_ms=15_000,
                min_window_ms=20_000,
                max_window_ms=900_000,
                pad_before_ms=250,
                pad_after_ms=250,
            ),
        ),
    )


def test_runtime_composes_handler_and_closes_resources_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    app = Closable("app", events)
    repository = Closable("repository", events)
    storage = Closable("storage", events)
    publisher = Closable("publisher", events)
    vad = SimpleNamespace(name="lazy-vad")
    registered_task = SimpleNamespace(name="registered-task")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(runtime_module, "create_celery_app", lambda settings: app)
    monkeypatch.setattr(
        runtime_module.SplitRepository,
        "create",
        lambda settings: repository,
    )
    monkeypatch.setattr(
        runtime_module.ObjectStorage,
        "create",
        lambda settings: storage,
    )
    monkeypatch.setattr(
        runtime_module.DiarizationPublisher,
        "create",
        lambda settings: publisher,
    )
    monkeypatch.setattr(
        runtime_module.PyannoteVad,
        "create",
        lambda policy, environment: vad,
    )

    def register(app_value: Any, handler: Any) -> Any:
        captured["app"] = app_value
        captured["handler"] = handler
        return registered_task

    monkeypatch.setattr(runtime_module, "register_split_task", register)
    monkeypatch.setattr(
        runtime_module,
        "clear_vad_pipeline_cache",
        lambda: events.append("clear:vad"),
    )

    runtime = runtime_module.TaskRuntime.create(_settings())

    assert captured["app"] is app
    assert runtime.app is app
    assert runtime.repository is repository
    assert runtime.storage is storage
    assert runtime.publisher is publisher
    assert runtime.vad is vad
    assert runtime.task is registered_task

    runtime.close()
    runtime.close()

    assert events == [
        "close:publisher",
        "close:storage",
        "close:repository",
        "close:app",
        "clear:vad",
    ]


def test_partial_runtime_failure_closes_created_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    app = Closable("app", events)
    repository = Closable("repository", events)
    storage = Closable("storage", events)

    monkeypatch.setattr(runtime_module, "create_celery_app", lambda settings: app)
    monkeypatch.setattr(
        runtime_module.SplitRepository,
        "create",
        lambda settings: repository,
    )
    monkeypatch.setattr(
        runtime_module.ObjectStorage,
        "create",
        lambda settings: storage,
    )

    def fail_publisher(settings: Any) -> Any:
        raise RuntimeError("publisher setup failed")

    monkeypatch.setattr(
        runtime_module.DiarizationPublisher,
        "create",
        fail_publisher,
    )

    with pytest.raises(RuntimeError, match="publisher setup failed"):
        runtime_module.TaskRuntime.create(_settings())

    assert events == [
        "close:storage",
        "close:repository",
        "close:app",
    ]
