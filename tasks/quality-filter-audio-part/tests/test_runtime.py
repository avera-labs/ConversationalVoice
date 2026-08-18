from pathlib import Path

import pytest

from voice_pipeline_quality_filter_audio_part.certification import ExecutionTarget
from voice_pipeline_quality_filter_audio_part.config import EnvironmentSettings, Settings, load_settings
from voice_pipeline_quality_filter_audio_part.runtime import TaskRuntime


class Closeable:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def close(self) -> None:
        self.events.append(self.name)


def test_runtime_closes_model_and_clients_once() -> None:
    events: list[str] = []
    runtime = TaskRuntime(
        app=Closeable("app", events),
        repository=Closeable("repository", events),
        storage=Closeable("storage", events),
        music_detector=Closeable("music", events),
        target=ExecutionTarget("x86_64", "cpu", "test"),
        task=object(),
    )
    runtime.close()
    runtime.close()
    assert events == ["music", "storage", "repository", "app"]


def settings(cache_dir: Path) -> Settings:
    loaded = load_settings(
        {
            "DATABASE_URL": "postgresql://unused/unused",
            "CELERY_BROKER_URL": "redis://unused/0",
            "S3_BUCKET": "unused",
            "S3_REGION": "unused",
            "MUSIC_MODEL_CACHE_DIR": str(cache_dir),
        }
    )
    return loaded


def test_model_cache_must_exist_and_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="cache directory"):
        TaskRuntime.create(settings(tmp_path / "missing"))


def test_model_cache_cannot_be_inside_project_workspace() -> None:
    project_root = Path(__file__).resolve().parents[2]
    with pytest.raises(RuntimeError, match="outside"):
        TaskRuntime.create(settings(project_root))
