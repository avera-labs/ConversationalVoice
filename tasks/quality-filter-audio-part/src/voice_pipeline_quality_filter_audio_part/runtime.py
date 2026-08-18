"""Process-local adapter composition and shutdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from celery import Celery
from celery.app.task import Task

from .celery_app import create_celery_app
from .certification import ExecutionTarget, detect_execution_target
from .config import DEFAULT_MUSIC_MODEL_CACHE_DIR, Settings
from .music import KerasMusicDetector
from .publisher import SeparateChunkPublisher
from .repository import QualityFilterRepository
from .storage import ObjectStorage
from .task import QualityFilterAudioPartHandler, register_quality_filter_task


@dataclass(slots=True)
class TaskRuntime:
    app: Celery
    repository: QualityFilterRepository
    storage: ObjectStorage
    music_detector: KerasMusicDetector
    target: ExecutionTarget
    task: Task
    publisher: SeparateChunkPublisher | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(cls, settings: Settings) -> TaskRuntime:
        cache_dir = settings.environment.music_model_cache_dir.resolve()
        project_root = Path(__file__).resolve().parents[4]
        bundled_model_dir = DEFAULT_MUSIC_MODEL_CACHE_DIR.resolve()
        if cache_dir != bundled_model_dir and (
            cache_dir == project_root or project_root in cache_dir.parents
        ):
            raise RuntimeError(
                "music model cache must be outside the project workspace"
            )
        if not cache_dir.is_dir() or cache_dir.is_symlink():
            raise RuntimeError("music model cache directory is invalid")
        target = detect_execution_target()
        app = create_celery_app(settings.environment)
        repository: QualityFilterRepository | None = None
        storage: ObjectStorage | None = None
        detector: KerasMusicDetector | None = None
        publisher: SeparateChunkPublisher | None = None
        try:
            repository = QualityFilterRepository.create(settings.environment)
            storage = ObjectStorage.create(settings.environment, settings.policy.task)
            detector = KerasMusicDetector(
                cache_dir=cache_dir,
                music_policy=settings.policy.music,
                quality_policy=settings.policy.quality,
            )
            detector.validate_artifacts()
            publisher = SeparateChunkPublisher.create(settings.environment)
            handler = QualityFilterAudioPartHandler(
                repository=repository,
                storage=storage,
                music_detector=detector,
                quality_policy=settings.policy.quality,
                planner_policy=settings.policy.planner,
                task_policy=settings.policy.task,
                publisher=publisher,
            )
            task = register_quality_filter_task(app, handler)
            return cls(app, repository, storage, detector, target, task, publisher)
        except Exception:
            if detector is not None:
                detector.close()
            if publisher is not None:
                publisher.close()
            if storage is not None:
                storage.close()
            if repository is not None:
                repository.close()
            app.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.music_detector.close()
        if self.publisher is not None:
            self.publisher.close()
        self.storage.close()
        self.repository.close()
        self.app.close()
