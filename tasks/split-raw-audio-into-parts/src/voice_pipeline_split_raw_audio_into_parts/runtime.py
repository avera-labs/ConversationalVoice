from __future__ import annotations

from dataclasses import dataclass, field

from celery import Celery
from celery.app.task import Task

from .celery_app import create_celery_app
from .config import ApplicationSettings
from .publisher import DiarizationPublisher
from .repository import SplitRepository
from .storage import ObjectStorage
from .task import SplitRawAudioIntoPartsHandler, register_split_task
from .vad import PyannoteVad, clear_vad_pipeline_cache


@dataclass(slots=True)
class TaskRuntime:
    """Own the process-local adapters used by one Celery worker."""

    app: Celery
    repository: SplitRepository
    storage: ObjectStorage
    publisher: DiarizationPublisher
    vad: PyannoteVad
    task: Task
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(cls, settings: ApplicationSettings) -> TaskRuntime:
        """Compose the worker while keeping model weights lazily loaded."""

        app = create_celery_app(settings.environment)
        repository: SplitRepository | None = None
        storage: ObjectStorage | None = None
        publisher: DiarizationPublisher | None = None
        try:
            repository = SplitRepository.create(settings.environment)
            storage = ObjectStorage.create(settings.environment)
            publisher = DiarizationPublisher.create(settings.environment)
            vad = PyannoteVad.create(settings.policy.vad, settings.environment)
            handler = SplitRawAudioIntoPartsHandler(
                repository=repository,
                storage=storage,
                vad=vad,
                publisher=publisher,
                windowing_policy=settings.policy.windowing,
            )
            task = register_split_task(app, handler)
            return cls(
                app=app,
                repository=repository,
                storage=storage,
                publisher=publisher,
                vad=vad,
                task=task,
            )
        except Exception:
            if publisher is not None:
                publisher.close()
            if storage is not None:
                storage.close()
            if repository is not None:
                repository.close()
            app.close()
            raise

    def close(self) -> None:
        """Release process-local clients and cached model state once."""

        if self._closed:
            return
        self._closed = True
        self.publisher.close()
        self.storage.close()
        self.repository.close()
        self.app.close()
        clear_vad_pipeline_cache()
