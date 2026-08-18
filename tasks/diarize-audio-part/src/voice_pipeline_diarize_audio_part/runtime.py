"""Process-local adapter composition and shutdown."""

from __future__ import annotations

from dataclasses import dataclass, field

from celery import Celery
from celery.app.task import Task

from .celery_app import create_celery_app
from .config import Settings
from .diarization import DiarizationEngine
from .publisher import QualityFilterPublisher
from .repository import DiarizationRepository
from .storage import ObjectStorage
from .task import DiarizeAudioPartHandler, register_diarization_task


@dataclass(slots=True)
class TaskRuntime:
    app: Celery
    repository: DiarizationRepository
    storage: ObjectStorage
    publisher: QualityFilterPublisher
    diarization: DiarizationEngine
    task: Task
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def create(cls, settings: Settings) -> TaskRuntime:
        app = create_celery_app(settings.environment)
        repository: DiarizationRepository | None = None
        storage: ObjectStorage | None = None
        publisher: QualityFilterPublisher | None = None
        diarization: DiarizationEngine | None = None
        try:
            repository = DiarizationRepository.create(settings.environment)
            storage = ObjectStorage.create(settings.environment)
            publisher = QualityFilterPublisher.create(settings.environment)
            diarization = DiarizationEngine(
                model=settings.policy.diarization.model,
                requested_device=settings.policy.diarization.device,
            )
            handler = DiarizeAudioPartHandler(
                repository=repository,
                storage=storage,
                diarization=diarization,
                publisher=publisher,
                diarization_policy=settings.policy.diarization,
                speaker_reference_policy=settings.policy.speaker_reference,
                task_policy=settings.policy.task,
            )
            task = register_diarization_task(app, handler)
            return cls(app, repository, storage, publisher, diarization, task)
        except Exception:
            if diarization is not None:
                diarization.close()
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
        self.diarization.close()
        self.publisher.close()
        self.storage.close()
        self.repository.close()
        self.app.close()
