from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from voice_pipeline_task_client import TaskPublisher

from .config import ApplicationSettings
from .repository import RawAudioRepository
from .services.audio_normalizer import AudioNormalizer
from .services.ingest import IngestService
from .services.storage import ObjectStorage


def normalize_database_url(database_url: str) -> str:
    """Select the installed psycopg 3 driver for generic PostgreSQL URLs."""
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


@dataclass(slots=True)
class ServiceResources:
    """Process-local clients and connection factories."""

    engine: Engine
    session_factory: sessionmaker[Session]
    storage: ObjectStorage
    task_publisher: TaskPublisher

    @classmethod
    def create(cls, settings: ApplicationSettings) -> ServiceResources:
        engine = create_engine(
            normalize_database_url(settings.environment.database_url),
            pool_pre_ping=True,
        )
        return cls(
            engine=engine,
            session_factory=sessionmaker(
                bind=engine,
                expire_on_commit=False,
            ),
            storage=ObjectStorage.create(settings.environment),
            task_publisher=TaskPublisher.create(
                client_name="voice-pipeline-ingest-api",
                broker_url=settings.environment.celery_broker_url,
            ),
        )

    def check_readiness(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        self.storage.check_readiness()
        self.task_publisher.check_readiness()

    def close(self) -> None:
        self.task_publisher.close()
        self.engine.dispose()


def get_resources(request: Request) -> ServiceResources:
    return request.app.state.resources


ResourcesDependency = Annotated[ServiceResources, Depends(get_resources)]


def get_task_publisher(resources: ResourcesDependency) -> TaskPublisher:
    return resources.task_publisher


TaskPublisherDependency = Annotated[
    TaskPublisher,
    Depends(get_task_publisher),
]


def get_raw_audio_repository(
    resources: ResourcesDependency,
) -> RawAudioRepository:
    return RawAudioRepository(resources.session_factory)


RawAudioRepositoryDependency = Annotated[
    RawAudioRepository,
    Depends(get_raw_audio_repository),
]


def get_ingest_service(
    request: Request,
    resources: ResourcesDependency,
    repository: RawAudioRepositoryDependency,
) -> IngestService:
    settings: ApplicationSettings = request.app.state.settings
    return IngestService(
        repository=repository,
        normalizer=AudioNormalizer(),
        storage=resources.storage,
        task_publisher=resources.task_publisher,
        max_upload_bytes=settings.policy.ingest.max_upload_bytes,
    )


IngestServiceDependency = Annotated[IngestService, Depends(get_ingest_service)]
