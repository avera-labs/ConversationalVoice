from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from voice_pipeline_ingest_api.dependencies import (
    get_ingest_service,
    get_raw_audio_repository,
)
from voice_pipeline_ingest_api.repository import (
    RawAudioRecord,
    RawAudioRepositoryError,
)
from voice_pipeline_ingest_api.routes.raw_audios import router
from voice_pipeline_ingest_api.services.audio_normalizer import (
    AudioNormalizationTimeout,
)
from voice_pipeline_ingest_api.services.ingest import (
    IngestRequest,
    IngestResult,
    IngestTaskPublicationError,
    ObjectStorageUploadError,
    UploadTooLargeError,
)

RAW_AUDIO_ID = UUID("12345678-1234-5678-1234-567812345678")
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _record(*, status: str = "pending") -> RawAudioRecord:
    return RawAudioRecord(
        id=RAW_AUDIO_ID,
        status=status,
        audio_uri=(
            "s3://test-bucket/raw_audios/12345678-1234-5678-1234-567812345678/audio.wav"
        ),
        content_sha1="a" * 40,
        title="Example",
        source_url="https://example.test/audio",
        lang="en",
        meta={"feed": "example"},
        duration_ms=100,
        size_bytes=3_244,
        error=None,
        created_at=NOW,
        updated_at=NOW,
    )


class FakeService:
    def __init__(self, result: IngestResult | None = None) -> None:
        self.result = result
        self.error: Exception | None = None
        self.requests: list[IngestRequest] = []
        self.upload_bytes: bytes | None = None

    def ingest(self, request: IngestRequest) -> IngestResult:
        self.requests.append(request)
        self.upload_bytes = request.source.read()
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class FakeRepository:
    def __init__(self, record: RawAudioRecord | None) -> None:
        self.record = record
        self.error: Exception | None = None
        self.queried: list[UUID] = []

    def get(self, raw_audio_id: UUID) -> RawAudioRecord | None:
        self.queried.append(raw_audio_id)
        if self.error is not None:
            raise self.error
        return self.record


def _app(*, service: FakeService, repository: FakeRepository) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_ingest_service] = lambda: service
    app.dependency_overrides[get_raw_audio_repository] = lambda: repository
    return app


def test_create_returns_202_and_parses_form_metadata() -> None:
    async def scenario() -> None:
        service = FakeService(
            IngestResult(
                record=_record(),
                task_id="task-123",
                deduplicated=False,
            )
        )
        app = _app(service=service, repository=FakeRepository(None))
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/raw-audios",
                files={"audio": ("episode.mp3", b"original-audio", "audio/mpeg")},
                data={
                    "title": "Example",
                    "source_url": "https://example.test/audio",
                    "lang": "en",
                    "meta": '{"feed":"example"}',
                },
            )

        assert response.status_code == 202
        assert response.json() == {
            "id": str(RAW_AUDIO_ID),
            "status": "pending",
            "content_sha1": "a" * 40,
            "task_id": "task-123",
            "deduplicated": False,
        }
        assert service.upload_bytes == b"original-audio"
        assert service.requests[0].meta == {"feed": "example"}

    asyncio.run(scenario())


def test_create_duplicate_returns_200_without_task_id() -> None:
    async def scenario() -> None:
        service = FakeService(
            IngestResult(
                record=_record(status="failed"),
                task_id=None,
                deduplicated=True,
            )
        )
        app = _app(service=service, repository=FakeRepository(None))
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/raw-audios",
                files={"audio": ("episode.wav", b"same-content", "audio/wav")},
            )

        assert response.status_code == 200
        assert response.json()["status"] == "failed"
        assert response.json()["task_id"] is None
        assert response.json()["deduplicated"] is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (UploadTooLargeError("Uploaded audio exceeds the size limit."), 413),
        (AudioNormalizationTimeout("Audio normalization timed out."), 504),
        (ObjectStorageUploadError("Unable to store normalized audio."), 502),
        (RawAudioRepositoryError("database unavailable"), 503),
        (IngestTaskPublicationError(RAW_AUDIO_ID), 503),
    ],
)
def test_create_maps_processing_failures(error, expected_status) -> None:
    async def scenario() -> None:
        service = FakeService()
        service.error = error
        app = _app(service=service, repository=FakeRepository(None))
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/raw-audios",
                files={"audio": ("episode.wav", b"audio", "audio/wav")},
            )

        assert response.status_code == expected_status
        assert "database unavailable" not in response.text
        if isinstance(error, IngestTaskPublicationError):
            assert response.json()["detail"]["raw_audio_id"] == str(RAW_AUDIO_ID)

    asyncio.run(scenario())


@pytest.mark.parametrize("meta", ["not-json", "[]", "null", "1"])
def test_create_rejects_meta_that_is_not_a_json_object(meta: str) -> None:
    async def scenario() -> None:
        service = FakeService()
        app = _app(service=service, repository=FakeRepository(None))
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/raw-audios",
                files={"audio": ("episode.wav", b"audio", "audio/wav")},
                data={"meta": meta},
            )

        assert response.status_code == 422
        assert service.requests == []

    asyncio.run(scenario())


def test_get_returns_complete_read_only_record() -> None:
    async def scenario() -> None:
        repository = FakeRepository(_record())
        service = FakeService()
        app = _app(service=service, repository=repository)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(f"/v1/raw-audios/{RAW_AUDIO_ID}")

        assert response.status_code == 200
        assert response.json() == {
            "id": str(RAW_AUDIO_ID),
            "status": "pending",
            "audio_uri": (
                "s3://test-bucket/raw_audios/"
                "12345678-1234-5678-1234-567812345678/audio.wav"
            ),
            "content_sha1": "a" * 40,
            "title": "Example",
            "source_url": "https://example.test/audio",
            "lang": "en",
            "meta": {"feed": "example"},
            "duration_ms": 100,
            "size_bytes": 3_244,
            "error": None,
            "created_at": "2026-01-02T03:04:05Z",
            "updated_at": "2026-01-02T03:04:05Z",
        }
        assert repository.queried == [RAW_AUDIO_ID]
        assert service.requests == []

    asyncio.run(scenario())


def test_get_returns_404_and_invalid_uuid_returns_422() -> None:
    async def scenario() -> None:
        repository = FakeRepository(None)
        app = _app(service=FakeService(), repository=repository)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            missing = await client.get(f"/v1/raw-audios/{RAW_AUDIO_ID}")
            invalid = await client.get("/v1/raw-audios/not-a-uuid")

        assert missing.status_code == 404
        assert invalid.status_code == 422

    asyncio.run(scenario())


def test_get_repository_failure_is_generic_and_retry_route_does_not_exist() -> None:
    async def scenario() -> None:
        repository = FakeRepository(None)
        repository.error = RawAudioRepositoryError(
            "postgresql://secret-user:secret-password@private-host/database"
        )
        app = _app(service=FakeService(), repository=repository)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            failed = await client.get(f"/v1/raw-audios/{RAW_AUDIO_ID}")
            retry = await client.post(f"/v1/raw-audios/{RAW_AUDIO_ID}/retry")

        assert failed.status_code == 503
        assert "secret-password" not in failed.text
        assert retry.status_code == 404

    asyncio.run(scenario())
