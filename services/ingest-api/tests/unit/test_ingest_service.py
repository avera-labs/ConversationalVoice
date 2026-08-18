from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from voice_pipeline_task_contracts import (
    SPLIT_RAW_AUDIO_INTO_PARTS,
    TaskContract,
)

from voice_pipeline_ingest_api.repository import (
    DuplicateContentError,
    RawAudioCreate,
    RawAudioRecord,
    RawAudioRepositoryError,
)
from voice_pipeline_ingest_api.services.audio_normalizer import (
    AudioNormalizationTimeout,
)
from voice_pipeline_ingest_api.services.ingest import (
    TASK_PUBLICATION_ERROR,
    IngestRequest,
    IngestService,
    IngestTaskPublicationError,
    ObjectStorageUploadError,
)
from voice_pipeline_ingest_api.services.wav_validation import WavMetadata

RAW_AUDIO_ID = UUID("12345678-1234-5678-1234-567812345678")
EXISTING_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
NOW = datetime(2026, 1, 2, tzinfo=UTC)


def _record(
    *,
    raw_audio_id: UUID = RAW_AUDIO_ID,
    content_sha1: str = "a" * 40,
    status: str = "pending",
) -> RawAudioRecord:
    return RawAudioRecord(
        id=raw_audio_id,
        status=status,
        audio_uri=f"s3://test-bucket/raw_audios/{raw_audio_id}/audio.wav",
        content_sha1=content_sha1,
        title="Title",
        source_url="https://example.test/audio",
        lang="en",
        meta={"feed": "example"},
        duration_ms=100,
        size_bytes=3_244,
        error=None,
        created_at=NOW,
        updated_at=NOW,
    )


class FakeRepository:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.existing: RawAudioRecord | None = None
        self.create_error: Exception | None = None
        self.created: RawAudioCreate | None = None
        self.failed: tuple[UUID, str] | None = None

    def find_by_content_sha1(self, content_sha1: str) -> RawAudioRecord | None:
        self.events.append("find")
        return self.existing

    def create(self, values: RawAudioCreate) -> RawAudioRecord:
        self.events.append("create")
        self.created = values
        if self.create_error is not None:
            raise self.create_error
        return _record(content_sha1=values.content_sha1)

    def mark_failed(self, raw_audio_id: UUID, error: str) -> None:
        self.events.append("mark_failed")
        self.failed = (raw_audio_id, error)


class FakeNormalizer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.error: Exception | None = None
        self.source_path: Path | None = None
        self.destination_path: Path | None = None
        self.original_bytes: bytes | None = None

    def normalize(self, source_path: Path, destination_path: Path) -> WavMetadata:
        self.events.append("normalize")
        self.source_path = source_path
        self.destination_path = destination_path
        self.original_bytes = source_path.read_bytes()
        destination_path.write_bytes(b"normalized-wav")
        if self.error is not None:
            raise self.error
        return WavMetadata(duration_ms=100, size_bytes=3_244, frame_count=1_600)


class FakeStorage:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.upload_error: Exception | None = None
        self.uploaded: tuple[UUID, bytes] | None = None
        self.deleted: list[UUID] = []

    def upload_normalized_audio(self, raw_audio_id: UUID, path: Path) -> str:
        self.events.append("upload")
        self.uploaded = (raw_audio_id, path.read_bytes())
        if self.upload_error is not None:
            raise self.upload_error
        return f"s3://test-bucket/raw_audios/{raw_audio_id}/audio.wav"

    def delete_normalized_audio(self, raw_audio_id: UUID) -> None:
        self.events.append("delete")
        self.deleted.append(raw_audio_id)


class FakePublisher:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.error: Exception | None = None
        self.published: list[tuple[TaskContract, UUID]] = []

    def publish(self, contract: TaskContract, identifier: UUID) -> str:
        self.events.append("publish")
        self.published.append((contract, identifier))
        if self.error is not None:
            raise self.error
        return "task-123"


def _service():
    events: list[str] = []
    repository = FakeRepository(events)
    normalizer = FakeNormalizer(events)
    storage = FakeStorage(events)
    publisher = FakePublisher(events)
    service = IngestService(
        repository=repository,
        normalizer=normalizer,
        storage=storage,
        task_publisher=publisher,
        max_upload_bytes=100,
        id_factory=lambda: RAW_AUDIO_ID,
    )
    return service, repository, normalizer, storage, publisher, events


def _request(content: bytes = b"original-audio") -> IngestRequest:
    return IngestRequest(
        source=io.BytesIO(content),
        filename="episode.wav",
        title="Title",
        source_url="https://example.test/audio",
        lang="en",
        meta={"feed": "example"},
    )


def test_ingest_runs_side_effects_in_strict_order_and_cleans_workspace() -> None:
    service, repository, normalizer, storage, publisher, events = _service()

    result = service.ingest(_request())

    expected_sha1 = hashlib.sha1(
        b"original-audio",
        usedforsecurity=False,
    ).hexdigest()
    assert events == ["find", "normalize", "upload", "create", "publish"]
    assert normalizer.original_bytes == b"original-audio"
    assert storage.uploaded == (RAW_AUDIO_ID, b"normalized-wav")
    assert repository.created == RawAudioCreate(
        id=RAW_AUDIO_ID,
        audio_uri=(
            "s3://test-bucket/raw_audios/12345678-1234-5678-1234-567812345678/audio.wav"
        ),
        content_sha1=expected_sha1,
        title="Title",
        source_url="https://example.test/audio",
        lang="en",
        meta={"feed": "example"},
        duration_ms=100,
        size_bytes=3_244,
    )
    assert publisher.published == [(SPLIT_RAW_AUDIO_INTO_PARTS, RAW_AUDIO_ID)]
    assert result.task_id == "task-123"
    assert not result.deduplicated
    assert normalizer.source_path is not None
    assert normalizer.source_path.name == "original.wav"
    assert normalizer.destination_path is not None
    assert not normalizer.source_path.exists()
    assert not normalizer.destination_path.exists()


def test_existing_digest_stops_before_normalization_and_cleans_workspace() -> None:
    service, repository, normalizer, storage, publisher, events = _service()
    existing = _record(raw_audio_id=EXISTING_ID, status="failed")
    repository.existing = existing

    result = service.ingest(_request())

    assert events == ["find"]
    assert result.record == existing
    assert result.task_id is None
    assert result.deduplicated
    assert normalizer.source_path is None
    assert storage.uploaded is None
    assert publisher.published == []


def test_unique_race_deletes_orphan_and_returns_existing_record() -> None:
    service, repository, _, storage, publisher, events = _service()
    existing = _record(raw_audio_id=EXISTING_ID)
    repository.create_error = DuplicateContentError(existing)

    result = service.ingest(_request())

    assert events == ["find", "normalize", "upload", "create", "delete"]
    assert storage.deleted == [RAW_AUDIO_ID]
    assert publisher.published == []
    assert result.record == existing
    assert result.deduplicated


def test_insert_failure_deletes_orphan_and_does_not_publish() -> None:
    service, repository, _, storage, publisher, events = _service()
    repository.create_error = RawAudioRepositoryError("database unavailable")

    with pytest.raises(RawAudioRepositoryError):
        service.ingest(_request())

    assert events == ["find", "normalize", "upload", "create", "delete"]
    assert storage.deleted == [RAW_AUDIO_ID]
    assert publisher.published == []


def test_upload_failure_attempts_compensation_and_stops() -> None:
    service, repository, _, storage, publisher, events = _service()
    storage.upload_error = RuntimeError("storage unavailable")

    with pytest.raises(ObjectStorageUploadError):
        service.ingest(_request())

    assert events == ["find", "normalize", "upload", "delete"]
    assert repository.created is None
    assert publisher.published == []


def test_normalization_timeout_cleans_files_and_has_no_later_side_effects() -> None:
    service, repository, normalizer, storage, publisher, events = _service()
    normalizer.error = AudioNormalizationTimeout("Audio normalization timed out.")

    with pytest.raises(AudioNormalizationTimeout):
        service.ingest(_request())

    assert events == ["find", "normalize"]
    assert repository.created is None
    assert storage.uploaded is None
    assert publisher.published == []
    assert normalizer.source_path is not None
    assert normalizer.destination_path is not None
    assert not normalizer.source_path.exists()
    assert not normalizer.destination_path.exists()


def test_publish_failure_marks_committed_record_failed_and_retains_wav() -> None:
    service, repository, _, storage, publisher, events = _service()
    publisher.error = RuntimeError("broker unavailable")

    with pytest.raises(IngestTaskPublicationError) as captured:
        service.ingest(_request())

    assert captured.value.raw_audio_id == RAW_AUDIO_ID
    assert events == [
        "find",
        "normalize",
        "upload",
        "create",
        "publish",
        "mark_failed",
    ]
    assert repository.failed == (RAW_AUDIO_ID, TASK_PUBLICATION_ERROR)
    assert storage.deleted == []
