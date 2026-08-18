from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO, Protocol
from uuid import UUID, uuid4

from voice_pipeline_task_contracts import (
    SPLIT_RAW_AUDIO_INTO_PARTS,
    TaskContract,
)

from ..repository import (
    DuplicateContentError,
    RawAudioCreate,
    RawAudioRecord,
    RawAudioRepositoryError,
)
from .wav_validation import WavMetadata

UPLOAD_COPY_CHUNK_BYTES = 1024 * 1024
UPLOAD_SUFFIX_MAX_CHARS = 10
TASK_PUBLICATION_ERROR = "Failed to publish the split_raw_audio_into_parts task."
logger = logging.getLogger(__name__)


class UploadValidationError(ValueError):
    """Base error for invalid uploaded audio content."""


class EmptyUploadError(UploadValidationError):
    """Raised when an upload contains no bytes."""


class UploadTooLargeError(UploadValidationError):
    """Raised when an upload exceeds the configured size limit."""


@dataclass(frozen=True, slots=True)
class UploadDigest:
    """Metadata calculated from the original uploaded bytes."""

    content_sha1: str
    size_bytes: int


class ObjectStorageUploadError(RuntimeError):
    """Raised when a normalized WAV cannot be stored."""


class IngestTaskPublicationError(RuntimeError):
    """Raised when a committed raw audio cannot be sent to Celery."""

    def __init__(self, raw_audio_id: UUID) -> None:
        super().__init__("Unable to start downstream processing.")
        self.raw_audio_id = raw_audio_id


class RawAudioRepositoryPort(Protocol):
    def find_by_content_sha1(self, content_sha1: str) -> RawAudioRecord | None: ...

    def create(self, values: RawAudioCreate) -> RawAudioRecord: ...

    def mark_failed(self, raw_audio_id: UUID, error: str) -> None: ...


class AudioNormalizerPort(Protocol):
    def normalize(self, source_path: Path, destination_path: Path) -> WavMetadata: ...


class ObjectStoragePort(Protocol):
    def upload_normalized_audio(self, raw_audio_id: UUID, path: Path) -> str: ...

    def delete_normalized_audio(self, raw_audio_id: UUID) -> None: ...


class TaskPublisherPort(Protocol):
    def publish(self, contract: TaskContract, identifier: UUID) -> str: ...


@dataclass(frozen=True, slots=True)
class IngestRequest:
    """Validated upload fields passed into ingest orchestration."""

    source: BinaryIO
    filename: str | None
    title: str | None
    source_url: str | None
    lang: str
    meta: dict[str, Any]


def _safe_upload_suffix(filename: str | None) -> str:
    if filename is None:
        return ".upload"
    suffix = Path(filename).suffix.lower()
    if (
        1 < len(suffix) <= UPLOAD_SUFFIX_MAX_CHARS
        and suffix[1:].isalnum()
        and suffix.isascii()
    ):
        return suffix
    return ".upload"


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Result of creating or deduplicating raw audio."""

    record: RawAudioRecord
    task_id: str | None
    deduplicated: bool


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def stream_copy_and_sha1(
    source: BinaryIO,
    destination: Path,
    *,
    max_bytes: int,
) -> UploadDigest:
    """Copy an upload to disk while hashing and enforcing its actual size."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")

    digest = hashlib.sha1(usedforsecurity=False)
    size_bytes = 0

    try:
        with destination.open("wb") as output:
            while chunk := source.read(UPLOAD_COPY_CHUNK_BYTES):
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise UploadTooLargeError("Uploaded audio exceeds the size limit.")
                digest.update(chunk)
                output.write(chunk)

        if size_bytes == 0:
            raise EmptyUploadError("Uploaded audio is empty.")
    except BaseException:
        _remove_partial_file(destination)
        raise

    return UploadDigest(content_sha1=digest.hexdigest(), size_bytes=size_bytes)


class IngestService:
    """Coordinate the complete raw audio ingest transaction boundary."""

    def __init__(
        self,
        *,
        repository: RawAudioRepositoryPort,
        normalizer: AudioNormalizerPort,
        storage: ObjectStoragePort,
        task_publisher: TaskPublisherPort,
        max_upload_bytes: int,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._repository = repository
        self._normalizer = normalizer
        self._storage = storage
        self._task_publisher = task_publisher
        self._max_upload_bytes = max_upload_bytes
        self._id_factory = id_factory

    def ingest(self, request: IngestRequest) -> IngestResult:
        with TemporaryDirectory(prefix="voice-pipeline-ingest-") as directory:
            workspace = Path(directory)
            source_path = workspace / f"original{_safe_upload_suffix(request.filename)}"
            normalized_path = workspace / "audio.wav"

            upload = stream_copy_and_sha1(
                request.source,
                source_path,
                max_bytes=self._max_upload_bytes,
            )
            existing = self._repository.find_by_content_sha1(upload.content_sha1)
            if existing is not None:
                return IngestResult(
                    record=existing,
                    task_id=None,
                    deduplicated=True,
                )

            raw_audio_id = self._id_factory()
            wav = self._normalizer.normalize(source_path, normalized_path)
            audio_uri = self._upload(raw_audio_id, normalized_path)

            try:
                record = self._repository.create(
                    RawAudioCreate(
                        id=raw_audio_id,
                        audio_uri=audio_uri,
                        content_sha1=upload.content_sha1,
                        title=request.title,
                        source_url=request.source_url,
                        lang=request.lang,
                        meta=request.meta,
                        duration_ms=wav.duration_ms,
                        size_bytes=wav.size_bytes,
                    )
                )
            except DuplicateContentError as exc:
                self._delete_unreferenced_audio(raw_audio_id)
                return IngestResult(
                    record=exc.record,
                    task_id=None,
                    deduplicated=True,
                )
            except RawAudioRepositoryError:
                self._delete_unreferenced_audio(raw_audio_id)
                raise

            try:
                task_id = self._task_publisher.publish(
                    SPLIT_RAW_AUDIO_INTO_PARTS,
                    raw_audio_id,
                )
            except Exception as exc:
                self._mark_publication_failed(raw_audio_id)
                raise IngestTaskPublicationError(raw_audio_id) from exc

            return IngestResult(
                record=record,
                task_id=task_id,
                deduplicated=False,
            )

    def _upload(self, raw_audio_id: UUID, normalized_path: Path) -> str:
        try:
            return self._storage.upload_normalized_audio(
                raw_audio_id,
                normalized_path,
            )
        except Exception as exc:
            self._delete_unreferenced_audio(raw_audio_id)
            raise ObjectStorageUploadError("Unable to store normalized audio.") from exc

    def _delete_unreferenced_audio(self, raw_audio_id: UUID) -> None:
        try:
            self._storage.delete_normalized_audio(raw_audio_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Unable to delete unreferenced normalized audio for raw_audio_id=%s.",
                raw_audio_id,
            )

    def _mark_publication_failed(self, raw_audio_id: UUID) -> None:
        try:
            self._repository.mark_failed(
                raw_audio_id,
                TASK_PUBLICATION_ERROR,
            )
        except RawAudioRepositoryError:
            logger.error(
                "Unable to mark task publication failure for raw_audio_id=%s.",
                raw_audio_id,
            )
