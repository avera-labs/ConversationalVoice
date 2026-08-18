from __future__ import annotations

import json
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status

from ..dependencies import IngestServiceDependency, RawAudioRepositoryDependency
from ..repository import RawAudioRecord, RawAudioRepositoryError
from ..schemas import CreateRawAudioResponse, RawAudioResponse
from ..services.audio_normalizer import (
    AudioNormalizationError,
    AudioNormalizationTimeout,
)
from ..services.ingest import (
    EmptyUploadError,
    IngestRequest,
    IngestTaskPublicationError,
    ObjectStorageUploadError,
    UploadTooLargeError,
)

router = APIRouter(prefix="/v1/raw-audios", tags=["raw-audios"])


def _parse_meta(value: str | None) -> dict[str, Any]:
    if value is None or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="meta must be a JSON object.",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="meta must be a JSON object.",
        )
    return parsed


def _raw_audio_response(record: RawAudioRecord) -> RawAudioResponse:
    return RawAudioResponse(
        id=record.id,
        status=record.status,
        audio_uri=record.audio_uri,
        content_sha1=record.content_sha1,
        title=record.title,
        source_url=record.source_url,
        lang=record.lang,
        meta=record.meta,
        duration_ms=record.duration_ms,
        size_bytes=record.size_bytes,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.post(
    "",
    response_model=CreateRawAudioResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_raw_audio(
    response: Response,
    service: IngestServiceDependency,
    audio: Annotated[UploadFile, File()],
    title: Annotated[str | None, Form()] = None,
    source_url: Annotated[str | None, Form()] = None,
    lang: Annotated[str, Form()] = "en",
    meta: Annotated[str | None, Form()] = None,
) -> CreateRawAudioResponse:
    """Accept, normalize, persist, and enqueue one podcast audio upload."""
    if not lang.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="lang must not be empty.",
        )

    try:
        result = service.ingest(
            IngestRequest(
                source=audio.file,
                filename=audio.filename,
                title=title,
                source_url=source_url,
                lang=lang,
                meta=_parse_meta(meta),
            )
        )
    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except EmptyUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AudioNormalizationTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except AudioNormalizationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except ObjectStorageUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except RawAudioRepositoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Raw audio persistence is unavailable.",
        ) from exc
    except IngestTaskPublicationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": str(exc),
                "raw_audio_id": str(exc.raw_audio_id),
            },
        ) from exc

    response.status_code = (
        status.HTTP_200_OK if result.deduplicated else status.HTTP_202_ACCEPTED
    )
    if result.record.content_sha1 is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stored raw audio has no content digest.",
        )
    return CreateRawAudioResponse(
        id=result.record.id,
        status=result.record.status,
        content_sha1=result.record.content_sha1,
        task_id=result.task_id,
        deduplicated=result.deduplicated,
    )


@router.get("/{raw_audio_id}", response_model=RawAudioResponse)
def get_raw_audio(
    raw_audio_id: UUID,
    repository: RawAudioRepositoryDependency,
) -> RawAudioResponse:
    """Return one raw audio without changing pipeline state."""
    try:
        record = repository.get(raw_audio_id)
    except RawAudioRepositoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Raw audio persistence is unavailable.",
        ) from exc
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Raw audio not found.",
        )
    return _raw_audio_response(record)
