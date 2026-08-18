from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ProbeResponse(BaseModel):
    """Response returned by service probes."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "ready"]


class TriggerTaskRequest(BaseModel):
    """Request to publish a registered task with one UUID argument."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_name: str
    id: UUID


class TriggerTaskResponse(BaseModel):
    """Accepted registered task publication."""

    model_config = ConfigDict(frozen=True)

    task_name: str
    id: UUID
    task_id: str


class CreateRawAudioResponse(BaseModel):
    """Response for newly accepted or deduplicated audio."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    status: str
    content_sha1: str
    task_id: str | None
    deduplicated: bool


class RawAudioResponse(BaseModel):
    """Read-only representation of a raw_audios row."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    status: str
    audio_uri: str | None
    content_sha1: str | None
    title: str | None
    source_url: str | None
    lang: str
    meta: dict[str, Any]
    duration_ms: int | None
    size_bytes: int | None
    error: str | None
    created_at: datetime
    updated_at: datetime
