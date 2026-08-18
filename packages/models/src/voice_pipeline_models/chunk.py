from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .audio_part import AudioPart


class Chunk(Base):
    """A clean two-speaker dialogue segment selected from an audio part."""

    __tablename__ = "chunks"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    audio_part_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("audio_parts.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'pending'")
    )
    audio_uri: Mapped[str] = mapped_column(Text, nullable=False)
    lang: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'en'"))
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    relative_start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    relative_end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    diarization_model: Mapped[str | None] = mapped_column(Text)
    diarizations: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    persona: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    final_results: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    audio_part: Mapped[AudioPart] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("audio_part_id", "chunk_index"),
        CheckConstraint("chunk_index >= 0"),
        CheckConstraint("relative_start_ms >= 0"),
        CheckConstraint("relative_end_ms > relative_start_ms"),
        CheckConstraint("duration_ms > 0"),
        CheckConstraint("duration_ms = relative_end_ms - relative_start_ms"),
        Index(
            "idx_chunks_status_created",
            "status",
            created_at.desc(),
        ),
    )
