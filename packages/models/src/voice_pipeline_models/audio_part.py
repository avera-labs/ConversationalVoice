from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
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
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .chunk import Chunk
    from .raw_audio import RawAudio


class AudioPart(Base):
    """A VAD-selected conversation window from a raw audio source."""

    __tablename__ = "audio_parts"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    raw_audio_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("raw_audios.id", ondelete="CASCADE"),
        nullable=False,
    )
    part_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'pending'")
    )
    audio_uri: Mapped[str] = mapped_column(Text, nullable=False)
    diarization_uri: Mapped[str | None] = mapped_column(Text)
    lang: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'en'")
    )
    relative_start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    relative_end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

    raw_audio: Mapped[RawAudio] = relationship(back_populates="audio_parts")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="audio_part",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        UniqueConstraint("raw_audio_id", "part_index"),
        CheckConstraint("part_index >= 0"),
        CheckConstraint(
            "status IN ('pending', 'diarizing', 'diarized', "
            "'filtering', 'completed', 'failed')",
            name="ck_audio_parts_status",
        ),
        CheckConstraint("relative_end_ms > relative_start_ms"),
        Index(
            "idx_audio_parts_status_created",
            "status",
            created_at.desc(),
        ),
    )

