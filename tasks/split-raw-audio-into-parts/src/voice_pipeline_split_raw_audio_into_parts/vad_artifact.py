from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from .wav_io import frame_to_milliseconds
from .windowing import FrameSpan

SCHEMA_VERSION = 1


class VadSegmentDocument(TypedDict):
    index: int
    start_ms: int
    end_ms: int
    duration_ms: int


class VadArtifactDocument(TypedDict):
    schema_version: int
    model: str
    audio_duration_ms: int
    segments: list[VadSegmentDocument]


def build_vad_artifact(
    *,
    model: str,
    audio_frame_count: int,
    segments: list[FrameSpan],
) -> VadArtifactDocument:
    """Build the stable JSON-compatible document for normalized VAD output."""

    if not model.strip():
        raise ValueError("model must not be empty")
    if audio_frame_count < 0:
        raise ValueError("audio_frame_count must not be negative")

    ordered = sorted(segments)
    previous_end = 0
    document_segments: list[VadSegmentDocument] = []
    for index, segment in enumerate(ordered):
        if segment.start_frame < previous_end:
            raise ValueError("segments must not overlap")
        if segment.end_frame > audio_frame_count:
            raise ValueError("segment exceeds audio_frame_count")

        start_ms = frame_to_milliseconds(segment.start_frame)
        end_ms = frame_to_milliseconds(segment.end_frame)
        document_segments.append(
            {
                "index": index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
            }
        )
        previous_end = segment.end_frame

    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "audio_duration_ms": frame_to_milliseconds(audio_frame_count),
        "segments": document_segments,
    }


def serialize_vad_artifact(document: VadArtifactDocument) -> bytes:
    """Serialize with stable field order, separators, encoding, and newline."""

    return (
        json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def write_vad_artifact(path: Path, document: VadArtifactDocument) -> None:
    """Write one local artifact ready for deterministic object storage upload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize_vad_artifact(document))
