"""Compatibility imports for the shared diarization artifact contract."""

from voice_pipeline_diarization_artifact import (
    DiarizationArtifact,
    RawTurn,
    Segment,
    SpeakerSummary,
    build_artifact,
)

__all__ = [
    "DiarizationArtifact",
    "RawTurn",
    "Segment",
    "SpeakerSummary",
    "build_artifact",
]
