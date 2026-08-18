"""Quality-filter view of the shared diarization contract."""

from voice_pipeline_diarization_artifact import (
    DiarizationArtifactError,
    DiarizationTurn,
    ParsedDiarizationArtifact,
    parse_artifact_bytes,
)

from .intervals import Interval, merge_intervals


def speech_union(turns: tuple[DiarizationTurn, ...]) -> tuple[Interval, ...]:
    return merge_intervals(tuple(Interval(turn.start_ms, turn.end_ms) for turn in turns))


__all__ = [
    "DiarizationArtifactError",
    "DiarizationTurn",
    "ParsedDiarizationArtifact",
    "parse_artifact_bytes",
    "speech_union",
]
