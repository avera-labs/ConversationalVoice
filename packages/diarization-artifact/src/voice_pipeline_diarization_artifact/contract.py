"""Strict version 1 diarization artifact writer and reader."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_MILLISECOND = Decimal("0.001")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "model",
    "audio_duration_seconds",
    "num_speakers",
    "total_speech_seconds",
    "segments",
    "speaker_summary",
}
_SEGMENT_FIELDS = {"speaker", "start", "end", "duration"}
_SUMMARY_FIELDS = {"speaker", "total_seconds", "percentage"}


class DiarizationArtifactError(ValueError):
    """Raised when a diarization artifact violates the shared contract."""


@dataclass(frozen=True, slots=True)
class RawTurn:
    start: float
    end: float
    speaker_label: str


@dataclass(frozen=True, slots=True)
class Segment:
    speaker: int
    start: float
    end: float
    duration: float


@dataclass(frozen=True, slots=True)
class SpeakerSummary:
    speaker: int
    total_seconds: float
    percentage: float


@dataclass(frozen=True, slots=True)
class DiarizationArtifact:
    schema_version: int
    model: str
    audio_duration_seconds: float
    num_speakers: int
    total_speech_seconds: float
    segments: tuple[Segment, ...]
    speaker_summary: tuple[SpeakerSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "audio_duration_seconds": self.audio_duration_seconds,
            "num_speakers": self.num_speakers,
            "total_speech_seconds": self.total_speech_seconds,
            "segments": [asdict(segment) for segment in self.segments],
            "speaker_summary": [asdict(summary) for summary in self.speaker_summary],
        }

    def to_json_bytes(self) -> bytes:
        return (json.dumps(self.to_dict(), ensure_ascii=True, indent=2) + "\n").encode(
            "utf-8"
        )

    def write(self, path: Path) -> int:
        payload = self.to_json_bytes()
        path.write_bytes(payload)
        return len(payload)


@dataclass(frozen=True, slots=True, order=True)
class DiarizationTurn:
    start_ms: int
    end_ms: int
    speaker: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class ParsedDiarizationArtifact:
    model: str
    duration_ms: int
    turns: tuple[DiarizationTurn, ...]


def _milliseconds_decimal(value: object, *, field: str) -> tuple[int, Decimal]:
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DiarizationArtifactError(f"{field} must be a decimal number") from exc
    if not number.is_finite():
        raise DiarizationArtifactError(f"{field} must be finite")
    milliseconds = number * 1000
    if milliseconds != milliseconds.to_integral_value():
        raise DiarizationArtifactError(f"{field} must have millisecond precision")
    return int(milliseconds), number


def _quantized_seconds(value: float) -> float:
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("timestamp must be finite")
    return float(number.quantize(_MILLISECOND, rounding=ROUND_HALF_UP))


def build_artifact(
    turns: Iterable[RawTurn], *, model: str, duration_ms: int
) -> DiarizationArtifact:
    if not isinstance(model, str) or not model.strip() or duration_ms <= 0:
        raise ValueError("invalid artifact metadata")
    duration_seconds = duration_ms / 1000.0
    checked: list[RawTurn] = []
    for turn in turns:
        start = float(turn.start)
        end = float(turn.end)
        label = turn.speaker_label
        if (
            not isinstance(label, str)
            or not label
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or end > duration_seconds + 0.0005
        ):
            raise ValueError("invalid diarization turn")
        checked.append(RawTurn(start=start, end=end, speaker_label=label))

    checked.sort(key=lambda item: (item.start, item.end, item.speaker_label))
    label_ids: dict[str, int] = {}
    segments: list[Segment] = []
    totals_ms: dict[int, int] = {}
    for turn in checked:
        speaker = label_ids.setdefault(turn.speaker_label, len(label_ids))
        start = _quantized_seconds(turn.start)
        end = _quantized_seconds(turn.end)
        start_ms = int(Decimal(str(start)) * 1000)
        end_ms = int(Decimal(str(end)) * 1000)
        if end_ms <= start_ms or end_ms > duration_ms:
            raise ValueError("turn is empty or out of bounds after normalization")
        segment_duration = (end_ms - start_ms) / 1000.0
        segments.append(Segment(speaker, start, end, segment_duration))
        totals_ms[speaker] = totals_ms.get(speaker, 0) + end_ms - start_ms

    total_speech_ms = sum(totals_ms.values())
    summaries = tuple(
        SpeakerSummary(
            speaker=speaker,
            total_seconds=round(total_ms / 1000.0, 2),
            percentage=round(
                (total_ms / total_speech_ms * 100.0) if total_speech_ms else 0.0,
                1,
            ),
        )
        for speaker, total_ms in sorted(totals_ms.items())
    )
    return DiarizationArtifact(
        schema_version=SCHEMA_VERSION,
        model=model.strip(),
        audio_duration_seconds=duration_ms / 1000.0,
        num_speakers=len(label_ids),
        total_speech_seconds=total_speech_ms / 1000.0,
        segments=tuple(segments),
        speaker_summary=summaries,
    )


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiarizationArtifactError(f"{field} must be an object")
    return value


def _require_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DiarizationArtifactError(f"{field} must be an integer")
    return value


def parse_artifact_bytes(
    payload: bytes, *, expected_duration_ms: int
) -> ParsedDiarizationArtifact:
    if expected_duration_ms <= 0 or not payload:
        raise DiarizationArtifactError("artifact metadata is invalid")
    try:
        document = json.loads(payload, parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiarizationArtifactError("artifact is not valid JSON") from exc
    root = _require_mapping(document, field="artifact")
    if set(root) != _TOP_LEVEL_FIELDS:
        raise DiarizationArtifactError("artifact fields do not match version 1")
    if root["schema_version"] != SCHEMA_VERSION:
        raise DiarizationArtifactError("artifact schema version is unsupported")
    model = root["model"]
    if not isinstance(model, str) or not model.strip():
        raise DiarizationArtifactError("model must be a non-empty string")
    duration_ms, _ = _milliseconds_decimal(
        root["audio_duration_seconds"], field="audio_duration_seconds"
    )
    if duration_ms != expected_duration_ms:
        raise DiarizationArtifactError("artifact duration does not match the database")

    raw_segments = root["segments"]
    raw_summaries = root["speaker_summary"]
    if not isinstance(raw_segments, list) or not isinstance(raw_summaries, list):
        raise DiarizationArtifactError("artifact collections must be arrays")

    turns: list[DiarizationTurn] = []
    totals_ms: dict[int, int] = {}
    previous: tuple[int, int, int] | None = None
    for index, raw in enumerate(raw_segments):
        item = _require_mapping(raw, field=f"segments[{index}]")
        if set(item) != _SEGMENT_FIELDS:
            raise DiarizationArtifactError("segment fields are invalid")
        speaker = _require_int(item["speaker"], field="speaker")
        start_ms, _ = _milliseconds_decimal(item["start"], field="start")
        end_ms, _ = _milliseconds_decimal(item["end"], field="end")
        segment_duration_ms, _ = _milliseconds_decimal(
            item["duration"], field="duration"
        )
        if start_ms < 0 or end_ms <= start_ms or end_ms > expected_duration_ms:
            raise DiarizationArtifactError("segment bounds are invalid")
        if segment_duration_ms != end_ms - start_ms:
            raise DiarizationArtifactError("segment duration is inconsistent")
        ordering = (start_ms, end_ms, speaker)
        if previous is not None and ordering < previous:
            raise DiarizationArtifactError("segments are not sorted")
        previous = ordering
        turns.append(DiarizationTurn(start_ms, end_ms, speaker))
        totals_ms[speaker] = totals_ms.get(speaker, 0) + segment_duration_ms

    speakers = sorted(totals_ms)
    if speakers != list(range(len(speakers))):
        raise DiarizationArtifactError("speaker identifiers must be contiguous")
    if _require_int(root["num_speakers"], field="num_speakers") != len(speakers):
        raise DiarizationArtifactError("num_speakers is inconsistent")
    total_speech_ms, _ = _milliseconds_decimal(
        root["total_speech_seconds"], field="total_speech_seconds"
    )
    if total_speech_ms != sum(totals_ms.values()):
        raise DiarizationArtifactError("total_speech_seconds is inconsistent")
    if len(raw_summaries) != len(speakers):
        raise DiarizationArtifactError("speaker_summary is inconsistent")
    for expected_speaker, raw in zip(speakers, raw_summaries, strict=True):
        item = _require_mapping(raw, field="speaker_summary")
        if set(item) != _SUMMARY_FIELDS:
            raise DiarizationArtifactError("speaker summary fields are invalid")
        if _require_int(item["speaker"], field="speaker") != expected_speaker:
            raise DiarizationArtifactError("speaker_summary is not sorted")
        expected_total = Decimal(totals_ms[expected_speaker]) / Decimal(1000)
        expected_total = expected_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        actual_total = item["total_seconds"]
        actual_total = actual_total if isinstance(actual_total, Decimal) else Decimal(str(actual_total))
        if actual_total != expected_total:
            raise DiarizationArtifactError("speaker total is inconsistent")
        expected_percentage = Decimal("0")
        if total_speech_ms:
            expected_percentage = (
                Decimal(totals_ms[expected_speaker])
                / Decimal(total_speech_ms)
                * Decimal(100)
            ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        actual_percentage = item["percentage"]
        actual_percentage = actual_percentage if isinstance(actual_percentage, Decimal) else Decimal(str(actual_percentage))
        if not actual_percentage.is_finite() or actual_percentage != expected_percentage:
            raise DiarizationArtifactError("speaker percentage is inconsistent")

    return ParsedDiarizationArtifact(model.strip(), duration_ms, tuple(turns))
