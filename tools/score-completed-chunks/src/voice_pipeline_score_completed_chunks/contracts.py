from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ScoringError
from .storage import StoredObject, parse_identity

GROUPS = ("separation", "reconstruction", "expansion")


@dataclass(frozen=True, slots=True)
class TrackDescriptor:
    speaker_id: int
    diarization_speaker_id: int
    sample_rate_hz: int
    duration_ms: int
    artifact: StoredObject


@dataclass(frozen=True, slots=True)
class GroupDescriptor:
    group: str
    language: str
    duration_ms: int
    transcript: StoredObject
    tracks: tuple[TrackDescriptor, TrackDescriptor]


@dataclass(frozen=True, slots=True)
class ReferenceDescriptor:
    speaker_id: int
    diarization_speaker_id: int
    source: str
    source_audio: StoredObject
    selection: tuple[tuple[int, int], ...]
    sample_rate_hz: int
    duration_ms: int
    size_bytes: int
    sha256: str


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScoringError("invalid_completed_contract", f"{name} is invalid")
    return value


def parse_group(final_results: dict[str, Any], group: str) -> GroupDescriptor:
    if group not in GROUPS:
        raise ScoringError("invalid_group", group)
    if group == "separation":
        return _parse_separation_group(final_results)
    key = "reconstruction" if group == "reconstruction" else "dialogue_extension"
    result = final_results.get(key)
    if not isinstance(result, dict):
        raise ScoringError("missing_completed_namespace", key)
    language = result.get("language")
    duration_ms = _positive_integer(result.get("actual_duration_ms"), "duration")
    artifacts = result.get("artifacts")
    if not isinstance(language, str) or not language or not isinstance(artifacts, dict):
        raise ScoringError("invalid_completed_contract", key)
    transcript = parse_identity(artifacts.get("transcript"), name=f"{group} transcript")
    raw_tracks = artifacts.get("speaker_audio")
    if not isinstance(raw_tracks, list) or len(raw_tracks) != 2:
        raise ScoringError("invalid_speaker_tracks", group)
    tracks: list[TrackDescriptor] = []
    for slot, raw in enumerate(raw_tracks):
        if not isinstance(raw, dict):
            raise ScoringError("invalid_speaker_tracks", group)
        speaker_id = raw.get("speaker_id")
        diarization_id = raw.get("diarization_speaker_id")
        sample_rate = raw.get("sample_rate_hz")
        track_duration = raw.get("duration_ms")
        if (
            speaker_id != slot
            or isinstance(diarization_id, bool)
            or not isinstance(diarization_id, int)
            or diarization_id < 0
            or sample_rate != 44100
            or track_duration != duration_ms
        ):
            raise ScoringError("invalid_speaker_tracks", group)
        tracks.append(
            TrackDescriptor(
                slot,
                diarization_id,
                sample_rate,
                duration_ms,
                parse_identity(raw, name=f"{group} speaker {slot}"),
            )
        )
    if tracks[0].diarization_speaker_id == tracks[1].diarization_speaker_id:
        raise ScoringError("invalid_speaker_mapping", group)
    return GroupDescriptor(
        group, language, duration_ms, transcript, (tracks[0], tracks[1])
    )


def _parse_separation_group(final_results: dict[str, Any]) -> GroupDescriptor:
    result = final_results.get("separation")
    transcription = final_results.get("transcription")
    if not isinstance(result, dict):
        raise ScoringError("missing_completed_namespace", "separation")
    if not isinstance(transcription, dict):
        raise ScoringError("missing_completed_namespace", "transcription")
    language = transcription.get("language")
    artifacts = transcription.get("artifacts")
    raw_tracks = result.get("speaker_audio")
    raw_inputs = transcription.get("input_speaker_audio")
    if (
        not isinstance(language, str)
        or not language
        or not isinstance(artifacts, dict)
        or not isinstance(raw_tracks, list)
        or len(raw_tracks) != 2
        or not isinstance(raw_inputs, list)
        or len(raw_inputs) != 2
    ):
        raise ScoringError("invalid_completed_contract", "separation")
    transcript = parse_identity(
        artifacts.get("transcript"), name="separation transcript"
    )
    tracks: list[TrackDescriptor] = []
    duration_ms: int | None = None
    for slot, (raw, transcription_input) in enumerate(
        zip(raw_tracks, raw_inputs, strict=True)
    ):
        if not isinstance(raw, dict) or not isinstance(transcription_input, dict):
            raise ScoringError("invalid_speaker_tracks", "separation")
        diarization_id = raw.get("diarization_speaker_id")
        track_duration = raw.get("duration_ms")
        if (
            raw.get("output_slot") != slot
            or isinstance(diarization_id, bool)
            or not isinstance(diarization_id, int)
            or diarization_id < 0
            or raw.get("sample_rate_hz") != 16000
            or isinstance(track_duration, bool)
            or not isinstance(track_duration, int)
            or track_duration <= 0
        ):
            raise ScoringError("invalid_speaker_tracks", "separation")
        artifact = parse_identity(raw, name=f"separation speaker {slot}")
        if transcription_input != {
            "output_slot": slot,
            "diarization_speaker_id": diarization_id,
            "uri": artifact.uri,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
        }:
            raise ScoringError("invalid_transcription_input", "separation")
        if duration_ms is None:
            duration_ms = track_duration
        elif track_duration != duration_ms:
            raise ScoringError("track_duration_mismatch", "separation")
        tracks.append(
            TrackDescriptor(
                slot,
                diarization_id,
                16000,
                track_duration,
                artifact,
            )
        )
    if tracks[0].diarization_speaker_id == tracks[1].diarization_speaker_id:
        raise ScoringError("invalid_speaker_mapping", "separation")
    assert duration_ms is not None
    return GroupDescriptor(
        "separation", language, duration_ms, transcript, (tracks[0], tracks[1])
    )


def parse_references(
    final_results: dict[str, Any],
) -> tuple[ReferenceDescriptor, ReferenceDescriptor]:
    extension = final_results.get("dialogue_extension")
    inputs = extension.get("inputs") if isinstance(extension, dict) else None
    raw_references = (
        inputs.get("speaker_references") if isinstance(inputs, dict) else None
    )
    if not isinstance(raw_references, list) or len(raw_references) != 2:
        raise ScoringError("invalid_speaker_references")
    references: list[ReferenceDescriptor] = []
    for slot, raw in enumerate(raw_references):
        if not isinstance(raw, dict):
            raise ScoringError("invalid_speaker_references")
        diarization_id = raw.get("diarization_speaker_id")
        source = raw.get("source")
        reference_audio = raw.get("reference_audio")
        selection = raw.get("selection")
        if (
            raw.get("speaker_id") != slot
            or isinstance(diarization_id, bool)
            or not isinstance(diarization_id, int)
            or diarization_id < 0
            or source not in {"diarization_reference", "separated_track_slice"}
            or not isinstance(reference_audio, dict)
            or not isinstance(selection, dict)
        ):
            raise ScoringError("invalid_speaker_references")
        raw_segments = selection.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ScoringError("invalid_reference_selection")
        segments: list[tuple[int, int]] = []
        for item in raw_segments:
            if not isinstance(item, dict):
                raise ScoringError("invalid_reference_selection")
            start, end = item.get("start_ms"), item.get("end_ms")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end <= start
            ):
                raise ScoringError("invalid_reference_selection")
            segments.append((start, end))
        sample_rate = reference_audio.get("sample_rate_hz")
        duration = reference_audio.get("duration_ms")
        size = reference_audio.get("size_bytes")
        sha256 = reference_audio.get("sha256")
        if (
            sample_rate != 16000
            or not isinstance(duration, int)
            or duration <= 0
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ScoringError("invalid_speaker_references")
        references.append(
            ReferenceDescriptor(
                slot,
                diarization_id,
                source,
                parse_identity(
                    raw.get("source_audio"), name=f"reference source {slot}"
                ),
                tuple(segments),
                sample_rate,
                duration,
                size,
                sha256,
            )
        )
    return references[0], references[1]


def validate_transcript(
    value: object,
    *,
    group: GroupDescriptor,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScoringError("invalid_transcript")
    if group.group == "separation":
        return _validate_separation_transcript(value, group=group)
    expected_timebase = (
        "reconstruction" if group.group == "reconstruction" else "dialogue_extension"
    )
    if (
        value.get("language") != group.language
        or value.get("timebase") != expected_timebase
        or value.get("duration_ms") != group.duration_ms
        or not isinstance(value.get("utterances"), list)
    ):
        raise ScoringError("invalid_transcript_identity")
    mapping = value.get("speaker_mapping")
    if not isinstance(mapping, list) or len(mapping) != 2:
        raise ScoringError("invalid_transcript_mapping")
    for track, item in zip(group.tracks, mapping, strict=True):
        if not isinstance(item, dict) or item != {
            "speaker_id": track.speaker_id,
            "diarization_speaker_id": track.diarization_speaker_id,
        }:
            raise ScoringError("invalid_transcript_mapping")
    return value


def _validate_separation_transcript(
    value: dict[str, Any], *, group: GroupDescriptor
) -> dict[str, Any]:
    if value.get("language") != group.language or value.get("timebase") != "chunk":
        raise ScoringError("invalid_transcript_identity")
    speakers = value.get("speakers")
    if not isinstance(speakers, list) or len(speakers) != 2:
        raise ScoringError("invalid_transcript_mapping")
    utterances: list[dict[str, Any]] = []
    for track, speaker in zip(group.tracks, speakers, strict=True):
        if (
            not isinstance(speaker, dict)
            or speaker.get("output_slot") != track.speaker_id
            or speaker.get("diarization_speaker_id") != track.diarization_speaker_id
            or not isinstance(speaker.get("utterances"), list)
        ):
            raise ScoringError("invalid_transcript_mapping")
        for utterance in speaker["utterances"]:
            if not isinstance(utterance, dict):
                raise ScoringError("invalid_transcript")
            start, end = utterance.get("start_ms"), utterance.get("end_ms")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > group.duration_ms
            ):
                raise ScoringError("invalid_transcript_interval")
            utterances.append({**utterance, "speaker_id": track.speaker_id})
    normalized = dict(value)
    normalized["utterances"] = utterances
    return normalized
