from __future__ import annotations

from dataclasses import dataclass

from .contracts import GroupDescriptor
from .errors import ScoringError


@dataclass(frozen=True, slots=True)
class InteractionUtterance:
    utterance_index: int
    speaker_id: int
    start_ms: int
    end_ms: int
    text: str
    source_index: int | None = None
    declared_type: str | None = None
    declared_placement: str | None = None
    declared_relation: str | None = None
    text_with_audio_tags: str | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScoringError("invalid_interaction_transcript", name)
    return value


def _string(value: object, name: str, *, empty: bool = True) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or (not empty and not value)
    ):
        raise ScoringError("invalid_interaction_transcript", name)
    return value


def normalize_interaction_utterances(
    transcript: dict,
    *,
    group: GroupDescriptor,
) -> tuple[InteractionUtterance, ...]:
    raw_utterances = transcript.get("utterances")
    if not isinstance(raw_utterances, list) or not raw_utterances:
        raise ScoringError("invalid_interaction_transcript", "utterances")
    materialized: list[InteractionUtterance] = []
    for fallback_index, raw in enumerate(raw_utterances):
        if not isinstance(raw, dict):
            raise ScoringError("invalid_interaction_transcript", "utterance")
        index = raw.get("utterance_index", fallback_index)
        index = _integer(index, "utterance_index")
        speaker_id = _integer(raw.get("speaker_id"), "speaker_id")
        start_ms = _integer(raw.get("start_ms"), "start_ms")
        end_ms = _integer(raw.get("end_ms"), "end_ms", minimum=1)
        if speaker_id not in {0, 1} or end_ms <= start_ms or end_ms > group.duration_ms:
            raise ScoringError("invalid_interaction_transcript", "interval")
        text = _string(raw.get("text", ""), "text")
        declared_type = raw.get("type")
        declared_placement = raw.get("placement")
        declared_relation = raw.get("relation")
        if declared_relation is not None and declared_relation not in {
            "leading",
            "gap",
            "overlap",
            "simultaneous",
        }:
            raise ScoringError("invalid_interaction_transcript", "relation")
        if declared_placement is None and declared_relation is not None:
            declared_placement = (
                "overlap_previous"
                if declared_relation in {"overlap", "simultaneous"}
                else "sequential"
            )
        tagged = raw.get("text_with_audio_tags")
        if declared_type is not None and declared_type not in {
            "dialogue",
            "backchannel",
            "paralinguistic",
        }:
            raise ScoringError("invalid_interaction_transcript", "type")
        if declared_placement is not None and declared_placement not in {
            "sequential",
            "overlap_previous",
        }:
            raise ScoringError("invalid_interaction_transcript", "placement")
        if tagged is not None and not isinstance(tagged, str):
            raise ScoringError("invalid_interaction_transcript", "text_with_audio_tags")
        materialized.append(
            InteractionUtterance(
                utterance_index=index,
                speaker_id=speaker_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                declared_type=declared_type,
                declared_placement=declared_placement,
                declared_relation=declared_relation,
                text_with_audio_tags=tagged,
            )
        )
    ordered = sorted(
        materialized,
        key=lambda item: (
            item.start_ms,
            item.end_ms,
            item.speaker_id,
            item.utterance_index,
        ),
    )
    if group.group != "separation" and [
        item.utterance_index for item in materialized
    ] != list(range(len(materialized))):
        raise ScoringError("invalid_interaction_transcript", "utterance_index")
    return tuple(ordered)


def attach_source_indices(
    source: tuple[InteractionUtterance, ...],
    reconstruction: tuple[InteractionUtterance, ...],
    raw_reconstruction: dict,
) -> tuple[InteractionUtterance, ...]:
    """Map reconstruction utterances to canonical source intervals exactly."""

    source_by_identity = {
        (item.speaker_id, item.start_ms, item.end_ms): source_index
        for source_index, item in enumerate(source)
    }
    raw_utterances = raw_reconstruction.get("utterances")
    if not isinstance(raw_utterances, list) or len(raw_utterances) != len(
        reconstruction
    ):
        raise ScoringError("invalid_reconstruction_source_mapping")
    raw_by_index = {
        _integer(item.get("utterance_index"), "utterance_index"): item
        for item in raw_utterances
        if isinstance(item, dict)
    }
    mapped: list[InteractionUtterance] = []
    used: set[int] = set()
    for item in reconstruction:
        raw = raw_by_index.get(item.utterance_index)
        if raw is None:
            raise ScoringError("invalid_reconstruction_source_mapping")
        source_start = _integer(raw.get("source_start_ms"), "source_start_ms")
        source_end = _integer(raw.get("source_end_ms"), "source_end_ms", minimum=1)
        source_index = source_by_identity.get(
            (item.speaker_id, source_start, source_end)
        )
        if source_index is None or source_index in used:
            raise ScoringError("invalid_reconstruction_source_mapping")
        used.add(source_index)
        mapped.append(
            InteractionUtterance(
                utterance_index=item.utterance_index,
                speaker_id=item.speaker_id,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                text=item.text,
                source_index=source_index,
                declared_type=item.declared_type,
                declared_placement=item.declared_placement,
                declared_relation=item.declared_relation,
                text_with_audio_tags=item.text_with_audio_tags,
            )
        )
    return tuple(mapped)


def source_with_indices(
    source: tuple[InteractionUtterance, ...],
) -> tuple[InteractionUtterance, ...]:
    return tuple(
        InteractionUtterance(
            utterance_index=item.utterance_index,
            speaker_id=item.speaker_id,
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            text=item.text,
            source_index=index,
            declared_type=item.declared_type,
            declared_placement=item.declared_placement,
            declared_relation=item.declared_relation,
            text_with_audio_tags=item.text_with_audio_tags,
        )
        for index, item in enumerate(source)
    )
