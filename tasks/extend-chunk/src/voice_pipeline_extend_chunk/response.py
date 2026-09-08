from __future__ import annotations

import math

from voice_pipeline_chunk_contracts import TaggedTextError, parse_text_with_audio_tags

from .interaction import canonical_backchannel, planned_counts, spoken_token_count

_COUNT_TOLERANCE = 0.30
_WIRE_FIELDS = {
    "utterance_index",
    "speaker_id",
    "text_with_audio_tags",
    "instruction",
    "type",
    "placement",
}


class CorrectionNeeded(Exception):
    def __init__(self, code: str, location: str, requirement: str):
        self.code = code
        self.location = location
        self.requirement = requirement


def normalize_response(wire: object, policy, *, interaction_targets=None) -> dict:
    utterances = _validate_envelope(wire, policy)
    normalized = [
        _normalize_utterance(raw, index) for index, raw in enumerate(utterances)
    ]
    _validate_conversation(normalized)
    if interaction_targets is not None:
        _validate_interaction_plan(normalized, interaction_targets)
    return {"utterances": normalized}


def _validate_envelope(wire: object, policy) -> list[dict]:
    if not isinstance(wire, dict) or set(wire) != {"utterances"}:
        raise CorrectionNeeded(
            "response_shape_invalid",
            "response",
            "Return exactly one object containing only the utterances array.",
        )
    utterances = wire["utterances"]
    if (
        not isinstance(utterances, list)
        or not policy.min_utterances <= len(utterances) <= policy.max_utterances
    ):
        raise CorrectionNeeded(
            "utterance_count_invalid",
            "utterances",
            "Return an utterance array that follows the supplied schema.",
        )
    return utterances


def _normalize_utterance(raw: object, index: int) -> dict:
    location = f"utterance_index={index}"
    if not isinstance(raw, dict):
        raise CorrectionNeeded(
            "response_shape_invalid", location, "Every utterance must be an object."
        )
    if "text" in raw:
        raise CorrectionNeeded(
            "forbidden_text_field",
            location,
            "Do not output text; output text_with_audio_tags and let the application derive text.",
        )
    if _WIRE_FIELDS - set(raw):
        raise CorrectionNeeded(
            "required_field_missing",
            location,
            "Include every required utterance field.",
        )
    if set(raw) != _WIRE_FIELDS:
        raise CorrectionNeeded(
            "response_shape_invalid",
            location,
            "Do not include fields outside the required utterance schema.",
        )
    if (
        isinstance(raw["utterance_index"], bool)
        or not isinstance(raw["utterance_index"], int)
        or raw["utterance_index"] != index
    ):
        raise CorrectionNeeded(
            "utterance_index_invalid",
            location,
            "utterance_index must start at zero and increase by one without gaps.",
        )

    tagged_text = raw["text_with_audio_tags"]
    if raw.get("type") == "backchannel":
        tagged_text = canonical_backchannel(index)
    try:
        tagged = parse_text_with_audio_tags(tagged_text)
    except TaggedTextError as exc:
        raise CorrectionNeeded(exc.code, location, exc.requirement) from exc

    instruction = raw["instruction"]
    if (
        not isinstance(instruction, str)
        or not instruction
        or instruction != instruction.strip()
        or "[" in instruction
        or "]" in instruction
    ):
        raise CorrectionNeeded(
            "instruction_invalid",
            location,
            "instruction must be one non-empty concise sentence with no square-bracket tags.",
        )
    utterance_type = raw["type"]
    placement = raw["placement"]
    speaker_id = raw["speaker_id"]
    if (
        isinstance(speaker_id, bool)
        or not isinstance(speaker_id, int)
        or speaker_id not in {0, 1}
    ):
        raise CorrectionNeeded(
            "speaker_id_invalid", location, "speaker_id must be the integer 0 or 1."
        )
    if utterance_type not in {"dialogue", "backchannel", "paralinguistic"}:
        raise CorrectionNeeded(
            "utterance_type_invalid",
            location,
            "type must be dialogue, backchannel, or paralinguistic.",
        )
    if placement not in {"sequential", "overlap_previous"}:
        raise CorrectionNeeded(
            "placement_invalid",
            location,
            "placement must be sequential or overlap_previous.",
        )
    if utterance_type != "paralinguistic" and not tagged.text:
        raise CorrectionNeeded(
            "spoken_text_empty",
            location,
            "Dialogue and backchannel utterances must contain non-empty spoken text after audio tags are removed.",
        )
    if utterance_type == "paralinguistic" and not tagged.tags:
        raise CorrectionNeeded(
            "paralinguistic_audio_tag_missing",
            location,
            "A paralinguistic utterance must contain at least one approved audio tag.",
        )
    if utterance_type == "backchannel" and placement != "overlap_previous":
        raise CorrectionNeeded(
            "backchannel_placement_invalid",
            location,
            "Every backchannel must use overlap_previous so the listener yields the floor.",
        )
    return {**raw, "text_with_audio_tags": tagged_text, "text": tagged.text}


def _validate_conversation(utterances: list[dict]) -> None:
    if {item["speaker_id"] for item in utterances} != {0, 1}:
        raise CorrectionNeeded(
            "both_speakers_required",
            "utterances",
            "Both fixed speakers 0 and 1 must appear.",
        )
    if utterances[0]["placement"] != "sequential":
        raise CorrectionNeeded(
            "first_placement_invalid",
            "utterance_index=0",
            "The first utterance must be sequential.",
        )
    for previous, current in zip(utterances, utterances[1:]):
        if current["placement"] != "overlap_previous":
            continue
        location = f"utterance_index={current['utterance_index']}"
        if current["speaker_id"] == previous["speaker_id"]:
            raise CorrectionNeeded(
                "self_overlap_invalid",
                location,
                "A speaker must not overlap its own previous utterance.",
            )
        if previous["type"] != "dialogue" or previous["placement"] != "sequential":
            raise CorrectionNeeded(
                "overlap_anchor_invalid",
                location,
                "Every overlap must immediately follow a sequential substantive dialogue anchor; never place overlaps consecutively.",
            )
        if spoken_token_count(previous["text"]) < 5:
            raise CorrectionNeeded(
                "overlap_anchor_too_short",
                location,
                "The preceding dialogue anchor needs more spoken context before the overlap begins.",
            )
        limit = 4 if current["type"] == "backchannel" else 12
        if spoken_token_count(current["text"]) > limit:
            raise CorrectionNeeded(
                "overlap_response_too_long",
                location,
                "Keep the overlapping response concise.",
            )


def _validate_interaction_plan(utterances: list[dict], interaction_targets) -> None:
    actual = planned_counts(utterances)
    expected = (
        interaction_targets.target_turn_count,
        interaction_targets.target_backchannel_count,
        interaction_targets.target_overlap_event_count,
    )
    for name, observed, target in zip(
        ("turn", "backchannel", "overlap"), actual, expected, strict=True
    ):
        lower, upper = _count_bounds(target)
        if not lower <= observed <= upper:
            raise CorrectionNeeded(
                f"{name}_target_missed",
                "utterances",
                "Regenerate the complete plan and match every supplied interaction target.",
            )


def _count_bounds(target: int) -> tuple[int, int]:
    lower = math.ceil(target * (1 - _COUNT_TOLERANCE))
    upper = max(
        1 if target == 0 else target, math.ceil(target * (1 + _COUNT_TOLERANCE))
    )
    return lower, upper
