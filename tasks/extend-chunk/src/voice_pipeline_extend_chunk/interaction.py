from __future__ import annotations

import re
from dataclasses import asdict, dataclass


_BACKCHANNEL_CUES = {
    "ah",
    "aha",
    "alright",
    "exactly",
    "good",
    "got it",
    "hmm",
    "mhm",
    "mm",
    "okay",
    "oh",
    "right",
    "sure",
    "uh huh",
    "uh-huh",
    "yeah",
    "yes",
    "嗯",
    "嗯嗯",
    "哦",
    "喔",
    "好",
    "好的",
    "对",
    "對",
    "没错",
    "沒錯",
    "是",
    "是的",
}
_WORD = re.compile(r"[\w-]+|[\u3400-\u9fff]", re.UNICODE)
_CANONICAL_BACKCHANNELS = (
    "Yeah.",
    "Right.",
    "Exactly.",
    "Okay.",
    "Mm-hmm.",
    "Uh-huh.",
)
_MINIMUM_OVERLAP_MS = 60
_OVERLAP_EVENT_MERGE_GAP_MS = 500


@dataclass(frozen=True, slots=True)
class InteractionTargets:
    reconstruction_effective_duration_ms: int
    reconstruction_turn_count: int
    reconstruction_backchannel_count: int
    reconstruction_overlap_event_count: int
    turns_per_minute: float
    backchannels_per_minute: float
    overlap_events_per_minute: float
    expansion_target_duration_ms: int
    target_turn_count: int
    target_backchannel_count: int
    target_overlap_event_count: int

    def prompt_payload(self) -> dict[str, int | float]:
        payload = asdict(self)
        for name in (
            "turns_per_minute",
            "backchannels_per_minute",
            "overlap_events_per_minute",
        ):
            payload[name] = round(float(payload[name]), 2)
        if self.target_overlap_event_count > 0:
            payload["target_overlap_spacing_seconds"] = round(
                self.expansion_target_duration_ms
                / self.target_overlap_event_count
                / 1000,
                2,
            )
            payload["target_overlap_anchor_interval"] = round(
                self.target_turn_count / self.target_overlap_event_count,
                2,
            )
        return payload


@dataclass(frozen=True, slots=True)
class _Turn:
    speaker_id: int
    start_ms: int
    end_ms: int
    texts: tuple[str, ...]


def derive_interaction_targets(transcript: dict, policy) -> InteractionTargets:
    """Estimate reconstruction interaction rates from transcript timing only.

    The production evaluator measures activity from audio. This intentionally cheap
    pre-generation estimate mirrors its turn and event definitions closely enough to
    condition the dialogue model without running VAD or loading an acoustic model.
    """

    utterances = sorted(
        transcript["utterances"],
        key=lambda item: (
            item["start_ms"],
            item["end_ms"],
            item["speaker_id"],
            item["utterance_index"],
        ),
    )
    effective_start = min(item["start_ms"] for item in utterances)
    effective_end = max(item["end_ms"] for item in utterances)
    effective_duration_ms = effective_end - effective_start
    if effective_duration_ms <= 0:
        raise ValueError("reconstruction interaction duration is invalid")

    turns = _build_turns(utterances)
    backchannel_count = _count_backchannels(turns)
    overlap_count = _count_overlap_events(utterances)
    minutes = effective_duration_ms / 60_000
    turns_per_minute = len(turns) / minutes
    backchannels_per_minute = backchannel_count / minutes
    overlaps_per_minute = overlap_count / minutes

    target_minutes = policy.target_duration_ms / 60_000
    target_turn_count = min(
        policy.max_utterances,
        max(policy.min_utterances, round(turns_per_minute * target_minutes)),
    )
    target_backchannels = min(
        max(0, target_turn_count // 3),
        round(backchannels_per_minute * target_minutes),
    )
    # Consecutive overlap requests are intentionally forbidden: alternating a
    # substantive anchor and an overlapping response is the densest safe layout.
    maximum_safe_overlaps = max(0, (target_turn_count - 1) // 2)
    target_overlaps = min(
        maximum_safe_overlaps,
        round(overlaps_per_minute * target_minutes),
    )
    return InteractionTargets(
        reconstruction_effective_duration_ms=effective_duration_ms,
        reconstruction_turn_count=len(turns),
        reconstruction_backchannel_count=backchannel_count,
        reconstruction_overlap_event_count=overlap_count,
        turns_per_minute=turns_per_minute,
        backchannels_per_minute=backchannels_per_minute,
        overlap_events_per_minute=overlaps_per_minute,
        expansion_target_duration_ms=policy.target_duration_ms,
        target_turn_count=target_turn_count,
        target_backchannel_count=target_backchannels,
        target_overlap_event_count=target_overlaps,
    )


def spoken_token_count(text: str) -> int:
    return len(_WORD.findall(text))


def is_backchannel_text(text: str) -> bool:
    return _is_backchannel_cue((text,))


def canonical_backchannel(utterance_index: int) -> str:
    return _CANONICAL_BACKCHANNELS[utterance_index % len(_CANONICAL_BACKCHANNELS)]


def planned_counts(utterances: list[dict]) -> tuple[int, int, int]:
    return (
        len(utterances),
        sum(item["type"] == "backchannel" for item in utterances),
        sum(item["placement"] == "overlap_previous" for item in utterances),
    )


def _build_turns(utterances: list[dict]) -> tuple[_Turn, ...]:
    per_speaker: list[list[dict]] = [[], []]
    for utterance in utterances:
        per_speaker[utterance["speaker_id"]].append(utterance)
    result: list[_Turn] = []
    for speaker_id, values in enumerate(per_speaker):
        start = end = 0
        texts: list[str] = []
        for utterance in values:
            if texts and utterance["start_ms"] - end > 100:
                result.append(_Turn(speaker_id, start, end, tuple(texts)))
                texts = []
            if not texts:
                start = utterance["start_ms"]
                end = utterance["end_ms"]
            else:
                end = max(end, utterance["end_ms"])
            texts.append(utterance["text"])
        if texts:
            result.append(_Turn(speaker_id, start, end, tuple(texts)))
    return tuple(sorted(result, key=lambda item: (item.start_ms, item.end_ms)))


def _count_backchannels(turns: tuple[_Turn, ...]) -> int:
    count = 0
    for index, current in enumerate(turns):
        candidates = [
            turn
            for turn in turns[:index]
            if turn.speaker_id != current.speaker_id
            and turn.start_ms <= current.start_ms
        ]
        if not candidates:
            continue
        anchor = max(candidates, key=lambda item: (item.start_ms, item.end_ms))
        gap = max(0, current.start_ms - anchor.end_ms)
        original_resumes = any(
            turn.speaker_id == anchor.speaker_id
            and current.end_ms <= turn.start_ms <= current.end_ms + 1_000
            for turn in turns[index + 1 :]
        )
        retains_floor = current.end_ms <= anchor.end_ms or original_resumes
        if (
            current.end_ms - current.start_ms <= 1_500
            and _is_backchannel_cue(current.texts)
            and retains_floor
            and gap <= 500
        ):
            count += 1
    return count


def _is_backchannel_cue(texts: tuple[str, ...]) -> bool:
    normalized = " ".join(
        re.sub(r"[^\w\u3400-\u9fff-]+", " ", text.casefold()).strip()
        for text in texts
    ).strip()
    if normalized in _BACKCHANNEL_CUES:
        return True
    tokens = normalized.split()
    return 0 < len(tokens) <= 3 and any(token in _BACKCHANNEL_CUES for token in tokens)


def _count_overlap_events(utterances: list[dict]) -> int:
    intersections: list[tuple[int, int]] = []
    left = sorted(
        (
            (item["start_ms"], item["end_ms"])
            for item in utterances
            if item["speaker_id"] == 0
        )
    )
    right = sorted(
        (
            (item["start_ms"], item["end_ms"])
            for item in utterances
            if item["speaker_id"] == 1
        )
    )
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if end - start >= _MINIMUM_OVERLAP_MS:
            intersections.append((start, end))
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    merged: list[tuple[int, int]] = []
    for start, end in intersections:
        if merged and start - merged[-1][1] <= _OVERLAP_EVENT_MERGE_GAP_MS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return len(merged)
