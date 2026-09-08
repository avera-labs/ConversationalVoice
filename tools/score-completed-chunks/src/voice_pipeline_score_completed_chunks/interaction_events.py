from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .interaction_config import InteractionConfig
from .interaction_transcript import InteractionUtterance
from .vad import Interval, intersect_interval, interval_total

TRANSITION_CATEGORIES = (
    "backchannel",
    "interruption",
    "other_overlap",
    "clean_transition",
    "delayed_other",
)
OVERLAP_CATEGORIES = {"backchannel", "interruption", "other_overlap"}

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


@dataclass(frozen=True, slots=True)
class AcousticUtterance:
    utterance_index: int
    source_index: int | None
    speaker_id: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class Turn:
    turn_id: int
    speaker_id: int
    start_ms: int
    end_ms: int
    utterance_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    texts: tuple[str, ...]

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class TransitionEvent:
    event_id: str
    category: str
    speaker_id: int
    other_speaker_id: int
    start_ms: int
    end_ms: int
    gap_ms: int
    overlap_duration_ms: int
    anchor_turn_id: int
    current_turn_id: int
    anchor_source_indices: tuple[int, ...]
    current_source_indices: tuple[int, ...]
    current_utterance_indices: tuple[int, ...]
    rule_trace: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StageAnalysis:
    duration_ms: int
    activities: tuple[tuple[Interval, ...], tuple[Interval, ...]]
    acoustic_utterances: tuple[AcousticUtterance, ...]
    turns: tuple[Turn, ...]
    transitions: tuple[TransitionEvent, ...]
    overlap_intervals: tuple[Interval, ...]
    overlap_event_intervals: tuple[Interval, ...]
    state_intervals: dict[str, tuple[Interval, ...]]


def _normalize_text(text: str) -> str:
    lowered = text.casefold().strip()
    return re.sub(r"[^\w\u3400-\u9fff-]+", " ", lowered).strip()


def _is_backchannel_cue(texts: tuple[str, ...]) -> bool:
    normalized = " ".join(_normalize_text(text) for text in texts).strip()
    if normalized in _BACKCHANNEL_CUES:
        return True
    tokens = normalized.split()
    return 0 < len(tokens) <= 3 and any(token in _BACKCHANNEL_CUES for token in tokens)


def _overlap_intervals(
    left: tuple[Interval, ...], right: tuple[Interval, ...]
) -> tuple[Interval, ...]:
    result: list[Interval] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        overlap = intersect_interval(left[left_index], right[right_index])
        if overlap is not None:
            result.append(overlap)
        if left[left_index].end_ms <= right[right_index].end_ms:
            left_index += 1
        else:
            right_index += 1
    return tuple(result)


def _merge_overlap_fragments(
    intervals: tuple[Interval, ...], *, maximum_gap_ms: int
) -> tuple[Interval, ...]:
    """Merge nearby qualifying fragments for event counting only.

    The merged envelopes must not be used to measure simultaneous-speech
    duration because the gaps between fragments are not overlap.
    """

    merged: list[Interval] = []
    for interval in intervals:
        if merged and interval.start_ms - merged[-1].end_ms <= maximum_gap_ms:
            previous = merged.pop()
            merged.append(
                Interval(previous.start_ms, max(previous.end_ms, interval.end_ms))
            )
        else:
            merged.append(interval)
    return tuple(merged)


def activity_state_intervals(
    activities: tuple[tuple[Interval, ...], tuple[Interval, ...]], duration_ms: int
) -> dict[str, tuple[Interval, ...]]:
    boundaries = {0, duration_ms}
    for speaker in activities:
        for interval in speaker:
            boundaries.update((interval.start_ms, interval.end_ms))
    ordered = sorted(boundaries)
    states: dict[str, list[Interval]] = {
        "silence": [],
        "a_only": [],
        "b_only": [],
        "overlap": [],
    }
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        active = [
            any(item.start_ms <= midpoint < item.end_ms for item in speaker)
            for speaker in activities
        ]
        state = (
            "overlap"
            if active == [True, True]
            else "a_only"
            if active == [True, False]
            else "b_only"
            if active == [False, True]
            else "silence"
        )
        candidate = Interval(start, end)
        if states[state] and states[state][-1].end_ms == start:
            previous = states[state].pop()
            states[state].append(Interval(previous.start_ms, end))
        else:
            states[state].append(candidate)
    return {key: tuple(value) for key, value in states.items()}


def _acoustic_utterances(
    utterances: tuple[InteractionUtterance, ...],
    activities: tuple[tuple[Interval, ...], tuple[Interval, ...]],
) -> tuple[AcousticUtterance, ...]:
    result: list[AcousticUtterance] = []
    for utterance in utterances:
        planned = Interval(utterance.start_ms, utterance.end_ms)
        overlaps = [
            active
            for active in activities[utterance.speaker_id]
            if intersect_interval(active, planned) is not None
        ]
        if not overlaps:
            continue
        result.append(
            AcousticUtterance(
                utterance_index=utterance.utterance_index,
                source_index=utterance.source_index,
                speaker_id=utterance.speaker_id,
                start_ms=min(item.start_ms for item in overlaps),
                end_ms=max(item.end_ms for item in overlaps),
                text=utterance.text,
            )
        )
    return tuple(
        sorted(result, key=lambda item: (item.start_ms, item.end_ms, item.speaker_id))
    )


def _turns(
    utterances: tuple[AcousticUtterance, ...], config: InteractionConfig
) -> tuple[Turn, ...]:
    per_speaker: list[list[AcousticUtterance]] = [[], []]
    for utterance in utterances:
        per_speaker[utterance.speaker_id].append(utterance)
    provisional: list[tuple[int, int, int, list[AcousticUtterance]]] = []
    for speaker_id, values in enumerate(per_speaker):
        current: list[AcousticUtterance] = []
        current_start = 0
        current_end = 0
        for utterance in sorted(values, key=lambda item: (item.start_ms, item.end_ms)):
            if (
                current
                and utterance.start_ms - current_end > config.merge_inactive_gap_ms
            ):
                provisional.append((speaker_id, current_start, current_end, current))
                current = []
            if not current:
                current_start = utterance.start_ms
                current_end = utterance.end_ms
            else:
                current_end = max(current_end, utterance.end_ms)
            current.append(utterance)
        if current:
            provisional.append((speaker_id, current_start, current_end, current))
    ordered = sorted(provisional, key=lambda item: (item[1], item[2], item[0]))
    return tuple(
        Turn(
            turn_id=index,
            speaker_id=speaker_id,
            start_ms=start,
            end_ms=end,
            utterance_indices=tuple(item.utterance_index for item in members),
            source_indices=tuple(
                item.source_index for item in members if item.source_index is not None
            ),
            texts=tuple(item.text for item in members),
        )
        for index, (speaker_id, start, end, members) in enumerate(ordered)
    )


def _transitions(
    turns: tuple[Turn, ...], config: InteractionConfig
) -> tuple[TransitionEvent, ...]:
    events: list[TransitionEvent] = []
    for current in turns:
        candidates = [
            turn
            for turn in turns
            if turn.speaker_id != current.speaker_id
            and turn.start_ms <= current.start_ms
            and turn.turn_id < current.turn_id
        ]
        if not candidates:
            continue
        anchor = max(
            candidates, key=lambda item: (item.start_ms, item.end_ms, item.turn_id)
        )
        overlap = max(0, min(anchor.end_ms, current.end_ms) - current.start_ms)
        gap = max(0, current.start_ms - anchor.end_ms)
        original_resumes = any(
            turn.speaker_id == anchor.speaker_id
            and current.end_ms
            <= turn.start_ms
            <= current.end_ms + config.floor_resumption_window_ms
            for turn in turns
        )
        retains_floor = current.end_ms <= anchor.end_ms or original_resumes
        short_response = (
            current.duration_ms <= config.short_response_maximum_duration_ms
        )
        lexical_cue = _is_backchannel_cue(current.texts)
        trace = [f"gap_ms={gap}", f"overlap_ms={overlap}"]
        if (
            short_response
            and lexical_cue
            and retains_floor
            and gap <= config.clean_transition_maximum_gap_ms
        ):
            category = "backchannel"
            trace.extend(("short_response", "lexical_cue", "floor_retained"))
        elif (
            overlap >= config.minimum_cross_speaker_overlap_ms
            and current.end_ms > anchor.end_ms
            and not original_resumes
        ):
            category = "interruption"
            trace.extend(("cross_speaker_overlap", "new_speaker_retains_floor"))
        elif overlap >= config.minimum_cross_speaker_overlap_ms:
            category = "other_overlap"
            trace.append("cross_speaker_overlap")
        elif gap <= config.clean_transition_maximum_gap_ms:
            category = "clean_transition"
            trace.append("bounded_gap")
        else:
            category = "delayed_other"
            trace.append("long_gap_or_other")
        events.append(
            TransitionEvent(
                event_id=f"transition-{len(events)}",
                category=category,
                speaker_id=current.speaker_id,
                other_speaker_id=anchor.speaker_id,
                start_ms=current.start_ms,
                end_ms=current.end_ms,
                gap_ms=gap,
                overlap_duration_ms=overlap,
                anchor_turn_id=anchor.turn_id,
                current_turn_id=current.turn_id,
                anchor_source_indices=anchor.source_indices,
                current_source_indices=current.source_indices,
                current_utterance_indices=current.utterance_indices,
                rule_trace=tuple(trace),
            )
        )
    return tuple(events)


def build_stage_analysis(
    *,
    utterances: tuple[InteractionUtterance, ...],
    activities: tuple[tuple[Interval, ...], tuple[Interval, ...]],
    duration_ms: int,
    config: InteractionConfig,
) -> StageAnalysis:
    acoustic = _acoustic_utterances(utterances, activities)
    turns = _turns(acoustic, config)
    transitions = _transitions(turns, config)
    overlap_intervals = tuple(
        interval
        for interval in _overlap_intervals(activities[0], activities[1])
        if interval.duration_ms >= config.minimum_cross_speaker_overlap_ms
    )
    overlap_event_intervals = _merge_overlap_fragments(
        overlap_intervals,
        maximum_gap_ms=config.overlap_event_merge_gap_ms,
    )
    return StageAnalysis(
        duration_ms=duration_ms,
        activities=activities,
        acoustic_utterances=acoustic,
        turns=turns,
        transitions=transitions,
        overlap_intervals=overlap_intervals,
        overlap_event_intervals=overlap_event_intervals,
        state_intervals=activity_state_intervals(activities, duration_ms),
    )


def effective_conversation_bounds(analysis: StageAnalysis) -> tuple[int, int]:
    intervals = [item for speaker in analysis.activities for item in speaker]
    if not intervals:
        return 0, analysis.duration_ms
    return min(item.start_ms for item in intervals), max(
        item.end_ms for item in intervals
    )


def stage_metrics(analysis: StageAnalysis) -> dict[str, object]:
    counts = Counter(event.category for event in analysis.transitions)
    eligible = len(analysis.transitions)
    effective_start_ms, effective_end_ms = effective_conversation_bounds(analysis)
    effective_duration_ms = effective_end_ms - effective_start_ms
    conversation_minutes = effective_duration_ms / 60_000
    active_ms = interval_total(analysis.activities[0]) + interval_total(
        analysis.activities[1]
    )
    active_minutes = active_ms / 60_000
    probabilities = (
        [
            counts[category] / eligible
            for category in TRANSITION_CATEGORIES
            if counts[category]
        ]
        if eligible
        else []
    )
    entropy = (
        math.exp(-sum(value * math.log(value) for value in probabilities))
        if probabilities
        else 0.0
    )
    return {
        "eligible_transition_count": eligible,
        "transition_counts": {
            category: counts[category] for category in TRANSITION_CATEGORIES
        },
        "transition_rates": {
            category: counts[category] / eligible if eligible else None
            for category in TRANSITION_CATEGORIES
        },
        "overlap_transition_count": sum(counts[value] for value in OVERLAP_CATEGORIES),
        "overlap_transition_rate": (
            sum(counts[value] for value in OVERLAP_CATEGORIES) / eligible
            if eligible
            else None
        ),
        "turn_count": len(analysis.turns),
        "turn_rate_per_minute": (
            len(analysis.turns) / conversation_minutes if conversation_minutes else None
        ),
        "backchannel_rate_per_minute": (
            counts["backchannel"] / conversation_minutes
            if conversation_minutes
            else None
        ),
        "interruption_rate_per_minute": (
            counts["interruption"] / conversation_minutes
            if conversation_minutes
            else None
        ),
        "overlap_event_count": len(analysis.overlap_event_intervals),
        "overlap_event_rate_per_minute": (
            len(analysis.overlap_event_intervals) / conversation_minutes
            if conversation_minutes
            else None
        ),
        "overlap_density_per_conversation_minute": (
            sum(counts[value] for value in OVERLAP_CATEGORIES) / conversation_minutes
            if conversation_minutes
            else None
        ),
        "conversation_duration_ms": effective_duration_ms,
        "track_duration_ms": analysis.duration_ms,
        "effective_start_ms": effective_start_ms,
        "effective_end_ms": effective_end_ms,
        "active_speech_duration_ms": active_ms,
        "active_speech_minutes": active_minutes,
        "overlap_durations_ms": [
            event.overlap_duration_ms
            for event in analysis.transitions
            if event.overlap_duration_ms > 0
        ],
        "inter_turn_gaps_ms": [
            event.gap_ms for event in analysis.transitions if event.gap_ms > 0
        ],
        "observed_transition_categories": sorted(
            category for category in TRANSITION_CATEGORIES if counts[category]
        ),
        "interaction_entropy_effective_categories": entropy,
        "effective_interaction_category_coverage": (
            entropy / len(TRANSITION_CATEGORIES) if eligible else None
        ),
        "observed_category_count": sum(
            bool(counts[value]) for value in TRANSITION_CATEGORIES
        ),
        "state_durations_ms": {
            key: interval_total(value)
            for key, value in analysis.state_intervals.items()
        },
    }
