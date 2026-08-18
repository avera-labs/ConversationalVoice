"""Strict two-speaker raw-window planning and region-scoped greedy merge."""

from __future__ import annotations

from dataclasses import dataclass, field

from voice_pipeline_diarization_artifact import DiarizationTurn

from .config import PlannerPolicy
from .intervals import Interval, overlap_ms


@dataclass(slots=True)
class MonoRun:
    speaker: int
    start_ms: int
    end_ms: int
    non_backchannel_starts: list[int] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class PlanResult:
    start_ms: int
    end_ms: int
    satisfied: bool
    next_position_ms: int


@dataclass(frozen=True, slots=True, order=True)
class RawWindow:
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    chunk_index: int
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def _overlapping(
    turns: tuple[DiarizationTurn, ...], start_ms: int, end_ms: int
) -> tuple[DiarizationTurn, ...]:
    return tuple(turn for turn in turns if turn.end_ms > start_ms and turn.start_ms < end_ms)


def _speaker_statistics(
    turns: tuple[DiarizationTurn, ...], start_ms: int, end_ms: int
) -> tuple[dict[int, int], dict[int, int]]:
    longest: dict[int, int] = {}
    totals: dict[int, int] = {}
    for turn in turns:
        if turn.start_ms < start_ms or turn.end_ms > end_ms:
            continue
        duration = turn.end_ms - turn.start_ms
        longest[turn.speaker] = max(longest.get(turn.speaker, 0), duration)
        totals[turn.speaker] = totals.get(turn.speaker, 0) + duration
    return longest, totals


def is_strict_two_speaker_window(
    turns: tuple[DiarizationTurn, ...],
    start_ms: int,
    end_ms: int,
    policy: PlannerPolicy,
) -> bool:
    overlapping = _overlapping(turns, start_ms, end_ms)
    speakers = {turn.speaker for turn in overlapping}
    if len(speakers) != 2:
        return False
    longest, totals = _speaker_statistics(turns, start_ms, end_ms)
    return all(
        longest.get(speaker, 0) >= policy.min_speaker_turn_ms
        and totals.get(speaker, 0) >= policy.min_speaker_total_ms
        for speaker in speakers
    )


def longest_effective_monologue(
    turns: tuple[DiarizationTurn, ...],
    start_ms: int,
    end_ms: int,
    policy: PlannerPolicy,
) -> MonoRun | None:
    runs: list[MonoRun] = []
    current: MonoRun | None = None
    previous: DiarizationTurn | None = None
    for turn in sorted(_overlapping(turns, start_ms, end_ms), key=lambda item: (item.start_ms, item.end_ms, item.speaker)):
        clipped_start = max(turn.start_ms, start_ms)
        clipped_end = min(turn.end_ms, end_ms)
        duration = turn.end_ms - turn.start_ms
        is_backchannel = duration < policy.backchannel_threshold_ms
        if current is None:
            if not is_backchannel:
                current = MonoRun(turn.speaker, clipped_start, clipped_end, [clipped_start])
            previous = turn
            continue
        if turn.speaker == current.speaker:
            current.end_ms = max(current.end_ms, clipped_end)
            if not is_backchannel:
                current.non_backchannel_starts.append(clipped_start)
            previous = turn
            continue
        if is_backchannel:
            previous = turn
            continue
        runs.append(current)
        new_start = clipped_start
        if (
            previous is not None
            and previous.speaker == turn.speaker
            and previous.end_ms - previous.start_ms < policy.backchannel_threshold_ms
            and previous.end_ms > start_ms
            and previous.start_ms < end_ms
        ):
            new_start = max(previous.start_ms, start_ms)
        current = MonoRun(turn.speaker, new_start, clipped_end, [clipped_start])
        previous = turn
    if current is not None:
        runs.append(current)
    return max(runs, key=lambda item: item.duration_ms) if runs else None


def _position_after_overflow(run: MonoRun, policy: PlannerPolicy) -> int:
    fitting = [
        start
        for start in run.non_backchannel_starts
        if run.end_ms - start <= policy.max_monologue_ms
    ]
    return min(fitting) if fitting else run.end_ms - policy.max_monologue_ms


def _extra_speaker_region(
    turns: tuple[DiarizationTurn, ...], start_ms: int, end_ms: int
) -> tuple[DiarizationTurn, int] | None:
    overlapping = _overlapping(turns, start_ms, end_ms)
    airtime: dict[int, int] = {}
    for turn in overlapping:
        duration = min(turn.end_ms, end_ms) - max(turn.start_ms, start_ms)
        if duration > 0:
            airtime[turn.speaker] = airtime.get(turn.speaker, 0) + duration
    if len(airtime) <= 2:
        return None
    extra_speaker = min(airtime, key=lambda speaker: (airtime[speaker], speaker))
    extra_turns = [turn for turn in overlapping if turn.speaker == extra_speaker]
    return min(extra_turns, key=lambda item: item.start_ms), max(item.end_ms for item in extra_turns)


def plan_one_window(
    turns: tuple[DiarizationTurn, ...],
    start_position_ms: int,
    region_end_ms: int,
    policy: PlannerPolicy,
) -> PlanResult:
    current_start = start_position_ms
    current_end = start_position_ms
    ordered = tuple(sorted(turns, key=lambda item: (item.start_ms, item.end_ms, item.speaker)))
    for _ in range(policy.max_inner_iterations):
        maximum_end = min(current_start + policy.max_planning_window_ms, region_end_ms)
        if maximum_end <= current_end:
            break
        extending = next(
            (
                turn
                for turn in ordered
                if turn.end_ms > current_end and turn.start_ms < maximum_end
            ),
            None,
        )
        if extending is None:
            break
        candidate_end = min(extending.end_ms, maximum_end)
        extra = _extra_speaker_region(ordered, current_start, candidate_end)
        if extra is not None:
            first_extra, extra_end = extra
            capped_end = max(current_start, first_extra.start_ms)
            scan_end = min(current_start + policy.max_planning_window_ms, region_end_ms)
            scanned = _extra_speaker_region(ordered, current_start, scan_end)
            if scanned is not None:
                extra_end = max(extra_end, scanned[1])
            next_position = max(extra_end, current_start + 1)
            satisfied = (
                capped_end - current_start >= policy.min_planning_window_ms
                and is_strict_two_speaker_window(ordered, current_start, capped_end, policy)
            )
            return PlanResult(current_start, capped_end, satisfied, next_position)
        monologue = longest_effective_monologue(ordered, current_start, candidate_end, policy)
        if monologue is not None and monologue.duration_ms > policy.max_monologue_ms:
            current_start = max(_position_after_overflow(monologue, policy), current_start + 1)
            current_end = max(current_end, current_start)
            continue
        current_end = candidate_end
        if current_end - current_start < policy.min_planning_window_ms:
            continue
        if is_strict_two_speaker_window(ordered, current_start, current_end, policy):
            return PlanResult(current_start, current_end, True, max(current_end, current_start + 1))
    fallback = max(current_end, current_start + policy.min_planning_window_ms)
    return PlanResult(current_start, current_end, False, min(fallback, region_end_ms))


def raw_windows_for_region(
    turns: tuple[DiarizationTurn, ...], region: Interval, policy: PlannerPolicy
) -> tuple[RawWindow, ...]:
    region_turns = _overlapping(turns, region.start_ms, region.end_ms)
    raw: list[RawWindow] = []
    position = region.start_ms
    iterations = 0
    maximum_iterations = 10 * region.duration_ms + 1000
    while position < region.end_ms and iterations < maximum_iterations:
        iterations += 1
        result = plan_one_window(region_turns, position, region.end_ms, policy)
        if result.satisfied:
            raw.append(RawWindow(result.start_ms, result.end_ms))
        position = max(result.next_position_ms, position + 1)
    return tuple(raw)


def _merge_is_valid(
    turns: tuple[DiarizationTurn, ...], start_ms: int, end_ms: int, policy: PlannerPolicy
) -> bool:
    speakers = {turn.speaker for turn in _overlapping(turns, start_ms, end_ms)}
    if len(speakers) != 2:
        return False
    monologue = longest_effective_monologue(turns, start_ms, end_ms, policy)
    return monologue is None or monologue.duration_ms <= policy.max_monologue_ms


def merge_region_windows(
    raw: tuple[RawWindow, ...],
    turns: tuple[DiarizationTurn, ...],
    policy: PlannerPolicy,
) -> tuple[RawWindow, ...]:
    if not raw:
        return ()
    output = [sorted(raw)[0]]
    for window in sorted(raw)[1:]:
        merged = RawWindow(output[-1].start_ms, max(output[-1].end_ms, window.end_ms))
        if _merge_is_valid(turns, merged.start_ms, merged.end_ms, policy):
            output[-1] = merged
        else:
            output.append(window)
    return tuple(output)


def plan_chunks(
    turns: tuple[DiarizationTurn, ...],
    regions: tuple[Interval, ...],
    policy: PlannerPolicy,
) -> tuple[ChunkDraft, ...]:
    windows: list[RawWindow] = []
    for region in regions:
        region_turns = _overlapping(turns, region.start_ms, region.end_ms)
        raw = raw_windows_for_region(region_turns, region, policy)
        windows.extend(merge_region_windows(raw, region_turns, policy))
    normalized = sorted(set(windows))
    for previous, following in zip(normalized, normalized[1:]):
        if following.start_ms < previous.end_ms:
            raise ValueError("planned windows overlap")
    return tuple(
        ChunkDraft(index, window.start_ms, window.end_ms)
        for index, window in enumerate(normalized)
    )
