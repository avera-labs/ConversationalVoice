from dataclasses import dataclass

from voice_pipeline_chunk_contracts import ChunkDiarization

from .config import WindowPolicy
from .errors import QualityRejection, RejectionCode


@dataclass(frozen=True, slots=True)
class Window:
    start_ms: int
    end_ms: int


def evidence(
    snapshot: ChunkDiarization, start: int, end: int, policy: WindowPolicy
) -> bool:
    longest = {speaker: 0 for speaker in snapshot.speaker_ids}
    totals = dict(longest)
    for segment in snapshot.segments:
        amount = max(0, min(segment.end_ms, end) - max(segment.start_ms, start))
        longest[segment.speaker] = max(longest[segment.speaker], amount)
        totals[segment.speaker] += amount
    return all(
        longest[s] >= policy.speaker_once_ms and totals[s] >= policy.speaker_total_ms
        for s in snapshot.speaker_ids
    )


def plan_windows(
    snapshot: ChunkDiarization, duration_ms: int, policy: WindowPolicy
) -> tuple[Window, ...]:
    if not evidence(snapshot, 0, duration_ms, policy):
        raise RuntimeError("chunk_evidence_contract_drift")
    output = []
    pos = 0
    while pos < duration_ms:
        end = min(pos + policy.initial_ms, duration_ms)
        while (
            not evidence(snapshot, pos, end, policy)
            and end < duration_ms
            and end - pos < policy.maximum_ms
        ):
            end = min(end + policy.extension_ms, duration_ms, pos + policy.maximum_ms)
        output.append(Window(pos, end))
        if end == duration_ms:
            break
        pos = end - policy.overlap_ms
    if (
        len(output) > 1
        and output[-1].end_ms - output[-1].start_ms < policy.initial_ms
        and output[-1].end_ms - output[-2].start_ms <= policy.maximum_ms
    ):
        output[-2] = Window(output[-2].start_ms, output[-1].end_ms)
        output.pop()
    if not all(evidence(snapshot, w.start_ms, w.end_ms, policy) for w in output):
        raise QualityRejection(RejectionCode.WINDOW_EVIDENCE_INSUFFICIENT)
    return tuple(output)
