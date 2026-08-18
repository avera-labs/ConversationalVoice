from dataclasses import dataclass

import numpy as np
from voice_pipeline_chunk_contracts import ChunkDiarization

from .config import AuditPolicy
from .errors import QualityRejection, RejectionCode


@dataclass(frozen=True, slots=True)
class AuditResult:
    reference_speaker_id: int
    relation: str
    mapping: tuple[int, int]


def audit_tracks(
    snapshot: ChunkDiarization,
    tracks: np.ndarray,
    sample_rate: int,
    policy: AuditPolicy,
) -> AuditResult:
    ids = snapshot.speaker_ids
    evidence = {ids[0]: [], ids[1]: []}
    events = []
    for s in snapshot.segments:
        events += [(s.start_ms, 1, s.speaker), (s.end_ms, -1, s.speaker)]
    events.sort()
    active = {}
    last = events[0][0] if events else 0
    intervals = []
    for time, delta, speaker in events:
        live = [s for s, c in active.items() if c > 0]
        if time > last and len(live) == 1:
            intervals.append((last, time, live[0]))
        active[speaker] = active.get(speaker, 0) + delta
        last = time
    relations = []
    for start, end, speaker in intervals:
        if end - start <= policy.pure_min_ms:
            continue
        a = (start + policy.trim_ms) * sample_rate // 1000
        b = (end - policy.trim_ms) * sample_rate // 1000
        if b <= a:
            continue
        rms = np.sqrt(np.mean(tracks[:, a:b].astype(np.float64) ** 2, axis=1) + 1e-20)
        ratio = float(max(rms) / max(min(rms), 1e-12))
        if ratio < policy.min_rms_ratio:
            continue
        winner = int(np.argmax(rms))
        evidence[speaker].append(winner)
        expected = 0 if speaker == ids[0] else 1
        relations.append("direct" if winner == expected else "swapped")
    if not relations or (
        policy.require_both_speakers and any(not evidence[s] for s in ids)
    ):
        raise QualityRejection(RejectionCode.AUDIT_INCONCLUSIVE)
    if len(set(relations)) != 1:
        raise QualityRejection(RejectionCode.TRACK_INCONSISTENT)
    relation = relations[0]
    mapping = ids if relation == "direct" else tuple(reversed(ids))
    return AuditResult(ids[0], relation, mapping)
