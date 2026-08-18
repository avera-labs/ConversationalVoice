from enum import StrEnum


class RejectionCode(StrEnum):
    WINDOW_EVIDENCE_INSUFFICIENT = "window_evidence_insufficient"
    ALIGNMENT_LOW_CONFIDENCE = "alignment_low_confidence"
    TRACK_INCONSISTENT = "track_inconsistent"
    AUDIT_INCONCLUSIVE = "audit_inconclusive"
    OUTPUT_QUALITY_REJECTED = "output_quality_rejected"


class QualityRejection(RuntimeError):
    def __init__(self, code: RejectionCode):
        super().__init__(code.value)
        self.code = code


def safe_error(code: str, maximum: int) -> str:
    return f"{code}: processing did not complete safely"[:maximum]
