"""Stable task errors safe for persistence."""

from enum import StrEnum


class FailureStage(StrEnum):
    INPUT = "input"
    CLAIM = "claim"
    DOWNLOAD = "download"
    VALIDATION = "validation"
    MUSIC = "music"
    SNR = "snr"
    PLANNING = "planning"
    CUT = "cut"
    UPLOAD = "upload"
    PERSISTENCE = "persistence"
    CLEANUP = "cleanup"


class ErrorCode(StrEnum):
    INVALID_AUDIO_PART_ID = "invalid_audio_part_id"
    AUDIO_PART_NOT_FOUND = "audio_part_not_found"
    INVALID_AUDIO_PART_STATE = "invalid_audio_part_state"
    INVALID_INPUT = "invalid_input"
    DOWNLOAD_FAILED = "download_failed"
    INVALID_AUDIO = "invalid_audio"
    INVALID_DIARIZATION = "invalid_diarization"
    MUSIC_DETECTION_FAILED = "music_detection_failed"
    SNR_FAILED = "snr_failed"
    PLANNING_FAILED = "planning_failed"
    CUT_FAILED = "cut_failed"
    UPLOAD_FAILED = "upload_failed"
    PERSISTENCE_FAILED = "persistence_failed"
    CLEANUP_FAILED = "cleanup_failed"


SAFE_MESSAGES = {
    ErrorCode.INVALID_AUDIO_PART_ID: "quality-filter input: invalid audio part identifier",
    ErrorCode.AUDIO_PART_NOT_FOUND: "quality-filter input: audio part not found",
    ErrorCode.INVALID_AUDIO_PART_STATE: "quality-filter input: invalid audio part status",
    ErrorCode.INVALID_INPUT: "quality-filter input: persisted input is invalid",
    ErrorCode.DOWNLOAD_FAILED: "quality-filter download: artifact download failed",
    ErrorCode.INVALID_AUDIO: "quality-filter validation: audio is invalid",
    ErrorCode.INVALID_DIARIZATION: "quality-filter validation: diarization artifact is invalid",
    ErrorCode.MUSIC_DETECTION_FAILED: "quality-filter music: model execution failed",
    ErrorCode.SNR_FAILED: "quality-filter snr: calculation failed",
    ErrorCode.PLANNING_FAILED: "quality-filter planning: window planning failed",
    ErrorCode.CUT_FAILED: "quality-filter cut: chunk audio creation failed",
    ErrorCode.UPLOAD_FAILED: "quality-filter upload: chunk audio upload failed",
    ErrorCode.PERSISTENCE_FAILED: "quality-filter persistence: state update failed",
    ErrorCode.CLEANUP_FAILED: "quality-filter cleanup: temporary workspace cleanup failed",
}


class TaskStageError(RuntimeError):
    def __init__(self, stage: FailureStage, code: ErrorCode) -> None:
        self.stage = stage
        self.code = code
        super().__init__(SAFE_MESSAGES[code])


def safe_message(code: ErrorCode, max_length: int) -> str:
    return SAFE_MESSAGES[code][:max_length]
