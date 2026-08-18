"""Reviewed, persistence-safe task errors."""

from __future__ import annotations

from enum import StrEnum


class FailureStage(StrEnum):
    INPUT = "input"
    CLAIM = "claim"
    DOWNLOAD = "download"
    INFERENCE = "inference"
    ARTIFACT = "artifact"
    UPLOAD = "upload"
    PERSISTENCE = "persistence"
    DOWNSTREAM_DISPATCH = "downstream_dispatch"
    CLEANUP = "cleanup"


class ErrorCode(StrEnum):
    INVALID_AUDIO_PART_ID = "invalid_audio_part_id"
    AUDIO_PART_NOT_FOUND = "audio_part_not_found"
    INVALID_AUDIO_PART_STATE = "invalid_audio_part_state"
    INVALID_INPUT_URI = "invalid_input_uri"
    DOWNLOAD_FAILED = "download_failed"
    INFERENCE_FAILED = "inference_failed"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    ARTIFACT_WRITE_FAILED = "artifact_write_failed"
    SPEAKER_REFERENCE_FAILED = "speaker_reference_failed"
    UPLOAD_FAILED = "upload_failed"
    PERSISTENCE_FAILED = "persistence_failed"
    DOWNSTREAM_DISPATCH_FAILED = "downstream_dispatch_failed"
    CLEANUP_FAILED = "cleanup_failed"


SAFE_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INVALID_AUDIO_PART_ID: "diarize-audio-part input: invalid audio part identifier",
    ErrorCode.AUDIO_PART_NOT_FOUND: "diarize-audio-part input: audio part not found",
    ErrorCode.INVALID_AUDIO_PART_STATE: "diarize-audio-part input: invalid audio part status",
    ErrorCode.INVALID_INPUT_URI: "diarize-audio-part download: invalid input artifact URI",
    ErrorCode.DOWNLOAD_FAILED: "diarize-audio-part download: input artifact download failed",
    ErrorCode.INFERENCE_FAILED: "diarize-audio-part inference: model execution failed",
    ErrorCode.INVALID_MODEL_OUTPUT: "diarize-audio-part artifact: invalid model output",
    ErrorCode.ARTIFACT_WRITE_FAILED: "diarize-audio-part artifact: artifact write failed",
    ErrorCode.SPEAKER_REFERENCE_FAILED: "diarize-audio-part artifact: speaker reference generation failed",
    ErrorCode.UPLOAD_FAILED: "diarize-audio-part upload: artifact upload failed",
    ErrorCode.PERSISTENCE_FAILED: "diarize-audio-part persistence: state update failed",
    ErrorCode.DOWNSTREAM_DISPATCH_FAILED: "diarize-audio-part downstream dispatch: message publish failed",
    ErrorCode.CLEANUP_FAILED: "diarize-audio-part cleanup: temporary workspace cleanup failed",
}


class TaskStageError(RuntimeError):
    """Error with a static message safe for persistence and terminal logs."""

    def __init__(self, stage: FailureStage, code: ErrorCode) -> None:
        self.stage = stage
        self.code = code
        super().__init__(SAFE_MESSAGES[code])


def safe_message(code: ErrorCode, max_length: int = 512) -> str:
    return SAFE_MESSAGES[code][:max_length]
