from __future__ import annotations

from enum import StrEnum


ERROR_PREFIX = "split-raw-audio-into-parts"
MAX_ERROR_LENGTH = 512


class FailureReason(StrEnum):
    """Stable reasons that may be persisted without exposing exception text."""

    INPUT_INVALID = "input_invalid"
    AUDIO_URI_MISSING = "audio_uri_missing"
    DOWNLOAD_FAILED = "download_failed"
    INFERENCE_FAILED = "inference_failed"
    VAD_ARTIFACT_FAILED = "vad_artifact_failed"
    GROUPING_FAILED = "grouping_failed"
    CUTTING_FAILED = "cutting_failed"
    UPLOAD_FAILED = "upload_failed"
    PERSISTENCE_FAILED = "persistence_failed"
    DOWNSTREAM_DISPATCH_FAILED = "downstream_dispatch_failed"


_SAFE_MESSAGES = {
    FailureReason.INPUT_INVALID: (
        "input",
        "the task input is invalid.",
    ),
    FailureReason.AUDIO_URI_MISSING: (
        "input",
        "the normalized WAV URI is missing.",
    ),
    FailureReason.DOWNLOAD_FAILED: (
        "download",
        "unable to fetch the normalized WAV.",
    ),
    FailureReason.INFERENCE_FAILED: (
        "inference",
        "voice activity detection failed.",
    ),
    FailureReason.VAD_ARTIFACT_FAILED: (
        "vad artifact",
        "unable to persist the VAD segments artifact.",
    ),
    FailureReason.GROUPING_FAILED: (
        "grouping",
        "unable to build conversation windows.",
    ),
    FailureReason.CUTTING_FAILED: (
        "cutting",
        "unable to create an audio part.",
    ),
    FailureReason.UPLOAD_FAILED: (
        "upload",
        "unable to persist an audio part.",
    ),
    FailureReason.PERSISTENCE_FAILED: (
        "persistence",
        "unable to commit audio part records.",
    ),
    FailureReason.DOWNSTREAM_DISPATCH_FAILED: (
        "downstream dispatch",
        "unable to publish a diarization task.",
    ),
}


def safe_failure_message(reason: FailureReason) -> str:
    """Return a bounded message assembled only from reviewed static text."""

    stage, detail = _SAFE_MESSAGES[reason]
    message = f"{ERROR_PREFIX} {stage}: {detail}"
    if len(message) > MAX_ERROR_LENGTH:
        raise AssertionError("Reviewed failure message exceeds its storage limit.")
    return message
