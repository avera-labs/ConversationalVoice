import pytest

from voice_pipeline_split_raw_audio_into_parts.errors import (
    ERROR_PREFIX,
    MAX_ERROR_LENGTH,
    FailureReason,
    safe_failure_message,
)


@pytest.mark.parametrize("reason", list(FailureReason))
def test_failure_messages_are_static_bounded_and_stage_specific(
    reason: FailureReason,
) -> None:
    message = safe_failure_message(reason)

    assert message.startswith(ERROR_PREFIX + " ")
    assert len(message) <= MAX_ERROR_LENGTH
    assert message.endswith(".")


def test_failure_message_api_does_not_accept_external_exception_text() -> None:
    message = safe_failure_message(FailureReason.DOWNLOAD_FAILED)

    assert "secret" not in message
    assert message == (
        "split-raw-audio-into-parts download: "
        "unable to fetch the normalized WAV."
    )
