from __future__ import annotations

import wave

import pytest

from voice_pipeline_ingest_api.services.wav_validation import (
    WavValidationError,
    validate_normalized_wav,
)


def _write_wav(
    path,
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16_000,
    frame_count: int = 1_600,
) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * frame_count * channels * sample_width)


def test_validate_normalized_wav_returns_metadata(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    _write_wav(path)

    metadata = validate_normalized_wav(path)

    assert metadata.duration_ms == 100
    assert metadata.frame_count == 1_600
    assert metadata.size_bytes == path.stat().st_size


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"channels": 2}, "mono"),
        ({"sample_width": 1}, "16-bit"),
        ({"sample_rate": 44_100}, "16 kHz"),
        ({"frame_count": 0}, "no audio frames"),
    ],
)
def test_validate_normalized_wav_rejects_contract_mismatch(
    tmp_path,
    overrides,
    message,
) -> None:
    path = tmp_path / "audio.wav"
    _write_wav(path, **overrides)

    with pytest.raises(WavValidationError, match=message):
        validate_normalized_wav(path)


def test_validate_normalized_wav_rejects_unreadable_file(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"not-a-wave")

    with pytest.raises(WavValidationError, match="readable WAV"):
        validate_normalized_wav(path)
