from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

NORMALIZED_SAMPLE_RATE_HZ = 16_000
NORMALIZED_CHANNELS = 1
NORMALIZED_SAMPLE_WIDTH_BYTES = 2
NORMALIZED_COMPRESSION_TYPE = "NONE"


class WavValidationError(ValueError):
    """Raised when a WAV does not match the normalized audio contract."""


@dataclass(frozen=True, slots=True)
class WavMetadata:
    """Validated metadata for a normalized WAV artifact."""

    duration_ms: int
    size_bytes: int
    frame_count: int


def validate_normalized_wav(path: Path) -> WavMetadata:
    """Validate that a non-empty WAV is 16 kHz mono 16-bit PCM."""
    try:
        size_bytes = path.stat().st_size
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression_type = wav_file.getcomptype()
            frame_count = wav_file.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise WavValidationError(
            "Normalized audio is not a readable WAV file."
        ) from exc

    if size_bytes == 0 or frame_count == 0:
        raise WavValidationError("Normalized WAV contains no audio frames.")
    if channels != NORMALIZED_CHANNELS:
        raise WavValidationError("Normalized WAV must be mono.")
    if sample_width != NORMALIZED_SAMPLE_WIDTH_BYTES:
        raise WavValidationError("Normalized WAV must use 16-bit samples.")
    if sample_rate != NORMALIZED_SAMPLE_RATE_HZ:
        raise WavValidationError("Normalized WAV must use a 16 kHz sample rate.")
    if compression_type != NORMALIZED_COMPRESSION_TYPE:
        raise WavValidationError("Normalized WAV must use PCM encoding.")

    duration_ms = round(frame_count * 1000 / sample_rate)
    return WavMetadata(
        duration_ms=duration_ms,
        size_bytes=size_bytes,
        frame_count=frame_count,
    )
