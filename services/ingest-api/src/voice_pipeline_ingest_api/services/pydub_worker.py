from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment

from .wav_validation import (
    NORMALIZED_CHANNELS,
    NORMALIZED_SAMPLE_RATE_HZ,
    NORMALIZED_SAMPLE_WIDTH_BYTES,
)


def normalize_with_pydub(source_path: str, destination_path: str) -> None:
    """Decode audio and export a normalized PCM WAV with pydub."""
    audio = AudioSegment.from_file(source_path)

    if audio.channels != NORMALIZED_CHANNELS:
        audio = audio.set_channels(NORMALIZED_CHANNELS)
    if audio.frame_rate > NORMALIZED_SAMPLE_RATE_HZ:
        audio = audio.set_frame_rate(NORMALIZED_SAMPLE_RATE_HZ)
    if audio.sample_width != NORMALIZED_SAMPLE_WIDTH_BYTES:
        audio = audio.set_sample_width(NORMALIZED_SAMPLE_WIDTH_BYTES)
    if audio.frame_rate != NORMALIZED_SAMPLE_RATE_HZ:
        audio = audio.set_frame_rate(NORMALIZED_SAMPLE_RATE_HZ)

    audio.export(Path(destination_path), format="wav")
