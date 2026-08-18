import wave
from pathlib import Path

import numpy as np
import pytest

from voice_pipeline_quality_filter_audio_part.wav_io import (
    WavError,
    read_normalized_wav,
    speech_samples,
    write_chunk_wav,
)


def write_wav(path: Path, duration_ms: int, *, sample_rate: int = 16000) -> None:
    samples = (np.sin(np.arange(duration_ms * sample_rate // 1000) / 20) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())


def test_read_and_cut_are_millisecond_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "chunk.wav"
    write_wav(source, 3000)
    audio = read_normalized_wav(source, expected_duration_ms=3000)
    assert speech_samples(audio, start_ms=1000, end_ms=2000).size == 16000
    write_chunk_wav(audio, output, start_ms=500, end_ms=2500)
    assert read_normalized_wav(output, expected_duration_ms=2000).duration_ms == 2000


def test_wrong_format_and_duration_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    write_wav(source, 1000, sample_rate=8000)
    with pytest.raises(WavError):
        read_normalized_wav(source, expected_duration_ms=1000)
