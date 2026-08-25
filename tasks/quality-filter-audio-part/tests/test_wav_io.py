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


def write_wav(
    path: Path,
    duration_ms: int,
    *,
    sample_rate: int = 16000,
    frame_count: int | None = None,
) -> None:
    count = duration_ms * sample_rate // 1000 if frame_count is None else frame_count
    samples = (np.sin(np.arange(count) / 20) * 12000).astype("<i2")
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


@pytest.mark.parametrize("frame_count", [15_984, 16_016])
def test_one_millisecond_duration_difference_is_normalized(
    tmp_path: Path, frame_count: int
) -> None:
    source = tmp_path / "source.wav"
    write_wav(source, 1000, frame_count=frame_count)

    audio = read_normalized_wav(source, expected_duration_ms=1000)

    assert audio.samples.size == 16_000
    assert audio.duration_ms == 1000


@pytest.mark.parametrize("frame_count", [15_983, 16_017])
def test_more_than_one_millisecond_duration_difference_is_rejected(
    tmp_path: Path, frame_count: int
) -> None:
    source = tmp_path / "source.wav"
    write_wav(source, 1000, frame_count=frame_count)

    with pytest.raises(WavError, match="duration"):
        read_normalized_wav(source, expected_duration_ms=1000)
