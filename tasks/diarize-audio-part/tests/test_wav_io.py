import hashlib
import wave
from pathlib import Path

import numpy as np
import pytest

from voice_pipeline_diarize_audio_part.speaker_reference import ReferenceSegment
from voice_pipeline_diarize_audio_part.wav_io import (
    WavError,
    read_normalized_wav,
    write_reference_wav,
)


def write_source(path: Path, samples: np.ndarray, *, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.astype("<i2", copy=False).tobytes())


def read_samples(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as reader:
        payload = reader.readframes(reader.getnframes())
        return (
            np.frombuffer(payload, dtype="<i2"),
            reader.getframerate(),
            reader.getnchannels(),
            reader.getsampwidth(),
        )


def test_reference_wav_preserves_samples_and_inserts_exact_silence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    samples = np.concatenate(
        (
            np.full(16000, 100, dtype="<i2"),
            np.full(16000, 200, dtype="<i2"),
            np.full(16000, 300, dtype="<i2"),
        )
    )
    write_source(source, samples)
    audio = read_normalized_wav(source, expected_duration_ms=3000)
    destination = tmp_path / "reference.wav"
    result = write_reference_wav(
        audio,
        destination,
        (ReferenceSegment(0, 1000), ReferenceSegment(2000, 3000)),
        inter_segment_silence_ms=500,
    )

    output, rate, channels, width = read_samples(destination)
    assert (rate, channels, width) == (16000, 1, 2)
    assert np.all(output[:16000] == 100)
    assert np.all(output[16000:24000] == 0)
    assert np.all(output[24000:] == 300)
    assert result.duration_ms == 2500
    assert result.size_bytes == destination.stat().st_size
    assert result.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_single_segment_has_no_leading_or_trailing_silence(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    write_source(source, np.full(16000, 123, dtype="<i2"))
    audio = read_normalized_wav(source, expected_duration_ms=1000)
    destination = tmp_path / "reference.wav"
    result = write_reference_wav(
        audio,
        destination,
        (ReferenceSegment(0, 1000),),
        inter_segment_silence_ms=500,
    )
    output, *_ = read_samples(destination)
    assert np.all(output == 123)
    assert result.duration_ms == 1000


def test_read_rejects_wrong_format_or_duration(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    write_source(source, np.zeros(8000, dtype="<i2"), sample_rate=8000)
    with pytest.raises(WavError):
        read_normalized_wav(source, expected_duration_ms=1000)

    write_source(source, np.zeros(16000, dtype="<i2"))
    with pytest.raises(WavError, match="duration"):
        read_normalized_wav(source, expected_duration_ms=999)


def test_write_rejects_out_of_bounds_segment_and_removes_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    write_source(source, np.zeros(16000, dtype="<i2"))
    audio = read_normalized_wav(source, expected_duration_ms=1000)
    destination = tmp_path / "reference.wav"
    with pytest.raises(WavError):
        write_reference_wav(
            audio,
            destination,
            (ReferenceSegment(0, 1001),),
            inter_segment_silence_ms=500,
        )
    assert not destination.exists()
