import wave

import numpy as np
import pytest

from voice_pipeline_transcribe_chunk.audio import read_speaker_wav


def write_wav(path, *, rate=16000, frames=16000):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(np.zeros(frames, dtype="<i2").tobytes())


def test_reads_exact_pcm_wav(tmp_path):
    path = tmp_path / "speaker.wav"
    write_wav(path)
    audio = read_speaker_wav(path, duration_ms=1000)
    assert audio.samples.shape == (16000,)
    assert audio.size_bytes == path.stat().st_size
    assert len(audio.sha256) == 64


def test_rejects_wrong_sample_rate(tmp_path):
    path = tmp_path / "speaker.wav"
    write_wav(path, rate=8000, frames=8000)
    with pytest.raises(ValueError):
        read_speaker_wav(path, duration_ms=1000)
