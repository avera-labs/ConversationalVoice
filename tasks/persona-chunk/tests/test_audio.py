import hashlib
import wave

import pytest

from voice_pipeline_persona_chunk.audio import validate_wav


def write_wav(path, *, frames=16000, rate=16000):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\0\0" * frames)


def test_validate_wav_returns_byte_identity(tmp_path):
    path = tmp_path / "audio.wav"
    write_wav(path)
    data = path.read_bytes()
    result = validate_wav(path, duration_ms=1000)
    assert result.size_bytes == len(data)
    assert result.sha256 == hashlib.sha256(data).hexdigest()


def test_validate_wav_rejects_wrong_rate_or_duration(tmp_path):
    path = tmp_path / "audio.wav"
    write_wav(path, rate=8000, frames=8000)
    with pytest.raises(ValueError, match="format or duration"):
        validate_wav(path, duration_ms=1000)
