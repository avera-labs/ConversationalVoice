import io
import wave

from voice_pipeline_reconstruct_chunk.audio import concatenate_reference, read_wav_bytes


def make_wav(frame: bytes, frame_count: int, rate: int = 16000):
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(frame * frame_count)
    return target.getvalue()


def test_reference_is_sample_then_exactly_one_second_silence_then_segment():
    sample = make_wav(b"\x01\x00", 1600)
    utterance = make_wav(b"\x02\x00", 3200)
    result = read_wav_bytes(
        concatenate_reference(sample, utterance), expected_rate=16000
    )
    assert result.frames[: 1600 * 2] == b"\x01\x00" * 1600
    assert result.frames[1600 * 2 : (1600 + 16000) * 2] == bytes(16000 * 2)
    assert result.frames[(1600 + 16000) * 2 :] == b"\x02\x00" * 3200
