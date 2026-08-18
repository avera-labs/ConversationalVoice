import wave

import numpy as np

from voice_pipeline_separate_chunk.audio import write_outputs


def test_raw_float_amplitude_above_one_is_scaled_without_rejection(tmp_path) -> None:
    tracks = np.stack(
        (
            np.full(24000, 2.0, dtype=np.float32),
            np.full(24000, -1.5, dtype=np.float32),
        )
    )
    paths = (tmp_path / "speaker-0.wav", tmp_path / "speaker-1.wav")

    metadata = write_outputs(tracks, 24000, paths, duration_ms=1000, peak=0.9)

    assert all(item["size_bytes"] > 0 for item in metadata)
    for path in paths:
        with wave.open(str(path), "rb") as reader:
            samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
        assert np.max(np.abs(samples.astype(np.int32))) <= round(0.9 * 32768)
