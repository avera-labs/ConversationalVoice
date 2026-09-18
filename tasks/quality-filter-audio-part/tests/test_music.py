from pathlib import Path

import numpy as np
import pytest

from voice_pipeline_quality_filter_audio_part.config import MusicPolicy
from voice_pipeline_quality_filter_audio_part.intervals import Interval
from voice_pipeline_quality_filter_audio_part.music import KerasMusicDetector, smooth_music_probabilities


def smooth(values: list[float], *, minimum: int = 2000, gap: int = 600):
    return smooth_music_probabilities(
        np.array(values),
        probability_threshold=0.2,
        hop_length=1,
        sample_rate=1000,
        minimum_interval_ms=minimum,
        gap_fill_ms=gap,
        duration_ms=len(values),
    )


def test_probability_threshold_is_inclusive() -> None:
    assert smooth([0.2] * 2001) == (Interval(0, 2001),)


def test_exact_minimum_music_duration_is_removed() -> None:
    assert smooth([0.2] * 2000) == ()


def test_gap_at_threshold_is_filled() -> None:
    values = [1.0] * 1001 + [0.0] * 600 + [1.0] * 1001
    assert smooth(values, minimum=1000) == (Interval(0, 2602),)


def test_gap_above_threshold_is_not_filled() -> None:
    values = [1.0] * 1001 + [0.0] * 601 + [1.0] * 1001
    assert smooth(values, minimum=1000) == (Interval(0, 1001), Interval(1602, 2603))


def test_non_finite_and_wrong_shape_predictions_are_rejected() -> None:
    with pytest.raises(ValueError):
        smooth_music_probabilities(
            np.array([0.3, np.nan]),
            probability_threshold=0.2,
            hop_length=1,
            sample_rate=1000,
            minimum_interval_ms=0,
            gap_fill_ms=0,
            duration_ms=2,
        )
    with pytest.raises(ValueError):
        smooth_music_probabilities(
            np.array([[0.3]]),
            probability_threshold=0.2,
            hop_length=1,
            sample_rate=1000,
            minimum_interval_ms=0,
            gap_fill_ms=0,
            duration_ms=1,
        )


def music_policy(**overrides) -> MusicPolicy:
    values = {
        "model_name": "model",
        "model_filename": "model.h5",
        "mean_filename": "mean.npy",
        "std_filename": "std.npy",
        "model_sha256": "",
        "mean_sha256": "",
        "std_sha256": "",
        "sample_rate": 22050,
        "fft_size": 1024,
        "hop_length": 512,
        "mel_bins": 80,
        "min_frequency_hz": 27.5,
        "max_frequency_hz": 8000.0,
    }
    values.update(overrides)
    return MusicPolicy(**values)


def test_runtime_requires_all_checksums(tmp_path: Path, quality_policy) -> None:
    detector = KerasMusicDetector(
        cache_dir=tmp_path,
        music_policy=music_policy(),
        quality_policy=quality_policy,
    )
    with pytest.raises(RuntimeError, match="checksums"):
        detector.validate_artifacts()


def default_music_policy() -> MusicPolicy:
    import tomllib

    from voice_pipeline_quality_filter_audio_part.config import DEFAULT_POLICY_PATH

    with DEFAULT_POLICY_PATH.open("rb") as file:
        return MusicPolicy(**tomllib.load(file)["music"])


def synthetic_signal(seconds: float, *, sample_rate: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    count = int(round(seconds * sample_rate))
    time = np.arange(count, dtype=np.float32) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * time) + 0.2 * np.sin(2 * np.pi * 660.0 * time)
    return (tone + 0.05 * generator.standard_normal(count)).astype(np.float32)


def test_detect_keeps_full_frame_sequence_on_every_call_with_real_model(
    monkeypatch, quality_policy
) -> None:
    """Regression: Keras 3 ``model.predict`` returned the full sequence for the
    first call in a process but one frame for any later call with a different
    sequence length, so the music gate was silently off for nearly every window
    after the first one a worker processed."""
    import voice_pipeline_quality_filter_audio_part.music as music_module
    from voice_pipeline_quality_filter_audio_part.config import DEFAULT_MUSIC_MODEL_CACHE_DIR

    policy = default_music_policy()
    frame_counts: list[int] = []
    real_smooth = music_module.smooth_music_probabilities

    def spy(probabilities, **kwargs):
        frame_counts.append(len(probabilities))
        return real_smooth(probabilities, **kwargs)

    monkeypatch.setattr(music_module, "smooth_music_probabilities", spy)
    detector = KerasMusicDetector(
        cache_dir=DEFAULT_MUSIC_MODEL_CACHE_DIR,
        music_policy=policy,
        quality_policy=quality_policy,
    )
    detector.validate_artifacts()
    try:
        for seconds, seed in ((10.0, 1), (6.0, 2), (10.0, 3)):
            waveform = synthetic_signal(seconds, sample_rate=16000, seed=seed)
            detector.detect(waveform, sample_rate=16000, duration_ms=int(seconds * 1000))
    finally:
        detector.close()

    def expected_frames(seconds: float) -> int:
        return 1 + int(round(seconds * policy.sample_rate)) // policy.hop_length

    assert frame_counts == [expected_frames(10.0), expected_frames(6.0), expected_frames(10.0)]
    assert min(frame_counts) > 1


def test_detect_calls_model_directly_instead_of_predict(quality_policy) -> None:
    policy = default_music_policy()

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, inputs, training=None):
            self.calls.append("call")
            assert training is False
            frames = np.asarray(inputs).shape[1]
            return np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (1, frames, 1))

        def predict(self, inputs, **kwargs):
            self.calls.append("predict")
            return np.array([[[0.0, 1.0]]], dtype=np.float32)

    detector = KerasMusicDetector(
        cache_dir=Path("/nonexistent"),
        music_policy=policy,
        quality_policy=quality_policy,
    )
    model = FakeModel()
    detector._model = model
    detector._mean = np.zeros(policy.mel_bins)
    detector._std = np.ones(policy.mel_bins)
    waveform = synthetic_signal(5.0, sample_rate=policy.sample_rate, seed=4)
    intervals = detector.detect(waveform, sample_rate=policy.sample_rate, duration_ms=5000)
    assert model.calls == ["call"]
    assert intervals == (Interval(0, 5000),)
