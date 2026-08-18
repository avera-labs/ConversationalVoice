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
