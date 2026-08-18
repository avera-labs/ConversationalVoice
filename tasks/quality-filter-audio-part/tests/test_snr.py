import numpy as np
import pytest

from voice_pipeline_quality_filter_audio_part.snr import wada_snr


def test_wada_snr_returns_finite_value_for_non_silent_waveform() -> None:
    waveform = np.sin(np.linspace(0, 100, 16000, dtype=np.float64))
    assert np.isfinite(wada_snr(waveform))


@pytest.mark.parametrize(
    "waveform",
    [np.array([]), np.zeros(100), np.array([0.0, np.nan])],
)
def test_invalid_waveform_is_rejected(waveform: np.ndarray) -> None:
    with pytest.raises(ValueError):
        wada_snr(waveform)
