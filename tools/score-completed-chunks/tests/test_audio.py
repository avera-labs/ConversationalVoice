from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import make_wav
from voice_pipeline_score_completed_chunks.audio import (
    extract_active_audio,
    merge_intervals,
    read_wav,
    slice_wav_payload,
)
from voice_pipeline_score_completed_chunks.errors import ScoringError
from voice_pipeline_score_completed_chunks.nisqa import NisqaScorer


def test_extract_active_audio_merges_overlap_and_adds_separator() -> None:
    audio = read_wav(make_wav(duration_ms=4000, sample_rate_hz=16000))
    transcript = {
        "utterances": [
            {"speaker_id": 0, "start_ms": 0, "end_ms": 900},
            {"speaker_id": 0, "start_ms": 800, "end_ms": 1800},
            {"speaker_id": 1, "start_ms": 1200, "end_ms": 2200},
            {"speaker_id": 0, "start_ms": 2500, "end_ms": 3300},
        ]
    }
    active = extract_active_audio(audio, transcript, speaker_id=0)
    assert active.active_duration_ms == 2600
    assert active.interval_count == 2
    assert active.samples.size == round(2700 * 16000 / 1000)


def test_short_active_speech_is_rejected() -> None:
    audio = read_wav(make_wav(duration_ms=2000))
    with pytest.raises(ScoringError, match="insufficient_active_speech"):
        extract_active_audio(
            audio,
            {"utterances": [{"speaker_id": 0, "start_ms": 0, "end_ms": 500}]},
            speaker_id=0,
        )


def test_slice_wav_payload_matches_worker_encoding() -> None:
    payload = make_wav(duration_ms=2000, sample_rate_hz=16000)
    sliced = slice_wav_payload(payload, segments=((250, 1250),))
    decoded = read_wav(sliced, expected_rate=16000)
    assert decoded.duration_ms == 1000
    assert len(sliced) == 32044
    assert hashlib.sha256(sliced).hexdigest()


def test_merge_intervals_rejects_out_of_bounds() -> None:
    with pytest.raises(ScoringError, match="active_interval_out_of_bounds"):
        merge_intervals([(0, 2001)], duration_ms=2000)


def test_nisqa_splits_long_audio_and_merges_short_tail(monkeypatch) -> None:
    sample_rate = 10
    samples = np.zeros(1005, dtype=np.float32)
    scorer = NisqaScorer.__new__(NisqaScorer)
    observed: list[int] = []

    def score_window(window, sample_rate_hz):
        assert sample_rate_hz == sample_rate
        observed.append(window.size)
        return np.full(5, window.size, dtype=np.float64)

    monkeypatch.setattr(scorer, "_score_window", score_window)
    result = scorer.score(samples, sample_rate)

    assert observed == [500, 505]
    assert result["nisqa_window_count"] == 2
    assert result["nisqa_mos"] == pytest.approx((500**2 + 505**2) / 1005)
