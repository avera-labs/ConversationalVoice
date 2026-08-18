"""Whole-part music detection and deterministic frame smoothing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import MusicPolicy, QualityPolicy
from .intervals import Interval, rational_to_milliseconds


class MusicDetector(Protocol):
    model_name: str

    def detect(
        self, waveform: np.ndarray, *, sample_rate: int, duration_ms: int
    ) -> tuple[Interval, ...]: ...


def smooth_music_probabilities(
    probabilities: np.ndarray,
    *,
    probability_threshold: float,
    hop_length: int,
    sample_rate: int,
    minimum_interval_ms: int,
    gap_fill_ms: int,
    duration_ms: int,
) -> tuple[Interval, ...]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("music probabilities are invalid")
    active = values >= probability_threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    intervals = [
        Interval(
            rational_to_milliseconds(start_frame * hop_length, sample_rate),
            min(duration_ms, rational_to_milliseconds(end_frame * hop_length, sample_rate)),
        )
        for start_frame, end_frame in runs
        if min(duration_ms, rational_to_milliseconds(end_frame * hop_length, sample_rate))
        > rational_to_milliseconds(start_frame * hop_length, sample_rate)
    ]
    merged: list[Interval] = []
    for interval in intervals:
        if merged and interval.start_ms - merged[-1].end_ms <= gap_fill_ms:
            merged[-1] = Interval(merged[-1].start_ms, interval.end_ms)
        else:
            merged.append(interval)
    return tuple(item for item in merged if item.duration_ms > minimum_interval_ms)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class KerasMusicDetector:
    """Lazily loaded BLSTM speech/music detector with pinned local artifacts."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        music_policy: MusicPolicy,
        quality_policy: QualityPolicy,
    ) -> None:
        self.model_name = music_policy.model_name
        self._cache_dir = cache_dir
        self._music_policy = music_policy
        self._quality_policy = quality_policy
        self._model = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _artifact(self, filename: str, expected_sha256: str) -> Path:
        path = self._cache_dir / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("music model artifact is missing")
        if expected_sha256 and _sha256(path) != expected_sha256:
            raise RuntimeError("music model artifact checksum does not match")
        return path

    def _load(self) -> None:
        if self._model is not None:
            return
        import keras

        model_path = self._artifact(self._music_policy.model_filename, self._music_policy.model_sha256)
        mean_path = self._artifact(self._music_policy.mean_filename, self._music_policy.mean_sha256)
        std_path = self._artifact(self._music_policy.std_filename, self._music_policy.std_sha256)
        mean = np.load(mean_path, allow_pickle=False)
        std = np.load(std_path, allow_pickle=False)
        if mean.shape != (self._music_policy.mel_bins,) or std.shape != mean.shape:
            raise RuntimeError("music normalization artifact shape is invalid")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std == 0):
            raise RuntimeError("music normalization artifact values are invalid")
        model = keras.Sequential(
            [
                keras.Input(shape=(None, self._music_policy.mel_bins)),
                *(
                    keras.layers.Bidirectional(
                        keras.layers.LSTM(
                            units,
                            dropout=0.054614,
                            implementation=1,
                            recurrent_activation="hard_sigmoid",
                            return_sequences=True,
                        )
                    )
                    for units in (175, 25, 100, 50)
                ),
                keras.layers.TimeDistributed(
                    keras.layers.Dense(2, activation="sigmoid")
                ),
            ],
            name="speech_music_detection_blstm",
        )
        model.load_weights(model_path)
        self._model = model
        self._mean = mean.astype(np.float64)
        self._std = std.astype(np.float64)

    def validate_artifacts(self) -> None:
        checksums = (
            self._music_policy.model_sha256,
            self._music_policy.mean_sha256,
            self._music_policy.std_sha256,
        )
        if not all(checksums):
            raise RuntimeError("music model artifact checksums are not configured")
        self._artifact(self._music_policy.model_filename, self._music_policy.model_sha256)
        self._artifact(self._music_policy.mean_filename, self._music_policy.mean_sha256)
        self._artifact(self._music_policy.std_filename, self._music_policy.std_sha256)

    def detect(
        self, waveform: np.ndarray, *, sample_rate: int, duration_ms: int
    ) -> tuple[Interval, ...]:
        self._load()
        import librosa

        samples = np.asarray(waveform, dtype=np.float32)
        if (
            samples.ndim != 1
            or samples.size == 0
            or not np.all(np.isfinite(samples))
            or sample_rate <= 0
            or duration_ms <= 0
        ):
            raise ValueError("music detector input is invalid")
        if sample_rate != self._music_policy.sample_rate:
            samples = librosa.resample(
                samples,
                orig_sr=sample_rate,
                target_sr=self._music_policy.sample_rate,
            )
        spectrum = np.abs(
            librosa.stft(
                samples,
                n_fft=self._music_policy.fft_size,
                hop_length=self._music_policy.hop_length,
            )
        ) ** 2
        mel_filter = librosa.filters.mel(
            sr=self._music_policy.sample_rate,
            n_fft=self._music_policy.fft_size,
            n_mels=self._music_policy.mel_bins,
            fmin=self._music_policy.min_frequency_hz,
            fmax=self._music_policy.max_frequency_hz,
        )
        bands = librosa.power_to_db(mel_filter @ spectrum, ref=1.0, amin=1e-7)
        normalized = (bands - self._mean[:, None]) / self._std[:, None]
        prediction = np.asarray(self._model.predict(normalized.T[None, ...], batch_size=1, verbose=0))[0]
        if prediction.ndim != 2 or 2 not in prediction.shape:
            raise RuntimeError("music model output shape is invalid")
        channels_by_frames = prediction.T if prediction.shape[-1] == 2 else prediction
        music_probabilities = channels_by_frames[1]
        return smooth_music_probabilities(
            music_probabilities,
            probability_threshold=self._quality_policy.music_probability_threshold,
            hop_length=self._music_policy.hop_length,
            sample_rate=self._music_policy.sample_rate,
            minimum_interval_ms=self._quality_policy.min_music_interval_ms,
            gap_fill_ms=self._quality_policy.music_gap_fill_ms,
            duration_ms=duration_ms,
        )

    def close(self) -> None:
        self._model = None
        self._mean = None
        self._std = None
