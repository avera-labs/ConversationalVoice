from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort

from .audio import resample
from .errors import ScoringError
from .model_files import DNSMOS_P808, DNSMOS_PRIMARY, ensure_model_file

SAMPLING_RATE = 16000
INPUT_LENGTH_SECONDS = 9.01


class DnsmosScorer:
    def __init__(self, model_cache: Path) -> None:
        directory = model_cache / "dnsmos"
        primary = ensure_model_file(DNSMOS_PRIMARY, directory / DNSMOS_PRIMARY.name)
        p808 = ensure_model_file(DNSMOS_P808, directory / DNSMOS_P808.name)
        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        self.providers = providers
        self.primary = ort.InferenceSession(str(primary), providers=providers)
        self.p808 = ort.InferenceSession(str(p808), providers=providers)

    @staticmethod
    def _melspec(audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLING_RATE,
            n_fft=321,
            hop_length=160,
            n_mels=120,
        )
        return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T

    @staticmethod
    def _calibrate(sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
        calibrated_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovr)
        calibrated_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig)
        calibrated_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak)
        return float(calibrated_sig), float(calibrated_bak), float(calibrated_ovr)

    def score(self, samples: np.ndarray, sample_rate_hz: int) -> dict[str, object]:
        audio = resample(samples, sample_rate_hz, SAMPLING_RATE)
        if audio.size == 0:
            raise ScoringError("dnsmos_empty_audio")
        required = round(INPUT_LENGTH_SECONDS * SAMPLING_RATE)
        repeated = audio.size < required
        while audio.size < required:
            audio = np.append(audio, audio)
        hops = int(np.floor(audio.size / SAMPLING_RATE) - INPUT_LENGTH_SECONDS) + 1
        scores: list[tuple[float, float, float, float]] = []
        try:
            for index in range(hops):
                segment = audio[
                    index * SAMPLING_RATE : index * SAMPLING_RATE + required
                ]
                if segment.size < required:
                    continue
                features = segment.astype(np.float32, copy=False)[None, :]
                mel = self._melspec(segment[:-160]).astype(np.float32)[None, :, :]
                p808 = float(self.p808.run(None, {"input_1": mel})[0][0][0])
                raw_sig, raw_bak, raw_ovr = self.primary.run(
                    None, {"input_1": features}
                )[0][0]
                sig, bak, ovr = self._calibrate(raw_sig, raw_bak, raw_ovr)
                scores.append((sig, bak, ovr, p808))
        except Exception as exc:
            raise ScoringError("dnsmos_failed") from exc
        if not scores:
            raise ScoringError("dnsmos_no_windows")
        means = np.mean(np.asarray(scores, dtype=np.float64), axis=0)
        if not np.isfinite(means).all():
            raise ScoringError("dnsmos_non_finite")
        result = {
            "dnsmos_sig": float(means[0]),
            "dnsmos_bak": float(means[1]),
            "dnsmos_ovrl": float(means[2]),
            "dnsmos_p808": float(means[3]),
            "dnsmos_repeated_input": repeated,
            "dnsmos_window_count": len(scores),
        }
        if not all(
            math.isfinite(value)
            for key, value in result.items()
            if key.startswith("dnsmos_") and isinstance(value, float)
        ):
            raise ScoringError("dnsmos_non_finite")
        return result

    def manifest(self) -> dict[str, object]:
        return {
            "implementation": "microsoft/DNS-Challenge DNSMOS/dnsmos_local.py",
            "upstream_commit": "591184a9fcb2cbdec02520fed81a32bbbf9d73ff",
            "personalized": False,
            "providers": self.providers,
            "primary_weights": {
                "name": DNSMOS_PRIMARY.name,
                "sha256": DNSMOS_PRIMARY.sha256,
                "size_bytes": DNSMOS_PRIMARY.size_bytes,
            },
            "p808_weights": {
                "name": DNSMOS_P808.name,
                "sha256": DNSMOS_P808.sha256,
                "size_bytes": DNSMOS_P808.size_bytes,
            },
        }
