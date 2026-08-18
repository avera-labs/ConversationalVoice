"""Waveform Amplitude Distribution Analysis SNR."""

from __future__ import annotations

import numpy as np

_DB_VALUES = np.arange(-20, 101, dtype=np.float64)
_G_VALUES = np.array([
    0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006,
    0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272,
    0.41526426, 0.4178192, 0.42077252, 0.42452799, 0.42918886, 0.43510373,
    0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236,
    0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496,
    0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985,
    0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349,
    1.01047155, 1.0362095, 1.06136425, 1.08579312, 1.1094819, 1.13277995,
    1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313,
    1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935,
    1.3605727, 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619,
    1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938,
    1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915,
    1.5229097, 1.528578, 1.53389835, 1.5391211, 1.5439065, 1.54858517,
    1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767,
    1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477,
    1.5941969, 1.59693155, 1.599446, 1.60185011, 1.60408668, 1.60627134,
    1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074, 1.61688905,
    1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143,
    1.62632463, 1.6274027, 1.62842767, 1.62945532, 1.6303307, 1.63128026,
    1.63204102,
], dtype=np.float64)


def wada_snr(waveform: np.ndarray) -> float:
    samples = np.asarray(waveform, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("waveform samples are invalid")
    peak = float(np.max(np.abs(samples)))
    if peak <= 0.0:
        raise ValueError("waveform samples are silent")
    epsilon = 1e-10
    normalized = samples / peak
    magnitude = np.maximum(np.abs(normalized), epsilon)
    statistic = float(np.log(max(epsilon, float(magnitude.mean()))) - np.log(magnitude).mean())
    indices = np.flatnonzero(_G_VALUES < statistic)
    if indices.size == 0:
        estimated = float(_DB_VALUES[0])
    elif indices[-1] == len(_DB_VALUES) - 1:
        estimated = float(_DB_VALUES[-1])
    else:
        index = int(indices[-1])
        estimated = float(
            _DB_VALUES[index]
            + (statistic - _G_VALUES[index])
            / (_G_VALUES[index + 1] - _G_VALUES[index])
        )
    factor = 10 ** (estimated / 10)
    energy = float(np.sum(normalized**2))
    noise = energy / (1 + factor)
    signal = energy * factor / (1 + factor)
    result = float(10 * np.log10(signal / (noise + epsilon)))
    if not np.isfinite(result):
        raise ValueError("SNR result is not finite")
    return result
