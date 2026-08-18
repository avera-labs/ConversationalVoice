import numpy as np


def merge(
    stitched: np.ndarray, current: np.ndarray, overlap: int, crossfade: int
) -> np.ndarray:
    if overlap <= 0:
        return np.concatenate((stitched, current), axis=1)
    overlap = min(overlap, stitched.shape[1], current.shape[1])
    crossfade = min(crossfade, overlap)
    start = stitched.shape[1] - overlap
    fade = np.linspace(0, 1, crossfade, dtype=np.float32)[None, :]
    blended = (
        stitched[:, start : start + crossfade] * (1 - fade)
        + current[:, :crossfade] * fade
    )
    return np.concatenate(
        (stitched[:, :start], blended, current[:, crossfade:]), axis=1
    )
