import os

import numpy as np
import pytest

from voice_pipeline_separate_chunk.config import load_settings
from voice_pipeline_separate_chunk.model import DialogueSidon

pytestmark = pytest.mark.model_capacity
if os.environ.get("RUN_MODEL_CAPACITY_TEST") != "1":
    pytest.skip(
        "Set RUN_MODEL_CAPACITY_TEST=1 through tests/capacity/run.sh.",
        allow_module_level=True,
    )


def test_maximum_window_on_cuda() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()
    settings = load_settings(
        {
            "DATABASE_URL": "postgresql://example/db",
            "CELERY_BROKER_URL": "redis://example/0",
            "S3_BUCKET": "bucket",
            "S3_REGION": "us-east-1",
        }
    )
    model = DialogueSidon(settings.policy.model)
    samples = np.random.default_rng(1).normal(0, 0.05, 120000 * 16).astype(np.float32)
    try:
        tracks, sample_rate = model.separate(samples, seed=1)
        assert sample_rate == 24000
        assert tracks.shape[0] == 2
        assert tracks.shape[1] > 120000 * sample_rate // 1000 - 1000
        assert np.isfinite(tracks).all()
    finally:
        model.close()
