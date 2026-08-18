import os

import numpy as np
import pytest

from voice_pipeline_separate_chunk.config import load_settings
from voice_pipeline_separate_chunk.model import DialogueSidon

pytestmark = pytest.mark.model_smoke
if os.environ.get("RUN_MODEL_SMOKE_TEST") != "1":
    pytest.skip(
        "Set RUN_MODEL_SMOKE_TEST=1 through tests/smoke/run.sh.",
        allow_module_level=True,
    )


def test_pinned_dialogue_sidon_cuda_inference():
    settings = load_settings(
        {
            "DATABASE_URL": "postgresql://example/db",
            "CELERY_BROKER_URL": "redis://example/0",
            "S3_BUCKET": "bucket",
            "S3_REGION": "us-east-1",
        }
    )
    samples = np.random.default_rng(1).normal(0, 0.03, 20 * 16000).astype(np.float32)
    tracks, sample_rate = DialogueSidon(settings.policy.model).separate(samples, seed=1)
    assert sample_rate == 24000
    assert tracks.shape[0] == 2
    assert tracks.shape[1] > 0
    assert np.isfinite(tracks).all()
