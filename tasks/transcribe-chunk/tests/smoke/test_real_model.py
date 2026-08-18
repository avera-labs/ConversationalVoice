import os

import numpy as np
import pytest

from voice_pipeline_transcribe_chunk.config import load_settings
from voice_pipeline_transcribe_chunk.model import ParakeetModel

pytestmark = pytest.mark.model_smoke


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_SMOKE_TEST") != "1",
    reason="Set RUN_MODEL_SMOKE_TEST=1 through tests/smoke/run.sh.",
)
def test_pinned_parakeet_loads_and_transcribes_on_cuda():
    import torch

    assert torch.cuda.is_available()
    settings = load_settings(
        {
            "DATABASE_URL": "postgresql://example/db",
            "CELERY_BROKER_URL": "redis://example/0",
            "S3_BUCKET": "bucket",
            "S3_REGION": "us-east-1",
        }
    )
    policy = settings.policy.model_copy(
        update={"model": settings.policy.model.model_copy(update={"device": "cuda"})}
    )
    model = ParakeetModel(policy)
    try:
        words = model.transcribe(np.zeros(16000, dtype=np.float32))
        assert isinstance(words, list)
    finally:
        model.close()
