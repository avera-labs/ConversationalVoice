import pytest

from voice_pipeline_transcribe_chunk.config import load_settings


@pytest.fixture
def policy():
    return load_settings(
        {
            "DATABASE_URL": "postgresql://example/db",
            "CELERY_BROKER_URL": "redis://example/0",
            "S3_BUCKET": "bucket",
            "S3_REGION": "us-east-1",
        }
    ).policy
