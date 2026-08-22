import pytest

from voice_pipeline_reconstruct_chunk.config import load_settings


@pytest.fixture
def settings():
    return load_settings(
        {
            "DATABASE_URL": "postgresql://localhost/test",
            "CELERY_BROKER_URL": "redis://localhost:6379/0",
            "S3_BUCKET": "bucket",
            "S3_REGION": "us-east-1",
            "OPENROUTER_API_KEY": "test-key",
        }
    )


@pytest.fixture
def policy(settings):
    return settings.policy
