import pytest

from voice_pipeline_extend_chunk.config import load_settings


@pytest.fixture
def settings():
    return load_settings(
        {
            "DATABASE_URL": "postgresql://example/db",
            "CELERY_BROKER_URL": "redis://example/0",
            "S3_BUCKET": "bucket",
            "S3_REGION": "us-east-1",
            "OPENROUTER_API_KEY": "openrouter-key",
        }
    )


@pytest.fixture
def policy(settings):
    return settings.policy
