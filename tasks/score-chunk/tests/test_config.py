from pathlib import Path

from voice_pipeline_score_chunk.config import load_settings


def test_default_policy_is_cpu_only_and_uses_openrouter_asr(tmp_path: Path) -> None:
    settings = load_settings(
        {
            "DATABASE_URL": "postgresql://database/test",
            "CELERY_BROKER_URL": "redis://broker/0",
            "S3_BUCKET": "bucket",
            "S3_REGION": "us-east-1",
            "OPENROUTER_API_KEY": "secret",
        }
    )
    assert settings.policy.config_version == "chunk-score-v2"
    assert settings.policy.asr.model == "qwen/qwen3-asr-1.7b"
    assert settings.policy.audio_tag.model == "google/gemini-3.7-flash"
    assert settings.policy.audio_tag.workers == 4
    assert "cuda" not in settings.policy.model_dump_json().lower()
