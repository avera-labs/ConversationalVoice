from datetime import UTC, datetime
from types import SimpleNamespace

import voice_pipeline_score_completed_chunks.service as service_module
from voice_pipeline_score_completed_chunks.service import (
    ChunkScoreService,
    scoring_code_fingerprint,
)


def test_scoring_code_fingerprint_is_stable_sha256() -> None:
    first = scoring_code_fingerprint()
    second = scoring_code_fingerprint()
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_score_preserves_evaluation_generation_time_and_source_time(monkeypatch):
    source_time = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    evaluation_time = "2026-09-02T10:20:30+00:00"
    captured: dict[str, object] = {}

    def fake_build_score_report(**kwargs):
        captured.update(kwargs)
        return {"generated_at": evaluation_time}

    monkeypatch.setattr(service_module, "build_score_report", fake_build_score_report)
    monkeypatch.setattr(service_module, "render_artifacts", lambda *_args: {})

    manifest_component = SimpleNamespace(manifest=lambda: {})
    service = ChunkScoreService.__new__(ChunkScoreService)
    service.acoustic = SimpleNamespace(
        score_chunk=lambda *_args, **_kwargs: ([], [], [])
    )
    service.interaction = SimpleNamespace(
        score_chunk=lambda *_args, **_kwargs: ([], [], [], []),
        manifest=lambda: {},
    )
    service.audio_tag = SimpleNamespace(
        score_chunk=lambda *_args, **_kwargs: ([], []),
        evaluator=manifest_component,
    )
    service._score_asr = lambda _chunk: ([], [])
    service.nisqa = manifest_component
    service.dnsmos = manifest_component
    service.speaker = manifest_component
    service.asr = manifest_component
    service.model_fingerprint = "model-fingerprint"
    service.code_fingerprint = "code-fingerprint"
    chunk = SimpleNamespace(
        chunk_id="chunk-1",
        language="en",
        updated_at=source_time,
    )

    report, artifacts = service.score(chunk)

    assert report["generated_at"] == evaluation_time
    assert artifacts == {}
    assert captured["manifest"]["source_updated_at"] == source_time.isoformat()
