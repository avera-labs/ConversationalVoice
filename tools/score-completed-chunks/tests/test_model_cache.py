from __future__ import annotations

from voice_pipeline_score_completed_chunks import nisqa as nisqa_module
from voice_pipeline_score_completed_chunks.nisqa import NisqaScorer
from voice_pipeline_score_completed_chunks.speaker_similarity import (
    SpeakerSimilarityScorer,
)


def test_nisqa_uses_configured_model_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        nisqa_module,
        "ensure_model_file",
        lambda specification, destination: destination,
    )

    scorer = NisqaScorer(tmp_path)

    assert scorer.model_path == tmp_path / "nisqa" / "nisqa.tar"


def test_wavlm_uses_configured_model_cache(tmp_path, monkeypatch) -> None:
    from transformers import AutoFeatureExtractor, WavLMForXVector

    calls: list[dict] = []

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_from_pretrained(*args, **kwargs):
        calls.append(kwargs)
        return FakeModel()

    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(WavLMForXVector, "from_pretrained", fake_from_pretrained)

    SpeakerSimilarityScorer("cpu", tmp_path)

    assert [call["cache_dir"] for call in calls] == [
        str(tmp_path / "huggingface"),
        str(tmp_path / "huggingface"),
    ]
