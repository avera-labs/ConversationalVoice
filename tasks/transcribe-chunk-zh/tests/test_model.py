from types import SimpleNamespace

import numpy as np
import pytest
import torch

from voice_pipeline_transcribe_chunk_zh.model import ParaformerModel


class FakeAutoModel:
    def __init__(self, adapter, output, logits, token_list=None):
        self.adapter = adapter
        self.model = SimpleNamespace(blank_id=0, sos=1, eos=2)
        self.output = output
        self.logits = logits
        vocabulary = token_list or ["<blank>", "<s>", "</s>", "你", "好"]
        self.kwargs = {
            "tokenizer": SimpleNamespace(
                ids2tokens=lambda token_ids: [vocabulary[index] for index in token_ids]
            )
        }

    def generate(self, **_kwargs):
        self.adapter._captured_logits = self.logits
        return self.output


def make_adapter(policy, output, logits, token_list=None):
    adapter = ParaformerModel(policy)
    adapter._auto_model = FakeAutoModel(adapter, output, logits, token_list)
    adapter._torch = torch
    return adapter


def test_transcribe_returns_per_character_timestamps_and_posteriors(policy):
    logits = torch.tensor([[[0.0, 0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0, 5.0]]])
    model = make_adapter(
        policy,
        [{"text": "你 好", "timestamp": [[100, 220], [220, 410]]}],
        logits,
    )
    result = model.transcribe(np.zeros(1600, dtype=np.float32))
    assert [item.text for item in result] == ["你", "好"]
    assert [(item.start_seconds, item.end_seconds) for item in result] == [
        (0.1, 0.22),
        (0.22, 0.41),
    ]
    assert all(0.9 < item.confidence < 1 for item in result)


def test_confidence_count_mismatch_fails_closed(policy):
    model = make_adapter(
        policy,
        [{"text": "你 好", "timestamp": [[100, 220], [220, 410]]}],
        torch.tensor([[[9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]]]),
    )
    with pytest.raises(RuntimeError, match="confidence count mismatch"):
        model.transcribe(np.zeros(1600, dtype=np.float32))


def test_transcribe_allows_empty_decoding(policy):
    model = make_adapter(
        policy,
        [{"text": "", "timestamp": []}],
        torch.tensor([[[0.0, 0.0, 5.0, 0.0, 0.0]]]),
    )

    assert model.transcribe(np.zeros(1600, dtype=np.float32)) == []


def test_transcribe_aligns_mixed_chinese_and_merged_english_bpe(policy):
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
            ]
        ]
    )
    model = make_adapter(
        policy,
        [
            {
                "text": "中 connotation 文",
                "timestamp": [[100, 220], [220, 600], [600, 760]],
            }
        ],
        logits,
        ["<blank>", "<s>", "</s>", "中", "con@@", "nota@@", "tion", "文"],
    )

    result = model.transcribe(np.zeros(1600, dtype=np.float32))

    assert [item.text for item in result] == ["中", "connotation", "文"]
    assert [(item.start_seconds, item.end_seconds) for item in result] == [
        (0.1, 0.22),
        (0.22, 0.6),
        (0.6, 0.76),
    ]
    assert result[1].confidence == pytest.approx(
        min(logits[0, index].softmax(dim=-1).max().item() for index in (1, 2, 3))
    )


@pytest.mark.parametrize(
    ("method_name", "tuple_output"),
    [("_seaco_decode_with_ASF", False), ("cal_decoder_with_predictor", True)],
)
def test_decoder_hook_captures_supported_funasr_paths(
    policy, method_name, tuple_output
):
    scores = torch.tensor([[[0.0, 1.0]]])

    def decode(*_args, **_kwargs):
        return (scores, object()) if tuple_output else scores

    core = SimpleNamespace(**{method_name: decode})
    model = ParaformerModel(policy)
    model._install_confidence_hook(core)
    getattr(core, method_name)()
    assert torch.equal(model._captured_logits, scores)
