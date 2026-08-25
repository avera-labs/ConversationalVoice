import pytest
from voice_pipeline_chunk_contracts import ChunkContractError, parse_chunk_language


@pytest.mark.parametrize("value", ["en", "zh"])
def test_accepts_only_canonical_language_codes(value):
    assert parse_chunk_language(value) == value


@pytest.mark.parametrize("value", ["cn", "zh-CN", "zh_CN", "", None])
def test_rejects_noncanonical_language_codes(value):
    with pytest.raises(ChunkContractError):
        parse_chunk_language(value)
