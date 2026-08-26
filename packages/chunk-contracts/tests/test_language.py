import pytest
from voice_pipeline_chunk_contracts import ChunkContractError, parse_chunk_language


@pytest.mark.parametrize("value", ["en", "zh", "es", "yue", "x-unsupported"])
def test_accepts_any_non_empty_canonical_language_identifier(value):
    assert parse_chunk_language(value) == value


@pytest.mark.parametrize("value", ["", " en", "en ", None, 1])
def test_rejects_malformed_language_identifiers(value):
    with pytest.raises(ChunkContractError):
        parse_chunk_language(value)
