import pytest
from voice_pipeline_chunk_contracts import ChunkContractError, parse_chunk_language


@pytest.mark.parametrize(
    "value", ["en", "zh", "es", "ja", "zh-CN", "zh-Hant-TW", "es-419"]
)
def test_accepts_iso_639_1_language_tags(value):
    assert parse_chunk_language(value) == value


@pytest.mark.parametrize(
    "value",
    ["", " en", "en ", "EN", "en_US", "xx", "yue", "x-unsupported", None, 1],
)
def test_rejects_malformed_language_identifiers(value):
    with pytest.raises(ChunkContractError):
        parse_chunk_language(value)
