from dataclasses import FrozenInstanceError

import pytest
from voice_pipeline_task_contracts import (
    ALL_TASKS,
    DIARIZE_AUDIO_PART,
    EXTEND_CHUNK,
    PERSONA_CHUNK,
    QUALITY_FILTER_AUDIO_PART,
    SEPARATE_CHUNK,
    SPLIT_RAW_AUDIO_INTO_PARTS,
    TASKS_BY_NAME,
    TRANSCRIBE_CHUNK,
    get_task_contract,
)


def test_split_raw_audio_into_parts_contract() -> None:
    assert SPLIT_RAW_AUDIO_INTO_PARTS.name == "split_raw_audio_into_parts"
    assert SPLIT_RAW_AUDIO_INTO_PARTS.queue == "split_raw_audio_into_parts"
    assert SPLIT_RAW_AUDIO_INTO_PARTS.uuid_argument == "raw_audio_id"


def test_diarize_audio_part_contract() -> None:
    assert DIARIZE_AUDIO_PART.name == "diarize_audio_part"
    assert DIARIZE_AUDIO_PART.queue == "diarize_audio_part"
    assert DIARIZE_AUDIO_PART.uuid_argument == "audio_part_id"


def test_quality_filter_audio_part_contract() -> None:
    assert QUALITY_FILTER_AUDIO_PART.name == "quality_filter_audio_part"
    assert QUALITY_FILTER_AUDIO_PART.queue == "quality_filter_audio_part"
    assert QUALITY_FILTER_AUDIO_PART.uuid_argument == "audio_part_id"


def test_separate_chunk_contract() -> None:
    assert SEPARATE_CHUNK.name == "separate_chunk"
    assert SEPARATE_CHUNK.queue == "separate_chunk"
    assert SEPARATE_CHUNK.uuid_argument == "chunk_id"


def test_transcribe_chunk_contract() -> None:
    assert TRANSCRIBE_CHUNK.name == "transcribe_chunk"
    assert TRANSCRIBE_CHUNK.queue == "transcribe_chunk"
    assert TRANSCRIBE_CHUNK.uuid_argument == "chunk_id"


def test_persona_chunk_contract() -> None:
    assert PERSONA_CHUNK.name == "persona_chunk"
    assert PERSONA_CHUNK.queue == "persona_chunk"
    assert PERSONA_CHUNK.uuid_argument == "chunk_id"


def test_extend_chunk_contract() -> None:
    assert EXTEND_CHUNK.name == "extend_chunk"
    assert EXTEND_CHUNK.queue == "extend_chunk"
    assert EXTEND_CHUNK.uuid_argument == "chunk_id"


def test_registered_names_and_queues_are_unique() -> None:
    assert len({contract.name for contract in ALL_TASKS}) == len(ALL_TASKS)
    assert len({contract.queue for contract in ALL_TASKS}) == len(ALL_TASKS)


def test_contracts_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        SPLIT_RAW_AUDIO_INTO_PARTS.name = "changed"  # type: ignore[misc]

    with pytest.raises(TypeError):
        TASKS_BY_NAME["changed"] = SPLIT_RAW_AUDIO_INTO_PARTS  # type: ignore[index]


def test_contract_lookup_accepts_only_registered_names() -> None:
    for contract in ALL_TASKS:
        assert get_task_contract(contract.name) is contract

    assert get_task_contract("unregistered_task") is None
