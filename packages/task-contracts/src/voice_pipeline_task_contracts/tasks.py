from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class TaskContract:
    """Stable routing metadata for a task with one UUID argument."""

    name: str
    queue: str
    uuid_argument: str


SPLIT_RAW_AUDIO_INTO_PARTS = TaskContract(
    name="split_raw_audio_into_parts",
    queue="split_raw_audio_into_parts",
    uuid_argument="raw_audio_id",
)

DIARIZE_AUDIO_PART = TaskContract(
    name="diarize_audio_part",
    queue="diarize_audio_part",
    uuid_argument="audio_part_id",
)

QUALITY_FILTER_AUDIO_PART = TaskContract(
    name="quality_filter_audio_part",
    queue="quality_filter_audio_part",
    uuid_argument="audio_part_id",
)

SEPARATE_CHUNK = TaskContract(
    name="separate_chunk",
    queue="separate_chunk",
    uuid_argument="chunk_id",
)

TRANSCRIBE_CHUNK = TaskContract(
    name="transcribe_chunk",
    queue="transcribe_chunk",
    uuid_argument="chunk_id",
)

PERSONA_CHUNK = TaskContract(
    name="persona_chunk",
    queue="persona_chunk",
    uuid_argument="chunk_id",
)

EXTEND_CHUNK = TaskContract(
    name="extend_chunk",
    queue="extend_chunk",
    uuid_argument="chunk_id",
)

ALL_TASKS = (
    SPLIT_RAW_AUDIO_INTO_PARTS,
    DIARIZE_AUDIO_PART,
    QUALITY_FILTER_AUDIO_PART,
    SEPARATE_CHUNK,
    TRANSCRIBE_CHUNK,
    PERSONA_CHUNK,
    EXTEND_CHUNK,
)


def _build_registry(
    contracts: tuple[TaskContract, ...],
) -> Mapping[str, TaskContract]:
    by_name = {contract.name: contract for contract in contracts}
    if len(by_name) != len(contracts):
        raise ValueError("Task names must be unique.")
    if len({contract.queue for contract in contracts}) != len(contracts):
        raise ValueError("Task queues must be unique.")
    return MappingProxyType(by_name)


TASKS_BY_NAME: Final = _build_registry(ALL_TASKS)


def get_task_contract(task_name: str) -> TaskContract | None:
    """Return a registered task contract without accepting arbitrary names."""
    return TASKS_BY_NAME.get(task_name)
