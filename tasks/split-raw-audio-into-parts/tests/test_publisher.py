from __future__ import annotations

from uuid import UUID

from voice_pipeline_task_contracts import DIARIZE_AUDIO_PART, TaskContract

from voice_pipeline_split_raw_audio_into_parts.publisher import (
    DiarizationPublisher,
)

AUDIO_PART_ID = UUID("12345678-1234-5678-1234-567812345678")


class FakeTaskPublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[TaskContract, UUID]] = []
        self.readiness_checks = 0
        self.closed = False

    def publish(self, contract: TaskContract, identifier: UUID) -> str:
        self.calls.append((contract, identifier))
        return "task-id"

    def check_readiness(self) -> None:
        self.readiness_checks += 1

    def close(self) -> None:
        self.closed = True


def test_publish_selects_diarization_contract_and_forwards_uuid() -> None:
    client = FakeTaskPublisher()
    publisher = DiarizationPublisher(client)  # type: ignore[arg-type]

    task_id = publisher.publish(AUDIO_PART_ID)

    assert task_id == "task-id"
    assert client.calls == [(DIARIZE_AUDIO_PART, AUDIO_PART_ID)]


def test_readiness_and_close_delegate_to_shared_client() -> None:
    client = FakeTaskPublisher()
    publisher = DiarizationPublisher(client)  # type: ignore[arg-type]

    publisher.check_readiness()
    publisher.close()

    assert client.readiness_checks == 1
    assert client.closed is True
