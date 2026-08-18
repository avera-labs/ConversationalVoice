from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from voice_pipeline_task_client import (
    DEFAULT_PUBLISH_POLICY,
    PublishPolicy,
    TaskPublicationError,
    TaskPublisher,
    UnknownTaskNameError,
)
from voice_pipeline_task_contracts import (
    DIARIZE_AUDIO_PART,
    TaskContract,
)

IDENTIFIER = UUID("12345678-1234-5678-1234-567812345678")


class FakeConnection:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.calls: list[int] = []

    def ensure_connection(self, *, max_retries: int) -> None:
        self.calls.append(max_retries)
        if self.error is not None:
            raise self.error


class FakeCelery:
    def __init__(self, *, task_id: str | None = "task-123") -> None:
        self.task_id = task_id
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error: Exception | None = None
        self.connection = FakeConnection()
        self.closed = False

    def send_task(self, name: str, **options: Any) -> SimpleNamespace:
        self.calls.append((name, options))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(id=self.task_id)

    def connection_for_write(self):
        return nullcontext(self.connection)

    def close(self) -> None:
        self.closed = True


def test_publish_serializes_one_uuid_and_routes_from_contract() -> None:
    app = FakeCelery()
    publisher = TaskPublisher(app)

    task_id = publisher.publish(DIARIZE_AUDIO_PART, IDENTIFIER)

    assert task_id == "task-123"
    assert app.calls == [
        (
            DIARIZE_AUDIO_PART.name,
            {
                "args": [str(IDENTIFIER)],
                "queue": DIARIZE_AUDIO_PART.queue,
                "retry": True,
                "retry_policy": DEFAULT_PUBLISH_POLICY.retry_policy(),
                "confirm_timeout": (
                    DEFAULT_PUBLISH_POLICY.confirm_timeout_seconds
                ),
            },
        )
    ]


def test_publish_rejects_unregistered_contract_before_broker_call() -> None:
    app = FakeCelery()
    publisher = TaskPublisher(app)
    unregistered = TaskContract(
        name=DIARIZE_AUDIO_PART.name,
        queue="unregistered_queue",
        uuid_argument="audio_part_id",
    )

    with pytest.raises(UnknownTaskNameError, match="not registered"):
        publisher.publish(unregistered, IDENTIFIER)

    assert app.calls == []


def test_publish_registered_rejects_unknown_name_before_broker_call() -> None:
    app = FakeCelery()
    publisher = TaskPublisher(app)

    with pytest.raises(UnknownTaskNameError, match="not registered"):
        publisher.publish_registered("unknown_task", IDENTIFIER)

    assert app.calls == []


def test_publish_errors_do_not_expose_provider_diagnostics() -> None:
    app = FakeCelery()
    app.error = RuntimeError("provider diagnostic marker")
    publisher = TaskPublisher(app)

    with pytest.raises(TaskPublicationError) as caught:
        publisher.publish(DIARIZE_AUDIO_PART, IDENTIFIER)

    assert "diagnostic marker" not in str(caught.value)


@pytest.mark.parametrize("task_id", [None, ""])
def test_publish_requires_a_non_empty_string_task_id(task_id: str | None) -> None:
    publisher = TaskPublisher(FakeCelery(task_id=task_id))

    with pytest.raises(TaskPublicationError, match="did not return"):
        publisher.publish(DIARIZE_AUDIO_PART, IDENTIFIER)


def test_publish_requires_uuid_object() -> None:
    publisher = TaskPublisher(FakeCelery())

    with pytest.raises(TypeError, match="must be a UUID"):
        publisher.publish(DIARIZE_AUDIO_PART, str(IDENTIFIER))  # type: ignore[arg-type]


def test_readiness_wraps_provider_error_and_close_releases_client() -> None:
    app = FakeCelery()
    app.connection.error = RuntimeError("provider diagnostic marker")
    publisher = TaskPublisher(app)

    with pytest.raises(TaskPublicationError) as caught:
        publisher.check_readiness()
    assert "diagnostic marker" not in str(caught.value)

    publisher.close()
    assert app.closed is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_retries", -1),
        ("interval_start_seconds", -0.1),
        ("interval_step_seconds", -0.1),
        ("interval_max_seconds", -0.1),
        ("confirm_timeout_seconds", -0.1),
    ],
)
def test_publish_policy_rejects_negative_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        PublishPolicy(**{field: value})  # type: ignore[arg-type]
