from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from celery import Celery
from voice_pipeline_task_contracts import TaskContract, get_task_contract


class TaskResultPort(Protocol):
    id: str | None


class BrokerConnectionPort(Protocol):
    def ensure_connection(self, *, max_retries: int) -> None: ...


class CeleryProducerPort(Protocol):
    def send_task(self, name: str, **options: Any) -> TaskResultPort: ...

    def connection_for_write(
        self,
    ) -> AbstractContextManager[BrokerConnectionPort]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PublishPolicy:
    """Bounded retry and confirmation settings for task publication."""

    max_retries: int = 3
    interval_start_seconds: float = 0
    interval_step_seconds: float = 0.2
    interval_max_seconds: float = 1
    confirm_timeout_seconds: float = 5

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative.")
        for name, value in (
            ("interval_start_seconds", self.interval_start_seconds),
            ("interval_step_seconds", self.interval_step_seconds),
            ("interval_max_seconds", self.interval_max_seconds),
            ("confirm_timeout_seconds", self.confirm_timeout_seconds),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative.")

    def retry_policy(self) -> dict[str, int | float]:
        return {
            "max_retries": self.max_retries,
            "interval_start": self.interval_start_seconds,
            "interval_step": self.interval_step_seconds,
            "interval_max": self.interval_max_seconds,
        }


DEFAULT_PUBLISH_POLICY = PublishPolicy()


class UnknownTaskNameError(ValueError):
    """Raised when a requested task is outside the shared registry."""


class TaskPublicationError(RuntimeError):
    """Raised when a registered task cannot be published safely."""


class TaskPublisher:
    """Publish registered UUID task contracts through one Celery client."""

    def __init__(
        self,
        celery_app: CeleryProducerPort,
        *,
        policy: PublishPolicy = DEFAULT_PUBLISH_POLICY,
    ) -> None:
        self._celery_app = celery_app
        self._policy = policy

    @classmethod
    def create(
        cls,
        *,
        client_name: str,
        broker_url: str,
        policy: PublishPolicy = DEFAULT_PUBLISH_POLICY,
    ) -> TaskPublisher:
        celery_app = Celery(client_name, broker=broker_url)
        celery_app.conf.update(
            accept_content=["json"],
            result_serializer="json",
            task_publish_retry=True,
            task_publish_retry_policy=policy.retry_policy(),
            task_serializer="json",
        )
        return cls(celery_app, policy=policy)

    def check_readiness(self) -> None:
        try:
            with self._celery_app.connection_for_write() as connection:
                connection.ensure_connection(max_retries=0)
        except Exception as exc:
            raise TaskPublicationError("Task broker is not ready.") from exc

    def publish(self, contract: TaskContract, identifier: UUID) -> str:
        if get_task_contract(contract.name) != contract:
            raise UnknownTaskNameError("Task contract is not registered.")
        if not isinstance(identifier, UUID):
            raise TypeError("identifier must be a UUID.")

        try:
            result = self._celery_app.send_task(
                contract.name,
                args=[str(identifier)],
                queue=contract.queue,
                retry=True,
                retry_policy=self._policy.retry_policy(),
                confirm_timeout=self._policy.confirm_timeout_seconds,
            )
        except Exception as exc:
            raise TaskPublicationError(
                "Unable to publish the requested task."
            ) from exc

        if not isinstance(result.id, str) or not result.id:
            raise TaskPublicationError(
                "The broker did not return a task identifier."
            )
        return result.id

    def publish_registered(self, task_name: str, identifier: UUID) -> str:
        contract = get_task_contract(task_name)
        if contract is None:
            raise UnknownTaskNameError("Task name is not registered.")
        return self.publish(contract, identifier)

    def close(self) -> None:
        self._celery_app.close()
