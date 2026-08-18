from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from voice_pipeline_task_client import (
    TaskPublicationError,
    UnknownTaskNameError,
)

from voice_pipeline_ingest_api.dependencies import get_task_publisher
from voice_pipeline_ingest_api.routes.tasks import router

ENTITY_ID = UUID("12345678-1234-5678-1234-567812345678")


class FakePublisher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, UUID]] = []
        self.error: Exception | None = None

    def publish_registered(self, task_name: str, entity_id: UUID) -> str:
        self.calls.append((task_name, entity_id))
        if self.error is not None:
            raise self.error
        return "task-123"


def _app(publisher: FakePublisher) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_task_publisher] = lambda: publisher
    return app


def test_trigger_task_returns_202_with_broker_task_id() -> None:
    async def scenario() -> None:
        publisher = FakePublisher()
        app = _app(publisher)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/tasks/trigger",
                json={
                    "task_name": "split_raw_audio_into_parts",
                    "id": str(ENTITY_ID),
                },
            )

        assert response.status_code == 202
        assert response.json() == {
            "task_name": "split_raw_audio_into_parts",
            "id": str(ENTITY_ID),
            "task_id": "task-123",
        }
        assert publisher.calls == [("split_raw_audio_into_parts", ENTITY_ID)]

    asyncio.run(scenario())


def test_trigger_task_rejects_unknown_task_name() -> None:
    async def scenario() -> None:
        publisher = FakePublisher()
        publisher.error = UnknownTaskNameError("Task name is not registered.")
        app = _app(publisher)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/tasks/trigger",
                json={"task_name": "unknown", "id": str(ENTITY_ID)},
            )

        assert response.status_code == 422
        assert response.json() == {"detail": "Task name is not registered."}

    asyncio.run(scenario())


def test_trigger_task_validates_uuid_and_forbids_extra_fields() -> None:
    async def scenario() -> None:
        publisher = FakePublisher()
        app = _app(publisher)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            invalid_uuid = await client.post(
                "/v1/tasks/trigger",
                json={"task_name": "diarize_audio_part", "id": "not-a-uuid"},
            )
            extra_field = await client.post(
                "/v1/tasks/trigger",
                json={
                    "task_name": "diarize_audio_part",
                    "id": str(ENTITY_ID),
                    "audio_uri": "unsupported value",
                },
            )

        assert invalid_uuid.status_code == 422
        assert extra_field.status_code == 422
        assert publisher.calls == []

    asyncio.run(scenario())


def test_trigger_task_returns_safe_503_for_broker_failure() -> None:
    async def scenario() -> None:
        publisher = FakePublisher()
        publisher.error = TaskPublicationError("provider diagnostic marker")
        app = _app(publisher)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/tasks/trigger",
                json={
                    "task_name": "diarize_audio_part",
                    "id": str(ENTITY_ID),
                },
            )

        assert response.status_code == 503
        assert response.json() == {"detail": "Task publication is unavailable."}
        assert "diagnostic marker" not in response.text

    asyncio.run(scenario())
