from __future__ import annotations

import asyncio
from dataclasses import dataclass

from httpx import ASGITransport, AsyncClient

from voice_pipeline_ingest_api.app import create_app
from voice_pipeline_ingest_api.config import (
    ApplicationSettings,
    EnvironmentSettings,
    IngestPolicy,
    PolicySettings,
)


@dataclass
class FakeResources:
    ready_error: Exception | None = None
    closed: bool = False

    def check_readiness(self) -> None:
        if self.ready_error is not None:
            raise self.ready_error

    def close(self) -> None:
        self.closed = True


def _settings() -> ApplicationSettings:
    return ApplicationSettings(
        policy=PolicySettings(
            ingest=IngestPolicy(
                max_upload_bytes=314_572_800,
                max_concurrent_requests=10,
            )
        ),
        environment=EnvironmentSettings(
            database_url="postgresql+psycopg://user:password@db/voice",
            celery_broker_url="redis://redis:6379/0",
            s3_bucket="test-bucket",
            s3_region="us-east-1",
        ),
    )


def test_health_and_readiness_succeed() -> None:
    async def scenario() -> None:
        resources = FakeResources()
        app = create_app(
            _settings(),
            resource_factory=lambda settings: resources,  # type: ignore[arg-type]
        )
        transport = ASGITransport(app=app)

        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client,
        ):
            assert (await client.get("/health")).json() == {"status": "ok"}
            assert (await client.get("/ready")).json() == {"status": "ready"}

        assert resources.closed

    asyncio.run(scenario())


def test_root_serves_swagger_api_documentation() -> None:
    async def scenario() -> None:
        app = create_app(
            _settings(),
            resource_factory=lambda settings: FakeResources(),  # type: ignore[arg-type]
        )
        transport = ASGITransport(app=app)

        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client,
        ):
            response = await client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "Swagger UI" in response.text
        assert "/openapi.json" in response.text

    asyncio.run(scenario())


def test_readiness_failure_is_generic() -> None:
    async def scenario() -> None:
        resources = FakeResources(
            ready_error=RuntimeError("redis://secret-user:secret-password@private-host")
        )
        app = create_app(
            _settings(),
            resource_factory=lambda settings: resources,  # type: ignore[arg-type]
        )
        transport = ASGITransport(app=app)

        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client,
        ):
            response = await client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {"detail": "Service dependencies are not ready."}
        assert "secret-password" not in response.text

    asyncio.run(scenario())
