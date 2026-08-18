from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import ApplicationSettings, load_application_settings
from .dependencies import ServiceResources
from .middleware import RequestConcurrencyMiddleware
from .routes import health_router, raw_audios_router, tasks_router

ResourceFactory = Callable[[ApplicationSettings], ServiceResources]
logger = logging.getLogger(__name__)


def create_app(
    settings: ApplicationSettings | None = None,
    *,
    resource_factory: ResourceFactory = ServiceResources.create,
) -> FastAPI:
    """Create a fully configured ingest API application."""

    resolved_settings = settings or load_application_settings()
    logger.info(
        ("Loaded ingest policy: max_upload_bytes=%d, max_concurrent_requests=%d"),
        resolved_settings.policy.ingest.max_upload_bytes,
        resolved_settings.policy.ingest.max_concurrent_requests,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources = resource_factory(resolved_settings)
        app.state.settings = resolved_settings
        app.state.resources = resources
        try:
            yield
        finally:
            resources.close()

    app = FastAPI(
        title="Voice Pipeline Ingest API",
        version="0.1.0",
        docs_url="/",
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestConcurrencyMiddleware,
        max_concurrent_requests=(
            resolved_settings.policy.ingest.max_concurrent_requests
        ),
    )
    app.include_router(health_router)
    app.include_router(raw_audios_router)
    app.include_router(tasks_router)
    return app
