from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, status

from ..dependencies import ResourcesDependency
from ..schemas import ProbeResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["service"])


@router.get("/health", response_model=ProbeResponse)
async def health() -> ProbeResponse:
    """Return process liveness without checking external dependencies."""

    return ProbeResponse(status="ok")


@router.get("/ready", response_model=ProbeResponse)
async def readiness(resources: ResourcesDependency) -> ProbeResponse:
    """Return readiness only when all external dependencies are reachable."""

    try:
        await asyncio.to_thread(resources.check_readiness)
    # Each dependency client exposes a different exception hierarchy.
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Dependency readiness check failed: %s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service dependencies are not ready.",
        ) from None
    return ProbeResponse(status="ready")
