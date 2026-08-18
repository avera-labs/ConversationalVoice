from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class RequestConcurrencyMiddleware:
    """Queue ingest requests behind a process-local semaphore."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_concurrent_requests: int,
        ingest_path: str = "/v1/raw-audios",
    ) -> None:
        if max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")

        self._app = app
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._ingest_path = ingest_path.rstrip("/") or "/"

    def _is_ingest_request(self, scope: Scope) -> bool:
        if scope.get("type") != "http":
            return False
        if scope.get("method") != "POST":
            return False
        path = str(scope.get("path", "")).rstrip("/") or "/"
        return path == self._ingest_path

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not self._is_ingest_request(scope):
            await self._app(scope, receive, send)
            return

        async with self._semaphore:
            await self._app(scope, receive, send)
