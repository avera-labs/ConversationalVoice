from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from voice_pipeline_ingest_api.middleware import RequestConcurrencyMiddleware


def test_ingest_requests_queue_at_the_process_limit() -> None:
    async def scenario() -> None:
        active = 0
        entered = 0
        maximum_active = 0
        first_ten_entered = asyncio.Event()
        release = asyncio.Event()

        app = FastAPI()

        @app.post("/v1/raw-audios")
        async def ingest() -> dict[str, bool]:
            nonlocal active, entered, maximum_active
            active += 1
            entered += 1
            maximum_active = max(maximum_active, active)
            if entered == 10:
                first_ten_entered.set()
            try:
                await release.wait()
                return {"ok": True}
            finally:
                active -= 1

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        limited_app = RequestConcurrencyMiddleware(
            app,
            max_concurrent_requests=10,
        )
        transport = ASGITransport(app=limited_app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            requests = [
                asyncio.create_task(client.post("/v1/raw-audios")) for _ in range(11)
            ]
            await asyncio.wait_for(first_ten_entered.wait(), timeout=1)
            await asyncio.sleep(0)

            assert entered == 10
            assert maximum_active == 10

            response = await client.get("/health")
            assert response.status_code == 200

            release.set()
            responses = await asyncio.gather(*requests)

        assert entered == 11
        assert all(response.status_code == 200 for response in responses)

    asyncio.run(scenario())


def test_trailing_slash_uses_the_same_ingest_limit() -> None:
    middleware = RequestConcurrencyMiddleware(
        app=lambda scope, receive, send: None,
        max_concurrent_requests=1,
    )

    assert middleware._is_ingest_request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/raw-audios/",
        }
    )


def test_cancelled_ingest_request_releases_its_slot() -> None:
    async def scenario() -> None:
        entered = 0
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        release = asyncio.Event()

        app = FastAPI()

        @app.post("/v1/raw-audios")
        async def ingest() -> dict[str, bool]:
            nonlocal entered
            entered += 1
            if entered == 1:
                first_entered.set()
            if entered == 2:
                second_entered.set()
            await release.wait()
            return {"ok": True}

        limited_app = RequestConcurrencyMiddleware(
            app,
            max_concurrent_requests=1,
        )
        transport = ASGITransport(app=limited_app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            first = asyncio.create_task(client.post("/v1/raw-audios"))
            await asyncio.wait_for(first_entered.wait(), timeout=1)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            second = asyncio.create_task(client.post("/v1/raw-audios"))
            await asyncio.wait_for(second_entered.wait(), timeout=1)
            release.set()
            response = await second

        assert response.status_code == 200

    asyncio.run(scenario())
