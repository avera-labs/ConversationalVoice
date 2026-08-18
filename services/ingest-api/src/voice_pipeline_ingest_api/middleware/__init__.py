"""ASGI middleware used by the ingest service."""

from .request_concurrency import RequestConcurrencyMiddleware

__all__ = ["RequestConcurrencyMiddleware"]
