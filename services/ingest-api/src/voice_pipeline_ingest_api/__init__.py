"""HTTP ingest service for normalized podcast audio."""

from .app import create_app

__all__ = ["create_app"]
