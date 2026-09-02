from __future__ import annotations

import logging
from uuid import UUID

from celery import Celery
from voice_pipeline_task_contracts import SCORE_CHUNK

from .repository import Disposition

logger = logging.getLogger(__name__)


class Handler:
    def __init__(self, repository, storage, service):
        self.repository = repository
        self.storage = storage
        self.service = service

    def __call__(self, value):
        try:
            identifier = UUID(value)
        except (AttributeError, TypeError, ValueError):
            self._finished(None, "invalid_chunk_id")
            raise
        claim = self.repository.claim(
            identifier, model_fingerprint=self.service.model_fingerprint
        )
        if claim.disposition is not Disposition.READY:
            self._finished(identifier, claim.disposition.value)
            return {
                "chunk_id": str(identifier),
                "outcome": claim.disposition.value,
                "evaluation": claim.evaluation,
            }
        assert claim.chunk is not None
        assert claim.source_fingerprint is not None
        if not claim.chunk.chunk_audio_uri:
            raise RuntimeError("missing_chunk_audio_uri")
        report, artifacts = self.service.score(claim.chunk)
        identities = self.storage.upload_artifacts(
            claim.chunk.chunk_audio_uri,
            model_fingerprint=self.service.model_fingerprint,
            source_fingerprint=claim.source_fingerprint,
            artifacts=artifacts,
        )
        evaluation = {
            "schema_version": 2,
            "status": report["status"],
            "language": claim.chunk.language,
            "model_fingerprint": self.service.model_fingerprint,
            "source_fingerprint": claim.source_fingerprint,
            "generated_at": report["generated_at"],
            "gpu_used": False,
            "local_asr_weights": False,
            "report": identities["score-report.json"],
            "artifacts": identities,
        }
        persisted = self.repository.complete(
            identifier,
            source_fingerprint_value=claim.source_fingerprint,
            evaluation=evaluation,
        )
        self._finished(identifier, "completed")
        return {
            "chunk_id": str(identifier),
            "outcome": "completed",
            "status": persisted["status"],
            "report": persisted["report"],
        }

    @staticmethod
    def _finished(identifier, outcome):
        logger.info(
            "score_chunk.finished",
            extra={
                "chunk_id": str(identifier) if identifier is not None else None,
                "outcome": outcome,
            },
        )


def register(app: Celery, handler):
    @app.task(
        name=SCORE_CHUNK.name,
        queue=SCORE_CHUNK.queue,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def score_chunk(_task, value):
        return handler(value)

    return score_chunk
