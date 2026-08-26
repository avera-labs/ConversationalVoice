from __future__ import annotations

import logging
from uuid import UUID

from celery import Celery
from voice_pipeline_task_contracts import RECONSTRUCT_CHUNK

from .artifacts import write_canonical_json
from .audio import render_tracks
from .errors import safe_error
from .inputs import InputLoader
from .outputs import OutputArtifacts
from .reconstruction import Reconstructor
from .reference import SpeakerReferenceUnavailable
from .repository import Disposition
from .workspace import Workspace

logger = logging.getLogger(__name__)


class Handler:
    def __init__(
        self,
        repository,
        storage,
        tags_client,
        tts_client,
        publisher,
        policy,
        workspace_parent=None,
        *,
        forced_aligner,
    ):
        self.repository = repository
        self.storage = storage
        self.publisher = publisher
        self.policy = policy
        self.workspace_parent = workspace_parent
        self.inputs = InputLoader(storage, policy)
        self.reconstructor = Reconstructor(
            tags_client, tts_client, forced_aligner, policy
        )
        self.outputs = OutputArtifacts(storage, policy)

    def __call__(self, value):
        identifier = self._parse_identifier(value)
        try:
            claim = self.repository.claim(identifier)
        except Exception:
            self._finished(identifier, "claim_failed")
            raise
        if claim.disposition is Disposition.READY_TO_DISPATCH:
            self._validate_completed(claim)
            if claim.status in {"reconstructed", "failed"}:
                return self._publish(identifier)
            outcome = "already_completed"
            self._finished(identifier, outcome)
            return {"chunk_id": str(identifier), "outcome": outcome}
        if claim.disposition is not Disposition.CLAIMED:
            self._finished(identifier, claim.disposition.value)
            return {"chunk_id": str(identifier), "outcome": claim.disposition.value}
        return self._process(identifier, claim)

    def _parse_identifier(self, value) -> UUID:
        try:
            return UUID(value)
        except (AttributeError, TypeError, ValueError):
            self._finished(None, "invalid_chunk_id")
            raise

    def _process(self, identifier, claim):
        workspace = None
        try:
            workspace = Workspace(
                self.policy.task.workspace_prefix, self.workspace_parent
            )
            loaded = self.inputs.load(claim, workspace)
            mapping = loaded.upstream.speaker_mapping
            reconstruction = self.reconstructor.reconstruct(
                loaded.transcript,
                loaded.separated_audio,
                loaded.references,
                speaker_mapping=mapping,
                source_duration_ms=claim.duration_ms,
                language=claim.lang,
            )
            transcript_meta = write_canonical_json(
                reconstruction.transcript, workspace.output_transcript
            )
            tracks = render_tracks(
                reconstruction.transcript["utterances"],
                reconstruction.generated_audio,
                speaker_mapping=mapping,
                track_paths=(workspace.track(0), workspace.track(1)),
                sample_rate_hz=self.policy.audio.output_sample_rate_hz,
            )
            output_uris = self.storage.output_uris(claim.chunk_audio_uri)
            manifest = self.outputs.build_manifest(
                claim,
                loaded,
                reconstruction,
                output_uris,
                transcript_meta,
                tracks,
            )
            manifest_meta = write_canonical_json(manifest, workspace.manifest)
            self._upload(output_uris, workspace, tracks)
            result = self.outputs.build_result(
                claim,
                reconstruction,
                output_uris,
                manifest_meta,
                transcript_meta,
                tracks,
            )
            self.outputs.validate_result(claim, result)
            self.repository.complete(claim, result)
            return self._publish(
                identifier,
                utterance_count=len(reconstruction.transcript["utterances"]),
                duration_ms=reconstruction.transcript["duration_ms"],
            )
        except SpeakerReferenceUnavailable:
            self.repository.reject(
                identifier,
                safe_error(
                    "speaker_reference_unavailable", self.policy.task.error_max_length
                ),
            )
            self._finished(identifier, "rejected")
            return {"chunk_id": str(identifier), "outcome": "rejected"}
        except Exception:
            self._persist_failure(identifier)
            self._finished(identifier, "failed")
            raise
        finally:
            self._close_workspace(workspace)

    def _upload(self, output_uris, workspace, tracks):
        for track, uri in zip(tracks, output_uris["speaker_audio"], strict=True):
            self.storage.upload_wav(uri, track.path)
        self.storage.upload_json(output_uris["transcript"], workspace.output_transcript)
        self.storage.upload_json(output_uris["manifest"], workspace.manifest)

    def _validate_completed(self, claim):
        self.inputs.validate(claim)
        if not isinstance(claim.reconstruction, dict):
            raise TypeError("invalid_completed_reconstruction")
        self.outputs.validate_result(claim, claim.reconstruction, current_policy=False)

    def _persist_failure(self, identifier):
        try:
            self.repository.fail(
                identifier,
                safe_error("reconstruction_failed", self.policy.task.error_max_length),
            )
        except Exception:
            logger.error("reconstruct_chunk.failure_persistence_failed")

    @staticmethod
    def _close_workspace(workspace):
        if workspace is None:
            return
        try:
            workspace.close()
        except Exception:
            logger.error("reconstruct_chunk.workspace_cleanup_failed")

    def _publish(self, identifier, **result):
        try:
            self.publisher.publish(identifier)
        except Exception:
            self.repository.fail_publication(
                identifier,
                safe_error(
                    "extend_chunk_publication_failed",
                    self.policy.task.error_max_length,
                ),
            )
            self._finished(identifier, "publication_failed")
            raise
        self._finished(identifier, "reconstructed")
        return {"chunk_id": str(identifier), "outcome": "reconstructed", **result}

    @staticmethod
    def _finished(identifier, outcome):
        logger.info(
            "reconstruct_chunk.finished",
            extra={
                "chunk_id": str(identifier) if identifier is not None else None,
                "outcome": outcome,
            },
        )


def register(app: Celery, handler):
    @app.task(
        name=RECONSTRUCT_CHUNK.name,
        queue=RECONSTRUCT_CHUNK.queue,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def reconstruct_chunk(_task, value):
        return handler(value)

    return reconstruct_chunk
