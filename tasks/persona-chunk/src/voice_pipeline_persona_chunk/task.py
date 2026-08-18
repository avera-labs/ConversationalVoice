from __future__ import annotations

import hashlib
import json
import logging
from uuid import UUID

from celery import Celery
from voice_pipeline_chunk_contracts import (
    parse_chunk_diarization,
    parse_persona_document,
    parse_persona_result,
    parse_separation_result,
    parse_transcription_artifact,
    parse_transcription_result,
)
from voice_pipeline_task_contracts import PERSONA_CHUNK

from .artifacts import (
    build_persona_document,
    canonical_json_bytes,
    write_canonical_json,
)
from .audio import encode_mp3, validate_wav
from .errors import safe_error
from .repository import Disposition
from .transcript import transcript_to_srt
from .workspace import Workspace

logger = logging.getLogger(__name__)


class Handler:
    def __init__(
        self, repository, storage, client, publisher, policy, workspace_parent=None
    ):
        self.repository = repository
        self.storage = storage
        self.client = client
        self.publisher = publisher
        self.policy = policy
        self.workspace_parent = workspace_parent

    def __call__(self, value):
        try:
            identifier = UUID(value)
        except (AttributeError, TypeError, ValueError):
            self._finished(None, "invalid_chunk_id")
            raise
        try:
            claim = self.repository.claim(identifier)
        except Exception:
            self._finished(identifier, "claim_failed")
            raise
        if claim.disposition is Disposition.ALREADY_COMPLETED:
            self._validate_completed_claim(claim)
        if claim.disposition is Disposition.READY_TO_DISPATCH:
            self._validate_completed_claim(claim)
            return self._publish(identifier)
        if claim.disposition is not Disposition.CLAIMED:
            self._finished(identifier, claim.disposition.value)
            return {"chunk_id": str(identifier), "outcome": claim.disposition.value}

        workspace = None
        try:
            workspace = Workspace(
                self.policy.task.workspace_prefix, self.workspace_parent
            )
            separation, _transcription, mapping, transcript_identity = (
                self._validate_upstream(claim)
            )
            self.storage.download(claim.audio_uri, workspace.audio)
            audio = validate_wav(workspace.audio, duration_ms=claim.duration_ms)
            expected_audio = separation.document["input_audio"]
            if (
                audio.size_bytes != expected_audio["size_bytes"]
                or audio.sha256 != expected_audio["sha256"]
            ):
                raise RuntimeError("input_audio_identity_mismatch")

            self.storage.download(transcript_identity[0], workspace.transcript)
            transcript_bytes = workspace.transcript.read_bytes()
            if (
                len(transcript_bytes) != transcript_identity[1]
                or hashlib.sha256(transcript_bytes).hexdigest()
                != transcript_identity[2]
            ):
                raise RuntimeError("input_transcript_identity_mismatch")
            transcript_document = parse_transcription_artifact(
                json.loads(transcript_bytes),
                kind="transcript",
                duration_ms=claim.duration_ms,
                speaker_mapping=mapping,
            )
            if canonical_json_bytes(transcript_document) != transcript_bytes:
                raise RuntimeError("input_transcript_is_not_canonical")
            srt = transcript_to_srt(transcript_document)
            encode_mp3(workspace.audio, workspace.mp3, self.policy.audio)
            wire, usage = self.client.analyze(workspace.mp3.read_bytes(), srt, mapping)
            persona = build_persona_document(wire, usage, mapping, self.policy)
            metadata = write_canonical_json(persona, workspace.persona)
            persona_uri = self.storage.persona_uri(claim.audio_uri)
            self.storage.upload_json(persona_uri, workspace.persona)
            result = self._result(
                claim,
                persona_uri,
                metadata.size_bytes,
                metadata.sha256,
                transcript_identity,
                expected_audio,
            )
            parse_persona_result(
                result,
                model_id=self.policy.openrouter.model,
                config_version=self.policy.config_version,
                input_audio=(claim.audio_uri, audio.size_bytes, audio.sha256),
                input_transcript=transcript_identity,
                artifact=(persona_uri, metadata.size_bytes, metadata.sha256),
            )
            self.repository.complete(claim, persona, result)
            return self._publish(identifier, speaker_count=2)
        except Exception:
            try:
                self.repository.fail(
                    identifier,
                    safe_error("persona_failed", self.policy.task.error_max_length),
                )
            except Exception:  # noqa: BLE001 - preserve the original failure
                logger.error("persona_chunk.failure_persistence_failed")
            self._finished(identifier, "failed")
            raise
        finally:
            if workspace is not None:
                try:
                    workspace.close()
                except Exception:  # noqa: BLE001 - cleanup must not replace outcome
                    logger.error("persona_chunk.workspace_cleanup_failed")

    def _validate_upstream(self, claim):
        if (
            claim.lang != "en"
            or not claim.audio_uri
            or not claim.duration_ms
            or claim.duration_ms <= 0
            or claim.end_ms - claim.start_ms != claim.duration_ms
            or not isinstance(claim.diarizations, dict)
            or not isinstance(claim.separation, dict)
            or not isinstance(claim.transcription, dict)
        ):
            raise RuntimeError("invalid_persona_input")
        snapshot = parse_chunk_diarization(
            claim.diarizations, duration_ms=claim.duration_ms
        )
        input_audio = claim.separation.get("input_audio")
        if not isinstance(input_audio, dict):
            raise TypeError("invalid_separation_input")
        separation = parse_separation_result(
            claim.separation,
            duration_ms=claim.duration_ms,
            speaker_ids=snapshot.speaker_ids,
            input_size_bytes=input_audio.get("size_bytes"),
            input_sha256=input_audio.get("sha256"),
            output_uris=self.storage.speaker_uris(claim.audio_uri),
        )
        mapping = tuple(
            item.diarization_speaker_id for item in separation.speaker_audio
        )
        artifacts = claim.transcription.get("artifacts")
        if not isinstance(artifacts, dict):
            raise TypeError("invalid_transcription_input")
        try:
            metadata = tuple(
                (artifacts[name]["size_bytes"], artifacts[name]["sha256"])
                for name in ("transcript", "word_alignment")
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError("invalid_transcription_input") from exc
        uris = self.storage.transcription_uris(claim.audio_uri)
        transcription = parse_transcription_result(
            claim.transcription,
            speaker_audio=separation.speaker_audio,
            artifact_uris=uris,
            artifact_metadata=metadata,
        )
        transcript = transcription["artifacts"]["transcript"]
        return (
            separation,
            transcription,
            mapping,
            (transcript["uri"], transcript["size_bytes"], transcript["sha256"]),
        )

    def _validate_completed_claim(self, claim):
        separation, _transcription, mapping, transcript_identity = (
            self._validate_upstream(claim)
        )
        if not isinstance(claim.persona, dict) or not isinstance(
            claim.persona_result, dict
        ):
            raise TypeError("invalid_completed_persona")
        model = claim.persona_result.get("model")
        if not isinstance(model, dict):
            raise TypeError("invalid_completed_persona")
        persisted_model_id = model.get("id")
        parse_persona_document(
            claim.persona,
            speaker_mapping=mapping,
            model_id=persisted_model_id,
            config_version=self.policy.config_version,
        )
        artifact = claim.persona_result.get("artifact")
        if not isinstance(artifact, dict):
            raise TypeError("invalid_completed_persona")
        input_audio = separation.document["input_audio"]
        persona_bytes = canonical_json_bytes(claim.persona)
        parse_persona_result(
            claim.persona_result,
            model_id=persisted_model_id,
            config_version=self.policy.config_version,
            input_audio=(
                claim.audio_uri,
                input_audio["size_bytes"],
                input_audio["sha256"],
            ),
            input_transcript=transcript_identity,
            artifact=(
                self.storage.persona_uri(claim.audio_uri),
                len(persona_bytes),
                hashlib.sha256(persona_bytes).hexdigest(),
            ),
        )

    def _result(self, claim, uri, size, sha, transcript, input_audio):
        return {
            "schema_version": 1,
            "backend": "openrouter",
            "model": {
                "id": self.policy.openrouter.model,
                "config_version": self.policy.config_version,
            },
            "language": "en",
            "input_audio": {
                "uri": claim.audio_uri,
                "size_bytes": input_audio["size_bytes"],
                "sha256": input_audio["sha256"],
            },
            "input_transcript": {
                "uri": transcript[0],
                "size_bytes": transcript[1],
                "sha256": transcript[2],
            },
            "artifact": {"uri": uri, "size_bytes": size, "sha256": sha},
        }

    def _publish(self, identifier, speaker_count=None):
        try:
            self.publisher.publish(identifier)
        except Exception:
            self.repository.fail_publication(
                identifier,
                safe_error(
                    "extension_publication_failed", self.policy.task.error_max_length
                ),
            )
            self._finished(identifier, "failed")
            raise
        self._finished(identifier, "persona_generated")
        result = {"chunk_id": str(identifier), "outcome": "persona_generated"}
        if speaker_count is not None:
            result["speaker_count"] = speaker_count
        return result

    @staticmethod
    def _finished(identifier, outcome):
        logger.info(
            "persona_chunk.finished",
            extra={
                "chunk_id": str(identifier) if identifier is not None else None,
                "outcome": outcome,
            },
        )


def register(app: Celery, handler):
    @app.task(
        name=PERSONA_CHUNK.name,
        queue=PERSONA_CHUNK.queue,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def persona_chunk(_task, value):
        return handler(value)

    return persona_chunk
