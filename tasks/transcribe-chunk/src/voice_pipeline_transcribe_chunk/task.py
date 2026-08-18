from __future__ import annotations

import logging
from uuid import UUID

from celery import Celery
from voice_pipeline_chunk_contracts import (
    parse_chunk_diarization,
    parse_separation_result,
    parse_transcription_result,
    validate_artifact_pair,
)
from voice_pipeline_task_contracts import TRANSCRIBE_CHUNK

from .artifacts import build_artifacts, model_identity, write_canonical_json
from .audio import read_speaker_wav
from .errors import safe_error
from .repository import Disposition
from .segments import plan_slices
from .utterances import build_utterances, normalize_words
from .workspace import Workspace

logger = logging.getLogger(__name__)


class Handler:
    def __init__(
        self, repository, storage, model, policy, workspace_parent=None, publisher=None
    ):
        self.repository = repository
        self.storage = storage
        self.model = model
        self.policy = policy
        self.workspace_parent = workspace_parent
        self.publisher = publisher

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
        if claim.disposition is Disposition.ALREADY_TRANSCRIBED:
            self._validate_completed_claim(claim)
        if claim.disposition is not Disposition.CLAIMED:
            self._finished(identifier, claim.disposition.value)
            return {"chunk_id": str(identifier), "outcome": claim.disposition.value}
        workspace = Workspace(self.policy.task.workspace_prefix, self.workspace_parent)
        try:
            if claim.lang != "en":
                raise RuntimeError("unsupported_chunk_language")
            if (
                not claim.audio_uri
                or not claim.duration_ms
                or claim.duration_ms <= 0
                or claim.end_ms - claim.start_ms != claim.duration_ms
            ):
                raise RuntimeError("invalid_chunk_input")
            snapshot = parse_chunk_diarization(
                claim.diarizations, duration_ms=claim.duration_ms
            )
            output_uris = self.storage.speaker_uris(claim.audio_uri)
            input_audio = claim.separation["input_audio"]
            separation = parse_separation_result(
                claim.separation,
                duration_ms=claim.duration_ms,
                speaker_ids=snapshot.speaker_ids,
                input_size_bytes=input_audio["size_bytes"],
                input_sha256=input_audio["sha256"],
                output_uris=output_uris,
            )
            audio_by_slot = []
            for speaker, destination in zip(
                separation.speaker_audio, workspace.speaker_paths, strict=True
            ):
                self.storage.download(speaker.uri, destination)
                audio = read_speaker_wav(destination, duration_ms=claim.duration_ms)
                if (
                    audio.size_bytes != speaker.size_bytes
                    or audio.sha256 != speaker.sha256
                ):
                    raise RuntimeError("speaker_audio_identity_mismatch")
                audio_by_slot.append(audio)
            speaker_outputs = []
            slice_count = 0
            for slot, (speaker, audio) in enumerate(
                zip(separation.speaker_audio, audio_by_slot, strict=True)
            ):
                slices = plan_slices(
                    snapshot,
                    speaker_id=speaker.diarization_speaker_id,
                    duration_ms=claim.duration_ms,
                    policy=self.policy.slices,
                )
                slice_count += len(slices)
                words = []
                for speech_slice in slices:
                    decoded = self.model.transcribe(
                        audio.samples[
                            speech_slice.start_ms * 16 : speech_slice.end_ms * 16
                        ]
                    )
                    words.extend(
                        normalize_words(
                            decoded,
                            offset_ms=speech_slice.start_ms,
                            duration_ms=claim.duration_ms,
                            policy=self.policy.utterance,
                        )
                    )
                utterances = build_utterances(words, self.policy.utterance)
                speaker_outputs.append(
                    (slot, speaker.diarization_speaker_id, words, utterances)
                )
            transcript, word_alignment = build_artifacts(
                policy=self.policy, speaker_outputs=speaker_outputs
            )
            mapping = tuple(
                item.diarization_speaker_id for item in separation.speaker_audio
            )
            validate_artifact_pair(
                transcript,
                word_alignment,
                duration_ms=claim.duration_ms,
                speaker_mapping=mapping,
            )
            transcript_meta = write_canonical_json(transcript, workspace.transcript)
            alignment_meta = write_canonical_json(
                word_alignment, workspace.word_alignment
            )
            artifact_uris = self.storage.artifact_uris(claim.audio_uri)
            self.storage.upload_json(artifact_uris[0], workspace.transcript)
            self.storage.upload_json(artifact_uris[1], workspace.word_alignment)
            artifact_metadata = (
                (transcript_meta.size_bytes, transcript_meta.sha256),
                (alignment_meta.size_bytes, alignment_meta.sha256),
            )
            result = {
                "schema_version": 1,
                "backend": "parakeet_tdt",
                "model": model_identity(self.policy),
                "language": "en",
                "input_speaker_audio": [
                    {
                        "output_slot": item.output_slot,
                        "diarization_speaker_id": item.diarization_speaker_id,
                        "uri": item.uri,
                        "size_bytes": item.size_bytes,
                        "sha256": item.sha256,
                    }
                    for item in separation.speaker_audio
                ],
                "artifacts": {
                    name: {
                        "uri": artifact_uris[index],
                        "size_bytes": artifact_metadata[index][0],
                        "sha256": artifact_metadata[index][1],
                    }
                    for index, name in enumerate(("transcript", "word_alignment"))
                },
            }
            parse_transcription_result(
                result,
                speaker_audio=separation.speaker_audio,
                artifact_uris=artifact_uris,
                artifact_metadata=artifact_metadata,
            )
            self.repository.complete(claim, result)
            if self.publisher is not None:
                self.publisher.publish(identifier)
            self._finished(identifier, "transcribed")
            return {
                "chunk_id": str(identifier),
                "outcome": "transcribed",
                "slice_count": slice_count,
                "word_count": sum(len(item[2]) for item in speaker_outputs),
                "utterance_count": sum(len(item[3]) for item in speaker_outputs),
            }
        except Exception:
            try:
                self.repository.fail(
                    identifier,
                    safe_error(
                        "transcription_failed", self.policy.task.error_max_length
                    ),
                )
            except Exception:  # noqa: BLE001 - preserve the original task failure
                logger.error("transcribe_chunk.failure_persistence_failed")
            self._finished(identifier, "failed")
            raise
        finally:
            try:
                workspace.close()
            except Exception:  # noqa: BLE001 - cleanup must not replace task outcome
                logger.error("transcribe_chunk.workspace_cleanup_failed")

    def _validate_completed_claim(self, claim):
        if (
            claim.lang != "en"
            or not claim.audio_uri
            or not claim.duration_ms
            or not isinstance(claim.diarizations, dict)
            or not isinstance(claim.separation, dict)
            or not isinstance(claim.transcription, dict)
        ):
            raise RuntimeError("invalid_completed_transcription")
        snapshot = parse_chunk_diarization(
            claim.diarizations, duration_ms=claim.duration_ms
        )
        output_uris = self.storage.speaker_uris(claim.audio_uri)
        input_audio = claim.separation.get("input_audio")
        if not isinstance(input_audio, dict):
            raise TypeError("invalid_completed_separation")
        separation = parse_separation_result(
            claim.separation,
            duration_ms=claim.duration_ms,
            speaker_ids=snapshot.speaker_ids,
            input_size_bytes=input_audio.get("size_bytes"),
            input_sha256=input_audio.get("sha256"),
            output_uris=output_uris,
        )
        artifact_uris = self.storage.artifact_uris(claim.audio_uri)
        artifacts = claim.transcription.get("artifacts")
        if not isinstance(artifacts, dict):
            raise TypeError("invalid_completed_transcription")
        try:
            metadata = tuple(
                (artifacts[name]["size_bytes"], artifacts[name]["sha256"])
                for name in ("transcript", "word_alignment")
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError("invalid_completed_transcription") from exc
        parse_transcription_result(
            claim.transcription,
            speaker_audio=separation.speaker_audio,
            artifact_uris=artifact_uris,
            artifact_metadata=metadata,
        )

    @staticmethod
    def _finished(identifier, outcome):
        logger.info(
            "transcribe_chunk.finished",
            extra={
                "chunk_id": str(identifier) if identifier is not None else None,
                "outcome": outcome,
            },
        )


def register(app: Celery, handler):
    @app.task(
        name=TRANSCRIBE_CHUNK.name,
        queue=TRANSCRIBE_CHUNK.queue,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def transcribe_chunk(_task, value):
        return handler(value)

    return transcribe_chunk
