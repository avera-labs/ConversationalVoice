from __future__ import annotations

import hashlib
import logging
from uuid import UUID

import numpy as np
from celery import Celery
from voice_pipeline_chunk_contracts import (
    build_chunk_diarization,
    parse_chunk_diarization,
    parse_separation_result,
)
from voice_pipeline_diarization_artifact import parse_artifact_bytes
from voice_pipeline_task_contracts import SEPARATE_CHUNK

from .audio import read_chunk, write_outputs
from .audit import audit_tracks
from .errors import QualityRejection, safe_error
from .stitching import merge
from .windows import plan_windows
from .workspace import Workspace

logger = logging.getLogger(__name__)


class Handler:
    def __init__(
        self,
        repository,
        storage,
        model,
        aligner,
        policy,
        workspace_parent=None,
        publisher=None,
    ):
        self.repository = repository
        self.storage = storage
        self.model = model
        self.aligner = aligner
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
            try:
                self.repository.fail(
                    identifier,
                    safe_error("separation_failed", self.policy.task.error_max_length),
                )
            except Exception:  # noqa: BLE001 - preserving the original claim failure
                logger.error("separate_chunk.failure_persistence_failed")
            self._finished(identifier, "failed")
            raise
        if claim.disposition.value == "already_separated":
            self._validate_completed_claim(claim)
        if claim.disposition.value != "claimed":
            self._finished(identifier, claim.disposition.value)
            return {"chunk_id": str(identifier), "outcome": claim.disposition.value}
        workspace = Workspace(self.policy.task.workspace_prefix, self.workspace_parent)
        try:
            self.storage.download(claim.audio_uri, workspace.audio)
            self.storage.download(
                claim.diarization_uri,
                workspace.diarization,
                maximum_bytes=self.policy.task.max_diarization_bytes,
            )
            audio = read_chunk(workspace.audio, claim.duration_ms)
            parent = parse_artifact_bytes(
                workspace.diarization.read_bytes(),
                expected_duration_ms=claim.part_duration_ms,
            )
            snapshot = build_chunk_diarization(
                parent.turns, start_ms=claim.start_ms, end_ms=claim.end_ms
            )
            windows = plan_windows(snapshot, claim.duration_ms, self.policy.window)
            seed_material = hashlib.shake_256(identifier.bytes).digest(8 * len(windows))
            stitched = None
            native_rate = None
            previous = None
            for index, window in enumerate(windows):
                samples = audio.samples[window.start_ms * 16 : window.end_ms * 16]
                seed = int.from_bytes(seed_material[index * 8 : (index + 1) * 8], "big")
                current, native_rate = self.model.separate(samples, seed=seed)
                target = round((window.end_ms - window.start_ms) * native_rate / 1000)
                current = (
                    current[:, :target]
                    if current.shape[1] >= target
                    else np.pad(current, ((0, 0), (0, target - current.shape[1])))
                )
                if stitched is None:
                    stitched = current
                else:
                    if self.aligner.align(previous, current, native_rate):
                        current = current[[1, 0]]
                    stitched = merge(
                        stitched,
                        current,
                        round(self.policy.window.overlap_ms * native_rate / 1000),
                        round(self.policy.window.crossfade_ms * native_rate / 1000),
                    )
                previous = current
            target = round(claim.duration_ms * native_rate / 1000)
            stitched = (
                stitched[:, :target]
                if stitched.shape[1] >= target
                else np.pad(stitched, ((0, 0), (0, target - stitched.shape[1])))
            )
            if not np.isfinite(stitched).all() or stitched.shape[0] != 2:
                raise RuntimeError("invalid_model_output")
            peak = np.max(np.abs(stitched), axis=1)
            if np.any(peak <= self.policy.output.silence_peak_max):
                from .errors import RejectionCode

                raise QualityRejection(RejectionCode.OUTPUT_QUALITY_REJECTED)
            audit = audit_tracks(snapshot, stitched, native_rate, self.policy.audit)
            metadata = write_outputs(
                stitched,
                native_rate,
                workspace.outputs,
                claim.duration_ms,
                self.policy.output.peak,
            )
            uris = self.storage.output_uris(claim.audio_uri)
            for uri, path in zip(uris, workspace.outputs, strict=True):
                self.storage.upload(uri, path)
            result = {
                "schema_version": 1,
                "backend": "dialogue_sidon",
                "model": {
                    "repo_id": self.policy.model.repo_id,
                    "revision": self.policy.model.revision,
                    "config_version": self.policy.config_version,
                    "inference_steps": self.policy.model.inference_steps,
                },
                "input_audio": {
                    "sample_rate_hz": 16000,
                    "duration_ms": claim.duration_ms,
                    "size_bytes": audio.size_bytes,
                    "sha256": audio.sha256,
                },
                "speaker_audio": [
                    {
                        "output_slot": slot,
                        "diarization_speaker_id": audit.mapping[slot],
                        "uri": uris[slot],
                        "sample_rate_hz": 16000,
                        "duration_ms": claim.duration_ms,
                        **metadata[slot],
                    }
                    for slot in range(2)
                ],
                "audit": {
                    "verdict": "ok",
                    "reference_speaker_id": audit.reference_speaker_id,
                    "consistent_relation": audit.relation,
                },
            }
            parse_separation_result(
                result,
                duration_ms=claim.duration_ms,
                speaker_ids=snapshot.speaker_ids,
                input_size_bytes=audio.size_bytes,
                input_sha256=audio.sha256,
                output_uris=uris,
            )
            self.repository.complete(claim, parent.model, snapshot.to_dict(), result)
            if claim.lang == "en" and self.publisher is not None:
                self.publisher.publish(identifier)
            self._finished(identifier, "separated")
            return {
                "chunk_id": str(identifier),
                "outcome": "separated",
                "output_slot_count": 2,
                "window_count": len(windows),
                "duration_ms": claim.duration_ms,
            }
        except QualityRejection as rejection:
            try:
                self.repository.reject(
                    identifier,
                    safe_error(rejection.code.value, self.policy.task.error_max_length),
                )
            except Exception:
                try:
                    self.repository.fail(
                        identifier,
                        safe_error(
                            "rejection_persistence_failed",
                            self.policy.task.error_max_length,
                        ),
                    )
                except Exception:  # noqa: BLE001 - preserving the original rejection
                    logger.error("separate_chunk.failure_persistence_failed")
                self._finished(identifier, "failed")
                raise
            self._finished(identifier, "rejected")
            return {
                "chunk_id": str(identifier),
                "outcome": "rejected",
                "rejection_code": rejection.code.value,
            }
        except Exception:
            try:
                self.repository.fail(
                    identifier,
                    safe_error("separation_failed", self.policy.task.error_max_length),
                )
            except Exception:  # noqa: BLE001 - preserving the original task failure
                logger.error("separate_chunk.failure_persistence_failed")
            self._finished(identifier, "failed")
            raise
        finally:
            try:
                workspace.close()
            except Exception:  # noqa: BLE001 - cleanup must not replace task outcome
                logger.error("separate_chunk.workspace_cleanup_failed")

    def _validate_completed_claim(self, claim):
        if (
            not claim.audio_uri
            or not claim.duration_ms
            or not isinstance(claim.diarizations, dict)
            or not isinstance(claim.separation, dict)
        ):
            raise RuntimeError("invalid_completed_separation")
        snapshot = parse_chunk_diarization(
            claim.diarizations, duration_ms=claim.duration_ms
        )
        input_audio = claim.separation.get("input_audio")
        if not isinstance(input_audio, dict):
            raise TypeError("invalid_completed_separation")
        parse_separation_result(
            claim.separation,
            duration_ms=claim.duration_ms,
            speaker_ids=snapshot.speaker_ids,
            input_size_bytes=input_audio.get("size_bytes"),
            input_sha256=input_audio.get("sha256"),
            output_uris=self.storage.output_uris(claim.audio_uri),
        )

    @staticmethod
    def _finished(identifier, outcome):
        logger.info(
            "separate_chunk.finished",
            extra={
                "chunk_id": str(identifier) if identifier is not None else None,
                "outcome": outcome,
            },
        )


def register(app: Celery, handler):
    @app.task(
        name=SEPARATE_CHUNK.name,
        queue=SEPARATE_CHUNK.queue,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def separate_chunk(_task, value):
        return handler(value)

    return separate_chunk
