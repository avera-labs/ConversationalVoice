"""Celery task lifecycle orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Protocol, TypeVar
from uuid import UUID

from celery import Celery
from celery.app.task import Task
from voice_pipeline_task_contracts import DIARIZE_AUDIO_PART

from .artifact import DiarizationArtifact, build_artifact
from .config import DiarizationPolicy, SpeakerReferencePolicy, TaskPolicy
from .diarization import InferenceResult
from .errors import ErrorCode, FailureStage, TaskStageError, safe_message
from .repository import (
    AudioPartClaim,
    AudioPartNotFoundError,
    ClaimDisposition,
    InvalidAudioPartStatusError,
)
from .speaker_reference import (
    ManifestSpeaker,
    ReferenceAudio,
    SpeakerReferenceManifest,
    plan_speaker_references,
)
from .storage import InvalidS3UriError
from .wav_io import read_normalized_wav, write_reference_wav
from .workspace import TaskWorkspace

logger = logging.getLogger(__name__)
Value = TypeVar("Value")
Clock = Callable[[], int]


class RepositoryPort(Protocol):
    def claim(self, audio_part_id: UUID) -> AudioPartClaim: ...
    def complete(self, audio_part_id: UUID, diarization_uri: str) -> None: ...
    def mark_processing_failed(self, audio_part_id: UUID, error: str) -> None: ...
    def mark_dispatch_failed(self, audio_part_id: UUID, error: str) -> None: ...


class StoragePort(Protocol):
    def download_audio(self, audio_uri: str, destination: Path) -> int: ...
    def upload_artifact(self, audio_uri: str, path: Path) -> str: ...
    def reference_audio_uri(self, audio_uri: str, speaker_id: int) -> str: ...
    def upload_reference_audio(
        self, audio_uri: str, speaker_id: int, path: Path
    ) -> str: ...
    def upload_reference_manifest(self, audio_uri: str, path: Path) -> str: ...


class DiarizationPort(Protocol):
    def infer(self, audio_path: Path) -> InferenceResult: ...


class PublisherPort(Protocol):
    def publish(self, audio_part_id: UUID) -> str: ...


WorkspaceFactory = Callable[..., TaskWorkspace]


@dataclass(frozen=True, slots=True)
class PreparedSpeakerReferences:
    audio_files: tuple[tuple[int, Path, str], ...]
    manifest_path: Path

    @property
    def speaker_count(self) -> int:
        return len(self.audio_files)


_TIMING_FIELDS = (
    "elapsed_claim_ms",
    "elapsed_download_ms",
    "elapsed_model_inference_ms",
    "elapsed_normalize_serialize_ms",
    "elapsed_reference_build_ms",
    "elapsed_upload_ms",
    "elapsed_persistence_ms",
    "elapsed_dispatch_ms",
    "elapsed_cleanup_ms",
)


class DiarizeAudioPartHandler:
    """Compose one status-owned diarization attempt."""

    def __init__(
        self,
        *,
        repository: RepositoryPort,
        storage: StoragePort,
        diarization: DiarizationPort,
        publisher: PublisherPort,
        diarization_policy: DiarizationPolicy,
        speaker_reference_policy: SpeakerReferencePolicy,
        task_policy: TaskPolicy,
        workspace_parent: Path | None = None,
        workspace_factory: WorkspaceFactory = TaskWorkspace,
        clock: Clock = perf_counter_ns,
        terminal_logger: logging.Logger = logger,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._diarization = diarization
        self._publisher = publisher
        self._diarization_policy = diarization_policy
        self._speaker_reference_policy = speaker_reference_policy
        self._task_policy = task_policy
        self._workspace_parent = workspace_parent
        self._workspace_factory = workspace_factory
        self._clock = clock
        self._logger = terminal_logger

    def __call__(self, audio_part_id: str) -> dict[str, object]:
        started = self._clock()
        summary = self._new_summary()
        try:
            try:
                parsed_id = UUID(audio_part_id)
            except (AttributeError, TypeError, ValueError) as exc:
                raise TaskStageError(
                    FailureStage.INPUT, ErrorCode.INVALID_AUDIO_PART_ID
                ) from exc
            summary["audio_part_id"] = str(parsed_id)
            try:
                claim = self._timed(
                    summary, "elapsed_claim_ms", self._repository.claim, parsed_id
                )
            except AudioPartNotFoundError as exc:
                raise TaskStageError(
                    FailureStage.INPUT, ErrorCode.AUDIO_PART_NOT_FOUND
                ) from exc
            except InvalidAudioPartStatusError as exc:
                raise TaskStageError(
                    FailureStage.INPUT, ErrorCode.INVALID_AUDIO_PART_STATE
                ) from exc
            except TaskStageError:
                raise
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.CLAIM, ErrorCode.PERSISTENCE_FAILED
                ) from exc

            if claim.disposition is ClaimDisposition.ALREADY_PROCESSING:
                summary.update(outcome="no_op", final_status=claim.status)
                return self._result(parsed_id, "already_processing", dispatched=False)
            if claim.disposition is ClaimDisposition.COMPLETED:
                summary.update(outcome="no_op", final_status="completed")
                return self._result(parsed_id, "already_completed", dispatched=False)
            if claim.disposition is ClaimDisposition.DISPATCH_READY:
                self._dispatch(parsed_id, summary)
                summary.update(outcome="succeeded", final_status="diarized")
                return self._result(parsed_id, "diarized", dispatched=True)
            return self._process_claimed(claim, summary)
        except TaskStageError as failure:
            summary.update(
                outcome="failed",
                failure_stage=failure.stage.value,
                error_code=failure.code.value,
            )
            raise
        finally:
            summary["elapsed_total_ms"] = self._elapsed_ms(started)
            level = logging.ERROR if summary["outcome"] == "failed" else logging.INFO
            try:
                self._logger.log(level, "diarize_audio_part.finished", extra=summary)
            except Exception:  # noqa: BLE001, S110
                pass

    def _process_claimed(
        self, claim: AudioPartClaim, summary: dict[str, Any]
    ) -> dict[str, object]:
        audio_part_id = claim.audio_part_id
        workspace: TaskWorkspace | None = None
        active_failure: TaskStageError | None = None
        completed = False
        try:
            if (
                not claim.audio_uri
                or claim.duration_ms is None
                or claim.duration_ms <= 0
            ):
                raise TaskStageError(FailureStage.DOWNLOAD, ErrorCode.INVALID_INPUT_URI)
            summary["audio_duration_seconds"] = round(claim.duration_ms / 1000.0, 3)
            try:
                workspace = self._workspace_factory(
                    prefix=self._task_policy.workspace_prefix,
                    parent=self._workspace_parent,
                )
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.ARTIFACT, ErrorCode.ARTIFACT_WRITE_FAILED
                ) from exc
            try:
                summary["input_bytes"] = self._timed(
                    summary,
                    "elapsed_download_ms",
                    self._storage.download_audio,
                    claim.audio_uri,
                    workspace.paths.audio_path,
                )
            except InvalidS3UriError as exc:
                raise TaskStageError(
                    FailureStage.DOWNLOAD, ErrorCode.INVALID_INPUT_URI
                ) from exc
            except TaskStageError:
                raise
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.DOWNLOAD, ErrorCode.DOWNLOAD_FAILED
                ) from exc

            try:
                inference = self._timed(
                    summary,
                    "elapsed_model_inference_ms",
                    self._diarization.infer,
                    workspace.paths.audio_path,
                )
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.INFERENCE, ErrorCode.INFERENCE_FAILED
                ) from exc
            summary.update(
                device=inference.device,
                accelerator=inference.accelerator,
                model_cache_hit=inference.model_cache_hit,
            )
            try:
                artifact = self._timed(
                    summary,
                    "elapsed_normalize_serialize_ms",
                    self._build_and_write,
                    inference,
                    claim.duration_ms,
                    workspace.paths.artifact_path,
                )
            except TaskStageError:
                raise
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.ARTIFACT, ErrorCode.INVALID_MODEL_OUTPUT
                ) from exc
            summary.update(
                speaker_count=artifact.num_speakers,
                segment_count=len(artifact.segments),
            )
            try:
                prepared_references = self._timed(
                    summary,
                    "elapsed_reference_build_ms",
                    self._build_speaker_references,
                    artifact,
                    claim.audio_uri,
                    claim.duration_ms,
                    workspace,
                )
            except TaskStageError:
                raise
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.ARTIFACT, ErrorCode.SPEAKER_REFERENCE_FAILED
                ) from exc
            summary["reference_speaker_count"] = prepared_references.speaker_count
            try:
                uri = self._timed(
                    summary,
                    "elapsed_upload_ms",
                    self._upload_artifacts,
                    claim.audio_uri,
                    workspace.paths.artifact_path,
                    prepared_references,
                )
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.UPLOAD, ErrorCode.UPLOAD_FAILED
                ) from exc
            try:
                self._timed(
                    summary,
                    "elapsed_persistence_ms",
                    self._repository.complete,
                    audio_part_id,
                    uri,
                )
                completed = True
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.PERSISTENCE, ErrorCode.PERSISTENCE_FAILED
                ) from exc
            self._dispatch(audio_part_id, summary)
            summary.update(outcome="succeeded", final_status="diarized")
            return self._result(
                audio_part_id,
                "diarized",
                dispatched=True,
                speaker_count=artifact.num_speakers,
                reference_speaker_count=prepared_references.speaker_count,
            )
        except TaskStageError as failure:
            active_failure = failure
            if failure.stage is FailureStage.DOWNSTREAM_DISPATCH and completed:
                summary["final_status"] = "failed"
            elif failure.stage is not FailureStage.CLEANUP:
                self._persist_processing_failure(audio_part_id, failure)
                summary["final_status"] = "failed"
            raise
        finally:
            if workspace is not None:
                cleanup_started = self._clock()
                try:
                    workspace.close()
                    summary["cleanup_succeeded"] = True
                except Exception as cleanup_error:
                    summary["cleanup_succeeded"] = False
                    if active_failure is not None:
                        active_failure.add_note(
                            "Temporary workspace cleanup also failed."
                        )
                    else:
                        raise TaskStageError(
                            FailureStage.CLEANUP, ErrorCode.CLEANUP_FAILED
                        ) from cleanup_error
                finally:
                    summary["elapsed_cleanup_ms"] = self._elapsed_ms(cleanup_started)

    def _dispatch(self, audio_part_id: UUID, summary: dict[str, Any]) -> None:
        try:
            self._timed(
                summary, "elapsed_dispatch_ms", self._publisher.publish, audio_part_id
            )
            summary["quality_filter_dispatched"] = True
        except Exception as exc:
            failure = TaskStageError(
                FailureStage.DOWNSTREAM_DISPATCH, ErrorCode.DOWNSTREAM_DISPATCH_FAILED
            )
            try:
                self._repository.mark_dispatch_failed(
                    audio_part_id,
                    safe_message(failure.code, self._task_policy.error_max_length),
                )
                summary["final_status"] = "failed"
            except Exception:  # noqa: BLE001
                failure.add_note("Dispatch failure state could not be persisted.")
            raise failure from exc

    def _persist_processing_failure(
        self, audio_part_id: UUID, failure: TaskStageError
    ) -> None:
        try:
            self._repository.mark_processing_failed(
                audio_part_id,
                safe_message(failure.code, self._task_policy.error_max_length),
            )
        except Exception:  # noqa: BLE001
            failure.add_note("Processing failure state could not be persisted.")

    def _build_and_write(
        self, inference: InferenceResult, duration_ms: int, path: Path
    ) -> DiarizationArtifact:
        try:
            artifact = build_artifact(
                inference.turns,
                model=self._diarization_policy.model,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            raise TaskStageError(
                FailureStage.ARTIFACT, ErrorCode.INVALID_MODEL_OUTPUT
            ) from exc
        try:
            artifact.write(path)
        except Exception as exc:
            raise TaskStageError(
                FailureStage.ARTIFACT, ErrorCode.ARTIFACT_WRITE_FAILED
            ) from exc
        return artifact

    def _build_speaker_references(
        self,
        artifact: DiarizationArtifact,
        audio_uri: str,
        duration_ms: int,
        workspace: TaskWorkspace,
    ) -> PreparedSpeakerReferences:
        try:
            audio = read_normalized_wav(
                workspace.paths.audio_path, expected_duration_ms=duration_ms
            )
            plans = plan_speaker_references(
                artifact.segments, self._speaker_reference_policy
            )
            speakers: list[ManifestSpeaker] = []
            audio_files: list[tuple[int, Path, str]] = []
            for plan in plans:
                path = workspace.speaker_reference_audio_path(plan.speaker_id)
                written = write_reference_wav(
                    audio,
                    path,
                    plan.segments,
                    inter_segment_silence_ms=(
                        self._speaker_reference_policy.inter_segment_silence_ms
                    ),
                )
                uri = self._storage.reference_audio_uri(audio_uri, plan.speaker_id)
                speakers.append(
                    ManifestSpeaker(
                        speaker_id=plan.speaker_id,
                        reference_audio=ReferenceAudio(
                            uri=uri,
                            sample_rate_hz=written.sample_rate_hz,
                            size_bytes=written.size_bytes,
                            sha256=written.sha256,
                            segments=plan.segments,
                            effective_duration_ms=plan.effective_duration_ms,
                            total_duration_ms=written.duration_ms,
                        ),
                    )
                )
                audio_files.append((plan.speaker_id, path, uri))
            manifest = SpeakerReferenceManifest(tuple(speakers))
            manifest.write(workspace.paths.speaker_reference_manifest_path)
            return PreparedSpeakerReferences(
                tuple(audio_files), workspace.paths.speaker_reference_manifest_path
            )
        except TaskStageError:
            raise
        except Exception as exc:
            raise TaskStageError(
                FailureStage.ARTIFACT, ErrorCode.SPEAKER_REFERENCE_FAILED
            ) from exc

    def _upload_artifacts(
        self,
        audio_uri: str,
        diarization_path: Path,
        references: PreparedSpeakerReferences,
    ) -> str:
        diarization_uri = self._storage.upload_artifact(audio_uri, diarization_path)
        for speaker_id, path, expected_uri in references.audio_files:
            actual_uri = self._storage.upload_reference_audio(
                audio_uri, speaker_id, path
            )
            if actual_uri != expected_uri:
                raise RuntimeError("speaker reference URI is inconsistent")
        self._storage.upload_reference_manifest(audio_uri, references.manifest_path)
        return diarization_uri

    def _timed(
        self,
        summary: dict[str, Any],
        field: str,
        operation: Callable[..., Value],
        *args: object,
    ) -> Value:
        started = self._clock()
        try:
            return operation(*args)
        finally:
            summary[field] = self._elapsed_ms(started)

    def _elapsed_ms(self, started: int) -> int:
        return max(0, (self._clock() - started) // 1_000_000)

    def _new_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "event": "diarize_audio_part.finished",
            "audio_part_id": None,
            "outcome": "failed",
            "final_status": None,
            "model": self._diarization_policy.model,
            "device": None,
            "accelerator": None,
            "model_cache_hit": None,
            "audio_duration_seconds": None,
            "input_bytes": None,
            "speaker_count": None,
            "segment_count": None,
            "reference_speaker_count": None,
            "quality_filter_dispatched": None,
            "cleanup_succeeded": None,
            "elapsed_total_ms": None,
            "failure_stage": None,
            "error_code": None,
        }
        summary.update({field: None for field in _TIMING_FIELDS})
        return summary

    @staticmethod
    def _result(
        audio_part_id: UUID,
        status: str,
        *,
        dispatched: bool,
        speaker_count: int | None = None,
        reference_speaker_count: int | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "audio_part_id": str(audio_part_id),
            "status": status,
            "quality_filter_dispatched": dispatched,
        }
        if speaker_count is not None:
            result["speaker_count"] = speaker_count
        if reference_speaker_count is not None:
            result["reference_speaker_count"] = reference_speaker_count
        return result


def register_diarization_task(
    app: Celery, handler: Callable[[str], dict[str, object]]
) -> Task:
    """Register the shared task contract against the composed handler."""

    # +-----------------------------------------------------------------------+
    # | diarize_audio_part(audio_part_id)                                     |
    # +-----------------------------------------------------------------------+
    # | audio_parts.audio_uri -> download normalized part WAV                 |
    # | BUT-FIT/DiariZen -> normalized speaker turns                          |
    # | derive safe pure-speaker references from normalized turns             |
    # | upload diarization.json, reference WAVs, and references.json          |
    # | commit diarization_uri + diarized                                     |
    # | publish quality_filter_audio_part(audio_part_id)                      |
    # | cleanup task workspace                                                |
    # | emit diarize_audio_part.finished                                      |
    # +-----------------------------------------------------------------------+
    @app.task(
        name=DIARIZE_AUDIO_PART.name,
        queue=DIARIZE_AUDIO_PART.queue,
        ignore_result=True,
    )
    def diarize_audio_part(audio_part_id: str) -> dict[str, object]:
        return handler(audio_part_id)

    return diarize_audio_part
