"""Celery task lifecycle orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol, TypeVar
from uuid import UUID, uuid4

from celery import Celery
from celery.app.task import Task
from voice_pipeline_task_contracts import QUALITY_FILTER_AUDIO_PART

from .config import PlannerPolicy, QualityPolicy, TaskPolicy
from .diarization import parse_artifact_bytes, speech_union
from .errors import ErrorCode, FailureStage, TaskStageError, safe_message
from .music import MusicDetector
from .planner import ChunkDraft, plan_chunks
from .quality import align_regions_to_turns, build_good_regions, decide_quality
from .repository import (
    AudioPartClaim,
    AudioPartNotFoundError,
    ClaimDisposition,
    InvalidAudioPartStatusError,
    PersistedChunk,
)
from .snr import wada_snr
from .wav_io import PcmAudio, read_normalized_wav, speech_samples, write_chunk_wav
from .workspace import TaskWorkspace

logger = logging.getLogger(__name__)
Value = TypeVar("Value")
Clock = Callable[[], int]


class RepositoryPort(Protocol):
    def claim(self, audio_part_id: UUID) -> AudioPartClaim: ...
    def complete(
        self, claim: AudioPartClaim, chunks: tuple[PersistedChunk, ...]
    ) -> tuple[UUID, ...]: ...
    def mark_failed(self, audio_part_id: UUID, error: str) -> None: ...


class StoragePort(Protocol):
    def download_audio(self, uri: str, destination: Path) -> int: ...
    def download_diarization(self, uri: str, destination: Path) -> int: ...
    def upload_chunk(self, audio_uri: str, chunk_index: int, path: Path) -> str: ...


class QualityFilterAudioPartHandler:
    def __init__(
        self,
        *,
        repository: RepositoryPort,
        storage: StoragePort,
        music_detector: MusicDetector,
        quality_policy: QualityPolicy,
        planner_policy: PlannerPolicy,
        task_policy: TaskPolicy,
        workspace_parent: Path | None = None,
        workspace_factory: Callable[..., TaskWorkspace] = TaskWorkspace,
        clock: Clock = perf_counter_ns,
        timing_logger: logging.Logger = logger,
        publisher: object | None = None,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._music_detector = music_detector
        self._quality_policy = quality_policy
        self._planner_policy = planner_policy
        self._task_policy = task_policy
        self._workspace_parent = workspace_parent
        self._workspace_factory = workspace_factory
        self._clock = clock
        self._logger = timing_logger
        self._publisher = publisher

    def __call__(self, audio_part_id: str) -> dict[str, object]:
        try:
            parsed_id = UUID(audio_part_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TaskStageError(
                FailureStage.INPUT, ErrorCode.INVALID_AUDIO_PART_ID
            ) from exc
        try:
            claim = self._repository.claim(parsed_id)
        except AudioPartNotFoundError as exc:
            raise TaskStageError(
                FailureStage.INPUT, ErrorCode.AUDIO_PART_NOT_FOUND
            ) from exc
        except InvalidAudioPartStatusError as exc:
            raise TaskStageError(
                FailureStage.INPUT, ErrorCode.INVALID_AUDIO_PART_STATE
            ) from exc
        except Exception as exc:
            raise TaskStageError(
                FailureStage.CLAIM, ErrorCode.PERSISTENCE_FAILED
            ) from exc
        if claim.disposition is not ClaimDisposition.CLAIMED:
            return {
                "audio_part_id": str(parsed_id),
                "outcome": claim.disposition.value,
                "created_count": 0,
            }
        return self._process_claimed(claim)

    def _process_claimed(self, claim: AudioPartClaim) -> dict[str, object]:
        workspace: TaskWorkspace | None = None
        active_failure: TaskStageError | None = None
        try:
            if (
                not claim.audio_uri
                or not claim.diarization_uri
                or claim.duration_ms is None
                or claim.duration_ms <= 0
                or not claim.lang
            ):
                raise TaskStageError(FailureStage.INPUT, ErrorCode.INVALID_INPUT)
            workspace = self._workspace_factory(
                prefix=self._task_policy.workspace_prefix, parent=self._workspace_parent
            )
            try:
                self._stage(
                    claim.audio_part_id,
                    "none",
                    "download_audio",
                    self._storage.download_audio,
                    claim.audio_uri,
                    workspace.paths.audio,
                )
                self._stage(
                    claim.audio_part_id,
                    "none",
                    "download_diarization",
                    self._storage.download_diarization,
                    claim.diarization_uri,
                    workspace.paths.diarization,
                )
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.DOWNLOAD, ErrorCode.DOWNLOAD_FAILED
                ) from exc
            try:
                audio = read_normalized_wav(
                    workspace.paths.audio, expected_duration_ms=claim.duration_ms
                )
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.VALIDATION, ErrorCode.INVALID_AUDIO
                ) from exc
            try:
                artifact = parse_artifact_bytes(
                    workspace.paths.diarization.read_bytes(),
                    expected_duration_ms=claim.duration_ms,
                )
                speech = speech_union(artifact.turns)
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.VALIDATION, ErrorCode.INVALID_DIARIZATION
                ) from exc
            try:
                music = self._stage(
                    claim.audio_part_id,
                    self._music_detector.model_name,
                    "music_detection",
                    self._music_detector.detect,
                    audio.waveform,
                    sample_rate=audio.sample_rate,
                    duration_ms=claim.duration_ms,
                )
            except TaskStageError:
                raise
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.MUSIC, ErrorCode.MUSIC_DETECTION_FAILED
                ) from exc
            try:
                snr_values = self._stage(
                    claim.audio_part_id,
                    "wada-snr",
                    "snr",
                    self._calculate_snr,
                    audio,
                    speech,
                )
            except Exception as exc:
                raise TaskStageError(FailureStage.SNR, ErrorCode.SNR_FAILED) from exc
            try:
                decisions = decide_quality(
                    speech, snr_values, music, self._quality_policy
                )
                regions = align_regions_to_turns(
                    build_good_regions(decisions, music, self._quality_policy),
                    artifact.turns,
                )
                drafts = self._stage(
                    claim.audio_part_id,
                    artifact.model,
                    "window_planning",
                    plan_chunks,
                    artifact.turns,
                    regions,
                    self._planner_policy,
                )
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.PLANNING, ErrorCode.PLANNING_FAILED
                ) from exc
            prepared = self._prepare_and_upload(claim, audio, drafts, workspace)
            try:
                workspace.close()
                workspace = None
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.CLEANUP, ErrorCode.CLEANUP_FAILED
                ) from exc
            try:
                chunk_ids = self._stage(
                    claim.audio_part_id,
                    "none",
                    "persist",
                    self._repository.complete,
                    claim,
                    prepared,
                )
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.PERSISTENCE, ErrorCode.PERSISTENCE_FAILED
                ) from exc
            if self._publisher is not None:
                try:
                    for chunk_id in chunk_ids:
                        self._publisher.publish(chunk_id)
                except Exception as exc:
                    raise TaskStageError(
                        FailureStage.PERSISTENCE, ErrorCode.PERSISTENCE_FAILED
                    ) from exc
            return {
                "audio_part_id": str(claim.audio_part_id),
                "outcome": "completed",
                "created_count": len(chunk_ids),
            }
        except TaskStageError as failure:
            active_failure = failure
            try:
                self._repository.mark_failed(
                    claim.audio_part_id,
                    safe_message(failure.code, self._task_policy.error_max_length),
                )
            except Exception:
                failure.add_note("Failure state could not be persisted.")
            raise
        except Exception as exc:
            failure = TaskStageError(FailureStage.VALIDATION, ErrorCode.INVALID_INPUT)
            active_failure = failure
            try:
                self._repository.mark_failed(
                    claim.audio_part_id,
                    safe_message(failure.code, self._task_policy.error_max_length),
                )
            except Exception:
                failure.add_note("Failure state could not be persisted.")
            raise failure from exc
        finally:
            if workspace is not None:
                try:
                    workspace.close()
                except Exception as exc:
                    if active_failure is not None:
                        active_failure.add_note(
                            "Temporary workspace cleanup also failed."
                        )
                    else:
                        raise TaskStageError(
                            FailureStage.CLEANUP, ErrorCode.CLEANUP_FAILED
                        ) from exc

    def _prepare_and_upload(
        self,
        claim: AudioPartClaim,
        audio: PcmAudio,
        drafts: tuple[ChunkDraft, ...],
        workspace: TaskWorkspace,
    ) -> tuple[PersistedChunk, ...]:
        output: list[PersistedChunk] = []
        for draft in drafts:
            chunk_id = uuid4()
            path = workspace.chunk_path(draft.chunk_index)
            try:
                self._stage(
                    claim.audio_part_id,
                    "none",
                    "cut_chunk_audio",
                    write_chunk_wav,
                    audio,
                    path,
                    start_ms=draft.start_ms,
                    end_ms=draft.end_ms,
                    chunk_id=chunk_id,
                )
            except Exception as exc:
                raise TaskStageError(FailureStage.CUT, ErrorCode.CUT_FAILED) from exc
            try:
                uri = self._stage(
                    claim.audio_part_id,
                    "none",
                    "upload_chunk_audio",
                    self._storage.upload_chunk,
                    claim.audio_uri,
                    draft.chunk_index,
                    path,
                    chunk_id=chunk_id,
                )
            except Exception as exc:
                raise TaskStageError(
                    FailureStage.UPLOAD, ErrorCode.UPLOAD_FAILED
                ) from exc
            output.append(
                PersistedChunk(
                    chunk_id,
                    draft.chunk_index,
                    uri,
                    claim.lang or "",
                    draft.duration_ms,
                    draft.start_ms,
                    draft.end_ms,
                )
            )
        return tuple(output)

    @staticmethod
    def _calculate_snr(audio: PcmAudio, intervals: tuple) -> tuple[float, ...]:
        return tuple(
            wada_snr(
                speech_samples(
                    audio, start_ms=interval.start_ms, end_ms=interval.end_ms
                )
            )
            for interval in intervals
        )

    def _stage(
        self,
        audio_part_id: UUID,
        model_name: str,
        stage: str,
        operation: Callable[..., Value],
        *args: object,
        chunk_id: UUID | None = None,
        **kwargs: object,
    ) -> Value:
        started = self._clock()
        try:
            return operation(*args, **kwargs)
        finally:
            elapsed_ms = max(0, (self._clock() - started + 500_000) // 1_000_000)
            extra = {
                "audio_part_id": str(audio_part_id),
                "model_name": model_name,
                "stage": stage,
                "elapsed_ms": elapsed_ms,
            }
            if chunk_id is not None:
                extra["chunk_id"] = str(chunk_id)
            try:
                self._logger.info("quality_filter_audio_part.timing", extra=extra)
            except Exception:
                pass


def register_quality_filter_task(
    app: Celery, handler: Callable[[str], dict[str, object]]
) -> Task:
    """Register the approved UUID-only task contract."""

    # +-----------------------------------------------------------------------+
    # | quality_filter_audio_part(audio_part_id)                              |
    # +-----------------------------------------------------------------------+
    # | part WAV + diarization.json -> speech union                           |
    # | whole-part music + per-speech WADA SNR                                |
    # | good regions -> strict two-speaker windows -> region-scoped merge     |
    # | upload chunk WAVs -> atomically create chunks + complete part         |
    # | commit chunks -> best-effort publish separate_chunk UUID tasks        |
    # | cleanup task workspace                                                |
    # +-----------------------------------------------------------------------+
    @app.task(
        name=QUALITY_FILTER_AUDIO_PART.name,
        queue=QUALITY_FILTER_AUDIO_PART.queue,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def quality_filter_audio_part(_task: Task, audio_part_id: str) -> dict[str, object]:
        return handler(audio_part_id)

    return quality_filter_audio_part
