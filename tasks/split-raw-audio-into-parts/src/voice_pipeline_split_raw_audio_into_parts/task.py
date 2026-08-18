from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from time import perf_counter
from typing import Protocol, TypeVar
from uuid import UUID

from celery import Celery
from celery.app.task import Task
from voice_pipeline_task_contracts import SPLIT_RAW_AUDIO_INTO_PARTS

from .config import WindowingPolicy
from .errors import FailureReason, safe_failure_message
from .repository import (
    AudioPartDraft,
    ClaimDisposition,
    PersistedAudioPart,
    RawAudioClaim,
)
from .vad import VadResult
from .vad_artifact import write_vad_artifact
from .wav_io import SAMPLE_RATE, WavClip, cut_wav_frames
from .windowing import IndexedWindow, build_windows
from .workspace import TaskWorkspace, task_workspace

SplitTaskHandler = Callable[[str], dict[str, object]]
WindowBuilder = Callable[..., list[IndexedWindow]]
WavCutter = Callable[..., WavClip]
Clock = Callable[[], float]
WorkspaceFactory = Callable[
    [Path | None],
    AbstractContextManager[TaskWorkspace],
]
ResultValue = TypeVar("ResultValue")
logger = logging.getLogger(__name__)
BYTES_PER_MB = 1_000_000


def _format_wav_duration(frame_count: int) -> str:
    """Format a 16 kHz frame count as HH:MM:SS.mmm."""

    total_ms = round(frame_count * 1_000 / SAMPLE_RATE)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    seconds, milliseconds = divmod(remainder_ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class RepositoryPort(Protocol):
    def claim(self, raw_audio_id: UUID) -> RawAudioClaim: ...

    def persist_parts_and_complete(
        self,
        raw_audio_id: UUID,
        drafts: list[AudioPartDraft],
    ) -> list[PersistedAudioPart]: ...

    def list_pending_audio_part_ids(self, raw_audio_id: UUID) -> list[UUID]: ...

    def count_audio_parts(self, raw_audio_id: UUID) -> int: ...

    def mark_failed(self, raw_audio_id: UUID, error: str) -> None: ...


class StoragePort(Protocol):
    def download_raw_audio(self, audio_uri: str, destination: Path) -> None: ...

    def upload_vad_segments(self, raw_audio_id: UUID, path: Path) -> str: ...

    def upload_audio_part(
        self,
        raw_audio_id: UUID,
        part_index: int,
        path: Path,
    ) -> str: ...


class VadPort(Protocol):
    def run(self, audio_path: Path) -> VadResult: ...


class PublisherPort(Protocol):
    def publish(self, audio_part_id: UUID) -> str: ...


class TaskStageError(RuntimeError):
    """Safe stage-specific failure re-raised for Celery observability."""

    def __init__(self, reason: FailureReason) -> None:
        self.reason = reason
        super().__init__(safe_failure_message(reason))


class SplitRawAudioIntoPartsHandler:
    """Compose status, VAD, artifacts, persistence, and downstream dispatch."""

    def __init__(
        self,
        *,
        repository: RepositoryPort,
        storage: StoragePort,
        vad: VadPort,
        publisher: PublisherPort,
        windowing_policy: WindowingPolicy,
        workspace_parent: Path | None = None,
        window_builder: WindowBuilder = build_windows,
        wav_cutter: WavCutter = cut_wav_frames,
        workspace_factory: WorkspaceFactory = task_workspace,
        clock: Clock = perf_counter,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._vad = vad
        self._publisher = publisher
        self._windowing_policy = windowing_policy
        self._workspace_parent = workspace_parent
        self._window_builder = window_builder
        self._wav_cutter = wav_cutter
        self._workspace_factory = workspace_factory
        self._clock = clock

    def __call__(self, raw_audio_id: str) -> dict[str, object]:
        try:
            parsed_id = UUID(raw_audio_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TaskStageError(FailureReason.INPUT_INVALID) from exc

        claim = self._repository.claim(parsed_id)
        if claim.disposition is ClaimDisposition.ALREADY_PROCESSING:
            return self._result(
                parsed_id,
                status="already_processing",
                audio_part_count=0,
                dispatch_count=0,
            )
        if claim.disposition is ClaimDisposition.COMPLETED:
            return self._recover_dispatch(parsed_id)
        return self._process_claimed(claim)

    def _process_claimed(self, claim: RawAudioClaim) -> dict[str, object]:
        raw_audio_id = claim.raw_audio_id
        try:
            if not claim.audio_uri:
                raise TaskStageError(FailureReason.AUDIO_URI_MISSING)
            if not claim.lang:
                raise TaskStageError(FailureReason.INPUT_INVALID)

            try:
                with self._workspace_factory(self._workspace_parent) as workspace:
                    self._run_stage(
                        FailureReason.DOWNLOAD_FAILED,
                        lambda: self._storage.download_raw_audio(
                            claim.audio_uri,
                            workspace.raw_audio_path,
                        ),
                    )
                    wav_size_bytes = self._run_stage(
                        FailureReason.DOWNLOAD_FAILED,
                        lambda: workspace.raw_audio_path.stat().st_size,
                    )
                    vad_started_at = self._clock()
                    vad_result = self._run_stage(
                        FailureReason.INFERENCE_FAILED,
                        lambda: self._vad.run(workspace.raw_audio_path),
                    )
                    vad_elapsed_seconds = max(
                        0.0,
                        self._clock() - vad_started_at,
                    )
                    self._log_vad_summary(
                        raw_audio_id=raw_audio_id,
                        wav_size_bytes=wav_size_bytes,
                        vad_result=vad_result,
                        elapsed_seconds=vad_elapsed_seconds,
                    )
                    self._run_stage(
                        FailureReason.VAD_ARTIFACT_FAILED,
                        lambda: write_vad_artifact(
                            workspace.vad_segments_path,
                            vad_result.artifact_document(),
                        ),
                    )
                    self._run_stage(
                        FailureReason.VAD_ARTIFACT_FAILED,
                        lambda: self._storage.upload_vad_segments(
                            raw_audio_id,
                            workspace.vad_segments_path,
                        ),
                    )
                    windows = self._run_stage(
                        FailureReason.GROUPING_FAILED,
                        lambda: self._window_builder(
                            vad_result.segments,
                            audio_frame_count=vad_result.audio_frame_count,
                            policy=self._windowing_policy,
                        ),
                    )
                    drafts = self._create_and_upload_parts(
                        raw_audio_id=raw_audio_id,
                        lang=claim.lang,
                        workspace=workspace,
                        windows=windows,
                    )
                    self._run_stage(
                        FailureReason.PERSISTENCE_FAILED,
                        lambda: self._repository.persist_parts_and_complete(
                            raw_audio_id,
                            drafts,
                        ),
                    )
            except TaskStageError:
                raise
            except Exception as exc:
                raise TaskStageError(FailureReason.CUTTING_FAILED) from exc

            return self._dispatch_completed(raw_audio_id)
        except TaskStageError as failure:
            self._mark_failed(raw_audio_id, failure)
            raise

    @staticmethod
    def _log_vad_summary(
        *,
        raw_audio_id: UUID,
        wav_size_bytes: int,
        vad_result: VadResult,
        elapsed_seconds: float,
    ) -> None:
        logger.info(
            (
                "VAD completed raw_audio_id=%s wav_size_mb=%.3f "
                "wav_duration=%s vad_model=%s vad_elapsed_seconds=%.3f"
            ),
            raw_audio_id,
            wav_size_bytes / BYTES_PER_MB,
            _format_wav_duration(vad_result.audio_frame_count),
            vad_result.model,
            elapsed_seconds,
        )

    def _create_and_upload_parts(
        self,
        *,
        raw_audio_id: UUID,
        lang: str,
        workspace: TaskWorkspace,
        windows: list[IndexedWindow],
    ) -> list[AudioPartDraft]:
        drafts: list[AudioPartDraft] = []
        for window in windows:
            part_path = workspace.audio_part_path(window.part_index)
            clip = self._run_stage(
                FailureReason.CUTTING_FAILED,
                lambda window=window, part_path=part_path: self._wav_cutter(
                    workspace.raw_audio_path,
                    part_path,
                    start_frame=window.start_frame,
                    end_frame=window.end_frame,
                ),
            )
            audio_uri = self._run_stage(
                FailureReason.UPLOAD_FAILED,
                lambda part_path=part_path, window=window: (
                    self._storage.upload_audio_part(
                        raw_audio_id,
                        window.part_index,
                        part_path,
                    )
                ),
            )
            drafts.append(
                AudioPartDraft(
                    part_index=window.part_index,
                    audio_uri=audio_uri,
                    lang=lang,
                    relative_start_ms=clip.relative_start_ms,
                    relative_end_ms=clip.relative_end_ms,
                    duration_ms=clip.duration_ms,
                )
            )
        return drafts

    def _recover_dispatch(self, raw_audio_id: UUID) -> dict[str, object]:
        try:
            return self._dispatch_completed(raw_audio_id)
        except TaskStageError as failure:
            self._mark_failed(raw_audio_id, failure)
            raise

    def _dispatch_completed(self, raw_audio_id: UUID) -> dict[str, object]:
        audio_part_count = self._run_stage(
            FailureReason.PERSISTENCE_FAILED,
            lambda: self._repository.count_audio_parts(raw_audio_id),
        )
        pending_ids = self._run_stage(
            FailureReason.PERSISTENCE_FAILED,
            lambda: self._repository.list_pending_audio_part_ids(raw_audio_id),
        )
        dispatch_count = self._run_stage(
            FailureReason.DOWNSTREAM_DISPATCH_FAILED,
            lambda: self._publish_all(pending_ids),
        )
        return self._result(
            raw_audio_id,
            status="split_completed",
            audio_part_count=audio_part_count,
            dispatch_count=dispatch_count,
        )

    def _publish_all(self, audio_part_ids: list[UUID]) -> int:
        for audio_part_id in audio_part_ids:
            self._publisher.publish(audio_part_id)
        return len(audio_part_ids)

    def _mark_failed(self, raw_audio_id: UUID, failure: TaskStageError) -> None:
        try:
            self._repository.mark_failed(raw_audio_id, str(failure))
        except Exception:
            failure.add_note("Raw audio failure state could not be persisted.")

    @staticmethod
    def _run_stage(
        reason: FailureReason,
        operation: Callable[[], ResultValue],
    ) -> ResultValue:
        try:
            return operation()
        except TaskStageError:
            raise
        except Exception as exc:
            raise TaskStageError(reason) from exc

    @staticmethod
    def _result(
        raw_audio_id: UUID,
        *,
        status: str,
        audio_part_count: int,
        dispatch_count: int,
    ) -> dict[str, object]:
        return {
            "raw_audio_id": str(raw_audio_id),
            "status": status,
            "audio_part_count": audio_part_count,
            "diarization_dispatch_count": dispatch_count,
        }


def register_split_task(app: Celery, handler: SplitTaskHandler) -> Task:
    """Register the shared Celery contract against a composed task handler."""

    # +------------------------------------------------------------------------+
    # | split_raw_audio_into_parts(raw_audio_id)                               |
    # +------------------------------------------------------------------------+
    # | INPUT DEPENDENCY: 16 kHz, mono, 16-bit PCM WAV                         |
    # | raw_audios.audio_uri -> download                                       |
    # | VAD MODEL: pyannote/segmentation-3.0 -> normalized speech segments     |
    # | persist vad_segments.json                                              |
    # | group windows -> cut/upload PCM WAV parts -> persist audio_parts       |
    # | send_task: diarize_audio_part(audio_part_id)                           |
    # +------------------------------------------------------------------------+
    @app.task(
        name=SPLIT_RAW_AUDIO_INTO_PARTS.name,
        queue=SPLIT_RAW_AUDIO_INTO_PARTS.queue,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
    )
    def split_raw_audio_into_parts(
        _task: Task,
        raw_audio_id: str,
    ) -> dict[str, object]:
        return handler(raw_audio_id)

    return split_raw_audio_into_parts
