from __future__ import annotations

import json
import logging
import struct
import wave
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from voice_pipeline_split_raw_audio_into_parts.config import WindowingPolicy
from voice_pipeline_split_raw_audio_into_parts.errors import (
    FailureReason,
    safe_failure_message,
)
from voice_pipeline_split_raw_audio_into_parts.repository import (
    AudioPartDraft,
    ClaimDisposition,
    RawAudioClaim,
)
from voice_pipeline_split_raw_audio_into_parts.task import (
    SplitRawAudioIntoPartsHandler,
    TaskStageError,
    _format_wav_duration,
)
from voice_pipeline_split_raw_audio_into_parts.vad import VadResult
from voice_pipeline_split_raw_audio_into_parts.wav_io import (
    CHANNEL_COUNT,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    cut_wav_frames,
)
from voice_pipeline_split_raw_audio_into_parts.windowing import FrameSpan


RAW_AUDIO_ID = UUID("12345678-1234-5678-1234-567812345678")
AUDIO_PART_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


def _write_wav(path: Path, frame_count: int = SAMPLE_RATE) -> None:
    samples = [index % 1_000 for index in range(frame_count)]
    payload = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(CHANNEL_COUNT)
        writer.setsampwidth(SAMPLE_WIDTH_BYTES)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(payload)


class FakeRepository:
    def __init__(
        self,
        events: list[str],
        *,
        disposition: ClaimDisposition = ClaimDisposition.CLAIMED,
        pending_ids: list[UUID] | None = None,
        total_count: int = 1,
        fail_at: str | None = None,
        audio_uri: str | None = "s3://test-bucket/raw.wav",
        lang: str | None = "en",
    ) -> None:
        self.events = events
        self.disposition = disposition
        self.pending_ids = list(pending_ids or [])
        self.total_count = total_count
        self.fail_at = fail_at
        self.audio_uri = audio_uri
        self.lang = lang
        self.drafts: list[AudioPartDraft] = []
        self.failure_error: str | None = None

    def _event(self, name: str) -> None:
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"unsafe {name} detail")

    def claim(self, raw_audio_id: UUID) -> RawAudioClaim:
        self._event("claim")
        return RawAudioClaim(
            raw_audio_id=raw_audio_id,
            disposition=self.disposition,
            status=self.disposition.value,
            audio_uri=(
                self.audio_uri
                if self.disposition is ClaimDisposition.CLAIMED
                else None
            ),
            lang=(
                self.lang
                if self.disposition is ClaimDisposition.CLAIMED
                else None
            ),
        )

    def persist_parts_and_complete(
        self,
        raw_audio_id: UUID,
        drafts: list[AudioPartDraft],
    ) -> list[Any]:
        self._event("persist")
        self.drafts = list(drafts)
        return []

    def list_pending_audio_part_ids(self, raw_audio_id: UUID) -> list[UUID]:
        self._event("pending")
        return list(self.pending_ids)

    def count_audio_parts(self, raw_audio_id: UUID) -> int:
        self._event("count")
        return self.total_count

    def mark_failed(self, raw_audio_id: UUID, error: str) -> None:
        self.events.append("mark_failed")
        if self.fail_at == "mark_failed":
            raise RuntimeError("unsafe failure update detail")
        self.failure_error = error


class FakeStorage:
    def __init__(self, events: list[str], *, fail_at: str | None = None) -> None:
        self.events = events
        self.fail_at = fail_at
        self.vad_document: dict[str, Any] | None = None

    def _event(self, name: str) -> None:
        self.events.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"unsafe {name} detail")

    def download_raw_audio(self, audio_uri: str, destination: Path) -> None:
        self._event("download")
        _write_wav(destination)

    def upload_vad_segments(self, raw_audio_id: UUID, path: Path) -> str:
        self._event("vad_upload")
        self.vad_document = json.loads(path.read_text(encoding="utf-8"))
        return f"s3://test-bucket/raw_audios/{raw_audio_id}/vad_segments.json"

    def upload_audio_part(
        self,
        raw_audio_id: UUID,
        part_index: int,
        path: Path,
    ) -> str:
        self._event("part_upload")
        assert path.stat().st_size > 0
        return (
            f"s3://test-bucket/raw_audios/{raw_audio_id}/"
            f"audio_parts/{part_index}/audio.wav"
        )


class FakeVad:
    def __init__(
        self,
        events: list[str],
        *,
        segments: tuple[FrameSpan, ...] = (FrameSpan(1_600, 14_400),),
        fail: bool = False,
    ) -> None:
        self.events = events
        self.segments = segments
        self.fail = fail

    def run(self, audio_path: Path) -> VadResult:
        self.events.append("vad")
        if self.fail:
            raise RuntimeError("unsafe model detail")
        return VadResult(
            model="pyannote/segmentation-3.0",
            audio_frame_count=SAMPLE_RATE,
            segments=self.segments,
        )


class FakePublisher:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail
        self.ids: list[UUID] = []

    def publish(self, audio_part_id: UUID) -> str:
        self.events.append("publish")
        self.ids.append(audio_part_id)
        if self.fail:
            raise RuntimeError("unsafe broker detail")
        return "task-id"


def _policy() -> WindowingPolicy:
    return WindowingPolicy(
        gap_threshold_ms=0,
        min_window_ms=100,
        max_window_ms=1_000,
        pad_before_ms=0,
        pad_after_ms=0,
    )


def _handler(
    tmp_path: Path,
    *,
    repository: FakeRepository,
    storage: FakeStorage,
    vad: FakeVad,
    publisher: FakePublisher,
    **kwargs: Any,
) -> SplitRawAudioIntoPartsHandler:
    return SplitRawAudioIntoPartsHandler(
        repository=repository,
        storage=storage,
        vad=vad,
        publisher=publisher,
        windowing_policy=_policy(),
        workspace_parent=tmp_path,
        **kwargs,
    )


def test_wav_duration_format_includes_hours_and_milliseconds() -> None:
    frame_count = 3_723_004 * (SAMPLE_RATE // 1_000)

    assert _format_wav_duration(frame_count) == "01:02:03.004"


def test_vad_summary_is_logged_on_one_line(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: list[str] = []
    repository = FakeRepository(events)
    clock_values = iter([100.0, 112.3456])
    handler = _handler(
        tmp_path,
        repository=repository,
        storage=FakeStorage(events),
        vad=FakeVad(events),
        publisher=FakePublisher(events),
        clock=lambda: next(clock_values),
    )

    with caplog.at_level(logging.INFO):
        handler(str(RAW_AUDIO_ID))

    summary_records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("VAD completed ")
    ]
    assert len(summary_records) == 1
    assert summary_records[0].getMessage() == (
        f"VAD completed raw_audio_id={RAW_AUDIO_ID} wav_size_mb=0.032 "
        "wav_duration=00:00:01.000 "
        "vad_model=pyannote/segmentation-3.0 vad_elapsed_seconds=12.346"
    )


def test_claimed_task_orders_artifacts_commit_and_dispatch(tmp_path: Path) -> None:
    events: list[str] = []
    repository = FakeRepository(events, pending_ids=[AUDIO_PART_ID])
    storage = FakeStorage(events)
    publisher = FakePublisher(events)
    handler = _handler(
        tmp_path,
        repository=repository,
        storage=storage,
        vad=FakeVad(events),
        publisher=publisher,
    )

    result = handler(str(RAW_AUDIO_ID))

    assert events == [
        "claim",
        "download",
        "vad",
        "vad_upload",
        "part_upload",
        "persist",
        "count",
        "pending",
        "publish",
    ]
    assert result == {
        "raw_audio_id": str(RAW_AUDIO_ID),
        "status": "split_completed",
        "audio_part_count": 1,
        "diarization_dispatch_count": 1,
    }
    assert storage.vad_document == {
        "schema_version": 1,
        "model": "pyannote/segmentation-3.0",
        "audio_duration_ms": 1_000,
        "segments": [
            {
                "index": 0,
                "start_ms": 100,
                "end_ms": 900,
                "duration_ms": 800,
            }
        ],
    }
    assert repository.drafts == [
        AudioPartDraft(
            part_index=0,
            audio_uri=(
                f"s3://test-bucket/raw_audios/{RAW_AUDIO_ID}/"
                "audio_parts/0/audio.wav"
            ),
            lang="en",
            relative_start_ms=100,
            relative_end_ms=900,
            duration_ms=800,
        )
    ]
    assert publisher.ids == [AUDIO_PART_ID]
    assert list(tmp_path.iterdir()) == []


def test_zero_windows_still_uploads_empty_vad_and_completes(tmp_path: Path) -> None:
    events: list[str] = []
    repository = FakeRepository(events, total_count=0)
    storage = FakeStorage(events)
    publisher = FakePublisher(events)
    handler = _handler(
        tmp_path,
        repository=repository,
        storage=storage,
        vad=FakeVad(events, segments=()),
        publisher=publisher,
    )

    result = handler(str(RAW_AUDIO_ID))

    assert storage.vad_document is not None
    assert storage.vad_document["segments"] == []
    assert repository.drafts == []
    assert "part_upload" not in events
    assert "publish" not in events
    assert result["audio_part_count"] == 0
    assert result["diarization_dispatch_count"] == 0
    assert list(tmp_path.iterdir()) == []


def test_invalid_uuid_fails_before_repository_access(tmp_path: Path) -> None:
    events: list[str] = []
    repository = FakeRepository(events)
    handler = _handler(
        tmp_path,
        repository=repository,
        storage=FakeStorage(events),
        vad=FakeVad(events),
        publisher=FakePublisher(events),
    )

    with pytest.raises(TaskStageError) as failure:
        handler("not-a-uuid")

    assert failure.value.reason is FailureReason.INPUT_INVALID
    assert events == []


def test_missing_audio_uri_is_a_safe_owned_failure(tmp_path: Path) -> None:
    events: list[str] = []
    repository = FakeRepository(events, audio_uri=None)
    handler = _handler(
        tmp_path,
        repository=repository,
        storage=FakeStorage(events),
        vad=FakeVad(events),
        publisher=FakePublisher(events),
    )

    with pytest.raises(TaskStageError) as failure:
        handler(str(RAW_AUDIO_ID))

    assert failure.value.reason is FailureReason.AUDIO_URI_MISSING
    assert repository.failure_error == safe_failure_message(
        FailureReason.AUDIO_URI_MISSING
    )
    assert events == ["claim", "mark_failed"]


def test_splitting_returns_without_io_or_dispatch(tmp_path: Path) -> None:
    events: list[str] = []
    repository = FakeRepository(
        events,
        disposition=ClaimDisposition.ALREADY_PROCESSING,
    )
    handler = _handler(
        tmp_path,
        repository=repository,
        storage=FakeStorage(events),
        vad=FakeVad(events),
        publisher=FakePublisher(events),
    )

    result = handler(str(RAW_AUDIO_ID))

    assert events == ["claim"]
    assert result["status"] == "already_processing"
    assert list(tmp_path.iterdir()) == []


def test_completed_only_recovers_pending_dispatch(tmp_path: Path) -> None:
    events: list[str] = []
    repository = FakeRepository(
        events,
        disposition=ClaimDisposition.COMPLETED,
        pending_ids=[AUDIO_PART_ID],
        total_count=3,
    )
    publisher = FakePublisher(events)
    handler = _handler(
        tmp_path,
        repository=repository,
        storage=FakeStorage(events),
        vad=FakeVad(events),
        publisher=publisher,
    )

    result = handler(str(RAW_AUDIO_ID))

    assert events == ["claim", "count", "pending", "publish"]
    assert publisher.ids == [AUDIO_PART_ID]
    assert result["audio_part_count"] == 3
    assert result["diarization_dispatch_count"] == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("failure_point", "reason"),
    [
        ("download", FailureReason.DOWNLOAD_FAILED),
        ("vad", FailureReason.INFERENCE_FAILED),
        ("vad_upload", FailureReason.VAD_ARTIFACT_FAILED),
        ("grouping", FailureReason.GROUPING_FAILED),
        ("cutting", FailureReason.CUTTING_FAILED),
        ("part_upload", FailureReason.UPLOAD_FAILED),
        ("persist", FailureReason.PERSISTENCE_FAILED),
        ("count", FailureReason.PERSISTENCE_FAILED),
        ("pending", FailureReason.PERSISTENCE_FAILED),
        ("publish", FailureReason.DOWNSTREAM_DISPATCH_FAILED),
    ],
)
def test_stage_failures_persist_safe_error_and_clean_workspace(
    tmp_path: Path,
    failure_point: str,
    reason: FailureReason,
) -> None:
    events: list[str] = []
    repository = FakeRepository(
        events,
        pending_ids=[AUDIO_PART_ID],
        fail_at=(
            failure_point
            if failure_point in {"persist", "count", "pending"}
            else None
        ),
    )
    storage = FakeStorage(
        events,
        fail_at=(
            failure_point
            if failure_point in {"download", "vad_upload", "part_upload"}
            else None
        ),
    )
    vad = FakeVad(events, fail=failure_point == "vad")
    publisher = FakePublisher(events, fail=failure_point == "publish")

    def window_builder(*args: Any, **kwargs: Any) -> list[Any]:
        if failure_point == "grouping":
            raise RuntimeError("unsafe grouping detail")
        from voice_pipeline_split_raw_audio_into_parts.windowing import build_windows

        return build_windows(*args, **kwargs)

    def wav_cutter(*args: Any, **kwargs: Any) -> Any:
        if failure_point == "cutting":
            raise RuntimeError("unsafe cutting detail")
        return cut_wav_frames(*args, **kwargs)

    handler = _handler(
        tmp_path,
        repository=repository,
        storage=storage,
        vad=vad,
        publisher=publisher,
        window_builder=window_builder,
        wav_cutter=wav_cutter,
    )

    with pytest.raises(TaskStageError) as failure:
        handler(str(RAW_AUDIO_ID))

    assert failure.value.reason is reason
    assert str(failure.value) == safe_failure_message(reason)
    assert repository.failure_error == safe_failure_message(reason)
    assert "unsafe" not in repository.failure_error
    assert events[-1] == "mark_failed"
    assert list(tmp_path.iterdir()) == []


def test_failure_state_error_does_not_replace_primary_failure(tmp_path: Path) -> None:
    events: list[str] = []
    repository = FakeRepository(events, fail_at="mark_failed")
    handler = _handler(
        tmp_path,
        repository=repository,
        storage=FakeStorage(events, fail_at="download"),
        vad=FakeVad(events),
        publisher=FakePublisher(events),
    )

    with pytest.raises(TaskStageError) as failure:
        handler(str(RAW_AUDIO_ID))

    assert failure.value.reason is FailureReason.DOWNLOAD_FAILED
    assert failure.value.__notes__ == [
        "Raw audio failure state could not be persisted."
    ]
