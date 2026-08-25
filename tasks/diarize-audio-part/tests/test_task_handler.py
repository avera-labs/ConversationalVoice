import json
import wave
from pathlib import Path
from uuid import UUID

import pytest

from voice_pipeline_diarize_audio_part.artifact import RawTurn
from voice_pipeline_diarize_audio_part.config import (
    DiarizationPolicy,
    SpeakerReferencePolicy,
    TaskPolicy,
)
from voice_pipeline_diarize_audio_part.diarization import InferenceResult
from voice_pipeline_diarize_audio_part.errors import ErrorCode, TaskStageError
from voice_pipeline_diarize_audio_part.repository import (
    AudioPartClaim,
    ClaimDisposition,
)
from voice_pipeline_diarize_audio_part.task import DiarizeAudioPartHandler

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")


class Repository:
    def __init__(
        self,
        disposition: ClaimDisposition = ClaimDisposition.CLAIMED,
        *,
        duration_ms: int = 2000,
    ) -> None:
        self.disposition = disposition
        self.duration_ms = duration_ms
        self.events: list[tuple] = []

    def claim(self, identifier: UUID) -> AudioPartClaim:
        self.events.append(("claim", identifier))
        status = {
            ClaimDisposition.CLAIMED: "diarizing",
            ClaimDisposition.DISPATCH_READY: "diarized",
            ClaimDisposition.ALREADY_PROCESSING: "diarizing",
            ClaimDisposition.COMPLETED: "completed",
        }[self.disposition]
        return AudioPartClaim(
            identifier,
            self.disposition,
            status,
            audio_uri=("s3://bucket/raw_audios/raw-id/audio_parts/0/audio.wav")
            if self.disposition is ClaimDisposition.CLAIMED
            else None,
            duration_ms=(
                self.duration_ms
                if self.disposition is ClaimDisposition.CLAIMED
                else None
            ),
        )

    def complete(self, identifier: UUID, uri: str) -> None:
        self.events.append(("complete", identifier, uri))

    def mark_processing_failed(self, identifier: UUID, error: str) -> None:
        self.events.append(("processing_failed", identifier, error))

    def mark_dispatch_failed(self, identifier: UUID, error: str) -> None:
        self.events.append(("dispatch_failed", identifier, error))


class Storage:
    def __init__(
        self,
        events: list[str],
        *,
        fail_download: bool = False,
        fail_reference_manifest: bool = False,
        duration_ms: int = 2000,
    ) -> None:
        self.events = events
        self.fail_download = fail_download
        self.fail_reference_manifest = fail_reference_manifest
        self.duration_ms = duration_ms
        self.reference_manifest: dict | None = None

    def download_audio(self, _uri: str, destination: Path) -> int:
        self.events.append("download")
        if self.fail_download:
            raise RuntimeError("sensitive endpoint detail")
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(bytes(16 * self.duration_ms * 2))
        return destination.stat().st_size

    def upload_artifact(self, audio_uri: str, path: Path) -> str:
        self.events.append("upload_diarization")
        assert path.read_bytes().endswith(b"\n")
        assert audio_uri.endswith("/audio.wav")
        return f"{audio_uri.removesuffix('audio.wav')}diarization.json"

    def reference_audio_uri(self, audio_uri: str, speaker_id: int) -> str:
        return (
            f"{audio_uri.removesuffix('audio.wav')}speaker-references/"
            f"speaker-{speaker_id}.wav"
        )

    def upload_reference_audio(
        self, audio_uri: str, speaker_id: int, path: Path
    ) -> str:
        self.events.append(f"upload_reference_{speaker_id}")
        assert path.stat().st_size > 44
        return self.reference_audio_uri(audio_uri, speaker_id)

    def upload_reference_manifest(self, _audio_uri: str, path: Path) -> str:
        self.events.append("upload_reference_manifest")
        if self.fail_reference_manifest:
            raise RuntimeError("sensitive storage detail")
        assert path.read_bytes().endswith(b"\n")
        self.reference_manifest = json.loads(path.read_bytes())
        return "s3://bucket/reference.json"


class Engine:
    def __init__(
        self, events: list[str], *, turns: tuple[RawTurn, ...] | None = None
    ) -> None:
        self.events = events
        self.turns = turns or (RawTurn(0.0, 1.0, "raw-label"),)

    def infer(self, _path: Path) -> InferenceResult:
        self.events.append("infer")
        return InferenceResult(
            turns=self.turns,
            device="cuda",
            accelerator="Test GPU",
            model_cache_hit=False,
        )


class Publisher:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def publish(self, _identifier: UUID) -> str:
        self.events.append("publish")
        if self.fail:
            raise RuntimeError("sensitive broker detail")
        return "message"


class CapturingLogger:
    def __init__(self, workspace_parent: Path) -> None:
        self.workspace_parent = workspace_parent
        self.calls: list[tuple[int, str, dict]] = []

    def log(self, level: int, message: str, *, extra: dict) -> None:
        assert list(self.workspace_parent.iterdir()) == []
        self.calls.append((level, message, dict(extra)))


def handler(
    tmp_path: Path,
    repository: Repository,
    storage: Storage,
    publisher: Publisher,
    logger: CapturingLogger,
    *,
    turns: tuple[RawTurn, ...] | None = None,
) -> DiarizeAudioPartHandler:
    return DiarizeAudioPartHandler(
        repository=repository,
        storage=storage,
        diarization=Engine(storage.events, turns=turns),
        publisher=publisher,
        diarization_policy=DiarizationPolicy(
            model="model",
            device="auto",
        ),
        speaker_reference_policy=SpeakerReferencePolicy(
            min_segment_ms=4000,
            edge_trim_ms=500,
            min_speaker_effective_ms=4000,
            max_speaker_effective_ms=30000,
            inter_segment_silence_ms=500,
        ),
        task_policy=TaskPolicy(error_max_length=512, workspace_prefix="owned-"),
        workspace_parent=tmp_path,
        terminal_logger=logger,
    )


def test_success_order_cleanup_and_terminal_summary(tmp_path: Path) -> None:
    events: list[str] = []
    repository = Repository()
    logger = CapturingLogger(tmp_path)
    result = handler(
        tmp_path,
        repository,
        Storage(events),
        Publisher(events),
        logger,
    )(str(IDENTIFIER))

    assert result == {
        "audio_part_id": str(IDENTIFIER),
        "status": "diarized",
        "quality_filter_dispatched": True,
        "speaker_count": 1,
        "reference_speaker_count": 0,
    }
    assert events == [
        "download",
        "infer",
        "upload_diarization",
        "upload_reference_manifest",
        "publish",
    ]
    assert [item[0] for item in repository.events] == ["claim", "complete"]
    assert len(logger.calls) == 1
    summary = logger.calls[0][2]
    assert summary["outcome"] == "succeeded"
    assert summary["device"] == "cuda"
    assert summary["speaker_count"] == 1
    assert summary["reference_speaker_count"] == 0
    assert summary["cleanup_succeeded"] is True
    assert summary["failure_stage"] is None


def test_download_failure_marks_failed_reraises_and_redacts_log(tmp_path: Path) -> None:
    events: list[str] = []
    repository = Repository()
    logger = CapturingLogger(tmp_path)
    with pytest.raises(TaskStageError) as raised:
        handler(
            tmp_path,
            repository,
            Storage(events, fail_download=True),
            Publisher(events),
            logger,
        )(str(IDENTIFIER))
    assert raised.value.code is ErrorCode.DOWNLOAD_FAILED
    assert [item[0] for item in repository.events] == ["claim", "processing_failed"]
    assert len(logger.calls) == 1
    assert "sensitive" not in repr(logger.calls[0][2])
    assert logger.calls[0][2]["elapsed_model_inference_ms"] is None


def test_qualifying_speaker_reference_is_uploaded_before_manifest_and_commit(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    repository = Repository(duration_ms=7000)
    storage = Storage(events, duration_ms=7000)
    logger = CapturingLogger(tmp_path)
    result = handler(
        tmp_path,
        repository,
        storage,
        Publisher(events),
        logger,
        turns=(RawTurn(0.0, 7.0, "speaker"),),
    )(str(IDENTIFIER))

    assert result["reference_speaker_count"] == 1
    assert events == [
        "download",
        "infer",
        "upload_diarization",
        "upload_reference_0",
        "upload_reference_manifest",
        "publish",
    ]
    assert [event[0] for event in repository.events] == ["claim", "complete"]
    speaker = storage.reference_manifest["speakers"][0]
    assert speaker["speaker_id"] == 0
    assert speaker["reference_audio"]["segments"] == [
        {"start_ms": 500, "end_ms": 6500, "duration_ms": 6000}
    ]
    assert speaker["reference_audio"]["effective_duration_ms"] == 6000


def test_reference_manifest_upload_failure_prevents_completion_and_dispatch(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    repository = Repository()
    logger = CapturingLogger(tmp_path)
    with pytest.raises(TaskStageError) as raised:
        handler(
            tmp_path,
            repository,
            Storage(events, fail_reference_manifest=True),
            Publisher(events),
            logger,
        )(str(IDENTIFIER))
    assert raised.value.code is ErrorCode.UPLOAD_FAILED
    assert [event[0] for event in repository.events] == [
        "claim",
        "processing_failed",
    ]
    assert "publish" not in events


def test_publish_failure_preserves_completed_artifact_state(tmp_path: Path) -> None:
    events: list[str] = []
    repository = Repository()
    logger = CapturingLogger(tmp_path)
    with pytest.raises(TaskStageError) as raised:
        handler(
            tmp_path,
            repository,
            Storage(events),
            Publisher(events, fail=True),
            logger,
        )(str(IDENTIFIER))
    assert raised.value.code is ErrorCode.DOWNSTREAM_DISPATCH_FAILED
    assert [item[0] for item in repository.events] == [
        "claim",
        "complete",
        "dispatch_failed",
    ]
    assert logger.calls[0][2]["final_status"] == "failed"


@pytest.mark.parametrize(
    ("disposition", "status", "published"),
    [
        (ClaimDisposition.ALREADY_PROCESSING, "already_processing", False),
        (ClaimDisposition.COMPLETED, "already_completed", False),
        (ClaimDisposition.DISPATCH_READY, "diarized", True),
    ],
)
def test_status_only_duplicate_and_recovery_paths(
    tmp_path: Path,
    disposition: ClaimDisposition,
    status: str,
    published: bool,
) -> None:
    events: list[str] = []
    repository = Repository(disposition)
    logger = CapturingLogger(tmp_path)
    result = handler(tmp_path, repository, Storage(events), Publisher(events), logger)(
        str(IDENTIFIER)
    )
    assert result["status"] == status
    assert ("publish" in events) is published
    assert len(logger.calls) == 1
    assert logger.calls[0][2]["elapsed_download_ms"] is None


def test_invalid_uuid_is_never_logged_raw(tmp_path: Path) -> None:
    events: list[str] = []
    logger = CapturingLogger(tmp_path)
    with pytest.raises(TaskStageError):
        handler(tmp_path, Repository(), Storage(events), Publisher(events), logger)(
            "secret-not-a-uuid"
        )
    assert logger.calls[0][2]["audio_part_id"] is None
    assert "secret-not-a-uuid" not in repr(logger.calls[0][2])


class RaisingLogger:
    def log(self, _level: int, _message: str, *, extra: dict) -> None:
        raise RuntimeError("logging backend unavailable")


def test_logging_backend_failure_does_not_change_result(tmp_path: Path) -> None:
    events: list[str] = []
    result = handler(
        tmp_path,
        Repository(ClaimDisposition.COMPLETED),
        Storage(events),
        Publisher(events),
        RaisingLogger(),
    )(str(IDENTIFIER))
    assert result["status"] == "already_completed"
