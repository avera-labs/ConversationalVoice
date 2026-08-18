import shutil
import wave
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from voice_pipeline_diarization_artifact import RawTurn, build_artifact

from voice_pipeline_quality_filter_audio_part.errors import ErrorCode, TaskStageError
from voice_pipeline_quality_filter_audio_part.repository import (
    AudioPartClaim,
    ClaimDisposition,
)
from voice_pipeline_quality_filter_audio_part.task import QualityFilterAudioPartHandler
from voice_pipeline_quality_filter_audio_part.workspace import TaskWorkspace

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")
AUDIO_URI = "s3://bucket/raw_audios/raw/audio_parts/0/audio.wav"
DIARIZATION_URI = "s3://bucket/raw_audios/raw/audio_parts/0/diarization.json"


def write_wav(path: Path, duration_ms: int) -> None:
    samples = (np.sin(np.arange(duration_ms * 16) / 20) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.tobytes())


class Repository:
    def __init__(self, disposition=ClaimDisposition.CLAIMED, *, fail_complete=False) -> None:
        self.disposition = disposition
        self.fail_complete = fail_complete
        self.events = []
        self.persisted = ()

    def claim(self, identifier):
        self.events.append(("claim", identifier))
        if self.disposition is ClaimDisposition.CLAIMED:
            return AudioPartClaim(
                identifier,
                self.disposition,
                "filtering",
                AUDIO_URI,
                DIARIZATION_URI,
                46000,
                "en",
            )
        return AudioPartClaim(identifier, self.disposition, self.disposition.value)

    def complete(self, claim, chunks):
        self.events.append(("complete", claim.audio_part_id))
        if self.fail_complete:
            raise RuntimeError("database details")
        self.persisted = chunks
        return tuple(chunk.chunk_id for chunk in chunks)

    def mark_failed(self, identifier, error):
        self.events.append(("failed", identifier, error))


class Storage:
    def __init__(
        self,
        source_audio: Path,
        source_diarization: Path,
        *,
        fail=False,
        fail_upload=False,
    ) -> None:
        self.source_audio = source_audio
        self.source_diarization = source_diarization
        self.fail = fail
        self.fail_upload = fail_upload
        self.uploads = []

    def download_audio(self, _uri, destination):
        if self.fail:
            raise RuntimeError("private endpoint detail")
        shutil.copyfile(self.source_audio, destination)

    def download_diarization(self, _uri, destination):
        shutil.copyfile(self.source_diarization, destination)

    def upload_chunk(self, _audio_uri, index, path):
        if self.fail_upload:
            raise RuntimeError("storage details")
        assert path.stat().st_size > 0
        self.uploads.append(index)
        return f"s3://bucket/chunks/{index}/audio.wav"


class Detector:
    model_name = "fake-music-model"

    def detect(self, _waveform, *, sample_rate, duration_ms):
        assert sample_rate == 16000
        assert duration_ms == 46000
        return ()


class FailingDetector(Detector):
    def detect(self, _waveform, *, sample_rate, duration_ms):
        raise RuntimeError("model details")


class FailingCleanupWorkspace(TaskWorkspace):
    def close(self) -> None:
        super().close()
        raise RuntimeError("cleanup details")


class CapturingLogger:
    def __init__(self) -> None:
        self.events = []

    def info(self, message, *, extra):
        self.events.append((message, dict(extra)))


def create_inputs(tmp_path: Path) -> tuple[Path, Path]:
    audio = tmp_path / "source.wav"
    diarization = tmp_path / "diarization.json"
    write_wav(audio, 46000)
    build_artifact(
        [
            RawTurn(0, 10, "a"),
            RawTurn(11, 21, "b"),
            RawTurn(22, 24, "a"),
            RawTurn(25, 35, "a"),
            RawTurn(36, 46, "b"),
        ],
        model="fake-diarization",
        duration_ms=46000,
    ).write(diarization)
    return audio, diarization


def make_handler(
    tmp_path,
    repository,
    storage,
    logger,
    quality_policy,
    planner_policy,
    task_policy,
):
    return QualityFilterAudioPartHandler(
        repository=repository,
        storage=storage,
        music_detector=Detector(),
        quality_policy=quality_policy,
        planner_policy=planner_policy,
        task_policy=task_policy,
        workspace_parent=tmp_path / "workspaces",
        timing_logger=logger,
    )


def test_complete_pipeline_uploads_and_persists_one_chunk(
    tmp_path,
    monkeypatch,
    quality_policy,
    planner_policy,
    task_policy,
) -> None:
    audio, diarization = create_inputs(tmp_path)
    workspace_parent = tmp_path / "workspaces"
    workspace_parent.mkdir()
    repository = Repository()
    storage = Storage(audio, diarization)
    logger = CapturingLogger()
    monkeypatch.setattr(
        "voice_pipeline_quality_filter_audio_part.task.wada_snr", lambda _samples: 20.0
    )
    result = make_handler(
        tmp_path,
        repository,
        storage,
        logger,
        quality_policy,
        planner_policy,
        task_policy,
    )(str(IDENTIFIER))
    assert result == {
        "audio_part_id": str(IDENTIFIER),
        "outcome": "completed",
        "created_count": 1,
    }
    assert storage.uploads == [0]
    assert repository.persisted[0].start_ms == 0
    assert repository.persisted[0].end_ms == 46000
    assert list(workspace_parent.iterdir()) == []
    assert logger.events
    assert all(
        {"audio_part_id", "model_name", "stage", "elapsed_ms"} <= set(extra)
        for _, extra in logger.events
    )
    assert all(message == "quality_filter_audio_part.timing" for message, _ in logger.events)
    chunk_events = [
        extra
        for _, extra in logger.events
        if extra["stage"] in {"cut_chunk_audio", "upload_chunk_audio"}
    ]
    assert len(chunk_events) == 2
    assert all("chunk_id" in extra for extra in chunk_events)


def test_download_failure_marks_failed_and_reraises(
    tmp_path,
    quality_policy,
    planner_policy,
    task_policy,
) -> None:
    audio, diarization = create_inputs(tmp_path)
    (tmp_path / "workspaces").mkdir()
    repository = Repository()
    logger = CapturingLogger()
    with pytest.raises(TaskStageError) as raised:
        make_handler(
            tmp_path,
            repository,
            Storage(audio, diarization, fail=True),
            logger,
            quality_policy,
            planner_policy,
            task_policy,
        )(str(IDENTIFIER))
    assert raised.value.code is ErrorCode.DOWNLOAD_FAILED
    assert [event[0] for event in repository.events] == ["claim", "failed"]
    assert "private endpoint" not in repr(logger.events)


@pytest.mark.parametrize(
    "disposition",
    [
        ClaimDisposition.ALREADY_PROCESSING,
        ClaimDisposition.ALREADY_COMPLETED,
        ClaimDisposition.NOT_READY,
    ],
)
def test_no_op_does_not_log_or_run_models(
    tmp_path,
    disposition,
    quality_policy,
    planner_policy,
    task_policy,
) -> None:
    audio, diarization = create_inputs(tmp_path)
    (tmp_path / "workspaces").mkdir()
    logger = CapturingLogger()
    result = make_handler(
        tmp_path,
        Repository(disposition),
        Storage(audio, diarization),
        logger,
        quality_policy,
        planner_policy,
        task_policy,
    )(str(IDENTIFIER))
    assert result["outcome"] == disposition.value
    assert logger.events == []


def test_invalid_uuid_is_not_logged(
    tmp_path,
    quality_policy,
    planner_policy,
    task_policy,
) -> None:
    audio, diarization = create_inputs(tmp_path)
    (tmp_path / "workspaces").mkdir()
    logger = CapturingLogger()
    with pytest.raises(TaskStageError):
        make_handler(
            tmp_path,
            Repository(),
            Storage(audio, diarization),
            logger,
            quality_policy,
            planner_policy,
            task_policy,
        )("secret-invalid-id")
    assert logger.events == []


@pytest.mark.parametrize(
    ("failure_point", "expected_code"),
    [
        ("audio", ErrorCode.INVALID_AUDIO),
        ("diarization", ErrorCode.INVALID_DIARIZATION),
        ("music", ErrorCode.MUSIC_DETECTION_FAILED),
        ("snr", ErrorCode.SNR_FAILED),
        ("planning", ErrorCode.PLANNING_FAILED),
        ("cut", ErrorCode.CUT_FAILED),
        ("upload", ErrorCode.UPLOAD_FAILED),
        ("persistence", ErrorCode.PERSISTENCE_FAILED),
        ("cleanup", ErrorCode.CLEANUP_FAILED),
    ],
)
def test_each_claimed_failure_stage_marks_failed_and_reraises(
    tmp_path,
    monkeypatch,
    quality_policy,
    planner_policy,
    task_policy,
    failure_point,
    expected_code,
) -> None:
    audio, diarization = create_inputs(tmp_path)
    (tmp_path / "workspaces").mkdir()
    repository = Repository(fail_complete=failure_point == "persistence")
    storage = Storage(audio, diarization, fail_upload=failure_point == "upload")
    detector = Detector()
    workspace_factory = TaskWorkspace
    if failure_point == "audio":
        audio.write_bytes(b"invalid")
    elif failure_point == "diarization":
        diarization.write_bytes(b"invalid")
    elif failure_point == "music":
        detector = FailingDetector()
    elif failure_point == "snr":
        monkeypatch.setattr(
            "voice_pipeline_quality_filter_audio_part.task.wada_snr",
            lambda _samples: (_ for _ in ()).throw(RuntimeError("snr details")),
        )
    else:
        monkeypatch.setattr(
            "voice_pipeline_quality_filter_audio_part.task.wada_snr", lambda _samples: 20.0
        )
    if failure_point == "planning":
        monkeypatch.setattr(
            "voice_pipeline_quality_filter_audio_part.task.plan_chunks",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("planning details")),
        )
    if failure_point == "cut":
        monkeypatch.setattr(
            "voice_pipeline_quality_filter_audio_part.task.write_chunk_wav",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cut details")),
        )
    if failure_point == "cleanup":
        workspace_factory = FailingCleanupWorkspace
    handler = QualityFilterAudioPartHandler(
        repository=repository,
        storage=storage,
        music_detector=detector,
        quality_policy=quality_policy,
        planner_policy=planner_policy,
        task_policy=task_policy,
        workspace_parent=tmp_path / "workspaces",
        workspace_factory=workspace_factory,
        timing_logger=CapturingLogger(),
    )
    with pytest.raises(TaskStageError) as raised:
        handler(str(IDENTIFIER))
    assert raised.value.code is expected_code
    assert repository.events[-1][0] == "failed"
    if failure_point in {"cut", "upload", "cleanup"}:
        assert not any(event[0] == "complete" for event in repository.events)
    assert list((tmp_path / "workspaces").iterdir()) == []


def test_empty_diarization_completes_with_zero_chunks(
    tmp_path,
    quality_policy,
    planner_policy,
    task_policy,
) -> None:
    audio = tmp_path / "source.wav"
    diarization = tmp_path / "diarization.json"
    write_wav(audio, 46000)
    build_artifact([], model="fake-diarization", duration_ms=46000).write(diarization)
    (tmp_path / "workspaces").mkdir()
    repository = Repository()
    result = make_handler(
        tmp_path,
        repository,
        Storage(audio, diarization),
        CapturingLogger(),
        quality_policy,
        planner_policy,
        task_policy,
    )(str(IDENTIFIER))
    assert result["created_count"] == 0
    assert repository.persisted == ()
