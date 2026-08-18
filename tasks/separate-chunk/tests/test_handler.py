import hashlib
import shutil
import wave
from dataclasses import replace
from uuid import UUID

import numpy as np
import pytest
from voice_pipeline_diarization_artifact import RawTurn, build_artifact

from voice_pipeline_separate_chunk import task as task_module
from voice_pipeline_separate_chunk.repository import Claim, Disposition
from voice_pipeline_separate_chunk.task import Handler

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")


def wav(path):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(np.zeros(20000 * 16, dtype="<i2").tobytes())


class Repo:
    def claim(self, identifier):
        return Claim(
            identifier,
            Disposition.CLAIMED,
            "separating",
            "s3://bucket/raw_audios/r/audio_parts/0/chunks/0/audio.wav",
            20000,
            0,
            20000,
            "en",
            20000,
            "s3://bucket/raw_audios/r/audio_parts/0/diarization.json",
        )

    def complete(self, *args):
        self.completed = args

    def reject(self, *args):
        self.rejected = args

    def fail(self, *args):
        self.failed = args


class Storage:
    def __init__(self, audio, diar):
        self.audio = audio
        self.diar = diar
        self.uploaded = []

    def download(self, uri, path, **_options):
        shutil.copyfile(self.diar if uri.endswith("json") else self.audio, path)

    def output_uris(self, _):
        return (
            "s3://bucket/raw_audios/r/audio_parts/0/chunks/0/results/separated/speaker-0.wav",
            "s3://bucket/raw_audios/r/audio_parts/0/chunks/0/results/separated/speaker-1.wav",
        )

    def upload(self, uri, path):
        self.uploaded.append((uri, path.stat().st_size))


class Model:
    def __init__(self):
        self.seeds = []

    def separate(self, samples, seed):
        self.seeds.append(seed)
        tracks = np.zeros((2, len(samples)), dtype=np.float32)
        tracks[0, : 9000 * 16] = 0.5
        tracks[1, 10000 * 16 : 19000 * 16] = 0.5
        return tracks, 16000


class Aligner:
    def align(self, *args):
        raise AssertionError("single window must not align")


class ClaimFailureRepo:
    def claim(self, _identifier):
        raise RuntimeError("contract drift details")

    def fail(self, *args):
        self.failed = args


class Publisher:
    def publish(self, identifier):
        self.published = identifier


class CompletedRepo:
    def __init__(self, separation):
        self.separation = separation

    def claim(self, identifier):
        return Claim(
            identifier,
            Disposition.ALREADY_SEPARATED,
            "separated",
            "s3://bucket/raw_audios/r/audio_parts/0/chunks/0/audio.wav",
            20000,
            0,
            20000,
            "en",
            diarizations={
                "schema_version": 1,
                "timebase": "chunk",
                "segments": [
                    {
                        "speaker": 0,
                        "start_ms": 0,
                        "end_ms": 9000,
                        "duration_ms": 9000,
                    },
                    {
                        "speaker": 1,
                        "start_ms": 10000,
                        "end_ms": 19000,
                        "duration_ms": 9000,
                    },
                ],
            },
            separation=self.separation,
        )


def completed_separation():
    uris = (
        "s3://bucket/raw_audios/r/audio_parts/0/chunks/0/results/separated/speaker-0.wav",
        "s3://bucket/raw_audios/r/audio_parts/0/chunks/0/results/separated/speaker-1.wav",
    )
    return {
        "schema_version": 1,
        "backend": "dialogue_sidon",
        "model": {
            "repo_id": "sarulab-speech/DialogueSidon",
            "revision": "a" * 40,
            "config_version": "sidon-v1",
            "inference_steps": 100,
        },
        "input_audio": {
            "sample_rate_hz": 16000,
            "duration_ms": 20000,
            "size_bytes": 1,
            "sha256": "b" * 64,
        },
        "speaker_audio": [
            {
                "output_slot": slot,
                "diarization_speaker_id": slot,
                "uri": uris[slot],
                "sample_rate_hz": 16000,
                "duration_ms": 20000,
                "size_bytes": 1,
                "sha256": chr(ord("c") + slot) * 64,
            }
            for slot in range(2)
        ],
        "audit": {
            "verdict": "ok",
            "reference_speaker_id": 0,
            "consistent_relation": "direct",
        },
    }


class NonEnglishRepo(Repo):
    def claim(self, identifier):
        return replace(super().claim(identifier), lang="zh")


def test_handler_separates_and_persists_mapping(tmp_path, policy):
    audio = tmp_path / "audio.wav"
    diar = tmp_path / "diarization.json"
    wav(audio)
    build_artifact(
        (RawTurn(0, 9, "a"), RawTurn(10, 19, "b")), model="fake", duration_ms=20000
    ).write(diar)
    repo = Repo()
    storage = Storage(audio, diar)
    model = Model()
    result = Handler(repo, storage, model, Aligner(), policy, tmp_path)(str(IDENTIFIER))
    assert result["outcome"] == "separated" and len(storage.uploaded) == 2
    separation = repo.completed[3]
    assert [item["diarization_speaker_id"] for item in separation["speaker_audio"]] == [
        0,
        1,
    ]
    assert model.seeds == [
        int.from_bytes(hashlib.shake_256(IDENTIFIER.bytes).digest(8), "big")
    ]


def test_english_chunk_dispatches_transcription_after_completion(tmp_path, policy):
    audio = tmp_path / "audio.wav"
    diar = tmp_path / "diarization.json"
    wav(audio)
    build_artifact(
        (RawTurn(0, 9, "a"), RawTurn(10, 19, "b")), model="fake", duration_ms=20000
    ).write(diar)
    repo = Repo()
    publisher = Publisher()
    result = Handler(
        repo,
        Storage(audio, diar),
        Model(),
        Aligner(),
        policy,
        tmp_path,
        publisher,
    )(str(IDENTIFIER))
    assert result["outcome"] == "separated"
    assert repo.completed
    assert publisher.published == IDENTIFIER


def test_non_english_chunk_is_not_dispatched(tmp_path, policy):
    audio = tmp_path / "audio.wav"
    diar = tmp_path / "diarization.json"
    wav(audio)
    build_artifact(
        (RawTurn(0, 9, "a"), RawTurn(10, 19, "b")), model="fake", duration_ms=20000
    ).write(diar)
    publisher = Publisher()
    result = Handler(
        NonEnglishRepo(),
        Storage(audio, diar),
        Model(),
        Aligner(),
        policy,
        tmp_path,
        publisher,
    )(str(IDENTIFIER))
    assert result["outcome"] == "separated"
    assert not hasattr(publisher, "published")


def test_claim_contract_failure_is_persisted_without_io(tmp_path, policy):
    repo = ClaimFailureRepo()
    with pytest.raises(RuntimeError, match="contract drift details"):
        Handler(repo, object(), object(), object(), policy, tmp_path)(str(IDENTIFIER))
    assert repo.failed[0] == IDENTIFIER
    assert repo.failed[1].startswith("separation_failed:")


def test_completed_separation_is_strictly_validated_without_io(tmp_path, policy):
    result = Handler(
        CompletedRepo(completed_separation()),
        Storage(tmp_path / "missing.wav", tmp_path / "missing.json"),
        object(),
        object(),
        policy,
        tmp_path,
    )(str(IDENTIFIER))
    assert result["outcome"] == "already_separated"


def test_partial_completed_separation_is_rejected(tmp_path, policy):
    with pytest.raises(TypeError, match="invalid_completed_separation"):
        Handler(
            CompletedRepo({}),
            Storage(tmp_path / "missing.wav", tmp_path / "missing.json"),
            object(),
            object(),
            policy,
            tmp_path,
        )(str(IDENTIFIER))


def test_workspace_cleanup_failure_does_not_replace_success(
    tmp_path, policy, monkeypatch
):
    audio = tmp_path / "audio.wav"
    diar = tmp_path / "diarization.json"
    wav(audio)
    build_artifact(
        (RawTurn(0, 9, "a"), RawTurn(10, 19, "b")), model="fake", duration_ms=20000
    ).write(diar)
    workspace_type = task_module.Workspace

    class CleanupFailureWorkspace(workspace_type):
        def close(self):
            raise OSError("cleanup failed")

    monkeypatch.setattr(task_module, "Workspace", CleanupFailureWorkspace)
    result = Handler(
        Repo(), Storage(audio, diar), Model(), Aligner(), policy, tmp_path
    )(str(IDENTIFIER))
    assert result["outcome"] == "separated"
