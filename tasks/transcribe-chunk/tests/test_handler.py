import hashlib
import shutil
import wave
from dataclasses import replace
from uuid import UUID

import numpy as np

from voice_pipeline_transcribe_chunk import task as task_module
from voice_pipeline_transcribe_chunk.repository import Claim, Disposition
from voice_pipeline_transcribe_chunk.task import Handler
from voice_pipeline_transcribe_chunk.utterances import DecodedWord

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")
BASE = "s3://bucket/raw_audios/r/audio_parts/0/chunks/0"


def make_wav(path, duration_ms=4000):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(np.zeros(duration_ms * 16, dtype="<i2").tobytes())
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def snapshot():
    return {
        "schema_version": 1,
        "timebase": "chunk",
        "segments": [
            {"speaker": 4, "start_ms": 100, "end_ms": 1000, "duration_ms": 900},
            {"speaker": 7, "start_ms": 2000, "end_ms": 3000, "duration_ms": 1000},
        ],
    }


def separation(size, sha):
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
            "duration_ms": 4000,
            "size_bytes": 1,
            "sha256": "b" * 64,
        },
        "speaker_audio": [
            {
                "output_slot": slot,
                "diarization_speaker_id": speaker,
                "uri": f"{BASE}/results/separated/speaker-{slot}.wav",
                "sample_rate_hz": 16000,
                "duration_ms": 4000,
                "size_bytes": size,
                "sha256": sha,
            }
            for slot, speaker in enumerate((4, 7))
        ],
        "audit": {
            "verdict": "ok",
            "reference_speaker_id": 4,
            "consistent_relation": "direct",
        },
    }


class Repo:
    def __init__(self, size, sha):
        self.claim_value = Claim(
            IDENTIFIER,
            Disposition.CLAIMED,
            "transcribing",
            UUID("22222222-2222-2222-2222-222222222222"),
            f"{BASE}/audio.wav",
            "en",
            4000,
            0,
            4000,
            snapshot(),
            separation(size, sha),
        )

    def claim(self, _identifier):
        return self.claim_value

    def complete(self, claim, result):
        self.completed = (claim, result)

    def fail(self, *args):
        self.failed = args


class Storage:
    def __init__(self, source):
        self.source = source
        self.uploads = []

    def speaker_uris(self, _audio_uri):
        return tuple(
            f"{BASE}/results/separated/speaker-{slot}.wav" for slot in range(2)
        )

    def artifact_uris(self, _audio_uri):
        return (
            f"{BASE}/results/transcript.json",
            f"{BASE}/results/word_alignment.json",
        )

    def download(self, _uri, path):
        shutil.copyfile(self.source, path)

    def upload_json(self, uri, path):
        self.uploads.append((uri, path.read_bytes()))


class Model:
    def __init__(self, empty=False):
        self.empty = empty
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(len(audio))
        return [] if self.empty else [DecodedWord("Hello.", 0.1, 0.4, 0.9)]


class Publisher:
    def publish(self, identifier):
        self.published = identifier


class FailingPublisher:
    def publish(self, _identifier):
        raise RuntimeError("publish failed")


def test_handler_transcribes_both_mapped_tracks(tmp_path, policy):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    repo = Repo(size, sha)
    storage = Storage(source)
    model = Model()
    result = Handler(repo, storage, model, policy, tmp_path)(str(IDENTIFIER))
    assert result["outcome"] == "transcribed"
    assert result["word_count"] == 2
    assert len(storage.uploads) == 2
    persisted = repo.completed[1]
    assert [
        item["diarization_speaker_id"] for item in persisted["input_speaker_audio"]
    ] == [4, 7]


def test_handler_publishes_persona_only_after_completion(tmp_path, policy):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    repo = Repo(size, sha)
    publisher = Publisher()
    Handler(repo, Storage(source), Model(), policy, tmp_path, publisher=publisher)(
        str(IDENTIFIER)
    )
    assert repo.completed
    assert publisher.published == IDENTIFIER


def test_publish_failure_is_raised_after_transcription_completion(tmp_path, policy):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    repo = Repo(size, sha)

    import pytest

    with pytest.raises(RuntimeError, match="publish failed"):
        Handler(
            repo,
            Storage(source),
            Model(),
            policy,
            tmp_path,
            publisher=FailingPublisher(),
        )(str(IDENTIFIER))
    assert repo.completed


def test_empty_model_output_is_successful(tmp_path, policy):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    repo = Repo(size, sha)
    result = Handler(repo, Storage(source), Model(empty=True), policy, tmp_path)(
        str(IDENTIFIER)
    )
    assert result["outcome"] == "transcribed"
    assert result["word_count"] == 0
    assert not hasattr(repo, "failed")


def test_completed_result_is_validated_without_io_or_model(tmp_path, policy):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    repo = Repo(size, sha)
    storage = Storage(source)
    Handler(repo, storage, Model(empty=True), policy, tmp_path)(str(IDENTIFIER))
    completed = repo.completed[1]
    repo.claim_value = replace(
        repo.claim_value,
        disposition=Disposition.ALREADY_TRANSCRIBED,
        status="transcribed",
        transcription=completed,
    )
    storage.uploads.clear()
    model = Model()
    result = Handler(repo, storage, model, policy, tmp_path)(str(IDENTIFIER))
    assert result["outcome"] == "already_transcribed"
    assert storage.uploads == []
    assert model.calls == []


def test_non_english_fails_before_download_or_model(tmp_path, policy):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    repo = Repo(size, sha)
    repo.claim_value = replace(repo.claim_value, lang="zh")
    storage = Storage(source)
    model = Model()
    import pytest

    with pytest.raises(RuntimeError, match="unsupported_chunk_language"):
        Handler(repo, storage, model, policy, tmp_path)(str(IDENTIFIER))
    assert storage.uploads == []
    assert model.calls == []
    assert repo.failed[0] == IDENTIFIER


def test_workspace_cleanup_failure_does_not_replace_success(
    tmp_path, policy, monkeypatch
):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    workspace_type = task_module.Workspace

    class CleanupFailureWorkspace(workspace_type):
        def close(self):
            raise OSError("cleanup failed")

    monkeypatch.setattr(task_module, "Workspace", CleanupFailureWorkspace)
    result = Handler(Repo(size, sha), Storage(source), Model(), policy, tmp_path)(
        str(IDENTIFIER)
    )
    assert result["outcome"] == "transcribed"
