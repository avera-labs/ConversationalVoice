import hashlib
import json
import shutil
import wave
from dataclasses import replace
from uuid import UUID

import numpy as np
import pytest

from voice_pipeline_transcribe_chunk_zh.alignment import DecodedUnit
from voice_pipeline_transcribe_chunk_zh.repository import Claim, Disposition
from voice_pipeline_transcribe_chunk_zh.task import Handler

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
            "zh",
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
        if self.empty:
            return []
        return [
            DecodedUnit("你", 0.1, 0.2, 0.9),
            DecodedUnit("好", 0.2, 0.4, 0.8),
        ]


class Punctuation:
    def restore(self, text):
        self.inputs = getattr(self, "inputs", []) + [text]
        return text + "。"


class Publisher:
    def __init__(self, repository):
        self.repository = repository

    def publish(self, identifier):
        assert hasattr(self.repository, "completed")
        self.published = identifier


@pytest.mark.parametrize("language", ["zh", "zh-CN", "zh-Hant-TW"])
def test_handler_preserves_chinese_language_tag_in_artifacts(
    tmp_path, policy, language
):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    repo = Repo(size, sha)
    repo.claim_value = replace(repo.claim_value, lang=language)
    storage = Storage(source)
    punctuation = Punctuation()
    publisher = Publisher(repo)
    result = Handler(
        repo,
        storage,
        Model(),
        punctuation,
        policy,
        tmp_path,
        publisher,
    )(str(IDENTIFIER))
    assert result["outcome"] == "transcribed"
    assert result["word_count"] == 4
    assert len(storage.uploads) == 2
    persisted = repo.completed[1]
    assert persisted["backend"] == "paraformer_zh"
    assert persisted["language"] == language
    assert publisher.published == IDENTIFIER
    assert punctuation.inputs == ["你好", "你好"]
    transcript = json.loads(storage.uploads[0][1])
    alignment = json.loads(storage.uploads[1][1])
    assert transcript["language"] == alignment["language"] == language
    assert [item["text"] for item in alignment["speakers"][0]["words"]] == [
        "你",
        "好。",
    ]
    assert transcript["speakers"][0]["utterances"] == [
        {
            "utterance_index": 0,
            "start_ms": 100,
            "end_ms": 400,
            "text": "你好。",
            "confidence": 0.85,
        }
    ]


def test_empty_tracks_complete_without_loading_punctuation(tmp_path, policy):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    repo = Repo(size, sha)
    punctuation = Punctuation()
    result = Handler(
        repo, Storage(source), Model(empty=True), punctuation, policy, tmp_path
    )(str(IDENTIFIER))
    assert result["outcome"] == "transcribed"
    assert result["word_count"] == 0
    assert not hasattr(punctuation, "inputs")


def test_non_zh_chunk_fails_before_audio_or_model(tmp_path, policy):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    repo = Repo(size, sha)
    repo.claim_value = replace(repo.claim_value, lang="en")
    storage = Storage(source)
    model = Model()
    with pytest.raises(RuntimeError, match="unsupported_chunk_language"):
        Handler(repo, storage, model, Punctuation(), policy, tmp_path)(str(IDENTIFIER))
    assert storage.uploads == []
    assert model.calls == []
    assert repo.failed[0] == IDENTIFIER


def test_completed_result_is_validated_without_model_io(tmp_path, policy):
    source = tmp_path / "source.wav"
    size, sha = make_wav(source)
    repo = Repo(size, sha)
    storage = Storage(source)
    Handler(repo, storage, Model(), Punctuation(), policy, tmp_path)(str(IDENTIFIER))
    completed = repo.completed[1]
    repo.claim_value = replace(
        repo.claim_value,
        disposition=Disposition.ALREADY_TRANSCRIBED,
        status="transcribed",
        transcription=completed,
    )
    storage.uploads.clear()
    model = Model()
    result = Handler(repo, storage, model, Punctuation(), policy, tmp_path)(
        str(IDENTIFIER)
    )
    assert result["outcome"] == "already_transcribed"
    assert storage.uploads == [] and model.calls == []
