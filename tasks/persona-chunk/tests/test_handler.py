import hashlib
import json
import shutil
import wave
from dataclasses import replace
from uuid import UUID

from voice_pipeline_persona_chunk import task as task_module
from voice_pipeline_persona_chunk.repository import Claim, Disposition
from voice_pipeline_persona_chunk.task import Handler

IDENTIFIER = UUID("11111111-1111-1111-1111-111111111111")
PART = UUID("22222222-2222-2222-2222-222222222222")
BASE = "s3://bucket/raw_audios/r/audio_parts/0/chunks/0"


def make_wav(path, duration_ms=1000):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0\0" * (duration_ms * 16))
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def snapshot():
    return {
        "schema_version": 1,
        "timebase": "chunk",
        "segments": [
            {"speaker": 4, "start_ms": 0, "end_ms": 500, "duration_ms": 500},
            {"speaker": 7, "start_ms": 500, "end_ms": 1000, "duration_ms": 500},
        ],
    }


def separation(audio_size, audio_sha):
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
            "duration_ms": 1000,
            "size_bytes": audio_size,
            "sha256": audio_sha,
        },
        "speaker_audio": [
            {
                "output_slot": slot,
                "diarization_speaker_id": speaker_id,
                "uri": f"{BASE}/results/separated/speaker-{slot}.wav",
                "sample_rate_hz": 16000,
                "duration_ms": 1000,
                "size_bytes": 10 + slot,
                "sha256": str(slot + 1) * 64,
            }
            for slot, speaker_id in enumerate((4, 7))
        ],
        "audit": {
            "verdict": "ok",
            "reference_speaker_id": 4,
            "consistent_relation": "direct",
        },
    }


def transcript():
    return {
        "schema_version": 1,
        "backend": "parakeet_tdt",
        "model": {
            "repo_id": "nvidia/parakeet-tdt-0.6b-v3",
            "revision": "b" * 40,
            "config_version": "parakeet-v1",
        },
        "language": "en",
        "timebase": "chunk",
        "speakers": [
            {
                "output_slot": 0,
                "diarization_speaker_id": 4,
                "utterances": [
                    {
                        "utterance_index": 0,
                        "start_ms": 0,
                        "end_ms": 400,
                        "text": "Hello",
                        "confidence": 0.9,
                    }
                ],
            },
            {
                "output_slot": 1,
                "diarization_speaker_id": 7,
                "utterances": [
                    {
                        "utterance_index": 0,
                        "start_ms": 500,
                        "end_ms": 900,
                        "text": "Hi",
                        "confidence": 0.8,
                    }
                ],
            },
        ],
    }


def transcription_result(sep, transcript_bytes):
    transcript_sha = hashlib.sha256(transcript_bytes).hexdigest()
    return {
        "schema_version": 1,
        "backend": "parakeet_tdt",
        "model": {
            "repo_id": "nvidia/parakeet-tdt-0.6b-v3",
            "revision": "b" * 40,
            "config_version": "parakeet-v1",
        },
        "language": "en",
        "input_speaker_audio": [
            {
                "output_slot": item["output_slot"],
                "diarization_speaker_id": item["diarization_speaker_id"],
                "uri": item["uri"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in sep["speaker_audio"]
        ],
        "artifacts": {
            "transcript": {
                "uri": f"{BASE}/results/transcript.json",
                "size_bytes": len(transcript_bytes),
                "sha256": transcript_sha,
            },
            "word_alignment": {
                "uri": f"{BASE}/results/word_alignment.json",
                "size_bytes": 1,
                "sha256": "c" * 64,
            },
        },
    }


class Repo:
    def __init__(self, audio_size, audio_sha, transcript_bytes):
        sep = separation(audio_size, audio_sha)
        self.claim_value = Claim(
            IDENTIFIER,
            Disposition.CLAIMED,
            "persona_generating",
            PART,
            f"{BASE}/audio.wav",
            "en",
            1000,
            0,
            1000,
            snapshot(),
            sep,
            transcription_result(sep, transcript_bytes),
        )

    def claim(self, _identifier):
        return self.claim_value

    def complete(self, claim, persona, result):
        self.completed = claim, persona, result

    def fail(self, *args):
        self.failed = args

    def fail_publication(self, *args):
        self.publication_failed = args


class Storage:
    def __init__(self, audio, transcript_bytes):
        self.audio = audio
        self.transcript_bytes = transcript_bytes
        self.downloaded = []
        self.uploads = []

    def speaker_uris(self, _audio_uri):
        return tuple(
            f"{BASE}/results/separated/speaker-{slot}.wav" for slot in range(2)
        )

    def transcription_uris(self, _audio_uri):
        return f"{BASE}/results/transcript.json", f"{BASE}/results/word_alignment.json"

    def persona_uri(self, _audio_uri):
        return f"{BASE}/results/persona.json"

    def download(self, uri, destination):
        self.downloaded.append(uri)
        if uri.endswith("audio.wav"):
            shutil.copyfile(self.audio, destination)
        elif uri.endswith("transcript.json"):
            destination.write_bytes(self.transcript_bytes)
        else:
            raise AssertionError(f"unexpected download: {uri}")

    def upload_json(self, uri, source):
        self.uploads.append((uri, source.read_bytes()))


class Client:
    def __init__(self):
        self.calls = []

    def analyze(self, mp3, srt, mapping):
        self.calls.append((mp3, srt, mapping))
        speaker = {
            "name": None,
            "age": None,
            "ethnicity": None,
            "gender": None,
            "tag": "You enjoy having a good conversation.",
            "alpha": "low",
            "evidence": None,
            "primary_emotion": "neutral",
            "secondary_emotion": None,
            "emotion_intensity": "low",
            "laugh": False,
            "cry": False,
            "whisper": False,
            "shout": False,
            "sigh": False,
            "overall_tone": "calm",
        }
        return (
            {
                "scene": {
                    "description": "A calm exchange.",
                    "overall_tone": "calm",
                    "emotion_intensity": "low",
                },
                "speakers": {"4": dict(speaker), "7": dict(speaker)},
            },
            {
                "model": "xiaomi/mimo-v2.5",
                "in_tokens": 1,
                "out_tokens": 2,
                "total_tokens": 3,
                "cost_usd": 0.001,
            },
        )


class Publisher:
    def __init__(self):
        self.calls = []

    def publish(self, identifier):
        self.calls.append(identifier)
        return "message-id"


class FailingPublisher:
    def publish(self, _identifier):
        raise RuntimeError("publish failed")


def test_handler_completes_without_downloading_word_alignment(
    tmp_path, policy, monkeypatch
):
    audio = tmp_path / "source.wav"
    size, sha = make_wav(audio)
    transcript_bytes = canonical(transcript())
    repo = Repo(size, sha, transcript_bytes)
    storage = Storage(audio, transcript_bytes)
    client = Client()

    def fake_encode(_source, destination, _policy):
        destination.write_bytes(b"mp3")
        return 3

    def fail_schema_validation(*_args, **_kwargs):
        raise AssertionError("model response must not be schema-validated")

    monkeypatch.setattr(task_module, "encode_mp3", fake_encode)
    monkeypatch.setattr(
        task_module,
        "parse_persona_document",
        fail_schema_validation,
    )
    publisher = Publisher()
    result = Handler(repo, storage, client, publisher, policy, tmp_path)(
        str(IDENTIFIER)
    )
    assert result == {
        "chunk_id": str(IDENTIFIER),
        "outcome": "persona_generated",
        "speaker_count": 2,
    }
    assert all(not uri.endswith("word_alignment.json") for uri in storage.downloaded)
    assert client.calls[0][2] == (4, 7)
    assert "[Speaker 4]: Hello" in client.calls[0][1]
    assert repo.completed[1]["speaker_mapping"][1]["diarization_speaker_id"] == 7
    assert repo.completed[2]["artifact"]["uri"].endswith("persona.json")
    assert publisher.calls == [IDENTIFIER]
    assert not hasattr(repo, "failed")


def test_completed_claim_is_validated_without_storage_or_provider_io(
    tmp_path, policy, monkeypatch
):
    audio = tmp_path / "source.wav"
    size, sha = make_wav(audio)
    transcript_bytes = canonical(transcript())
    repo = Repo(size, sha, transcript_bytes)
    storage = Storage(audio, transcript_bytes)
    client = Client()
    monkeypatch.setattr(
        task_module, "encode_mp3", lambda _s, d, _p: d.write_bytes(b"mp3")
    )
    Handler(repo, storage, client, Publisher(), policy, tmp_path)(str(IDENTIFIER))
    _, persona, result = repo.completed
    repo.claim_value = replace(
        repo.claim_value,
        disposition=Disposition.ALREADY_COMPLETED,
        status="completed",
        persona=persona,
        persona_result=result,
    )
    storage.downloaded.clear()
    storage.uploads.clear()
    client.calls.clear()
    outcome = Handler(repo, storage, client, Publisher(), policy, tmp_path)(
        str(IDENTIFIER)
    )
    assert outcome["outcome"] == "already_completed"
    assert storage.downloaded == []
    assert storage.uploads == []
    assert client.calls == []


def test_durable_persona_retry_only_republishes_successor(
    tmp_path, policy, monkeypatch
):
    audio = tmp_path / "source.wav"
    size, sha = make_wav(audio)
    transcript_bytes = canonical(transcript())
    repo = Repo(size, sha, transcript_bytes)
    storage = Storage(audio, transcript_bytes)
    monkeypatch.setattr(
        task_module, "encode_mp3", lambda _s, d, _p: d.write_bytes(b"mp3")
    )
    Handler(repo, storage, Client(), Publisher(), policy, tmp_path)(str(IDENTIFIER))
    _, persona, result = repo.completed
    repo.claim_value = replace(
        repo.claim_value,
        disposition=Disposition.READY_TO_DISPATCH,
        status="failed",
        persona=persona,
        persona_result=result,
    )
    storage.downloaded.clear()
    storage.uploads.clear()
    client = Client()
    publisher = Publisher()

    outcome = Handler(repo, storage, client, publisher, policy, tmp_path)(
        str(IDENTIFIER)
    )

    assert outcome["outcome"] == "persona_generated"
    assert publisher.calls == [IDENTIFIER]
    assert storage.downloaded == []
    assert client.calls == []


def test_publication_failure_marks_durable_persona_failed(
    tmp_path, policy, monkeypatch
):
    audio = tmp_path / "source.wav"
    size, sha = make_wav(audio)
    transcript_bytes = canonical(transcript())
    repo = Repo(size, sha, transcript_bytes)
    monkeypatch.setattr(
        task_module, "encode_mp3", lambda _s, d, _p: d.write_bytes(b"mp3")
    )

    import pytest

    with pytest.raises(RuntimeError, match="publish failed"):
        Handler(
            repo,
            Storage(audio, transcript_bytes),
            Client(),
            FailingPublisher(),
            policy,
            tmp_path,
        )(str(IDENTIFIER))
    assert repo.completed
    assert repo.publication_failed[0] == IDENTIFIER


def test_completed_claim_uses_persisted_model_after_configuration_change(
    tmp_path, policy, monkeypatch
):
    audio = tmp_path / "source.wav"
    size, sha = make_wav(audio)
    transcript_bytes = canonical(transcript())
    repo = Repo(size, sha, transcript_bytes)
    storage = Storage(audio, transcript_bytes)
    monkeypatch.setattr(
        task_module, "encode_mp3", lambda _s, d, _p: d.write_bytes(b"mp3")
    )
    Handler(repo, storage, Client(), Publisher(), policy, tmp_path)(str(IDENTIFIER))
    _, persona, result = repo.completed
    persona["usage"]["model"] = "example/previous-model"
    result["model"]["id"] = "example/previous-model"
    persona_bytes = canonical(persona)
    result["artifact"]["size_bytes"] = len(persona_bytes)
    result["artifact"]["sha256"] = hashlib.sha256(persona_bytes).hexdigest()
    repo.claim_value = replace(
        repo.claim_value,
        disposition=Disposition.ALREADY_COMPLETED,
        status="completed",
        persona=persona,
        persona_result=result,
    )
    outcome = Handler(repo, storage, Client(), Publisher(), policy, tmp_path)(
        str(IDENTIFIER)
    )
    assert outcome["outcome"] == "already_completed"


def test_completed_claim_rejects_persona_artifact_hash_drift(
    tmp_path, policy, monkeypatch
):
    audio = tmp_path / "source.wav"
    size, sha = make_wav(audio)
    transcript_bytes = canonical(transcript())
    repo = Repo(size, sha, transcript_bytes)
    storage = Storage(audio, transcript_bytes)
    monkeypatch.setattr(
        task_module, "encode_mp3", lambda _s, d, _p: d.write_bytes(b"mp3")
    )
    Handler(repo, storage, Client(), Publisher(), policy, tmp_path)(str(IDENTIFIER))
    _, persona, result = repo.completed
    result["artifact"]["sha256"] = "f" * 64
    repo.claim_value = replace(
        repo.claim_value,
        disposition=Disposition.ALREADY_COMPLETED,
        status="completed",
        persona=persona,
        persona_result=result,
    )
    import pytest

    with pytest.raises(ValueError, match="artifact identity"):
        Handler(repo, storage, Client(), Publisher(), policy, tmp_path)(str(IDENTIFIER))


def test_transcript_identity_mismatch_marks_claimed_row_failed(
    tmp_path, policy, monkeypatch
):
    audio = tmp_path / "source.wav"
    size, sha = make_wav(audio)
    transcript_bytes = canonical(transcript())
    repo = Repo(size, sha, transcript_bytes)
    storage = Storage(audio, transcript_bytes + b" ")
    client = Client()
    monkeypatch.setattr(
        task_module, "encode_mp3", lambda _s, d, _p: d.write_bytes(b"mp3")
    )
    import pytest

    with pytest.raises(RuntimeError, match="input_transcript_identity_mismatch"):
        Handler(repo, storage, client, Publisher(), policy, tmp_path)(str(IDENTIFIER))
    assert repo.failed[0] == IDENTIFIER
    assert client.calls == []


def test_workspace_creation_failure_marks_claimed_row_failed(
    tmp_path, policy, monkeypatch
):
    audio = tmp_path / "source.wav"
    size, sha = make_wav(audio)
    transcript_bytes = canonical(transcript())
    repo = Repo(size, sha, transcript_bytes)

    def fail_workspace(*_args, **_kwargs):
        raise OSError("workspace unavailable")

    monkeypatch.setattr(task_module, "Workspace", fail_workspace)
    import pytest

    with pytest.raises(OSError, match="workspace unavailable"):
        Handler(
            repo,
            Storage(audio, transcript_bytes),
            Client(),
            Publisher(),
            policy,
            tmp_path,
        )(str(IDENTIFIER))
    assert repo.failed[0] == IDENTIFIER
