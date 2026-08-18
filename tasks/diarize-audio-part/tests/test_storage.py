from pathlib import Path

import pytest

from voice_pipeline_diarize_audio_part.storage import (
    InvalidS3UriError,
    ObjectStorage,
    parse_s3_uri,
)


@pytest.mark.parametrize(
    "uri",
    [
        "https://bucket/key",
        "s3://other/key",
        "s3://bucket//key",
        "s3://bucket/a/../key",
        "s3://bucket/key?token=value",
        "s3://bucket/key%20name",
    ],
)
def test_rejects_noncanonical_or_cross_bucket_uri(uri: str) -> None:
    with pytest.raises(InvalidS3UriError):
        parse_s3_uri(uri, expected_bucket="bucket")


def test_download_and_deterministic_upload(tmp_path: Path) -> None:
    calls: list[tuple] = []

    class Client:
        def download_file(self, bucket: str, key: str, path: str) -> None:
            calls.append(("download", bucket, key))
            Path(path).write_bytes(b"wav")

        def upload_file(
            self, path: str, bucket: str, key: str, ExtraArgs: dict
        ) -> None:
            calls.append(("upload", bucket, key, ExtraArgs))

    storage = ObjectStorage(Client(), "bucket")
    audio = tmp_path / "audio.wav"
    assert storage.download_audio("s3://bucket/input/audio.wav", audio) == 3
    artifact = tmp_path / "diarization.json"
    artifact.write_text("{}\n", encoding="utf-8")
    audio_uri = (
        "s3://bucket/raw_audios/11111111-1111-1111-1111-111111111111/"
        "audio_parts/0/audio.wav"
    )
    uri = storage.upload_artifact(audio_uri, artifact)
    assert (
        uri == "s3://bucket/raw_audios/11111111-1111-1111-1111-111111111111/"
        "audio_parts/0/diarization.json"
    )
    assert calls[-1][-1] == {"ContentType": "application/json"}

    reference_audio = tmp_path / "speaker-3.wav"
    reference_audio.write_bytes(b"reference")
    assert storage.upload_reference_audio(audio_uri, 3, reference_audio).endswith(
        "/speaker-references/speaker-3.wav"
    )
    assert calls[-1][-1] == {"ContentType": "audio/wav"}

    manifest = tmp_path / "references.json"
    manifest.write_text('{"schema_version":1,"speakers":[]}\n', encoding="utf-8")
    assert storage.upload_reference_manifest(audio_uri, manifest).endswith(
        "/speaker-references/references.json"
    )
    assert calls[-1][-1] == {"ContentType": "application/json"}


def test_reference_keys_and_uri_are_deterministic() -> None:
    storage = ObjectStorage(object(), "bucket")
    audio_uri = "s3://bucket/raw_audios/id/audio_parts/2/audio.wav"
    assert storage.reference_manifest_key(audio_uri) == (
        "raw_audios/id/audio_parts/2/speaker-references/references.json"
    )
    assert storage.reference_audio_key(audio_uri, 7) == (
        "raw_audios/id/audio_parts/2/speaker-references/speaker-7.wav"
    )
    assert storage.reference_audio_uri(audio_uri, 7) == (
        "s3://bucket/raw_audios/id/audio_parts/2/speaker-references/speaker-7.wav"
    )


@pytest.mark.parametrize("speaker_id", [-1, True, 1.5])
def test_reference_audio_key_rejects_invalid_speaker_id(speaker_id: object) -> None:
    storage = ObjectStorage(object(), "bucket")
    with pytest.raises(ValueError):
        storage.reference_audio_key("s3://bucket/input/audio.wav", speaker_id)  # type: ignore[arg-type]


def test_artifact_key_rejects_cross_bucket_audio_uri() -> None:
    storage = ObjectStorage(object(), "bucket")
    with pytest.raises(InvalidS3UriError):
        storage.artifact_key("s3://other/raw_audios/id/audio_parts/0/audio.wav")
