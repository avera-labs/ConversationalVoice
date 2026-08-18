from pathlib import Path

import pytest

from voice_pipeline_quality_filter_audio_part.storage import InvalidS3UriError, ObjectStorage, parse_s3_uri


class Client:
    def __init__(self) -> None:
        self.uploads = []

    def upload_file(self, path, bucket, key, ExtraArgs):
        self.uploads.append((path, bucket, key, ExtraArgs))


@pytest.mark.parametrize(
    "uri",
    ["https://bucket/key", "s3://other/key", "s3://bucket/a/../b", "s3://bucket/key?x=1"],
)
def test_invalid_uri_is_rejected(uri: str) -> None:
    with pytest.raises(InvalidS3UriError):
        parse_s3_uri(uri, expected_bucket="bucket")


def test_chunk_key_and_upload_are_deterministic(tmp_path: Path) -> None:
    client = Client()
    storage = ObjectStorage(client, "bucket", max_diarization_bytes=1000)
    source = "s3://bucket/raw_audios/id/audio_parts/2/audio.wav"
    assert storage.chunk_key(source, 3) == "raw_audios/id/audio_parts/2/chunks/3/audio.wav"
    chunk = tmp_path / "chunk.wav"
    chunk.write_bytes(b"wav")
    assert storage.upload_chunk(source, 3, chunk) == (
        "s3://bucket/raw_audios/id/audio_parts/2/chunks/3/audio.wav"
    )
    assert client.uploads[0][3] == {"ContentType": "audio/wav"}
