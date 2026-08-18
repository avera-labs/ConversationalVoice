from pathlib import Path

import pytest

from voice_pipeline_separate_chunk.storage import ObjectStorage


class Client:
    def __init__(self, payload: bytes = b"audio") -> None:
        self.payload = payload

    def download_file(self, _bucket: str, _key: str, destination: str) -> None:
        Path(destination).write_bytes(self.payload)


@pytest.mark.parametrize(
    "uri",
    (
        "https://bucket/key",
        "s3://other/key",
        "s3://bucket/a/../b",
        "s3://bucket/a%2Fb",
        "s3://bucket/a\\b",
        "s3://bucket/key?x=1",
    ),
)
def test_rejects_noncanonical_uri(uri: str) -> None:
    with pytest.raises(ValueError, match="invalid S3 URI"):
        ObjectStorage(Client(), "bucket")._key(uri)


def test_oversized_download_is_removed(tmp_path: Path) -> None:
    destination = tmp_path / "artifact"
    with pytest.raises(OSError, match="object size is invalid"):
        ObjectStorage(Client(b"12345"), "bucket").download(
            "s3://bucket/key", destination, maximum_bytes=4
        )
    assert not destination.exists()
