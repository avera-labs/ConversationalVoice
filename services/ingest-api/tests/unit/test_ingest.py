from __future__ import annotations

import hashlib
import io

import pytest

from voice_pipeline_ingest_api.services.ingest import (
    UPLOAD_COPY_CHUNK_BYTES,
    EmptyUploadError,
    UploadTooLargeError,
    stream_copy_and_sha1,
)


class TrackingStream(io.BytesIO):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.requested_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.requested_sizes.append(size)
        return super().read(size)


def test_stream_copy_hashes_original_bytes_in_fixed_chunks(tmp_path) -> None:
    content = b"original-audio" * 100_000
    source = TrackingStream(content)
    destination = tmp_path / "original.upload"

    result = stream_copy_and_sha1(
        source,
        destination,
        max_bytes=len(content),
    )

    assert destination.read_bytes() == content
    assert (
        result.content_sha1
        == hashlib.sha1(
            content,
            usedforsecurity=False,
        ).hexdigest()
    )
    assert result.size_bytes == len(content)
    assert set(source.requested_sizes) == {UPLOAD_COPY_CHUNK_BYTES}


def test_stream_copy_rejects_oversized_actual_content_and_removes_partial_file(
    tmp_path,
) -> None:
    destination = tmp_path / "original.upload"

    with pytest.raises(UploadTooLargeError):
        stream_copy_and_sha1(
            io.BytesIO(b"123456"),
            destination,
            max_bytes=5,
        )

    assert not destination.exists()


def test_stream_copy_rejects_empty_content_and_removes_file(tmp_path) -> None:
    destination = tmp_path / "original.upload"

    with pytest.raises(EmptyUploadError):
        stream_copy_and_sha1(
            io.BytesIO(b""),
            destination,
            max_bytes=100,
        )

    assert not destination.exists()
