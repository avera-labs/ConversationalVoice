import pytest

from voice_pipeline_persona_chunk.storage import ObjectStorage


def test_storage_derives_canonical_result_uris():
    storage = ObjectStorage(object(), "bucket")
    audio = "s3://bucket/raw_audios/r/audio_parts/0/chunks/3/audio.wav"
    assert storage.persona_uri(audio).endswith("/chunks/3/results/persona.json")
    assert storage.transcription_uris(audio) == (
        "s3://bucket/raw_audios/r/audio_parts/0/chunks/3/results/transcript.json",
        "s3://bucket/raw_audios/r/audio_parts/0/chunks/3/results/word_alignment.json",
    )


@pytest.mark.parametrize(
    "uri",
    [
        "https://bucket/key",
        "s3://other/key",
        "s3://bucket/a/../key",
        "s3://bucket/a%2Fkey",
        "s3://bucket//key",
    ],
)
def test_storage_rejects_noncanonical_or_cross_bucket_uris(uri):
    with pytest.raises(ValueError, match="invalid S3 URI"):
        ObjectStorage(object(), "bucket").persona_uri(uri)
