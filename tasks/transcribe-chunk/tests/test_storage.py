from voice_pipeline_transcribe_chunk.storage import ObjectStorage


class Client:
    def upload_file(self, *args, **kwargs):
        self.upload = (args, kwargs)


def test_storage_derives_canonical_keys(tmp_path):
    storage = ObjectStorage(Client(), "bucket")
    audio = "s3://bucket/raw_audios/r/audio_parts/0/chunks/2/audio.wav"
    assert storage.speaker_uris(audio) == (
        "s3://bucket/raw_audios/r/audio_parts/0/chunks/2/results/separated/speaker-0.wav",
        "s3://bucket/raw_audios/r/audio_parts/0/chunks/2/results/separated/speaker-1.wav",
    )
    assert storage.artifact_uris(audio) == (
        "s3://bucket/raw_audios/r/audio_parts/0/chunks/2/results/transcript.json",
        "s3://bucket/raw_audios/r/audio_parts/0/chunks/2/results/word_alignment.json",
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    storage.upload_json(storage.artifact_uris(audio)[0], artifact)
    assert storage.client.upload[1]["ExtraArgs"] == {"ContentType": "application/json"}
