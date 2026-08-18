from voice_pipeline_extend_chunk.storage import ObjectStorage


class Client:
    pass


def test_deterministic_input_and_output_uris():
    storage = ObjectStorage(Client(), "bucket")
    part = "s3://bucket/raw_audios/r/audio_parts/2/audio.wav"
    chunk = f"{part.removesuffix('audio.wav')}chunks/3/audio.wav"
    assert storage.reference_manifest_uri(part).endswith(
        "/audio_parts/2/speaker-references/references.json"
    )
    assert storage.reference_audio_uri(part, 7).endswith(
        "/speaker-references/speaker-7.wav"
    )
    outputs = storage.output_uris(chunk)
    assert outputs["script"].endswith("/dialogue-extension/script.json")
    assert outputs["speaker_audio"] == (
        outputs["script"].replace("script.json", "speaker-0.wav"),
        outputs["script"].replace("script.json", "speaker-1.wav"),
    )
