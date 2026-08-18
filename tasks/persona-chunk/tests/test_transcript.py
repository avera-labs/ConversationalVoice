from voice_pipeline_persona_chunk.transcript import transcript_to_srt


def test_srt_merges_speakers_with_stable_order():
    document = {
        "speakers": [
            {
                "output_slot": 0,
                "diarization_speaker_id": 7,
                "utterances": [
                    {
                        "utterance_index": 0,
                        "start_ms": 1000,
                        "end_ms": 1500,
                        "text": "Later",
                    }
                ],
            },
            {
                "output_slot": 1,
                "diarization_speaker_id": 4,
                "utterances": [
                    {
                        "utterance_index": 0,
                        "start_ms": 10,
                        "end_ms": 900,
                        "text": "First",
                    }
                ],
            },
        ]
    }
    result = transcript_to_srt(document)
    assert result.startswith("1\n00:00:00,010 --> 00:00:00,900\n[Speaker 4]: First")
    assert "[Speaker 7]: Later" in result
