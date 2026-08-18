import json
import math

import pytest

from voice_pipeline_diarize_audio_part.artifact import RawTurn, build_artifact


def test_artifact_is_sorted_mapped_and_stable() -> None:
    artifact = build_artifact(
        [
            RawTurn(2.0, 3.0, "z"),
            RawTurn(0.1, 1.2349, "a"),
            RawTurn(0.5, 2.5, "z"),
        ],
        model="model-name",
        duration_ms=4000,
    )

    assert [(item.speaker, item.start, item.end) for item in artifact.segments] == [
        (0, 0.1, 1.235),
        (1, 0.5, 2.5),
        (1, 2.0, 3.0),
    ]
    assert artifact.num_speakers == 2
    assert artifact.total_speech_seconds == 4.135
    assert [item.speaker for item in artifact.speaker_summary] == [0, 1]
    payload = artifact.to_json_bytes()
    assert payload.endswith(b"\n")
    assert list(json.loads(payload)) == [
        "schema_version",
        "model",
        "audio_duration_seconds",
        "num_speakers",
        "total_speech_seconds",
        "segments",
        "speaker_summary",
    ]
    assert b"speaker_label" not in payload


def test_empty_annotation_is_valid() -> None:
    artifact = build_artifact([], model="model-name", duration_ms=1000)
    assert artifact.num_speakers == 0
    assert artifact.segments == ()
    assert artifact.speaker_summary == ()


@pytest.mark.parametrize(
    "turn",
    [
        RawTurn(-1.0, 1.0, "a"),
        RawTurn(1.0, 1.0, "a"),
        RawTurn(0.0, 2.0, "a"),
        RawTurn(math.nan, 1.0, "a"),
        RawTurn(0.0, math.inf, "a"),
        RawTurn(0.0, 0.0001, "a"),
        RawTurn(0.0, 1.0, ""),
    ],
)
def test_invalid_turn_is_rejected(turn: RawTurn) -> None:
    with pytest.raises(ValueError):
        build_artifact([turn], model="model-name", duration_ms=1000)
