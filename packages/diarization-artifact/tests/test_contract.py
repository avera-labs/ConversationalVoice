import json

import pytest

from voice_pipeline_diarization_artifact import (
    DiarizationArtifactError,
    RawTurn,
    build_artifact,
    parse_artifact_bytes,
)


def test_writer_and_reader_share_version_one_contract() -> None:
    artifact = build_artifact(
        [RawTurn(0.1, 1.2349, "a"), RawTurn(1.5, 2.5, "b")],
        model="model-name",
        duration_ms=3000,
    )
    parsed = parse_artifact_bytes(artifact.to_json_bytes(), expected_duration_ms=3000)
    assert parsed.model == "model-name"
    assert [(turn.speaker, turn.start_ms, turn.end_ms) for turn in parsed.turns] == [
        (0, 100, 1235),
        (1, 1500, 2500),
    ]


def test_empty_artifact_is_valid() -> None:
    artifact = build_artifact([], model="model-name", duration_ms=1000)
    parsed = parse_artifact_bytes(artifact.to_json_bytes(), expected_duration_ms=1000)
    assert parsed.turns == ()


@pytest.mark.parametrize("mutation", ["extra", "duration", "precision", "speaker"])
def test_reader_rejects_contract_drift(mutation: str) -> None:
    document = build_artifact(
        [RawTurn(0.0, 1.0, "a")], model="model-name", duration_ms=1000
    ).to_dict()
    if mutation == "extra":
        document["unexpected"] = True
    elif mutation == "duration":
        document["segments"][0]["duration"] = 0.5
    elif mutation == "precision":
        document["segments"][0]["start"] = 0.0001
    else:
        document["segments"][0]["speaker"] = 2
    with pytest.raises(DiarizationArtifactError):
        parse_artifact_bytes(
            json.dumps(document).encode("utf-8"), expected_duration_ms=1000
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "version",
        "database_duration",
        "bounds",
        "ordering",
        "num_speakers",
        "speech_total",
        "summary_total",
        "summary_percentage",
        "not_finite",
    ],
)
def test_reader_rejects_invalid_metadata_and_ordering(mutation: str) -> None:
    document = build_artifact(
        [RawTurn(0.0, 1.0, "a"), RawTurn(1.0, 2.0, "b")],
        model="model-name",
        duration_ms=2000,
    ).to_dict()
    expected_duration_ms = 2000
    if mutation == "missing":
        del document["model"]
    elif mutation == "version":
        document["schema_version"] = 2
    elif mutation == "database_duration":
        expected_duration_ms = 2001
    elif mutation == "bounds":
        document["segments"][1]["end"] = 2.001
    elif mutation == "ordering":
        document["segments"].reverse()
    elif mutation == "num_speakers":
        document["num_speakers"] = 1
    elif mutation == "speech_total":
        document["total_speech_seconds"] = 1.5
    elif mutation == "summary_total":
        document["speaker_summary"][0]["total_seconds"] = 0.5
    elif mutation == "summary_percentage":
        document["speaker_summary"][0]["percentage"] = 49.9
    else:
        document["segments"][0]["start"] = float("nan")
    with pytest.raises(DiarizationArtifactError):
        parse_artifact_bytes(
            json.dumps(document).encode("utf-8"),
            expected_duration_ms=expected_duration_ms,
        )
