import pytest

from voice_pipeline_diarize_audio_part.certification import (
    validate_certification_target,
)


@pytest.mark.parametrize(
    ("target", "architecture", "accelerator"),
    [
        ("a100", "x86_64", "NVIDIA A100-SXM4-80GB"),
        ("gh200", "aarch64", "NVIDIA GH200 480GB"),
    ],
)
def test_required_platforms_are_accepted(
    target: str, architecture: str, accelerator: str
) -> None:
    validate_certification_target(
        target=target,
        architecture=architecture,
        cuda_available=True,
        accelerator=accelerator,
    )


@pytest.mark.parametrize(
    ("target", "architecture", "cuda_available", "accelerator"),
    [
        ("a100", "x86_64", True, "NVIDIA A10G"),
        ("a100", "aarch64", True, "NVIDIA A100"),
        ("gh200", "aarch64", False, "NVIDIA GH200"),
    ],
)
def test_wrong_hardware_cannot_produce_certification(
    target: str,
    architecture: str,
    cuda_available: bool,
    accelerator: str,
) -> None:
    with pytest.raises(RuntimeError):
        validate_certification_target(
            target=target,
            architecture=architecture,
            cuda_available=cuda_available,
            accelerator=accelerator,
        )
