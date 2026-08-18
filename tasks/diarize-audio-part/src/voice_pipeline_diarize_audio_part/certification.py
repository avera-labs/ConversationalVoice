"""Release-certification platform identity checks."""

from __future__ import annotations

CERTIFICATION_TARGETS = {
    "a100": ("x86_64", "A100"),
    "gh200": ("aarch64", "GH200"),
}


def validate_certification_target(
    *,
    target: str,
    architecture: str,
    cuda_available: bool,
    accelerator: str,
) -> None:
    """Reject reports produced on hardware other than the declared target."""
    expected = CERTIFICATION_TARGETS.get(target)
    if expected is None:
        raise ValueError("certification target must be a100 or gh200")
    expected_architecture, accelerator_marker = expected
    if architecture.lower() != expected_architecture:
        raise RuntimeError("certification host architecture does not match the target")
    if not cuda_available:
        raise RuntimeError("certification requires CUDA")
    if accelerator_marker not in accelerator:
        raise RuntimeError("certification accelerator does not match the target")
