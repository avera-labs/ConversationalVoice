"""Automatic execution-target discovery."""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionTarget:
    architecture: str
    device: str
    accelerator: str


def detect_execution_target() -> ExecutionTarget:
    architecture = platform.machine().lower()
    aliases = {"amd64": "x86_64", "arm64": "aarch64"}
    architecture = aliases.get(architecture, architecture)
    if architecture not in {"x86_64", "aarch64"}:
        raise RuntimeError("machine architecture is unsupported")
    try:
        import tensorflow as tf

        devices = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        raise RuntimeError("TensorFlow execution target discovery failed") from exc
    if devices:
        return ExecutionTarget(architecture, "gpu", devices[0].name)
    return ExecutionTarget(architecture, "cpu", platform.processor() or architecture)
