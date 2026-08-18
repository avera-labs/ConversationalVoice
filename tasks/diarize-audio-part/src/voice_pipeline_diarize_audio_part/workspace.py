"""Task-scoped temporary workspaces and restart-time orphan cleanup."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    root: Path
    audio_path: Path
    artifact_path: Path
    speaker_references: Path
    speaker_reference_manifest_path: Path


class TaskWorkspace:
    def __init__(self, *, prefix: str, parent: Path | None = None) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=parent)
        root = Path(self._temporary.name)
        speaker_references = root / "speaker-references"
        speaker_references.mkdir()
        self.paths = WorkspacePaths(
            root,
            root / "audio.wav",
            root / "diarization.json",
            speaker_references,
            speaker_references / "references.json",
        )

    def speaker_reference_audio_path(self, speaker_id: int) -> Path:
        if isinstance(speaker_id, bool) or speaker_id < 0:
            raise ValueError("speaker ID must be a non-negative integer")
        return self.paths.speaker_references / f"speaker-{speaker_id}.wav"

    def __enter__(self) -> WorkspacePaths:
        return self.paths

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        self._temporary.cleanup()


def cleanup_orphaned_workspaces(*, prefix: str, parent: Path | None = None) -> int:
    root = Path(tempfile.gettempdir()) if parent is None else parent
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("Temporary workspace root is invalid.")
    removed = 0
    for entry in root.iterdir():
        if (
            not entry.name.startswith(prefix)
            or entry.is_symlink()
            or not entry.is_dir()
        ):
            continue
        shutil.rmtree(entry)
        removed += 1
    return removed
