"""Task-scoped workspace and bounded orphan cleanup."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    root: Path
    audio: Path
    diarization: Path
    chunks: Path


class TaskWorkspace:
    def __init__(self, *, prefix: str, parent: Path | None = None) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=parent)
        root = Path(self._temporary.name)
        chunks = root / "chunks"
        chunks.mkdir()
        self.paths = WorkspacePaths(root, root / "audio.wav", root / "diarization.json", chunks)

    def chunk_path(self, chunk_index: int) -> Path:
        return self.paths.chunks / f"{chunk_index}.wav"

    def close(self) -> None:
        self._temporary.cleanup()


def cleanup_orphaned_workspaces(*, prefix: str, parent: Path | None = None) -> int:
    root = Path(tempfile.gettempdir()) if parent is None else parent
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("Temporary workspace root is invalid.")
    removed = 0
    for entry in root.iterdir():
        if not entry.name.startswith(prefix) or entry.is_symlink() or not entry.is_dir():
            continue
        shutil.rmtree(entry)
        removed += 1
    return removed
