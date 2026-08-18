from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator


@dataclass(frozen=True, slots=True)
class TaskWorkspace:
    """Paths owned by one task attempt and removed when the attempt exits."""

    root: Path
    raw_audio_path: Path
    vad_segments_path: Path
    audio_parts_directory: Path

    def audio_part_path(self, part_index: int) -> Path:
        if part_index < 0:
            raise ValueError("part_index must not be negative")
        directory = self.audio_parts_directory / str(part_index)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "audio.wav"


@contextmanager
def task_workspace(parent: Path | None = None) -> Iterator[TaskWorkspace]:
    """Create and always remove a task-local workspace."""

    parent_directory = None if parent is None else str(parent)
    with TemporaryDirectory(
        prefix="split-raw-audio-",
        dir=parent_directory,
    ) as temporary_directory:
        root = Path(temporary_directory)
        audio_parts_directory = root / "audio_parts"
        audio_parts_directory.mkdir()
        yield TaskWorkspace(
            root=root,
            raw_audio_path=root / "raw_audio.wav",
            vad_segments_path=root / "vad_segments.json",
            audio_parts_directory=audio_parts_directory,
        )
