import shutil
import tempfile
from pathlib import Path


class Workspace:
    def __init__(self, prefix, parent=None):
        self.temp = tempfile.TemporaryDirectory(prefix=prefix, dir=parent)
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio.wav"
        self.diarization = self.root / "diarization.json"
        self.outputs = (self.root / "speaker-0.wav", self.root / "speaker-1.wav")

    def close(self):
        self.temp.cleanup()


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
