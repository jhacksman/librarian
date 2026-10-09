"""Resolve persistent index storage independently of the source corpus."""

import os
import sys
from pathlib import Path


def index_path(index: Path | None = None, data_dir: Path | None = None) -> Path:
    """Explicit index, explicit data directory, environment, then user-local default."""
    if index is not None:
        return index.expanduser().resolve()
    if data_dir is None:
        configured = os.environ.get("LIBRARIAN_DATA_DIR")
        if configured:
            data_dir = Path(configured)
        elif sys.platform == "darwin":
            data_dir = Path.home() / "Library" / "Application Support" / "Librarian"
        elif sys.platform == "win32":
            data_dir = (
                Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
                / "Librarian"
            )
        else:
            data_dir = (
                Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
                / "librarian"
            )
    return data_dir.expanduser().resolve() / "library.sqlite"
