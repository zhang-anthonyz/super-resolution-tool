"""Application root resolution for source and PyInstaller frozen builds."""
from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    """Return the directory that contains app resources and vendor binaries.

    In a PyInstaller one-dir build, ``sys.executable`` points to the packaged
    executable. In a source checkout, this module is located at
    ``<root>/app/core/paths.py``, so two parents up is the repository root.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]
