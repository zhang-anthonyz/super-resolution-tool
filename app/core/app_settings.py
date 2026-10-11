"""Small JSON settings store for desktop UI preferences."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def user_data_dir() -> Path:
    """Return a per-user writable directory for application state."""
    override = os.environ.get("SRT_SETTINGS_DIR")
    if override:
        return Path(override)
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "SuperResolutionTool"
    return Path.home() / ".super-resolution-tool"


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


def load_settings() -> dict[str, Any]:
    path = settings_path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_settings(settings: dict[str, Any]) -> Path:
    path = settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
    except OSError:
        pass
    return path
