"""Task logging: console output plus an optional GUI sink.

Processing modules log every step through :class:`TaskLog`. Each record goes to
the standard ``logging`` tree — where :func:`configure_logging` installs a
timestamped console handler (and a rotating file handler when a writable log
directory exists) — *and* to an optional sink callback. The Qt workers wire
that sink to the run-log panel, so the GUI shows exactly the same step trace
that the console prints instead of a coarser progress-only summary.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Callable, Optional

#: Namespace that owns every log line this application produces.
LOGGER_NAMESPACE = "app"

#: Console/log-file format: time, level, module, message.
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

#: Third-party loggers that are chatty at INFO/DEBUG and add nothing useful.
_QUIET_LOGGERS = (
    "PIL",
    "PIL.Image",
    "matplotlib",
    "urllib3",
)

_LEVEL_BY_NAME = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "success": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

#: UI log panel levels (see ``app.ui.theme.LOG_LEVEL_COLORS``).
UI_LEVEL_BY_NAME = {
    "debug": "Info",
    "info": "Info",
    "warning": "Warning",
    "error": "Error",
    "success": "Success",
}

_configured = False


def _resolve_level(level: Optional[str]) -> int:
    if level is None:
        level = os.environ.get("SRT_LOG_LEVEL", "")
    name = str(level).strip().lower()
    if name in _LEVEL_BY_NAME:
        return _LEVEL_BY_NAME[name]
    if name.isdigit():
        return int(name)
    return logging.INFO


def log_directory() -> Optional[str]:
    """Return the per-user directory that receives ``*.log`` files, if any.

    Packaged builds run windowed (no console), so a file log is the only trace
    available after the fact. Anything unwritable degrades to console-only.
    """
    try:
        from .app_settings import user_data_dir

        directory = user_data_dir() / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory)
    except Exception:
        return None


def configure_logging(
    level: Optional[str] = None,
    *,
    log_dir: Optional[str] = None,
    force: bool = False,
) -> None:
    """Install console (and file) handlers on the application log namespace.

    Idempotent: calling it again only updates the level, so the GUI can raise
    verbosity at runtime without stacking duplicate handlers.
    """
    global _configured

    numeric_level = _resolve_level(level)
    namespace = logging.getLogger(LOGGER_NAMESPACE)
    namespace.setLevel(numeric_level)
    # Keep the log namespace in charge of its own records (the root logger may
    # be owned by an embedding host, e.g. a frozen GUI shell).
    namespace.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if not _configured or force:
        for handler in list(namespace.handlers):
            namespace.removeHandler(handler)
            handler.close()

        stream = sys.stdout
        if stream is not None:
            # Chinese step messages must not crash the worker on a cp936
            # console, and a frozen windowed build has no stream at all.
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except (OSError, ValueError):
                    pass
            console = logging.StreamHandler(stream)
            console.setFormatter(formatter)
            namespace.addHandler(console)

        directory = log_dir if log_dir is not None else log_directory()
        if directory:
            try:
                file_handler = logging.handlers.RotatingFileHandler(
                    os.path.join(str(directory), "super-resolution-tool.log"),
                    maxBytes=4 * 1024 * 1024,
                    backupCount=3,
                    encoding="utf-8",
                )
                file_handler.setFormatter(formatter)
                namespace.addHandler(file_handler)
            except Exception:  # noqa: BLE001 - the file log is best-effort
                pass

        if not namespace.handlers:  # keep records from reaching a host's root
            namespace.addHandler(logging.NullHandler())

        for name in _QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        _configured = True

    for handler in namespace.handlers:
        handler.setLevel(numeric_level)


def format_bytes(num_bytes: float) -> str:
    """Human-readable size, e.g. ``12.3 MB``."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} GB"


def format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. ``0.42s`` or ``1m 05s``."""
    value = max(0.0, float(seconds))
    if value < 60.0:
        return f"{value:.2f}s"
    minutes, remainder = divmod(value, 60.0)
    return f"{int(minutes)}m {remainder:04.1f}s"


class TaskLog:
    """Step logger for one processing task.

    ``sink`` receives ``(ui_level, message)`` tuples; the GUI workers pass a
    Qt signal so the run-log panel mirrors the console trace.
    """

    def __init__(
        self,
        logger_name: str,
        sink: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._logger = logging.getLogger(logger_name)
        self._sink = sink

    @property
    def sink(self) -> Optional[Callable[[str, str], None]]:
        return self._sink

    def bind(self, sink: Optional[Callable[[str, str], None]]) -> "TaskLog":
        """Return a copy of this logger that additionally reports to ``sink``."""
        return TaskLog(self._logger.name, sink)

    def _emit(self, level: str, message: str, *args) -> str:
        text = message % args if args else message
        self._logger.log(_LEVEL_BY_NAME[level], "%s", text)
        if self._sink is not None and level != "debug":
            self._sink(UI_LEVEL_BY_NAME[level], text)
        return text

    def debug(self, message: str, *args) -> None:
        self._emit("debug", message, *args)

    def info(self, message: str, *args) -> None:
        self._emit("info", message, *args)

    def success(self, message: str, *args) -> None:
        self._emit("success", message, *args)

    def warning(self, message: str, *args) -> None:
        self._emit("warning", message, *args)

    def error(self, message: str, *args) -> None:
        self._emit("error", message, *args)
