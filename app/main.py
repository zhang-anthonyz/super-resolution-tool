"""Application entry point for the PySide6 desktop interface."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication, QMessageBox

from app.core.environment_checker import EnvironmentCheckError, check_environment
from app.core.types import GpuInfo
from app.ui.main_window import MainWindow


def create_main_window(gpu: Optional[GpuInfo] = None) -> MainWindow:
    """Create the main window using a supplied GPU or an environment check."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setStyle("Fusion")
    if gpu is None:
        gpu = check_environment()
    return MainWindow(gpu)


def main(argv: Optional[list[str]] = None) -> int:
    app = QApplication.instance() or QApplication(list(argv or sys.argv))
    app.setApplicationName("Super Resolution Tool")
    app.setStyle("Fusion")

    try:
        gpu = check_environment()
    except EnvironmentCheckError as exc:
        QMessageBox.critical(
            None,
            "环境检查失败",
            f"未检测到独立显卡\n\n{exc.message}",
        )
        return 1

    window = MainWindow(gpu)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
