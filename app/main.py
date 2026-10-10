"""Application entry point for the PySide6 desktop interface."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication, QMessageBox

from app.core.environment_checker import EnvironmentCheckError, check_environment
from app.core.paths import app_root
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


def _run_headless_smoke() -> int:
    """Exercise model, DirectML, and vendored FFmpeg without opening the UI."""
    import tempfile
    import traceback

    from PIL import Image

    from app.models.model_manager import load_model, scene_default
    from app.pipelines.ffmpeg_utils import run_ffmpeg
    from app.pipelines.image_pipeline import upscale_single
    from app.core.types import Scene

    try:
        ffmpeg_path = app_root() / "vendor" / "ffmpeg.exe"
        if not ffmpeg_path.is_file():
            raise FileNotFoundError(f"Packaged FFmpeg is missing: {ffmpeg_path}")
        default_model = scene_default(Scene.REALISTIC)
        model_path = app_root() / default_model.model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Packaged model is missing: {model_path}")

        process = run_ffmpeg(["ffmpeg", "-version"])
        _, stderr = process.communicate(timeout=30)
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Packaged FFmpeg failed: {detail}")

        with tempfile.TemporaryDirectory(prefix="sr_smoke_") as temp_dir:
            root = Path(temp_dir)
            source = root / "input.png"
            output = root / "output_x4.png"
            Image.new("RGB", (32, 24), (80, 120, 160)).save(source)

            engine = load_model(default_model)
            result = upscale_single(source, output, engine, scale=4)
            if not result.ok or not output.is_file():
                raise RuntimeError(f"Headless upscale failed: {result.error}")

            with Image.open(output) as image:
                if image.size != (128, 96):
                    raise RuntimeError(f"Unexpected smoke output size: {image.size}")

        return 0
    except Exception as exc:
        traceback.print_exc()
        print(f"HEADLESS_SMOKE_FAILED: {exc}", file=sys.stderr)
        return 1


def main(argv: Optional[list[str]] = None) -> int:
    if "--headless" in (argv or sys.argv):
        return _run_headless_smoke()

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
