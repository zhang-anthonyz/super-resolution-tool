import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from app.core.types import GpuInfo, GpuType, GpuVendor, ProgressInfo, SuperResResult
from app.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def make_gpu() -> GpuInfo:
    return GpuInfo(
        vendor=GpuVendor.AMD,
        name="AMD Radeon RX 7900 XT",
        type=GpuType.DISCRETE,
        dedicated_vram_bytes=24 * 1024**3,
        vendor_id=0x1002,
    )


def test_progress_bars_start_static_at_zero(qapp):
    window = MainWindow(make_gpu())
    try:
        assert window.image_progress.minimum() == 0
        assert window.image_progress.maximum() == 100
        assert window.image_progress.value() == 0
        assert window.video_progress.minimum() == 0
        assert window.video_progress.maximum() == 100
        assert window.video_progress.value() == 0
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_progress_handler_uses_task_total_and_current(qapp):
    window = MainWindow(make_gpu())
    try:
        window._on_image_progress(ProgressInfo(current=2, total=5, message="2/5"))
        assert window.image_progress.maximum() == 5
        assert window.image_progress.value() == 2

        window._on_video_progress(ProgressInfo(current=3, total=10, message="3/10"))
        assert window.video_progress.maximum() == 10
        assert window.video_progress.value() == 3
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_image_progress_rounds_fractional_value_and_logs_tile_progress(qapp):
    window = MainWindow(make_gpu())
    try:
        window._on_image_progress(
            ProgressInfo(
                current=2.5,
                total=5,
                message="第 3/5 张 · tile 1/2 · a.png",
                image_index=3,
                image_total=5,
                source_name="a.png",
                phase="tile",
            )
        )

        assert window.image_progress.maximum() == 5
        assert window.image_progress.value() == 3
        assert "处理 a.png" in window.image_log.text_edit.toPlainText()
        assert "tile 1/2" in window.image_log.text_edit.toPlainText()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_image_finished_logs_save_and_summary(qapp):
    window = MainWindow(make_gpu())
    try:
        output = Path("output/a_x2.png")
        window._on_image_finished(
            [
                SuperResResult(
                    ok=True,
                    output_path=output,
                    elapsed_seconds=1.25,
                    input_path=Path("input/a.png"),
                )
            ]
        )

        text = window.image_log.text_edit.toPlainText()
        assert f"保存到 {output}" in text
        assert "成功 1 / 失败 0" in text
        assert "总耗时 1.2s" in text
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
