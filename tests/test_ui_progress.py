import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from app.core.types import GpuInfo, GpuType, GpuVendor, ProgressInfo
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
