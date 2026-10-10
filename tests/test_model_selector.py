import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from app.core.types import GpuInfo, GpuType, GpuVendor
from app.ui.main_window import MainWindow
from app.ui.model_selector import is_denoise_model


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


def _select_model(selector, name: str) -> None:
    index = selector.model_combo.findData(name)
    assert index >= 0
    selector.model_combo.setCurrentIndex(index)


def test_tabs_have_independent_model_selectors_and_denoise_visibility(qapp):
    window = MainWindow(make_gpu())
    try:
        assert window.image_model_selector is not None
        assert window.video_model_selector is not None
        assert window.import_model_button is not None

        assert window.image_model_selector.selected_model_name() == "realesr-general-x4v3"
        assert window.image_model_selector.denoise_row.isHidden() is False
        assert window.image_model_selector.denoise_strength() == 0.5
        assert window.image_scale_selector.scale() == 4

        _select_model(window.image_model_selector, "RealESRGAN_x2plus")
        assert window.image_scale_selector.scale() == 2
        assert window.image_model_selector.denoise_row.isHidden() is True
        assert window.image_model_selector.current_model().denoise_pair_path is None

        # 图片 Tab 的选择不影响视频 Tab。
        assert window.video_model_selector.selected_model_name() == "realesr-general-x4v3"
        assert window.video_model_selector.denoise_row.isHidden() is False
        assert window.video_scale_selector.scale() == 4

        _select_model(window.image_model_selector, "RealESRGAN_x4plus_anime_6B")
        assert window.image_scale_selector.scale() == 4
        assert window.image_model_selector.denoise_row.isHidden() is True

        _select_model(window.image_model_selector, "realesr-general-x4v3")
        assert window.image_model_selector.denoise_row.isHidden() is False
        assert is_denoise_model(window.image_model_selector.selected_registry_model()) is True
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_model_selector_defaults_to_general_with_denoise_slider(qapp):
    window = MainWindow(make_gpu())
    try:
        selector = window.video_model_selector
        assert selector.model_combo.count() == 5
        assert selector.selected_model_name() == "realesr-general-x4v3"
        assert selector.denoise_slider.value() == 50
        assert selector.current_model().denoise_pair_path is not None
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
