import os
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SRT_SETTINGS_DIR", tempfile.mkdtemp(prefix="srt-ui-tests-"))
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from app.core.types import GpuInfo, GpuType, GpuVendor, Scene
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
        assert window.image_model_selector.sharpen_row.isHidden() is False
        assert window.image_model_selector.denoise_strength() == 0.0
        assert window.image_model_selector.sharpen_strength() == 0.25
        assert window.image_scale_selector.scale() == 4

        _select_model(window.image_model_selector, "RealESRGAN_x2plus")
        assert window.image_scale_selector.scale() == 2
        assert window.image_model_selector.denoise_row.isHidden() is False
        assert window.image_model_selector.sharpen_row.isHidden() is False
        assert window.image_model_selector.current_model().denoise_pair_path is None

        # 图片 Tab 的选择不影响视频 Tab。
        assert window.video_model_selector.selected_model_name() == "realesr-general-x4v3"
        assert window.video_model_selector.denoise_row.isHidden() is False
        assert window.video_scale_selector.scale() == 4

        _select_model(window.image_model_selector, "RealESRGAN_x4plus_anime_6B")
        assert window.image_scale_selector.scale() == 4
        assert window.image_model_selector.denoise_row.isHidden() is False
        assert window.image_model_selector.sharpen_row.isHidden() is False

        _select_model(window.image_model_selector, "realesr-general-x4v3")
        assert window.image_model_selector.denoise_row.isHidden() is False
        assert is_denoise_model(window.image_model_selector.selected_registry_model()) is False
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_model_selector_defaults_to_general_with_denoise_slider(qapp):
    window = MainWindow(make_gpu())
    try:
        selector = window.video_model_selector
        assert selector.model_combo.count() == 8  # 7 个内置超分模型 + 自定义
        assert selector.selected_model_name() == "realesr-general-x4v3"
        assert selector.denoise_slider.value() == 0
        # 降噪模型默认不挂载：只有显式选择后才随模型一起下发
        assert selector.current_model().denoise_pair_path is None
        assert selector.layout().spacing() == 10
        assert selector.model_combo.minimumHeight() >= 28
        assert selector.denoise_row.isVisibleTo(selector) is True
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_denoise_model_choice_attaches_pair_to_runtime_meta(qapp):
    window = MainWindow(make_gpu())
    try:
        selector = window.image_model_selector
        assert selector.denoise_model_name() is None
        assert selector.denoise_combo.count() == 3  # 无 + 两个降噪模型
        assert selector.current_model().denoise_pair_path is None

        selector.set_denoise_model("SwinIR_colorDN25")
        model = selector.current_model()

        assert model.denoise_pair_path is not None
        assert model.denoise_pair_path.name == "SwinIR_colorDN25.onnx"
        assert model.denoise_pair_scale == 1
        assert is_denoise_model(selector.current_denoise_model()) is True

        selector.set_denoise_model("realesr-general-wdn-x4v3")
        assert selector.current_model().denoise_pair_scale == 4

        selector.set_denoise_model(None)
        assert selector.current_model().denoise_pair_path is None
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_scene_selector_drives_default_model(qapp):
    window = MainWindow(make_gpu())
    try:
        selector = window.image_model_selector
        anime_index = selector.scene_combo.findData(Scene.ANIME_2D)
        assert anime_index >= 0

        selector.scene_combo.setCurrentIndex(anime_index)

        assert selector.selected_model_name() == "RealESRGAN_x4plus_anime_6B"
        assert selector.current_scene() == Scene.ANIME_2D
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_manual_custom_selection_clears_previous_model_path(qapp):
    window = MainWindow(make_gpu())
    try:
        selector = window.image_model_selector
        assert selector.model_path_edit.text()

        custom_index = selector.model_combo.findData(None)
        assert custom_index >= 0
        selector.model_combo.setCurrentIndex(custom_index)

        assert selector.model_path_edit.text() == ""
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
