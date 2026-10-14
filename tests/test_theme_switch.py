"""The theme picker must take effect immediately.

Regression cover for a bug where picking 浅色 / 深色 in 设置 → 外观 changed
nothing until the app was restarted: the window kept a constructor-supplied
mode in a separate "override" field and applied that in preference to the
live choice, so every click was silently discarded.
"""
import os
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SRT_SETTINGS_DIR", tempfile.mkdtemp(prefix="srt-theme-tests-"))
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from app.core.types import GpuInfo, GpuType, GpuVendor
from app.ui import theme
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


@pytest.fixture
def window(qapp):
    win = MainWindow(make_gpu(), theme_mode="light")
    win.resize(1240, 860)
    win.show()
    for _ in range(2):
        qapp.processEvents()
    yield win
    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_picking_dark_applies_without_a_restart(qapp, window):
    window._set_theme_mode("dark")
    for _ in range(2):
        qapp.processEvents()

    assert theme.current_mode() == "dark"
    assert theme.DARK_BG_APP in qapp.styleSheet()
    assert window._theme_buttons["dark"].isChecked()


def test_picking_light_switches_back(qapp, window):
    window._set_theme_mode("dark")
    qapp.processEvents()
    window._set_theme_mode("light")
    qapp.processEvents()

    assert theme.current_mode() == "light"
    assert theme.DARK_BG_APP not in qapp.styleSheet()
    assert window._theme_buttons["light"].isChecked()


def test_switch_is_visible_in_the_rendered_pixels(qapp, window):
    """The strongest form of the contract: the window actually repaints."""
    window._set_theme_mode("light")
    qapp.processEvents()
    light = window.grab().toImage()
    light_pixel = light.pixelColor(4, light.height() - 4)

    window._set_theme_mode("dark")
    qapp.processEvents()
    dark = window.grab().toImage()
    dark_pixel = dark.pixelColor(4, dark.height() - 4)

    assert light_pixel.lightness() > dark_pixel.lightness() + 40, (
        f"canvas did not darken: light={light_pixel.name()} "
        f"dark={dark_pixel.name()}"
    )


def test_choice_survives_a_window_restart(qapp, window):
    """The choice is written to settings and read back on construction."""
    window._set_theme_mode("dark")
    qapp.processEvents()
    assert window._settings["theme_mode"] == "dark"

    rebuilt = MainWindow(make_gpu())
    try:
        assert rebuilt._theme_mode == "dark"
        assert theme.current_mode() == "dark"
    finally:
        rebuilt.close()
        rebuilt.deleteLater()
        qapp.processEvents()


def test_progress_bar_is_idle_until_it_has_work(qapp, window):
    """No background repainting at 0%: the shimmer timer must stay off."""
    assert not window.image_progress._timer.isActive()
    assert not window.video_progress._timer.isActive()

    window.image_progress.setValue(500)
    assert window.image_progress._timer.isActive()

    window.image_progress.setValue(window.image_progress.maximum())
    window.image_progress.set_state("success")
    for _ in range(40):  # let the completion pulse play out
        qapp.processEvents()
        window.image_progress._on_tick()
    assert not window.image_progress._timer.isActive()
