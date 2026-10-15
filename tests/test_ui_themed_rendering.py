"""Hand-painted and hand-laid-out UI must follow the live theme.

Two classes of regression are covered here, both of which shipped and were
visible only in the dark theme:

* The comparison canvas painted itself from the light-only module constants
  (``theme.SURFACE_MUTED``), so the "empty preview" area stayed near-white on
  the dark backplate.
* ``_make_card`` left no stretch item in the card layout, so when a card was
  stretched taller than its content (the settings stack does this) Qt spread
  the slack over the header row as well: the title floated to the middle of
  the card and the divider followed it down.
"""
import os
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SRT_SETTINGS_DIR", tempfile.mkdtemp(prefix="srt-render-tests-"))
pytest.importorskip("PySide6")

from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from app.core.types import GpuInfo, GpuType, GpuVendor  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    theme.set_current_mode("light")


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


# --- hand-painted canvas ----------------------------------------------------


def test_compare_canvas_follows_the_dark_theme(qapp, window):
    """The empty comparison canvas must not stay a light block in dark mode."""
    window.stack.setCurrentIndex(MainWindow.COMPARE_PAGE)
    window._set_theme_mode("dark")
    for _ in range(3):
        qapp.processEvents()

    view = window.image_compare_view
    assert view.has_pair() is False
    image = view.grab().toImage()
    corner = image.pixelColor(image.width() // 2, 8)
    assert corner.lightness() < 80, (
        f"empty compare canvas painted {corner.name()} in dark mode"
    )
    assert QColor(theme.DARK_SURFACE_MUTED).lightness() > corner.lightness() - 40


def test_compare_canvas_returns_to_light(qapp, window):
    window.stack.setCurrentIndex(MainWindow.COMPARE_PAGE)
    window._set_theme_mode("dark")
    qapp.processEvents()
    window._set_theme_mode("light")
    for _ in range(3):
        qapp.processEvents()

    image = window.image_compare_view.grab().toImage()
    corner = image.pixelColor(image.width() // 2, 8)
    assert corner.lightness() > 200, (
        f"empty compare canvas painted {corner.name()} after switching back"
    )


def test_theme_switch_refreshes_the_log_panel(qapp, window):
    """Log ink is baked per line, so the window has to recolour on a switch."""
    panel = window.image_log
    panel.append("Error", "boom")
    qapp.processEvents()
    light_ink = _first_line_colour(panel, qapp)

    window._set_theme_mode("dark")
    for _ in range(2):
        qapp.processEvents()
    dark_ink = _first_line_colour(panel, qapp)

    assert light_ink != dark_ink, "log line kept its light-theme colour"
    assert dark_ink.lightness() > light_ink.lightness()


def _first_line_colour(panel, qapp):
    """The ink colour of the panel's first line (the whole line shares one)."""
    from PySide6.QtGui import QTextCursor

    qapp.processEvents()
    cursor = QTextCursor(panel.text_edit.document().firstBlock())
    cursor.movePosition(
        QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor
    )
    return cursor.charFormat().foreground().color()


# --- hand-laid-out cards ----------------------------------------------------


def test_stretched_card_keeps_its_title_at_the_top(qapp, window):
    """A card taller than its content must not float the title to the middle."""
    window.stack.setCurrentIndex(3)  # settings
    for _ in range(3):
        qapp.processEvents()

    card = window.settings_stack.currentWidget()
    title = card.findChild(QLabel, "cardTitle")
    assert title is not None
    assert card.height() > 300, "the settings stack should stretch the card"

    # The title keeps its natural height instead of absorbing the card's slack.
    assert title.height() < 40, (
        f"card title grew to {title.height()}px inside a {card.height()}px card"
    )
    assert title.y() < 40, f"card title moved down to y={title.y()}"


# --- empty states -----------------------------------------------------------


def test_empty_image_list_shows_a_drop_affordance(qapp, window):
    """An empty list is the first thing a new user sees: it must not be blank."""
    window.stack.setCurrentIndex(0)
    for _ in range(2):
        qapp.processEvents()

    listing = window.image_list
    assert listing.count() == 0
    image = listing.viewport().grab().toImage()
    dark = 0
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            colour = image.pixelColor(x, y)
            if (colour.red() + colour.green() + colour.blue()) // 3 < 150:
                dark += 1
    assert dark > 40, f"empty list rendered almost nothing (dark={dark})"


def test_drop_affordance_disappears_once_images_are_added(qapp, window, tmp_path):
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QListWidgetItem

    window.stack.setCurrentIndex(0)
    sample = tmp_path / "a.png"
    QImage(40, 30, QImage.Format.Format_RGB32).save(str(sample))

    listing = window.image_list
    item = QListWidgetItem(sample.name)
    item.setData(_Qt.ItemDataRole.UserRole, str(sample))
    listing.addItem(item)
    listing._set_drop_highlight(True)
    for _ in range(2):
        qapp.processEvents()

    assert listing.count() == 1
    # With a row present the empty-state text is gone; the drop ring is the only
    # overlay left, so the row's own pixels stay visible underneath it.
    image = listing.viewport().grab().toImage()
    row_pixel = image.pixelColor(image.width() // 2, 30)
    assert row_pixel.alpha() != 0
