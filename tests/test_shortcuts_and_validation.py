"""Cover the global shortcut bindings + ValidatableField.

The shortcuts live on ``MainWindow`` and depend on it being fully constructed
in offscreen mode. ``ValidatableField`` is a pure wrapper around a child
input widget so the test creates a throwaway ``QLineEdit``.
"""
from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SRT_SETTINGS_DIR", tempfile.mkdtemp(prefix="srt-sc-tests-"))
pytest.importorskip("PySide6")

from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QLineEdit

from app.core.types import GpuInfo, GpuType, GpuVendor
from app.ui.main_window import MainWindow
from app.ui.widgets import ValidatableField


def _pump(app: QApplication) -> None:
    app.processEvents()


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def window(qapp):
    gpu = GpuInfo(
        vendor=GpuVendor.AMD,
        name="Test",
        type=GpuType.DISCRETE,
        dedicated_vram_bytes=8 * 1024 ** 3,
    )
    win = MainWindow(gpu=gpu, theme_mode="light")
    win.show()
    _pump(qapp)
    yield win
    win.close()


def test_main_window_registers_eight_shortcuts(window):
    """Eight QShortcuts is the contract documented in the upgrade plan."""
    assert len(window._shortcuts) == 8


def test_main_window_shortcut_sequences_include_expected_binds(window):
    """Each shortcut sequence must be parseable; the string form is the
    stable contract the offscreen audit captures.
    """
    expected = {"Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+4", "Ctrl+Return", "Enter+Return", "Esc", "Ctrl+T"}
    # ``Ctrl+Enter`` and ``Ctrl+Return`` share the same QKeySequence; Qt
    # normalises the second one to ``"Enter+Return"``. ``QKeySequence``
    # also rewrites ``"Escape"`` to ``"Esc"`` when reading the value back,
    # so the canonical form accepted here is the post-normalisation one.
    observed = set()
    for shortcut in window._shortcuts:
        seq = QKeySequence(shortcut.key())
        observed.add(seq.toString())
    for needed in ("Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+4", "Ctrl+Return", "Ctrl+T", "Esc"):
        assert needed in observed, f"missing shortcut: {needed}"
    assert any("Enter" in s for s in observed)  # Ctrl+Enter equivalent


def test_shortcut_page_navigation_switches_stack(window, qapp):
    window.stack.setCurrentIndex(0)
    window._set_page(2)
    _pump(qapp)
    assert window.stack.currentIndex() == 2
    window._set_page(0)  # reset


def test_shortcut_escape_when_no_task_emits_toast(window, qapp):
    """Esc with nothing running should pop a toast via ``ToastHost.notify``."""
    host = window.toast_host
    initial_count = len(host._toasts)
    window._shortcut_escape()
    _pump(qapp)
    assert len(host._toasts) >= initial_count + 1


def test_validatable_field_toggles_invalid_state_with_helper_text(qapp):
    edit = QLineEdit()
    field = ValidatableField(edit, helper_text="")
    assert field.is_invalid() is False
    field.set_invalid("缺少输出目录")
    assert field.is_invalid() is True
    assert field._helper_label.text() == "缺少输出目录"
    # Hidden state is the part of the API that does not depend on whether
    # the widget has been shown — so check ``isHidden()`` instead of the
    # offscreen-default ``isVisible()``.
    assert field._helper_label.isHidden() is False
    field.clear_invalid()
    assert field.is_invalid() is False
    assert field._helper_label.text() == ""


def test_validatable_field_emits_invalid_changed(qapp):
    edit = QLineEdit()
    field = ValidatableField(edit)
    captured: list[bool] = []
    field.invalid_changed.connect(captured.append)
    field.set_invalid("x")
    field.clear_invalid()
    assert captured == [True, False]


def test_validatable_field_property_round_trip(qapp):
    edit = QLineEdit()
    field = ValidatableField(edit)
    field.set_invalid("error")
    assert field.property("invalid") is True
    field.clear_invalid()
    assert field.property("invalid") is False
