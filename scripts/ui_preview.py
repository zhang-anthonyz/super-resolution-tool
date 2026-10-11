"""Offscreen UI preview + rendering diagnostics for the PySide6 shell.

Renders every page of the main window to PNG files without opening a real
window (top and scrolled-to-bottom views), then verifies programmatically that

* no interactive control collapsed to an unclickable size,
* combo chevrons and spin buttons actually painted pixels,
* surfaces (cards, sidebar, canvas) use the intended theme colours,
* no widget leaked Fusion's default grey (#EFEFEF) over the light system,
* text labels do not paint an opaque block over their card.

Usage:
    python scripts/ui_preview.py [--out-dir D:\\model_output\\images] [--tag after]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QWidget,
)

from app.core.types import GpuInfo, GpuType, GpuVendor  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

INTERACTIVE = (
    QPushButton,
    QComboBox,
    QLineEdit,
    QSlider,
    QSpinBox,
    QCheckBox,
    QListWidget,
    QProgressBar,
)

#: Controls whose intended height is deliberately below the 18px click floor.
MIN_HEIGHT = {QProgressBar: 8}

#: objectName -> (acceptable dominant colours, widget class)
SURFACES = {
    "sidebar": ({theme.SURFACE}, QFrame),
    "headerCard": ({theme.SURFACE}, QFrame),
    # The log card is mostly its muted log view.
    "card": ({theme.SURFACE, theme.SURFACE_MUTED}, QFrame),
}

#: Labels that sit on a white card and must stay transparent.
CARD_LABELS = {
    "fieldLabel",
    "cardTitle",
    "cardHint",
    "mutedLabel",
    "statusLabel",
    "valueStrong",
}

#: Fusion's default window colour; any occurrence means a stale palette leak.
PALETTE_LEAK = "#EFEFEF"


def make_gpu() -> GpuInfo:
    return GpuInfo(
        vendor=GpuVendor.AMD,
        name="AMD Radeon RX 7900 XT",
        type=GpuType.DISCRETE,
        dedicated_vram_bytes=24 * 1024**3,
        vendor_id=0x1002,
    )


def region_luminance(image: QImage, x: int, y: int, w: int, h: int) -> int:
    """Darkest pixel in a region; low values mean something was drawn there."""
    lowest = 255
    for py in range(max(0, y), min(image.height(), y + h), 2):
        for px in range(max(0, x), min(image.width(), x + w), 2):
            color = image.pixelColor(px, py)
            lowest = min(lowest, (color.red() + color.green() + color.blue()) // 3)
    return lowest


def dominant_color(widget) -> tuple[str, int]:
    """Most frequent colour inside a widget as (#RRGGBB, share in percent)."""
    image = widget.grab().toImage()
    counts: dict[str, int] = {}
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            name = image.pixelColor(x, y).name().upper()
            counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values()) or 1
    color, count = max(counts.items(), key=lambda item: item[1])
    return color, round(count * 100 / total)


def count_color(widget, target: str) -> tuple[int, int]:
    """Count pixels equal to target inside a widget; returns (hits, total)."""
    image = widget.grab().toImage()
    hits = 0
    total = 0
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            total += 1
            if image.pixelColor(x, y).name().upper() == target:
                hits += 1
    return hits, total


def widget_origin(widget) -> QPoint:
    return widget.mapTo(widget.window(), QPoint(0, 0))


def is_fully_visible(widget) -> bool:
    """True when nothing of the widget is clipped (scroll viewport included)."""
    region = widget.visibleRegion()
    if region.isEmpty():
        return False
    bounds = region.boundingRect()
    rect = widget.rect()
    if bounds.width() < rect.width() or bounds.height() < rect.height():
        return False
    origin = widget_origin(widget)
    return origin.x() >= 0 and origin.y() >= 0


def check_geometry(stage: str, window) -> list[str]:
    problems: list[str] = []
    for widget_class in INTERACTIVE:
        for widget in window.findChildren(widget_class):
            if not widget.isVisibleTo(window) or not is_fully_visible(widget):
                continue
            minimum = MIN_HEIGHT.get(widget_class, 18)
            size = widget.size()
            if size.height() < minimum or size.width() < 18:
                name = widget.objectName() or "-"
                problems.append(
                    f"{stage}: {widget_class.__name__}#{name} collapsed to "
                    f"{size.width()}x{size.height()}"
                )
    return problems


def check_indicators(stage: str, window, image: QImage) -> list[str]:
    """Verify combo chevrons and spin buttons painted pixels on screen."""
    problems: list[str] = []
    for widget_class, label in ((QComboBox, "chevron"), (QSpinBox, "spin buttons")):
        for widget in window.findChildren(widget_class):
            if not widget.isVisibleTo(window) or not widget.isEnabled():
                continue
            if not is_fully_visible(widget):
                continue
            rect = widget.visibleRegion().boundingRect()
            origin = widget_origin(widget)
            luminance = region_luminance(
                image,
                origin.x() + rect.x() + rect.width() - 24,
                origin.y() + 2,
                22,
                rect.height() - 4,
            )
            if luminance > 200:
                problems.append(
                    f"{stage}: {widget_class.__name__} {label} region looks blank "
                    f"(lum={luminance}, name={widget.objectName() or '-'})"
                )
    return problems


def check_surfaces(stage: str, window) -> list[str]:
    problems: list[str] = []
    for object_name, (expected, widget_class) in SURFACES.items():
        for widget in window.findChildren(widget_class, object_name):
            if not widget.isVisibleTo(window):
                continue
            color, share = dominant_color(widget)
            allowed = {value.upper() for value in expected}
            if color not in allowed or share < 50:
                problems.append(
                    f"{stage}: surface {object_name} expected one of "
                    f"{sorted(allowed)} to dominate, got {color} ({share}%)"
                )
    return problems


def check_palette_leak(stage: str, window) -> list[str]:
    problems: list[str] = []
    targets = ((QFrame, "card"), (QFrame, "headerCard"), (QFrame, "sidebar"))
    for widget_class, object_name in targets:
        for widget in window.findChildren(widget_class, object_name):
            if not widget.isVisibleTo(window):
                continue
            hits, total = count_color(widget, PALETTE_LEAK)
            if total and hits * 100 // total > 0:
                problems.append(
                    f"{stage}: {object_name} uses {PALETTE_LEAK} on "
                    f"{hits * 100 // total}% of its pixels"
                )
    return problems


def check_canvas(stage: str, window) -> list[str]:
    """The canvas colour must show through, never a palette default."""
    problems: list[str] = []
    image = window.grab().toImage()
    content = window.findChild(QWidget, "contentArea")
    if content is None:
        return problems
    origin = widget_origin(content)
    samples = [
        ("content right margin", origin.x() + content.width() - 10, origin.y() + content.height() // 2),
        ("content top margin", origin.x() + content.width() // 2, origin.y() + 4),
    ]
    for name, x, y in samples:
        color = image.pixelColor(max(0, x), max(0, y)).name().upper()
        if color != theme.BG_APP.upper():
            problems.append(
                f"{stage}: canvas {name} is {color}, expected {theme.BG_APP.upper()}"
            )
    return problems


def check_card_labels(stage: str, window) -> list[str]:
    """Labels on white cards must not paint their own background."""
    problems: list[str] = []
    for label in window.findChildren(QLabel):
        if not label.isVisibleTo(window) or label.objectName() not in CARD_LABELS:
            continue
        if not label.text() or not is_fully_visible(label):
            continue
        image = label.grab().toImage()
        if image.width() < 8 or image.height() < 4:
            continue
        for x in (2, 3, 4):
            if x >= image.width():
                break
            color = image.pixelColor(x, image.height() // 2)
            if color.alpha() < 20:
                continue  # transparent over the card: exactly what we want
            if (
                abs(color.red() - 255) + abs(color.green() - 255) + abs(color.blue() - 255)
                > 24
            ):
                problems.append(
                    f"{stage}: label {label.objectName()} painted "
                    f"{color.name().upper()} over its card "
                    f"(text={label.text()[:12]!r})"
                )
                break
    return problems


def check_nav_selection(stage: str, window) -> list[str]:
    problems: list[str] = []
    checked = 0
    for button in window.findChildren(QPushButton, "navButton"):
        if not button.isChecked():
            continue
        checked += 1
        color, share = dominant_color(button)
        if color != theme.ACCENT_SOFT.upper() or share < 50:
            problems.append(
                f"{stage}: nav button {button.text()!r} background {color} "
                f"({share}%), expected {theme.ACCENT_SOFT.upper()}"
            )
    if checked != 1:
        problems.append(f"{stage}: expected exactly one selected nav button, found {checked}")
    return problems


def check_text_present(stage: str, window) -> list[str]:
    """Every card must render some text pixels (catches clipped/blank paint)."""
    problems: list[str] = []
    for widget_class, object_name in ((QFrame, "card"), (QFrame, "headerCard")):
        for widget in window.findChildren(widget_class, object_name):
            if not widget.isVisibleTo(window):
                continue
            image = widget.grab().toImage()
            dark = 0
            for y in range(image.height()):
                for x in range(image.width()):
                    color = image.pixelColor(x, y)
                    if (color.red() + color.green() + color.blue()) // 3 < 140:
                        dark += 1
            if dark < 25:
                title = widget.findChild(QLabel, "cardTitle")
                problems.append(
                    f"{stage}: {object_name} "
                    f"({title.text() if title else 'header'}) rendered almost no "
                    f"text pixels (dark={dark})"
                )
    return problems


def check_no_horizontal_overflow(window) -> list[str]:
    """Scroll pages must fit the viewport: horizontal clipping is not scrollable."""
    problems: list[str] = []
    for page in (window.image_tab, window.video_tab, window.settings_tab):
        if not isinstance(page, QScrollArea):
            continue
        content = page.widget()
        if content is None:
            continue
        overflow = content.width() - page.viewport().width()
        if overflow > 2:
            problems.append(
                f"page content is {content.width()}px wide but the viewport is "
                f"{page.viewport().width()}px ({overflow}px clipped, unscrollable)"
            )
    return problems


def audit(stage: str, window, image: QImage) -> list[str]:
    problems: list[str] = []
    problems.extend(check_geometry(stage, window))
    problems.extend(check_indicators(stage, window, image))
    problems.extend(check_surfaces(stage, window))
    problems.extend(check_palette_leak(stage, window))
    problems.extend(check_canvas(stage, window))
    problems.extend(check_card_labels(stage, window))
    problems.extend(check_nav_selection(stage, window))
    problems.extend(check_text_present(stage, window))
    problems.extend(check_no_horizontal_overflow(window))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=r"D:\model_output\images")
    parser.add_argument("--tag", default="preview")
    parser.add_argument("--width", type=int, default=1240)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument(
        "--list-controls",
        action="store_true",
        help="print every on-screen interactive control and its size",
    )
    args = parser.parse_args()

    os.environ.setdefault("SRT_SETTINGS_DIR", tempfile.mkdtemp(prefix="srt-preview-cfg-"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    theme.apply_app_theme(app)

    window = MainWindow(make_gpu())
    window.resize(args.width, args.height)
    window.show()
    app.processEvents()

    pager = getattr(window, "stack", None) or getattr(window, "tabs", None)
    pages = [window.image_tab, window.video_tab, window.settings_tab]
    page_names = ["image", "video", "settings"]
    problems: list[str] = []

    for index, (page, page_name) in enumerate(zip(pages, page_names)):
        if pager is not None:
            pager.setCurrentIndex(index)
        app.processEvents()

        scroll = page if isinstance(page, QScrollArea) else None
        bar = scroll.verticalScrollBar() if scroll is not None else None

        states: list[tuple[str, int]] = [("", 0)]
        if bar is not None and bar.maximum() > 0:
            states.append(("-mid", bar.maximum() // 2))
            states.append(("-bottom", bar.maximum()))

        for suffix, offset in states:
            if bar is not None:
                bar.setValue(offset)
            app.processEvents()
            image = window.grab().toImage()
            if suffix == "":
                image.save(str(out_dir / f"ui-{args.tag}-{index + 1}-{page_name}.png"))
            elif suffix == "-bottom":
                image.save(
                    str(out_dir / f"ui-{args.tag}-{index + 1}-{page_name}-bottom.png")
                )
            stage = page_name if not suffix else f"{page_name}{suffix}"
            problems.extend(audit(stage, window, image))

    # The custom-scale spin box only enables itself in the 自定义 state; enable
    # it once on screen to confirm its spin buttons still render.
    if pager is not None:
        pager.setCurrentIndex(0)
    spin = window.image_scale_selector.custom_spin
    spin.setEnabled(True)
    app.processEvents()
    problems.extend(check_indicators("custom-spin", window, window.grab().toImage()))

    print("\n=== SUMMARY ===")
    if problems:
        for problem in problems:
            print(f"PROBLEM {problem}")
    else:
        print("all checks passed")

    if args.list_controls:
        print("\n=== CONTROLS ===")
        for widget_class in INTERACTIVE:
            for widget in window.findChildren(widget_class):
                if not widget.isVisibleTo(window) or not is_fully_visible(widget):
                    continue
                origin = widget_origin(widget)
                print(
                    f"{widget_class.__name__:<14} "
                    f"{widget.objectName() or '-':<14} "
                    f"{widget.size().width()}x{widget.size().height()} "
                    f"at=({origin.x()},{origin.y()})"
                )
    window.close()
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
