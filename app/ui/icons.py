"""Vector icon set for the desktop shell.

The glyphs are *drawn* rather than shipped as assets. The window has two theme
variants and up to five states per button (normal / hover / checked / disabled),
so every icon would otherwise need a separate recoloured file — and each one
would go stale the moment a theme switch repaints the app. Drawing from a
coordinate map means an icon always renders in the tokens of the theme that is
live at paint time.

All glyphs share Lucide's 24x24 grid and a round-capped ~1.7px stroke, so a
hand-mixed set still reads as one family at 16-18px.

Public surface:

* :func:`icon` — a themed :class:`QIcon` with per-state pixmaps, ready for
  ``QPushButton.setIcon``.
* :func:`paint_glyph` — draw a glyph into any painter (used for the in-canvas
  empty states, where a widget-level icon does not apply).
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from . import theme

#: Coordinate grid every glyph is authored on.
GRID = 24.0
#: Stroke width in grid units, matching Lucide's 2px-at-24px look at icon sizes.
STROKE = 1.7
#: Default logical icon size in px for buttons.
DEFAULT_SIZE = 18

#: States a themed icon is generated for. ``active`` doubles for hover, focus
#: and press; ``on`` is the checked state of a toggle button.
_STATES = ("normal", "active", "on", "disabled")


# === glyph geometry =========================================================
# Each painter receives a QPainter already scaled to the 24x24 grid with the
# stroke pen set; they only emit geometry.


def _draw_image(p: QPainter) -> None:
    """Picture frames — the 图片超分 page."""
    p.drawRoundedRect(QRectF(3.0, 3.0, 18.0, 18.0), 2.6, 2.6)
    p.drawEllipse(QPointF(8.8, 8.8), 1.8, 1.8)
    path = QPainterPath(QPointF(21.0, 15.0))
    path.lineTo(17.9, 11.9)
    path.lineTo(15.1, 11.9)
    path.lineTo(6.0, 21.0)
    p.drawPath(path)


def _draw_video(p: QPainter) -> None:
    """Film camera — the 视频超分 page."""
    p.drawRoundedRect(QRectF(2.0, 6.0, 14.0, 12.0), 2.2, 2.2)
    path = QPainterPath(QPointF(22.0, 8.0))
    path.lineTo(16.0, 12.0)
    path.lineTo(22.0, 16.0)
    path.closeSubpath()
    p.drawPath(path)


def _draw_compare(p: QPainter) -> None:
    """Framed surface with a centre split — before/after comparison."""
    p.drawRoundedRect(QRectF(3.0, 4.0, 18.0, 16.0), 2.4, 2.4)
    p.drawLine(QPointF(12.0, 4.0), QPointF(12.0, 20.0))


def _draw_settings(p: QPainter) -> None:
    """Sliders — settings."""
    p.drawLine(QPointF(20.0, 6.5), QPointF(12.5, 6.5))
    p.drawLine(QPointF(11.5, 17.5), QPointF(4.0, 17.5))
    p.drawEllipse(QPointF(8.0, 6.5), 3.0, 3.0)
    p.drawEllipse(QPointF(16.0, 17.5), 3.0, 3.0)


def _draw_plus(p: QPainter) -> None:
    p.drawLine(QPointF(5.0, 12.0), QPointF(19.0, 12.0))
    p.drawLine(QPointF(12.0, 5.0), QPointF(12.0, 19.0))


def _draw_folder_plus(p: QPainter) -> None:
    """Folder with a plus — "add a folder"."""
    path = QPainterPath(QPointF(2.6, 19.4))
    path.lineTo(2.6, 5.6)
    path.lineTo(8.8, 5.6)
    path.lineTo(11.2, 8.6)
    path.lineTo(21.4, 8.6)
    path.lineTo(21.4, 19.4)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(12.0, 11.6), QPointF(12.0, 16.4))
    p.drawLine(QPointF(9.6, 14.0), QPointF(14.4, 14.0))


def _draw_close(p: QPainter) -> None:
    p.drawLine(QPointF(6.5, 6.5), QPointF(17.5, 17.5))
    p.drawLine(QPointF(17.5, 6.5), QPointF(6.5, 17.5))


def _draw_play(p: QPainter) -> None:
    """Start — the run button."""
    path = QPainterPath(QPointF(7.0, 4.2))
    path.lineTo(19.6, 12.0)
    path.lineTo(7.0, 19.8)
    path.closeSubpath()
    p.drawPath(path)


def _draw_fit(p: QPainter) -> None:
    """Corner brackets — fit to window."""
    for points in (
        ((8.5, 3.0), (5.0, 3.0), (3.0, 5.0), (3.0, 8.5)),
        ((15.5, 3.0), (19.0, 3.0), (21.0, 5.0), (21.0, 8.5)),
        ((3.0, 15.5), (3.0, 19.0), (5.0, 21.0), (8.5, 21.0)),
        ((21.0, 15.5), (21.0, 19.0), (19.0, 21.0), (15.5, 21.0)),
    ):
        path = QPainterPath(QPointF(*points[0]))
        for point in points[1:]:
            path.lineTo(QPointF(*point))
        p.drawPath(path)


def _draw_drop(p: QPainter) -> None:
    """Picture frame with a downward arrow entering it — the drop target."""
    p.drawRoundedRect(QRectF(3.0, 5.0, 18.0, 14.0), 2.4, 2.4)
    p.drawLine(QPointF(12.0, 8.4), QPointF(12.0, 13.4))
    p.drawLine(QPointF(9.7, 11.2), QPointF(12.0, 13.5))
    p.drawLine(QPointF(14.3, 11.2), QPointF(12.0, 13.5))


_PAINTERS = {
    "image": _draw_image,
    "video": _draw_video,
    "compare": _draw_compare,
    "settings": _draw_settings,
    "plus": _draw_plus,
    "folder-plus": _draw_folder_plus,
    "close": _draw_close,
    "play": _draw_play,
    "fit": _draw_fit,
    "drop": _draw_drop,
}


# === drawing ===============================================================

def paint_glyph(
    painter: QPainter,
    name: str,
    rect: QRectF,
    colour: str,
    *,
    stroke: float = STROKE,
) -> None:
    """Draw ``name`` centred in ``rect`` with a ``stroke``-px line width."""
    renderer = _PAINTERS.get(name)
    if renderer is None:
        raise KeyError(f"unknown icon: {name!r}")

    side = min(rect.width(), rect.height())
    if side <= 0:
        return
    scale = side / GRID
    painter.save()
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.translate(
            rect.center().x() - side / 2.0, rect.center().y() - side / 2.0
        )
        painter.scale(scale, scale)
        pen = QPen(QColor(colour), stroke / scale)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        renderer(painter)
    finally:
        painter.restore()


def glyph_pixmap(
    name: str,
    size: int,
    colour: str,
    ratio: float = 1.0,
    *,
    stroke: float = STROKE,
) -> QPixmap:
    """A transparent pixmap holding one glyph in ``colour``."""
    pixels = max(1, int(round(size * ratio)))
    pixmap = QPixmap(pixels, pixels)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        paint_glyph(painter, name, QRectF(0.0, 0.0, float(size), float(size)), colour, stroke=stroke)
    finally:
        painter.end()
    return pixmap


# === themed QIcon ==========================================================

#: ``(name, size, mode, colours) -> QIcon`` memo. Small and bounded by the
#: handful of icons the window uses, so it is never pruned.
_CACHE: dict[tuple, QIcon] = {}


def _screen_ratio() -> float:
    """Device pixel ratio of the primary screen (1.0 offscreen / in tests)."""
    try:
        from PySide6.QtWidgets import QApplication  # noqa: WPS433

        app = QApplication.instance()
        if app is None:
            return 1.0
        screen = app.primaryScreen()
        return float(screen.devicePixelRatio()) if screen is not None else 1.0
    except Exception:
        return 1.0


def themed_colours() -> dict[str, str]:
    """Colour per state for an icon that sits on a glass (non-accent) surface."""
    tokens = theme.tokens_for(theme.current_mode())
    return {
        "normal": tokens["TEXT_SECONDARY"],
        "active": tokens["TEXT_STRONG"],
        "on": tokens["ACCENT_SOFT_TEXT"],
        "disabled": tokens["TEXT_FAINT"],
    }


def on_accent_colours() -> dict[str, str]:
    """Colour per state for an icon on the saturated primary button."""
    tokens = theme.tokens_for(theme.current_mode())
    return {state: tokens["SURFACE"] for state in _STATES}


def icon(
    name: str,
    size: int = DEFAULT_SIZE,
    colours: dict[str, str] | None = None,
) -> QIcon:
    """A themed icon with pixmaps for every button state.

    ``colours`` maps any of ``normal`` / ``active`` / ``on`` / ``disabled`` to a
    colour; the gaps fall back to :func:`themed_colours`. Qt picks the pixmap
    per state itself, so a hovered or checked button does not need any wiring.
    """
    resolved = themed_colours() if colours is None else {**themed_colours(), **colours}
    ratio = _screen_ratio()
    key = (
        name,
        size,
        round(ratio, 3),
        theme.current_mode(),
        tuple(sorted(resolved.items())),
    )
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    result = QIcon()
    # Active covers hover/focus/press; Selected covers the checked state of the
    # toggle buttons; both On and Off variants are supplied because styles
    # differ in which pair they ask for.
    plans = (
        ("normal", QIcon.Mode.Normal, QIcon.State.Off),
        ("normal", QIcon.Mode.Normal, QIcon.State.On),
        ("active", QIcon.Mode.Active, QIcon.State.Off),
        ("on", QIcon.Mode.Active, QIcon.State.On),
        ("active", QIcon.Mode.Selected, QIcon.State.Off),
        ("on", QIcon.Mode.Selected, QIcon.State.On),
        ("disabled", QIcon.Mode.Disabled, QIcon.State.Off),
        ("disabled", QIcon.Mode.Disabled, QIcon.State.On),
    )
    for state, mode, icon_state in plans:
        colour = resolved.get(state) or resolved["normal"]
        result.addPixmap(
            glyph_pixmap(name, size, colour, ratio),
            mode,
            icon_state,
        )
    _CACHE[key] = result
    return result
