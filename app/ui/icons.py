"""Vector icon set for the desktop shell.

The "liquid glass" surface has been replaced with a precision-instrument
aesthetic. Glyphs sit on a 24x24 Lucide-style grid but use a thinner 1.4px
stroke, slightly tighter joins, and a few extra glyphs that the new layout
asks for (motion enable / pause / refresh / pipeline / info).

Every glyph is drawn from a coordinate map so an icon always renders in
the tokens of the theme that is live at paint time — no separate recoloured
asset per state.

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
#: Stroke width in grid units. Was 1.7 (Lucide-canonical); the new "STUDIO"
#: aesthetic wants a thinner, more technical line.
STROKE = 1.4
#: Default logical icon size in px for buttons.
DEFAULT_SIZE = 16


#: States a themed icon is generated for. ``active`` doubles for hover, focus
#: and press; ``on`` is the checked state of a toggle button.
_STATES = ("normal", "active", "on", "disabled")


# === glyph geometry =========================================================
# Each painter receives a QPainter already scaled to the 24x24 grid with the
# stroke pen set; they only emit geometry.


def _draw_image(p: QPainter) -> None:
    """Picture frames — the 图片超分 page."""
    p.drawRoundedRect(QRectF(3.0, 3.5, 18.0, 17.0), 2.0, 2.0)
    p.drawEllipse(QPointF(8.6, 8.6), 1.8, 1.8)
    path = QPainterPath(QPointF(20.6, 15.4))
    path.lineTo(17.9, 12.7)
    path.lineTo(15.1, 12.7)
    path.lineTo(6.6, 21.2)
    p.drawPath(path)


def _draw_video(p: QPainter) -> None:
    """Film camera — the 视频超分 page."""
    p.drawRoundedRect(QRectF(2.5, 6.5, 14.0, 11.0), 1.8, 1.8)
    path = QPainterPath(QPointF(21.5, 8.0))
    path.lineTo(16.5, 12.0)
    path.lineTo(21.5, 16.0)
    path.closeSubpath()
    p.drawPath(path)


def _draw_compare(p: QPainter) -> None:
    """Framed surface with a centre split — before/after comparison."""
    p.drawRoundedRect(QRectF(3.0, 4.5, 18.0, 15.0), 2.0, 2.0)
    p.drawLine(QPointF(12.0, 4.5), QPointF(12.0, 19.5))


def _draw_settings(p: QPainter) -> None:
    """Sliders — settings."""
    p.drawLine(QPointF(20.0, 7.0), QPointF(12.0, 7.0))
    p.drawLine(QPointF(12.0, 17.0), QPointF(4.0, 17.0))
    p.drawEllipse(QPointF(8.0, 7.0), 2.6, 2.6)
    p.drawEllipse(QPointF(16.0, 17.0), 2.6, 2.6)


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
    path = QPainterPath(QPointF(7.5, 4.5))
    path.lineTo(19.0, 12.0)
    path.lineTo(7.5, 19.5)
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
    p.drawRoundedRect(QRectF(3.0, 5.0, 18.0, 14.0), 2.0, 2.0)
    p.drawLine(QPointF(12.0, 8.4), QPointF(12.0, 13.4))
    p.drawLine(QPointF(9.7, 11.2), QPointF(12.0, 13.5))
    p.drawLine(QPointF(14.3, 11.2), QPointF(12.0, 13.5))


def _draw_pause(p: QPainter) -> None:
    """Pause — the cancel button."""
    p.drawRoundedRect(QRectF(6.5, 4.5, 4.0, 15.0), 1.0, 1.0)
    p.drawRoundedRect(QRectF(13.5, 4.5, 4.0, 15.0), 1.0, 1.0)


def _draw_refresh(p: QPainter) -> None:
    """Circular arrow — refresh / re-probe."""
    p.drawArc(QRectF(4.0, 4.5, 16.0, 15.0), 30 * 16, 280 * 16)
    path = QPainterPath(QPointF(15.0, 4.0))
    path.lineTo(20.5, 4.0)
    path.lineTo(20.5, 9.5)
    p.drawPath(path)


def _draw_check(p: QPainter) -> None:
    """Tick — success / accepted."""
    path = QPainterPath(QPointF(4.5, 12.5))
    path.lineTo(10.0, 18.0)
    path.lineTo(19.5, 7.5)
    p.drawPath(path)


def _draw_alert(p: QPainter) -> None:
    """Triangle with exclamation — warning."""
    path = QPainterPath(QPointF(12.0, 3.5))
    path.lineTo(21.5, 19.5)
    path.lineTo(2.5, 19.5)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(12.0, 10.0), QPointF(12.0, 14.5))
    p.drawLine(QPointF(12.0, 16.6), QPointF(12.0, 16.8))


def _draw_info(p: QPainter) -> None:
    """Lowercase i — info / hint."""
    p.drawEllipse(QPointF(12.0, 12.0), 9.0, 9.0)
    p.drawLine(QPointF(12.0, 10.5), QPointF(12.0, 16.5))
    p.drawEllipse(QPointF(12.0, 7.5), 0.7, 0.7)


def _draw_pipeline(p: QPainter) -> None:
    """Pipeline: three nodes with connecting lines — model workflow."""
    p.drawEllipse(QPointF(5.0, 12.0), 3.0, 3.0)
    p.drawEllipse(QPointF(19.0, 12.0), 3.0, 3.0)
    p.drawEllipse(QPointF(12.0, 12.0), 3.0, 3.0)
    p.drawLine(QPointF(8.0, 12.0), QPointF(9.0, 12.0))
    p.drawLine(QPointF(15.0, 12.0), QPointF(16.0, 12.0))


def _draw_chevron_right(p: QPainter) -> None:
    p.drawPolyline(
        QPainterPath(QPointF(9.5, 5.5))
        .lineTo(QPointF(15.5, 12.0))
        .lineTo(QPointF(9.5, 18.5))
        .toSubpathPolygons()[0]
        if False
        else None
    )
    # Fallback: drawPolyline with explicit points (QPainterPath .toSubpathPolygons
    # was a hack from an earlier attempt — keep the call simple).
    from PySide6.QtGui import QPolygonF
    p.drawPolyline(
        QPolygonF(
            [
                QPointF(9.5, 5.5),
                QPointF(15.5, 12.0),
                QPointF(9.5, 18.5),
            ]
        )
    )


def _draw_chevron_down(p: QPainter) -> None:
    from PySide6.QtGui import QPolygonF
    p.drawPolyline(
        QPolygonF(
            [
                QPointF(5.5, 9.5),
                QPointF(12.0, 15.5),
                QPointF(18.5, 9.5),
            ]
        )
    )


def _draw_arrow_up_right(p: QPainter) -> None:
    """Output arrow — used on the upload/result hint."""
    p.drawLine(QPointF(7.0, 17.0), QPointF(17.0, 7.0))
    path = QPainterPath(QPointF(10.0, 7.0))
    path.lineTo(QPointF(17.0, 7.0))
    path.lineTo(QPointF(17.0, 14.0))
    p.drawPath(path)


def _draw_eye(p: QPainter) -> None:
    """Eye — the comparison view icon."""
    path = QPainterPath(QPointF(2.5, 12.0))
    path.cubicTo(QPointF(6.0, 5.0), QPointF(18.0, 5.0), QPointF(21.5, 12.0))
    path.cubicTo(QPointF(18.0, 19.0), QPointF(6.0, 19.0), QPointF(2.5, 12.0))
    path.closeSubpath()
    p.drawPath(path)
    p.drawEllipse(QPointF(12.0, 12.0), 3.0, 3.0)


def _draw_magnifier(p: QPainter) -> None:
    """Magnifier — the lens mode of compare."""
    p.drawEllipse(QPointF(10.5, 10.5), 6.5, 6.5)
    p.drawLine(QPointF(15.4, 15.4), QPointF(20.0, 20.0))


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
    "pause": _draw_pause,
    "refresh": _draw_refresh,
    "check": _draw_check,
    "alert": _draw_alert,
    "info": _draw_info,
    "pipeline": _draw_pipeline,
    "chevron-right": _draw_chevron_right,
    "chevron-down": _draw_chevron_down,
    "arrow-up-right": _draw_arrow_up_right,
    "eye": _draw_eye,
    "magnifier": _draw_magnifier,
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
    """Colour per state for an icon that sits on a hairline (non-accent) surface."""
    tokens = theme.tokens_for(theme.current_mode())
    return {
        "normal": tokens["TEXT_SECONDARY"],
        "active": tokens["TEXT_STRONG"],
        "on": tokens["ACCENT"],
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