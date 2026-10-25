"""Custom widgets for the desktop shell.

The "liquid glass" surface has been replaced with a precision-instrument
aesthetic ("STUDIO"):

* :class:`BackdropWidget` replaces the flat ``QWidget#appRoot``. It paints a
  deep charcoal canvas overlaid with a 1px engineering grid + a faint film
  grain — the texture under every card.
* :class:`HairlineFrame` (was ``GlassFrame``) is the card. A 1px border and a
  single inner top highlight read as a precision-engineered surface instead
  of frosted glass.
* :class:`RippleButton` (was ``GlassButton``) keeps the click ripple but the
  chrome is hairline-bordered and the press fades the entire button.
* :class:`HairlineProgressBar` (was ``GlassProgressBar``) self-paints a thin
  chunk with a refined shimmer and a 100% celebration.
* :class:`StaggerAnimator` and :func:`stagger_list_items` are pure helpers for
  fade-in + translateY animations on a ``QListWidget``.

The :class:`ChevronComboBox` keeps its job: re-painting the missing
``::down-arrow`` indicator once the stylesheet touches ``QComboBox``.
"""
from __future__ import annotations

import math
import random
from typing import Iterable

from PySide6.QtCore import (
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from . import theme


#: Horizontal distance from the right edge to the chevron centre.
CHEVRON_INSET = 16


# === ChevronComboBox (unchanged) ============================================


class ChevronComboBox(QComboBox):
    """Combo box with a hand-drawn chevron indicator.

    The native drop-down button is suppressed by the QSS
    (``QComboBox::drop-down { width: 0 }``); this widget only paints the
    accent chevron, centred at the right inset.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        self._paint_chevron()

    def _paint_chevron(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        tokens = theme.tokens_for(theme.current_mode())
        if not self.isEnabled():
            color = QColor(tokens["TEXT_FAINT"])
        else:
            # Use the accent at full strength on the focus border; the
            # chevron gets the muted text colour otherwise.
            if self.hasFocus():
                color = QColor(tokens["ACCENT"])
            else:
                color = QColor(tokens["TEXT_SECONDARY"])
        pen = QPen(color, 1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        center_x = float(self.width() - CHEVRON_INSET)
        center_y = self.height() / 2.0
        chevron = QPolygonF(
            [
                QPointF(center_x - 4.0, center_y - 2.0),
                QPointF(center_x, center_y + 2.4),
                QPointF(center_x + 4.0, center_y - 2.0),
            ]
        )
        painter.drawPolyline(chevron)
        painter.end()


class PlusMinusSpinBox(QSpinBox):
    """A QSpinBox whose step buttons show explicit "+" and "−" glyphs.

    Earlier iterations tried to fight QSS subcontrol-positioning on the
    up/down buttons, but the QSpinBox internals are not stable enough
    for that to work cross-platform — the up-button can end up covering
    the whole widget, the down-button can vanish, and a click in the
    lower half can still step *up*. The reliable fix is to take the
    click handling into our own hands: route left-clicks in the right
    half of the widget to ``stepUp`` / ``stepDown`` based on which
    half they fall in, and skip the QSS subcontrols entirely. The
    native up/down arrows are hidden via QSS so the user only ever
    sees our own glyphs.
    """

    # Pixel width reserved for the step buttons on the right edge.
    BUTTON_WIDTH = 22

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.isEnabled()
            and self._is_in_button_area(event.position().toPoint())
        ):
            if self._is_in_up_area(event.position().toPoint()):
                self.stepUp()
            else:
                self.stepDown()
            event.accept()
            return
        super().mousePressEvent(event)

    def _is_in_button_area(self, point) -> bool:
        return point.x() >= self.width() - self.BUTTON_WIDTH

    def _is_in_up_area(self, point) -> bool:
        return point.y() < self.height() / 2

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        self._paint_glyphs()

    def _paint_glyphs(self) -> None:
        """Draw a "+" in the up half and a "−" in the down half."""
        painter = QPainter(self)
        try:
            painter.setClipping(False)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            tokens = theme.tokens_for(theme.current_mode())
            if not self.isEnabled():
                color = QColor(tokens["TEXT_FAINT"])
            else:
                color = QColor(tokens["TEXT_SECONDARY"])

            cx = self.width() - self.BUTTON_WIDTH / 2
            r = 4.0
            painter.setPen(
                QPen(color, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            )

            up_cy = self.height() * 0.25
            down_cy = self.height() * 0.75
            painter.drawLine(QPointF(cx - r, up_cy), QPointF(cx + r, up_cy))
            painter.drawLine(QPointF(cx, up_cy - r), QPointF(cx, up_cy + r))
            painter.drawLine(
                QPointF(cx - r, down_cy), QPointF(cx + r, down_cy)
            )
        finally:
            painter.end()


# === BackdropWidget ========================================================
#: Engineering grid: 24px square cells, drawn as 1px hairlines. The grid
#: stays *just* above the canvas so it reads as "instrumentation", not
#: "graph paper". Slight per-cell pixel offset keeps the grid from forming
#: moiré with the device pixel ratio at 1.0.
_GRID_PX = 24
_GRID_ALPHA_LIGHT = 22
_GRID_ALPHA_DARK = 26

#: Film-grain noise — pre-baked into a small tile so painting it stays cheap.
#: The tile is rebuilt on theme switch (alpha changes) and kept otherwise.
_GRAIN_TILE_PX = 192
_GRAIN_DENSITY = 0.05


def _build_grain_tile(mode: str) -> QPixmap:
    """Bake a small greyscale-noise tile once and stamp it across the canvas."""
    tokens = theme.tokens_for(mode)
    r, g, b = (int(part) for part in tokens["GRAIN_RGB"].split(","))
    alpha = float(tokens["GRAIN_ALPHA"])
    tile = QPixmap(_GRAIN_TILE_PX, _GRAIN_TILE_PX)
    tile.fill(Qt.GlobalColor.transparent)
    painter = QPainter(tile)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rng = random.Random(0x535455)  # deterministic per process
        for y in range(_GRAIN_TILE_PX):
            for x in range(_GRAIN_TILE_PX):
                if rng.random() > _GRAIN_DENSITY:
                    continue
                speck = QColor(r, g, b)
                speck.setAlphaF(alpha)
                painter.fillRect(x, y, 1, 1, speck)
    finally:
        painter.end()
    return tile


class BackdropWidget(QWidget):
    """Root widget painting a deep canvas with a 1px grid + film grain.

    The mesh-gradient backdrop is gone: it read as wallpaper, not as a working
    surface. The replacement is two cheap static layers — a 24px engineering
    grid and a procedural noise tile — that read as "instrument bezel" without
    distracting from the cards on top. Both layers are cached so a repaint
    does not cost a full re-render.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("appRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._cache: QPixmap | None = None
        self._cache_key: tuple[int, int, str] | None = None
        self._grain_tile: QPixmap | None = None
        self._grain_tile_mode: str | None = None

    def invalidate(self) -> None:
        """Drop the cached backdrop (call after a theme change)."""
        self._cache = None
        self._cache_key = None
        self._grain_tile = None
        self._grain_tile_mode = None
        self.update()

    def _grain_for_mode(self) -> QPixmap:
        mode = theme.current_mode()
        if self._grain_tile is None or self._grain_tile_mode != mode:
            self._grain_tile = _build_grain_tile(mode)
            self._grain_tile_mode = mode
        return self._grain_tile

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        ratio = self.devicePixelRatioF()
        key = (self.width(), self.height(), theme.current_mode())
        if self._cache is None or self._cache_key != key:
            self._cache = self._render_canvas(ratio)
            self._cache_key = key
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        painter.end()
        super().paintEvent(event)

    def _render_canvas(self, ratio: float) -> QPixmap:
        pixmap = QPixmap(
            max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio))
        )
        pixmap.setDevicePixelRatio(ratio)
        # Base fill — the BG_APP token.
        tokens = theme.tokens_for(theme.current_mode())
        pixmap.fill(QColor(tokens["BG_APP"]))

        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

            # 1px engineering grid. Drawn straight onto the pixmap at full
            # resolution; the alpha is set in stone so it never washes out
            # the BG colour.
            grid = QColor(tokens["GRID_RGB"])
            grid_alpha = (
                _GRID_ALPHA_DARK if theme.current_mode() == "dark"
                else _GRID_ALPHA_LIGHT
            )
            grid.setAlpha(grid_alpha)
            painter.setPen(QPen(grid, 1.0 / ratio))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            width = self.width()
            height = self.height()
            for x in range(_GRID_PX, width, _GRID_PX):
                painter.drawLine(QPointF(x, 0.0), QPointF(x, float(height)))
            for y in range(_GRID_PX, height, _GRID_PX):
                painter.drawLine(QPointF(0.0, y), QPointF(float(width), y))

            # Film grain — stamped across the canvas. Qt handles tiled draws
            # natively so this stays under a frame even on 4K displays.
            grain = self._grain_for_mode()
            painter.drawTiledPixmap(0, 0, width, height, grain)
        finally:
            painter.end()
        return pixmap


# === HairlineFrame =========================================================


class HairlineFrame(QFrame):
    """A precise card surface — 1px border + 1px inner top highlight.

    The QSS handles the background + border colour; this widget adds the
    inner highlight (a 1px line along the top edge) that turns a flat
    rectangle into a "panel".

    ``setFrameShape(NoFrame)`` is still required: Qt Fusion's default
    :class:`QFrame` paints a panel bevel otherwise, and the QSS
    translucent fill is drawn underneath it.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            tokens = theme.tokens_for(theme.current_mode())
            # 1px inner top highlight — slightly lighter than the border.
            rim = QColor(tokens["BORDER_STRONG"])
            rim.setAlpha(140 if theme.current_mode() == "dark" else 120)
            pen = QPen(rim, 1.0)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(1.0, 0.6), QPointF(self.width() - 1.0, 0.6)
            )
        finally:
            painter.end()


#: Public alias for callers that still import the old name. Keeps any
#: test or external import working without a sweeping rename.
GlassFrame = HairlineFrame


# === RippleButton ==========================================================


class RippleButton(QPushButton):
    """A ``QPushButton`` that paints a click ripple over the stylesheet.

    The chrome stays in the QSS so a theme switch repaints correctly; the
    ripple is purely additive and lives 280ms.
    """

    RIPPLE_MS = 280

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ripple_point: QPoint | None = None
        self._ripple_start = 0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._ripple_point = event.position().toPoint()
            self._ripple_start = self._elapsed_ms()
            self._timer.start()
            self.update()
        super().mousePressEvent(event)

    def _elapsed_ms(self) -> int:
        import time

        return int(time.monotonic() * 1000)

    def _tick(self) -> None:
        if self._ripple_point is None:
            self._timer.stop()
            return
        elapsed = self._elapsed_ms() - self._ripple_start
        if elapsed >= self.RIPPLE_MS:
            self._ripple_point = None
            self._timer.stop()
            self.update()
            return
        self.update(self._ripple_bounds())

    def _ripple_bounds(self) -> QRect:
        elapsed = self._elapsed_ms() - self._ripple_start
        progress = max(0.0, min(1.0, elapsed / self.RIPPLE_MS))
        radius = self._ripple_radius(progress)
        point = self._ripple_point or QPoint(0, 0)
        return QRect(
            int(point.x() - radius), int(point.y() - radius),
            int(radius * 2) + 2, int(radius * 2) + 2,
        )

    def _ripple_radius(self, progress: float) -> float:
        reach = math.hypot(self.width(), self.height()) * 1.05
        return max(1.0, reach * (1 - (1 - progress) ** 2.4))

    def _ripple_color(self) -> QColor:
        # On the saturated primary a white flash; on the hairline buttons a
        # tinted one, because white on white is invisible.
        if self.objectName() == "primaryButton":
            return QColor(255, 255, 255, 140)
        tokens = theme.tokens_for(theme.current_mode())
        if theme.current_mode() == "dark":
            return QColor(94, 234, 212, 96)
        return QColor(14, 143, 110, 60)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self._ripple_point is None:
            return
        elapsed = self._elapsed_ms() - self._ripple_start
        if elapsed < 0 or elapsed >= self.RIPPLE_MS:
            return
        progress = elapsed / self.RIPPLE_MS
        radius = self._ripple_radius(progress)
        base = self._ripple_color()
        fade = (1.0 - progress) ** 1.5
        centre = QColor(base)
        centre.setAlpha(max(0, int(base.alpha() * fade)))
        shoulder = QColor(base)
        shoulder.setAlpha(max(0, int(base.alpha() * fade * 0.45)))
        edge = QColor(base)
        edge.setAlpha(0)

        gradient = QRadialGradient(QPointF(self._ripple_point), radius)
        gradient.setColorAt(0.0, centre)
        gradient.setColorAt(0.55, shoulder)
        gradient.setColorAt(1.0, edge)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            path = QPainterPath()
            path.addRoundedRect(
                QRectF(self.rect()), 6.0, 6.0
            )
            painter.fillPath(path, QBrush(gradient))
        finally:
            painter.end()


#: Public alias for callers that still import the old name.
GlassButton = RippleButton


# === HairlineProgressBar ===================================================


def _lighten(color: QColor, factor: float) -> QColor:
    """Blend ``color`` toward white by ``factor`` (0 = unchanged, 1 = white)."""
    return QColor(
        int(round(color.red() + (255 - color.red()) * factor)),
        int(round(color.green() + (255 - color.green()) * factor)),
        int(round(color.blue() + (255 - color.blue()) * factor)),
    )


class HairlineProgressBar(QProgressBar):
    """Progress bar with a flowing sheen chunk and a 100% celebration.

    The default ``QProgressBar::chunk`` style can only animate by changing
    geometry; replacing it with a custom paint allows a gradient that *flows*
    across the chunk at ~60fps and a brief lift on completion.

    The animation timer is on only while the bar is actually moving (or
    celebrating) — once the task is done, or before it starts, the chunk is
    static and nothing repaints.
    """

    SHIMMER_INTERVAL_MS = 32
    COMPLETION_PULSE_MS = 600
    #: Horizontal drift of the sheen, in px per tick.
    SHIMMER_SPEED = 0.9

    #: Chunk colour families, keyed by state and resolved against the live
    #: theme at paint time. Fills come from the medium-saturation PROGRESS_*
    #: tokens (the text-safe ACCENT/SUCCESS… family reads almost black as a
    #: fat fill); the sheen band is the fill lightened toward white, so it
    #: stays in the same hue family in every state.
    STATE_TOKENS: dict[str, str] = {
        "normal": "PROGRESS_NORMAL",
        "success": "PROGRESS_SUCCESS",
        "warning": "PROGRESS_WARNING",
        "error": "PROGRESS_ERROR",
        "cancelled": "PROGRESS_NEUTRAL",
    }

    #: How much the sheen lightens the fill, per mode. Kept low: the band is
    #: a quiet ripple, not a mirror stripe.
    SHEEN_LIGHTEN = {"light": 0.42, "dark": 0.30}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._phase = 0
        self._state = "normal"
        self._pulse_phase: float = 0.0
        self._pulse_active = False

        self._timer = QTimer(self)
        self._timer.setInterval(self.SHIMMER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

    # Public API ------------------------------------------------------------
    def set_state(self, state: str) -> None:
        """Set the chunk colour family. Accepts: normal|success|warning|error|cancelled."""
        if state not in self.STATE_TOKENS:
            raise ValueError(f"unknown progress-bar state: {state}")
        self._state = state
        # A successful run that is already at 100% gets the flash.
        if state == "success" and self.value() == self.maximum():
            self._start_completion_pulse()
        self._sync_animation()
        self.update()

    def _chunk_palette(self) -> dict[str, QColor]:
        mode = theme.current_mode()
        tokens = theme.tokens_for(mode)
        base = QColor(tokens[self.STATE_TOKENS[self._state]])
        return {"base": base, "sheen": _lighten(base, self.SHEEN_LIGHTEN[mode])}

    def _animating(self) -> bool:
        """Whether the chunk is worth repainting 30 times a second."""
        return self.minimum() < self.value() < self.maximum() or self._pulse_active

    def _sync_animation(self) -> None:
        if self._animating():
            if not self._timer.isActive():
                self._timer.start()
        elif self._timer.isActive():
            self._timer.stop()

    def _on_tick(self) -> None:
        self._phase = (self._phase + 1) % 4096
        if self._pulse_active:
            self._pulse_phase += self.SHIMMER_INTERVAL_MS / self.COMPLETION_PULSE_MS
            if self._pulse_phase >= 1.0:
                self._pulse_active = False
        self._sync_animation()
        self.update()

    def setValue(self, value: int) -> None:  # noqa: N802 - Qt naming
        super().setValue(value)
        if value >= self.maximum() and self._state == "success":
            self._start_completion_pulse()
        self._sync_animation()
        self.update()

    def setRange(self, minimum: int, maximum: int) -> None:  # noqa: N802
        super().setRange(minimum, maximum)
        self._sync_animation()

    def _start_completion_pulse(self) -> None:
        self._pulse_active = True
        self._pulse_phase = 0.0
        self._sync_animation()

    # Painting --------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._draw_background(painter)
            if self.maximum() > self.minimum():
                self._draw_chunk(painter)
        finally:
            painter.end()

    def _draw_background(self, painter: QPainter) -> None:
        rect = QRectF(self.rect())
        radius = rect.height() / 2.0
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        tokens = theme.tokens_for(theme.current_mode())
        painter.fillPath(path, QColor(tokens["PROGRESS_TRACK"]))

    def _draw_chunk(self, painter: QPainter) -> None:
        span = self.maximum() - self.minimum()
        progress = (self.value() - self.minimum()) / span
        if progress <= 0:
            return
        full_rect = QRectF(self.rect())
        chunk_width = full_rect.width() * progress
        if chunk_width <= 0:
            return

        chunk_rect = QRectF(
            full_rect.x(), full_rect.y(), chunk_width, full_rect.height()
        )
        chunk_path = QPainterPath()
        radius = full_rect.height() / 2.0
        chunk_path.addRoundedRect(chunk_rect, radius, radius)

        palette = self._chunk_palette()
        base = palette["base"]
        sheen = palette["sheen"]

        if self._animating():
            period = self._sheen_period()
            phase = (self._phase * self.SHIMMER_SPEED) % period
            gradient = QLinearGradient(
                QPointF(chunk_rect.x() + phase - period,
                        chunk_rect.center().y()),
                QPointF(chunk_rect.x() + phase, chunk_rect.center().y()),
            )
            from PySide6.QtGui import QGradient
            gradient.setSpread(QGradient.Spread.RepeatSpread)
            gradient.setColorAt(0.0, base)
            gradient.setColorAt(0.5, sheen)
            gradient.setColorAt(1.0, base)
            painter.fillPath(chunk_path, QBrush(gradient))
        else:
            painter.fillPath(chunk_path, base)

        if self._pulse_active:
            flash = QColor(255, 255, 255)
            flash.setAlpha(max(0, int(96 * (1.0 - self._pulse_phase))))
            painter.fillPath(chunk_path, flash)

    #: Width of one sheen band. At least one bar-width so a wide bar shows a
    #: single quiet ripple instead of repeated mirror stripes.
    def _sheen_period(self) -> float:
        return max(260.0, float(self.width()))


#: Public alias for callers that still import the old name.
GlassProgressBar = HairlineProgressBar


# === StaggerAnimator =======================================================


def stagger_list_items(
    list_widget: QListWidget,
    item_indices: Iterable[int],
    *,
    delay_ms: int = 32,
    duration_ms: int = 240,
) -> None:
    """Cascade a sliding accent bar across a set of newly-added rows."""
    targets = [int(i) for i in item_indices]
    if not targets:
        return

    delegate = list_widget.itemDelegate()
    if not isinstance(delegate, _StaggerItemDelegate):
        delegate = _StaggerItemDelegate(list_widget)
        list_widget.setItemDelegate(delegate)

    delegate.schedule(targets, delay_ms=delay_ms, duration_ms=duration_ms)


class _StaggerItemDelegate(QStyledItemDelegate):
    """Item delegate that paints a sliding accent bar on rows currently entering."""

    TICK_MS = 16

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._delay_ms: int = 32
        self._duration_ms: int = 240
        self._start_ms: dict[int, int] = {}
        self._progress: dict[int, float] = {}
        self._tick_count: int = 0
        self._timer = QTimer(parent)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._on_tick)

    def schedule(
        self, rows: Iterable[int], *, delay_ms: int, duration_ms: int
    ) -> None:
        self._delay_ms = max(0, delay_ms)
        self._duration_ms = max(1, duration_ms)
        for index, row in enumerate(rows):
            start = self._tick_count + (index * self._delay_ms) // self.TICK_MS
            self._start_ms[row] = start
            self._progress[row] = 0.0
        if rows and not self._timer.isActive():
            self._timer.start()

    def _on_tick(self) -> None:
        self._tick_count += 1
        finished: list[int] = []
        for row, start in list(self._start_ms.items()):
            local = (self._tick_count - start) * self.TICK_MS
            if local <= 0:
                progress = 0.0
            elif local >= self._duration_ms:
                progress = 1.0
            else:
                progress = local / self._duration_ms
            self._progress[row] = progress
            if progress >= 1.0:
                finished.append(row)
        for row in finished:
            self._start_ms.pop(row, None)
            self._progress.pop(row, None)

        parent = self.parent()
        if parent is not None:
            parent.viewport().update()
        if not self._start_ms and self._timer.isActive():
            self._timer.stop()

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index,
    ) -> None:
        super().paint(painter, option, index)
        progress = self._progress.get(index.row())
        if progress is None or progress >= 1.0:
            return
        rect = option.rect
        accent = QColor(theme.tokens_for(theme.current_mode())["ACCENT"])
        alpha = int(140 * (1.0 - progress))
        accent.setAlpha(max(0, min(255, alpha)))
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(accent)
            bar_width = 2
            offset = int(8 * (1.0 - progress))
            bar_rect = QRect(
                rect.left() - offset,
                rect.top() + 4,
                bar_width,
                rect.height() - 8,
            )
            painter.drawRoundedRect(bar_rect, 1, 1)
        finally:
            painter.restore()


# === ValidatableField =======================================================


class ValidatableField(QFrame):
    """A composable container that adds inline validation to its inner widget."""

    #: Emits the new invalid state whenever it flips. ``True`` means the
    #: field is currently considered invalid (so a form-level "开始" button
    #: can wire this to ``setEnabled``).
    invalid_changed = Signal(bool)

    _DANGER_RIM_LIGHT = "DANGER_SOFT_BORDER"
    _DANGER_RIM_DARK = "DANGER_SOFT_BORDER"
    _ERROR_TEXT_LIGHT = "DANGER"
    _ERROR_TEXT_DARK = "DANGER"

    def __init__(
        self,
        input_widget: QWidget,
        helper_text: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("validatableField")
        self._input = input_widget
        self._invalid = False
        self._error_text = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        outer.addWidget(input_widget)

        self._helper_label = QLabel(self)
        self._helper_label.setObjectName("fieldHelper")
        self._helper_label.setWordWrap(True)
        if helper_text:
            self._helper_label.setText(helper_text)
        else:
            self._helper_label.hide()
        outer.addWidget(self._helper_label)

        self._refresh_style()

    # ------------------------------------------------------------------ API
    def input_widget(self) -> QWidget:
        return self._input

    def is_invalid(self) -> bool:
        return self._invalid

    def set_invalid(self, message: str) -> None:
        self._error_text = message
        self._helper_label.setText(message)
        self._helper_label.show()
        if not self._invalid:
            self._invalid = True
            self.setProperty("invalid", True)
            self._refresh_style()
            self.invalid_changed.emit(True)

    def clear_invalid(self) -> None:
        self._error_text = ""
        self._helper_label.setText("")
        self._helper_label.hide()
        if self._invalid:
            self._invalid = False
            self.setProperty("invalid", False)
            self._refresh_style()
            self.invalid_changed.emit(False)

    def refresh_style(self) -> None:
        """Re-pick the danger colour from the current theme tokens."""
        self._refresh_style()

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        if not self._invalid:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        tokens = theme.tokens_for(theme.current_mode())
        rim_token = (
            self._DANGER_RIM_DARK
            if theme.current_mode() == "dark"
            else self._DANGER_RIM_LIGHT
        )
        rim_color = QColor(tokens[rim_token])
        centre = QPoint(self.width() // 2, self.height() - 1)
        radius = max(self.width(), 60) * 0.55
        gradient = QRadialGradient(centre, radius)
        gradient.setColorAt(0.0, QColor(rim_color.red(), rim_color.green(), rim_color.blue(), 32))
        gradient.setColorAt(1.0, QColor(rim_color.red(), rim_color.green(), rim_color.blue(), 0))
        painter.fillRect(self.rect(), gradient)
        painter.end()

    # ------------------------------------------------------------------ internals
    def _refresh_style(self) -> None:
        tokens = theme.tokens_for(theme.current_mode())
        text_token = (
            self._ERROR_TEXT_DARK
            if theme.current_mode() == "dark"
            else self._ERROR_TEXT_LIGHT
        )
        error_colour = tokens[text_token]
        helper_colour = tokens["TEXT_MUTED"]
        qss = (
            f"QLabel#fieldHelper {{"
            f"  color: {error_colour if self._invalid else helper_colour};"
            f"  background: transparent;"
            f"  border: none;"
            f"  font-size: 11px;"
            f"  padding-left: 2px;"
            f"}}"
        )
        self._helper_label.setStyleSheet(qss)