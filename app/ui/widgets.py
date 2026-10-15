"""Custom widgets for the desktop shell.

Three classes replace the styled defaults so the user gets the
"liquid glass" feel end-to-end:

* :class:`BackdropWidget` replaces the flat ``QWidget#appRoot``. It paints a
  soft mesh-gradient backdrop under every surface — cards and sidebar then sit
  on it as translucent glass.
* :class:`GlassFrame` and :class:`GlassButton` are the card and the interactive
  surface. They paint a faint rim highlight that follows the cursor (card) or
  a click ripple (button).
* :class:`GlassProgressBar` self-paints a flowing multi-stop chunk.
* :class:`StaggerAnimator` and :func:`stagger_list_items` are pure helpers for
  fade-in + translateY animations on a ``QListWidget``.

The :class:`ChevronComboBox` keeps its job: re-painting the missing
``::down-arrow`` indicator once the stylesheet touches ``QComboBox``.
"""
from __future__ import annotations

import math
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
    QGradient,
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
CHEVRON_INSET = 17


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
        pen = QPen(color, 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        center_x = float(self.width() - CHEVRON_INSET)
        center_y = self.height() / 2.0
        chevron = QPolygonF(
            [
                QPointF(center_x - 4.5, center_y - 2.4),
                QPointF(center_x, center_y + 2.8),
                QPointF(center_x + 4.5, center_y - 2.4),
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
        # The native step buttons still exist; we want them hidden so the
        # user can't accidentally click them and step in the wrong direction.
        # The QSS handles that with ``QSpinBox::up-button,
        # QSpinBox::down-button { width: 0 }`` and matching the arrows.
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
        # Outside the button strip: let the QSpinBox line-edit behaviour
        # run as normal (focus, cursor placement, etc.).
        super().mousePressEvent(event)

    def _is_in_button_area(self, point) -> bool:
        return point.x() >= self.width() - self.BUTTON_WIDTH

    def _is_in_up_area(self, point) -> bool:
        return point.y() < self.height() / 2

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        self._paint_glyphs()

    def _paint_glyphs(self) -> None:
        """Draw a "+" in the up half and a "−" in the down half.

        Both glyphs live entirely within the right-hand button strip,
        so they can never spill into the line-edit area.
        """
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
            r = 5.0
            painter.setPen(
                QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
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
#: Coloured blobs painted as radial gradients under the rest of the window.
#: Low chroma and wide falloff so they read as depth, not as wallpaper.
#: Each entry is (colour, centre x, centre y, radius) as fractions.
_LIGHT_BLOBS = (
    ("#DBEAFE", 0.12, 0.06, 0.75),   # blue wash, top-left
    ("#EDE9FE", 0.95, 0.92, 0.70),   # violet wash, bottom-right
    ("#CCFBF1", 0.58, 0.14, 0.55),   # faint cyan, top-middle
    ("#EFF6FF", 0.50, 0.60, 0.90),   # broad centre lift
)
_DARK_BLOBS = (
    ("#1E3A8A", 0.12, 0.06, 0.78),   # deep blue, top-left
    ("#312E81", 0.95, 0.92, 0.72),   # indigo, bottom-right
    ("#0E7490", 0.58, 0.14, 0.55),   # teal, top-middle
    ("#111C33", 0.50, 0.60, 0.90),   # broad centre lift
)

#: Blob alpha: enough to tint the canvas, far from a hard edge.
_BLOB_ALPHA_LIGHT = 0.55
_BLOB_ALPHA_DARK = 0.62


class BackdropWidget(QWidget):
    """Root widget painting a soft mesh-gradient backdrop.

    Sits behind every child (sidebar, header, cards). The mesh is rendered
    once into a pixmap and blitted afterwards — the gradient would otherwise
    cost four full-window fills on every repaint of the root widget.

    Deliberately *static*: an earlier version drifted the blobs a few pixels
    on a timer, which read as a jump-cut rather than motion.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("appRoot")
        # Every pixel is painted (base fill + blobs cover the whole rect), so
        # Qt can skip its own background erase.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._cache: QPixmap | None = None
        self._cache_key: tuple[int, int, int, str] | None = None

    def invalidate(self) -> None:
        """Drop the cached mesh (call after a theme change)."""
        self._cache = None
        self._cache_key = None
        self.update()

    def _blobs(self) -> tuple[tuple[str, float, float, float], ...]:
        return _DARK_BLOBS if theme.current_mode() == "dark" else _LIGHT_BLOBS

    def _base_color(self) -> QColor:
        return QColor(theme.tokens_for(theme.current_mode())["BG_APP"])

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        ratio = self.devicePixelRatioF()
        key = (self.width(), self.height(), round(ratio * 100), theme.current_mode())
        if self._cache is None or self._cache_key != key:
            self._cache = self._render_mesh(ratio)
            self._cache_key = key
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        painter.end()
        super().paintEvent(event)

    def _render_mesh(self, ratio: float) -> QPixmap:
        pixmap = QPixmap(
            max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio))
        )
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(self._base_color())

        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            rect = QRectF(0.0, 0.0, float(self.width()), float(self.height()))
            alpha = (
                _BLOB_ALPHA_DARK if theme.current_mode() == "dark"
                else _BLOB_ALPHA_LIGHT
            )
            for color, x_share, y_share, radius_share in self._blobs():
                centre = QPointF(rect.width() * x_share, rect.height() * y_share)
                radius = max(rect.width(), rect.height()) * radius_share
                gradient = QRadialGradient(centre, radius)
                inner = QColor(color)
                inner.setAlphaF(alpha)
                middle = QColor(color)
                middle.setAlphaF(alpha * 0.45)
                gradient.setColorAt(0.0, inner)
                gradient.setColorAt(0.55, middle)
                gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
                painter.fillRect(rect, QBrush(gradient))
        finally:
            painter.end()
        return pixmap


# === GlassFrame ============================================================


class GlassFrame(QFrame):
    """A translucent card surface, identical in look to ``LogPanel``.

    No custom painting: the body is filled by the QSS
    ``QFrame#card { background-color: rgba($GLASS_RGB, $CARD_ALPHA);
    border-radius: 12px }`` rule. The inner rim is the only paint added,
    and it's a 1px white line along the rounded edge — the same
    highlight that :class:`LogPanel` gets from the QSS alone.

    ``setFrameShape(NoFrame)`` is still required: Qt Fusion's default
    :class:`QFrame` paints a panel bevel otherwise, and the QSS
    translucent fill is drawn underneath it. The bevel is what was
    showing up as a "black border" in the user's screenshots.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Hard-disable the QFrame panel bevel — the source of the visible
        # black border in the Fusion style.
        self.setFrameShape(QFrame.Shape.NoFrame)


# === GlassButton ===========================================================


class GlassButton(QPushButton):
    """A ``QPushButton`` that paints a click ripple over the stylesheet.

    All chrome, hover, and pressed visuals stay in the QSS so restyling keeps
    working; the ripple is purely additive and lives 300ms.
    """

    RIPPLE_MS = 300

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
        # ``time.monotonic`` in milliseconds is plenty for a 300ms window.
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
        # Repaint only the ring's bounding box, not the whole button: a full
        # repaint per frame is what made the press feel heavy.
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
        # Ease-out expansion: fast at first, settling as it fades.
        reach = math.hypot(self.width(), self.height()) * 1.05
        return max(1.0, reach * (1 - (1 - progress) ** 2.4))

    def _ripple_color(self) -> QColor:
        # On the saturated primary a white flash; on the pale glass buttons a
        # tinted one, because white on white is invisible.
        if self.objectName() == "primaryButton":
            return QColor(255, 255, 255, 150)
        if theme.current_mode() == "dark":
            return QColor(96, 165, 250, 92)
        return QColor(37, 99, 235, 58)

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
            # Clipped to the button's own rounded shape.
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()), 8.0, 8.0)
            painter.fillPath(path, QBrush(gradient))
        finally:
            painter.end()


# === GlassProgressBar ======================================================


class GlassProgressBar(QProgressBar):
    """Progress bar with a flowing multi-stop chunk and a 100% celebration.

    The default ``QProgressBar::chunk`` style can only animate by changing
    geometry; replacing it with a custom paint allows a gradient that *flows*
    across the chunk at ~60fps and a brief colour pulse at completion.

    The animation timer is on only while the bar is actually moving (or
    celebrating) — once the task is done, or before it starts, the chunk is
    static and nothing repaints.
    """

    SHIMMER_INTERVAL_MS = 32
    COMPLETION_PULSE_MS = 600
    #: Horizontal drift of the sheen, in px per tick.
    SHIMMER_SPEED = 0.9

    #: Chunk colour families, keyed by state and resolved against the live
    #: theme at paint time: a solid ``base`` with a brighter ``sheen`` band
    #: sweeping across it. Kept to two colours per family so the sweep stays
    #: subtle — a full base→bright→dark ramp reads as a colour glitch, not as
    #: flow. ``sheen`` is fixed per family because it is a lighter tint of the
    #: base, not a palette role.
    STATE_TOKENS: dict[str, tuple[str, str]] = {
        "normal": ("ACCENT", "ACCENT_LIGHT"),
        "success": ("SUCCESS", "#86EFAC"),
        "warning": ("WARNING", "#FCD34D"),
        "error": ("DANGER", "#FCA5A5"),
        "cancelled": ("NEUTRAL", "#CBD5E1"),
    }

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

    def _chunk_palette(self) -> dict[str, str]:
        base_token, sheen = self.STATE_TOKENS[self._state]
        tokens = theme.tokens_for(theme.current_mode())
        return {"base": tokens[base_token], "sheen": sheen}

    def _animating(self) -> bool:
        """Whether the chunk is worth repainting 30 times a second.

        At rest (0%, or finished) the gradient is static, so the timer stays
        off — otherwise both bars would repaint forever at an idle desktop.
        """
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
        if theme.current_mode() == "dark":
            painter.fillPath(path, QColor(148, 163, 184, 36))
        else:
            painter.fillPath(path, QColor(148, 163, 184, 60))

    #: Width of one sheen band, in px. Fixed rather than proportional to the
    #: chunk so the sweep keeps the same visual speed at every progress level.
    SHEEN_PERIOD = 190.0

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
        base = QColor(palette["base"])
        sheen = QColor(palette["sheen"])

        if self._animating():
            # A bright band travelling left→right over a solid fill. The
            # gradient repeats with a period of SHEEN_PERIOD and the first and
            # last stops are the same colour, so the wrap is invisible — an
            # earlier version slid a 3-stop ramp, which jumped once per cycle.
            phase = (self._phase * self.SHIMMER_SPEED) % self.SHEEN_PERIOD
            gradient = QLinearGradient(
                QPointF(chunk_rect.x() + phase - self.SHEEN_PERIOD,
                        chunk_rect.center().y()),
                QPointF(chunk_rect.x() + phase, chunk_rect.center().y()),
            )
            gradient.setSpread(QGradient.Spread.RepeatSpread)
            gradient.setColorAt(0.0, base)
            gradient.setColorAt(0.5, sheen)
            gradient.setColorAt(1.0, base)
            painter.fillPath(chunk_path, QBrush(gradient))
        else:
            painter.fillPath(chunk_path, base)

        if self._pulse_active:
            # Completion flash: a brief lift on the finished chunk.
            flash = QColor(255, 255, 255)
            flash.setAlpha(max(0, int(96 * (1.0 - self._pulse_phase))))
            painter.fillPath(chunk_path, flash)


# === StaggerAnimator =======================================================


def stagger_list_items(
    list_widget: QListWidget,
    item_indices: Iterable[int],
    *,
    delay_ms: int = 35,
    duration_ms: int = 280,
) -> None:
    """Cascade a sliding accent bar across a set of newly-added rows.

    Installs a :class:`_StaggerItemDelegate` on the list, schedules one entry
    per row (offset by ``delay_ms``), and stops the driving ``QTimer`` once
    every row has finished.

    Test contracts (``addItem``, ``count()``, ``item(row)``, ``clear()``) are
    untouched: we only install an item delegate and toggle per-row opacity.
    """
    targets = [int(i) for i in item_indices]
    if not targets:
        return

    delegate = list_widget.itemDelegate()
    if not isinstance(delegate, _StaggerItemDelegate):
        delegate = _StaggerItemDelegate(list_widget)
        list_widget.setItemDelegate(delegate)

    delegate.schedule(targets, delay_ms=delay_ms, duration_ms=duration_ms)


class _StaggerItemDelegate(QStyledItemDelegate):
    """Item delegate that paints a sliding accent bar on rows currently entering.

    Per-row state is two dicts:
      * ``_start_ms[row]`` — when the row entered (in tick counts of 16ms)
      * ``_progress[row]`` — current 0..1 phase of the slide-in

    A single ``QTimer`` ticks the entries every 16ms (60fps). Once a row's
    phase reaches 1.0 the entry is dropped from the schedules. The delegate's
    ``paint`` only paints the accent bar when the row has a current progress,
    so settled rows look identical to the default delegate.
    """

    TICK_MS = 16

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._delay_ms: int = 35
        self._duration_ms: int = 280
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
        alpha = int(120 * (1.0 - progress))
        accent.setAlpha(max(0, min(255, alpha)))
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(accent)
            bar_width = 3
            offset = int(8 * (1.0 - progress))
            bar_rect = QRect(
                rect.left() - offset,
                rect.top() + 4,
                bar_width,
                rect.height() - 8,
            )
            painter.drawRoundedRect(bar_rect, 2, 2)
        finally:
            painter.restore()


# === ValidatableField =======================================================


class ValidatableField(QFrame):
    """A composable container that adds inline validation to its inner widget.

    Wraps a child ``QLineEdit``/``QComboBox``/``QSpinBox``-style input so that
    missing/invalid input drives an outline ``DANGER_SOFT_BORDER`` colour, an
    inline error label below the widget, and a single ``invalid`` property
    the form code can bind to.

    Conventions (see memory *qframe-panel-bevel-and-painted-shadow*):

    - This widget inherits ``QFrame`` so it must call ``setFrameShape(NoFrame)``
      before any QSS — otherwise Fusion's panel bevel would re-appear.
    - The outline colour is driven by widget-level QSS keyed on the
      ``invalid`` dynamic property; the parent stylesheet stays unaware.
    """

    #: Emits the new invalid state whenever it flips. ``True`` means the
    #: field is currently considered invalid (so a form-level "开始" button
    #: can wire this to ``setEnabled``).
    invalid_changed = Signal(bool)

    #: Tokens that drive the inline QSS — looked up at apply time so a
    #: runtime theme switch keeps the danger colour aligned with the rest of
    #: the glass palette.
    _DANGER_RIM_LIGHT = "DANGER_SOFT_BORDER"
    _DANGER_RIM_DARK = "DANGER_SOFT_BORDER"
    _ERROR_TEXT_LIGHT = "ACCENT_SOFT_TEXT"
    _ERROR_TEXT_DARK = "DANGER_SOFT_BORDER"

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
        # Painted shadow underneath (memory convention, not QGraphicsEffect).
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(2, 2, -2, 6)
        if rect.width() <= 0 or rect.height() <= 0:
            painter.end()
            return
        tokens = theme.tokens_for(theme.current_mode())
        rim_token = (
            self._DANGER_RIM_DARK
            if theme.current_mode() == "dark"
            else self._DANGER_RIM_LIGHT
        )
        rim_color = QColor(tokens[rim_token])
        centre = QPoint(rect.x() + rect.width() // 2, rect.bottom() - 1)
        radius = max(rect.width(), 60) * 0.55
        gradient = QRadialGradient(centre, radius)
        gradient.setColorAt(
            0.0,
            QColor(rim_color.red(), rim_color.green(), rim_color.blue(), 28),
        )
        gradient.setColorAt(
            1.0,
            QColor(rim_color.red(), rim_color.green(), rim_color.blue(), 0),
        )
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
