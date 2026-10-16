"""Transient toast notifications stacked in the bottom-right of the main window.

Replaces most of the legacy ``QMessageBox.warning`` / ``information`` call
sites so the workflow is not interrupted by modal dialogs. Four message kinds
are distinguished (``info`` / ``warning`` / ``error`` / ``success``) and each
maps to one of the theme accent tokens, so the visual matches the rest of
the glass-style surface.

Conventions (see project memory *qframe-panel-bevel-and-painted-shadow*):

- Every ``QFrame``-derived widget calls ``setFrameShape(NoFrame)`` so the
  Fusion panel bevel cannot leak through and the QSS background wins.
- The drop shadow is **painted** in ``paintEvent`` via a radial gradient
  rather than a ``QGraphicsDropShadowEffect`` (which is expensive and breaks
  the offscreen ``ui_preview.py`` audit path).
- Toast colours come from ``theme.tokens_for(theme.current_mode())`` and are
  re-applied at runtime; the host listens for theme changes by walking the
  live Toast set.

Public entry points:

- :func:`install_toast_host` — call once from ``MainWindow.__init__``.
- :meth:`ToastHost.notify` — the only API most callers should use.
"""
from __future__ import annotations

import weakref
from typing import Literal

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QPainter,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import theme

_Kind = Literal["info", "warning", "error", "success"]

#: Theme token that drives the rim and icon colour for each kind.
_KIND_TO_ACCENT = {
    "info": "ACCENT",
    "warning": "WARNING",
    "error": "DANGER",
    "success": "SUCCESS",
}

#: Compact glyph drawn in the toast icon slot. Lowercase Latin letters so the
#: icon never falls back to a tofu box on Windows servers missing fonts.
_GLYPHS: dict[str, str] = {
    "info": "i",
    "warning": "!",
    "error": "x",
    "success": "✓",
}

#: Painted-shadow alpha underneath the pill body.
_SHADOW_ALPHA = 28

#: Maximum visible toast count. Older ones are dropped silently.
_MAX_TOASTS = 6

#: Default auto-dismiss timeout in ms.
_DEFAULT_TIMEOUT_MS = 4500

#: Slide / fade animation duration in ms.
_ANIM_MS = 220

#: Inset from the parent's bottom-right corner (matches the sidebar gutter).
_HOST_INSET = 18

#: Track every live toast so the host can reposition and re-theme them.
#: Weak refs so a deleted toast does not leak the Python wrapper into a
#: ``RuntimeError`` the next time someone walks the list.
_LIVE: list["weakref.ref[Toast]"] = []


# ============================================================================
# Helpers
# ============================================================================

def _hex_to_rgba(token: str, alpha: float | str = 1.0) -> str:
    """Convert a theme colour (``#RRGGBB`` or ``r, g, b``) to a CSS rgba string.

    Theme tokens in this project are hex; some legacy ones are comma strings.
    Both shapes are accepted so the helper is robust to drift in the token file.
    ``alpha`` may be a numeric or a string token (e.g. ``"0.55"``); both are
    formatted with a single-pass normalize so we never depend on Python's
    ``g`` spec which only applies to numbers.
    """
    token = token.strip()
    if token.startswith("#"):
        hex_color = token.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = (int(hex_color[i : i + 2], 16) for i in range(0, 6, 2))
        else:
            return token
    elif "," in token:
        parts = [p.strip() for p in token.split(",")]
        if len(parts) >= 3:
            r, g, b = (int(float(parts[i])) for i in range(3))
        else:
            return token
    else:
        return token
    alpha_str = str(float(alpha))
    return f"rgba({r}, {g}, {b}, {alpha_str})"


def _build_toast_qss(kind: str, tokens: dict[str, str]) -> str:
    """Per-widget QSS for a single toast, picked up by ``setStyleSheet``.

    Object names keep the rules narrow so a toast QSS never leaks onto a
    sibling ``QFrame`` in the same window. The body is a hairline-bordered
    rectangle on the SURFACE tone — no glass blur, no translucent fill —
    matching the rest of the "STUDIO" aesthetic.
    """
    accent = tokens[_KIND_TO_ACCENT.get(kind, "ACCENT")]
    fill = tokens["SURFACE"]
    rim = tokens["BORDER_STRONG"]
    text_colour = tokens["TEXT"]
    muted = tokens["TEXT_MUTED"]
    hover = tokens["SURFACE_HOVER"]
    return (
        f"QFrame#toastPill {{"
        f"  background-color: {fill};"
        f"  border: 1px solid {rim};"
        f"  border-left: 2px solid {accent};"
        f"  border-radius: 6px;"
        f"}}"
        f"QLabel#toastMessage {{"
        f"  color: {text_colour};"
        f"  background: transparent;"
        f"  border: none;"
        f"}}"
        f"QLabel#toastIcon {{"
        f"  color: {accent};"
        f"  font-weight: 700;"
        f"  font-size: 13px;"
        f"  background: transparent;"
        f"  border: none;"
        f"}}"
        f"QPushButton#toastClose {{"
        f"  background: transparent;"
        f"  border: none;"
        f"  color: {muted};"
        f"  border-radius: 4px;"
        f"  font-weight: 700;"
        f"}}"
        f"QPushButton#toastClose:hover {{"
        f"  background: {hover};"
        f"  color: {text_colour};"
        f"}}"
    )


# ============================================================================
# Toast (single pill)
# ============================================================================

class Toast(QFrame):
    """A single pill-shaped toast row.

    Self-paints a radial-gradient shadow underneath its body so it reads as
    lifted off whatever surface it sits on (memory convention). The QSS
    background colour and rim come from :func:`_build_toast_qss`; the icon and
    close ``×`` are styled the same way.
    """

    closed = Signal()

    def __init__(
        self,
        text: str,
        kind: _Kind = "info",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Memory: must call NoFrame before any QSS background or Fusion will
        # paint a hard bevel under the rim.
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("toastPill")
        self._kind: _Kind = kind
        # Set by :meth:`ToastHost._dismiss`; read by the deferred fade-in
        # callbacks so they don't touch a C++ widget that ``deleteLater``
        # has already cleaned up.
        self._deleted = False
        self._build_ui(text)
        self.setMinimumWidth(240)
        self.setMaximumWidth(360)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
        _LIVE.append(weakref.ref(self, lambda _ref: self._remove_dead()))
        self._refresh_style()
        self.destroyed.connect(self._on_destroyed)

    # ------------------------------------------------------------------ build
    def _build_ui(self, text: str) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 9, 8, 9)
        layout.setSpacing(10)

        icon = QLabel(self)
        icon.setObjectName("toastIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedWidth(18)
        icon.setText(_GLYPHS.get(self._kind, "*"))
        layout.addWidget(icon)

        message = QLabel(self)
        message.setObjectName("toastMessage")
        message.setWordWrap(True)
        message.setTextFormat(Qt.TextFormat.PlainText)
        message.setText(text)
        layout.addWidget(message, 1)

        close_btn = QPushButton("x", self)
        close_btn.setObjectName("toastClose")
        close_btn.setFixedSize(22, 22)
        close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setToolTip("关闭")
        close_btn.clicked.connect(self.request_close)
        layout.addWidget(close_btn)

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Memory: painted radial gradient drop shadow, not QGraphicsEffect.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(2, 2, -2, 4)
        if rect.width() > 0 and rect.height() > 0:
            tokens = theme.tokens_for(theme.current_mode())
            accent = tokens[_KIND_TO_ACCENT.get(self._kind, "ACCENT")]
            colour = QColor(accent)
            centre = QPoint(rect.x() + rect.width() // 2, rect.bottom() - 1)
            radius = max(rect.width(), 80) * 0.6
            gradient = QRadialGradient(centre, radius)
            gradient.setColorAt(
                0.0,
                QColor(colour.red(), colour.green(), colour.blue(), _SHADOW_ALPHA),
            )
            gradient.setColorAt(
                1.0,
                QColor(colour.red(), colour.green(), colour.blue(), 0),
            )
            painter.fillRect(self.rect(), gradient)
        painter.end()
        super().paintEvent(event)

    # ------------------------------------------------------------------ API
    def kind(self) -> _Kind:
        return self._kind

    def request_close(self) -> None:
        self.closed.emit()

    def _refresh_style(self) -> None:
        tokens = theme.tokens_for(theme.current_mode())
        self.setStyleSheet(_build_toast_qss(self._kind, tokens))

    @staticmethod
    def refresh_all() -> None:
        """Re-pick theme tokens for every visible toast."""
        for ref in _LIVE:
            toast = ref()
            if toast is None:
                continue
            try:
                toast._refresh_style()  # noqa: SLF001 - intentional internal call
            except RuntimeError:
                # C++ widget deleted between weak-ref resolution and the
                # QSS rebuild — skip silently.
                continue

    def _remove_dead(self) -> None:
        """Prune weak refs whose target wrapper is gone."""
        _LIVE[:] = [r for r in _LIVE if r() is not None]

    def _on_destroyed(self, *_unused) -> None:
        self._remove_dead()


# ============================================================================
# Toast host (stacking region)
# ============================================================================

class ToastHost(QWidget):
    """Bottom-right stacking region that hosts :class:`Toast` rows.

    The host is a child of the main window. It stays out of mouse hit-testing
    outside the toast rows and reflows itself whenever the parent is resized
    or a row is added/removed.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setObjectName("toastHost")

        # Drop ourselves from the global registry once the C++ widget is gone;
        # otherwise subsequent ``install_toast_host`` calls test
        # ``host.parentWidget() is parent`` on a deleted object and crash.
        self.destroyed.connect(self._on_host_destroyed)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        # Bottom stretch so freshly added toasts push the column downward.
        self._layout.addStretch(1)

        self._toasts: list[Toast] = []

    # ------------------------------------------------------------------ API
    def notify(
        self,
        text: str,
        kind: _Kind = "info",
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> Toast:
        toast = Toast(text, kind, self)
        self._toasts.append(toast)
        # Insert above the bottom stretch so new rows accumulate upward.
        self._layout.insertWidget(self._layout.count() - 1, toast)
        toast.show()
        self._animate_in(toast)

        while len(self._toasts) > _MAX_TOASTS:
            self._dismiss(self._toasts[0], animate=False)

        self.reposition()

        toast.closed.connect(lambda t=toast: self._dismiss(t, animate=True))
        if timeout_ms > 0:
            from PySide6.QtCore import QTimer  # local import to avoid cycle

            QTimer.singleShot(timeout_ms, lambda t=toast: self._dismiss(t, animate=True))
        return toast

    def reposition(self) -> None:
        """Pin the host to the bottom-right of its parent and reflow rows."""
        parent = self.parentWidget()
        if parent is None:
            return
        if not self._toasts:
            self.setGeometry(parent.width(), parent.height(), 0, 0)
            return
        max_w = max(min(t.sizeHint().width(), 360) for t in self._toasts)
        max_w = max(max_w, 240)
        row_h = 0
        for t in self._toasts:
            hint = t.sizeHint()
            row_h += hint.height() + 8
        row_h -= 8
        host_w = max_w
        host_h = row_h
        x = parent.width() - host_w - _HOST_INSET
        y = parent.height() - host_h - _HOST_INSET
        self.setGeometry(x, y, host_w, host_h)
        # Resize each row to match the host width so text wraps consistently.
        for t in self._toasts:
            t.setFixedWidth(host_w)

    def _on_host_destroyed(self, *_unused) -> None:
        # Drop the weak entry from the host registry; the prune rebuilds
        # the list without reaching into a deleted object.
        _prune_hosts()

    # ------------------------------------------------------------------ internals
    def _animate_in(self, toast: Toast) -> None:
        """Animate opacity 0→1 + 8px slide-in. ``windowOpacity`` is the cheap
        path; it does not require a graphics effect and shows in the offscreen
        ``ui_preview.py`` screenshots."""
        from PySide6.QtCore import QTimer  # local import to avoid cycle

        if toast._deleted:
            return
        toast.setWindowOpacity(0.0)
        QTimer.singleShot(0, lambda t=toast: self._start_fade_in(t))

    def _start_fade_in(self, toast: Toast) -> None:
        if toast._deleted:
            return
        try:
            anim = QPropertyAnimation(toast, b"windowOpacity", self)
        except RuntimeError:
            # shiboken throws when the underlying C++ object was already
            # deleted by a previous ``deleteLater``; tests trigger this by
            # calling ``notify(..., timeout_ms=0)`` immediately.
            return
        anim.setDuration(_ANIM_MS)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _dismiss(self, toast: Toast, *, animate: bool) -> None:
        if toast not in self._toasts:
            return
        self._toasts.remove(toast)
        toast._deleted = True
        if animate:
            self._start_fade_out(toast)
        else:
            self._layout.removeWidget(toast)
            toast.hide()
            toast.deleteLater()
        self.reposition()

    def _start_fade_out(self, toast: Toast) -> None:
        try:
            anim = QPropertyAnimation(toast, b"windowOpacity", self)
        except RuntimeError:
            self._layout.removeWidget(toast)
            return
        anim.setDuration(_ANIM_MS)
        anim.setStartValue(toast.windowOpacity())
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _finalise() -> None:
            self._layout.removeWidget(toast)
            toast.hide()
            toast.deleteLater()

        anim.finished.connect(_finalise)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


# ============================================================================
# Public install helper
# ============================================================================

#: One host per top-level window. Uses weak refs so a deleted host never
#: trips up the next lookup.
_HOSTS: list["weakref.ref[ToastHost]"] = []


def install_toast_host(parent: QWidget) -> ToastHost:
    """Create a :class:`ToastHost` on ``parent`` and return it.

    Idempotent for the same parent — a second call returns the existing host
    instead of stacking overlays.
    """
    _prune_hosts()
    for ref in _HOSTS:
        host = ref()
        if host is None:
            continue
        if host.parentWidget() is parent:
            return host
    host = ToastHost(parent)
    _HOSTS.append(weakref.ref(host))
    host.setGeometry(parent.width(), parent.height(), 0, 0)
    host.raise_()
    return host


def _prune_hosts() -> None:
    """Remove weak refs whose target has been GC'd.

    Tests rely on the registry staying correct across many creations of
    throw-away parent widgets; without this prune the lookup below would
    touch a deleted C++ object and ``PySide6`` raises ``RuntimeError``.
    """
    _HOSTS[:] = [r for r in _HOSTS if r() is not None]


def notify(
    parent: QWidget,
    text: str,
    kind: _Kind = "info",
    timeout_ms: int = _DEFAULT_TIMEOUT_MS,
) -> Toast:
    """Convenience: install (or reuse) a :class:`ToastHost` on ``parent``
    and show ``text``. Equivalent to ``install_toast_host(parent).notify(...)``.
    """
    return install_toast_host(parent).notify(text, kind, timeout_ms)
