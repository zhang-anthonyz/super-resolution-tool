"""Shared custom widgets for the desktop shell.

Qt drops the native drop-down arrow as soon as a stylesheet touches a
``QComboBox`` (the base style no longer draws ``::down-arrow``), which left
every combo box in the app looking inert. ``ChevronComboBox`` paints the
indicator itself so the affordance survives the stylesheet.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QComboBox, QWidget

from . import theme

#: Horizontal distance from the right edge to the chevron centre.
CHEVRON_INSET = 17


class ChevronComboBox(QComboBox):
    """Combo box with a hand-drawn chevron indicator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        self._paint_chevron()

    def _paint_chevron(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        color = QColor(theme.TEXT_MUTED if self.isEnabled() else theme.TEXT_FAINT)
        pen = QPen(color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        center_x = float(self.width() - CHEVRON_INSET)
        center_y = self.height() / 2.0
        chevron = QPolygonF(
            [
                QPointF(center_x - 4.0, center_y - 2.0),
                QPointF(center_x, center_y + 2.5),
                QPointF(center_x + 4.0, center_y - 2.0),
            ]
        )
        painter.drawPolyline(chevron)
        painter.end()
