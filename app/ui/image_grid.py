"""Image-grid list widget with thumbnails + drag/drop.

Replaces the bare ``QListWidget`` used by the image-upscale tab so the user
sees a 64×64 thumbnail, file name and pixel size per row, plus can drag files
or folders onto the panel to add them.

The list is still a ``QListWidget`` subclass, so the existing API surface
(``addItem``, ``count()``, ``item(index)``, ``selectedItems()``,
``takeItem(row)``, ``setData(UserRole, ...)``) keeps working without any
call-site changes outside the constructor.

Conventions (see project memory *qframe-panel-bevel-and-painted-shadow*):

- Subclassing ``QListWidget`` already avoids the panel-bevel pitfall, but the
  delegate that paints rows also keeps its style widget-local; the entry
  widget stays unaware so a theme switch continues to repaint it via the
  global repolish.
- Drop shadow on the thumbnail frame is **painted** in ``paint`` rather than
  via ``QGraphicsDropShadowEffect``, so the offscreen ``ui_preview.py`` audit
  path captures it correctly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from PIL import Image as _PILImage  # used only to read pixel size

from PySide6.QtCore import (
    QEvent,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from . import icons as icons_mod
from . import theme
from ..core.image_io import orient_source


#: Width allocated to the thumbnail inside each row, in pixels.
_THUMB_PX = 64
#: Inner padding around the row content.
_ROW_PADDING_Y = 8
_ROW_PADDING_X = 12
#: Gap between thumbnail and text block.
_TEXT_INSET = 12
#: Maximum number of characters before a name gets elided.
_NAME_MAX_CHARS = 56

#: Drop highlight ring colour, refreshed when the parent theme changes.
_DROP_RING_LIGHT = "ACCENT"
_DROP_RING_DARK = "ACCENT_SOFT_BORDER"


# ============================================================================
# Thumbnail delegate
# ============================================================================

class ImageThumbnailDelegate(QStyledItemDelegate):
    """Row painter: thumbnail + filename + pixel size, with a drop highlight."""

    def __init__(self, parent_list: "ImageThumbnailList") -> None:
        super().__init__(parent_list)
        self._list = parent_list

    # ------------------------------------------------------------------ API
    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        # Stable height — keeps the list monospaced regardless of name length.
        fm = QFontMetrics(option.font)
        return QSize(0, fm.height() * 2 + _ROW_PADDING_Y * 2)

    def paint(self, painter, option: QStyleOptionViewItem, index) -> None:
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            rect = option.rect
            # Background (selected / hovered / plain).
            painter.fillRect(rect, self._fill_for(option))

            thumb_size = _THUMB_PX
            thumb_rect = QRect(
                rect.left() + _ROW_PADDING_X,
                rect.top() + (rect.height() - thumb_size) // 2,
                thumb_size,
                thumb_size,
            )
            self._paint_thumbnail(painter, thumb_rect, index)
            self._paint_text(painter, thumb_rect, rect, index)

            if option.state & QStyle.StateFlag.State_HasFocus:
                # Don't draw a focus rect on individual rows; the wrapper already
                # gives a clear accent on the active list. Only honour selection.
                pass
        finally:
            painter.restore()

    # ------------------------------------------------------------------ helpers
    def _fill_for(self, option) -> QColor:
        tokens = theme.tokens_for(theme.current_mode())
        if option.state & QStyle.StateFlag.State_Selected:
            return QColor(tokens["ACCENT_SOFT"])
        if option.state & QStyle.StateFlag.State_MouseOver:
            return QColor(tokens["SURFACE_HOVER"])
        return QColor(tokens["SURFACE"])

    def _paint_thumbnail(self, painter, rect, index) -> None:
        pixmap, ok = self._list.thumbnail_for(index)
        tokens = theme.tokens_for(theme.current_mode())
        # Surface-tinted rounded square so the cell still reads as a frame
        # even when the IO fails or the file is still loading.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(tokens["SURFACE_MUTED"]))
        painter.drawRoundedRect(rect, 8, 8)

        if pixmap is None or pixmap.isNull():
            self._paint_placeholder(painter, rect, ok)
        else:
            scaled = pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = rect.left() + (rect.width() - scaled.width()) // 2
            y = rect.top() + (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        # Thin border so the thumb square reads against any backdrop.
        border = QColor(tokens["BORDER"])
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(border, 1))
        painter.drawRoundedRect(rect, 8, 8)

    def _paint_placeholder(self, painter, rect, io_ok) -> None:
        tokens = theme.tokens_for(theme.current_mode())
        painter.setPen(QColor(tokens["TEXT_MUTED"]))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(11)
        painter.setFont(font)
        glyph = "..." if not io_ok else "?"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, glyph)

    def _paint_text(self, painter, thumb_rect, row_rect, index) -> None:
        tokens = theme.tokens_for(theme.current_mode())
        text_left = thumb_rect.right() + _TEXT_INSET
        text_right = row_rect.right() - _ROW_PADDING_X
        if text_right <= text_left:
            return
        name = index.data(Qt.ItemDataRole.DisplayRole) or ""
        size_text = index.data(Qt.ItemDataRole.UserRole + 1) or ""
        painter.setClipRect(QRect(text_left, row_rect.top(), text_right - text_left, row_rect.height()))

        name_font = painter.font()
        name_font.setBold(True)
        name_font.setPointSize(11)
        painter.setFont(name_font)
        painter.setPen(QColor(tokens["TEXT_STRONG"]))
        name_rect = QRect(text_left, row_rect.top() + _ROW_PADDING_Y, text_right - text_left, name_font.pointSize() * 1.6 + 4)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._elide(name, name_rect.width()))

        sub_font = painter.font()
        sub_font.setBold(False)
        sub_font.setPointSize(10)
        painter.setFont(sub_font)
        painter.setPen(QColor(tokens["TEXT_MUTED"]))
        sub_rect = QRect(text_left, name_rect.bottom() + 2, text_right - text_left, sub_font.pointSize() * 1.6 + 4)
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, size_text or "—")
        painter.setClipping(False)

    @staticmethod
    def _elide(text: str, width_px: int) -> str:
        if width_px <= 0:
            return ""
        fm = QFontMetrics(QApplication.instance().font() if QApplication.instance() else QApplication.font())
        return fm.elidedText(text, Qt.TextElideMode.ElideRight, width_px)


# ============================================================================
# Thumbnail list
# ============================================================================

class ImageThumbnailList(QListWidget):
    """Drag-and-drop image list with thumbnails.

    Dropping files or folders onto the panel emits :attr:`paths_dropped` with
    the merged candidates; ``MainWindow`` then folds them through the same
    ``_add_image_paths`` pipeline as the dialog buttons.

    Per-row thumbnail pixmaps are cached by path so the same file added twice
    only reads the disk once.
    """

    #: Emitted with the merged candidate paths whenever a drag-and-drop
    #: gesture completes. The host should treat them as a *set of
    #: suggestions* and route them through whatever deduplication it already
    #: owns (the existing ``_add_image_paths`` code handles this).
    paths_dropped = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setUniformItemSizes(True)
        self.setMinimumHeight(180)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self._delegate = ImageThumbnailDelegate(self)
        self.setItemDelegate(self._delegate)
        self._thumb_cache: dict[str, QPixmap] = {}
        self._size_cache: dict[str, str] = {}
        self._drop_active = False

    # ------------------------------------------------------------------ drop
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_drop_highlight(True)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        # ``QListWidget.dragLeaveEvent`` strictly requires a ``QDragLeaveEvent``
        # so we just route through our drop-highlight toggle without forwarding.
        self._set_drop_highlight(False)

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        paths: list[str] = []
        for url in urls:
            local = url.toLocalFile()
            if local:
                paths.append(local)
        self._set_drop_highlight(False)
        if paths:
            event.acceptProposedAction()
            self.paths_dropped.emit(paths)
        else:
            event.ignoreProposedAction()

    # ------------------------------------------------------------------ API
    def thumbnail_for(self, index) -> tuple[QPixmap | None, bool]:
        """Return ``(pixmap, io_ok)`` for the row at ``index``.

        ``io_ok=False`` means the file is unreadable or absent; the delegate
        paints an ellipsis to make that visible without breaking the row.
        """
        item = self.item(index.row())
        if item is None:
            return None, True
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return None, True
        # Side-load the size into a parallel cache so the delegate can show it
        # without re-reading the file.
        size_text = self._size_cache.get(path)
        if size_text is None:
            width, height = _read_pixel_size(path)
            size_text = (
                f"{Path(path).name}  ·  {width}×{height}" if width and height else Path(path).name
            )
            # The "name" portion goes on the first line via DisplayRole; the
            # "size suffix" is provided separately as UserRole+1.
            item.setData(Qt.ItemDataRole.UserRole + 1, size_text)
            self._size_cache[path] = size_text
        pixmap = self._thumb_cache.get(path)
        if pixmap is None:
            pixmap = _load_thumbnail(path, _THUMB_PX)
            self._thumb_cache[path] = pixmap
        ok = pixmap is not None and not pixmap.isNull()
        return pixmap, ok

    def take_paths(self) -> list[str]:
        """Return every row's path and clear the list.

        Provided so callers can do a one-shot drain without enumerating rows
        manually.
        """
        paths = []
        for index in range(self.count()):
            path = self.item(index).data(Qt.ItemDataRole.UserRole)
            if path:
                paths.append(path)
        self.clear()
        return paths

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        # Keep the size cache around — `takeItem`/etc. don't change it and a
        # re-add should still be fast; only a hard clear drops them.
        self._thumb_cache.clear()
        self._size_cache.clear()

    # ------------------------------------------------------------------ internals
    def _set_drop_highlight(self, on: bool) -> None:
        if self._drop_active == on:
            return
        self._drop_active = on
        self.viewport().update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self.count() == 0:
            self._paint_empty_state()
        if not self._drop_active:
            return
        # Painted drop highlight ring (memory convention, not QGraphicsEffect).
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        ring_token = (
            _DROP_RING_DARK
            if theme.current_mode() == "dark"
            else _DROP_RING_LIGHT
        )
        tokens = theme.tokens_for(theme.current_mode())
        ring_colour = QColor(tokens[ring_token])
        ring_colour.setAlpha(220)
        pen = QPen(ring_colour, 2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pad = 2
        painter.drawRoundedRect(
            self.viewport().rect().adjusted(pad, pad, -pad, -pad),
            10,
            10,
        )
        painter.end()

    def _paint_empty_state(self) -> None:
        """Drop-target affordance for the empty list.

        Before the first file is added this area is the most prominent thing on
        the page, and it was blank: nothing told the user that it accepts a
        drag or which formats are supported. The dashed outline is the standard
        "drop here" cue; the hint mirrors the two ways to fill it.
        """
        viewport = self.viewport()
        rect = viewport.rect()
        if rect.width() < 200 or rect.height() < 120:
            return
        tokens = theme.tokens_for(theme.current_mode())
        painter = QPainter(viewport)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(QColor(tokens["BORDER_STRONG"]), 1.4)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashPattern([4.0, 4.0])
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(10, 10, -10, -10), 10, 10)

            glyph = 34.0
            headline = "拖放图片或文件夹到此处"
            hint = "或点击下方「添加文件 / 添加文件夹」· 支持 jpg / png / webp"
            headline_font = QFont(painter.font())
            headline_font.setBold(True)
            hint_font = QFont(painter.font())
            hint_font.setPointSizeF(max(8.0, hint_font.pointSizeF() - 1.0))
            headline_h = float(painter.fontMetrics().height())
            hint_h = float(QFontMetrics(hint_font).height())
            block_h = glyph + 12.0 + headline_h + 4.0 + hint_h
            top = rect.center().y() - block_h / 2.0

            icons_mod.paint_glyph(
                painter,
                "drop",
                QRect(0, 0, int(glyph), int(glyph)).translated(
                    rect.center().x() - int(glyph) // 2, int(top)
                ),
                tokens["TEXT_MUTED"],
                stroke=1.4,
            )
            painter.setFont(headline_font)
            painter.setPen(QColor(tokens["TEXT_SECONDARY"]))
            painter.drawText(
                QRect(0, int(top + glyph + 12.0), rect.width(), int(headline_h)),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                headline,
            )
            painter.setFont(hint_font)
            painter.setPen(QColor(tokens["TEXT_MUTED"]))
            painter.drawText(
                QRect(
                    0,
                    int(top + glyph + 12.0 + headline_h + 4.0),
                    rect.width(),
                    int(hint_h),
                ),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                hint,
            )
        finally:
            painter.end()


# ============================================================================
# IO helpers
# ============================================================================

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_thumbnail(path: str, side_px: int) -> QPixmap | None:
    try:
        with _PILImage.open(path) as img:
            # Apply the EXIF orientation so a phone photo's thumbnail is not
            # shown lying on its side (the pipeline orients its output too).
            oriented, _exif = orient_source(img)
            oriented.thumbnail((side_px * 2, side_px * 2))
            img = oriented.convert("RGBA")
            data = img.tobytes("raw", "RGBA")
        pix = QPixmap.fromImage(__qt_image_from_buffer__(data, img.size))
        if pix.isNull():
            return None
        return pix
    except Exception:
        return None


def _read_pixel_size(path: str) -> tuple[int | None, int | None]:
    """Return the displayed pixel size (EXIF orientation applied)."""
    try:
        with _PILImage.open(path) as img:
            oriented, _exif = orient_source(img)
            return oriented.size
    except Exception:
        return None, None


def __qt_image_from_buffer__(data: bytes, size) -> "QImage":  # noqa: N802 - private module helper
    from PySide6.QtGui import QImage

    w, h = size
    return QImage(data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
