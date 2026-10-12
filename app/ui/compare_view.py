"""Before/after comparison preview for a finished single-image task.

Two ways to compare the input with the super-resolved output, both driven by the
mouse:

* **像素块对比** — the image is shown fit-to-view with a marker under the cursor,
  and a strip below shows the same small region from *both* images side by side,
  so the difference is visible pixel by pixel while the mouse moves.
* **垂直滑块对比** — one aligned view with a draggable vertical divider: the left
  side shows the original, the right side the super-resolved result.

Both views draw the original scaled into the output's coordinate space (the
usual way SR results are presented: "how does this beat a plain resize?"), and
that mapping telescopes to a single uniform ratio per axis, so the two sides
stay pixel-aligned whatever the output size ended up being.

Preview images are capped in memory: a 4x output of a large photo can reach tens
of megapixels, and two of those as QImage would cost hundreds of megabytes. Past
the cap both images are scaled by the same factor and the view says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, QSize, QSizeF, Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .widgets import ChevronComboBox

#: Preview budget per image (QImage costs 3 bytes per pixel here).
MAX_PREVIEW_PIXELS = 20_000_000

MODE_BLOCK = "block"
MODE_WIPE = "wipe"

#: Half-width of the grab zone of the vertical divider, in view pixels.
SPLITTER_GRAB_PX = 10

#: Zoom limits, as multiples of the fit-to-view scale. Below 1 the image is
#: smaller than the view (a small 2x output can be shown at true 1:1 that way).
MIN_ZOOM = 0.1
MAX_ZOOM = 16.0

#: Height share of the sample strip in block mode, clamped so the strip stays
#: readable in a short window and never eats the image in a tall one.
_STRIP_RATIO = 0.34
_STRIP_MIN_HEIGHT = 150.0
_STRIP_MAX_RATIO = 0.5

#: Pixels of margin around the fitted image.
_VIEW_MARGIN = 6


@dataclass(frozen=True)
class PreviewPair:
    """Input/output images prepared for display."""

    input_path: Path
    output_path: Path
    original: QImage
    output: QImage
    original_true_size: tuple[int, int]
    output_true_size: tuple[int, int]
    preview_scale: float  # 1.0 when nothing had to be downscaled

    @property
    def scaled_note(self) -> str:
        if self.preview_scale >= 0.999:
            return "1:1 预览"
        return f"预览已缩放至 {self.preview_scale * 100:.0f}%（原图 {self.input_true_text}）"

    @property
    def input_true_text(self) -> str:
        return f"{self.original_true_size[0]}×{self.original_true_size[1]}"

    @property
    def output_true_text(self) -> str:
        return f"{self.output_true_size[0]}×{self.output_true_size[1]}"


def _load_qimage(path: Path, max_pixels: int) -> tuple[QImage, tuple[int, int], float]:
    """Load one image for display, downscaling above the memory budget.

    Pillow does the decoding (it already handles every format the pipeline can
    write, without depending on Qt's image plugins being bundled).
    """
    with Image.open(path) as source:
        image = source.convert("RGB")
        true_size = (image.width, image.height)
        scale = 1.0
        pixels = image.width * image.height
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            # Flooring both edges keeps width * height <= max_pixels.
            target = (
                max(1, int(image.width * scale)),
                max(1, int(image.height * scale)),
            )
            image = image.resize(target, Image.Resampling.LANCZOS)
            scale = image.width / true_size[0]
        array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))

    height, width = array.shape[:2]
    # .copy() detaches the QImage from the numpy buffer, which dies here.
    qimage = QImage(
        array.data, width, height, 3 * width, QImage.Format.Format_RGB888
    ).copy()
    return qimage, true_size, scale


def load_preview_pair(
    input_path: str | Path,
    output_path: str | Path,
    max_pixels: int = MAX_PREVIEW_PIXELS,
) -> PreviewPair:
    """Decode both images for the comparison view."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    original, original_size, original_scale = _load_qimage(input_path, max_pixels)
    output, output_size, output_scale = _load_qimage(output_path, max_pixels)
    return PreviewPair(
        input_path=input_path,
        output_path=output_path,
        original=original,
        output=output,
        original_true_size=original_size,
        output_true_size=output_size,
        preview_scale=min(original_scale, output_scale),
    )


class PreviewLoadWorker(QThread):
    """Loads a preview pair off the GUI thread (a 4x PNG can take seconds)."""

    loaded = Signal(object, object)  # (token, PreviewPair)
    failed = Signal(object, str)  # (token, message)

    def __init__(self, token, input_path: Path, output_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._token = token
        self._input_path = Path(input_path)
        self._output_path = Path(output_path)

    def run(self) -> None:
        try:
            pair = load_preview_pair(self._input_path, self._output_path)
        except Exception as exc:  # noqa: BLE001 - reported to the run log
            self.failed.emit(self._token, str(exc) or exc.__class__.__name__)
            return
        self.loaded.emit(self._token, pair)


class ComparisonView(QWidget):
    """Interactive input/output comparison widget (block strip or wipe)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._pair: Optional[PreviewPair] = None
        self._mode = MODE_BLOCK
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._split = 0.5
        self._cursor = QPointF(0.0, 0.0)
        self._cursor_inside = False
        self._dragging: Optional[str] = None
        self._drag_origin = QPointF(0.0, 0.0)
        self._pan_origin = QPointF(0.0, 0.0)

        self._output_layer: Optional[QPixmap] = None
        self._original_layer: Optional[QPixmap] = None
        self._layer_origin = QPointF(0.0, 0.0)
        self._layer_key = None

    # ---- state ---------------------------------------------------------------

    def has_pair(self) -> bool:
        return self._pair is not None

    def set_pair(self, pair: Optional[PreviewPair]) -> None:
        self._pair = pair
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        # Start on the image centre so the strip has something to show before
        # the first mouse move.
        self._cursor = (
            QPointF(pair.output.width() / 2.0, pair.output.height() / 2.0)
            if pair is not None
            else QPointF(0.0, 0.0)
        )
        self._cursor_inside = False
        self._invalidate_layers()
        self.update()

    def pair(self) -> Optional[PreviewPair]:
        return self._pair

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_BLOCK, MODE_WIPE):
            raise ValueError(f"unknown comparison mode: {mode}")
        if mode == self._mode:
            return
        self._mode = mode
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._invalidate_layers()
        self.setCursor(
            Qt.CursorShape.SplitHCursor if mode == MODE_WIPE else Qt.CursorShape.CrossCursor
        )
        self.update()

    def split_ratio(self) -> float:
        return self._split

    def set_split_ratio(self, ratio: float) -> None:
        self._split = max(0.0, min(1.0, float(ratio)))
        self.update()

    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))
        self._invalidate_layers()
        self.update()

    def fit_to_view(self) -> None:
        """Show the whole image (zoom factor 1)."""
        self._pan = QPointF(0.0, 0.0)
        self.set_zoom(1.0)

    def one_to_one(self) -> None:
        """Show output pixels 1:1 (one image pixel per screen pixel)."""
        self._pan = QPointF(0.0, 0.0)
        self.set_zoom(1.0 / max(self._fit_scale(), 1e-6))

    def sample_point(self) -> QPointF:
        """Cursor position in output-image pixels (the point being compared)."""
        return QPointF(self._cursor)

    # ---- geometry ------------------------------------------------------------

    def strip_height(self) -> float:
        """Height of the sample strip in block mode."""
        height = float(self.rect().height())
        if self._mode != MODE_BLOCK:
            return 0.0
        return min(height * _STRIP_MAX_RATIO, max(_STRIP_MIN_HEIGHT, height * _STRIP_RATIO))

    def view_rect(self) -> QRectF:
        """Rect holding the main image area (the strip takes the bottom in block)."""
        rect = QRectF(self.rect())
        if self._mode == MODE_BLOCK:
            rect.setHeight(max(1.0, rect.height() - self.strip_height()))
        return rect

    def strip_rect(self) -> QRectF:
        rect = QRectF(self.rect())
        rect.setTop(self.view_rect().bottom())
        return rect

    def _fit_scale(self) -> float:
        """Scale that fits the whole output image into the view area."""
        if self._pair is None or self._pair.output.isNull():
            return 1.0
        area = self.view_rect().adjusted(
            _VIEW_MARGIN, _VIEW_MARGIN, -_VIEW_MARGIN, -_VIEW_MARGIN
        )
        if area.width() <= 1 or area.height() <= 1:
            return 1.0
        return min(
            area.width() / self._pair.output.width(),
            area.height() / self._pair.output.height(),
        )

    def image_scale(self) -> float:
        return self._fit_scale() * self._zoom

    def image_rect(self) -> QRectF:
        """Where the output image lands in widget coordinates."""
        if self._pair is None:
            return QRectF()
        scale = self.image_scale()
        width = self._pair.output.width() * scale
        height = self._pair.output.height() * scale
        area = self.view_rect()
        return QRectF(
            area.left() + (area.width() - width) / 2.0 + self._pan.x(),
            area.top() + (area.height() - height) / 2.0 + self._pan.y(),
            width,
            height,
        )

    def viewport_to_output(self, point: QPointF) -> QPointF:
        """Map a widget point to output-image pixels (may fall outside)."""
        if self._pair is None:
            return QPointF(0.0, 0.0)
        rect = self.image_rect()
        scale = self.image_scale()
        if scale <= 0:
            return QPointF(0.0, 0.0)
        return QPointF(
            (point.x() - rect.left()) / scale,
            (point.y() - rect.top()) / scale,
        )

    def clamp_to_output(self, point: QPointF) -> QPointF:
        if self._pair is None:
            return QPointF(0.0, 0.0)
        return QPointF(
            min(max(point.x(), 0.0), self._pair.output.width() - 1.0),
            min(max(point.y(), 0.0), self._pair.output.height() - 1.0),
        )

    def original_ratio(self) -> QPointF:
        """Output-image pixel -> original-image pixel factor (per axis)."""
        if self._pair is None or self._pair.output.isNull():
            return QPointF(1.0, 1.0)
        return QPointF(
            self._pair.original.width() / self._pair.output.width(),
            self._pair.original.height() / self._pair.output.height(),
        )

    def sample_block(self, cell: QSizeF) -> QRectF:
        """Region of the output image shown in one strip cell, in image pixels.

        The cell is ``_zoom`` times larger than the region it shows, so the
        strip magnifies the output's own pixels.
        """
        if self._pair is None or self._zoom <= 0 or cell.isEmpty():
            return QRectF()
        width = max(16.0, min(float(self._pair.output.width()), cell.width() / self._zoom))
        height = max(16.0, min(float(self._pair.output.height()), cell.height() / self._zoom))
        left = min(
            max(self._cursor.x() - width / 2.0, 0.0),
            max(0.0, self._pair.output.width() - width),
        )
        top = min(
            max(self._cursor.y() - height / 2.0, 0.0),
            max(0.0, self._pair.output.height() - height),
        )
        return QRectF(left, top, width, height)

    # ---- painting ------------------------------------------------------------

    def _invalidate_layers(self) -> None:
        self._layer_key = None
        self._output_layer = None
        self._original_layer = None

    def _visible_source_rect(self) -> QRectF:
        """Output-image pixels that can land inside the view area.

        Rendering only this region keeps the cached pixmap at viewport size no
        matter how far the user zooms in.
        """
        assert self._pair is not None
        scale = self.image_scale()
        area = self.view_rect()
        rect = self.image_rect()
        left = (area.left() - rect.left()) / scale
        top = (area.top() - rect.top()) / scale
        width = area.width() / scale
        height = area.height() / scale
        return QRectF(left, top, width, height).intersected(
            QRectF(self._pair.output.rect())
        )

    def _render_layers(self) -> None:
        """Cache the visible slice of both images for fast repaints."""
        key = (
            self.rect().size().toTuple(),
            round(self._zoom, 4),
            round(self._pan.x(), 2),
            round(self._pan.y(), 2),
            self._mode,
            id(self._pair),
        )
        if self._pair is None or key == self._layer_key:
            return
        source = self._visible_source_rect()
        scale = self.image_scale()
        rect = self.image_rect()
        # Put the slice back where it belongs in widget coordinates.
        target = QRectF(
            rect.left() + source.left() * scale,
            rect.top() + source.top() * scale,
            source.width() * scale,
            source.height() * scale,
        )
        ratio = self.original_ratio()
        original_source = QRectF(
            source.left() * ratio.x(),
            source.top() * ratio.y(),
            source.width() * ratio.x(),
            source.height() * ratio.y(),
        ).intersected(QRectF(self._pair.original.rect()))
        size = QSize(max(1, int(target.width())), max(1, int(target.height())))
        for name, image, region in (
            ("_output_layer", self._pair.output, source),
            ("_original_layer", self._pair.original, original_source),
        ):
            pixmap = QPixmap(size)
            pixmap.fill(QColor(theme.SURFACE_MUTED))
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(QRectF(0, 0, size.width(), size.height()), image, region)
            painter.end()
            setattr(self, name, pixmap)
        self._layer_origin = target.topLeft()
        self._layer_key = key

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.SURFACE_MUTED))
        if self._pair is None:
            painter.setPen(QColor(theme.TEXT_SECONDARY))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "任务完成后这里显示原图与超分结果的对比预览",
            )
            painter.end()
            return

        self._render_layers()
        if self._mode == MODE_WIPE:
            self._paint_wipe(painter)
        else:
            self._paint_block(painter)
        painter.end()

    def _paint_wipe(self, painter: QPainter) -> None:
        rect = self.image_rect()
        if self._output_layer is not None:
            painter.drawPixmap(self._layer_origin, self._output_layer)
        split_x = self.view_rect().left() + self.view_rect().width() * self._split
        # Left of the divider: the original, rendered from the matching source
        # slice so both sides stay aligned pixel for pixel.
        painter.save()
        painter.setClipRect(
            QRectF(self.view_rect().left(), rect.top(), split_x - self.view_rect().left(),
                   rect.height())
        )
        if self._original_layer is not None:
            painter.drawPixmap(self._layer_origin, self._original_layer)
        painter.restore()

        painter.setPen(QPen(QColor("#ffffff"), 3))
        painter.drawLine(
            QPointF(split_x, self.view_rect().top()),
            QPointF(split_x, self.view_rect().bottom()),
        )
        painter.setPen(QPen(QColor(theme.ACCENT), 1.5))
        painter.drawLine(
            QPointF(split_x, self.view_rect().top()),
            QPointF(split_x, self.view_rect().bottom()),
        )
        self._draw_splitter_handle(painter, split_x)
        self._draw_caption(painter, self.view_rect().left() + 10, "原图", Qt.AlignmentFlag.AlignLeft)
        self._draw_caption(
            painter, self.view_rect().right() - 10, "超分输出", Qt.AlignmentFlag.AlignRight
        )

    def _draw_splitter_handle(self, painter: QPainter, x: float) -> None:
        rect = self.view_rect()
        center = QPointF(x, rect.center().y())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.ACCENT))
        painter.drawEllipse(center, 11.0, 11.0)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        for offset in (-3.5, 3.5):
            painter.drawLine(
                QPointF(center.x() + offset, center.y() - 4.5),
                QPointF(center.x() + offset, center.y() + 4.5),
            )

    def _paint_block(self, painter: QPainter) -> None:
        rect = self.image_rect()
        if self._output_layer is not None:
            painter.drawPixmap(self._layer_origin, self._output_layer)

        strip = self.strip_rect()
        painter.fillRect(strip, QColor(theme.SURFACE))
        half = strip.width() / 2.0
        left_cell = QRectF(strip.left(), strip.top(), half, strip.height())
        right_cell = QRectF(strip.left() + half, strip.top(), half, strip.height())
        block = self.sample_block(left_cell.size())

        # Marker showing which output pixels the strip below is showing.
        scale = self.image_scale()
        marker = QRectF(
            rect.left() + block.left() * scale,
            rect.top() + block.top() * scale,
            max(2.0, block.width() * scale),
            max(2.0, block.height() * scale),
        )
        painter.setPen(QPen(QColor(theme.ACCENT), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(marker)

        painter.save()
        painter.setClipRect(right_cell)
        painter.drawPixmap(right_cell.topLeft(), self._crop_output(block, right_cell.size()))
        painter.restore()
        painter.save()
        painter.setClipRect(left_cell)
        painter.drawPixmap(left_cell.topLeft(), self._crop_original(block, left_cell.size()))
        painter.restore()

        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.drawLine(
            QPointF(left_cell.right(), strip.top()),
            QPointF(left_cell.right(), strip.bottom()),
        )
        self._draw_caption(painter, left_cell.left() + 8, "原图（同尺寸）", Qt.AlignmentFlag.AlignLeft, in_strip=True)
        self._draw_caption(
            painter, right_cell.right() - 8, "超分输出", Qt.AlignmentFlag.AlignRight, in_strip=True
        )
        if self._pair is not None:
            xy = f"输出坐标 {int(self._cursor.x())}, {int(self._cursor.y())} · 取样 {int(block.width())}×{int(block.height())} px · 放大 {self._zoom:g}x"
            self._draw_caption(
                painter,
                strip.left() + 8,
                xy,
                Qt.AlignmentFlag.AlignLeft,
                in_strip=True,
                bottom=True,
            )

    def _crop_output(self, block: QRectF, size) -> QPixmap:
        return self._crop(self._pair.output, block, size)

    def _crop_original(self, block: QRectF, size) -> QPixmap:
        ratio = self.original_ratio()
        source = QRectF(
            block.left() * ratio.x(),
            block.top() * ratio.y(),
            block.width() * ratio.x(),
            block.height() * ratio.y(),
        )
        source = source.intersected(QRectF(self._pair.original.rect()))
        return self._crop(self._pair.original, source, size)

    @staticmethod
    def _crop(image: QImage, region: QRectF, size) -> QPixmap:
        target = QSize(
            max(1, int(round(size.width()))), max(1, int(round(size.height())))
        )
        pixmap = QPixmap(target)
        pixmap.fill(QColor(theme.SURFACE_MUTED))
        if region.width() < 1 or region.height() < 1:
            return pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(QRectF(0, 0, target.width(), target.height()), image, region)
        painter.end()
        return pixmap

    def _draw_caption(
        self,
        painter: QPainter,
        x: float,
        text: str,
        alignment,
        in_strip: bool = False,
        bottom: bool = False,
    ) -> None:
        area = self.strip_rect() if in_strip else self.view_rect()
        y = area.bottom() - 22 if bottom else area.top() + 22
        font = QFont(painter.font())
        font.setPointSizeF(max(8.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        box = QRectF(x - 120, y - 16, 240, 20)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 110))
        painter.drawRoundedRect(box, 6.0, 6.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(box, alignment | Qt.AlignmentFlag.AlignVCenter, text)

    # ---- interaction ---------------------------------------------------------

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        position = QPointF(event.position())
        if self._dragging == "split":
            span = max(1.0, self.view_rect().width())
            self.set_split_ratio((position.x() - self.view_rect().left()) / span)
            return
        if self._dragging == "pan":
            delta = position - self._drag_origin
            self._pan = self._pan_origin + delta
            self._invalidate_layers()
            self.update()
            return

        if self._mode == MODE_WIPE:
            self._update_hover_cursor(position)
            return
        inside = self.view_rect().contains(position)
        self._cursor_inside = inside
        if inside:
            self._cursor = self.clamp_to_output(self.viewport_to_output(position))
        self.update()

    def _update_hover_cursor(self, position: QPointF) -> None:
        near_split = abs(position.x() - (self.view_rect().left() + self.view_rect().width() * self._split)) <= SPLITTER_GRAB_PX
        self.setCursor(
            Qt.CursorShape.SplitHCursor if near_split else Qt.CursorShape.OpenHandCursor
        )

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._mode == MODE_BLOCK:
            self._cursor_inside = False
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._pair is None or event.button() != Qt.MouseButton.LeftButton:
            return
        position = QPointF(event.position())
        if self._mode == MODE_WIPE:
            split_x = self.view_rect().left() + self.view_rect().width() * self._split
            if abs(position.x() - split_x) <= SPLITTER_GRAB_PX:
                self._dragging = "split"
                self.setCursor(Qt.CursorShape.SplitHCursor)
                return
            self._dragging = "pan"
            self._drag_origin = position
            self._pan_origin = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._dragging = None
        if self._mode == MODE_WIPE:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._pair is None:
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        self.set_zoom(self._zoom * (1.25 ** steps))
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._invalidate_layers()
        super().resizeEvent(event)


class ComparisonPanel(QWidget):
    """Mode selector + status line + :class:`ComparisonView`.

    One panel sits in the image tab, another inside :class:`ComparisonDialog`;
    the two are kept in step by the window (see ``_sync_compare_panels``).
    """

    mode_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(8)
        caption = QLabel("对比方式")
        caption.setObjectName("fieldLabel")
        self.mode_combo = ChevronComboBox()
        self.mode_combo.addItem("像素块对比", MODE_BLOCK)
        self.mode_combo.addItem("垂直滑块对比", MODE_WIPE)
        self.mode_combo.setMinimumWidth(150)
        self.status = QLabel("等待任务完成")
        self.status.setObjectName("hintLabel")
        self.status.setWordWrap(True)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        row.addWidget(caption)
        row.addWidget(self.mode_combo)
        row.addWidget(self.status, 1)
        row.addLayout(self.actions)
        layout.addLayout(row)

        self.view = ComparisonView()
        self.view.setMinimumHeight(300)
        layout.addWidget(self.view, 1)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_index_changed)

    def _on_mode_index_changed(self) -> None:
        mode = str(self.mode_combo.currentData() or MODE_BLOCK)
        self.view.set_mode(mode)
        self.mode_changed.emit(mode)

    def add_action(self, button: QWidget) -> None:
        """Append a host-provided toolbar button (fit / 1:1 / close / …)."""
        self.actions.addWidget(button)

    def set_mode(self, mode: str) -> None:
        index = self.mode_combo.findData(mode)
        if index >= 0 and index != self.mode_combo.currentIndex():
            self.mode_combo.setCurrentIndex(index)

    def set_summary(self, text: str) -> None:
        self.status.setText(text)

    def summary(self) -> str:
        return self.status.text()

    def set_pair(self, pair: Optional[PreviewPair]) -> None:
        self.view.set_pair(pair)


class ComparisonDialog(QDialog):
    """Big, resizable window for comfortable pixel peeping.

    The inline card stays available for a quick look; this window is what you
    open when you actually want to study the difference (it remembers its
    geometry, and can be maximised like any other window).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("对比预览 · 原图 / 超分输出")
        self.setSizeGripEnabled(True)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        self.panel = ComparisonPanel(self)
        layout.addWidget(self.panel, 1)
        self.panel.add_action(ghost_button("适应窗口", self.panel.view.fit_to_view))
        self.panel.add_action(ghost_button("1:1 像素", self.panel.view.one_to_one))
        self.panel.add_action(ghost_button("最大化", self.showMaximized))
        self.panel.add_action(ghost_button("关闭", self.close))
        self.resize(1360, 900)

    def view(self) -> ComparisonView:
        return self.panel.view


def ghost_button(text: str, slot, parent: Optional[QWidget] = None) -> QPushButton:
    """Secondary button matching the app's theme."""
    button = QPushButton(text, parent)
    button.setObjectName("ghostButton")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.clicked.connect(slot)
    return button


def format_pair_summary(pair: PreviewPair) -> str:
    """One log line describing what the preview is comparing."""
    return (
        f"对比预览：{pair.input_path.name}（{pair.input_true_text}）→ "
        f"{pair.output_path.name}（{pair.output_true_text}）· {pair.scaled_note}"
    )


__all__ = [
    "MAX_PREVIEW_PIXELS",
    "MODE_BLOCK",
    "MODE_WIPE",
    "ComparisonDialog",
    "ComparisonPanel",
    "ComparisonView",
    "PreviewLoadWorker",
    "format_pair_summary",
    "ghost_button",
    "PreviewPair",
    "format_pair_summary",
    "load_preview_pair",
]
