"""Before/after comparison preview for a finished single-image task.

Two ways to compare, both modelled on Upscayl's comparison views:

* **放大镜对比** (``MODE_LENS``) — the output is shown fit-to-view and the mouse
  is the magnifier: a marker marks the spot under the cursor, and two magnified
  cells float right next to the cursor showing the same region from the original
  and from the super-resolved output. The eye never leaves the cursor, which is
  what makes pixel peeping work on a big image: there is no separate strip to
  look at, and the lens can be flipped above the cursor near the bottom edge.
* **滑块对比** (``MODE_SLIDER``) — one aligned view with a draggable vertical
  divider: the left side shows the original, the right side the output. The
  wheel zooms around the cursor, dragging elsewhere pans, and the pan is clamped
  so the image can never be dragged out of sight.

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
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import theme

#: Preview budget per image (QImage costs 3 bytes per pixel here).
MAX_PREVIEW_PIXELS = 20_000_000

MODE_LENS = "lens"
MODE_SLIDER = "slider"

#: Mode entries in display order, shared by the mode switch and the window.
MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    (MODE_LENS, "放大镜对比"),
    (MODE_SLIDER, "滑块对比"),
)

#: Names used before the lens/slider rework; settings files may still hold them.
_LEGACY_MODES = {"block": MODE_LENS, "wipe": MODE_SLIDER}

#: Half-width of the grab zone of the vertical divider, in view pixels.
SPLITTER_GRAB_PX = 10

#: Zoom limits, as multiples of the fit-to-view scale. Below 1 the image is
#: smaller than the view (a small 2x output can be shown at true 1:1 that way).
MIN_ZOOM = 0.1
MAX_ZOOM = 10.0

#: Lens magnification is "output pixels per cell pixel" — 4 means every output
#: pixel is drawn 4x4 view pixels, the same factor Upscayl's lens uses.
LENS_DEFAULT_ZOOM = 4.0
LENS_MIN_ZOOM = 1.0
LENS_MAX_ZOOM = 10.0

#: Gap between the two lens cells, and their size limits in view pixels.
LENS_GAP_PX = 8.0
LENS_CELL_MIN_PX = 88.0
LENS_CELL_MAX_PX = 184.0

#: Below this the cells would be too small to read; the marker is all that is
#: drawn (cramped windows).
LENS_CELL_DRAW_MIN_PX = 56.0

#: Vertical distance from the cursor to the lens cells.
LENS_OFFSET_PX = 26.0

#: The marker may shrink below this when the sampled region is tiny on screen.
LENS_MARKER_MIN_PX = 24.0

#: The zoom slider works in percent of the fit-to-view scale; every step is the
#: same 1x..10x range in both modes (lens magnification / view zoom).
ZOOM_SLIDER_MIN = 100
ZOOM_SLIDER_MAX = 1000
ZOOM_SLIDER_STEP = 10

#: Pixels of margin around the fitted image.
_VIEW_MARGIN = 6

HINT_LENS = "把鼠标移到图上：放大镜跟随光标，同步放大原图与输出"
HINT_SLIDER = "拖动分割线左右对照 · 拖其它位置平移 · 滚轮以光标为中心缩放 · 双击复位"


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
    """Interactive input/output comparison canvas (lens or slider)."""

    #: Emitted when the mode, the view zoom or the lens magnification changed,
    #: so host controls (the zoom slider) can follow.
    view_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._pair: Optional[PreviewPair] = None
        self._mode = MODE_LENS
        self._zoom = 1.0
        self._lens_zoom = LENS_DEFAULT_ZOOM
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
        # Start on the image centre so the lens has something to show before
        # the first mouse move.
        self._cursor = (
            QPointF(pair.output.width() / 2.0, pair.output.height() / 2.0)
            if pair is not None
            else QPointF(0.0, 0.0)
        )
        self._cursor_inside = False
        self._invalidate_layers()
        self.update()
        self.view_changed.emit()

    def pair(self) -> Optional[PreviewPair]:
        return self._pair

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        mode = _LEGACY_MODES.get(str(mode), str(mode))
        if mode not in (MODE_LENS, MODE_SLIDER):
            raise ValueError(f"unknown comparison mode: {mode}")
        if mode == self._mode:
            return
        self._mode = mode
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._invalidate_layers()
        self._apply_mode_cursor()
        self.update()
        self.view_changed.emit()

    def _apply_mode_cursor(self) -> None:
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if self._mode == MODE_LENS
            else Qt.CursorShape.OpenHandCursor
        )

    def split_ratio(self) -> float:
        return self._split

    def set_split_ratio(self, ratio: float) -> None:
        self._split = max(0.0, min(1.0, float(ratio)))
        self.update()

    def zoom(self) -> float:
        """View zoom in slider mode, as a multiple of the fit-to-view scale."""
        return self._zoom

    def set_zoom(self, zoom: float) -> None:
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))
        if zoom == self._zoom:
            return
        self._zoom = zoom
        self._clamp_pan()
        self._invalidate_layers()
        self.update()
        self.view_changed.emit()

    def lens_zoom(self) -> float:
        """Lens magnification: output pixels per lens-cell pixel."""
        return self._lens_zoom

    def set_lens_zoom(self, zoom: float) -> None:
        zoom = max(LENS_MIN_ZOOM, min(LENS_MAX_ZOOM, float(zoom)))
        if zoom == self._lens_zoom:
            return
        self._lens_zoom = zoom
        self.update()
        self.view_changed.emit()

    def fit_to_view(self) -> None:
        """Show the whole image (zoom factor 1)."""
        self._pan = QPointF(0.0, 0.0)
        self._zoom = 1.0
        self._clamp_pan()
        self._invalidate_layers()
        self.update()
        self.view_changed.emit()

    def one_to_one(self) -> None:
        """Show output pixels 1:1 (one image pixel per screen pixel)."""
        self._pan = QPointF(0.0, 0.0)
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, 1.0 / max(self._fit_scale(), 1e-6)))
        self._clamp_pan()
        self._invalidate_layers()
        self.update()
        self.view_changed.emit()

    def zoom_at(self, point: QPointF, factor: float) -> None:
        """Zoom so the output pixel under ``point`` stays under ``point``."""
        if self._pair is None:
            return
        before = self.viewport_to_output(point)
        self.set_zoom(self._zoom * factor)
        scale = self.image_scale()
        after = self.viewport_to_output(point)
        self._pan = QPointF(
            self._pan.x() + (after.x() - before.x()) * scale,
            self._pan.y() + (after.y() - before.y()) * scale,
        )
        self._clamp_pan()
        self._invalidate_layers()
        self.update()

    def pan(self) -> QPointF:
        return QPointF(self._pan)

    def sample_point(self) -> QPointF:
        """Cursor position in output-image pixels (the point being compared)."""
        return QPointF(self._cursor)

    # ---- geometry ------------------------------------------------------------

    def view_rect(self) -> QRectF:
        """Rect holding the image canvas."""
        return QRectF(self.rect())

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

    def _pan_limits(self) -> QPointF:
        """How far the image may be panned before a gap would open in the view."""
        if self._pair is None or self._pair.output.isNull():
            return QPointF(0.0, 0.0)
        scale = self.image_scale()
        area = self.view_rect()
        return QPointF(
            max(0.0, (self._pair.output.width() * scale - area.width()) / 2.0),
            max(0.0, (self._pair.output.height() * scale - area.height()) / 2.0),
        )

    def _clamp_pan(self) -> None:
        """Keep the image in reach: no panning it away from the view."""
        limits = self._pan_limits()
        self._pan = QPointF(
            min(max(self._pan.x(), -limits.x()), limits.x()),
            min(max(self._pan.y(), -limits.y()), limits.y()),
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

    def sample_block(self, cell: QSizeF, zoom: Optional[float] = None) -> QRectF:
        """Region of the output image shown in one cell, in image pixels.

        The cell is ``zoom`` times larger than the region it shows, so the cell
        magnifies the output's own pixels. Defaults to the lens magnification.
        """
        if self._pair is None or cell.isEmpty():
            return QRectF()
        factor = self._lens_zoom if zoom is None else max(float(zoom), 1e-6)
        if factor <= 0:
            return QRectF()
        width = max(16.0, min(float(self._pair.output.width()), cell.width() / factor))
        height = max(16.0, min(float(self._pair.output.height()), cell.height() / factor))
        left = min(
            max(self._cursor.x() - width / 2.0, 0.0),
            max(0.0, self._pair.output.width() - width),
        )
        top = min(
            max(self._cursor.y() - height / 2.0, 0.0),
            max(0.0, self._pair.output.height() - height),
        )
        return QRectF(left, top, width, height)

    def lens_cell_size(self) -> float:
        """Side of one magnified lens cell, in view pixels."""
        area = self.view_rect()
        inner = min(
            area.width() - 2 * _VIEW_MARGIN, area.height() - 2 * _VIEW_MARGIN
        )
        if inner <= 0:
            return 0.0
        cell = min(LENS_CELL_MAX_PX, max(LENS_CELL_MIN_PX, inner * 0.42))
        # Two cells plus the gap must fit the view on both axes.
        cell = min(cell, (area.width() - 2 * _VIEW_MARGIN - LENS_GAP_PX) / 2.0)
        cell = min(cell, inner)
        return max(0.0, cell)

    def lens_panels(self) -> tuple[QRectF, QRectF]:
        """The two magnified cells (original, output) in widget coordinates.

        They sit just past the marker like Upscayl's lens, flip above it when
        the cursor is near the bottom edge, and stay fully inside the view.
        """
        cell = self.lens_cell_size()
        if cell < 1.0:
            return QRectF(), QRectF()
        area = self.view_rect()
        marker = self.lens_marker_rect()
        total = 2.0 * cell + LENS_GAP_PX
        left = min(
            max(self._cursor.x() - total / 2.0, area.left() + _VIEW_MARGIN),
            area.right() - _VIEW_MARGIN - total,
        )
        # Offset from the marker, not the cursor: the marker's height depends on
        # the zoom, and the cells must never sit on top of it.
        top = marker.bottom() + LENS_OFFSET_PX
        if top + cell > area.bottom() - _VIEW_MARGIN:
            top = marker.top() - LENS_OFFSET_PX - cell
        top = min(
            max(top, area.top() + _VIEW_MARGIN),
            area.bottom() - _VIEW_MARGIN - cell,
        )
        return (
            QRectF(left, top, cell, cell),
            QRectF(left + cell + LENS_GAP_PX, top, cell, cell),
        )

    def lens_marker_rect(self) -> QRectF:
        """On-screen rectangle of the output pixels the lens magnifies."""
        if self._pair is None:
            return QRectF()
        cell = self.lens_cell_size()
        if cell < 1.0:
            return QRectF()
        block = self.sample_block(QSizeF(cell, cell))
        scale = self.image_scale()
        image = self.image_rect()
        marker = QRectF(
            image.left() + block.left() * scale,
            image.top() + block.top() * scale,
            block.width() * scale,
            block.height() * scale,
        )
        # Keep a tiny sampled region visible, and the marker over the image.
        for axis in ("x", "y"):
            size = marker.width() if axis == "x" else marker.height()
            if size < LENS_MARKER_MIN_PX:
                grow = (LENS_MARKER_MIN_PX - size) / 2.0
                if axis == "x":
                    marker.setLeft(marker.left() - grow)
                    marker.setWidth(LENS_MARKER_MIN_PX)
                else:
                    marker.setTop(marker.top() - grow)
                    marker.setHeight(LENS_MARKER_MIN_PX)
        left = min(
            max(marker.left(), image.left()),
            max(image.left(), image.right() - marker.width()),
        )
        top = min(
            max(marker.top(), image.top()),
            max(image.top(), image.bottom() - marker.height()),
        )
        return QRectF(left, top, marker.width(), marker.height())

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
        if self._mode == MODE_SLIDER:
            self._paint_slider(painter)
        else:
            self._paint_lens(painter)
        painter.end()

    def _paint_slider(self, painter: QPainter) -> None:
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
        # Side labels sit in the bottom corners, clear of the image's centre.
        bottom = self.view_rect().bottom() - 18
        self._draw_pill(painter, "原图", self.view_rect().left() + 12, bottom)
        self._draw_pill(
            painter, "超分输出", self.view_rect().right() - 12, bottom, right=True
        )
        if self._cursor_inside:
            self._draw_pill(
                painter,
                self._readout_text(),
                self.view_rect().left() + 12,
                self.view_rect().top() + 18,
            )

    def _paint_lens(self, painter: QPainter) -> None:
        rect = self.image_rect()
        if self._output_layer is not None:
            painter.drawPixmap(self._layer_origin, self._output_layer)

        if not self._cursor_inside:
            self._draw_pill(
                painter,
                HINT_LENS,
                self.view_rect().left() + 12,
                self.view_rect().top() + 18,
            )
            return

        # Marker on the image showing which pixels the lens is showing.
        marker = self.lens_marker_rect()
        painter.setPen(QPen(QColor(theme.ACCENT), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(marker)

        cell = self.lens_cell_size()
        if cell < LENS_CELL_DRAW_MIN_PX:
            return
        original_rect, output_rect = self.lens_panels()
        block = self.sample_block(QSizeF(cell, cell))
        for target, pixmap, label in (
            (original_rect, self._crop_original(block, target_size=original_rect.size()), "原图"),
            (output_rect, self._crop_output(block, target_size=output_rect.size()), "超分输出"),
        ):
            painter.drawPixmap(target.topLeft(), pixmap)
            painter.setPen(QPen(QColor(theme.SURFACE), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(target.adjusted(1.0, 1.0, -1.0, -1.0))
            painter.setPen(QPen(QColor(theme.BORDER_STRONG), 1))
            painter.drawRect(target)
            self._draw_pill(
                painter,
                label,
                target.center().x(),
                target.bottom() - 14,
            )
        self._draw_pill(
            painter,
            self._readout_text(),
            self.view_rect().left() + 12,
            self.view_rect().top() + 18,
        )

    def _readout_text(self) -> str:
        """Small status pill: where the pointer is and at what magnification."""
        if self._mode == MODE_LENS:
            detail = f"放大 {self._lens_zoom:g}x"
        else:
            detail = f"缩放 {self._zoom * 100:.0f}%"
        return f"输出坐标 {int(self._cursor.x())}, {int(self._cursor.y())} · {detail}"

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

    def _crop_output(self, block: QRectF, target_size) -> QPixmap:
        return self._crop(self._pair.output, block, target_size)

    def _crop_original(self, block: QRectF, target_size) -> QPixmap:
        ratio = self.original_ratio()
        source = QRectF(
            block.left() * ratio.x(),
            block.top() * ratio.y(),
            block.width() * ratio.x(),
            block.height() * ratio.y(),
        )
        source = source.intersected(QRectF(self._pair.original.rect()))
        return self._crop(self._pair.original, source, target_size)

    @staticmethod
    def _crop(image: QImage, region: QRectF, target_size) -> QPixmap:
        target = QSize(
            max(1, int(round(target_size.width()))),
            max(1, int(round(target_size.height()))),
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

    def _draw_pill(
        self,
        painter: QPainter,
        text: str,
        x: float,
        y: float,
        right: bool = False,
    ) -> None:
        """Translucent label pill whose vertical centre sits on ``y``."""
        font = QFont(painter.font())
        font.setPointSizeF(max(8.0, font.pointSizeF() - 1.0))
        # The pill changes pen, brush and font; callers draw on with theirs.
        painter.save()
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = float(metrics.horizontalAdvance(text)) + 16.0
        height = float(metrics.height()) + 4.0
        box = QRectF(x - width if right else x, y - height / 2.0, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.drawRoundedRect(box, 6.0, 6.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

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
            self._clamp_pan()
            self._invalidate_layers()
            self.update()
            return

        inside = self.view_rect().contains(position)
        self._cursor_inside = inside and self._pair is not None
        if inside:
            self._cursor = self.clamp_to_output(self.viewport_to_output(position))
        self._update_hover_cursor(position)
        self.update()

    def _update_hover_cursor(self, position: QPointF) -> None:
        if self._mode == MODE_LENS:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        near_split = abs(position.x() - (self.view_rect().left() + self.view_rect().width() * self._split)) <= SPLITTER_GRAB_PX
        self.setCursor(
            Qt.CursorShape.SplitHCursor if near_split else Qt.CursorShape.OpenHandCursor
        )

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._cursor_inside = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            self._pair is None
            or self._mode != MODE_SLIDER
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return
        position = QPointF(event.position())
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
        if self._mode == MODE_SLIDER and self._pair is not None:
            self._apply_mode_cursor()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Double click snaps back to the fit view (in either mode)."""
        if self._pair is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self.fit_to_view()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._pair is None:
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        factor = 1.25 ** steps
        if self._mode == MODE_LENS:
            self.set_lens_zoom(self._lens_zoom * factor)
        else:
            self.zoom_at(QPointF(event.position()), factor)
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._clamp_pan()
        self._invalidate_layers()
        super().resizeEvent(event)


class ModeSwitch(QWidget):
    """Segmented control choosing between the two comparison modes."""

    changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._mode = MODE_LENS
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for mode, label in MODE_OPTIONS:
            button = QPushButton(label)
            button.setObjectName("segmentButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setMinimumHeight(30)
            button.setToolTip(
                HINT_LENS if mode == MODE_LENS else HINT_SLIDER
            )
            layout.addWidget(button)
            self._group.addButton(button)
            self._buttons[mode] = button
            button.clicked.connect(lambda _checked=False, mode=mode: self.set_mode(mode))
        self._buttons[MODE_LENS].setChecked(True)

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        mode = _LEGACY_MODES.get(str(mode), str(mode))
        if mode not in self._buttons:
            raise ValueError(f"unknown comparison mode: {mode}")
        self._buttons[mode].setChecked(True)
        if mode == self._mode:
            return
        self._mode = mode
        self.changed.emit(mode)


class ComparisonPanel(QWidget):
    """Mode switch + zoom slider + status line + :class:`ComparisonView`.

    One panel sits on the comparison page, another inside
    :class:`ComparisonDialog`; the two are kept in step by the window (see
    ``_sync_compare_panels``).
    """

    mode_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.setSpacing(8)
        caption = QLabel("对比方式")
        caption.setObjectName("fieldLabel")
        self.mode_switch = ModeSwitch()
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        row.addWidget(caption)
        row.addWidget(self.mode_switch)
        row.addStretch(1)
        row.addLayout(self.actions)
        layout.addLayout(row)

        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(8)
        self.zoom_caption = QLabel("放大")
        self.zoom_caption.setObjectName("fieldLabel")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(ZOOM_SLIDER_MIN, ZOOM_SLIDER_MAX)
        self.zoom_slider.setSingleStep(ZOOM_SLIDER_STEP)
        self.zoom_slider.setPageStep(ZOOM_SLIDER_STEP * 10)
        self.zoom_slider.setFixedWidth(220)
        self.zoom_value = QLabel()
        self.zoom_value.setObjectName("valueStrong")
        self.zoom_value.setMinimumWidth(56)
        self.status = QLabel("等待任务完成")
        self.status.setObjectName("hintLabel")
        self.status.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        zoom_row.addWidget(self.zoom_caption)
        zoom_row.addWidget(self.zoom_slider)
        zoom_row.addWidget(self.zoom_value)
        zoom_row.addStretch(1)
        zoom_row.addWidget(self.status, 1)
        layout.addLayout(zoom_row)

        self.view = ComparisonView()
        layout.addWidget(self.view, 1)

        self.mode_switch.changed.connect(self._on_mode_switch_changed)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        self.view.view_changed.connect(self._sync_zoom_controls)
        self._sync_zoom_controls()

    # ---- controls ------------------------------------------------------------

    def _on_mode_switch_changed(self, mode: str) -> None:
        self.view.set_mode(mode)
        self._sync_zoom_controls()
        self.mode_changed.emit(mode)

    def _on_zoom_slider_changed(self, value: int) -> None:
        if self._syncing:
            return
        if self.view.mode() == MODE_LENS:
            self.view.set_lens_zoom(value / 100.0)
        else:
            self.view.set_zoom(value / 100.0)

    def _sync_zoom_controls(self) -> None:
        """Point the slider at whichever magnification the current mode uses."""
        lens = self.view.mode() == MODE_LENS
        current = self.view.lens_zoom() if lens else self.view.zoom()
        self._syncing = True
        self.zoom_slider.setValue(
            int(min(
                max(round(current * 100), ZOOM_SLIDER_MIN),
                ZOOM_SLIDER_MAX,
            ))
        )
        self._syncing = False
        if lens:
            self.zoom_value.setText(f"{self.view.lens_zoom():g}x")
            self.zoom_caption.setText("放大镜倍率")
            self.zoom_slider.setToolTip("放大镜倍率（滚轮同样可调）")
        else:
            self.zoom_value.setText(f"{self.view.zoom() * 100:.0f}%")
            self.zoom_caption.setText("视图缩放")
            self.zoom_slider.setToolTip("相对适应窗口的缩放（滚轮以光标为中心缩放）")

    def add_action(self, button: QWidget) -> None:
        """Append a host-provided toolbar button (fit / 1:1 / close / …)."""
        self.actions.addWidget(button)

    def mode(self) -> str:
        return self.view.mode()

    def set_mode(self, mode: str) -> None:
        self.mode_switch.set_mode(mode)

    def set_summary(self, text: str) -> None:
        self.status.setText(text)

    def summary(self) -> str:
        return self.status.text()

    def set_pair(self, pair: Optional[PreviewPair]) -> None:
        self.view.set_pair(pair)


class ComparisonDialog(QDialog):
    """Big, resizable window for comfortable pixel peeping.

    The comparison page already fills the window; this dialog is for a second
    screen or a side-by-side look (it remembers its geometry, and can be
    maximised like any other window).
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
    "MODE_LENS",
    "MODE_OPTIONS",
    "MODE_SLIDER",
    "ComparisonDialog",
    "ComparisonPanel",
    "ComparisonView",
    "ModeSwitch",
    "PreviewLoadWorker",
    "format_pair_summary",
    "ghost_button",
    "PreviewPair",
    "format_pair_summary",
    "load_preview_pair",
]
