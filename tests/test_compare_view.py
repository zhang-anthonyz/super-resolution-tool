"""Input/output comparison preview: loader, geometry mapping, interaction."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SRT_SETTINGS_DIR", tempfile.mkdtemp(prefix="srt-compare-tests-"))
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from app.core.types import GpuInfo, GpuType, GpuVendor, SuperResResult  # noqa: E402
from app.pipelines.image_pipeline import (  # noqa: E402
    COMPRESSION_PROFILE_HIGH,
    _save_image,
)
from app.ui import compare_view, main_window, theme  # noqa: E402
from app.ui.compare_view import (  # noqa: E402
    MODE_LENS,
    MODE_OPTIONS,
    MODE_SLIDER,
    ComparisonView,
    format_pair_summary,
    load_preview_pair,
)
from app.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    theme.apply_app_theme(app)
    yield app


def make_gpu() -> GpuInfo:
    return GpuInfo(
        vendor=GpuVendor.AMD,
        name="AMD Radeon RX 7900 XT",
        type=GpuType.DISCRETE,
        dedicated_vram_bytes=24 * 1024**3,
        vendor_id=0x1002,
    )


def _write_pair(tmp_path: Path, size=(64, 48), scale=4) -> tuple[Path, Path]:
    """A smooth original and a larger, sharper "super-resolved" output."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    original = np.stack(
        [
            (xx * 255 // max(1, width - 1)).astype(np.uint8),
            (yy * 255 // max(1, height - 1)).astype(np.uint8),
            np.full((height, width), 128, np.uint8),
        ],
        axis=-1,
    )
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.fromarray(original).save(input_path)
    output = Image.fromarray(original).resize(
        (width * scale, height * scale), Image.Resampling.NEAREST
    )
    output.save(output_path)
    return input_path, output_path


def _mouse(event_type, position, button=Qt.MouseButton.NoButton, buttons=None):
    point = QPointF(position)
    pressed = buttons if buttons is not None else button
    return QMouseEvent(
        event_type,
        point,
        point,
        point,
        button,
        pressed,
        Qt.KeyboardModifier.NoModifier,
    )


def _hover(view: ComparisonView, position: QPointF) -> None:
    view.mouseMoveEvent(_mouse(QMouseEvent.Type.MouseMove, position))


def test_preview_loads_both_sides_and_reports_sizes(tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)

    pair = load_preview_pair(input_path, output_path)

    assert (pair.original.width(), pair.original.height()) == (64, 48)
    assert (pair.output.width(), pair.output.height()) == (256, 192)
    assert pair.original_true_size == (64, 48)
    assert pair.output_true_size == (256, 192)
    assert pair.preview_scale == 1.0
    assert pair.scaled_note == "1:1 预览"
    assert "64×48" in format_pair_summary(pair)
    assert "256×192" in format_pair_summary(pair)


def test_huge_output_is_downscaled_within_the_memory_budget(tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)

    pair = load_preview_pair(input_path, output_path, max_pixels=10_000)

    pixels = pair.output.width() * pair.output.height()
    assert pixels <= 10_000
    # Both sides keep their aspect ratio and stay aligned.
    assert abs(
        pair.output.width() / pair.output.height() - 256 / 192
    ) < 0.02
    assert pair.preview_scale < 1.0
    assert "预览已缩放至" in pair.scaled_note


def test_missing_output_reports_an_error(tmp_path):
    input_path, _output_path = _write_pair(tmp_path)

    with pytest.raises(Exception):
        load_preview_pair(input_path, tmp_path / "nope.png")


def test_view_maps_widget_points_to_output_pixels(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    pair = load_preview_pair(input_path, output_path)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(pair)

    centre = view.image_rect().center()

    assert view.viewport_to_output(centre).x() == pytest.approx(128.0, abs=0.01)
    assert view.viewport_to_output(centre).y() == pytest.approx(96.0, abs=0.01)
    # The original is one quarter of the output on each axis.
    assert view.original_ratio().x() == pytest.approx(0.25)
    assert view.original_ratio().y() == pytest.approx(0.25)
    # The canvas is the whole widget in both modes (no strip eating the view).
    assert view.view_rect().height() == pytest.approx(view.height())


def test_sample_block_keeps_the_cursor_inside_the_image(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))
    cell = QSizeF(200.0, 120.0)

    # A 1:1 cell shows the cell's own pixels.
    view.set_lens_zoom(1.0)
    view._cursor = QPointF(0.0, 0.0)
    block = view.sample_block(cell)
    assert block.left() == 0.0 and block.top() == 0.0
    assert block.width() == pytest.approx(200.0)
    assert block.right() <= 256.0

    view._cursor = QPointF(255.0, 191.0)
    block = view.sample_block(cell)
    assert block.right() == pytest.approx(256.0)
    assert block.bottom() == pytest.approx(192.0)


def test_sample_block_magnifies_by_the_lens_zoom(qapp, tmp_path):
    """The lens shows fewer output pixels the more it magnifies."""
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))
    cell = QSizeF(160.0, 160.0)

    assert view.lens_zoom() == pytest.approx(compare_view.LENS_DEFAULT_ZOOM)
    magnified = view.sample_block(cell)
    assert magnified.width() == pytest.approx(160.0 / compare_view.LENS_DEFAULT_ZOOM)

    view.set_lens_zoom(1.0)
    assert view.sample_block(cell).width() == pytest.approx(160.0)


def test_lens_tracks_the_mouse(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))

    rect = view.image_rect()
    _hover(view, rect.center() + QPointF(10.0, 8.0))

    assert view.sample_point().x() > 128.0
    assert view.sample_point().y() > 96.0


def _panel_group(view) -> QRectF:
    return view.lens_panels()[0].united(view.lens_panels()[1])


def test_lens_panels_hold_the_middle_of_the_canvas(qapp, tmp_path):
    """The pair is a fixed place to look, not a pane that chases the cursor."""
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))
    area = view.view_rect()

    # Pointer in a far corner: the pair ignores it and stays centred.
    rect = view.image_rect()
    _hover(view, QPointF(rect.left() + 2.0, rect.top() + 2.0))
    group = _panel_group(view)
    assert group.center().x() == pytest.approx(area.center().x(), abs=1.0)
    assert group.center().y() == pytest.approx(area.center().y(), abs=1.0)

    original_cell, output_cell = view.lens_panels()
    for cell_rect in (original_cell, output_cell):
        assert area.contains(cell_rect), (cell_rect, area)
        assert cell_rect.width() == pytest.approx(view.lens_cell_size())
    assert output_cell.left() > original_cell.left()
    assert output_cell.top() == pytest.approx(original_cell.top())

    # Moving the pointer around (away from the pair) must not move the pair.
    other = QPointF(rect.right() - 2.0, rect.bottom() - 2.0)
    _hover(view, other)
    assert _panel_group(view) == group


def test_lens_panels_lean_only_a_little_for_the_pointer(qapp, tmp_path):
    """The pair gives the pointer room, but never wanders off to an edge."""
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))
    area = view.view_rect()
    limit = view.lens_cell_size() * compare_view.LENS_DODGE_LIMIT_RATIO

    for x_share in (0.1, 0.35, 0.5, 0.65, 0.9):
        for y_share in (0.05, 0.3, 0.42, 0.5, 0.58, 0.7, 0.95):
            _hover(
                view,
                QPointF(
                    area.left() + area.width() * x_share,
                    area.top() + area.height() * y_share,
                ),
            )
            group = _panel_group(view)
            marker = view.lens_marker_rect()
            offset = group.center().y() - area.center().y()
            assert area.contains(group), (x_share, y_share, group, area)
            # Never further than the cap from the middle of the canvas, and
            # never off the canvas: the pair is a fixture, not a follower.
            assert abs(offset) <= limit + 1.0, (x_share, y_share, offset, limit)
            if marker.intersects(group):
                # Still overlapping: buried too deep to be freed within the cap,
                # so the pair leans as far as it is allowed to.
                assert abs(offset) >= limit - 1.0, (x_share, y_share, offset)


def test_lens_panels_lean_away_from_the_pointer(qapp, tmp_path):
    """Pointer just inside either edge of the pair: it moves off the spot, the
    short way, and the spot is fully visible again."""
    input_path, output_path = _write_pair(tmp_path, size=(160, 120), scale=16)
    for above in (True, False):
        # A fresh view each time: which side the pair leans to is sticky until
        # the pointer leaves it, exactly so a crossing does not flip it.
        view = ComparisonView()
        view.resize(800, 500)
        view.set_pair(load_preview_pair(input_path, output_path))
        area = view.view_rect()
        cell = view.lens_cell_size()
        home = QRectF(
            area.center().x() - (2 * cell + compare_view.LENS_GAP_PX) / 2.0,
            area.center().y() - cell / 2.0,
            2 * cell + compare_view.LENS_GAP_PX,
            cell,
        )
        _hover(
            view,
            QPointF(
                area.center().x(),
                home.top() + 8.0 if above else home.bottom() - 8.0,
            ),
        )

        group = _panel_group(view)
        offset = group.center().y() - area.center().y()
        assert not view.lens_marker_rect().intersects(group), (above, group)
        assert abs(offset) <= cell * compare_view.LENS_DODGE_LIMIT_RATIO + 1.0
        # A pointer above the middle pushes the pair down and vice versa.
        assert (offset > 0) is above, (above, offset)


def test_lens_panels_hold_their_side_while_the_pointer_crosses(qapp, tmp_path):
    """Crossing the pair's middle must not throw it to the other side."""
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))
    area = view.view_rect()

    offsets = []
    for step in range(41):
        y_share = 0.3 + 0.4 * step / 40
        _hover(view, QPointF(area.center().x(), area.top() + area.height() * y_share))
        offsets.append(_panel_group(view).center().y() - area.center().y())

    # It moves only in one direction while the pointer walks through it.
    signs = {1 if offset > 0 else -1 if offset < 0 else 0 for offset in offsets}
    assert signs <= {1, 0} or signs <= {-1, 0}, offsets


def test_lens_overlay_disappears_when_the_mouse_leaves(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))

    _hover(view, view.image_rect().center())
    assert view._cursor_inside is True

    view.leaveEvent(None)  # noqa: SLF001 - the event is not used

    assert view._cursor_inside is False


def test_lens_cell_size_is_bounded_by_the_view(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path)
    pair = load_preview_pair(input_path, output_path)
    view = ComparisonView()
    view.set_pair(pair)

    view.resize(800, 1100)  # tall: cells grow but stay usable
    assert compare_view.LENS_CELL_MIN_PX <= view.lens_cell_size() <= compare_view.LENS_CELL_MAX_PX

    view.resize(420, 320)  # cramped: the two cells still fit side by side
    cell = view.lens_cell_size()
    assert cell >= compare_view.LENS_CELL_MIN_PX
    original_cell, output_cell = view.lens_panels()
    assert original_cell.width() + output_cell.width() <= view.width()


def test_wheel_changes_lens_magnification_not_the_view(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))

    assert view.mode() == MODE_LENS
    before = view.lens_zoom()
    view.wheelEvent(_wheel(view.image_rect().center(), 120))

    assert view.lens_zoom() > before
    assert view.zoom() == pytest.approx(1.0)  # the fitted view is untouched


def _wheel(position, delta):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QWheelEvent

    point = QPointF(position)
    return QWheelEvent(
        point,  # position
        point,  # globalPosition
        QPoint(0, 0),  # pixelDelta
        QPoint(0, delta),  # angleDelta
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_slider_mode_drags_the_divider_and_pans_elsewhere(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))
    view.set_mode(MODE_SLIDER)

    area = view.view_rect()
    middle = QPointF(area.left() + area.width() * 0.5, area.center().y())
    view.mousePressEvent(_mouse(QMouseEvent.Type.MouseButtonPress, middle, Qt.MouseButton.LeftButton))
    view.mouseMoveEvent(
        _mouse(
            QMouseEvent.Type.MouseMove,
            QPointF(area.left() + area.width() * 0.25, area.center().y()),
            Qt.MouseButton.LeftButton,
        )
    )
    view.mouseReleaseEvent(_mouse(QMouseEvent.Type.MouseButtonRelease, middle, Qt.MouseButton.LeftButton))
    assert view.split_ratio() == pytest.approx(0.25, abs=0.01)

    # Dragging far from the divider pans instead of moving it.
    split_before = view.split_ratio()
    far = QPointF(area.left() + area.width() * 0.9, area.center().y())
    view.mousePressEvent(_mouse(QMouseEvent.Type.MouseButtonPress, far, Qt.MouseButton.LeftButton))
    view.mouseMoveEvent(
        _mouse(
            QMouseEvent.Type.MouseMove,
            QPointF(far.x() + 40.0, far.y()),
            Qt.MouseButton.LeftButton,
        )
    )
    view.mouseReleaseEvent(_mouse(QMouseEvent.Type.MouseButtonRelease, far, Qt.MouseButton.LeftButton))
    assert view.split_ratio() == split_before


def test_wheel_zoom_keeps_the_pixel_under_the_cursor(qapp, tmp_path):
    """Zooming anchors on the cursor, so the detail being inspected stays put.

    (Already zoomed in: at the fit view the pan clamp wins over anchoring, which
    is what every image viewer does.)
    """
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))
    view.set_mode(MODE_SLIDER)
    view.set_zoom(2.0)

    point = view.image_rect().center() + QPointF(40.0, -24.0)
    before = view.viewport_to_output(point)
    view.wheelEvent(_wheel(point, 120))
    after = view.viewport_to_output(point)

    assert view.zoom() > 2.0
    assert after.x() == pytest.approx(before.x(), abs=0.6)
    assert after.y() == pytest.approx(before.y(), abs=0.6)


def test_pan_is_clamped_so_the_image_stays_in_view(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))
    view.set_mode(MODE_SLIDER)

    # Fitted: the image is smaller than the view, so there is nothing to pan.
    area = view.view_rect()
    start = QPointF(area.left() + area.width() * 0.9, area.center().y())
    view.mousePressEvent(_mouse(QMouseEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton))
    view.mouseMoveEvent(
        _mouse(QMouseEvent.Type.MouseMove, start + QPointF(-300.0, -200.0), Qt.MouseButton.LeftButton)
    )
    view.mouseReleaseEvent(_mouse(QMouseEvent.Type.MouseButtonRelease, start, Qt.MouseButton.LeftButton))
    assert view.pan() == QPointF(0.0, 0.0)

    # Zoomed in: dragging all the way cannot drag the image out of the view.
    view.set_zoom(4.0)
    view.mousePressEvent(_mouse(QMouseEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton))
    view.mouseMoveEvent(
        _mouse(
            QMouseEvent.Type.MouseMove,
            start + QPointF(5000.0, 5000.0),
            Qt.MouseButton.LeftButton,
        )
    )
    view.mouseReleaseEvent(_mouse(QMouseEvent.Type.MouseButtonRelease, start, Qt.MouseButton.LeftButton))
    visible = view.image_rect().intersected(view.view_rect())
    assert visible.width() >= view.view_rect().width() - 1.0
    assert visible.height() >= view.view_rect().height() - 1.0


def test_mode_switch_resets_zoom_and_rejects_unknown_modes(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))

    view.set_zoom(3.0)
    view.set_mode(MODE_SLIDER)

    assert view.mode() == MODE_SLIDER
    assert view.zoom() == 1.0
    with pytest.raises(ValueError):
        view.set_mode("stereo")


def test_old_mode_names_still_map_to_lens_and_slider(qapp, tmp_path):
    """Settings written before the rework said block/wipe."""
    input_path, output_path = _write_pair(tmp_path)
    view = ComparisonView()
    view.set_pair(load_preview_pair(input_path, output_path))

    view.set_mode("wipe")
    assert view.mode() == MODE_SLIDER
    view.set_mode("block")
    assert view.mode() == MODE_LENS


def test_zoom_is_clamped_and_one_to_one_matches_output_pixels(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))

    view.set_zoom(999.0)
    assert view.zoom() == compare_view.MAX_ZOOM
    view.set_zoom(0.0)
    assert view.zoom() == compare_view.MIN_ZOOM

    view.set_lens_zoom(999.0)
    assert view.lens_zoom() == compare_view.LENS_MAX_ZOOM

    view.one_to_one()
    assert view.image_scale() == pytest.approx(1.0, abs=0.02)

    view.fit_to_view()
    assert view.zoom() == pytest.approx(1.0)
    # This image is smaller than the view, so fitting scales it up.
    assert view.image_scale() > 1.0


def test_rendering_stays_cheap_when_zoomed_in(qapp, tmp_path):
    """The cached layers cover the viewport, not the whole image at zoom."""
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(640, 400)
    view.set_pair(load_preview_pair(input_path, output_path))
    view.one_to_one()
    view.grab()

    layer = view._output_layer
    assert layer.width() <= view.width() + 2
    assert layer.height() <= view.height() + 2


def test_rendering_paints_content_in_both_modes(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(640, 400)
    view.set_pair(load_preview_pair(input_path, output_path))

    for mode in (MODE_LENS, MODE_SLIDER):
        view.set_mode(mode)
        image = view.grab().toImage()
        colors = {image.pixel(x, y) for x in range(0, 200, 7) for y in range(0, 200, 7)}
        assert len(colors) > 3, f"{mode} painted a flat surface"


def test_lens_paints_both_magnified_cells(qapp, tmp_path):
    """Both cells must show the sampled content, not a background or a wash."""
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(640, 480)
    pair = load_preview_pair(input_path, output_path)
    view.set_pair(pair)
    _hover(view, view.image_rect().center())
    image = view.grab().toImage()

    original_cell, output_cell = view.lens_panels()
    block = view.sample_block(QSizeF(view.lens_cell_size(), view.lens_cell_size()))
    ratio = view.original_ratio()
    sandbox = ((pair.output, output_cell), (pair.original, original_cell))
    for source, cell_rect in sandbox:
        block_center = block.center()
        source_point = QPointF(
            block_center.x() * (ratio.x() if source is pair.original else 1.0),
            block_center.y() * (ratio.y() if source is pair.original else 1.0),
        )
        expected = source.pixelColor(int(source_point.x()), int(source_point.y()))
        point = cell_rect.center()
        painted = image.pixelColor(int(point.x()), int(point.y()))
        for channel in ("red", "green", "blue"):
            assert abs(
                getattr(painted, channel)() - getattr(expected, channel)()
            ) <= 32, (painted.name(), expected.name())


class _RecordingPreviewWorker:
    """Stand-in for PreviewLoadWorker so the wiring can be driven by hand."""

    instances = []

    def __init__(self, token, input_path, output_path, parent=None):
        self.token = token
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.loaded = _Signal()
        self.failed = _Signal()
        self.started = False
        _RecordingPreviewWorker.instances.append(self)

    def start(self):
        self.started = True

    def deleteLater(self):
        return None


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in self.slots:
            slot(*args)


@pytest.fixture
def recording_workers(monkeypatch):
    _RecordingPreviewWorker.instances = []
    monkeypatch.setattr(
        compare_view, "PreviewLoadWorker", _RecordingPreviewWorker
    )
    yield _RecordingPreviewWorker
    _RecordingPreviewWorker.instances = []


def test_preview_page_opens_after_a_single_image_task(qapp, recording_workers, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(32, 24), scale=2)
    window = MainWindow(make_gpu())
    try:
        assert window.stack.currentIndex() == 0

        window._on_image_finished(
            [SuperResResult(ok=True, input_path=input_path, output_path=output_path)]
        )

        # The comparison is a page of its own and takes over the canvas.
        assert window.stack.currentIndex() == MainWindow.COMPARE_PAGE
        worker = recording_workers.instances[-1]
        assert worker.started is True
        assert worker.input_path == input_path
        assert worker.output_path == output_path
        assert "载入" in window.image_compare_status.text()

        pair = load_preview_pair(input_path, output_path)
        worker.loaded.emit(worker.token, pair)

        assert window.image_compare_view.pair() is pair
        assert "32×24" in window.image_compare_status.text()
        assert "对比预览" in window.image_log.text_edit.toPlainText()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_batch_tasks_do_not_offer_the_preview(qapp, recording_workers, tmp_path):
    input_path, output_path = _write_pair(tmp_path)
    window = MainWindow(make_gpu())
    try:
        window._on_image_finished(
            [
                SuperResResult(ok=True, input_path=input_path, output_path=output_path),
                SuperResResult(ok=True, input_path=input_path, output_path=output_path),
            ]
        )

        assert window.image_compare_view.pair() is None
        assert window.stack.currentIndex() == 0
        assert recording_workers.instances == []
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_stale_preview_loads_are_ignored(qapp, recording_workers, tmp_path):
    input_path, output_path = _write_pair(tmp_path)
    window = MainWindow(make_gpu())
    try:
        window._on_image_finished(
            [SuperResResult(ok=True, input_path=input_path, output_path=output_path)]
        )
        stale = recording_workers.instances[-1]
        window.close_image_compare()  # bumps the token

        pair = load_preview_pair(input_path, output_path)
        stale.loaded.emit(stale.token, pair)

        assert window.image_compare_view.pair() is None
        assert window.stack.currentIndex() == 0
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_preview_failure_is_reported_in_the_log(qapp, recording_workers, tmp_path):
    input_path, output_path = _write_pair(tmp_path)
    window = MainWindow(make_gpu())
    try:
        window._on_image_finished(
            [SuperResResult(ok=True, input_path=input_path, output_path=output_path)]
        )
        worker = recording_workers.instances[-1]

        worker.failed.emit(worker.token, "broken png")

        assert "对比图载入失败" in window.image_compare_status.text()
        assert "对比预览不可用：broken png" in window.image_log.text_edit.toPlainText()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_mode_switch_in_the_window_updates_the_view(qapp, recording_workers, tmp_path):
    input_path, output_path = _write_pair(tmp_path)
    window = MainWindow(make_gpu())
    try:
        switch = window.image_compare_panel.mode_switch
        assert [mode for mode, _label in MODE_OPTIONS] == [MODE_LENS, MODE_SLIDER]
        assert window.image_compare_view.mode() == MODE_LENS

        switch.set_mode(MODE_SLIDER)
        assert window.image_compare_view.mode() == MODE_SLIDER
        switch.set_mode(MODE_LENS)
        assert window.image_compare_view.mode() == MODE_LENS
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_zoom_control_follows_the_mode(qapp, recording_workers, tmp_path):
    window = MainWindow(make_gpu())
    try:
        panel = window.image_compare_panel
        # Lens mode: the slider is the lens magnification (4x by default).
        assert panel.zoom_slider.value() == pytest.approx(400, abs=1)
        assert "4" in panel.zoom_value.text()

        panel.zoom_slider.setValue(800)
        assert window.image_compare_view.lens_zoom() == pytest.approx(8.0)

        panel.mode_switch.set_mode(MODE_SLIDER)
        assert panel.zoom_slider.value() == 100  # fitted = 100%
        assert "%" in panel.zoom_value.text()

        panel.zoom_slider.setValue(250)
        assert window.image_compare_view.zoom() == pytest.approx(2.5)
        # Each mode keeps its own magnification.
        panel.mode_switch.set_mode(MODE_LENS)
        assert panel.zoom_slider.value() == pytest.approx(800, abs=1)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_saved_outputs_from_the_pipeline_are_previewable(qapp, tmp_path):
    """The pipeline's own writer output must load back for comparison."""
    input_path, _unused = _write_pair(tmp_path, size=(32, 24), scale=2)
    original = np.asarray(Image.open(input_path).convert("RGB"))
    upscaled = np.repeat(np.repeat(original, 2, axis=0), 2, axis=1)
    output_path = tmp_path / "out.webp"
    _save_image(upscaled, output_path, compression_profile=COMPRESSION_PROFILE_HIGH)

    pair = load_preview_pair(input_path, output_path)

    assert pair.output_true_size == (64, 48)
    assert pair.output.width() == 64


def test_compare_page_controls_stay_usable(qapp, recording_workers, tmp_path):
    """The page owns the whole canvas; its controls must lay out to clickable
    sizes and the view must get real space."""
    input_path, output_path = _write_pair(tmp_path)
    window = MainWindow(make_gpu())
    try:
        window.resize(1240, 950)
        window.show()
        for _ in range(3):
            qapp.processEvents()
        window.grab()

        window._on_image_finished(
            [SuperResResult(ok=True, input_path=input_path, output_path=output_path)]
        )
        for _ in range(3):
            qapp.processEvents()
        window.grab()

        view = window.image_compare_view
        assert window.stack.currentIndex() == MainWindow.COMPARE_PAGE
        assert view.width() >= 300 and view.height() >= 300, (
            view.width(),
            view.height(),
        )
        buttons = [
            child
            for child in window.image_compare_card.findChildren(QPushButton)
            if not child.isHidden()
        ]
        assert len(buttons) >= 6  # 2 mode + 适应窗口 / 1:1 / 大窗口 / 返回
        for button in buttons:
            assert button.width() >= 18 and button.height() >= 18, button.text()
        assert window.image_compare_panel.zoom_slider.width() >= 100
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_compare_page_gives_the_view_the_full_canvas(qapp, recording_workers, tmp_path):
    """The old inline card shared a scroll page with the settings and the log;
    on its own page the view gets the height instead of a 560px minimum."""
    input_path, output_path = _write_pair(tmp_path)
    window = MainWindow(make_gpu())
    try:
        window.resize(1240, 1000)
        window.show()
        for _ in range(3):
            qapp.processEvents()
        window.grab()
        window._on_image_finished(
            [SuperResResult(ok=True, input_path=input_path, output_path=output_path)]
        )
        for _ in range(3):
            qapp.processEvents()
        window.grab()

        view = window.image_compare_view
        # The image page still has its own layout; the comparison is separate.
        assert view.height() >= 400, view.height()
        assert view.height() < window.image_tab.height()
        assert view.lens_cell_size() >= compare_view.LENS_CELL_MIN_PX
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_big_window_button_opens_a_matching_comparison(
    qapp, recording_workers, tmp_path
):
    input_path, output_path = _write_pair(tmp_path, size=(32, 24), scale=2)
    window = MainWindow(make_gpu())
    try:
        window._on_image_finished(
            [SuperResResult(ok=True, input_path=input_path, output_path=output_path)]
        )
        worker = recording_workers.instances[-1]
        pair = load_preview_pair(input_path, output_path)
        worker.loaded.emit(worker.token, pair)

        window.open_image_compare_window()

        dialog = window._compare_dialog
        assert dialog is not None
        assert dialog.view().pair() is pair
        assert dialog.view().mode() == window.image_compare_view.mode()
        assert dialog.view().lens_zoom() == window.image_compare_view.lens_zoom()
        assert dialog.width() >= 1000 and dialog.height() >= 700
        assert "×" in dialog.panel.summary()

        # Mode changes in the big window drive the page view too.
        dialog.panel.set_mode(compare_view.MODE_SLIDER)
        assert window.image_compare_view.mode() == compare_view.MODE_SLIDER

        dialog.close()
        assert window._settings["image_compare_window"]["width"] >= 1000
        assert window.image_compare_view.mode() == compare_view.MODE_SLIDER
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_big_window_needs_a_pair(qapp, recording_workers, tmp_path):
    window = MainWindow(make_gpu())
    try:
        window.open_image_compare_window()

        assert window._compare_dialog is None
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_big_window_restores_its_saved_size(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(
        main_window,
        "load_settings",
        lambda: {
            "image_compare_window": {
                "width": 1111,
                "height": 777,
                "maximized": False,
            }
        },
    )
    input_path, output_path = _write_pair(tmp_path, size=(32, 24), scale=2)
    window = MainWindow(make_gpu())
    try:
        window.image_compare_view.set_pair(load_preview_pair(input_path, output_path))

        window.open_image_compare_window()

        dialog = window._compare_dialog
        assert (dialog.width(), dialog.height()) == (1111, 777)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_saving_settings_keeps_the_comparison_window_geometry(
    qapp, monkeypatch, tmp_path
):
    """Regression: _window_settings used to rebuild the dict from scratch and
    drop image_compare_window, so the big window's size never reached the file."""
    written: dict = {}
    monkeypatch.setattr(
        main_window, "save_settings", lambda payload: written.update(payload)
    )
    window = MainWindow(make_gpu())
    try:
        window.image_compare_panel.set_mode(MODE_SLIDER)
        window.image_compare_view.set_lens_zoom(6.0)
        window.image_compare_view.set_split_ratio(0.3)
        window._settings["image_compare_window"] = {
            "width": 1234,
            "height": 890,
            "maximized": False,
        }

        window._save_window_settings()

        assert written["image_compare_window"]["width"] == 1234
        assert written["image_compare"]["mode"] == MODE_SLIDER
        assert written["image_compare"]["lens_zoom"] == pytest.approx(6.0)
        assert written["image_compare"]["split"] == pytest.approx(0.3)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_saved_comparison_preferences_are_restored(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(
        main_window,
        "load_settings",
        lambda: {
            "image_compare": {"mode": "wipe", "lens_zoom": 7.0, "split": 0.25},
        },
    )
    window = MainWindow(make_gpu())
    try:
        # "wipe" is the pre-rework name for the slider view.
        assert window.image_compare_view.mode() == MODE_SLIDER
        assert window.image_compare_view.lens_zoom() == pytest.approx(7.0)
        assert window.image_compare_view.split_ratio() == pytest.approx(0.25)
        assert window.image_compare_panel.mode_switch.mode() == MODE_SLIDER
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
