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

from PySide6.QtCore import QPointF, QSizeF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from app.core.types import GpuInfo, GpuType, GpuVendor, SuperResResult  # noqa: E402
from app.pipelines.image_pipeline import (  # noqa: E402
    COMPRESSION_PROFILE_HIGH,
    _save_image,
)
from app.ui import compare_view, main_window, theme  # noqa: E402
from app.ui.compare_view import (  # noqa: E402
    MODE_BLOCK,
    MODE_WIPE,
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


def test_sample_block_keeps_the_cursor_inside_the_image(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))
    cell = QSizeF(200.0, 120.0)

    # Cursor pushed into the top-left corner: the block must stay in bounds.
    view._cursor = QPointF(0.0, 0.0)
    block = view.sample_block(cell)
    assert block.left() == 0.0 and block.top() == 0.0
    assert block.width() == pytest.approx(200.0)
    assert block.right() <= 256.0

    view._cursor = QPointF(255.0, 191.0)
    block = view.sample_block(cell)
    assert block.right() == pytest.approx(256.0)
    assert block.bottom() == pytest.approx(192.0)


def test_block_mode_tracks_the_mouse(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))

    rect = view.image_rect()
    view.mouseMoveEvent(
        _mouse(QMouseEvent.Type.MouseMove, rect.center() + QPointF(10.0, 8.0))
    )

    assert view.sample_point().x() > 128.0
    assert view.sample_point().y() > 96.0
    # Only the top area holds the image; the strip is below it.
    assert view.view_rect().bottom() < view.strip_rect().bottom()


def test_wipe_mode_drags_the_divider_and_pans_elsewhere(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))
    view.set_mode(MODE_WIPE)

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


def test_mode_switch_resets_zoom_and_rejects_unknown_modes(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))

    view.set_zoom(3.0)
    view.set_mode(MODE_WIPE)

    assert view.mode() == MODE_WIPE
    assert view.zoom() == 1.0
    with pytest.raises(ValueError):
        view.set_mode("stereo")


def test_zoom_is_clamped_and_one_to_one_matches_output_pixels(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(64, 48), scale=4)
    view = ComparisonView()
    view.resize(800, 500)
    view.set_pair(load_preview_pair(input_path, output_path))

    view.set_zoom(999.0)
    assert view.zoom() == compare_view.MAX_ZOOM
    view.set_zoom(0.0)
    assert view.zoom() == compare_view.MIN_ZOOM

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

    for mode in (MODE_BLOCK, MODE_WIPE):
        view.set_mode(mode)
        image = view.grab().toImage()
        colors = {image.pixel(x, y) for x in range(0, 200, 7) for y in range(0, 200, 7)}
        assert len(colors) > 3, f"{mode} painted a flat surface"


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


def test_preview_appears_after_a_single_image_task(qapp, recording_workers, tmp_path):
    input_path, output_path = _write_pair(tmp_path, size=(32, 24), scale=2)
    window = MainWindow(make_gpu())
    try:
        assert window.image_compare_card.isHidden() is True

        window._on_image_finished(
            [SuperResResult(ok=True, input_path=input_path, output_path=output_path)]
        )

        assert window.image_compare_card.isHidden() is False
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

        assert window.image_compare_card.isHidden() is True
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
        assert window.image_compare_card.isHidden() is True
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
        combo = window.image_compare_mode_combo
        assert [combo.itemData(index) for index in range(combo.count())] == [
            MODE_BLOCK,
            MODE_WIPE,
        ]

        combo.setCurrentIndex(combo.findData(MODE_WIPE))
        assert window.image_compare_view.mode() == MODE_WIPE
        combo.setCurrentIndex(combo.findData(MODE_BLOCK))
        assert window.image_compare_view.mode() == MODE_BLOCK
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


def test_compare_card_controls_stay_usable_when_it_appears(
    qapp, recording_workers, tmp_path
):
    """The card is hidden until a task finishes; its controls must lay out to
    clickable sizes and the view must get real space once it is shown."""
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

        combo = window.image_compare_mode_combo
        view = window.image_compare_view
        assert not window.image_compare_card.isHidden()
        assert combo.width() >= 120 and combo.height() >= 24, (
            combo.width(),
            combo.height(),
        )
        assert view.width() >= 300 and view.height() >= 200, (
            view.width(),
            view.height(),
        )
        buttons = [
            child
            for child in window.image_compare_card.findChildren(QPushButton)
            if not child.isHidden()
        ]
        assert len(buttons) >= 3  # 适应窗口 / 1:1 像素 / 关闭预览
        for button in buttons:
            assert button.width() >= 18 and button.height() >= 18, button.text()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_inline_preview_is_tall_enough_to_be_usable(qapp, recording_workers, tmp_path):
    """The card lives at the end of a scroll page; a short view squeezed the
    block strip into the image area and made previewing cramped."""
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
        assert view.minimumHeight() >= 500, view.minimumHeight()
        assert view.height() >= 500, view.height()
        # In block mode the image area keeps the larger share of that.
        assert view.view_rect().height() > view.strip_height()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_block_strip_height_is_bounded(qapp, tmp_path):
    input_path, output_path = _write_pair(tmp_path)
    pair = load_preview_pair(input_path, output_path)
    view = ComparisonView()
    view.set_pair(pair)

    view.resize(800, 240)  # cramped window
    assert view.strip_height() >= 100.0, view.strip_height()
    assert view.view_rect().height() >= 100.0

    view.resize(800, 1600)  # tall window: the strip must not eat the image
    assert view.strip_height() <= 1600 * 0.5
    assert view.view_rect().height() > view.strip_height()


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
        assert dialog.width() >= 1000 and dialog.height() >= 700
        assert "×" in dialog.panel.summary()

        # Mode changes in the big window drive the inline card too.
        dialog.panel.set_mode(compare_view.MODE_WIPE)
        assert window.image_compare_view.mode() == compare_view.MODE_WIPE

        dialog.close()
        assert window._settings["image_compare_window"]["width"] >= 1000
        assert window.image_compare_view.mode() == compare_view.MODE_WIPE
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
