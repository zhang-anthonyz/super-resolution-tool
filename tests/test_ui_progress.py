import os
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SRT_SETTINGS_DIR", tempfile.mkdtemp(prefix="srt-ui-tests-"))
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from app.pipelines.image_pipeline import PERCENT_MAX
from app.ui import main_window
from app.core.types import (
    Backend,
    GpuInfo,
    GpuType,
    GpuVendor,
    ModelMeta,
    ProgressInfo,
    Scene,
    SuperResResult,
)
from app.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def make_gpu() -> GpuInfo:
    return GpuInfo(
        vendor=GpuVendor.AMD,
        name="AMD Radeon RX 7900 XT",
        type=GpuType.DISCRETE,
        dedicated_vram_bytes=24 * 1024**3,
        vendor_id=0x1002,
    )


def test_progress_bars_start_static_at_zero(qapp):
    window = MainWindow(make_gpu())
    try:
        assert window.image_progress.minimum() == 0
        assert window.image_progress.maximum() == 1000
        assert window.image_progress.value() == 0
        assert window.video_progress.minimum() == 0
        assert window.video_progress.maximum() == 1000
        assert window.video_progress.value() == 0
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_progress_handler_uses_task_total_and_current(qapp):
    window = MainWindow(make_gpu())
    try:
        window._on_image_progress(ProgressInfo(current=2, total=5, message="2/5"))
        assert window.image_progress.maximum() == 1000
        assert window.image_progress.value() == 400

        window._on_video_progress(ProgressInfo(current=3, total=10, message="3/10"))
        assert window.video_progress.maximum() == 1000
        assert window.video_progress.value() == 300
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_image_progress_rounds_fractional_value_and_logs_the_image_once(qapp):
    """Tile detail comes from the pipeline's step log, so progress only needs to
    announce each image once — echoing every tile callback as well would double
    every line in the panel."""
    window = MainWindow(make_gpu())
    try:
        progress = ProgressInfo(
            current=2.5,
            total=5,
            message="第 3/5 张 · tile 1/2 · a.png",
            image_index=3,
            image_total=5,
            source_name="a.png",
            phase="tile",
        )
        window._on_image_progress(progress)
        window._on_image_progress(progress)

        assert window.image_progress.maximum() == 1000
        assert window.image_progress.value() == 500
        text = window.image_log.text_edit.toPlainText()
        assert text.count("处理 a.png") == 1
        assert "tile 1/2" not in text
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_pipeline_step_log_lines_reach_the_panel(qapp):
    window = MainWindow(make_gpu())
    try:
        window._on_image_log("Info", "[1/2] 推理完成 · 0.46s · 4096x4096 · 36.6 MP/s")
        window._on_image_log("Error", "失败 b.png：模型输出尺寸与 tile 倍率不一致")

        text = window.image_log.text_edit.toPlainText()
        assert "[1/2] 推理完成 · 0.46s · 4096x4096 · 36.6 MP/s" in text
        assert "模型输出尺寸与 tile 倍率不一致" in text
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_image_finished_logs_save_and_summary(qapp):
    window = MainWindow(make_gpu())
    try:
        output = Path("output/a_x2.png")
        window._on_image_finished(
            [
                SuperResResult(
                    ok=True,
                    output_path=output,
                    elapsed_seconds=1.25,
                    input_path=Path("input/a.png"),
                )
            ]
        )

        text = window.image_log.text_edit.toPlainText()
        assert f"保存到 {output}" in text
        assert "成功 1 / 失败 0" in text
        assert "总耗时 1.2s" in text
        assert window.image_progress.maximum() == 1000
        assert window.image_progress.value() == 1000
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_engine_sessions_are_reused_between_tasks(qapp):
    """A repeated task must not pay the model load again (0.3s for the light
    models, ~3s for SwinIR)."""
    from app.ui import workers

    workers._ENGINE_CACHE.clear()
    meta = ModelMeta(
        name="cached-model",
        scale=4,
        scene=Scene.REALISTIC,
        backend=Backend.CPU,
        model_path=Path("models/onnx/realesr-general-x4v3.onnx"),
        input_name="input",
        output_name="output",
    )
    loaded = []

    class FakeEngine:
        def __init__(self, backend):
            self.backend = backend
            self.resets = 0
            loaded.append(self)

        def load_model(self, model):
            self.model = model

        def reset(self):
            self.resets += 1

        def cancel(self):
            pass

    original = workers.OnnxEngine
    workers.OnnxEngine = FakeEngine
    try:
        first = workers._load_engine(Backend.CPU, meta)
        second = workers._load_engine(Backend.CPU, meta)
        other = workers._load_engine(
            Backend.CPU, replace(meta, model_path=Path("models/onnx/other.onnx"))
        )

        assert first is second, "same model + backend should reuse the session"
        assert len(loaded) == 2, "a different model must load a fresh session"
        assert first.resets == 1
        assert other is not first
        assert workers._ENGINE_CACHE == {
            (Backend.CPU.value, str(Path("models/onnx/other.onnx"))): other
        }
    finally:
        workers.OnnxEngine = original
        workers._ENGINE_CACHE.clear()


def test_output_size_options_include_percentage_scaling(qapp):
    window = MainWindow(make_gpu())
    try:
        combo = window.image_resize_mode_combo
        assert [combo.itemData(i) for i in range(combo.count())] == [
            "keep",
            "restore",
            "percent",
        ]
        assert "百分比" in combo.itemText(2)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_percent_field_only_appears_for_percentage_mode(qapp):
    window = MainWindow(make_gpu())
    try:
        spin = window.image_resize_percent_spin
        assert spin.isHidden(), "percent field must start hidden"

        window.image_resize_mode_combo.setCurrentIndex(
            window.image_resize_mode_combo.findData("restore")
        )
        assert spin.isHidden()

        window.image_resize_mode_combo.setCurrentIndex(
            window.image_resize_mode_combo.findData("percent")
        )
        assert not spin.isHidden()
        assert spin.value() == 50  # default keeps the common case one click away
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_resize_options_map_to_the_pipeline_arguments(qapp):
    window = MainWindow(make_gpu())
    try:
        combo = window.image_resize_mode_combo
        combo.setCurrentIndex(combo.findData("keep"))
        assert window.resize_options() == (False, None)

        combo.setCurrentIndex(combo.findData("restore"))
        assert window.resize_options() == (True, None)

        combo.setCurrentIndex(combo.findData("percent"))
        window.image_resize_percent_spin.setValue(75)
        assert window.resize_options() == (False, 75.0)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_resize_percent_round_trips_through_settings(qapp, monkeypatch):
    monkeypatch.setattr(
        main_window,
        "load_settings",
        lambda: {"image_resize_mode": "percent", "image_resize_percent": 75},
    )
    window = MainWindow(make_gpu())
    try:
        assert window.resize_mode() == "percent"
        assert window.resize_percent() == 75.0
        assert window._window_settings()["image_resize_percent"] == 75
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_saved_resize_percent_is_clamped(qapp, monkeypatch):
    monkeypatch.setattr(
        main_window,
        "load_settings",
        lambda: {"image_resize_mode": "percent", "image_resize_percent": 9999},
    )
    window = MainWindow(make_gpu())
    try:
        assert window.image_resize_percent_spin.value() == PERCENT_MAX
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_worker_receives_the_resize_percent(qapp, monkeypatch, tmp_path):
    source = tmp_path / "a.png"
    source.write_bytes(b"not needed: the worker is never started")
    window = MainWindow(make_gpu())
    try:
        window.image_output_dir_edit.setText(str(tmp_path / "out"))
        window._image_path_set = lambda: {source}
        window._image_paths = lambda: [source]
        window.image_resize_mode_combo.setCurrentIndex(
            window.image_resize_mode_combo.findData("percent")
        )
        window.image_resize_percent_spin.setValue(60)

        started = {}

        class _SignalStub:
            def connect(self, *args, **kwargs):
                return None

        class RecordingWorker:
            progress = runtime = log = finished = error = cancelled = _SignalStub()

            def __init__(self, **kwargs):
                started.update(kwargs)

            def start(self):
                return None

        monkeypatch.setattr(main_window, "ImageUpscaleWorker", RecordingWorker)
        model = window.image_model_selector.current_model()
        assert model.model_path.is_file(), "test needs a real built-in model"

        window.start_image_upscale()

        assert started["resize_percent"] == 60.0
        assert started["resize_to_original"] is False
    finally:
        window._image_worker = None  # the stub is not a QThread; nothing to close
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_percent_field_gets_a_usable_size_and_sits_in_its_row(qapp):
    """The percentage field appears inside the 输出尺寸 row; it must not be
    squeezed into an unclickable size next to the combo."""
    window = MainWindow(make_gpu())
    try:
        window.resize(1240, 900)
        window.show()
        for _ in range(3):
            qapp.processEvents()
        window.grab()  # forces the scroll page to run its layout
        for _ in range(3):
            qapp.processEvents()

        combo = window.image_resize_mode_combo
        spin = window.image_resize_percent_spin
        combo.setCurrentIndex(combo.findData("percent"))
        qapp.processEvents()
        window.grab()
        qapp.processEvents()

        row = spin.parentWidget()
        assert spin.width() >= 44 and spin.height() >= 22, (spin.width(), spin.height())
        assert abs(
            combo.geometry().center().y() - spin.geometry().center().y()
        ) <= 2, "percent field must sit on the same line as the combo"
        assert combo.width() + spin.width() + 8 == row.width(), (
            "combo + percent field must fill the row (no clipping)",
            combo.width(),
            spin.width(),
            row.width(),
        )
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
