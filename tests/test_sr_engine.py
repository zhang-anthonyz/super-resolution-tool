from pathlib import Path

import numpy as np
import pytest

from app.core.sr_engine import OnnxEngine, SuperResCancelled
from app.core.types import Backend, ModelMeta, Scene


class FakeSession:
    def __init__(self, scale=1):
        self.scale = scale
        self.feeds = []
        self.calls = 0
        self.fail_once_with_oom = False

    def run(self, output_names, feeds):
        self.calls += 1
        self.feeds.append(dict(feeds))
        if self.fail_once_with_oom:
            self.fail_once_with_oom = False
            raise RuntimeError("DML out of memory")
        tile = feeds["input"]
        return [
            np.repeat(
                np.repeat(tile, self.scale, axis=2),
                self.scale,
                axis=3,
            )
        ]


class _FakeMeta:
    """Minimal stand-in for an onnxruntime NodeArg."""

    def __init__(self, name, type_):
        self.name = name
        self.type = type_


class FakeTypedSession(FakeSession):
    """Fake session that reports an ONNX input dtype like a real session."""

    def __init__(self, scale=1, input_type="tensor(float)"):
        super().__init__(scale)
        self.input_type = input_type

    def get_inputs(self):
        return [_FakeMeta("input", self.input_type)]

    def get_outputs(self):
        return [_FakeMeta("output", self.input_type)]


class FakeValueSession:
    def __init__(self, scale, value):
        self.scale = scale
        self.value = float(value)

    def run(self, output_names, feeds):
        tile = feeds["input"]
        shape = (
            tile.shape[0],
            tile.shape[1],
            tile.shape[2] * self.scale,
            tile.shape[3] * self.scale,
        )
        return [np.full(shape, self.value, dtype=np.float32)]


def make_model(
    scale=2,
    tile_size=256,
    scale_input=None,
    denoise_input=None,
) -> ModelMeta:
    return ModelMeta(
        name="fake",
        scale=scale,
        scene=Scene.REALISTIC,
        backend=Backend.DIRECTML,
        model_path=Path("fake.onnx"),
        input_name="input",
        output_name="output",
        scale_input=scale_input,
        denoise_input=denoise_input,
        tile_size=tile_size,
    )


def make_engine(fake_session, model):
    engine = OnnxEngine(Backend.DIRECTML)
    engine._session = fake_session
    engine._model = model
    engine._input_name = "input"
    engine._output_name = "output"
    engine._output_channels = 3
    return engine


def make_paired_engine(weak_value=0.2, strong_value=0.6):
    model = make_model(scale=2)
    weak_session = FakeValueSession(scale=2, value=weak_value)
    strong_session = FakeValueSession(scale=2, value=strong_value)
    engine = make_engine(weak_session, model)
    engine._pair_session = strong_session
    engine._pair_input_name = "input"
    engine._pair_output_name = "output"
    engine._pair_output_channels = 3
    return engine


def test_upscale_image_produces_expected_nearest_output():
    session = FakeSession(scale=2)
    model = make_model(scale=2)
    engine = make_engine(session, model)
    image = np.array(
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )

    output = engine.upscale_image(image, scale=2)
    expected = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

    assert output.shape == (4, 4, 3)
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output, expected)


def test_progress_callback_reports_every_tile():
    session = FakeSession(scale=2)
    model = make_model(scale=2, tile_size=3)
    engine = make_engine(session, model)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    reported = []

    engine.upscale_image(
        image,
        scale=2,
        progress_cb=lambda info: reported.append((info.current, info.total)),
    )

    assert len(reported) == 16
    assert [current for current, _ in reported] == list(range(1, 17))
    assert {total for _, total in reported} == {16}


def test_cancel_before_tile_raises_without_running_session():
    session = FakeSession(scale=2)
    model = make_model(scale=2)
    engine = make_engine(session, model)
    engine.cancel()

    with pytest.raises(SuperResCancelled):
        engine.upscale_image(np.zeros((2, 2, 3), dtype=np.uint8), scale=2)

    assert session.calls == 0


def test_scale_and_denoise_inputs_are_added_to_feeds():
    session = FakeSession(scale=2)
    model = make_model(scale=2, scale_input="scale", denoise_input="denoise")
    engine = make_engine(session, model)

    engine.upscale_image(np.zeros((2, 2, 3), dtype=np.uint8), scale=2)

    assert set(session.feeds[0]) == {"input", "scale", "denoise"}
    np.testing.assert_array_equal(session.feeds[0]["scale"], [2.0])
    # denoise_input 现在由"降噪强度"驱动（upscale_image 默认 0.5），不再硬编码 0.0
    np.testing.assert_array_equal(session.feeds[0]["denoise"], [0.5])


def test_fp16_engine_feeds_float16_tiles_and_returns_uint8():
    session = FakeSession(scale=2)
    model = make_model(scale=2)
    engine = make_engine(session, model)
    engine._fp16 = True
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    output = engine.upscale_image(image, scale=2)

    assert session.feeds[0]["input"].dtype == np.float16
    assert output.shape == (4, 4, 3)
    assert output.dtype == np.uint8


def test_oom_during_large_tile_falls_back_to_256():
    session = FakeSession(scale=2)
    session.fail_once_with_oom = True
    model = make_model(scale=2, tile_size=512)
    engine = make_engine(session, model)
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    output = engine.upscale_image(image, scale=2)

    assert output.shape == (600, 600, 3)
    assert session.calls == 5
    widths = [feed["input"].shape[3] for feed in session.feeds]
    assert widths == [300, 256, 54, 256, 54]


def test_denoise_strength_interpolates_between_weak_and_strong_outputs():
    image = np.full((2, 2, 3), 51, dtype=np.uint8)

    weak = make_paired_engine().upscale_image(image, scale=2, denoise_strength=0.0)
    strong = make_paired_engine().upscale_image(image, scale=2, denoise_strength=1.0)
    balanced = make_paired_engine().upscale_image(image, scale=2, denoise_strength=0.5)

    np.testing.assert_array_equal(weak, np.full_like(weak, 51))
    np.testing.assert_array_equal(strong, np.full_like(strong, 153))
    np.testing.assert_array_equal(balanced, np.full_like(balanced, 102))


def test_denoise_strength_is_ignored_without_pair_session():
    image = np.full((2, 2, 3), 51, dtype=np.uint8)
    model = make_model(scale=2)
    engine = make_engine(FakeValueSession(scale=2, value=0.2), model)

    output = engine.upscale_image(image, scale=2, denoise_strength=1.0)

    np.testing.assert_array_equal(output, np.full_like(output, 51))


def test_load_model_rejects_missing_runtime_provider(tmp_path, monkeypatch):
    import app.core.sr_engine as sr_engine

    model_path = tmp_path / "fake.onnx"
    model_path.write_bytes(b"not-a-real-model")
    model = make_model(scale=2)
    model = ModelMeta(
        name=model.name,
        scale=model.scale,
        scene=model.scene,
        backend=Backend.CUDA,
        model_path=model_path,
        tile_size=model.tile_size,
    )
    class FakeOnnxRuntime:
        @staticmethod
        def get_available_providers():
            return []

    monkeypatch.setattr(
        sr_engine,
        "load_onnxruntime_for_backend",
        lambda backend: FakeOnnxRuntime,
    )

    engine = OnnxEngine(Backend.CUDA)

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        engine.load_model(model)


def _engine_with_typed_pair(main_type: str, pair_type: str, pair_scale: int = 2):
    """Engine whose main and denoise sessions advertise different dtypes."""
    model = make_model(scale=pair_scale)
    main = FakeTypedSession(scale=pair_scale, input_type=main_type)
    pair = FakeTypedSession(scale=pair_scale, input_type=pair_type)
    engine = make_engine(main, model)
    engine._fp16 = engine._is_fp16_session(main)  # noqa: SLF001
    engine._pair_session = pair  # noqa: SLF001
    engine._pair_fp16 = engine._is_fp16_session(pair)  # noqa: SLF001
    engine._pair_input_name = "input"  # noqa: SLF001
    engine._pair_output_name = "output"  # noqa: SLF001
    engine._pair_output_channels = 3  # noqa: SLF001
    engine._pair_scale = pair_scale  # noqa: SLF001
    return engine, main, pair


def test_denoise_session_gets_its_own_dtype_in_post_blend():
    """fp32 main + fp16 denoiser must not feed float32 into the pair session."""
    engine, main, pair = _engine_with_typed_pair("tensor(float)", "tensor(float16)")

    engine.upscale_image(
        np.zeros((4, 4, 3), dtype=np.uint8), scale=2, denoise_strength=0.5
    )

    assert main.feeds[0]["input"].dtype == np.float32
    assert pair.feeds[0]["input"].dtype == np.float16


def test_pre_denoise_feeds_the_pair_session_dtype():
    """fp16 main + fp32 SwinIR denoiser: the pair session needs float32."""
    engine, _main, pair = _engine_with_typed_pair(
        "tensor(float16)", "tensor(float)", pair_scale=1
    )

    engine.pre_denoise(np.zeros((8, 8, 3), dtype=np.uint8), 1.0)

    assert engine._pair_session.feeds[0]["input"].dtype == np.float32  # noqa: SLF001
    assert pair.feeds[0]["input"].dtype == np.float32


def test_inference_and_pair_dtype_are_tracked_separately():
    engine, _main, _pair = _engine_with_typed_pair(
        "tensor(float16)", "tensor(float)", pair_scale=1
    )

    assert engine._inference_dtype() is np.float16  # noqa: SLF001
    assert engine._pair_dtype() is np.float32  # noqa: SLF001


def test_tiled_run_releases_disk_buffers_synchronously(monkeypatch):
    """Cleanup must happen inside the call, not whenever GC gets around to it."""
    import tempfile

    from app.core import sr_engine

    monkeypatch.setattr(sr_engine, "_DISK_BUFFER_THRESHOLD_BYTES", 1)
    released = []
    monkeypatch.setattr(
        sr_engine.OnnxEngine,
        "_release_temp_buffers",
        staticmethod(lambda handles: released.append(len(handles))),
    )

    engine = make_engine(FakeSession(scale=2), make_model(scale=2))
    engine.upscale_image(np.zeros((8, 8, 3), dtype=np.uint8), scale=2)

    # 累加器 + 权重 + 均值缓冲都在磁盘上，一次调用内全部释放
    assert len(released) == 1 and released[0] >= 2, released


def test_tiled_run_leaves_no_temp_dirs(monkeypatch):
    """Disk scratch buffers (huge images) must not leak into %TEMP%."""
    import gc
    import tempfile

    from app.core import sr_engine

    monkeypatch.setattr(sr_engine, "_DISK_BUFFER_THRESHOLD_BYTES", 1)
    before = set(Path(tempfile.gettempdir()).glob("srt-buffer-*"))

    engine = make_engine(FakeSession(scale=2), make_model(scale=2))
    engine.upscale_image(np.zeros((8, 8, 3), dtype=np.uint8), scale=2)
    gc.collect()

    after = set(Path(tempfile.gettempdir()).glob("srt-buffer-*"))
    assert after - before == set(), f"leaked: {after - before}"


def test_cancelled_tiled_run_also_removes_disk_buffers(monkeypatch):
    import tempfile

    from app.core import sr_engine

    monkeypatch.setattr(sr_engine, "_DISK_BUFFER_THRESHOLD_BYTES", 1)
    before = set(Path(tempfile.gettempdir()).glob("srt-buffer-*"))

    engine = make_engine(FakeSession(scale=2), make_model(scale=2))
    engine.cancel()

    with pytest.raises(SuperResCancelled):
        engine.upscale_image(np.zeros((8, 8, 3), dtype=np.uint8), scale=2)

    after = set(Path(tempfile.gettempdir()).glob("srt-buffer-*"))
    assert after - before == set(), f"leaked: {after - before}"
