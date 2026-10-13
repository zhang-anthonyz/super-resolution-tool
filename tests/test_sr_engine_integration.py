from pathlib import Path

import numpy as np
import pytest

from app.core.sr_engine import OnnxEngine
from app.core.types import Backend, ModelMeta, Scene


def _save_nearest_2x_model(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        ["N", "C", "H", "W"],
    )
    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        ["N", "C", "H2", "W2"],
    )
    scales = helper.make_tensor(
        "scales",
        TensorProto.FLOAT,
        [4],
        [1.0, 1.0, 2.0, 2.0],
    )
    node = helper.make_node(
        "Resize",
        inputs=["input", "", "scales"],
        outputs=["output"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph(
        [node],
        "nearest2x",
        [input_info],
        [output_info],
        [scales],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _dml_available() -> bool:
    try:
        import onnxruntime
    except ImportError:
        return False
    return "DmlExecutionProvider" in onnxruntime.get_available_providers()


@pytest.mark.skipif(not _dml_available(), reason="DirectML provider is not available")
def test_directml_nearest_2x_integration(tmp_path):
    pytest.importorskip("onnx")
    model_path = tmp_path / "nearest2x.onnx"
    _save_nearest_2x_model(model_path)

    model = ModelMeta(
        name="nearest2x",
        scale=2,
        scene=Scene.REALISTIC,
        backend=Backend.DIRECTML,
        model_path=model_path,
        input_name="input",
        output_name="output",
        tile_size=256,
    )
    engine = OnnxEngine(Backend.DIRECTML)
    engine.load_model(model)
    # 关键断言：确认 DirectML 真正生效，而非静默 fallback 到 CPU
    assert "DmlExecutionProvider" in engine._session.get_providers()

    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    output = engine.upscale_image(image, scale=2)
    expected = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

    assert output.shape == (128, 128, 3)
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output, expected)


@pytest.mark.skipif(
    not Path("models/onnx/RealESRGAN_x2plus.onnx").is_file(),
    reason="RealESRGAN_x2plus ONNX is not shipped in this checkout",
)
def test_realesrgan_x2plus_loaded_and_shape_correct():
    """The official RealESRGAN x2plus (fp16) loads under DirectML and
    multiplies H/W by 2 on a small even-sized input. Pixel-unshuffle
    inside the model means H/W must be even; the engine's tiling already
    honours this.
    """
    model = ModelMeta(
        name="RealESRGAN_x2plus",
        scale=2,
        scene=Scene.REALISTIC,
        backend=Backend.DIRECTML,
        model_path=Path("models/onnx/RealESRGAN_x2plus.onnx"),
        input_name="input",
        output_name="output",
        tile_size=256,
    )
    engine = OnnxEngine(Backend.DIRECTML)
    engine.load_model(model)
    assert "DmlExecutionProvider" in engine._session.get_providers()
    assert engine.precision_label == "fp16"

    rng = np.random.default_rng(2026)
    # tile_size=256 splits larger images, so 64x64 lands in a single tile.
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    output = engine.upscale_image(image, scale=2)
    assert output.shape == (128, 128, 3)
    assert output.dtype == np.uint8
    # The model is a learned 2x super-resolver, not a nearest resize, so the
    # output must differ from `np.repeat`-of-input at more than a few pixels.
    nearest = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    assert np.abs(output.astype(np.int16) - nearest.astype(np.int16)).mean() > 1.0
    # And it must not be saturated/garbage: the per-pixel std stays in the
    # natural range for a photo (4 MP equivalent of detail).
    assert 5.0 < output.std() < 80.0


_CUGAN_2X_CHECKOUTS = tuple(
    f"realcugan-2x-{variant}"
    for variant in ("no-denoise", "denoise1x", "denoise2x", "denoise3x", "conservative")
)

_cugan_available = all(
    Path(f"models/onnx/{name}.onnx").is_file() for name in _CUGAN_2X_CHECKOUTS
)


@pytest.mark.skipif(not _cugan_available, reason="Real-CUGAN 2x ONNX files are not shipped")
def test_realcugan_2x_variants_load_and_differ_by_denoise_strength():
    """All five Real-CUGAN 2x denoise variants run under DirectML, produce the
    right shape, and are genuinely different models (a stronger denoise setting
    must leave a smoother image than a weaker one)."""
    from app.models.model_manager import builtin_models

    registry = {model.name: model for model in builtin_models()}
    rng = np.random.default_rng(11)
    # Anime-ish test patch: flat blocks, a hard colour edge, and grain.
    height = width = 96
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.stack(
        [
            (xx * 2 % 256).astype(np.uint8),
            (yy * 2 % 256).astype(np.uint8),
            ((xx + yy) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    image[30:60, 30:70] = (240, 30, 60)
    grain = rng.integers(-10, 10, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    outputs: dict[str, np.ndarray] = {}
    for name in _CUGAN_2X_CHECKOUTS:
        meta = registry[name]
        assert meta.scale == 2 and meta.scene == Scene.ANIME_2D
        engine = OnnxEngine(Backend.DIRECTML)
        engine.load_model(meta)
        assert "DmlExecutionProvider" in engine._session.get_providers()
        assert engine.precision_label == "fp16"
        result = engine.upscale_image(image, scale=2)
        assert result.shape == (height * 2, width * 2, 3)
        assert result.dtype == np.uint8
        outputs[name] = result

    # 每一档都是独立权重：与 no-denoise 档的输出必须有可见差别。
    baseline = outputs["realcugan-2x-no-denoise"].astype(np.int16)
    for name in _CUGAN_2X_CHECKOUTS[1:]:
        diff = np.abs(outputs[name].astype(np.int16) - baseline)
        assert diff.mean() > 1.0, (name, float(diff.mean()))

    def smoothness(a: np.ndarray) -> float:
        return float(np.abs(np.diff(a.astype(np.float32), axis=1)).mean())

    # 强降噪档必须比无降噪档更平滑（这是档位命名的含义所在）。
    assert smoothness(outputs["realcugan-2x-denoise3x"]) < smoothness(
        outputs["realcugan-2x-no-denoise"]
    )
