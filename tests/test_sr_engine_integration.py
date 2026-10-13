from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.sr_engine import OnnxEngine
from app.core.types import Backend, ModelMeta, Scene


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(
        np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2)
    )
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _low_frequency_plate(size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """A flat-block / slow-gradient image plus its bicubic 1/2 downscale.

    Deliberately low-frequency: a correct 2x model reconstructs such content to
    25-28 dB (measured), while the DirectML miscomputation bug scored 6-8 dB —
    so this pair separates "runs" from "runs correctly" in a shape-blind way.
    """
    height = width = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    image = np.stack(
        [
            128 + 60 * np.sin(2 * np.pi * xx / width),
            128 + 60 * np.sin(2 * np.pi * yy / height),
            128 + 60 * np.cos(2 * np.pi * (xx + yy) / (width + height)),
        ],
        axis=-1,
    )
    image[24:72, 24:72] = (200, 90, 70)
    image[8:24, 8:88] = (60, 160, 210)
    image = np.clip(image, 0, 255).astype(np.uint8)
    lr = np.asarray(
        Image.fromarray(image).resize(
            (width // 2, height // 2), Image.Resampling.BICUBIC
        )
    )
    return image, lr


#: 保真度下限：正确模型 ≥25 dB，坏掉的 DirectML 计算 6-8 dB，取 18 dB 留裕度。
_FIDELITY_FLOOR_DB = 18.0


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

    image, lr = _low_frequency_plate()

    output = engine.upscale_image(lr, scale=2)
    assert output.shape == (image.shape[0], image.shape[1], 3)
    assert output.dtype == np.uint8
    # The model is a learned 2x super-resolver, not a nearest resize, so the
    # output must differ from `np.repeat`-of-input at more than a few pixels.
    nearest = np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1)
    assert np.abs(output.astype(np.int16) - nearest.astype(np.int16)).mean() > 1.0
    # Fidelity: the reconstruction must match the source's low-frequency
    # content (shape-and-std-only checks once let a DirectML miscomputation
    # through with PSNR ~7 dB).
    assert _psnr(image, output) > _FIDELITY_FLOOR_DB, _psnr(image, output)


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
    must leave a smoother image than a weaker one).

    Crucially this also checks *fidelity*: the output must actually reconstruct
    the input's low-frequency content. A DirectML execution-provider bug once
    made these models run silently wrong (PSNR ~7 dB) while still producing the
    correct shape and plausible-looking variants, so shape-only assertions are
    not enough.
    """
    from app.models.model_manager import builtin_models

    registry = {model.name: model for model in builtin_models()}

    image, lr = _low_frequency_plate()
    height, width = image.shape[:2]

    outputs: dict[str, np.ndarray] = {}
    for name in _CUGAN_2X_CHECKOUTS:
        meta = registry[name]
        assert meta.scale == 2 and meta.scene == Scene.ANIME_2D
        engine = OnnxEngine(Backend.DIRECTML)
        engine.load_model(meta)
        assert "DmlExecutionProvider" in engine._session.get_providers()
        assert engine.precision_label == "fp16"
        result = engine.upscale_image(lr, scale=2)
        assert result.shape == (height, width, 3)  # (h//2 * 2, w//2 * 2)
        assert result.dtype == np.uint8
        # 保真度：输出必须重建出输入的低频内容（坏掉的 DML 版本只有 6-8 dB）
        assert _psnr(image, result) > _FIDELITY_FLOOR_DB, (name, _psnr(image, result))
        outputs[name] = result

    # 每一档都是独立权重：与 no-denoise 档的输出必须有可见差别。
    baseline = outputs["realcugan-2x-no-denoise"].astype(np.int16)
    for name in _CUGAN_2X_CHECKOUTS[1:]:
        diff = np.abs(outputs[name].astype(np.int16) - baseline)
        assert diff.mean() > 1.0, (name, float(diff.mean()))

    def smoothness(a: np.ndarray) -> float:
        return float(np.abs(np.diff(a.astype(np.float32), axis=1)).mean())

    # 降噪档位语义：在**带噪声**的输入上，强降噪档必须显著更平滑。
    # （干净图上各档平滑度并不单调——没有噪声可降时档位差异不体现在这里——
    # 所以断言只在加噪场景下做，与 Set5 noisy2x 基线的结论一致。）
    rng = np.random.default_rng(11)
    noisy_lr = np.clip(
        lr.astype(np.float64) + rng.normal(0.0, 25.0, lr.shape), 0, 255
    ).astype(np.uint8)

    smoothed: dict[str, float] = {}
    for name in ("realcugan-2x-no-denoise", "realcugan-2x-denoise3x"):
        engine = OnnxEngine(Backend.DIRECTML)
        engine.load_model(registry[name])
        smoothed[name] = smoothness(engine.upscale_image(noisy_lr, scale=2))

    assert smoothed["realcugan-2x-denoise3x"] < smoothed["realcugan-2x-no-denoise"], smoothed
