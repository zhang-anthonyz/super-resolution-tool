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
