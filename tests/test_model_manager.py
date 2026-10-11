from pathlib import Path

import pytest

from app.models.model_manager import (
    builtin_models,
    denoise_models,
    download_model,
    import_custom_model,
    import_custom_model_to_library,
    registered_custom_models,
    scene_default,
)
from app.core.types import Backend, ModelMeta, Scene
from app.ui.ui_logic import resolve_project_path


def test_builtin_models_cover_real_esrgan_and_swinir():
    models = builtin_models()

    assert len(models) == 7
    by_name = {model.name: model for model in models}
    assert set(by_name) == {
        "realesr-general-x4v3",
        "RealESRGAN_x2plus",
        "RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x4plus_fp16",
        "SwinIR_x4_classical",
        "SwinIR_x4_realSR_GAN",
        "4x-UltraSharp_fp16",
    }

    assert by_name["RealESRGAN_x2plus"].scale == 2
    assert by_name["SwinIR_x4_classical"].scale == 4
    assert by_name["4x-UltraSharp_fp16"].scale == 4
    for model in models:
        assert model.scale in {2, 4}
        assert model.backend == Backend.DIRECTML
        assert model.model_path.suffix == ".onnx"
        assert model.kind == "sr"
        assert "降噪" in model.description or "无降噪" in model.description


def test_denoise_models_are_registered_separately():
    models = denoise_models()
    by_name = {model.name: model for model in models}

    assert set(by_name) == {"realesr-general-wdn-x4v3", "SwinIR_colorDN25"}
    assert all(model.kind == "denoise" for model in models)
    # 1x 降噪模型走"输入尺度预降噪"，x4 模型与主模型输出按强度插值
    assert by_name["SwinIR_colorDN25"].scale == 1
    assert by_name["realesr-general-wdn-x4v3"].scale == 4
    # 降噪模型不得混入超分模型下拉列表
    assert not (set(by_name) & {model.name for model in builtin_models()})


def test_scene_default_maps_realistic_to_real_esrgan():
    model = scene_default(Scene.REALISTIC)

    assert model.name == "realesr-general-x4v3"
    assert model.scale == 4


def test_scene_default_maps_all_scenes_to_registered_models():
    assert scene_default(Scene.REALISTIC).name == "realesr-general-x4v3"
    assert scene_default(Scene.ANIME_2D).name == "RealESRGAN_x4plus_anime_6B"
    assert scene_default(Scene.CG_3D).name == "realesr-general-x4v3"


def _save_scale_aware_identity_model(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    image_input = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        ["N", 3, "H", "W"],
    )
    scale_input = helper.make_tensor_value_info(
        "scale",
        TensorProto.FLOAT,
        [],
    )
    output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        ["N", 3, "H", "W"],
    )
    node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["output"],
    )
    graph = helper.make_graph(
        [node],
        "scale_aware_identity",
        [image_input, scale_input],
        [output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_import_custom_model_reads_metadata_and_detects_scale_input(tmp_path):
    onnx_path = tmp_path / "custom.onnx"
    meta_path = tmp_path / "custom.json"
    _save_scale_aware_identity_model(onnx_path)
    meta_path.write_text(
        (
            '{"name": "custom", "scene": "anime_2d", "backend": "cpu", '
            '"tile_size": 256, "description": "test model"}'
        ),
        encoding="utf-8",
    )

    model = import_custom_model(onnx_path, meta_path)

    assert model.name == "custom"
    assert model.scene == Scene.ANIME_2D
    assert model.backend == Backend.CPU
    assert model.model_path == onnx_path
    assert model.scale is None
    assert model.scale_input == "scale"
    assert model.tile_size == 256
    assert model.description == "test model"


def test_import_custom_model_to_library_registers_model(tmp_path):
    onnx_path = tmp_path / "registered.onnx"
    _save_scale_aware_identity_model(onnx_path)
    models_dir = tmp_path / "library"
    registry_path = tmp_path / "custom_models.json"

    model = import_custom_model_to_library(
        onnx_path,
        models_dir=models_dir,
        registry_path=registry_path,
    )

    assert model.name == "registered"
    assert model.model_path.parent == models_dir
    assert model.scale_input == "scale"
    models = registered_custom_models(registry_path)
    assert [item.name for item in models] == ["registered"]
    assert resolve_project_path(models[0].model_path) == model.model_path


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_download_model_writes_and_verifies_sha256(tmp_path, monkeypatch):
    import app.models.model_manager as manager

    payload = b"fake-model-bytes"
    dest = tmp_path / "model.onnx"
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()

    def fake_urlopen(url, *args, **kwargs):
        return _FakeResponse([payload])

    monkeypatch.setattr(manager.urllib.request, "urlopen", fake_urlopen)

    result = download_model("https://example.invalid/model.onnx", dest, sha256=digest)

    assert result == dest
    assert dest.read_bytes() == payload


def test_download_model_rejects_sha256_mismatch(tmp_path, monkeypatch):
    import app.models.model_manager as manager

    payload = b"fake-model-bytes"
    dest = tmp_path / "model.onnx"

    def fake_urlopen(url, *args, **kwargs):
        return _FakeResponse([payload])

    monkeypatch.setattr(manager.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_model(
            "https://example.invalid/model.onnx",
            dest,
            sha256="0" * 64,
        )

    assert not dest.exists()
