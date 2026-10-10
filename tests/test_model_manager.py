from pathlib import Path

import pytest

from app.models.model_manager import (
    builtin_models,
    download_model,
    import_custom_model,
    scene_default,
)
from app.core.types import Backend, ModelMeta, Scene


def test_builtin_models_contains_real_esrgan_x4plus():
    models = builtin_models()

    assert len(models) == 1
    model = models[0]
    assert model.name == "RealESRGAN_x4plus"
    assert model.scale == 4
    assert model.scene == Scene.REALISTIC
    assert model.backend == Backend.DIRECTML
    assert model.model_path == Path("models/onnx/RealESRGAN_x4plus.onnx")


def test_scene_default_maps_realistic_to_real_esrgan():
    model = scene_default(Scene.REALISTIC)

    assert model.name == "RealESRGAN_x4plus"
    assert model.scale == 4


def test_scene_default_rejects_unregistered_scene():
    with pytest.raises(ValueError, match="No default model"):
        scene_default(Scene.ANIME_2D)


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

