import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core import sr_engine
from app.core.sr_engine import SuperResCancelled, VulkanEngine
from app.core.types import Backend, ModelMeta, Scene


def make_model(name: str = "RealESRGAN_x4plus", scale: int | None = 4) -> ModelMeta:
    return ModelMeta(
        name=name,
        scale=scale,
        scene=Scene.REALISTIC,
        backend=Backend.VULKAN,
        model_path=Path(f"{name}.onnx"),
    )


def _prepare_vendor(tmp_path: Path) -> tuple[Path, Path]:
    vendor = tmp_path / "vendor"
    executable = vendor / "realesrgan-ncnn-vulkan.exe"
    model_dir = vendor / "ncnn-models"
    executable.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"fake")
    return executable, model_dir


def test_ncnn_model_name_mapping():
    assert VulkanEngine._ncnn_model_name(make_model("RealESRGAN_x4plus"), 4) == (
        "realesrgan-x4plus"
    )
    assert VulkanEngine._ncnn_model_name(
        make_model("RealESRGAN_x4plus_anime_6B"), 4
    ) == "realesrgan-x4plus-anime"
    assert VulkanEngine._ncnn_model_name(make_model("animevideov3"), 3) == (
        "realesr-animevideov3-x3"
    )
    with pytest.raises(ValueError):
        VulkanEngine._ncnn_model_name(make_model("custom-model"), 4)


def test_load_model_records_ncnn_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(sr_engine, "app_root", lambda: tmp_path)
    executable, model_dir = _prepare_vendor(tmp_path)
    engine = VulkanEngine(Backend.VULKAN)

    engine.load_model(make_model())

    assert engine._ncnn_path == executable
    assert engine._model_dir == model_dir


def test_load_model_requires_ncnn_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(sr_engine, "app_root", lambda: tmp_path)
    engine = VulkanEngine(Backend.VULKAN)

    with pytest.raises(FileNotFoundError, match="ncnn"):
        engine.load_model(make_model())


def test_upscale_image_uses_ncnn_cli_and_reports_file_progress(tmp_path, monkeypatch):
    executable, model_dir = _prepare_vendor(tmp_path)
    engine = VulkanEngine(Backend.VULKAN)
    engine._model = make_model()
    engine._ncnn_path = executable
    engine._model_dir = model_dir

    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        with Image.open(input_path) as source:
            source_array = np.asarray(source.convert("RGB"))
        upscaled = np.repeat(np.repeat(source_array, 2, axis=0), 2, axis=1)
        Image.fromarray(upscaled, "RGB").save(output_path)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sr_engine.subprocess, "run", fake_run)
    image = np.array(
        [[[0, 10, 20], [30, 40, 50]], [[60, 70, 80], [90, 100, 110]]],
        dtype=np.uint8,
    )
    progress = []

    output = engine.upscale_image(
        image,
        scale=2,
        progress_cb=lambda info: progress.append((info.current, info.total)),
    )

    assert commands
    command = commands[0]
    assert command[command.index("-n") + 1] == "realesrgan-x4plus"
    assert command[command.index("-s") + 1] == "2"
    assert output.shape == (4, 4, 3)
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(
        output,
        np.repeat(np.repeat(image, 2, axis=0), 2, axis=1),
    )
    assert progress == [(0, 1), (1, 1)]


def test_upscale_image_checks_cancelled_before_spawning_subprocess(tmp_path, monkeypatch):
    executable, model_dir = _prepare_vendor(tmp_path)
    engine = VulkanEngine(Backend.VULKAN)
    engine._model = make_model()
    engine._ncnn_path = executable
    engine._model_dir = model_dir
    engine.cancel()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess should not run after cancellation")

    monkeypatch.setattr(sr_engine.subprocess, "run", fail_if_called)

    with pytest.raises(SuperResCancelled):
        engine.upscale_image(np.zeros((2, 2, 3), dtype=np.uint8), scale=2)


def test_upscale_image_raises_on_ncnn_failure(tmp_path, monkeypatch):
    executable, model_dir = _prepare_vendor(tmp_path)
    engine = VulkanEngine(Backend.VULKAN)
    engine._model = make_model()
    engine._ncnn_path = executable
    engine._model_dir = model_dir

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="vulkan failed")

    monkeypatch.setattr(sr_engine.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="vulkan failed"):
        engine.upscale_image(np.zeros((2, 2, 3), dtype=np.uint8), scale=2)
