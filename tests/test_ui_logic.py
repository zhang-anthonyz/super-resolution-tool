from pathlib import Path

import pytest

from app.core.types import Backend, GpuInfo, GpuType, GpuVendor, Scene
from app.ui.ui_logic import (
    DEFAULT_MODEL_RELATIVE_PATH,
    backend_choices,
    build_model_meta,
    collect_image_files,
    default_model_path,
    effective_scale,
    fixed_scale_for_model,
    format_log_line,
    gpu_summary,
    is_8bit_sdr,
    recommended_backend_index,
    video_8bit_notice,
)


def make_gpu(
    vendor: GpuVendor = GpuVendor.AMD,
    name: str = "AMD Radeon RX 7900 XT",
) -> GpuInfo:
    return GpuInfo(
        vendor=vendor,
        name=name,
        type=GpuType.DISCRETE,
        dedicated_vram_bytes=24 * 1024**3,
        vendor_id=0x1002,
    )


@pytest.mark.parametrize("scene", list(Scene))
def test_scene_default_model_mapping(scene):
    assert default_model_path(scene) == DEFAULT_MODEL_RELATIVE_PATH

    meta = build_model_meta(default_model_path(scene), scene, Backend.DIRECTML)
    assert meta.name == "RealESRGAN_x4plus"
    assert meta.scene == scene
    assert meta.scale == 4


def test_backend_choices_show_amd_recommended_backend_and_unready_vulkan():
    gpu = make_gpu()
    choices = backend_choices(gpu, [Backend.DIRECTML, Backend.CPU])

    assert [choice.backend for choice in choices] == [
        Backend.DIRECTML,
        Backend.VULKAN,
        Backend.CPU,
    ]
    assert choices[0].available is True
    assert choices[0].display_label == "DirectML（推荐）"
    assert choices[1].available is False
    assert choices[1].display_label == "Vulkan（未就绪）"
    assert recommended_backend_index(choices) == 0


def test_backend_choices_keep_vulkan_when_runtime_available():
    gpu = make_gpu()
    choices = backend_choices(gpu, [Backend.DIRECTML, Backend.VULKAN, Backend.CPU])

    assert [choice.available for choice in choices] == [True, True, True]


def test_8bit_sdr_prompt_logic():
    assert is_8bit_sdr(8, False) is True
    assert is_8bit_sdr(10, False) is False
    assert is_8bit_sdr(8, True) is False
    assert video_8bit_notice() == "仅支持 8bit SDR 视频，10bit/HDR 暂不支持"


def test_fixed_scale_and_effective_scale():
    model = Path("models/onnx/RealESRGAN_x4plus.onnx")
    assert fixed_scale_for_model(model) == 4
    assert effective_scale(model, 2) == (4, True)
    assert effective_scale(Path("custom.onnx"), 3) == (3, False)


def test_collect_image_files_filters_and_recurses(tmp_path):
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "b.jpg").write_bytes(b"")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.webp").write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("no", encoding="utf-8")

    files = collect_image_files([tmp_path])

    assert [path.name for path in files] == ["a.png", "b.jpg", "c.webp"]


def test_format_log_line_has_timestamp_and_level():
    assert format_log_line("Info", "hello", timestamp="12:00:00") == (
        "[12:00:00] [Info] hello"
    )


def test_gpu_summary_contains_amd_and_model_name():
    summary = gpu_summary(make_gpu(), [Backend.DIRECTML, Backend.CPU])

    assert "AMD" in summary
    assert "RX 7900 XT" in summary
    assert "DirectML（推荐）" in summary

