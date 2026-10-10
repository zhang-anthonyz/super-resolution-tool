from pathlib import Path

import pytest

from app.core.paths import app_root
from app.core.types import Backend, GpuInfo, GpuType, GpuVendor, Scene
from app.ui.ui_logic import (
    backend_choices,
    build_model_meta,
    collect_image_files,
    copy_model_to_library,
    default_model_path,
    effective_scale,
    fixed_scale_for_model,
    format_log_line,
    gpu_summary,
    is_8bit_sdr,
    model_options,
    model_scales,
    progress_bar_value,
    recommended_backend_index,
    supported_model_formats,
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
    path = default_model_path(scene)
    assert path.parent == app_root() / "models" / "onnx"
    assert path.suffix == ".onnx"

    meta = build_model_meta(path, scene, Backend.DIRECTML)
    assert meta.name == path.stem
    assert meta.scene == scene
    assert meta.scale == 4


def test_model_scales_are_fixed_or_scale_aware():
    fixed = build_model_meta(
        app_root() / "models" / "onnx" / "realesr-general-x4v3.onnx",
        Scene.REALISTIC,
        Backend.DIRECTML,
    )
    assert model_scales(fixed) == [4]

    scale_aware = build_model_meta(
        app_root() / "models" / "onnx" / "custom.onnx",
        Scene.REALISTIC,
        Backend.DIRECTML,
        fixed_scale=None,
    )
    assert model_scales(scale_aware) == [2, 3, 4]


def test_model_options_expose_fp16_registry_with_denoise_hints():
    options = model_options()

    assert [name for name, _ in options] == [
        "realesr-general-x4v3",
        "RealESRGAN_x2plus",
        "RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x4plus_fp16",
    ]
    assert any("降噪" in display for _, display in options)


def test_supported_model_formats_and_copy_model_to_library(tmp_path):
    formats = supported_model_formats()
    assert ".onnx" in formats
    assert ".json" in formats

    source = tmp_path / "custom.onnx"
    source.write_bytes(b"onnx")
    target = copy_model_to_library(source, tmp_path / "library")
    assert target == (tmp_path / "library" / "custom.onnx")
    assert target.read_bytes() == b"onnx"

    renamed = copy_model_to_library(source, tmp_path / "library")
    assert renamed == (tmp_path / "library" / "custom (1).onnx")
    assert renamed.exists()


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


def test_progress_bar_value_rounds_half_up():
    assert progress_bar_value(2.5) == 3
    assert progress_bar_value(2.4) == 2
    assert progress_bar_value(0.0) == 0


def test_gpu_summary_contains_amd_and_model_name():
    summary = gpu_summary(make_gpu(), [Backend.DIRECTML, Backend.CPU])

    assert "AMD" in summary
    assert "RX 7900 XT" in summary
    assert "DirectML（推荐）" in summary
