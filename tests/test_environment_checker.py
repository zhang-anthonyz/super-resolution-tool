import json
import subprocess
from pathlib import Path

import pytest

from app.core import environment_checker
from app.core.environment_checker import (
    EnvironmentCheckError,
    _bit_depth_from_pix_fmt,
    _vendor_from_id,
    available_backends,
    check_environment,
    detect_video_bit_depth,
    find_discrete_gpu,
)
from app.core.types import Backend, GpuInfo, GpuType, GpuVendor


def make_gpu(
    vendor: GpuVendor = GpuVendor.AMD,
    gpu_type: GpuType = GpuType.DISCRETE,
    dedicated_vram_bytes: int = 8 * 1024**3,
    name: str = "Test GPU",
) -> GpuInfo:
    return GpuInfo(
        vendor=vendor,
        name=name,
        type=gpu_type,
        dedicated_vram_bytes=dedicated_vram_bytes,
        vendor_id=0x1002,
    )


def test_find_discrete_gpu_returns_largest_discrete():
    small = make_gpu(dedicated_vram_bytes=4 * 1024**3)
    large = make_gpu(dedicated_vram_bytes=24 * 1024**3, name="RX 7900 XT")
    integrated = make_gpu(
        vendor=GpuVendor.INTEL,
        gpu_type=GpuType.INTEGRATED,
        dedicated_vram_bytes=0,
        name="UHD Graphics",
    )

    assert find_discrete_gpu([integrated, small, large]) is large
    assert find_discrete_gpu([integrated]) is None


def test_gpu_info_dedicated_vram_gb_conversion():
    gpu = make_gpu(dedicated_vram_bytes=24 * 1024**3)
    assert gpu.dedicated_vram_gb == 24.0


def test_vendor_id_mapping():
    assert _vendor_from_id(0x10DE) == GpuVendor.NVIDIA
    assert _vendor_from_id(0x1002) == GpuVendor.AMD
    assert _vendor_from_id(0x8086) == GpuVendor.INTEL
    assert _vendor_from_id(0x1234) == GpuVendor.UNKNOWN


def test_pix_fmt_bit_depth_parsing():
    assert _bit_depth_from_pix_fmt("yuv420p") == 8
    assert _bit_depth_from_pix_fmt("yuv420p10le") == 10
    assert _bit_depth_from_pix_fmt("yuv420p12le") == 12
    assert _bit_depth_from_pix_fmt("p010le") == 10
    assert _bit_depth_from_pix_fmt("nv12") == 8
    assert _bit_depth_from_pix_fmt("") == 8


def test_detect_video_bit_depth_parses_ffprobe_json(tmp_path, monkeypatch):
    ffprobe = tmp_path / "ffprobe.exe"
    ffprobe.write_bytes(b"")
    video = tmp_path / "sample.mkv"
    video.write_bytes(b"")
    monkeypatch.setattr(environment_checker, "_ffprobe_path", lambda: ffprobe)

    stdout = json.dumps(
        {
            "streams": [
                {
                    "pix_fmt": "yuv420p10le",
                    "color_primaries": "bt2020",
                    "color_transfer": "smpte2084",
                }
            ]
        }
    )

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(environment_checker.subprocess, "run", fake_run)

    assert detect_video_bit_depth(str(video)) == (10, True)


def test_detect_video_bit_depth_returns_sdr_default_when_ffprobe_missing(
    tmp_path, monkeypatch
):
    missing_ffprobe = tmp_path / "missing.exe"
    video = tmp_path / "sample.mkv"
    video.write_bytes(b"")
    monkeypatch.setattr(environment_checker, "_ffprobe_path", lambda: missing_ffprobe)

    assert detect_video_bit_depth(str(video)) == (8, False)


def test_available_backends_removes_vulkan_when_ncnn_missing(monkeypatch):
    gpu = make_gpu(vendor=GpuVendor.AMD)
    monkeypatch.setattr(environment_checker, "_ncnn_path", lambda: Path("missing.exe"))

    assert available_backends(gpu) == [Backend.DIRECTML, Backend.CPU]


def test_available_backends_keeps_vulkan_when_ncnn_exists(tmp_path, monkeypatch):
    ncnn = tmp_path / "realesrgan-ncnn-vulkan.exe"
    ncnn.write_bytes(b"")
    gpu = make_gpu(vendor=GpuVendor.AMD)
    monkeypatch.setattr(environment_checker, "_ncnn_path", lambda: ncnn)

    assert available_backends(gpu) == [Backend.DIRECTML, Backend.VULKAN, Backend.CPU]


def test_check_environment_raises_without_discrete_gpu(monkeypatch):
    integrated = make_gpu(
        vendor=GpuVendor.INTEL,
        gpu_type=GpuType.INTEGRATED,
        dedicated_vram_bytes=0,
        name="UHD Graphics",
    )
    monkeypatch.setattr(environment_checker, "detect_gpus", lambda: [integrated])

    with pytest.raises(EnvironmentCheckError) as exc_info:
        check_environment()

    assert exc_info.value.code == "NO_DISCRETE_GPU"


def test_amd_integrated_name_detection():
    from app.core.environment_checker import _is_amd_integrated_name

    assert _is_amd_integrated_name("AMD Radeon(TM) Graphics") is True
    assert _is_amd_integrated_name("AMD Radeon Graphics") is True
    assert _is_amd_integrated_name("Radeon Vega 8 Graphics") is True
    assert _is_amd_integrated_name("AMD Radeon RX 7900 XT") is False
    assert _is_amd_integrated_name("Radeon Pro W6800") is False


def test_classify_gpu_type_amd_igpu():
    from app.core.environment_checker import _classify_gpu_type

    # AMD 核显报告少量 dedicated（512MB），应判为核显
    assert (
        _classify_gpu_type(GpuVendor.AMD, "AMD Radeon(TM) Graphics", 509_374_464)
        == GpuType.INTEGRATED
    )
    # AMD 独显
    assert (
        _classify_gpu_type(GpuVendor.AMD, "AMD Radeon RX 7900 XT", 21_413_277_696)
        == GpuType.DISCRETE
    )
    # NVIDIA 独显
    assert (
        _classify_gpu_type(GpuVendor.NVIDIA, "NVIDIA GeForce RTX 4090", 24 * 1024**3)
        == GpuType.DISCRETE
    )
    # Intel 核显
    assert _classify_gpu_type(GpuVendor.INTEL, "Intel UHD Graphics", 0) == GpuType.INTEGRATED
    # Intel 独显
    assert (
        _classify_gpu_type(GpuVendor.INTEL, "Intel Arc A770", 16 * 1024**3)
        == GpuType.DISCRETE
    )


def test_dedupe_gpus():
    from app.core.environment_checker import _dedupe_gpus

    a = GpuInfo(
        vendor=GpuVendor.AMD,
        name="RX 7900 XT",
        type=GpuType.DISCRETE,
        dedicated_vram_bytes=21_413_277_696,
        vendor_id=0x1002,
        device_id=0x744C,
    )
    a_dup = GpuInfo(
        vendor=GpuVendor.AMD,
        name="RX 7900 XT",
        type=GpuType.DISCRETE,
        dedicated_vram_bytes=21_413_277_696,
        vendor_id=0x1002,
        device_id=0x744C,
    )
    igpu = GpuInfo(
        vendor=GpuVendor.AMD,
        name="Radeon Graphics",
        type=GpuType.INTEGRATED,
        dedicated_vram_bytes=0,
        vendor_id=0x1002,
        device_id=0x164E,
    )

    result = _dedupe_gpus([a, a_dup, igpu])
    assert len(result) == 2
    assert result[0].device_id == 0x744C
    assert result[1].device_id == 0x164E
