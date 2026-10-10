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
