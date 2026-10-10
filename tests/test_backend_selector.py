import logging

import pytest

from app.core.backend_selector import BackendSelector
from app.core.types import Backend, GpuInfo, GpuType, GpuVendor


def make_gpu(vendor: GpuVendor) -> GpuInfo:
    return GpuInfo(
        vendor=vendor,
        name="Test GPU",
        type=GpuType.DISCRETE,
        dedicated_vram_bytes=8 * 1024**3,
        vendor_id=0x1002,
    )


@pytest.mark.parametrize(
    ("vendor", "expected"),
    [
        (GpuVendor.NVIDIA, Backend.CUDA),
        (GpuVendor.AMD, Backend.DIRECTML),
        (GpuVendor.INTEL, Backend.DIRECTML),
        (GpuVendor.UNKNOWN, Backend.CPU),
    ],
)
def test_default_backend_routes_by_vendor(vendor, expected):
    selector = BackendSelector(make_gpu(vendor))
    assert selector.default_backend() == expected


def test_select_uses_default_when_preferred_is_none():
    selector = BackendSelector(make_gpu(GpuVendor.AMD))
    assert selector.select(None) == Backend.DIRECTML


def test_select_returns_available_preferred_backend():
    selector = BackendSelector(make_gpu(GpuVendor.AMD))
    assert selector.select(Backend.DIRECTML) == Backend.DIRECTML


def test_select_falls_back_and_logs_when_preferred_is_unavailable(caplog):
    selector = BackendSelector(make_gpu(GpuVendor.AMD))

    with caplog.at_level(logging.WARNING, logger="app.core.backend_selector"):
        result = selector.select(Backend.CUDA)

    assert result == Backend.DIRECTML
    assert "cuda" in caplog.text
    assert "directml" in caplog.text


def test_load_onnxruntime_directml_returns_module_with_dml_provider():
    selector = BackendSelector(make_gpu(GpuVendor.AMD))
    onnxruntime = selector.load_onnxruntime(Backend.DIRECTML)

    assert onnxruntime is not None
    assert "DmlExecutionProvider" in onnxruntime.get_available_providers()


def test_load_onnxruntime_cpu_returns_module():
    selector = BackendSelector(make_gpu(GpuVendor.UNKNOWN))
    onnxruntime = selector.load_onnxruntime(Backend.CPU)

    assert "CPUExecutionProvider" in onnxruntime.get_available_providers()


def test_load_onnxruntime_rejects_vulkan_backend():
    selector = BackendSelector(make_gpu(GpuVendor.AMD))

    with pytest.raises(ValueError, match="不使用 onnxruntime"):
        selector.load_onnxruntime(Backend.VULKAN)
