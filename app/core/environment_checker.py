"""环境检测模块（W1）。

职责：启动时检测独立显卡、厂商、显存、后端可用性，并检测视频位深/HDR。
"""
from __future__ import annotations

import ctypes
import json
import logging
import re
import subprocess
import sys
from ctypes import (
    POINTER,
    WINFUNCTYPE,
    Structure,
    byref,
    c_long,
    c_size_t,
    c_ubyte,
    c_uint,
    c_uint16,
    c_uint32,
    c_ulong,
    c_void_p,
    cast,
)
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from .types import Backend, GpuInfo, GpuType, GpuVendor

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EnvironmentCheckError(Exception):
    """环境不满足运行条件时抛出（如无独立显卡）。"""

    def __init__(self, message: str, code: str = "ENV_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


def _vendor_from_id(vendor_id: int) -> GpuVendor:
    """将 DXGI VendorId 映射为 GpuVendor。"""
    return {
        0x10DE: GpuVendor.NVIDIA,
        0x1002: GpuVendor.AMD,
        0x8086: GpuVendor.INTEL,
    }.get(int(vendor_id), GpuVendor.UNKNOWN)


def _bit_depth_from_pix_fmt(pix_fmt: str) -> int:
    """从 ffprobe pix_fmt 解析位深，无法识别时按 8bit 处理。"""
    if not pix_fmt:
        return 8

    fmt = pix_fmt.lower().strip()
    known_8bit = {
        "nv12",
        "nv21",
        "yuv420p",
        "yuvj420p",
        "yuva420p",
        "yuv422p",
        "yuv444p",
        "rgb24",
        "bgr24",
        "rgba",
        "bgra",
    }
    if fmt in known_8bit:
        return 8

    # p010/p012/p016 等格式的数字在 p0 后，而不是末尾。
    p_match = re.fullmatch(r"p0(\d{2})(?:le|be)?", fmt)
    if p_match:
        return int(p_match.group(1))

    base = re.sub(r"(?:le|be)$", "", fmt)
    match = re.search(r"(\d{1,2})$", base)
    if match and base.startswith(("yuv", "yuva", "gray", "gbr", "gbrap")):
        return int(match.group(1))
    return 8


def _is_hdr_from_stream(stream: dict[str, Any]) -> bool:
    """根据 ffprobe stream 字段判断 HDR。"""
    color_transfer = str(stream.get("color_transfer") or "").lower()
    color_primaries = str(stream.get("color_primaries") or "").lower()
    return color_transfer in {"smpte2084", "arib-std-b67"} or color_primaries == "bt2020"


def _load_config() -> dict[str, Any]:
    """读取项目根目录 config.yaml；配置缺失或损坏时返回空字典。"""
    config_path = _PROJECT_ROOT / "config.yaml"
    if not config_path.is_file():
        return {}

    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    try:
        import yaml
    except ImportError:
        yaml = None

    if yaml is not None:
        try:
            data = yaml.safe_load(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.warning("配置文件解析失败，将使用默认路径", exc_info=True)

    # 无 PyYAML 时仅解析 paths 平铺项，足够满足外部工具路径需求。
    result: dict[str, Any] = {}
    current_section: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            current_section = line.strip().rstrip(":").strip() or None
            result.setdefault(current_section or "", {})
            continue
        if ":" not in line or current_section is None:
            continue
        key, value = line.split(":", 1)
        result.setdefault(current_section, {})[key.strip()] = value.strip().strip("'\"")
    return result


def _config_path(section: str, key: str) -> Optional[str]:
    data = _load_config()
    value = data.get(section, {}).get(key) if isinstance(data.get(section), dict) else None
    return str(value) if value not in (None, "") else None


def _resolve_vendor_path(key: str, default_name: str) -> Path:
    """解析外部工具路径，优先 config，其次项目根目录 vendor。"""
    configured = _config_path("paths", key)
    raw = configured or str(_PROJECT_ROOT / "vendor" / default_name)
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _ffprobe_path() -> Path:
    return _resolve_vendor_path("ffprobe", "ffprobe.exe")


def _ncnn_path() -> Path:
    return _resolve_vendor_path("ncnn", "realesrgan-ncnn-vulkan.exe")


class _GUID(Structure):
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", c_uint16),
        ("Data3", c_uint16),
        ("Data4", c_ubyte * 8),
    ]


class _LUID(Structure):
    _fields_ = [
        ("LowPart", c_uint32),
        ("HighPart", c_long),
    ]


class _DXGIAdapterDesc(Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", c_uint),
        ("DeviceId", c_uint),
        ("SubSysId", c_uint),
        ("Revision", c_uint),
        ("DedicatedVideoMemory", c_size_t),
        ("DedicatedSystemMemory", c_size_t),
        ("SharedSystemMemory", c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", c_uint),
    ]


_IID_IDXGIFACTORY = _GUID(
    0x7B7166EC,
    0x21C7,
    0x44AE,
    (c_ubyte * 8)(0xB2, 0x1A, 0xC9, 0xAE, 0x32, 0x1A, 0xE3, 0x69),
)


def _com_method(interface: c_void_p, index: int, prototype):
    """读取 COM 接口 vtable 中的方法并绑定 this 指针。"""
    vtable = cast(interface, POINTER(POINTER(c_void_p))).contents
    return prototype(vtable[index])


def _release_com(interface: c_void_p) -> None:
    try:
        release = _com_method(interface, 2, WINFUNCTYPE(c_ulong, c_void_p))
        release(interface)
    except Exception:
        logger.debug("释放 COM 接口失败", exc_info=True)


def _detect_gpus_via_dxgi() -> list[GpuInfo]:
    """用 DXGI 枚举所有适配器，返回精确显存和厂商信息。"""
    if sys.platform != "win32":
        return []

    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
    except OSError:
        logger.warning("无法加载 dxgi.dll")
        return []

    create_factory = dxgi.CreateDXGIFactory
    create_factory.argtypes = [POINTER(_GUID), POINTER(c_void_p)]
    create_factory.restype = c_long

    factory = c_void_p()
    hr = create_factory(byref(_IID_IDXGIFACTORY), byref(factory))
    if hr < 0 or not factory:
        logger.warning("CreateDXGIFactory 失败: 0x%08X", hr & 0xFFFFFFFF)
        return []

    enum_adapters = _com_method(
        factory,
        7,
        WINFUNCTYPE(c_long, c_void_p, c_uint, POINTER(c_void_p)),
    )

    gpus: list[GpuInfo] = []
    try:
        for index in range(64):
            adapter = c_void_p()
            hr = enum_adapters(factory, c_uint(index), byref(adapter))
            if hr < 0 or not adapter:
                break

            try:
                get_desc = _com_method(
                    adapter,
                    8,
                    WINFUNCTYPE(c_long, c_void_p, POINTER(_DXGIAdapterDesc)),
                )
                desc = _DXGIAdapterDesc()
                hr = get_desc(adapter, byref(desc))
                if hr < 0:
                    continue

                name = str(desc.Description).rstrip("\x00").strip()
                vendor_id = int(desc.VendorId)
                device_id = int(desc.DeviceId)
                dedicated_bytes = int(desc.DedicatedVideoMemory)
                vendor = _vendor_from_id(vendor_id)

                if vendor == GpuVendor.INTEL and dedicated_bytes == 0 and _is_intel_discrete_name(name):
                    gpu_type = GpuType.DISCRETE
                elif dedicated_bytes > 0:
                    gpu_type = GpuType.DISCRETE
                else:
                    gpu_type = GpuType.INTEGRATED

                gpus.append(
                    GpuInfo(
                        vendor=vendor,
                        name=name or "Unknown GPU",
                        type=gpu_type,
                        dedicated_vram_bytes=dedicated_bytes,
                        vendor_id=vendor_id,
                        device_id=device_id,
                    )
                )
            finally:
                _release_com(adapter)
    finally:
        _release_com(factory)

    return gpus


def _is_intel_discrete_name(name: str) -> bool:
    lowered = name.lower()
    markers = ("arc", "iris xe max", "a770", "a750", "a580", "a380", "b580", "b570")
    return any(marker in lowered for marker in markers)


def _parse_pnp_vendor(pnp_device_id: str) -> Optional[int]:
    match = re.search(r"VEN_([0-9A-Fa-f]{4})", pnp_device_id or "")
    return int(match.group(1), 16) if match else None


def _detect_gpus_via_wmi() -> list[GpuInfo]:
    """WMI 兜底路径。AdapterRAM 为 DWORD，仅在不适用 DXGI 时使用。"""
    if sys.platform != "win32":
        return []

    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,AdapterCompatibility,DriverVersion,PNPDeviceID | "
        "ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("WMI 显卡枚举失败", exc_info=True)
        return []

    if completed.returncode != 0 or not completed.stdout.strip():
        return []

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        logger.warning("WMI 返回结果不是有效 JSON", exc_info=True)
        return []

    controllers = payload if isinstance(payload, list) else [payload]
    gpus: list[GpuInfo] = []
    for item in controllers:
        if not isinstance(item, dict):
            continue

        name = str(item.get("Name") or "Unknown GPU").strip()
        vendor_id = _parse_pnp_vendor(str(item.get("PNPDeviceID") or ""))
        if vendor_id is None:
            compatibility = str(item.get("AdapterCompatibility") or "").lower()
            if "nvidia" in compatibility:
                vendor_id = 0x10DE
            elif "amd" in compatibility or "advanced micro devices" in compatibility:
                vendor_id = 0x1002
            elif "intel" in compatibility:
                vendor_id = 0x8086
            else:
                vendor_id = 0

        adapter_ram = int(item.get("AdapterRAM") or 0)
        vendor = _vendor_from_id(vendor_id)
        dedicated_bytes = adapter_ram
        if vendor == GpuVendor.INTEL and dedicated_bytes == 0 and _is_intel_discrete_name(name):
            gpu_type = GpuType.DISCRETE
        elif dedicated_bytes > 0:
            gpu_type = GpuType.DISCRETE
        else:
            gpu_type = GpuType.INTEGRATED

        gpus.append(
            GpuInfo(
                vendor=vendor,
                name=name,
                type=gpu_type,
                dedicated_vram_bytes=dedicated_bytes,
                driver_version=str(item.get("DriverVersion") or ""),
                vendor_id=vendor_id,
            )
        )
    return gpus


def _coerce_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _enrich_nvidia_gpus(gpus: list[GpuInfo]) -> list[GpuInfo]:
    """NVIDIA 显卡可选地通过 pynvml 复核名称、显存与驱动版本。"""
    nvidia_indexes = [i for i, gpu in enumerate(gpus) if gpu.vendor == GpuVendor.NVIDIA]
    if not nvidia_indexes:
        return gpus

    try:
        import pynvml
    except Exception:
        logger.debug("未安装 pynvml，跳过 NVIDIA 信息复核", exc_info=True)
        return gpus

    try:
        pynvml.nvmlInit()
    except Exception:
        logger.debug("nvmlInit 失败，跳过 NVIDIA 信息复核", exc_info=True)
        return gpus

    updated = list(gpus)
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        driver_version = ""
        try:
            driver_version = _coerce_text(pynvml.nvmlSystemGetDriverVersion())
        except Exception:
            logger.debug("读取 NVIDIA 驱动版本失败", exc_info=True)

        for nvml_index, gpu_index in enumerate(nvidia_indexes):
            if nvml_index >= device_count:
                break
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
                name = _coerce_text(pynvml.nvmlDeviceGetName(handle))
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                updated[gpu_index] = replace(
                    updated[gpu_index],
                    name=name,
                    dedicated_vram_bytes=int(memory.total),
                    driver_version=driver_version,
                )
            except Exception:
                logger.debug("读取第 %s 张 NVIDIA 显卡信息失败", nvml_index, exc_info=True)
    except Exception:
        logger.debug("pynvml 显卡枚举失败", exc_info=True)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            logger.debug("nvmlShutdown 失败", exc_info=True)

    return updated


def detect_gpus() -> list[GpuInfo]:
    """枚举所有显卡，返回 GpuInfo 列表。"""
    try:
        gpus = _detect_gpus_via_dxgi()
    except Exception:
        logger.warning("DXGI 显卡枚举失败，尝试 WMI 兜底", exc_info=True)
        gpus = []

    if not gpus:
        try:
            gpus = _detect_gpus_via_wmi()
        except Exception:
            logger.warning("WMI 显卡枚举失败", exc_info=True)
            gpus = []

    try:
        return _enrich_nvidia_gpus(gpus)
    except Exception:
        logger.warning("NVIDIA 显卡信息复核失败", exc_info=True)
        return gpus


def find_discrete_gpu(gpus: list[GpuInfo]) -> Optional[GpuInfo]:
    """从列表中选出首选独立显卡（多卡时选显存最大者）。"""
    discrete = [g for g in gpus if g.type == GpuType.DISCRETE]
    if not discrete:
        return None
    return max(discrete, key=lambda g: g.dedicated_vram_bytes)


def check_environment() -> GpuInfo:
    """启动自检入口。

    返回首选独立显卡 GpuInfo；无独立显卡时抛 EnvironmentCheckError
    （code=NO_DISCRETE_GPU）。
    """
    gpu = find_discrete_gpu(detect_gpus())
    if gpu is None:
        raise EnvironmentCheckError(
            "未检测到满足运行条件的独立显卡",
            code="NO_DISCRETE_GPU",
        )
    return gpu


def available_backends(gpu: GpuInfo) -> list[Backend]:
    """根据显卡厂商与驱动返回该机器可用的后端列表。"""
    if gpu.vendor == GpuVendor.NVIDIA:
        backends = [Backend.CUDA, Backend.DIRECTML, Backend.CPU]
    elif gpu.vendor == GpuVendor.AMD:
        backends = [Backend.DIRECTML, Backend.VULKAN, Backend.CPU]
    elif gpu.vendor == GpuVendor.INTEL:
        backends = [Backend.DIRECTML, Backend.CPU]
    else:
        backends = [Backend.CPU]

    if Backend.VULKAN in backends and not _ncnn_path().is_file():
        backends.remove(Backend.VULKAN)

    return backends


def detect_video_bit_depth(video_path: str) -> tuple[int, bool]:
    """用 ffprobe 检测视频位深与 HDR。

    返回 (bit_depth, is_hdr)。找不到 ffprobe、视频文件或解析失败时返回
    (8, False)，不抛出异常。
    """
    ffprobe = _ffprobe_path()
    if not ffprobe.is_file():
        logger.warning("ffprobe 不存在: %s，按 8bit SDR 处理", ffprobe)
        return 8, False

    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=pix_fmt,color_primaries,color_transfer",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("ffprobe 执行失败，按 8bit SDR 处理", exc_info=True)
        return 8, False

    if completed.returncode != 0:
        logger.warning("ffprobe 返回码异常: %s", completed.returncode)
        return 8, False

    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        if isinstance(streams, list):
            stream = streams[0] if streams else {}
        elif isinstance(streams, dict):
            stream = streams
        else:
            stream = {}
    except (json.JSONDecodeError, AttributeError):
        logger.warning("ffprobe 返回结果无法解析，按 8bit SDR 处理", exc_info=True)
        return 8, False

    bit_depth = _bit_depth_from_pix_fmt(str(stream.get("pix_fmt") or ""))
    is_hdr = _is_hdr_from_stream(stream)
    return bit_depth, is_hdr
