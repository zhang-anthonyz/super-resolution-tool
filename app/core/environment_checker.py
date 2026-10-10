"""环境检测模块（W1）。

职责：启动时检测独立显卡、厂商、显存、后端可用性。
本文件定义接口契约，具体实现由 Codex 按契约填充。
"""
from __future__ import annotations

from typing import Optional

from .types import Backend, GpuInfo, GpuType, GpuVendor


class EnvironmentCheckError(Exception):
    """环境不满足运行条件时抛出（如无独立显卡）。"""

    def __init__(self, message: str, code: str = "ENV_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


def detect_gpus() -> list[GpuInfo]:
    """枚举所有显卡，返回 GpuInfo 列表。

    实现要点：
    - DXGI 枚举适配器读 ``DedicatedVideoMemory``（精确；WMI 的 AdapterRAM 有 4GB 截断缺陷）
    - VendorId 映射：0x10DE=NVIDIA、0x1002=AMD、0x8086=Intel
    - 独显判定：``DedicatedVideoMemory > 0``；Intel 独显（Arc/Iris Xe MAX）不能误判为核显，
      需结合型号名与显存大小（UHD/Iris 核显的 DedicatedVideoMemory 为 0）
    - NVIDIA 额外用 pynvml 复核显存与驱动版本
    """
    raise NotImplementedError  # TODO: Codex 实现


def find_discrete_gpu(gpus: list[GpuInfo]) -> Optional[GpuInfo]:
    """从列表中选出首选独立显卡（多卡时选显存最大者）。"""
    discrete = [g for g in gpus if g.type == GpuType.DISCRETE]
    if not discrete:
        return None
    return max(discrete, key=lambda g: g.dedicated_vram_bytes)


def check_environment() -> GpuInfo:
    """启动自检入口。

    返回首选独立显卡 GpuInfo；无独立显卡时抛 EnvironmentCheckError（code=NO_DISCRETE_GPU）。
    """
    raise NotImplementedError  # TODO: Codex 实现


def available_backends(gpu: GpuInfo) -> list[Backend]:
    """根据显卡厂商与驱动返回该机器可用的后端列表（供 BackendSelector/UI 使用）。

    期望输出：
    - NVIDIA: [CUDA, DIRECTML, CPU]（可选 TENSORRT）
    - AMD:    [DIRECTML, VULKAN, CPU]
    - Intel:  [DIRECTML, CPU]（可选 OPENVINO）
    检测方式：NVIDIA 用 pynvml；AMD/Intel 用 onnxruntime 的 get_available_providers()
    判断是否含 DmlExecutionProvider；Vulkan 检测 ncnn 可执行文件与 Vulkan 驱动。
    """
    raise NotImplementedError  # TODO: Codex 实现


def detect_video_bit_depth(video_path: str) -> tuple[int, bool]:
    """用 ffprobe 检测视频位深与 HDR。

    返回 (bit_depth, is_hdr)。is_hdr 依据 color_transfer（smpte2084/arib-std-b67）或
    color_primaries（bt2020）判定。bit_depth > 8 或 is_hdr 均视为「V1 不支持」。
    """
    raise NotImplementedError  # TODO: Codex 实现
