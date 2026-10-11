"""核心共享类型定义（契约层）。

本模块定义所有模块间传递的数据结构与枚举，是各 Wave 实现的公共契约。
字段一经确定不得随意更改；如需变更，必须同步更新所有依赖方。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class GpuVendor(str, Enum):
    """显卡厂商。"""

    NVIDIA = "nvidia"   # VendorId 0x10DE
    AMD = "amd"         # VendorId 0x1002
    INTEL = "intel"     # VendorId 0x8086
    UNKNOWN = "unknown"


class GpuType(str, Enum):
    """显卡类型。"""

    DISCRETE = "discrete"       # 独立显卡（满足运行条件）
    INTEGRATED = "integrated"   # 核显（不满足运行条件）
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GpuInfo:
    """单张显卡信息（由环境检测模块产出）。"""

    vendor: GpuVendor
    name: str
    type: GpuType
    dedicated_vram_bytes: int          # 独显显存精确值（DXGI DedicatedVideoMemory）
    driver_version: str = ""
    vendor_id: int = 0
    device_id: int = 0

    @property
    def dedicated_vram_gb(self) -> float:
        return round(self.dedicated_vram_bytes / (1024 ** 3), 2)


class Backend(str, Enum):
    """推理后端。"""

    CUDA = "cuda"           # onnxruntime-gpu，NVIDIA 主力
    TENSORRT = "tensorrt"   # onnxruntime-gpu TensorRT，NVIDIA 可选
    DIRECTML = "directml"   # onnxruntime-directml，AMD/Intel 主力，NVIDIA 备选
    OPENVINO = "openvino"   # Intel 可选
    VULKAN = "vulkan"       # ncnn / Anime4KCPP，AMD 第二后端
    CPU = "cpu"             # 兜底


class Scene(str, Enum):
    """超分场景预设。"""

    REALISTIC = "realistic"   # 真实：照片/实拍/人像/风景
    ANIME_2D = "anime_2d"     # 2D 动漫/漫画/插画
    CG_3D = "cg_3d"           # 3D/CG/游戏画面


class VideoCodec(str, Enum):
    """输出视频编码器（用户可选，默认 H.264）。"""

    H264 = "h264"
    HEVC = "hevc"


@dataclass(frozen=True)
class ModelMeta:
    """超分模型元数据（内置模型与自定义导入模型统一用此结构）。"""

    name: str
    scale: Optional[int]               # 固定倍率；scale-aware 模型（如 SRMD）填 None
    scene: Scene                       # 适用场景
    backend: Backend                   # 可用后端（Anime4K 仅 VULKAN/CPU）
    model_path: Path                   # 模型路径（.onnx 或 ncnn .param/.bin 目录）
    denoise_pair_path: Optional[Path] = None  # 降噪模型路径（ONNX 后端使用）
    # 降噪模型自身的倍率：1 表示"在输入尺度做预降噪"，None 表示与主模型同倍率
    # （同倍率时按降噪强度与主模型输出做插值，即原来的 pair 混合路径）。
    denoise_pair_scale: Optional[int] = None
    kind: str = "sr"                        # "sr" 超分模型；"denoise" 纯降噪模型
    input_name: str = "input"
    output_name: str = "output"
    scale_input: Optional[str] = None       # SRMD 的 "scale"
    denoise_input: Optional[str] = None     # general-x4v3 的 "denoise"
    channel_order: str = "RGB"
    normalize: tuple[float, float] = (0.0, 1.0)
    tile_size: int = 512
    description: str = ""


@dataclass
class SuperResResult:
    """单张图片超分结果。"""

    ok: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    input_path: Optional[Path] = None


@dataclass
class ProgressInfo:
    """进度回调载荷。"""

    current: float
    total: int
    message: str = ""
    cancelled: bool = False
    image_index: Optional[int] = None
    image_total: Optional[int] = None
    source_name: Optional[str] = None
    phase: Optional[str] = None

    @property
    def percent(self) -> float:
        return (self.current / self.total * 100) if self.total else 0.0
