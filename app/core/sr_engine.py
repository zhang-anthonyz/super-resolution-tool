"""超分推理引擎（SuperResEngine）。

统一接口，屏蔽多后端差异：ONNX(CUDA/DirectML/OpenVINO) 与 Vulkan(ncnn/Anime4KCPP)。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from .types import Backend, ModelMeta, ProgressInfo


class SuperResEngine(ABC):
    """超分引擎抽象基类。所有后端实现此接口。"""

    def __init__(self, backend: Backend):
        self.backend = backend
        self._cancelled = False

    def cancel(self) -> None:
        """请求取消当前任务（线程安全）。"""
        self._cancelled = True

    def reset(self) -> None:
        """清除取消标志，供下一次任务复用。"""
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @abstractmethod
    def load_model(self, model: ModelMeta) -> None:
        """加载模型到显存/内存。"""
        ...

    @abstractmethod
    def upscale_image(
        self,
        image: np.ndarray,          # RGB、HWC、uint8
        scale: int = 4,
        progress_cb: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> np.ndarray:                # 返回 RGB、HWC、uint8
        """超分单张图片。

        实现要求：tile 分块（默认 512，按显存余量自适应下调）、fp16/fp32 策略、
        循环内检查 ``cancelled`` 并抛 ``SuperResCancelled``。
        """
        ...


class SuperResCancelled(Exception):
    """任务被用户取消。"""


class OnnxEngine(SuperResEngine):
    """ONNX Runtime 后端（CUDA / DirectML / OpenVINO / CPU 共用）。"""

    def __init__(self, backend: Backend):
        super().__init__(backend)

    def load_model(self, model: ModelMeta) -> None:
        raise NotImplementedError  # TODO: Codex 实现

    def upscale_image(self, image, scale=4, progress_cb=None):
        raise NotImplementedError  # TODO: Codex 实现


class VulkanEngine(SuperResEngine):
    """Vulkan 后端（ncnn Real-ESRGAN + Anime4KCPP）。

    V1 采用 subprocess 调 CLI（接受文件级/帧级粗粒度进度与取消控制）。
    """

    def __init__(self, backend: Backend):
        super().__init__(backend)

    def load_model(self, model: ModelMeta) -> None:
        raise NotImplementedError  # TODO: Codex 实现

    def upscale_image(self, image, scale=4, progress_cb=None):
        raise NotImplementedError  # TODO: Codex 实现
