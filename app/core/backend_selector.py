"""后端选择器（BackendSelector）。

职责：依据显卡厂商 + 用户选择，返回当前生效的后端；管理双 onnxruntime 的动态加载。
"""
from __future__ import annotations

import logging
from typing import Optional

from .environment_checker import available_backends
from .types import Backend, GpuInfo, GpuVendor

logger = logging.getLogger(__name__)


class BackendSelector:
    """根据厂商与用户偏好选择推理后端。"""

    def __init__(self, gpu: GpuInfo):
        self.gpu = gpu

    def default_backend(self) -> Backend:
        """厂商默认后端：NVIDIA→CUDA、AMD→DirectML、Intel→DirectML。"""
        if self.gpu.vendor == GpuVendor.NVIDIA:
            return Backend.CUDA
        if self.gpu.vendor in {GpuVendor.AMD, GpuVendor.INTEL}:
            return Backend.DIRECTML
        return Backend.CPU

    def select(self, preferred: Optional[Backend]) -> Backend:
        """返回生效后端；preferred 为空时用默认，不可用时回退默认并记录原因。"""
        default = self.default_backend()
        if preferred is None:
            return default

        available = available_backends(self.gpu)
        if preferred in available:
            return preferred

        logger.warning(
            "后端 %s 在厂商 %s 上不可用，回退到默认后端 %s",
            preferred.value,
            self.gpu.vendor.value,
            default.value,
        )
        return default

    def load_onnxruntime(self, backend: Backend):
        """按后端动态加载对应 onnxruntime 包，返回 onnxruntime 模块。

        双运行时共存实现要点：
        - ``runtime/cuda/`` 放置 onnxruntime-gpu 解包产物（CUDA/TensorRT EP）
        - ``runtime/dml/``  放置 onnxruntime-directml 解包产物（DirectML EP）
        - 两个包模块名均为 ``onnxruntime``，不能同 venv 共存，故用 importlib + sys.path
          动态加载；注意 onnxruntime 的 ``onnxruntime.capi`` 子包需一并处理。
        """
        if backend == Backend.DIRECTML:
            target_provider = "DmlExecutionProvider"
        elif backend in {Backend.CUDA, Backend.TENSORRT}:
            target_provider = "CUDAExecutionProvider"
        elif backend == Backend.OPENVINO:
            target_provider = "OpenVINOExecutionProvider"
        elif backend == Backend.CPU:
            target_provider = "CPUExecutionProvider"
        elif backend == Backend.VULKAN:
            raise ValueError("Vulkan 后端不使用 onnxruntime")
        else:
            raise ValueError(f"不支持的 ONNX Runtime 后端: {backend}")

        # W6 打包阶段实现双运行时目录加载：
        #   - runtime/cuda/ 中的 onnxruntime-gpu 产物
        #   - runtime/dml/ 中的 onnxruntime-directml 产物
        # 当前开发阶段二者只安装其一，因此直接导入当前环境中的 onnxruntime。
        # TODO(W6): 根据 backend 将对应 runtime 目录临时加入 sys.path，并通过
        # importlib 动态加载/隔离 ``onnxruntime`` 与 ``onnxruntime.capi``。
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError("onnxruntime 未安装，无法加载推理后端") from exc

        providers = onnxruntime.get_available_providers()
        if target_provider not in providers:
            raise RuntimeError(
                f"当前 onnxruntime 未提供 {target_provider}；可用 providers={providers}"
            )
        return onnxruntime
