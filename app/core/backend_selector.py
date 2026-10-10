"""后端选择器（BackendSelector）。

职责：依据显卡厂商 + 用户选择，返回当前生效的后端；管理双 onnxruntime 的动态加载。
"""
from __future__ import annotations

from typing import Optional

from .types import Backend, GpuInfo


class BackendSelector:
    """根据厂商与用户偏好选择推理后端。"""

    def __init__(self, gpu: GpuInfo):
        self.gpu = gpu

    def default_backend(self) -> Backend:
        """厂商默认后端：NVIDIA→CUDA、AMD→DirectML、Intel→DirectML。"""
        raise NotImplementedError  # TODO: Codex 实现

    def select(self, preferred: Optional[Backend]) -> Backend:
        """返回生效后端；preferred 为空时用默认，不可用时回退默认并记录原因。"""
        raise NotImplementedError  # TODO: Codex 实现

    def load_onnxruntime(self, backend: Backend):
        """按后端动态加载对应 onnxruntime 包，返回 onnxruntime 模块。

        双运行时共存实现要点：
        - ``runtime/cuda/`` 放置 onnxruntime-gpu 解包产物（CUDA/TensorRT EP）
        - ``runtime/dml/``  放置 onnxruntime-directml 解包产物（DirectML EP）
        - 两个包模块名均为 ``onnxruntime``，不能同 venv 共存，故用 importlib + sys.path
          动态加载；注意 onnxruntime 的 ``onnxruntime.capi`` 子包需一并处理。
        """
        raise NotImplementedError  # TODO: Codex 实现
