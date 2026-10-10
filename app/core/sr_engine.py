"""超分推理引擎（SuperResEngine）。

统一接口，屏蔽多后端差异：ONNX(CUDA/DirectML/OpenVINO) 与 Vulkan(ncnn/Anime4KCPP)。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from .tiling import postprocess_tensor, preprocess_image, tile_regions
from .types import Backend, ModelMeta, ProgressInfo

logger = logging.getLogger(__name__)


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
        self._session = None
        self._model: Optional[ModelMeta] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._output_channels: Optional[int] = None
        self._fp16 = False
        self._precision = "fp16" if backend == Backend.CUDA else "fp32"

    def load_model(self, model: ModelMeta) -> None:
        if not model.model_path.is_file():
            raise FileNotFoundError(f"ONNX 模型不存在: {model.model_path}")

        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError("onnxruntime 未安装，无法加载 ONNX 模型") from exc

        providers = self._providers_for_backend()
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        try:
            session = onnxruntime.InferenceSession(
                str(model.model_path),
                sess_options=session_options,
                providers=providers,
            )
        except Exception as exc:
            # CUDA 机器上 fp16 组合可能失败（驱动/算子支持不足），先退回
            # fp32 的 CPU provider，保证至少能加载模型；DirectML 默认 fp32，
            # 不在这里做无意义的 fp16 尝试。
            if self.backend != Backend.CUDA:
                raise
            logger.warning("CUDA provider 加载失败，回退到 CPU fp32: %s", exc)
            self._precision = "fp32"
            session = onnxruntime.InferenceSession(
                str(model.model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )

        self._session = session
        self._model = model
        self._fp16 = self._is_fp16_session(session)
        self._precision = "fp16" if self._fp16 else "fp32"
        self._input_name = self._resolve_io_name(model.input_name, session.get_inputs())
        self._output_name = self._resolve_io_name(model.output_name, session.get_outputs())
        self._output_channels = self._resolve_output_channels(session)

    @staticmethod
    def _is_fp16_session(session) -> bool:
        inputs = session.get_inputs()
        if not inputs:
            return False
        input_type = inputs[0].type.lower()
        return "float16" in input_type

    def _inference_dtype(self):
        return np.float16 if self._fp16 else np.float32

    @staticmethod
    def _resolve_io_name(declared: str, metas) -> str:
        names = {meta.name for meta in metas}
        if declared and declared in names:
            return declared
        if metas:
            return metas[0].name
        return declared

    @staticmethod
    def _resolve_output_channels(session) -> Optional[int]:
        outputs = session.get_outputs()
        if not outputs:
            return None
        shape = outputs[0].shape
        if len(shape) >= 2 and isinstance(shape[1], int) and shape[1] > 0:
            return int(shape[1])
        return None

    def _providers_for_backend(self):
        if self.backend == Backend.DIRECTML:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if self.backend == Backend.CUDA:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.backend == Backend.OPENVINO:
            return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def upscale_image(self, image, scale=4, progress_cb=None):
        if self._session is None or self._model is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model")
        if scale <= 0:
            raise ValueError("scale 必须为正整数")

        target_scale = int(self._model.scale or scale)
        tensor = preprocess_image(
            image,
            channel_order=self._model.channel_order,
            normalize=self._model.normalize,
            dtype=self._inference_dtype(),
        )

        tile_size = int(self._model.tile_size or 512)
        try:
            return self._upscale_tiled(tensor, target_scale, tile_size, progress_cb)
        except (MemoryError, RuntimeError) as exc:
            if tile_size <= 256 or not self._looks_like_oom(exc):
                raise
            logger.warning(
                "tile_size=%s 时显存不足，自动降级到 256: %s",
                tile_size,
                exc,
            )
            return self._upscale_tiled(tensor, target_scale, 256, progress_cb)

    def _upscale_tiled(
        self,
        tensor: np.ndarray,
        target_scale: int,
        tile_size: int,
        progress_cb: Optional[Callable[[ProgressInfo], None]],
    ) -> np.ndarray:
        _, channels, height, width = tensor.shape
        regions = tile_regions(height, width, tile_size=tile_size, overlap=10)
        total = len(regions)

        out_channels = self._output_channels or channels
        output_height = height * target_scale
        output_width = width * target_scale
        accumulator = np.zeros(
            (1, out_channels, output_height, output_width),
            dtype=np.float32,
        )
        weights = np.zeros((1, 1, output_height, output_width), dtype=np.float32)

        for index, region in enumerate(regions, start=1):
            if self.cancelled:
                raise SuperResCancelled()

            tile = tensor[
                :,
                :,
                region.top : region.bottom,
                region.left : region.right,
            ]
            feeds = {
                self._input_name: np.ascontiguousarray(
                    tile,
                    dtype=self._inference_dtype(),
                )
            }
            self._add_control_inputs(feeds, target_scale)

            output = self._session.run([self._output_name], feeds)[0]
            output = np.asarray(output, dtype=np.float32)
            if output.ndim == 3:
                output = output[None, ...]
            if output.ndim != 4:
                raise ValueError(f"模型输出应为 NCHW，实际 shape={output.shape}")

            expected_h = region.height * target_scale
            expected_w = region.width * target_scale
            if output.shape[2:] != (expected_h, expected_w):
                raise ValueError(
                    "模型输出尺寸与 tile 倍率不一致: "
                    f"{output.shape[2:]} != {(expected_h, expected_w)}"
                )
            if output.shape[1] != out_channels:
                raise ValueError(
                    f"模型输出通道数不一致: {output.shape[1]} != {out_channels}"
                )

            out_top = region.top * target_scale
            out_left = region.left * target_scale
            out_bottom = region.bottom * target_scale
            out_right = region.right * target_scale
            accumulator[:, :, out_top:out_bottom, out_left:out_right] += output
            weights[:, :, out_top:out_bottom, out_left:out_right] += 1.0

            if progress_cb is not None:
                progress_cb(
                    ProgressInfo(
                        current=index,
                        total=total,
                        message=f"tile {index}/{total}",
                    )
                )

            if self.cancelled:
                raise SuperResCancelled()

        averaged = accumulator / weights
        return postprocess_tensor(averaged)

    def _add_control_inputs(self, feeds: dict, target_scale: int) -> None:
        model = self._model
        if model.scale_input:
            feeds[model.scale_input] = np.asarray([target_scale], dtype=np.float32)
        if model.denoise_input:
            feeds[model.denoise_input] = np.asarray([0.0], dtype=np.float32)

    @staticmethod
    def _looks_like_oom(exc: BaseException) -> bool:
        text = str(exc).lower()
        markers = (
            "out of memory",
            "oom",
            "insufficient memory",
            "memory allocation",
            "allocate",
        )
        return any(marker in text for marker in markers)


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
