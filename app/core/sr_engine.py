"""超分推理引擎（SuperResEngine）。

统一接口，屏蔽多后端差异：ONNX(CUDA/DirectML/OpenVINO) 与 Vulkan(ncnn/Anime4KCPP)。
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

# 解除 Pillow 解压炸弹限制（Vulkan 后端读写大图）
Image.MAX_IMAGE_PIXELS = None

from .paths import app_root
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
        denoise_strength: float = 0.5,
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
        self._pair_session = None
        self._model: Optional[ModelMeta] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._output_channels: Optional[int] = None
        self._pair_input_name: Optional[str] = None
        self._pair_output_name: Optional[str] = None
        self._pair_output_channels: Optional[int] = None
        self._fp16 = False
        self._precision = "fp16" if backend == Backend.CUDA else "fp32"

    def load_model(self, model: ModelMeta) -> None:
        model_path = self._resolve_model_path(model.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

        pair_path = self._resolve_pair_path(model)
        if pair_path is not None and not pair_path.is_file():
            raise FileNotFoundError(f"ONNX 降噪对模型不存在: {pair_path}")

        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError("onnxruntime 未安装，无法加载 ONNX 模型") from exc

        providers = self._providers_for_backend()
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        def create_session(path: Path, active_providers):
            return onnxruntime.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=active_providers,
            )

        active_providers = providers
        try:
            session = create_session(model_path, active_providers)
        except Exception as exc:
            # CUDA 机器上 fp16 组合可能失败（驱动/算子支持不足），先退回
            # fp32 的 CPU provider，保证至少能加载模型；DirectML 默认 fp32，
            # 不在这里做无意义的 fp16 尝试。
            if self.backend != Backend.CUDA:
                raise
            logger.warning("CUDA provider 加载失败，回退到 CPU fp32: %s", exc)
            self._precision = "fp32"
            active_providers = ["CPUExecutionProvider"]
            session = create_session(model_path, active_providers)

        pair_session = None
        if pair_path is not None:
            pair_session = create_session(pair_path, active_providers)

        self._session = session
        self._model = model
        self._fp16 = self._is_fp16_session(session)
        self._precision = "fp16" if self._fp16 else "fp32"
        self._input_name = self._resolve_io_name(model.input_name, session.get_inputs())
        self._output_name = self._resolve_io_name(model.output_name, session.get_outputs())
        self._output_channels = self._resolve_output_channels(session)
        if pair_session is not None:
            self._pair_session = pair_session
            self._pair_input_name = self._resolve_io_name(
                model.input_name, pair_session.get_inputs()
            )
            self._pair_output_name = self._resolve_io_name(
                model.output_name, pair_session.get_outputs()
            )
            self._pair_output_channels = self._resolve_output_channels(pair_session)
        else:
            self._pair_session = None
            self._pair_input_name = None
            self._pair_output_name = None
            self._pair_output_channels = None

    @staticmethod
    def _resolve_model_path(path: Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        root_candidate = app_root() / path
        if root_candidate.exists():
            return root_candidate
        return path

    def _resolve_pair_path(self, model: ModelMeta) -> Optional[Path]:
        if model.denoise_pair_path is None:
            return None

        raw = Path(model.denoise_pair_path)
        if raw.is_absolute():
            return raw

        sibling = Path(model.model_path).parent / raw
        if sibling.is_file():
            return sibling

        root_candidate = app_root() / raw
        if root_candidate.is_file():
            return root_candidate
        return raw

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

    def upscale_image(self, image, scale=4, progress_cb=None, denoise_strength: float = 0.5):
        if self._session is None or self._model is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model")
        if scale <= 0:
            raise ValueError("scale 必须为正整数")

        denoise_strength = max(0.0, min(1.0, float(denoise_strength)))
        target_scale = int(self._model.scale or scale)
        tensor = preprocess_image(
            image,
            channel_order=self._model.channel_order,
            normalize=self._model.normalize,
            dtype=self._inference_dtype(),
        )

        tile_size = int(self._model.tile_size or 512)
        try:
            return self._upscale_tiled(
                tensor,
                target_scale,
                tile_size,
                progress_cb,
                denoise_strength,
            )
        except (MemoryError, RuntimeError) as exc:
            if tile_size <= 256 or not self._looks_like_oom(exc):
                raise
            logger.warning(
                "tile_size=%s 时显存不足，自动降级到 256: %s",
                tile_size,
                exc,
            )
            return self._upscale_tiled(
                tensor,
                target_scale,
                256,
                progress_cb,
                denoise_strength,
            )

    def _upscale_tiled(
        self,
        tensor: np.ndarray,
        target_scale: int,
        tile_size: int,
        progress_cb: Optional[Callable[[ProgressInfo], None]],
        denoise_strength: float,
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
            feeds = self._feeds_for_tile(self._input_name, tile, target_scale)
            output = self._run_tile(self._session, self._output_name, feeds)
            output = self._validate_output(output, region, target_scale, out_channels)

            if self._pair_session is not None:
                pair_feeds = self._feeds_for_tile(
                    self._pair_input_name,
                    tile,
                    target_scale,
                )
                pair_output = self._run_tile(
                    self._pair_session,
                    self._pair_output_name,
                    pair_feeds,
                )
                pair_output = self._validate_output(
                    pair_output,
                    region,
                    target_scale,
                    out_channels,
                )
                output = (
                    output * (1.0 - denoise_strength)
                    + pair_output * denoise_strength
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

    def _feeds_for_tile(
        self,
        input_name: Optional[str],
        tile: np.ndarray,
        target_scale: int,
    ) -> dict:
        if input_name is None:
            raise RuntimeError("模型输入名尚未解析")
        feeds = {
            input_name: np.ascontiguousarray(
                tile,
                dtype=self._inference_dtype(),
            )
        }
        self._add_control_inputs(feeds, target_scale)
        return feeds

    def _run_tile(self, session, output_name: Optional[str], feeds: dict) -> np.ndarray:
        if output_name is None:
            raise RuntimeError("模型输出名尚未解析")
        output = session.run([output_name], feeds)[0]
        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 3:
            output = output[None, ...]
        if output.ndim != 4:
            raise ValueError(f"模型输出应为 NCHW，实际 shape={output.shape}")
        return output

    def _validate_output(
        self,
        output: np.ndarray,
        region,
        target_scale: int,
        out_channels: int,
    ) -> np.ndarray:
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
        return output

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
        self._model: Optional[ModelMeta] = None
        self._ncnn_path: Optional[Path] = None
        self._model_dir: Optional[Path] = None

    @staticmethod
    def _ncnn_model_name(model: ModelMeta, scale: int) -> str:
        """Map the UI model name to an ncnn model name accepted by ``-n``."""
        name = model.name
        lowered = name.lower()

        if "animevideov3" in lowered:
            return f"realesr-animevideov3-x{int(scale)}"
        if "anime" in lowered:
            return "realesrgan-x4plus-anime"
        if "x4plus" in lowered:
            return "realesrgan-x4plus"
        raise ValueError(
            f"Vulkan 后端不支持模型「{name}」（ncnn 无对应模型），请切换 DirectML 后端"
        )

    def load_model(self, model: ModelMeta) -> None:
        ncnn_path = app_root() / "vendor" / "realesrgan-ncnn-vulkan.exe"
        model_dir = app_root() / "vendor" / "ncnn-models"
        if not ncnn_path.is_file():
            raise FileNotFoundError(f"ncnn 可执行文件不存在: {ncnn_path}")
        if not model_dir.is_dir():
            raise FileNotFoundError(f"ncnn 模型目录不存在: {model_dir}")

        self._model = model
        self._ncnn_path = ncnn_path
        self._model_dir = model_dir

    def upscale_image(self, image, scale=4, progress_cb=None, denoise_strength: float = 0.5):
        if self._model is None or self._ncnn_path is None or self._model_dir is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model")
        if scale not in (2, 3, 4):
            raise ValueError("Vulkan ncnn 后端仅支持 2x/3x/4x 倍率")
        if self.cancelled:
            raise SuperResCancelled()

        ncnn_model = self._ncnn_model_name(self._model, int(scale))
        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise TypeError("Vulkan 后端要求 uint8 RGB 输入")

        with tempfile.TemporaryDirectory(prefix="srt-vulkan-") as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "input.png"
            output_path = temp_root / "output.png"
            Image.fromarray(image, "RGB").save(input_path)

            if progress_cb is not None:
                progress_cb(
                    ProgressInfo(
                        current=0,
                        total=1,
                        message="Vulkan 处理中 0/1",
                    )
                )

            command = [
                str(self._ncnn_path),
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "-m",
                self._model_dir.name,
                "-n",
                ncnn_model,
                "-s",
                str(int(scale)),
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    creationflags=creationflags,
                    cwd=str(self._ncnn_path.parent),
                    check=False,
                )
            except OSError as exc:
                raise RuntimeError(f"无法启动 Vulkan ncnn 进程: {exc}") from exc

            if self.cancelled:
                raise SuperResCancelled()

            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(
                    f"Vulkan ncnn 执行失败（exit={completed.returncode}）: {detail}"
                )
            if not output_path.is_file():
                raise RuntimeError("Vulkan ncnn 未生成输出文件")

            with Image.open(output_path) as output_image:
                result = np.asarray(output_image.convert("RGB"))

            if self.cancelled:
                raise SuperResCancelled()

            if progress_cb is not None:
                progress_cb(
                    ProgressInfo(
                        current=1,
                        total=1,
                        message="Vulkan 处理完成 1/1",
                    )
                )
            return result
