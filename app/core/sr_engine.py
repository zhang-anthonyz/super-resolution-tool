"""超分推理引擎（SuperResEngine）。

统一接口，屏蔽多后端差异：ONNX(CUDA/DirectML/OpenVINO/CPU)。
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

# 解除 Pillow 解压炸弹限制（超分需要读写大图）
Image.MAX_IMAGE_PIXELS = None

from .paths import app_root
from .runtime_capabilities import (
    load_onnxruntime_for_backend,
    provider_for_backend,
    provider_label,
)
from .tiling import axis_starts, blend_tile, postprocess_tensor, preprocess_image
from .types import Backend, ModelMeta, ProgressInfo

logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _format_eta(seconds: float) -> str:
    """Compact remaining-time text for the progress line."""
    if seconds < 10.0:
        return f"{seconds:.1f}s"
    return f"{seconds:.0f}s"


#: Input-space pixels shared by neighbouring tiles. Bigger bands hide seams
#: better but re-infer more pixels; 10 is the Real-ESRGAN default.
_TILE_OVERLAP = 10


@dataclass(frozen=True)
class DenoisePlan:
    """How one denoise request is served by an engine.

    Exactly one of the three strengths is non-zero:

    ``pre_strength``       a scale-1 denoise model runs on the *input* image
                           (cheap: 1/scale² of the output pixels, and noise is
                           removed before the SR model can amplify it);
    ``post_strength``      a same-scale denoise model is interpolated with the
                           super-resolved output;
    ``classical_strength`` no denoise model applies, so the built-in
                           edge-preserving filter smooths the output.
    """

    pre_strength: float = 0.0
    post_strength: float = 0.0
    classical_strength: float = 0.0


def resolve_denoise_plan(engine, strength: float, target_scale: int) -> DenoisePlan:
    """Ask an engine how to serve a denoise request.

    Falls back to the built-in filter for engines (or test doubles) that do not
    implement :meth:`SuperResEngine.plan_denoise`.
    """
    planner = getattr(engine, "plan_denoise", None)
    if callable(planner):
        return planner(strength, target_scale)
    return DenoisePlan(classical_strength=_clamp01(strength))


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


    def plan_denoise(self, strength: float, target_scale: int) -> DenoisePlan:
        """Decide how a denoise request is served.

        Base implementation: no denoise model is available on this backend, so
        the built-in edge-preserving filter handles it.
        """
        return DenoisePlan(classical_strength=_clamp01(strength))


class SuperResCancelled(Exception):
    """任务被用户取消。"""


class OnnxEngine(SuperResEngine):
    """ONNX Runtime 后端（CUDA / DirectML / OpenVINO / CPU 共用）。"""

    def __init__(self, backend: Backend):
        super().__init__(backend)
        self._session = None
        self._pair_session = None
        self._pair_scale: Optional[int] = None
        self._model: Optional[ModelMeta] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._output_channels: Optional[int] = None
        self._pair_input_name: Optional[str] = None
        self._pair_output_name: Optional[str] = None
        self._pair_output_channels: Optional[int] = None
        self._fp16 = False
        self._pair_fp16 = False
        self._precision = "fp16" if backend == Backend.CUDA else "fp32"
        self._active_providers: tuple[str, ...] = ()
        self._logged_geometry: Optional[tuple] = None

    @property
    def active_providers(self) -> tuple[str, ...]:
        """Actual ONNX Runtime providers used by the loaded sessions."""
        return self._active_providers

    @property
    def runtime_label(self) -> str:
        """Short label describing the actual active runtime."""
        if not self._active_providers:
            return self.backend.value
        return ", ".join(provider_label(item) for item in self._active_providers)

    @property
    def precision_label(self) -> str:
        """Inference precision of the loaded session (``fp16`` / ``fp32``)."""
        if self._model is None:
            return ""
        return self._precision

    def load_model(self, model: ModelMeta) -> None:
        model_path = self._resolve_model_path(model.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

        pair_path = self._resolve_pair_path(model)
        if pair_path is not None and not pair_path.is_file():
            raise FileNotFoundError(f"ONNX 降噪对模型不存在: {pair_path}")

        try:
            onnxruntime = load_onnxruntime_for_backend(self.backend)
        except ImportError as exc:
            raise RuntimeError("onnxruntime 未安装，无法加载 ONNX 模型") from exc

        providers = self._providers_for_backend()
        target_provider = provider_for_backend(self.backend)
        try:
            available_providers = tuple(onnxruntime.get_available_providers())
        except Exception:
            available_providers = ()
        if target_provider is None:
            raise RuntimeError(f"不支持的 ONNX Runtime 后端: {self.backend.value}")
        if target_provider not in available_providers:
            raise RuntimeError(
                f"当前运行时未提供 {target_provider}；"
                f"可用 providers={list(available_providers)}"
            )

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
        session = create_session(model_path, active_providers)

        pair_session = None
        if pair_path is not None:
            pair_session = create_session(pair_path, active_providers)

        self._session = session
        self._model = model
        self._active_providers = tuple(session.get_providers())
        self._fp16 = self._is_fp16_session(session)
        self._precision = "fp16" if self._fp16 else "fp32"
        self._input_name = self._resolve_io_name(model.input_name, session.get_inputs())
        self._output_name = self._resolve_io_name(model.output_name, session.get_outputs())
        self._output_channels = self._resolve_output_channels(session)
        if pair_session is not None:
            self._pair_session = pair_session
            self._pair_fp16 = self._is_fp16_session(pair_session)
            if self._pair_fp16 != self._fp16:
                # 主模型与降噪模型精度可以不同（如 fp16 的 Real-ESRGAN 配
                # fp32 的 SwinIR 降噪），两侧各按自己 session 的 dtype 喂入。
                logger.info(
                    "降噪模型精度 %s 与主模型 %s 不同，按各自 session 精度喂入",
                    "fp16" if self._pair_fp16 else "fp32",
                    "fp16" if self._fp16 else "fp32",
                )
            self._pair_scale = (
                model.denoise_pair_scale
                if model.denoise_pair_scale is not None
                else model.scale
            )
            self._pair_input_name = self._resolve_io_name(
                model.input_name, pair_session.get_inputs()
            )
            self._pair_output_name = self._resolve_io_name(
                model.output_name, pair_session.get_outputs()
            )
            self._pair_output_channels = self._resolve_output_channels(pair_session)
        else:
            self._pair_session = None
            self._pair_fp16 = False
            self._pair_scale = None
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
        get_inputs = getattr(session, "get_inputs", None)
        if get_inputs is None:  # test doubles / duck-typed sessions
            return False
        inputs = get_inputs()
        if not inputs:
            return False
        input_type = inputs[0].type.lower()
        return "float16" in input_type

    def _inference_dtype(self):
        return np.float16 if self._fp16 else np.float32

    def _pair_dtype(self):
        """Input dtype of the denoise session.

        The denoise model has its own precision: the built-in Real-ESRGAN
        family is fp16 while the SwinIR denoisers are fp32, so feeding the main
        model's dtype into the pair session raises
        ``Unexpected input data type. Actual: (tensor(float16)), expected:
        (tensor(float))``.
        """
        return np.float16 if self._pair_fp16 else np.float32

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
        provider = provider_for_backend(self.backend)
        if provider is None:
            raise ValueError(f"不支持的 ONNX Runtime 后端: {self.backend}")
        return [provider]

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

    @property
    def denoiser_scale(self) -> Optional[int]:
        """Scale of the attached denoise model (None when none is loaded)."""
        return self._pair_scale

    @property
    def has_denoiser(self) -> bool:
        return self._pair_session is not None

    def pre_denoise(self, image: np.ndarray, strength: float = 0.0) -> np.ndarray:
        """Denoise at the input scale with a scale-1 denoise model.

        Returns uint8 RGB; a no-op when no scale-1 denoiser is attached, so
        callers can invoke it unconditionally before upscaling.
        """
        rgb = np.asarray(image)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        strength = float(max(0.0, min(1.0, float(strength))))
        if strength <= 0.0 or self._pair_session is None or self._pair_scale != 1:
            return rgb

        tensor = preprocess_image(
            rgb,
            channel_order=self._model.channel_order,
            normalize=self._model.normalize,
            dtype=self._pair_dtype(),
        )
        denoised = self._run_tiled(
            session=self._pair_session,
            input_name=self._pair_input_name,
            output_name=self._pair_output_name,
            out_channels=self._pair_output_channels,
            tensor=tensor,
            target_scale=1,
            tile_size=int(self._model.tile_size or 512),
            progress_cb=None,
            denoise_value=strength,
            dtype=self._pair_dtype(),
        )
        blended = (
            denoised.astype(np.float32) * strength
            + rgb.astype(np.float32) * (1.0 - strength)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def plan_denoise(self, strength: float, target_scale: int) -> DenoisePlan:
        """Route the request to the pre / post / classical denoise path."""
        strength = _clamp01(strength)
        if strength <= 0.0:
            return DenoisePlan()
        if self._pair_session is None:
            return DenoisePlan(classical_strength=strength)
        if self._pair_scale == 1:
            return DenoisePlan(pre_strength=strength)
        if self._pair_scale in (None, target_scale):
            return DenoisePlan(post_strength=strength)
        # Denoise model scale matches neither the input nor the output.
        logger.warning(
            "降噪模型倍率 %s 与任务倍率 %s 不匹配，回退到内置降噪",
            self._pair_scale,
            target_scale,
        )
        return DenoisePlan(classical_strength=strength)

    def _upscale_tiled(
        self,
        tensor: np.ndarray,
        target_scale: int,
        tile_size: int,
        progress_cb: Optional[Callable[[ProgressInfo], None]],
        denoise_strength: float,
    ) -> np.ndarray:
        """Super-resolve tile by tile, optionally blending a same-scale denoiser."""
        pair = None
        if (
            self._pair_session is not None
            and denoise_strength > 0.0
            and self._pair_scale in (None, target_scale)
        ):
            pair = (
                self._pair_session,
                self._pair_input_name,
                self._pair_output_name,
                self._pair_output_channels,
                denoise_strength,
            )
        return self._run_tiled(
            session=self._session,
            input_name=self._input_name,
            output_name=self._output_name,
            out_channels=self._output_channels,
            tensor=tensor,
            target_scale=target_scale,
            tile_size=tile_size,
            progress_cb=progress_cb,
            pair=pair,
            denoise_value=denoise_strength,
        )

    def _run_tiled(
        self,
        *,
        session,
        input_name: Optional[str],
        output_name: Optional[str],
        out_channels: Optional[int],
        tensor: np.ndarray,
        target_scale: int,
        tile_size: int,
        progress_cb: Optional[Callable[[ProgressInfo], None]] = None,
        pair: Optional[tuple] = None,
        denoise_value: float = 0.0,
        dtype=None,
    ) -> np.ndarray:
        """Tiled inference for any session, writing tiles straight into uint8.

        Tiles are written into the output image as they finish, cross-fading the
        overlap bands against their already-written left/above neighbours. That
        keeps peak memory at one output image instead of the former
        accumulator + weight + divide buffers (three full float32 images, which
        for a 4x output of a large photo meant gigabytes spilled to disk).

        ``dtype`` is the input dtype of the session being run; it defaults to
        the main model's precision and must be overridden when running the
        denoise session (which may be fp32 while the main model is fp16).
        """
        if dtype is None:
            dtype = self._inference_dtype()
        _, channels, height, width = tensor.shape
        y_starts = axis_starts(height, tile_size=tile_size, overlap=_TILE_OVERLAP)
        x_starts = axis_starts(width, tile_size=tile_size, overlap=_TILE_OVERLAP)
        total = len(y_starts) * len(x_starts)
        out_channels = out_channels or channels
        output = np.empty(
            (height * target_scale, width * target_scale, out_channels),
            dtype=np.uint8,
        )

        index = 0
        started = time.perf_counter()
        geometry = (tile_size, _TILE_OVERLAP, height, width, target_scale, dtype)
        if geometry != self._logged_geometry:
            # Logged once per distinct shape: a video runs the same geometry for
            # every frame, and repeating this line per frame would bury the log.
            self._logged_geometry = geometry
            logger.info(
                "tile 推理：%s 块（%s 行 x %s 列，tile %s，重叠 %s）· 输入 %sx%s · "
                "输出 %sx%s · %s",
                total,
                len(y_starts),
                len(x_starts),
                tile_size,
                _TILE_OVERLAP,
                width,
                height,
                width * target_scale,
                height * target_scale,
                np.dtype(dtype).name,
            )
        for row, top in enumerate(y_starts):
            bottom = min(top + tile_size, height)
            out_top = top * target_scale
            out_bottom = bottom * target_scale
            # How many output rows/columns this tile shares with the tile to the
            # left / above it (zero when there is no neighbour to fade against).
            band_y = (
                0
                if row == 0
                else max(
                    0,
                    min(y_starts[row - 1] + tile_size, height) * target_scale
                    - out_top,
                )
            )
            for column, left in enumerate(x_starts):
                if self.cancelled:
                    raise SuperResCancelled()
                tile_started = time.perf_counter()
                right = min(left + tile_size, width)
                out_left = left * target_scale
                out_right = right * target_scale
                band_x = (
                    0
                    if column == 0
                    else max(
                        0,
                        min(x_starts[column - 1] + tile_size, width) * target_scale
                        - out_left,
                    )
                )

                tile = tensor[:, :, top:bottom, left:right]
                feeds = self._feeds_for_tile(
                    input_name,
                    tile,
                    target_scale,
                    denoise_value=denoise_value,
                    dtype=dtype,
                )
                output_tile = self._run_tile(session, output_name, feeds)
                output_tile = self._validate_output(
                    output_tile,
                    bottom - top,
                    right - left,
                    target_scale,
                    out_channels,
                )

                if pair is not None:
                    (
                        pair_session,
                        pair_input,
                        pair_output,
                        pair_channels,
                        pair_strength,
                    ) = pair
                    pair_feeds = self._feeds_for_tile(
                        pair_input,
                        tile,
                        target_scale,
                        denoise_value=denoise_value,
                        dtype=self._pair_dtype(),
                    )
                    blended = self._run_tile(pair_session, pair_output, pair_feeds)
                    blended = self._validate_output(
                        blended,
                        bottom - top,
                        right - left,
                        target_scale,
                        pair_channels or out_channels,
                    )
                    output_tile = (
                        output_tile * (1.0 - pair_strength)
                        + blended * pair_strength
                    )

                rgb = postprocess_tensor(output_tile)
                blend_tile(
                    output[out_top:out_bottom, out_left:out_right],
                    rgb,
                    band_x=band_x,
                    band_y=band_y,
                )

                index += 1
                elapsed = time.perf_counter() - started
                tile_ms = (elapsed / index) * 1000.0
                if progress_cb is not None:
                    remaining = max(0, total - index) * elapsed / index
                    progress_cb(
                        ProgressInfo(
                            current=index,
                            total=total,
                            message=(
                                f"tile {index}/{total} · {tile_ms:.0f}ms/块 · "
                                f"剩余 {_format_eta(remaining)}"
                            ),
                        )
                    )
                logger.debug(
                    "tile %s/%s 完成 · %.0fms · 累计 %.2fs",
                    index,
                    total,
                    (time.perf_counter() - tile_started) * 1000.0,
                    elapsed,
                )

                if self.cancelled:
                    raise SuperResCancelled()

        logger.debug("tile 推理完成：%.2fs", time.perf_counter() - started)
        return output

    def _feeds_for_tile(
        self,
        input_name: Optional[str],
        tile: np.ndarray,
        target_scale: int,
        denoise_value: float = 0.0,
        dtype=None,
    ) -> dict:
        if input_name is None:
            raise RuntimeError("模型输入名尚未解析")
        if dtype is None:
            dtype = self._inference_dtype()
        feeds = {
            input_name: np.ascontiguousarray(
                tile,
                dtype=dtype,
            )
        }
        self._add_control_inputs(feeds, target_scale, denoise_value, dtype=dtype)
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

    @staticmethod
    def _validate_output(
        output: np.ndarray,
        tile_height: int,
        tile_width: int,
        target_scale: int,
        out_channels: int,
    ) -> np.ndarray:
        expected_h = tile_height * target_scale
        expected_w = tile_width * target_scale
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

    def _add_control_inputs(
        self,
        feeds: dict,
        target_scale: int,
        denoise_value: float = 0.0,
        dtype=None,
    ) -> None:
        if dtype is None:
            dtype = np.float32
        model = self._model
        if model.scale_input:
            feeds[model.scale_input] = np.asarray([target_scale], dtype=dtype)
        if model.denoise_input:
            # 噪声等级条件化模型（DRUNet 等）的强度输入：由"降噪强度"驱动，
            # 不再硬编码 0.0（该通路此前从未接线）。
            feeds[model.denoise_input] = np.asarray(
                [float(denoise_value)], dtype=dtype
            )

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
