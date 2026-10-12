"""QThread-backed asynchronous workers for image and video upscaling."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from PySide6.QtCore import QThread, Signal

from ..core.sr_engine import OnnxEngine, SuperResCancelled, VulkanEngine
from ..core.types import Backend, ModelMeta, VideoCodec
from ..pipelines.image_pipeline import upscale_batch
from ..pipelines.video_pipeline import upscale_video

logger = logging.getLogger(__name__)

#: Loading an ONNX session costs 0.3s for the lightweight ESGAN models and
#: ~3s for SwinIR, and users typically run several tasks in a row against the
#: same model. Sessions are therefore kept for reuse; one entry only, so VRAM
#: held by an unused model is released as soon as a different one is loaded.
_ENGINE_CACHE: dict[tuple[str, str], OnnxEngine | VulkanEngine] = {}
_ENGINE_CACHE_LOCK = threading.Lock()


def _cache_key(backend: Backend, model: ModelMeta) -> tuple[str, str]:
    return (backend.value, str(model.model_path))


def _load_engine(backend: Backend, model: ModelMeta) -> OnnxEngine | VulkanEngine:
    key = _cache_key(backend, model)
    with _ENGINE_CACHE_LOCK:
        cached = _ENGINE_CACHE.get(key)
        if cached is not None:
            cached.reset()
            logger.info("复用已加载的推理会话：%s", model.name)
            return cached
        _ENGINE_CACHE.clear()

    if backend == Backend.VULKAN:
        engine = VulkanEngine(backend)
    else:
        engine = OnnxEngine(backend)
    engine.load_model(model)

    with _ENGINE_CACHE_LOCK:
        _ENGINE_CACHE[key] = engine
    return engine


def _runtime_label(engine: OnnxEngine | VulkanEngine) -> str:
    if isinstance(engine, OnnxEngine):
        return engine.runtime_label
    return "Vulkan ncnn"


class ImageUpscaleWorker(QThread):
    """Run ``upscale_batch`` in a worker thread.

    ``finished`` carries ``list[SuperResResult]``. Cancellation is requested
    through a ``threading.Event`` and, when an engine is available, through
    ``OnnxEngine.cancel`` so inference can stop between tiles as well.
    """

    progress = Signal(object)
    runtime = Signal(str)
    log = Signal(str, str)  # (ui_level, message) for the step-by-step run log
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        paths: Iterable[str | Path],
        out_dir: str | Path,
        model: ModelMeta,
        scale: int,
        backend: Backend,
        overwrite: bool = False,
        denoise_strength: float = 0.5,
        sharpen_strength: float = 0.0,
        resize_to_original: bool = False,
        compression_profile: str = "high",
        resize_percent: Optional[float] = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._paths = [Path(path) for path in paths]
        self._out_dir = Path(out_dir)
        self._model = model
        self._scale = int(scale)
        self._backend = backend
        self._overwrite = bool(overwrite)
        self._denoise_strength = float(denoise_strength)
        self._sharpen_strength = float(sharpen_strength)
        self._resize_to_original = bool(resize_to_original)
        self._compression_profile = str(compression_profile)
        self._resize_percent = (
            float(resize_percent) if resize_percent is not None else None
        )
        self._cancel_event = threading.Event()
        self._engine: OnnxEngine | VulkanEngine | None = None

    def cancel(self) -> None:
        self._cancel_event.set()
        engine = self._engine
        if engine is not None:
            engine.cancel()

    def run(self) -> None:
        try:
            self._engine = _load_engine(self._backend, self._model)
            self.runtime.emit(_runtime_label(self._engine))
            results = upscale_batch(
                self._paths,
                self._out_dir,
                self._engine,
                self._scale,
                progress_cb=self.progress.emit,
                cancel_flag=self._cancel_event,
                overwrite=self._overwrite,
                denoise_strength=self._denoise_strength,
                sharpen_strength=self._sharpen_strength,
                resize_to_original=self._resize_to_original,
                compression_profile=self._compression_profile,
                resize_percent=self._resize_percent,
                log_sink=self.log.emit,
            )
            self.finished.emit(results)
        except SuperResCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc) or exc.__class__.__name__)


class VideoUpscaleWorker(QThread):
    """Run ``upscale_video`` in a worker thread."""

    progress = Signal(object)
    runtime = Signal(str)
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        input_path: str | Path,
        out_path: str | Path,
        model: ModelMeta,
        scale: int,
        backend: Backend,
        codec: VideoCodec,
        denoise_strength: float = 0.5,
        sharpen_strength: float = 0.0,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._input_path = Path(input_path)
        self._out_path = Path(out_path)
        self._model = model
        self._scale = int(scale)
        self._backend = backend
        self._codec = codec
        self._denoise_strength = float(denoise_strength)
        self._sharpen_strength = float(sharpen_strength)
        self._cancel_event = threading.Event()
        self._engine: OnnxEngine | VulkanEngine | None = None

    def cancel(self) -> None:
        self._cancel_event.set()
        engine = self._engine
        if engine is not None:
            engine.cancel()

    def run(self) -> None:
        try:
            self._engine = _load_engine(self._backend, self._model)
            self.runtime.emit(_runtime_label(self._engine))
            result = upscale_video(
                self._input_path,
                self._out_path,
                self._engine,
                self._scale,
                codec=self._codec,
                progress_cb=self.progress.emit,
                cancel_flag=self._cancel_event,
                denoise_strength=self._denoise_strength,
                sharpen_strength=self._sharpen_strength,
            )
            if result.ok:
                self.finished.emit(result)
            else:
                self.error.emit(result.error or "视频超分失败")
        except SuperResCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc) or exc.__class__.__name__)
