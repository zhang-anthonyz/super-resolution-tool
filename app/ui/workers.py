"""QThread-backed asynchronous workers for image and video upscaling."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import QThread, Signal

from ..core.sr_engine import OnnxEngine, SuperResCancelled
from ..core.types import Backend, ModelMeta, VideoCodec
from ..pipelines.image_pipeline import upscale_batch
from ..pipelines.video_pipeline import upscale_video


def _load_engine(backend: Backend, model: ModelMeta) -> OnnxEngine:
    if backend == Backend.VULKAN:
        raise RuntimeError("Vulkan 后端暂未就绪，请选择其他后端")
    engine = OnnxEngine(backend)
    engine.load_model(model)
    return engine


class ImageUpscaleWorker(QThread):
    """Run ``upscale_batch`` in a worker thread.

    ``finished`` carries ``list[SuperResResult]``. Cancellation is requested
    through a ``threading.Event`` and, when an engine is available, through
    ``OnnxEngine.cancel`` so inference can stop between tiles as well.
    """

    progress = Signal(object)
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
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._paths = [Path(path) for path in paths]
        self._out_dir = Path(out_dir)
        self._model = model
        self._scale = int(scale)
        self._backend = backend
        self._overwrite = bool(overwrite)
        self._cancel_event = threading.Event()
        self._engine: OnnxEngine | None = None

    def cancel(self) -> None:
        self._cancel_event.set()
        engine = self._engine
        if engine is not None:
            engine.cancel()

    def run(self) -> None:
        try:
            self._engine = _load_engine(self._backend, self._model)
            results = upscale_batch(
                self._paths,
                self._out_dir,
                self._engine,
                self._scale,
                progress_cb=self.progress.emit,
                cancel_flag=self._cancel_event,
                overwrite=self._overwrite,
            )
            self.finished.emit(results)
        except SuperResCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc) or exc.__class__.__name__)


class VideoUpscaleWorker(QThread):
    """Run ``upscale_video`` in a worker thread."""

    progress = Signal(object)
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
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._input_path = Path(input_path)
        self._out_path = Path(out_path)
        self._model = model
        self._scale = int(scale)
        self._backend = backend
        self._codec = codec
        self._cancel_event = threading.Event()
        self._engine: OnnxEngine | None = None

    def cancel(self) -> None:
        self._cancel_event.set()
        engine = self._engine
        if engine is not None:
            engine.cancel()

    def run(self) -> None:
        try:
            self._engine = _load_engine(self._backend, self._model)
            result = upscale_video(
                self._input_path,
                self._out_path,
                self._engine,
                self._scale,
                codec=self._codec,
                progress_cb=self.progress.emit,
                cancel_flag=self._cancel_event,
            )
            if result.ok:
                self.finished.emit(result)
            else:
                self.error.emit(result.error or "视频超分失败")
        except SuperResCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc) or exc.__class__.__name__)

