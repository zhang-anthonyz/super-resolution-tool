"""Runtime provider probing shared by backend selection and inference engines."""
from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from .paths import app_root
from .types import Backend


PROVIDER_BY_BACKEND = {
    Backend.CUDA: "CUDAExecutionProvider",
    Backend.TENSORRT: "CUDAExecutionProvider",
    Backend.DIRECTML: "DmlExecutionProvider",
    Backend.OPENVINO: "OpenVINOExecutionProvider",
    Backend.CPU: "CPUExecutionProvider",
}

PROVIDER_LABELS = {
    "CUDAExecutionProvider": "CUDA",
    "DmlExecutionProvider": "DirectML",
    "OpenVINOExecutionProvider": "OpenVINO",
    "CPUExecutionProvider": "CPU",
}

_RUNTIME_DIR_NAMES = {
    Backend.CUDA: "cuda",
    Backend.TENSORRT: "cuda",
    Backend.DIRECTML: "dml",
    Backend.OPENVINO: "openvino",
}


@lru_cache(maxsize=1)
def available_onnxruntime_providers() -> tuple[str, ...]:
    """Return ONNX Runtime execution providers available to this process."""
    try:
        import onnxruntime
    except Exception:
        return ()

    try:
        return tuple(str(provider) for provider in onnxruntime.get_available_providers())
    except Exception:
        return ()


def provider_for_backend(backend: Backend) -> Optional[str]:
    """Return the ONNX Runtime provider required by ``backend``."""
    return PROVIDER_BY_BACKEND.get(backend)


def backend_runtime_dir(backend: Backend) -> Optional[Path]:
    """Return the bundled runtime directory used by ``backend``, if any."""
    dirname = _RUNTIME_DIR_NAMES.get(backend)
    if dirname is None:
        return None
    return app_root() / "runtime" / dirname


def _bundled_runtime_exists(backend: Backend) -> bool:
    runtime_dir = backend_runtime_dir(backend)
    if runtime_dir is None or not runtime_dir.is_dir():
        return False
    return (runtime_dir / "onnxruntime").is_dir()


def _quiet_onnxruntime_logger(onnxruntime) -> None:
    """Stop ORT's native logger from spamming the console.

    Session creation prints a two-line warning per model ("Some nodes were not
    assigned to the preferred execution providers...", which is expected for the
    CPU shape ops of these graphs). Keep only errors, or plain INFO when the
    user asked for verbose output.
    """
    import logging
    import os

    verbose = os.environ.get("SRT_LOG_LEVEL", "").strip().lower() == "debug"
    setter = getattr(onnxruntime, "set_default_logger_severity", None)
    if callable(setter):
        try:
            setter(1 if verbose else 3)
        except Exception:
            pass
    logging.getLogger("onnxruntime").setLevel(
        logging.INFO if verbose else logging.ERROR
    )


def load_onnxruntime_for_backend(backend: Backend):
    """Load the ONNX Runtime distribution selected for ``backend``.

    Development checkouts usually have one active onnxruntime package. Packaged
    builds may also ship isolated ``runtime/cuda``, ``runtime/dml``, or
    ``runtime/openvino`` directories; when present they take priority.
    """
    runtime_dir = backend_runtime_dir(backend)
    if runtime_dir is not None and _bundled_runtime_exists(backend):
        runtime_path = str(runtime_dir)
        if runtime_path not in sys.path:
            sys.path.insert(0, runtime_path)
        for name in list(sys.modules):
            if name == "onnxruntime" or name.startswith("onnxruntime."):
                del sys.modules[name]
        onnxruntime = importlib.import_module("onnxruntime")
    else:
        onnxruntime = importlib.import_module("onnxruntime")
    _quiet_onnxruntime_logger(onnxruntime)
    return onnxruntime


def backend_available(
    backend: Backend,
    providers: Optional[Iterable[str]] = None,
) -> bool:
    """Return whether ``backend`` can be served by the detected ORT providers."""
    provider = provider_for_backend(backend)
    if provider is None:
        return False
    provider_set = (
        set(available_onnxruntime_providers())
        if providers is None
        else {str(item) for item in providers}
    )
    if provider in provider_set:
        return True
    return providers is None and _bundled_runtime_exists(backend)


def provider_label(provider: str) -> str:
    """Return a short user-facing label for an ORT provider id."""
    return PROVIDER_LABELS.get(provider, provider)
