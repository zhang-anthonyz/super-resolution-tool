"""Pure UI helper logic shared by the main window and workers.

The functions in this module intentionally avoid Qt imports so the UI rules
can be unit tested without a QApplication or an offscreen platform.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..core.paths import app_root
from ..core.types import Backend, GpuInfo, GpuVendor, ModelMeta, Scene


DEFAULT_MODEL_RELATIVE_PATH = Path("models/onnx/RealESRGAN_x4plus.onnx")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}

SCENE_OPTIONS: tuple[tuple[Scene, str], ...] = (
    (Scene.REALISTIC, "真实"),
    (Scene.ANIME_2D, "2D 动漫"),
    (Scene.CG_3D, "3D·CG"),
)

SCENE_DEFAULT_MODEL_NAMES = {
    Scene.REALISTIC: "RealESRGAN_x4plus",
    Scene.ANIME_2D: "RealESRGAN_x4plus",
    Scene.CG_3D: "RealESRGAN_x4plus",
}

VENDOR_LABELS = {
    GpuVendor.NVIDIA: "NVIDIA",
    GpuVendor.AMD: "AMD",
    GpuVendor.INTEL: "Intel",
    GpuVendor.UNKNOWN: "未知厂商",
}

BASE_BACKEND_LABELS = {
    Backend.CUDA: "CUDA",
    Backend.TENSORRT: "TensorRT",
    Backend.DIRECTML: "DirectML",
    Backend.OPENVINO: "OpenVINO",
    Backend.VULKAN: "Vulkan",
    Backend.CPU: "CPU",
}

VENDOR_BACKEND_CANDIDATES = {
    GpuVendor.NVIDIA: (Backend.CUDA, Backend.DIRECTML, Backend.CPU),
    GpuVendor.AMD: (Backend.DIRECTML, Backend.VULKAN, Backend.CPU),
    GpuVendor.INTEL: (Backend.DIRECTML, Backend.OPENVINO, Backend.CPU),
    GpuVendor.UNKNOWN: (Backend.CPU,),
}


@dataclass(frozen=True)
class BackendChoice:
    """A backend option with its vendor-aware display label."""

    backend: Backend
    label: str
    available: bool = True

    @property
    def display_label(self) -> str:
        if self.available:
            return self.label
        return f"{self.label}（未就绪）"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a relative path against the application root."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = app_root() / path
    return path


def scene_label(scene: Scene) -> str:
    return dict(SCENE_OPTIONS)[scene]


def default_model_name(scene: Scene) -> str:
    if not isinstance(scene, Scene):
        raise TypeError("scene must be a Scene enum value")
    return SCENE_DEFAULT_MODEL_NAMES[scene]


def default_model_path(scene: Scene) -> Path:
    """Return the currently registered default model for a scene preset."""
    default_model_name(scene)
    return app_root() / DEFAULT_MODEL_RELATIVE_PATH


def fixed_scale_for_model(model_path: str | Path) -> Optional[int]:
    """Return the fixed scale encoded by known model file names.

    Unknown or scale-aware models return ``None`` and use the scale selected
    by the user.
    """
    name = Path(model_path).name.lower()
    if "x4plus" in name or name.startswith("realesrgan_x4"):
        return 4
    return None


def build_model_meta(
    model_path: str | Path,
    scene: Scene,
    backend: Backend,
    fixed_scale: Optional[int] = None,
) -> ModelMeta:
    """Build a runtime model descriptor for the selected UI settings."""
    path = resolve_project_path(model_path)
    if fixed_scale is None:
        fixed_scale = fixed_scale_for_model(path)
    return ModelMeta(
        name=path.stem,
        scale=fixed_scale,
        scene=scene,
        backend=backend,
        model_path=path,
        input_name="input",
        output_name="output",
        scale_input=None,
        denoise_input=None,
        channel_order="RGB",
        normalize=(0.0, 1.0),
        tile_size=512,
        description="User-selected ONNX super-resolution model",
    )


def effective_scale(model_path: str | Path, requested_scale: int) -> tuple[int, bool]:
    """Return the scale to use and whether it differs from the request."""
    requested_scale = int(requested_scale)
    fixed = fixed_scale_for_model(model_path)
    if fixed is not None:
        return fixed, fixed != requested_scale
    return requested_scale, False


def backend_label(backend: Backend, gpu: Optional[GpuInfo] = None) -> str:
    label = BASE_BACKEND_LABELS[backend]
    if gpu is None:
        return label
    if backend == Backend.CUDA and gpu.vendor == GpuVendor.NVIDIA:
        return "CUDA（推荐）"
    if backend == Backend.DIRECTML and gpu.vendor in {GpuVendor.AMD, GpuVendor.INTEL}:
        return "DirectML（推荐）"
    return label


def backend_choices(
    gpu: GpuInfo,
    available: Sequence[Backend] = (),
) -> list[BackendChoice]:
    """Return vendor candidates, marking runtime-unavailable ones as disabled."""
    available_set = set(available)
    candidates = VENDOR_BACKEND_CANDIDATES.get(gpu.vendor, (Backend.CPU,))
    return [
        BackendChoice(
            backend=backend,
            label=backend_label(backend, gpu),
            available=backend in available_set,
        )
        for backend in candidates
    ]


def recommended_backend_index(choices: Sequence[BackendChoice]) -> int:
    for index, choice in enumerate(choices):
        if choice.available:
            return index
    return 0


def gpu_summary(gpu: GpuInfo, available: Sequence[Backend]) -> str:
    choices = backend_choices(gpu, available)
    backend_text = "、".join(
        choice.display_label for choice in choices if choice.available
    ) or "无可用后端"
    return (
        f"{VENDOR_LABELS[gpu.vendor]} / {gpu.name} / "
        f"{gpu.dedicated_vram_gb:.1f} GB / 可用后端：{backend_text}"
    )


def collect_image_files(paths: Iterable[str | Path]) -> list[Path]:
    """Collect supported image files recursively, preserving sorted order."""
    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    files: set[Path] = set()
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_file():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                files.add(path)
            continue
        if path.is_dir():
            files.update(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
            )
    return sorted(files, key=lambda item: str(item))


def is_8bit_sdr(bit_depth: int, is_hdr: bool) -> bool:
    """Return whether media meets the current 8-bit SDR pipeline constraint."""
    return int(bit_depth) <= 8 and not bool(is_hdr)


def video_8bit_notice() -> str:
    return "仅支持 8bit SDR 视频，10bit/HDR 暂不支持"


def format_log_line(
    level: str,
    message: str,
    timestamp: Optional[str] = None,
) -> str:
    """Format a timestamped, level-prefixed UI log line."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] [{level}] {message}"
