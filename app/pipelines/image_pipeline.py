"""Single-image and batch image super-resolution pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

# 超分工具需处理大图（4x 超分后像素激增），解除 Pillow 解压炸弹限制
Image.MAX_IMAGE_PIXELS = None

from ..core.sr_engine import SuperResCancelled, resolve_denoise_plan
from ..core.postprocess import finishing_touch
from ..core.types import ProgressInfo, SuperResResult

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

RESIZE_MODE_KEEP = "keep"
RESIZE_MODE_RESTORE = "restore"

COMPRESSION_PROFILE_HIGH = "high"
COMPRESSION_PROFILE_COMPRESSED = "compressed"
COMPRESSION_PROFILE_LOSSLESS = "lossless"


def _format_for_extension(extension: str) -> str:
    if extension in {".jpg", ".jpeg"}:
        return "JPEG"
    if extension == ".png":
        return "PNG"
    if extension == ".webp":
        return "WEBP"
    raise ValueError(f"Unsupported output image extension: {extension}")


def _compression_options(image_format: str, profile: str) -> dict[str, Any]:
    if image_format == "JPEG":
        if profile == COMPRESSION_PROFILE_COMPRESSED:
            return {
                "quality": 82,
                "optimize": True,
                "progressive": True,
                "subsampling": "4:2:0",
            }
        if profile == COMPRESSION_PROFILE_LOSSLESS:
            return {
                "quality": 100,
                "optimize": True,
                "progressive": False,
                "subsampling": 0,
            }
        return {"quality": 95, "optimize": True, "subsampling": "4:2:2"}

    if image_format == "PNG":
        level = 9 if profile in {
            COMPRESSION_PROFILE_COMPRESSED,
            COMPRESSION_PROFILE_LOSSLESS,
        } else 6
        return {"compress_level": level, "optimize": True}

    if image_format == "WEBP":
        if profile == COMPRESSION_PROFILE_COMPRESSED:
            return {"quality": 78, "method": 6}
        if profile == COMPRESSION_PROFILE_LOSSLESS:
            return {"lossless": True, "method": 6}
        return {"quality": 92, "method": 6}

    return {}


def _save_image(
    image: np.ndarray,
    out_path: Path,
    compression_profile: str = COMPRESSION_PROFILE_HIGH,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_format = _format_for_extension(out_path.suffix.lower())
    pil_image = Image.fromarray(np.asarray(image), "RGB")

    save_options: dict[str, Any] = {
        "format": image_format,
        **_compression_options(image_format, compression_profile),
    }
    pil_image.save(out_path, **save_options)


def upscale_single(
    image_path: str | Path,
    out_path: str | Path,
    engine: Any,
    scale: int,
    denoise_strength: float = 0.5,
    progress_cb: Optional[Callable[[ProgressInfo], None]] = None,
    resize_to_original: bool = False,
    compression_profile: str = COMPRESSION_PROFILE_HIGH,
    sharpen_strength: float = 0.0,
) -> SuperResResult:
    """Upscale one image and return a result object."""
    image_path = Path(image_path)
    out_path = Path(out_path)
    started = time.perf_counter()

    try:
        _report_phase(progress_cb, 0.05, "decode", "读取图片")
        if out_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported output image extension: {out_path.suffix}")

        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
            original_size = source.size
        _report_phase(progress_cb, 0.15, "decode", "图片读取完成")

        def tile_cb(tile_info: ProgressInfo) -> None:
            tile_total = max(int(tile_info.total), 1)
            tile_current = max(0.0, float(tile_info.current))
            fraction = 0.15 + 0.75 * min(tile_current / tile_total, 1.0)
            _report_phase(
                progress_cb,
                fraction,
                "tile",
                tile_info.message or "推理中",
            )

        # 降噪路由：1x 降噪模型在输入尺度先降噪；同倍率降噪模型在引擎内与输出
        # 插值；两者都不适用时才回退到内置的边缘保持降噪。
        plan = resolve_denoise_plan(engine, denoise_strength, scale)
        if plan.pre_strength > 0.0:
            _report_phase(progress_cb, 0.16, "denoise", "输入尺度降噪")
            image = engine.pre_denoise(image, plan.pre_strength)
            _report_phase(progress_cb, 0.18, "denoise", "输入尺度降噪完成")

        upscaled = engine.upscale_image(
            image,
            scale,
            denoise_strength=denoise_strength,
            progress_cb=tile_cb,
        )
        _report_phase(progress_cb, 0.88, "inference", "推理完成")
        if plan.classical_strength > 0.0 or sharpen_strength > 0.0:
            upscaled = finishing_touch(
                upscaled,
                denoise_strength=plan.classical_strength,
                sharpen_strength=sharpen_strength,
            )
            _report_phase(progress_cb, 0.92, "finish", "降噪与锐化")
        if resize_to_original:
            pil_image = Image.fromarray(np.asarray(upscaled), "RGB")
            upscaled = np.asarray(
                pil_image.resize(original_size, Image.Resampling.LANCZOS)
            )
            _report_phase(progress_cb, 0.94, "resize", "回缩到原图尺寸")
        _report_phase(progress_cb, 0.96, "save", "正在保存")
        _save_image(upscaled, out_path, compression_profile=compression_profile)
        _report_phase(progress_cb, 0.99, "save", "保存完成")
        return SuperResResult(
            ok=True,
            output_path=out_path,
            elapsed_seconds=time.perf_counter() - started,
            input_path=image_path,
        )
    except SuperResCancelled:
        raise
    except Exception as exc:
        return SuperResResult(
            ok=False,
            error=str(exc),
            elapsed_seconds=time.perf_counter() - started,
            input_path=image_path,
        )


def _normalize_paths(paths: Any) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _collect_files(paths: Any) -> list[Path]:
    files: list[Path] = []
    for raw_path in _normalize_paths(paths):
        path = Path(raw_path)
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
            continue
        if path.is_dir():
            directory_files = [
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            files.extend(sorted(directory_files, key=lambda child: str(child)))
    return files


def _should_cancel(cancel_flag: Any) -> bool:
    if cancel_flag is None:
        return False
    if callable(cancel_flag):
        return bool(cancel_flag())
    is_set = getattr(cancel_flag, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    return bool(cancel_flag)


def _report_phase(
    progress_cb: Optional[Callable[[ProgressInfo], None]],
    fraction: float,
    phase: str,
    message: str,
) -> None:
    if progress_cb is None:
        return
    progress_cb(
        ProgressInfo(
            current=min(max(float(fraction), 0.0), 1.0),
            total=1,
            message=message,
            phase=phase,
        )
    )


def upscale_batch(
    paths: Any,
    out_dir: str | Path,
    engine: Any,
    scale: int,
    progress_cb: Optional[Callable[[ProgressInfo], None]] = None,
    cancel_flag: Any = None,
    overwrite: bool = False,
    denoise_strength: float = 0.5,
    resize_to_original: bool = False,
    compression_profile: str = COMPRESSION_PROFILE_HIGH,
    sharpen_strength: float = 0.0,
) -> list[SuperResResult]:
    """Upscale supported files recursively and return one result per input."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _collect_files(paths)
    total = len(files)
    results: list[SuperResResult] = []

    for index, source in enumerate(files, start=1):
        if _should_cancel(cancel_flag) or getattr(engine, "cancelled", False):
            _report_image_progress(
                progress_cb,
                index,
                total,
                source,
                0.0,
                phase="cancelled",
                message="已取消",
            )
            raise SuperResCancelled()

        suffix = "_restored" if resize_to_original else ""
        out_path = out_dir / f"{source.stem}_x{scale}{suffix}{source.suffix.lower()}"
        if out_path.exists() and not overwrite:
            results.append(
                SuperResResult(ok=True, output_path=out_path, input_path=source)
            )
            _report_image_progress(
                progress_cb,
                index,
                total,
                source,
                1.0,
                phase="skipped",
                message="已跳过",
            )
            continue

        def image_cb(image_info: ProgressInfo) -> None:
            fraction = (
                float(image_info.current) / float(image_info.total)
                if image_info.total
                else 0.0
            )
            _report_image_progress(
                progress_cb,
                index,
                total,
                source,
                fraction,
                phase=image_info.phase or "image",
                message=image_info.message,
            )

        result = upscale_single(
            source,
            out_path,
            engine,
            scale,
            denoise_strength=denoise_strength,
            progress_cb=image_cb,
            resize_to_original=resize_to_original,
            compression_profile=compression_profile,
            sharpen_strength=sharpen_strength,
        )
        results.append(result)
        if result.ok:
            _report_image_progress(
                progress_cb,
                index,
                total,
                source,
                1.0,
                phase="image",
                message="完成",
            )

    return results


def _report_image_progress(
    progress_cb: Optional[Callable[[ProgressInfo], None]],
    image_index: int,
    image_total: int,
    source: Path,
    fraction: float,
    phase: str,
    message: str,
) -> None:
    if progress_cb is None:
        return
    fraction = min(max(float(fraction), 0.0), 1.0)
    current = (image_index - 1) + fraction
    detail = f"第 {image_index}/{image_total} 张"
    if message:
        detail += f" · {message}"
    detail += f" · {source.name}"
    progress_cb(
        ProgressInfo(
            current=min(float(image_total), max(0.0, current)),
            total=image_total,
            message=detail,
            image_index=image_index,
            image_total=image_total,
            source_name=source.name,
            phase=phase,
        )
    )
