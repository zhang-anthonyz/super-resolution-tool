"""Single-image and batch image super-resolution pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

# 超分工具需处理大图（4x 超分后像素激增），解除 Pillow 解压炸弹限制
Image.MAX_IMAGE_PIXELS = None

from ..core.sr_engine import SuperResCancelled
from ..core.types import ProgressInfo, SuperResResult

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _format_for_extension(extension: str) -> str:
    if extension in {".jpg", ".jpeg"}:
        return "JPEG"
    if extension == ".png":
        return "PNG"
    if extension == ".webp":
        return "WEBP"
    raise ValueError(f"Unsupported output image extension: {extension}")


def _save_image(image: np.ndarray, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_format = _format_for_extension(out_path.suffix.lower())
    pil_image = Image.fromarray(np.asarray(image), "RGB")

    save_options: dict[str, Any] = {"format": image_format}
    if image_format == "JPEG":
        save_options["quality"] = 95
    pil_image.save(out_path, **save_options)


def upscale_single(
    image_path: str | Path,
    out_path: str | Path,
    engine: Any,
    scale: int,
    denoise_strength: float = 0.5,
    progress_cb: Optional[Callable[[ProgressInfo], None]] = None,
) -> SuperResResult:
    """Upscale one image and return a result object."""
    image_path = Path(image_path)
    out_path = Path(out_path)
    started = time.perf_counter()

    try:
        if out_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported output image extension: {out_path.suffix}")

        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        upscaled = engine.upscale_image(
            image,
            scale,
            denoise_strength=denoise_strength,
            progress_cb=progress_cb,
        )
        _save_image(upscaled, out_path)
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


def _report_progress(
    progress_cb: Optional[Callable[[ProgressInfo], None]],
    current: float,
    total: int,
    source: Path,
    skipped: bool = False,
    cancelled: bool = False,
) -> None:
    if progress_cb is None:
        return
    message = f"{current:.0f}/{total} {source.name}"
    if skipped:
        message += " skipped"
    progress_cb(
        ProgressInfo(
            current=current,
            total=total,
            message=message,
            cancelled=cancelled,
            image_index=int(current),
            image_total=total,
            source_name=source.name,
            phase="skipped" if skipped else "image",
        )
    )


def _report_tile_progress(
    progress_cb: Optional[Callable[[ProgressInfo], None]],
    image_index: int,
    image_total: int,
    source: Path,
    tile_info: ProgressInfo,
) -> None:
    if progress_cb is None:
        return

    tile_total = max(int(tile_info.total), 1)
    tile_current = max(0.0, float(tile_info.current))
    overall_current = (image_index - 1) + (tile_current / tile_total)
    overall_current = min(float(image_total), max(0.0, overall_current))
    progress_cb(
        ProgressInfo(
            current=overall_current,
            total=image_total,
            message=(
                f"第 {image_index}/{image_total} 张 · tile "
                f"{int(tile_current)}/{tile_total} · {source.name}"
            ),
            image_index=image_index,
            image_total=image_total,
            source_name=source.name,
            phase="tile",
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
) -> list[SuperResResult]:
    """Upscale supported files recursively and return one result per input."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _collect_files(paths)
    total = len(files)
    results: list[SuperResResult] = []

    for index, source in enumerate(files, start=1):
        if _should_cancel(cancel_flag) or getattr(engine, "cancelled", False):
            _report_progress(
                progress_cb,
                index,
                total,
                source,
                cancelled=True,
            )
            raise SuperResCancelled()

        out_path = out_dir / f"{source.stem}_x{scale}{source.suffix.lower()}"
        if out_path.exists() and not overwrite:
            results.append(
                SuperResResult(ok=True, output_path=out_path, input_path=source)
            )
            _report_progress(progress_cb, index, total, source, skipped=True)
            continue

        def tile_cb(tile_info: ProgressInfo) -> None:
            _report_tile_progress(progress_cb, index, total, source, tile_info)

        result = upscale_single(
            source,
            out_path,
            engine,
            scale,
            denoise_strength=denoise_strength,
            progress_cb=tile_cb,
        )
        results.append(result)
        _report_progress(progress_cb, index, total, source)

    return results
