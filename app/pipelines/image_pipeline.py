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
from ..core.task_log import TaskLog, format_bytes, format_duration
from ..core.types import ProgressInfo, SuperResResult

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

RESIZE_MODE_KEEP = "keep"
RESIZE_MODE_RESTORE = "restore"
RESIZE_MODE_PERCENT = "percent"

#: Percentage-of-the-super-resolved-result resize: allowed range and default.
PERCENT_MIN = 5
PERCENT_MAX = 400
PERCENT_DEFAULT = 50

COMPRESSION_PROFILE_HIGH = "high"
COMPRESSION_PROFILE_COMPRESSED = "compressed"
COMPRESSION_PROFILE_LOSSLESS = "lossless"

#: How many tile-progress lines one image may put in the run log. Longer jobs
#: report at an even cadence instead of flooding the panel with hundreds of
#: lines (every tile still reaches the progress bar).
MAX_TILE_LOG_LINES = 12

logger = TaskLog("app.pipelines.image_pipeline")


def _format_for_extension(extension: str) -> str:
    if extension in {".jpg", ".jpeg"}:
        return "JPEG"
    if extension == ".png":
        return "PNG"
    if extension == ".webp":
        return "WEBP"
    raise ValueError(f"Unsupported output image extension: {extension}")


def _compression_options(image_format: str, profile: str) -> dict[str, Any]:
    """Encoder options for one output format and compression profile.

    ``optimize`` is deliberately absent for PNG and WEBP: Pillow's PNG
    ``optimize`` runs libpng's per-row filter search, which measured **~11x
    slower** (14.9s vs 1.3s for a 4096x4096 output) for a ~14% smaller file, and
    WEBP's ``method`` 6 costs several times more than 4 for the same bytes.
    Compression level alone already sits within a few percent of the smallest
    file, so the default profile buys 10x the save speed for ~14% more bytes.
    """
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
        # PNG is lossless at every level; the level only trades size for time.
        level = 9 if profile == COMPRESSION_PROFILE_COMPRESSED else 6
        return {"compress_level": level}

    if image_format == "WEBP":
        if profile == COMPRESSION_PROFILE_COMPRESSED:
            return {"quality": 78, "method": 4}
        if profile == COMPRESSION_PROFILE_LOSSLESS:
            return {"lossless": True, "method": 4}
        return {"quality": 92, "method": 4}

    return {}


def _save_image(
    image: np.ndarray,
    out_path: Path,
    compression_profile: str = COMPRESSION_PROFILE_HIGH,
) -> dict[str, Any]:
    """Encode ``image`` and return the encoder options that were used."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_format = _format_for_extension(out_path.suffix.lower())
    pil_image = Image.fromarray(np.asarray(image), "RGB")

    save_options: dict[str, Any] = {
        "format": image_format,
        **_compression_options(image_format, compression_profile),
    }
    pil_image.save(out_path, **save_options)
    return save_options


def _engine_summary(engine: Any) -> str:
    """Model/runtime details for the log, tolerant of engine test doubles."""
    parts: list[str] = []
    runtime = getattr(engine, "runtime_label", None)
    if runtime:
        parts.append(f"运行时 {runtime}")
    precision = getattr(engine, "precision_label", None)
    if precision:
        parts.append(precision)
    return " · ".join(parts)


def percentage_size(
    size: tuple[int, int],
    percent: float,
) -> tuple[int, int]:
    """Scale ``size`` by a percentage, keeping the aspect ratio.

    Both axes get the same factor, so the result stays proportional; each edge
    is rounded to at least one pixel. A factor of exactly 100% is a no-op.
    """
    factor = float(percent) / 100.0
    width = max(1, int(round(size[0] * factor)))
    height = max(1, int(round(size[1] * factor)))
    return (width, height)


def _resize_suffix(percent: float) -> str:
    """Output-file suffix for a percentage resize (empty for a no-op)."""
    if float(percent) == 100.0:
        return ""
    return f"_p{float(percent):g}"


def _is_percentage_noop(percent: float) -> bool:
    return float(percent) == 100.0


def resize_summary(
    resize_to_original: bool = False,
    resize_percent: Optional[float] = None,
) -> str:
    """Human-readable description of the requested output size handling."""
    if resize_percent is not None and not _is_percentage_noop(resize_percent):
        return f"按超分结果的 {float(resize_percent):g}% 等比例缩放"
    if resize_to_original:
        return "回缩到原图尺寸"
    return "保持超分尺寸"


def _denoise_route(plan: Any, denoise_strength: float) -> str:
    """Describe how the requested denoise strength will be served."""
    strength = float(denoise_strength)
    if strength <= 0.0:
        return "关闭"
    if plan.pre_strength > 0.0:
        return f"降噪模型预降噪（输入尺度）强度 {plan.pre_strength:.2f}"
    if plan.post_strength > 0.0:
        return f"降噪模型同倍率插值 强度 {plan.post_strength:.2f}"
    return f"内置边缘保持降噪 强度 {plan.classical_strength:.2f}"


def _tile_log_interval(total: int) -> int:
    """Report every N-th tile so one image logs at most MAX_TILE_LOG_LINES.

    The last tile always reports, which is why the cadence leaves room for it.
    """
    room = max(1, MAX_TILE_LOG_LINES - 1)
    if total <= MAX_TILE_LOG_LINES:
        return 1
    return max(1, total // room)


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
    resize_percent: Optional[float] = None,
    log_sink: Optional[Callable[[str, str], None]] = None,
    image_index: Optional[int] = None,
    image_total: Optional[int] = None,
) -> SuperResResult:
    """Upscale one image and return a result object.

    Output size: ``resize_percent`` scales the super-resolved result by that
    percentage (aspect ratio preserved, 100% = unchanged); otherwise
    ``resize_to_original`` restores the input's pixel dimensions.
    """
    image_path = Path(image_path)
    out_path = Path(out_path)
    log = logger.bind(log_sink) if log_sink is not None else logger
    tag = f"[{image_index}/{image_total}] " if image_index else ""
    started = time.perf_counter()

    try:
        _report_phase(progress_cb, 0.05, "decode", "读取图片")
        if out_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported output image extension: {out_path.suffix}")

        step_started = time.perf_counter()
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
            original_size = source.size
            source_format = source.format or image_path.suffix.lstrip(".")
        decode_seconds = time.perf_counter() - step_started
        log.info(
            "%s解码 %s · %sx%s · %s · %s · %.3fs",
            tag,
            image_path.name,
            original_size[0],
            original_size[1],
            source_format,
            format_bytes(image_path.stat().st_size),
            decode_seconds,
        )
        _report_phase(progress_cb, 0.15, "decode", "图片读取完成")

        # 降噪路由：1x 降噪模型在输入尺度先降噪；同倍率降噪模型在引擎内与输出
        # 插值；两者都不适用时才回退到内置的边缘保持降噪。
        plan = resolve_denoise_plan(engine, denoise_strength, scale)
        log.info(
            "%s设置 倍率 %sx · 输出尺寸 %s · 降噪 %s · 锐化 %.2f · %s",
            tag,
            scale,
            resize_summary(resize_to_original, resize_percent),
            _denoise_route(plan, denoise_strength),
            sharpen_strength,
            _engine_summary(engine) or "后端未知",
        )

        denoise_seconds = 0.0
        if plan.pre_strength > 0.0:
            _report_phase(progress_cb, 0.16, "denoise", "输入尺度降噪")
            step_started = time.perf_counter()
            image = engine.pre_denoise(image, plan.pre_strength)
            denoise_seconds = time.perf_counter() - step_started
            log.info(
                "%s输入尺度预降噪完成 · %.3fs · 输出 %sx%s",
                tag,
                denoise_seconds,
                image.shape[1],
                image.shape[0],
            )
            _report_phase(progress_cb, 0.18, "denoise", "输入尺度降噪完成")

        tile_interval = None

        def tile_cb(tile_info: ProgressInfo) -> None:
            nonlocal tile_interval
            tile_total = max(int(tile_info.total), 1)
            tile_current = max(0.0, float(tile_info.current))
            fraction = 0.15 + 0.75 * min(tile_current / tile_total, 1.0)
            _report_phase(
                progress_cb,
                fraction,
                "tile",
                tile_info.message or "推理中",
            )
            if tile_interval is None:
                tile_interval = _tile_log_interval(tile_total)
                log.info(
                    "%s推理开始 · 共 %s 块 tile · 输出 %sx%s",
                    tag,
                    tile_total,
                    image.shape[1] * scale,
                    image.shape[0] * scale,
                )
            elif (
                int(tile_current) % tile_interval == 0
                or int(tile_current) == tile_total
            ):
                log.info("%s%s", tag, tile_info.message)

        step_started = time.perf_counter()
        upscaled = engine.upscale_image(
            image,
            scale,
            denoise_strength=denoise_strength,
            progress_cb=tile_cb,
        )
        inference_seconds = time.perf_counter() - step_started
        output_pixels = upscaled.shape[0] * upscaled.shape[1]
        log.info(
            "%s推理完成 · %.3fs · %sx%s · %.1f MP/s",
            tag,
            inference_seconds,
            upscaled.shape[1],
            upscaled.shape[0],
            output_pixels / 1e6 / max(inference_seconds, 1e-6),
        )
        _report_phase(progress_cb, 0.88, "inference", "推理完成")

        finish_seconds = 0.0
        if plan.classical_strength > 0.0 or sharpen_strength > 0.0:
            step_started = time.perf_counter()
            upscaled = finishing_touch(
                upscaled,
                denoise_strength=plan.classical_strength,
                sharpen_strength=sharpen_strength,
            )
            finish_seconds = time.perf_counter() - step_started
            detail = []
            if plan.classical_strength > 0.0:
                detail.append(f"边缘保持降噪 {plan.classical_strength:.2f}")
            if sharpen_strength > 0.0:
                detail.append(f"边缘锐化 {sharpen_strength:.2f}")
            log.info(
                "%s后处理完成 · %s · %.3fs",
                tag,
                " + ".join(detail),
                finish_seconds,
            )
            _report_phase(progress_cb, 0.92, "finish", "降噪与锐化")

        resize_seconds = 0.0
        if resize_percent is not None and not _is_percentage_noop(resize_percent):
            target = percentage_size(
                (upscaled.shape[1], upscaled.shape[0]), resize_percent
            )
            step_started = time.perf_counter()
            pil_image = Image.fromarray(np.asarray(upscaled), "RGB")
            upscaled = np.asarray(pil_image.resize(target, Image.Resampling.LANCZOS))
            resize_seconds = time.perf_counter() - step_started
            log.info(
                "%s按超分结果的 %g%% 缩放 → %sx%s · %.3fs",
                tag,
                float(resize_percent),
                target[0],
                target[1],
                resize_seconds,
            )
            _report_phase(
                progress_cb,
                0.94,
                "resize",
                f"按 {float(resize_percent):g}% 等比例缩放",
            )
        elif resize_to_original:
            step_started = time.perf_counter()
            pil_image = Image.fromarray(np.asarray(upscaled), "RGB")
            upscaled = np.asarray(
                pil_image.resize(original_size, Image.Resampling.LANCZOS)
            )
            resize_seconds = time.perf_counter() - step_started
            log.info(
                "%s回缩到原图尺寸 %sx%s · %.3fs",
                tag,
                original_size[0],
                original_size[1],
                resize_seconds,
            )
            _report_phase(progress_cb, 0.94, "resize", "回缩到原图尺寸")

        _report_phase(progress_cb, 0.96, "save", "正在保存")
        step_started = time.perf_counter()
        save_options = _save_image(
            upscaled, out_path, compression_profile=compression_profile
        )
        save_seconds = time.perf_counter() - step_started
        detail = [
            f"{key}={value}"
            for key, value in save_options.items()
            if key != "format"
        ]
        encoded_bytes = out_path.stat().st_size
        log.info(
            "%s编码 %s（%s）→ %s · %.3fs · %.1f MB/s",
            tag,
            save_options["format"],
            " ".join(detail) or compression_profile,
            format_bytes(encoded_bytes),
            save_seconds,
            encoded_bytes / 1e6 / max(save_seconds, 1e-6),
        )
        _report_phase(progress_cb, 0.99, "save", "保存完成")

        elapsed = time.perf_counter() - started
        log.info(
            "%s完成 %s · 总耗时 %s（解码 %.2fs / 推理 %.2fs / 后处理 %.2fs / "
            "编码 %.2fs）",
            tag,
            out_path.name,
            format_duration(elapsed),
            decode_seconds,
            inference_seconds,
            finish_seconds + denoise_seconds + resize_seconds,
            save_seconds,
        )
        return SuperResResult(
            ok=True,
            output_path=out_path,
            elapsed_seconds=elapsed,
            input_path=image_path,
        )
    except SuperResCancelled:
        raise
    except Exception as exc:
        log.error("%s失败 %s：%s", tag, image_path.name, exc)
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
    resize_percent: Optional[float] = None,
    log_sink: Optional[Callable[[str, str], None]] = None,
) -> list[SuperResResult]:
    """Upscale supported files recursively and return one result per input."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _collect_files(paths)
    total = len(files)
    results: list[SuperResResult] = []
    log = logger.bind(log_sink) if log_sink is not None else logger
    batch_started = time.perf_counter()

    log.info(
        "批量任务开始 · %s 张图片 · 输出目录 %s",
        total,
        out_dir,
    )

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
            log.warning("任务已取消 · 完成 %s/%s 张", index - 1, total)
            raise SuperResCancelled()

        suffix = ""
        if resize_percent is not None:
            suffix = _resize_suffix(resize_percent)
        if not suffix and resize_to_original:
            suffix = "_restored"
        out_path = out_dir / f"{source.stem}_x{scale}{suffix}{source.suffix.lower()}"
        if out_path.exists() and not overwrite:
            results.append(
                SuperResResult(ok=True, output_path=out_path, input_path=source)
            )
            log.info("跳过（已存在）：%s", out_path.name)
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
            resize_percent=resize_percent,
            log_sink=log_sink,
            image_index=index,
            image_total=total,
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

    elapsed = time.perf_counter() - batch_started
    ok_count = sum(1 for result in results if result.ok)
    log.success(
        "批量完成 · 成功 %s / 失败 %s · 总耗时 %s · 平均 %s/张",
        ok_count,
        len(results) - ok_count,
        format_duration(elapsed),
        format_duration(elapsed / max(len(results), 1)),
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
