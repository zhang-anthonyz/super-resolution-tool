"""Single-video super-resolution pipeline using rawvideo FFmpeg pipes."""
from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..core.environment_checker import detect_gpus, detect_video_bit_depth, find_discrete_gpu
from ..core.sr_engine import SuperResCancelled, resolve_denoise_plan
from ..core.postprocess import finishing_touch
from ..core.types import (
    Backend,
    GpuVendor,
    ProgressInfo,
    SuperResResult,
    VideoCodec,
)
from .ffmpeg_utils import encoder_available, encoder_for_vendor, probe_video, run_ffmpeg

logger = logging.getLogger(__name__)


class _EncoderError(RuntimeError):
    """Raised when the selected FFmpeg encoder cannot complete startup."""


class _StderrCollector:
    """Drain subprocess stderr continuously to avoid pipe backpressure."""

    def __init__(self, process: Optional[subprocess.Popen]) -> None:
        self._process = process
        self._chunks: list[bytes] = []
        self._thread: Optional[threading.Thread] = None
        if process is not None and process.stderr is not None:
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()

    def _read(self) -> None:
        stream = self._process.stderr if self._process is not None else None
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                self._chunks.append(chunk)
        except (OSError, ValueError):
            return

    def text(self) -> str:
        if self._thread is not None:
            self._thread.join(timeout=2)
        data = b"".join(self._chunks)
        return data.decode("utf-8", errors="replace").strip()


def output_dimensions(width: int, height: int, scale: int) -> tuple[int, int]:
    """Return integer output dimensions for an integer upscale factor."""
    return int(width) * int(scale), int(height) * int(scale)


def rotation_turns(rotation: Optional[int]) -> int:
    """Number of counter-clockwise quarter turns a display matrix asks for.

    ``probe_video`` reports the rotation as counter-clockwise degrees
    (normalised to 0 / 90 / 180 / 270), matching what FFmpeg's own autorotate
    applies. Pixel-exact reference (verified against FFmpeg): 90° maps to
    ``numpy.rot90(k=1)``, 180° to ``k=2`` and 270° to ``k=3``.
    """
    if rotation is None:
        return 0
    return (int(rotation) % 360) // 90


def rotated_dimensions(width: int, height: int, turns: int) -> tuple[int, int]:
    """Dimensions after applying ``turns`` counter-clockwise quarter turns."""
    if turns % 2:
        return int(height), int(width)
    return int(width), int(height)


def rotated_sar(sar: str, turns: int) -> str:
    """Sample aspect ratio after a quarter turn.

    A quarter turn transposes the pixel grid, so the pixel aspect ratio has
    to be inverted to keep the display aspect ratio intact (``2:1`` becomes
    ``1:2``). This is exactly what FFmpeg's own transcode produces.
    """
    if not turns % 2 or not sar:
        return sar
    if sar in {"0:1", "1:0"}:          # unknown / unset aspect ratio
        return sar
    numerator, _, denominator = sar.partition(":")
    if not numerator.isdigit() or not denominator.isdigit():
        return sar
    return f"{denominator}:{numerator}"


def _fallback_encoder(codec: VideoCodec | str) -> str:
    if isinstance(codec, str):
        codec = VideoCodec(codec.lower())
    return "libx264" if codec == VideoCodec.H264 else "libx265"


def _should_cancel(cancel_flag: Any) -> bool:
    if cancel_flag is None:
        return False
    if callable(cancel_flag):
        return bool(cancel_flag())
    is_set = getattr(cancel_flag, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    return bool(cancel_flag)


def _resolve_vendor(engine: Any) -> GpuVendor:
    backend = getattr(engine, "backend", None)
    if backend in (Backend.CUDA, Backend.TENSORRT):
        return GpuVendor.NVIDIA
    if backend == Backend.OPENVINO:
        return GpuVendor.INTEL
    if backend == Backend.DIRECTML:
        try:
            gpu = find_discrete_gpu(detect_gpus())
        except Exception:
            logger.debug("GPU detection failed while choosing video encoder", exc_info=True)
            gpu = None
        if gpu is not None and gpu.vendor != GpuVendor.UNKNOWN:
            return gpu.vendor
        # DirectML is commonly used on AMD in this project; unknown hardware
        # still gets a hardware route attempt through encoder_available.
        return GpuVendor.AMD
    return GpuVendor.UNKNOWN


def _select_encoder(vendor: GpuVendor, codec: VideoCodec) -> str:
    preferred = encoder_for_vendor(vendor, codec)
    fallback = _fallback_encoder(codec)
    if preferred == fallback:
        return preferred
    if encoder_available(preferred):
        return preferred
    logger.warning(
        "FFmpeg encoder %s is unavailable; falling back to %s",
        preferred,
        fallback,
    )
    return fallback


def _encoder_quality_args(encoder: str) -> list[str]:
    if encoder == "libx264":
        return ["-crf", "18", "-preset", "veryfast"]
    if encoder == "libx265":
        return ["-crf", "20", "-preset", "veryfast"]
    if encoder in {"h264_nvenc", "hevc_nvenc"}:
        return ["-cq", "21", "-preset", "p4"]
    if encoder in {"h264_amf", "hevc_amf"}:
        return ["-rc", "cqp", "-qp_i", "18", "-qp_p", "20"]
    if encoder in {"h264_qsv", "hevc_qsv"}:
        return ["-global_quality", "22"]
    return []


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_stderr(process: subprocess.Popen) -> str:
    stderr = process.stderr
    if stderr is None:
        return ""
    try:
        return stderr.read().decode("utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return ""


def _terminate(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
    except OSError:
        logger.debug("Failed to terminate FFmpeg process", exc_info=True)


def _report_progress(
    progress_cb: Optional[Callable[[ProgressInfo], None]],
    current: int,
    total: int,
) -> None:
    if progress_cb is None:
        return
    if total > 0:
        message = f"frame {current}/{total}"
        info_total = total
    else:
        message = f"processed {current} frames"
        info_total = 0
    progress_cb(ProgressInfo(current=current, total=info_total, message=message))


def _run_to_completion(args: list[str], description: str) -> None:
    process = run_ffmpeg(args)
    _, stderr = process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
        raise RuntimeError(f"{description} failed: {detail}")


def _encode_video(
    input_path: Path,
    output_path: Path,
    engine: Any,
    scale: int,
    width: int,
    height: int,
    fps: float,
    encoder: str,
    total_frames: int,
    progress_cb: Optional[Callable[[ProgressInfo], None]],
    cancel_flag: Any,
    denoise_strength: float = 0.5,
    sharpen_strength: float = 0.0,
    sar: str = "",
    rotation: Optional[int] = None,
) -> None:
    # The display matrix is applied by us, not by FFmpeg: the decoder runs with
    # ``-noautorotate`` so the frames always arrive in the *coded* orientation
    # that ``probe_video`` reported, and the rotation is applied to the decoded
    # frame below. Letting FFmpeg autorotate instead would hand us frames whose
    # height/width are swapped for a quarter turn while the probe still reports
    # the coded size, and the raw frame would then be reshaped with the wrong
    # stride (measured: the picture comes out diagonally sheared).
    turns = rotation_turns(rotation)
    display_width, display_height = rotated_dimensions(width, height, turns)
    out_width, out_height = output_dimensions(display_width, display_height, scale)
    frame_size = width * height * 3

    decoder_args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-noautorotate",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    encoder_args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{out_width}x{out_height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        *_encoder_quality_args(encoder),
    ]
    if sar and sar not in {"0:1", "1:1"}:
        output_sar = rotated_sar(sar, turns)
        if output_sar not in {"0:1", "1:1"}:
            encoder_args.extend(["-vf", f"setsar={output_sar.replace(':', '/')}"])
    # No display matrix is written to the output: the rotation is already
    # baked into the pixels (see the decoder note above), and FFmpeg's
    # ``-metadata:s:v:0 rotate=`` tag is dropped by modern MP4/MOV muxers
    # anyway, so a matrix could not be preserved reliably.
    encoder_args.extend(["-y", str(output_path)])

    try:
        decoder = run_ffmpeg(decoder_args)
    except OSError as exc:
        raise RuntimeError("Failed to start FFmpeg decoder") from exc

    try:
        encoder_process = run_ffmpeg(encoder_args)
    except OSError as exc:
        _terminate(decoder)
        raise _EncoderError(f"Failed to start FFmpeg encoder {encoder}") from exc

    decoder_stderr = _StderrCollector(decoder)
    encoder_stderr = _StderrCollector(encoder_process)

    plan = resolve_denoise_plan(engine, denoise_strength, scale)
    frames_written = 0
    try:
        while True:
            if _should_cancel(cancel_flag) or getattr(engine, "cancelled", False):
                raise SuperResCancelled()

            raw_frame = _read_exact(decoder.stdout, frame_size)
            if not raw_frame:
                break
            if len(raw_frame) != frame_size:
                raise RuntimeError(
                    f"Truncated raw video frame: expected {frame_size} bytes, "
                    f"got {len(raw_frame)}"
                )

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            if turns % 4:
                # Rotate into display orientation before upscaling. This is the
                # step that was missing: FFmpeg delivers the *coded* frame
                # (because of ``-noautorotate``) while the model and the
                # encoder both work in display orientation, so the rotation has
                # to happen here. ``np.rot90`` matches FFmpeg's own autorotate
                # pixel for pixel (verified for 90°, 180° and 270°).
                frame = np.ascontiguousarray(np.rot90(frame, turns))
            if plan.pre_strength > 0.0:
                frame = engine.pre_denoise(frame, plan.pre_strength)
            upscaled = engine.upscale_image(
                frame,
                scale,
                denoise_strength=denoise_strength,
            )
            if plan.classical_strength > 0.0 or sharpen_strength > 0.0:
                upscaled = finishing_touch(
                    upscaled,
                    denoise_strength=plan.classical_strength,
                    sharpen_strength=sharpen_strength,
                )
            upscaled = np.asarray(upscaled)
            if upscaled.dtype != np.uint8:
                raise RuntimeError("Upscale engine must return an uint8 RGB frame")
            if upscaled.shape[:2] != (out_height, out_width):
                raise RuntimeError(
                    f"Upscale engine returned invalid dimensions: "
                    f"{upscaled.shape[:2]} != {(out_height, out_width)}"
                )

            try:
                encoder_process.stdin.write(np.ascontiguousarray(upscaled).tobytes())
                encoder_process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                _terminate(decoder)
                _terminate(encoder_process)
                detail = encoder_stderr.text()
                raise _EncoderError(
                    f"FFmpeg encoder {encoder} exited during writing: {detail}"
                ) from exc

            frames_written += 1
            _report_progress(progress_cb, frames_written, total_frames)

        decoder_stdout_closed = decoder.stdout is not None
        if decoder_stdout_closed:
            decoder.stdout.close()
        decoder_returncode = decoder.wait()
        if decoder_returncode != 0:
            _terminate(encoder_process)
            detail = decoder_stderr.text()
            raise RuntimeError(f"FFmpeg decoding failed: {detail}")
    except Exception:
        _terminate(decoder)
        _terminate(encoder_process)
        raise

    if encoder_process.stdin is not None:
        encoder_process.stdin.close()
    encoder_returncode = encoder_process.wait()
    if encoder_returncode != 0:
        detail = encoder_stderr.text()
        raise _EncoderError(f"FFmpeg encoder {encoder} failed: {detail}")


def _audio_extension(input_path: Path, out_path: Path) -> str:
    if out_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        return ".m4a"
    if out_path.suffix.lower() == ".mkv":
        return ".mka"
    return ".mka"


def _remux_audio(
    input_path: Path,
    video_only: Path,
    out_path: Path,
    audio_path: Path,
) -> None:
    _run_to_completion(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "copy",
            str(audio_path),
        ],
        "audio extraction",
    )
    _run_to_completion(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video_only),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        "audio remux",
    )


def _remux_without_audio(
    input_path: Path,
    video_only: Path,
    out_path: Path,
) -> None:
    _run_to_completion(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video_only),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        "video remux",
    )


def upscale_video(
    input_path: str | Path,
    out_path: str | Path,
    engine: Any,
    scale: int,
    codec: VideoCodec = VideoCodec.H264,
    progress_cb: Optional[Callable[[ProgressInfo], None]] = None,
    cancel_flag: Any = None,
    denoise_strength: float = 0.5,
    sharpen_strength: float = 0.0,
) -> SuperResResult:
    """Upscale a single video frame by frame.

    Only 8-bit SDR input is accepted. Video is decoded to raw RGB24, upscaled,
    encoded through a vendor-selected FFmpeg encoder, and finally remuxed with
    copied audio when present.
    """
    input_path = Path(input_path)
    out_path = Path(out_path)
    started = time.perf_counter()

    try:
        if _should_cancel(cancel_flag) or getattr(engine, "cancelled", False):
            raise SuperResCancelled()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        bit_depth, is_hdr = detect_video_bit_depth(str(input_path))
        if bit_depth > 8 or is_hdr:
            return SuperResResult(
                ok=False,
                error="仅支持 8bit SDR，检测到 10bit/HDR 视频",
                elapsed_seconds=time.perf_counter() - started,
            )

        info = probe_video(input_path)
        width = int(info["width"])
        height = int(info["height"])
        fps = float(info.get("fps") or 0.0)
        if fps <= 0:
            fps = 25.0
        total_frames = int(info.get("total_frames") or 0)
        sar = str(info.get("sar") or "")
        rotation = info.get("rotation")
        rotation = int(rotation) if rotation is not None else None
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions: {width}x{height}")

        vendor = _resolve_vendor(engine)
        encoder = _select_encoder(vendor, codec)
        fallback = _fallback_encoder(codec)

        with tempfile.TemporaryDirectory(prefix="srt-video-") as temp_dir:
            temp_root = Path(temp_dir)
            video_only = temp_root / "video_only.mp4"
            _report_progress(progress_cb, 0, total_frames)

            try:
                _encode_video(
                    input_path=input_path,
                    output_path=video_only,
                    engine=engine,
                    scale=int(scale),
                    width=width,
                    height=height,
                    fps=fps,
                    encoder=encoder,
                    total_frames=total_frames,
                    progress_cb=progress_cb,
                    cancel_flag=cancel_flag,
                    denoise_strength=denoise_strength,
                    sharpen_strength=sharpen_strength,
                    sar=sar,
                    rotation=rotation,
                )
            except _EncoderError:
                if encoder == fallback:
                    raise
                video_only.unlink(missing_ok=True)
                logger.warning(
                    "Encoder %s failed during encoding; retrying with %s",
                    encoder,
                    fallback,
                )
                _report_progress(progress_cb, 0, total_frames)
                _encode_video(
                    input_path=input_path,
                    output_path=video_only,
                    engine=engine,
                    scale=int(scale),
                    width=width,
                    height=height,
                    fps=fps,
                    encoder=fallback,
                    total_frames=total_frames,
                    progress_cb=progress_cb,
                    cancel_flag=cancel_flag,
                    denoise_strength=denoise_strength,
                    sharpen_strength=sharpen_strength,
                    sar=sar,
                    rotation=rotation,
                )

            if info["has_audio"]:
                audio_path = temp_root / f"audio{_audio_extension(input_path, out_path)}"
                _remux_audio(input_path, video_only, out_path, audio_path)
            else:
                _remux_without_audio(input_path, video_only, out_path)

        return SuperResResult(
            ok=True,
            output_path=out_path,
            elapsed_seconds=time.perf_counter() - started,
        )
    except SuperResCancelled:
        raise
    except Exception as exc:
        logger.exception("Video upscaling failed")
        return SuperResResult(
            ok=False,
            error=str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )
