"""FFmpeg helpers for media probing, encoder routing, and subprocess startup."""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from ..core.paths import app_root
from ..core.types import GpuVendor, VideoCodec

logger = logging.getLogger(__name__)


def _vendor_executable_path(name: str) -> Optional[Path]:
    """Return the packaged/source vendor executable when it exists."""
    candidate = app_root() / "vendor" / f"{name}.exe"
    return candidate if candidate.is_file() else None


def _first_executable(command: list[str]) -> str:
    if not command:
        raise ValueError("ffmpeg command must not be empty")
    name = command[0]
    explicit = Path(name)
    if explicit.is_absolute():
        return str(explicit)

    vendor_executable = _vendor_executable_path(name)
    if vendor_executable is not None:
        return str(vendor_executable)

    resolved = shutil.which(name)
    return str(resolved) if resolved else str(name)


def run_ffmpeg(args: list[str], **popen_kwargs: Any) -> subprocess.Popen:
    """Start an FFmpeg subprocess without a shell.

    ``args`` is a full argv list whose first item is either an executable name
    such as ``ffmpeg`` or an explicit path. Remaining items are passed through
    unchanged.
    """
    if not isinstance(args, (list, tuple)) or not args:
        raise ValueError("run_ffmpeg requires a non-empty argument list")

    executable = _first_executable([str(arg) for arg in args])
    command = [executable, *[str(arg) for arg in args[1:]]]

    defaults: dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    defaults.update(popen_kwargs)
    defaults["shell"] = False
    return subprocess.Popen(command, **defaults)


def _parse_fraction(value: Any) -> float:
    if value in (None, "", "0/0"):
        return 0.0
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return 0.0
            return float(numerator) / denominator_value
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _stream_total_frames(stream: dict[str, Any], fps: float) -> int:
    for key in ("nb_frames", "nb_read_frames"):
        try:
            value = int(stream.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value

    duration = stream.get("duration")
    if duration is None:
        duration = stream.get("DURATION")
    try:
        duration_seconds = float(duration)
    except (TypeError, ValueError):
        duration_seconds = 0.0
    if duration_seconds > 0 and fps > 0:
        return int(round(duration_seconds * fps))
    return 0


def probe_video(path: str | Path) -> dict[str, Any]:
    """Return basic media information for the first video stream.

    Keys returned are ``width``, ``height``, ``fps``, ``pix_fmt``,
    ``color_transfer``, ``color_primaries``, ``total_frames``, and
    ``has_audio``. ``total_frames`` may be zero when FFprobe cannot determine
    it, which is normal for variable-frame-rate media.
    """
    executable = _vendor_executable_path("ffprobe") or shutil.which("ffprobe")
    if not executable:
        raise FileNotFoundError("ffprobe not found in PATH")

    command = [
        str(executable),
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed for {path}: {detail}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path}") from exc

    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        raise RuntimeError(f"ffprobe returned unexpected streams for {path}")

    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError(f"No video stream found in {path}")
    video = video_streams[0]

    try:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"ffprobe returned invalid dimensions for {path}") from exc

    fps = _parse_fraction(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    if fps <= 0:
        fps = _parse_fraction(video.get("r_frame_rate"))

    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    duration = video.get("duration")
    if duration is None:
        duration = video.get("DURATION")
    if duration is None:
        duration = payload.get("format", {}).get("duration")
    try:
        duration_seconds = float(duration)
    except (TypeError, ValueError):
        duration_seconds = 0.0

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "pix_fmt": str(video.get("pix_fmt") or ""),
        "color_transfer": str(video.get("color_transfer") or ""),
        "color_primaries": str(video.get("color_primaries") or ""),
        "total_frames": _stream_total_frames(video, fps),
        "has_audio": has_audio,
        "duration": duration_seconds,
    }


def encoder_for_vendor(vendor: GpuVendor | str, codec: VideoCodec | str) -> str:
    """Map a GPU vendor and target codec to an FFmpeg encoder name."""
    if isinstance(vendor, str):
        vendor = GpuVendor(vendor.lower())
    if isinstance(codec, str):
        codec = VideoCodec(codec.lower())

    if codec == VideoCodec.H264:
        return {
            GpuVendor.NVIDIA: "h264_nvenc",
            GpuVendor.AMD: "h264_amf",
            GpuVendor.INTEL: "h264_qsv",
            GpuVendor.UNKNOWN: "libx264",
        }[vendor]
    if codec == VideoCodec.HEVC:
        return {
            GpuVendor.NVIDIA: "hevc_nvenc",
            GpuVendor.AMD: "hevc_amf",
            GpuVendor.INTEL: "hevc_qsv",
            GpuVendor.UNKNOWN: "libx265",
        }[vendor]
    raise ValueError(f"Unsupported video codec: {codec}")


def encoder_available(encoder: str) -> bool:
    """Return whether FFmpeg advertises ``encoder`` in ``-encoders`` output."""
    try:
        process = run_ffmpeg(["ffmpeg", "-hide_banner", "-encoders"])
    except (OSError, ValueError):
        return False

    try:
        stdout, _ = process.communicate(timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False

    if process.returncode != 0:
        return False
    text = stdout.decode("utf-8", errors="replace")
    pattern = rf"(?m)^\s*[A-Z.]+\s+{re.escape(encoder)}\s"
    return bool(re.search(pattern, text))
