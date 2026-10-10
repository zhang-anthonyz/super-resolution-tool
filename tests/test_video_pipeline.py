import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.core.sr_engine import SuperResCancelled
from app.core.types import Backend, GpuVendor, VideoCodec
from app.pipelines import video_pipeline
from app.pipelines.ffmpeg_utils import probe_video
from app.pipelines.video_pipeline import (
    _encode_video,
    _resolve_vendor,
    output_dimensions,
    upscale_video,
)


class NearestEngine:
    def __init__(self, backend=None, cancelled=False):
        self.backend = backend
        self.cancelled = cancelled
        self.calls = 0
        self.denoise_strengths = []

    def upscale_image(self, image, scale=4, progress_cb=None, denoise_strength=0.5):
        self.calls += 1
        self.denoise_strengths.append(denoise_strength)
        if self.cancelled:
            raise SuperResCancelled()
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def _run_ffmpeg_generation(args, **kwargs):
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        **kwargs,
    )


def _make_video(path: Path, size=32, rate=2, frames=2, audio=False) -> None:
    ffmpeg = "ffmpeg"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={size}x{size}:rate={rate}",
    ]
    if audio:
        command += [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            str(frames / rate),
        ]
    command += [
        "-frames:v",
        str(frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if audio:
        command += ["-c:a", "aac", "-shortest"]
    command.append(str(path))
    _run_ffmpeg_generation(command)


def test_output_dimensions_calculation():
    assert output_dimensions(320, 240, 4) == (1280, 960)
    assert output_dimensions(1280, 720, 2) == (2560, 1440)


def test_upscale_video_blocks_10bit_or_hdr(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"")
    output_path = tmp_path / "output.mp4"
    engine = NearestEngine()
    monkeypatch.setattr(video_pipeline, "detect_video_bit_depth", lambda path: (10, True))

    result = upscale_video(input_path, output_path, engine, scale=2)

    assert result.ok is False
    assert result.error == "仅支持 8bit SDR，检测到 10bit/HDR 视频"
    assert engine.calls == 0
    assert not output_path.exists()


def test_upscale_video_uses_rawvideo_pipes_and_reports_frames(tmp_path, monkeypatch):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _make_video(input_path, size=32, rate=2, frames=2)
    engine = NearestEngine(backend=Backend.CPU)
    progress = []

    result = upscale_video(
        input_path,
        output_path,
        engine,
        scale=2,
        progress_cb=lambda info: progress.append((info.current, info.total)),
    )

    assert result.ok is True
    assert result.output_path == output_path
    assert output_path.is_file()
    assert engine.calls == 2
    assert engine.denoise_strengths == [0.5, 0.5]
    assert progress == [(0, 2), (1, 2), (2, 2)]
    output_info = probe_video(output_path)
    assert (output_info["width"], output_info["height"]) == (64, 64)
    assert output_info["has_audio"] is False


def test_upscale_video_remuxes_audio(tmp_path):
    input_path = tmp_path / "input_audio.mp4"
    output_path = tmp_path / "output_audio.mp4"
    _make_video(input_path, size=32, rate=4, frames=2, audio=True)

    result = upscale_video(
        input_path,
        output_path,
        NearestEngine(backend=Backend.CPU),
        scale=2,
    )

    assert result.ok is True
    assert output_path.is_file()
    output_info = probe_video(output_path)
    assert (output_info["width"], output_info["height"]) == (64, 64)
    assert output_info["has_audio"] is True


def test_upscale_video_cancel_flag_propagates_and_cleans_up(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _make_video(input_path, size=32, rate=4, frames=4)
    cancel_calls = 0

    def cancel_flag():
        nonlocal cancel_calls
        cancel_calls += 1
        return cancel_calls >= 3

    engine = NearestEngine(backend=Backend.CPU)

    with pytest.raises(SuperResCancelled):
        upscale_video(
            input_path,
            output_path,
            engine,
            scale=2,
            cancel_flag=cancel_flag,
        )

    assert not output_path.exists()
    assert engine.calls == 1


def test_resolve_vendor_from_engine_and_gpu(tmp_path, monkeypatch):
    class CudaEngine:
        backend = Backend.CUDA

    class DmlEngine:
        backend = Backend.DIRECTML

    assert _resolve_vendor(CudaEngine()) == GpuVendor.NVIDIA
    assert _resolve_vendor(NearestEngine(backend=None)) == GpuVendor.UNKNOWN

    monkeypatch.setattr(
        video_pipeline,
        "detect_gpus",
        lambda: [object()],
    )
    class FakeGpu:
        vendor = GpuVendor.INTEL

    fake_gpu = FakeGpu()
    monkeypatch.setattr(video_pipeline, "find_discrete_gpu", lambda gpus: fake_gpu)
    assert _resolve_vendor(DmlEngine()) == GpuVendor.INTEL
