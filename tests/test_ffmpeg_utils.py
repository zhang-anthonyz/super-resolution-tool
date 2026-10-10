import json
import subprocess

import pytest

from app.core.types import GpuVendor, VideoCodec
from app.pipelines import ffmpeg_utils
from app.pipelines.ffmpeg_utils import (
    encoder_available,
    encoder_for_vendor,
    probe_video,
    run_ffmpeg,
)


@pytest.mark.parametrize(
    ("vendor", "expected"),
    [
        (GpuVendor.NVIDIA, "h264_nvenc"),
        (GpuVendor.AMD, "h264_amf"),
        (GpuVendor.INTEL, "h264_qsv"),
        (GpuVendor.UNKNOWN, "libx264"),
    ],
)
def test_encoder_for_vendor_h264_routes(vendor, expected):
    assert encoder_for_vendor(vendor, VideoCodec.H264) == expected


@pytest.mark.parametrize(
    ("vendor", "expected"),
    [
        (GpuVendor.NVIDIA, "hevc_nvenc"),
        (GpuVendor.AMD, "hevc_amf"),
        (GpuVendor.INTEL, "hevc_qsv"),
        (GpuVendor.UNKNOWN, "libx265"),
    ],
)
def test_encoder_for_vendor_hevc_routes(vendor, expected):
    assert encoder_for_vendor(vendor, VideoCodec.HEVC) == expected


def test_encoder_for_vendor_accepts_strings():
    assert encoder_for_vendor("nvidia", "h264") == "h264_nvenc"
    assert encoder_for_vendor("unknown", "hevc") == "libx265"


def test_probe_video_parses_ffprobe_json(tmp_path, monkeypatch):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr(ffmpeg_utils, "_vendor_executable_path", lambda name: None)
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda name: f"C:/{name}.exe")

    stdout = json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "25/1",
                    "pix_fmt": "yuv420p",
                    "color_transfer": "bt709",
                    "color_primaries": "bt709",
                    "nb_frames": "50",
                },
                {"codec_type": "audio"},
            ]
        }
    )

    def fake_run(command, **kwargs):
        assert command[0] == "C:/ffprobe.exe"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", fake_run)

    result = probe_video(video)

    assert result == {
        "width": 1280,
        "height": 720,
        "fps": 25.0,
        "pix_fmt": "yuv420p",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "total_frames": 50,
        "has_audio": True,
        "duration": 0.0,
    }


def test_probe_video_reports_no_audio(tmp_path, monkeypatch):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr(ffmpeg_utils, "_vendor_executable_path", lambda name: None)
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda name: f"C:/{name}.exe")
    stdout = json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 64,
                    "height": 64,
                    "avg_frame_rate": "30/1",
                    "pix_fmt": "yuv420p",
                }
            ]
        }
    )
    monkeypatch.setattr(
        ffmpeg_utils.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=stdout, stderr=""
        ),
    )

    result = probe_video(video)

    assert result["has_audio"] is False
    assert result["total_frames"] == 0


def test_run_ffmpeg_resolves_name_without_shell(monkeypatch):
    monkeypatch.setattr(ffmpeg_utils, "_vendor_executable_path", lambda name: None)
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda name: f"C:/tools/{name}.exe")
    captured = {}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs

    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", FakeProcess)

    process = run_ffmpeg(["ffmpeg", "-version"])

    assert isinstance(process, FakeProcess)
    assert captured["command"] == ["C:/tools/ffmpeg.exe", "-version"]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["stdin"] == subprocess.PIPE


def test_encoder_available_parses_encoder_list(monkeypatch):
    class FakeProcess:
        returncode = 0

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, timeout=None):
            return b" V....D h264_amf AMD AMF H.264 Encoder\n", b""

    monkeypatch.setattr(ffmpeg_utils, "run_ffmpeg", FakeProcess)

    assert encoder_available("h264_amf") is True
    assert encoder_available("missing_encoder") is False
