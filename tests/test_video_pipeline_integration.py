import shutil
import subprocess

import pytest

from app.core.types import Scene
from app.models.model_manager import load_model, scene_default
from app.pipelines.ffmpeg_utils import probe_video
from app.pipelines.video_pipeline import upscale_video


def _dml_available() -> bool:
    try:
        import onnxruntime
    except ImportError:
        return False
    return "DmlExecutionProvider" in onnxruntime.get_available_providers()


def _make_two_second_video(path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=25",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100",
        "-t",
        "2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not in PATH")
@pytest.mark.skipif(not _dml_available(), reason="DirectML provider is not available")
def test_video_upscale_directml_realesrgan_integration(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _make_two_second_video(input_path)

    engine = load_model(scene_default(Scene.REALISTIC))

    result = upscale_video(
        input_path,
        output_path,
        engine,
        scale=4,
    )

    assert result.ok is True, result.error
    assert output_path.is_file()
    output_info = probe_video(output_path)
    assert (output_info["width"], output_info["height"]) == (1280, 960)
    assert output_info["has_audio"] is True
    duration = output_info.get("duration", 0.0)
    assert 1.8 <= duration <= 2.2
