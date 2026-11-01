"""Regression tests for the three orientation / transparency fixes.

* video: a stream carrying a display matrix was reshaped with the *coded*
  dimensions while FFmpeg delivered the rotated frame, so the picture came
  out diagonally sheared (and the task still reported success);
* images: the EXIF orientation was ignored, so a phone photo was written out
  lying on its side;
* images: the alpha channel was dropped, so transparency turned opaque.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.image_io import (
    composite_over_white,
    exif_orientation,
    extract_alpha,
    orient_source,
)
from app.pipelines.image_pipeline import upscale_single
from app.pipelines.video_pipeline import (
    rotated_dimensions,
    rotated_sar,
    rotation_turns,
)

FFMPEG = "ffmpeg"


class _NearestEngine:
    """Repeats pixels so the output size follows the input exactly."""

    def __init__(self):
        self.cancelled = False
        self.shapes: list[tuple[int, ...]] = []

    def upscale_image(self, image, scale=4, progress_cb=None, denoise_strength=0.5):
        self.shapes.append(tuple(image.shape))
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def test_exif_orientation_reads_the_tag():
    image = Image.new("RGB", (4, 4))
    assert exif_orientation(image) == 1
    exif = image.getexif()
    exif[0x0112] = 6
    image.info["exif"] = exif.tobytes()
    assert exif_orientation(image) == 6


def test_orient_source_transposes_and_drops_the_orientation_tag():
    image = Image.new("RGB", (64, 48), (10, 20, 30))
    exif = image.getexif()
    exif[0x0112] = 6
    exif[271] = "Cam"
    image.info["exif"] = exif.tobytes()

    oriented, exif_bytes = orient_source(image)

    assert oriented.size == (48, 64)
    assert oriented.getexif().get(0x0112) is None
    assert oriented.getexif().get(271) == "Cam"
    assert exif_bytes


def test_orient_source_leaves_an_upright_image_untouched():
    image = Image.new("RGB", (64, 48), (10, 20, 30))
    oriented, _exif = orient_source(image)
    assert oriented.size == (64, 48)


def test_extract_alpha_returns_none_for_opaque_images():
    assert extract_alpha(Image.new("RGB", (4, 4))) is None
    assert extract_alpha(Image.new("RGBA", (4, 4), (1, 2, 3, 255))) is None


def test_extract_alpha_reads_a_transparency_mask():
    alpha = extract_alpha(Image.new("RGBA", (4, 4), (1, 2, 3, 128)))
    assert alpha is not None
    assert alpha.mode == "L"
    assert alpha.getextrema() == (128, 128)


def test_composite_over_white_flattens_transparency():
    flattened = composite_over_white(Image.new("RGBA", (2, 2), (0, 0, 0, 0)))
    assert flattened.mode == "RGB"
    assert np.asarray(flattened).min() == 255


# --------------------------------------------------------------------------
# Image pipeline: EXIF orientation
# --------------------------------------------------------------------------


def _write_jpeg_with_orientation(path: Path, orientation: int) -> None:
    image = Image.new("RGB", (64, 48), (200, 30, 30))
    pixels = image.load()
    for y in range(8):                      # marker in the top-left corner
        for x in range(8):
            pixels[x, y] = (0, 255, 0)
    exif = image.getexif()
    exif[0x0112] = orientation
    exif[271] = "TestCam"
    image.save(path, exif=exif, quality=95)


def test_upscale_single_applies_the_exif_orientation(tmp_path):
    source = tmp_path / "phone.jpg"
    output = tmp_path / "phone_x4.jpg"
    _write_jpeg_with_orientation(source, 6)

    result = upscale_single(source, output, _NearestEngine(), scale=4)

    assert result.ok is True, result.error
    with Image.open(output) as saved:
        # Orientation 6 means the stored 64x48 picture is displayed upright.
        assert saved.size == (192, 256)
        assert saved.getexif().get(0x0112) is None
        assert saved.getexif().get(271) == "TestCam"
        array = np.asarray(saved.convert("RGB"))
    # The green marker started top-left; after orientation 6 it is top-right.
    assert tuple(array[0, -1]) != (200, 30, 30)


def test_upscale_single_keeps_an_upright_image_unchanged(tmp_path):
    source = tmp_path / "plain.jpg"
    output = tmp_path / "plain_x2.jpg"
    Image.new("RGB", (16, 16), (40, 50, 60)).save(source)

    result = upscale_single(source, output, _NearestEngine(), scale=2)

    assert result.ok is True
    with Image.open(output) as saved:
        assert saved.size == (32, 32)


# --------------------------------------------------------------------------
# Image pipeline: alpha channel
# --------------------------------------------------------------------------


def _write_rgba_png(path: Path) -> None:
    image = Image.new("RGBA", (32, 32), (10, 20, 30, 255))
    pixels = image.load()
    for y in range(32):
        for x in range(16, 32):             # right half transparent
            pixels[x, y] = (200, 100, 50, 0)
    image.save(path)


def test_upscale_single_preserves_transparency(tmp_path):
    source = tmp_path / "logo.png"
    output = tmp_path / "logo_x4.png"
    _write_rgba_png(source)

    result = upscale_single(source, output, _NearestEngine(), scale=4)

    assert result.ok is True, result.error
    with Image.open(output) as saved:
        assert saved.mode == "RGBA"
        array = np.asarray(saved)
    assert array[64, 10, 3] == 255          # opaque left half
    assert array[64, 120, 3] == 0           # transparent right half


def test_upscale_single_composites_transparency_for_jpeg(tmp_path):
    source = tmp_path / "logo.png"
    output = tmp_path / "logo_x4.jpg"
    _write_rgba_png(source)

    result = upscale_single(source, output, _NearestEngine(), scale=4)

    assert result.ok is True, result.error
    with Image.open(output) as saved:
        assert saved.mode == "RGB"
        array = np.asarray(saved)
    assert tuple(array[64, 10]) == (10, 20, 30)
    assert tuple(array[64, 120]) == (255, 255, 255)


def test_upscale_single_keeps_transparency_through_a_resize(tmp_path):
    source = tmp_path / "logo.png"
    output = tmp_path / "logo_x4.png"
    _write_rgba_png(source)

    result = upscale_single(
        source, output, _NearestEngine(), scale=4, resize_percent=50
    )

    assert result.ok is True, result.error
    with Image.open(output) as saved:
        assert saved.size == (64, 64)
        array = np.asarray(saved)
    assert array[32, 5, 3] == 255
    assert array[32, 60, 3] == 0


def test_upscale_single_carries_the_icc_profile_over(tmp_path):
    """A wide-gamut photo must not silently lose its colour profile."""
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    source = tmp_path / "wide.jpg"
    output = tmp_path / "wide_x2.jpg"
    Image.new("RGB", (16, 16), (40, 50, 60)).save(source, icc_profile=icc)

    result = upscale_single(source, output, _NearestEngine(), scale=2)

    assert result.ok is True, result.error
    with Image.open(output) as saved:
        assert "icc_profile" in saved.info


# --------------------------------------------------------------------------
# Video pipeline: display matrix
# --------------------------------------------------------------------------


def test_rotation_turns_normalises_the_matrix_angle():
    assert rotation_turns(None) == 0
    assert rotation_turns(0) == 0
    assert rotation_turns(90) == 1
    assert rotation_turns(180) == 2
    assert rotation_turns(270) == 3
    assert rotation_turns(-90) == 3


def test_rotated_dimensions_swap_on_a_quarter_turn():
    assert rotated_dimensions(64, 48, 0) == (64, 48)
    assert rotated_dimensions(64, 48, 1) == (48, 64)
    assert rotated_dimensions(64, 48, 2) == (64, 48)
    assert rotated_dimensions(64, 48, 3) == (48, 64)


def test_rotated_sar_inverts_on_a_quarter_turn():
    assert rotated_sar("2:1", 0) == "2:1"
    assert rotated_sar("2:1", 1) == "1:2"
    assert rotated_sar("2:1", 2) == "2:1"
    assert rotated_sar("2:1", 3) == "1:2"
    assert rotated_sar("0:1", 1) == "0:1"       # unknown aspect ratio
    assert rotated_sar("", 1) == ""


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg is not in PATH",
)
def test_probe_video_reads_the_display_matrix(tmp_path):
    from app.pipelines.ffmpeg_utils import probe_video

    source = tmp_path / "plain.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5", "-t", "1",
            "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(source),
        ],
        check=True,
        capture_output=True,
    )
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-display_rotation", "90", "-i", str(source), "-c", "copy", str(rotated),
        ],
        check=True,
        capture_output=True,
    )

    info = probe_video(rotated)

    assert info["width"] == 64 and info["height"] == 48
    assert info["rotation"] == 90
