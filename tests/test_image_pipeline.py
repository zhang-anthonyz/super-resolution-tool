from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.sr_engine import SuperResCancelled
from app.core.types import ProgressInfo, SuperResResult
from app.pipelines.image_pipeline import (
    SUPPORTED_EXTENSIONS,
    upscale_batch,
    upscale_single,
)


class _NearestEngine:
    def __init__(self, fail_on_call=None):
        self.fail_on_call = set(fail_on_call or [])
        self.cancelled = False
        self.calls = 0
        self.denoise_strengths = []

    def upscale_image(self, image, scale=4, progress_cb=None, denoise_strength=0.5):
        self.calls += 1
        self.denoise_strengths.append(denoise_strength)
        if self.calls in self.fail_on_call:
            raise RuntimeError("synthetic failure")
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


class _TileProgressEngine:
    def __init__(self, tile_progress):
        self.tile_progress = list(tile_progress)
        self.cancelled = False

    def upscale_image(self, image, scale=4, progress_cb=None, denoise_strength=0.5):
        for current, total in self.tile_progress:
            if progress_cb is not None:
                progress_cb(ProgressInfo(current=current, total=total, message=f"tile {current}/{total}"))
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def _write_image(path: Path, size=(2, 2), color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_supported_extensions_are_expected():
    assert SUPPORTED_EXTENSIONS == {".jpg", ".jpeg", ".png", ".webp"}


def test_upscale_single_writes_correct_dimensions(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    _write_image(source, size=(2, 2))

    result = upscale_single(source, output, _NearestEngine(), scale=2)

    assert result.ok is True
    assert result.output_path == output
    assert result.error is None
    with Image.open(output) as image:
        assert image.size == (4, 4)
        assert image.mode == "RGB"


def test_upscale_single_captures_open_failure(tmp_path):
    missing = tmp_path / "missing.png"
    output = tmp_path / "output.png"

    result = upscale_single(missing, output, _NearestEngine(), scale=2)

    assert result.ok is False
    assert result.output_path is None
    assert result.error


def test_upscale_single_passes_denoise_strength(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    _write_image(source, size=(2, 2))
    engine = _NearestEngine()

    result = upscale_single(
        source,
        output,
        engine,
        scale=2,
        denoise_strength=0.25,
    )

    assert result.ok is True
    assert engine.denoise_strengths == [0.25]


def test_upscale_single_forwards_tile_progress(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    _write_image(source, size=(2, 2))
    engine = _TileProgressEngine([(1, 2), (2, 2)])
    reported = []

    result = upscale_single(
        source,
        output,
        engine,
        scale=2,
        progress_cb=lambda info: reported.append(info),
    )

    assert result.ok is True
    tile_reports = [info.message for info in reported if info.phase == "tile"]
    assert tile_reports == ["tile 1/2", "tile 2/2"]
    assert [info.phase for info in reported[:2]] == ["decode", "decode"]
    assert reported[-1].phase == "save"
    assert reported[-1].current == 0.99


def test_upscale_single_can_restore_original_resolution(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "output_restored.png"
    _write_image(source, size=(3, 2))

    result = upscale_single(
        source,
        output,
        _NearestEngine(),
        scale=4,
        resize_to_original=True,
    )

    assert result.ok is True
    with Image.open(output) as image:
        assert image.size == (3, 2)


def test_upscale_batch_filters_extensions_and_names_outputs(tmp_path):
    source_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    _write_image(source_dir / "a.png", size=(2, 2))
    _write_image(source_dir / "b.jpg", size=(2, 2))
    _write_image(source_dir / "sub" / "c.webp", size=(2, 2))
    (source_dir / "ignore.txt").write_text("not an image", encoding="utf-8")

    results = upscale_batch([source_dir], out_dir, _NearestEngine(), scale=2)

    assert len(results) == 3
    assert all(result.ok for result in results)
    assert (out_dir / "a_x2.png").is_file()
    assert (out_dir / "b_x2.jpg").is_file()
    assert (out_dir / "c_x2.webp").is_file()
    assert not (out_dir / "ignore_x2.txt").exists()
    with Image.open(out_dir / "a_x2.png") as image:
        assert image.size == (4, 4)


def test_upscale_batch_failure_does_not_abort_other_files(tmp_path):
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    _write_image(source_a, size=(2, 2))
    _write_image(source_b, size=(2, 2))
    out_dir = tmp_path / "output"

    engine = _NearestEngine(fail_on_call={2})
    results = upscale_batch([source_a, source_b], out_dir, engine, scale=2)

    assert len(results) == 2
    assert results[0].ok is True
    assert results[1].ok is False
    assert results[1].error == "synthetic failure"
    assert (out_dir / "a_x2.png").is_file()
    assert (out_dir / "b_x2.png").is_file() is False


def test_upscale_batch_reports_current_and_total(tmp_path):
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    _write_image(source_a, size=(2, 2))
    _write_image(source_b, size=(2, 2))
    reported = []

    upscale_batch(
        [source_a, source_b],
        tmp_path / "output",
        _NearestEngine(),
        scale=2,
        progress_cb=lambda info: reported.append(
            (info.current, info.total, info.phase)
        ),
    )

    assert [item for item in reported if item[2] == "image"] == [
        (1.0, 2, "image"),
        (2.0, 2, "image"),
    ]
    assert reported[0][2] == "decode"
    assert reported[-1][2] == "image"


def test_upscale_batch_maps_tile_progress_to_continuous_overall_progress(tmp_path):
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    _write_image(source_a, size=(2, 2))
    _write_image(source_b, size=(2, 2))
    engine = _TileProgressEngine([(1, 2), (2, 2)])
    reported = []

    upscale_batch(
        [source_a, source_b],
        tmp_path / "output",
        engine,
        scale=2,
        progress_cb=lambda info: reported.append(
            (info.current, info.total, info.phase, info.image_index, info.message)
        ),
    )

    tile_reports = [item for item in reported if item[2] == "tile"]
    assert len(tile_reports) == 4
    assert tile_reports[0][0] == pytest.approx(0.525)
    assert tile_reports[1][0] == pytest.approx(0.9)
    assert tile_reports[2][0] == pytest.approx(1.525)
    assert tile_reports[3][0] == pytest.approx(1.9)
    assert "第 1/2 张 · tile 1/2" in tile_reports[0][4]
    assert "第 2/2 张 · tile 2/2" in tile_reports[3][4]


def test_upscale_batch_cancel_flag_propagates_cancellation(tmp_path):
    source = tmp_path / "a.png"
    _write_image(source, size=(2, 2))

    with pytest.raises(SuperResCancelled):
        upscale_batch(
            [source],
            tmp_path / "output",
            _NearestEngine(),
            scale=2,
            cancel_flag=lambda: True,
        )

    assert not (tmp_path / "output" / "a_x2.png").exists()


def test_upscale_batch_skips_existing_output_by_default(tmp_path):
    source = tmp_path / "a.png"
    _write_image(source, size=(2, 2))
    out_dir = tmp_path / "output"
    engine = _NearestEngine()

    first = upscale_batch([source], out_dir, engine, scale=2)
    second = upscale_batch([source], out_dir, engine, scale=2)

    assert first[0].ok is True
    assert second[0].ok is True
    assert engine.calls == 1
    assert (out_dir / "a_x2.png").is_file()
