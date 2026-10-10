from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.sr_engine import SuperResCancelled
from app.core.types import SuperResResult
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

    def upscale_image(self, image, scale=4, progress_cb=None):
        self.calls += 1
        if self.calls in self.fail_on_call:
            raise RuntimeError("synthetic failure")
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
        progress_cb=lambda info: reported.append((info.current, info.total)),
    )

    assert reported == [(1, 2), (2, 2)]


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
