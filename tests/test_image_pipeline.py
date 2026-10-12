from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.sr_engine import SuperResCancelled
from app.core.types import ProgressInfo, SuperResResult
from app.pipelines.image_pipeline import (
    COMPRESSION_PROFILE_COMPRESSED,
    COMPRESSION_PROFILE_HIGH,
    COMPRESSION_PROFILE_LOSSLESS,
    MAX_TILE_LOG_LINES,
    SUPPORTED_EXTENSIONS,
    percentage_size,
    resize_summary,
    _compression_options,
    _save_image,
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


class _ManyTileEngine:
    """Reports a configurable number of tiles to exercise tile-log throttling."""

    def __init__(self, total: int = 200):
        self.total = total
        self.cancelled = False

    def upscale_image(self, image, scale=4, progress_cb=None, denoise_strength=0.5):
        for current in range(1, self.total + 1):
            if progress_cb is not None:
                progress_cb(
                    ProgressInfo(
                        current=current,
                        total=self.total,
                        message=f"tile {current}/{self.total} · 20ms/块 · 剩余 2s",
                    )
                )
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def test_png_profile_does_not_request_pillow_optimize():
    """Pillow's PNG `optimize` runs a per-row filter search: measured 14.9s vs
    1.3s for a 4096x4096 output, for a ~14% smaller file."""
    for profile in (
        COMPRESSION_PROFILE_HIGH,
        COMPRESSION_PROFILE_COMPRESSED,
        COMPRESSION_PROFILE_LOSSLESS,
    ):
        options = _compression_options("PNG", profile)
        assert "optimize" not in options
        assert options["compress_level"] in {6, 9}


def test_png_compressed_profile_uses_the_highest_level():
    assert _compression_options("PNG", COMPRESSION_PROFILE_COMPRESSED) == {
        "compress_level": 9
    }
    assert _compression_options("PNG", COMPRESSION_PROFILE_HIGH) == {
        "compress_level": 6
    }


def test_webp_stays_on_a_fast_encoder_method():
    for profile in (
        COMPRESSION_PROFILE_HIGH,
        COMPRESSION_PROFILE_COMPRESSED,
        COMPRESSION_PROFILE_LOSSLESS,
    ):
        assert _compression_options("WEBP", profile)["method"] <= 4


def test_saved_png_round_trips_pixel_exactly(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    out = tmp_path / "out.png"

    _save_image(image, out, compression_profile=COMPRESSION_PROFILE_COMPRESSED)

    with Image.open(out) as saved:
        assert saved.size == (32, 32)
        np.testing.assert_array_equal(np.asarray(saved.convert("RGB")), image)


def test_saved_outputs_load_for_every_format(tmp_path):
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    for name in ("out.jpg", "out.webp", "out.png"):
        _save_image(image, tmp_path / name, compression_profile=COMPRESSION_PROFILE_HIGH)
        with Image.open(tmp_path / name) as saved:
            assert saved.size == (16, 16)


def test_step_log_reaches_the_sink_with_step_names(tmp_path):
    source = tmp_path / "input.png"
    _write_image(source, size=(4, 4))
    lines = []

    result = upscale_single(
        source,
        tmp_path / "out.png",
        _NearestEngine(),
        scale=2,
        denoise_strength=0.0,
        log_sink=lambda level, message: lines.append((level, message)),
    )

    assert result.ok is True
    text = "\n".join(message for _, message in lines)
    assert "解码" in text
    assert "设置 倍率" in text
    assert "推理完成" in text
    assert "编码 PNG" in text
    assert "完成" in text
    assert {level for level, _ in lines} <= {"Info", "Warning", "Error", "Success"}


def test_tile_log_lines_are_throttled(tmp_path):
    source = tmp_path / "input.png"
    _write_image(source, size=(4, 4))
    lines = []

    upscale_single(
        source,
        tmp_path / "out.png",
        _ManyTileEngine(total=200),
        scale=2,
        log_sink=lambda level, message: lines.append(message),
    )

    tile_lines = [line for line in lines if line.startswith("tile ")]
    assert len(tile_lines) <= MAX_TILE_LOG_LINES
    assert tile_lines[-1].startswith("tile 200/200")


def test_batch_reports_a_summary_and_skips_existing_outputs(tmp_path):
    source_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    _write_image(source_dir / "a.png", size=(2, 2))
    _write_image(source_dir / "b.png", size=(2, 2))
    lines = []

    upscale_batch(
        [source_dir],
        out_dir,
        _NearestEngine(),
        scale=2,
        log_sink=lambda level, message: lines.append((level, message)),
    )
    upscale_batch(
        [source_dir],
        out_dir,
        _NearestEngine(),
        scale=2,
        log_sink=lambda level, message: lines.append((level, message)),
    )

    text = "\n".join(message for _, message in lines)
    assert "批量任务开始 · 2 张图片" in text
    assert "跳过（已存在）" in text
    assert any(level == "Success" and "批量完成" in msg for level, msg in lines)


def test_failed_image_logs_the_error_and_keeps_the_batch_going(tmp_path):
    source = tmp_path / "a.png"
    _write_image(source, size=(2, 2))
    lines = []

    results = upscale_batch(
        [source],
        tmp_path / "output",
        _NearestEngine(fail_on_call={1}),
        scale=2,
        log_sink=lambda level, message: lines.append((level, message)),
    )

    assert results[0].ok is False
    assert any(
        level == "Error" and "synthetic failure" in message for level, message in lines
    )


def test_percentage_size_keeps_the_aspect_ratio():
    assert percentage_size((4096, 4096), 50) == (2048, 2048)
    assert percentage_size((4032, 3024), 62.5) == (2520, 1890)
    assert percentage_size((1000, 500), 200) == (2000, 1000)
    # A very small factor still yields a valid (1px) image.
    assert percentage_size((4, 4), 5) == (1, 1)


def test_resize_summary_describes_each_mode():
    assert resize_summary(False, None) == "保持超分尺寸"
    assert resize_summary(True, None) == "回缩到原图尺寸"
    assert resize_summary(False, 50) == "按超分结果的 50% 等比例缩放"
    # 100% is the same as keeping the super-resolved size.
    assert resize_summary(False, 100) == "保持超分尺寸"


def test_upscale_single_scales_the_super_resolved_output(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "out.png"
    _write_image(source, size=(5, 4))

    result = upscale_single(
        source, output, _NearestEngine(), scale=4, resize_percent=50
    )

    assert result.ok is True
    with Image.open(output) as image:
        # 4x of 5x4 is 20x16, half of that is 10x8.
        assert image.size == (10, 8)


def test_percentage_resize_overshoots_by_choice(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "out.png"
    _write_image(source, size=(3, 2))

    result = upscale_single(
        source, output, _NearestEngine(), scale=2, resize_percent=200
    )

    assert result.ok is True
    with Image.open(output) as image:
        assert image.size == (12, 8)  # 6x4 super-resolved, doubled


def test_percentage_of_100_resizes_nothing(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "out.png"
    _write_image(source, size=(3, 2))

    upscale_single(source, output, _NearestEngine(), scale=2, resize_percent=100)

    with Image.open(output) as image:
        assert image.size == (6, 4)

    lines = []
    upscale_single(
        source,
        tmp_path / "out2.png",
        _NearestEngine(),
        scale=2,
        resize_percent=100,
        log_sink=lambda level, message: lines.append(message),
    )
    assert not any("缩放 →" in line for line in lines)


def test_percentage_resize_logs_the_resulting_size(tmp_path):
    source = tmp_path / "input.png"
    _write_image(source, size=(4, 4))
    lines = []

    upscale_single(
        source,
        tmp_path / "out.png",
        _NearestEngine(),
        scale=2,
        resize_percent=25,
        log_sink=lambda level, message: lines.append(message),
    )

    text = "\n".join(lines)
    assert "输出尺寸 按超分结果的 25% 等比例缩放" in text
    assert "按超分结果的 25% 缩放 → 2x2" in text


def test_batch_names_percentage_outputs_with_the_percent(tmp_path):
    source_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    _write_image(source_dir / "a.png", size=(4, 4))

    results = upscale_batch(
        [source_dir], out_dir, _NearestEngine(), scale=2, resize_percent=50
    )

    assert len(results) == 1
    assert results[0].output_path == out_dir / "a_x2_p50.png"
    assert (out_dir / "a_x2_p50.png").is_file()


def test_batch_percentage_of_100_keeps_the_plain_name(tmp_path):
    source_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    _write_image(source_dir / "a.png", size=(4, 4))

    upscale_batch(
        [source_dir], out_dir, _NearestEngine(), scale=2, resize_percent=100
    )

    assert (out_dir / "a_x2.png").is_file()
