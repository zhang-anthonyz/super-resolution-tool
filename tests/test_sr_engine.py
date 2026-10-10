from pathlib import Path

import numpy as np
import pytest

from app.core.sr_engine import OnnxEngine, SuperResCancelled
from app.core.types import Backend, ModelMeta, Scene


class FakeSession:
    def __init__(self, scale=1):
        self.scale = scale
        self.feeds = []
        self.calls = 0
        self.fail_once_with_oom = False

    def run(self, output_names, feeds):
        self.calls += 1
        self.feeds.append(dict(feeds))
        if self.fail_once_with_oom:
            self.fail_once_with_oom = False
            raise RuntimeError("DML out of memory")
        tile = feeds["input"]
        return [
            np.repeat(
                np.repeat(tile, self.scale, axis=2),
                self.scale,
                axis=3,
            )
        ]


def make_model(
    scale=2,
    tile_size=256,
    scale_input=None,
    denoise_input=None,
) -> ModelMeta:
    return ModelMeta(
        name="fake",
        scale=scale,
        scene=Scene.REALISTIC,
        backend=Backend.DIRECTML,
        model_path=Path("fake.onnx"),
        input_name="input",
        output_name="output",
        scale_input=scale_input,
        denoise_input=denoise_input,
        tile_size=tile_size,
    )


def make_engine(fake_session, model):
    engine = OnnxEngine(Backend.DIRECTML)
    engine._session = fake_session
    engine._model = model
    engine._input_name = "input"
    engine._output_name = "output"
    engine._output_channels = 3
    return engine


def test_upscale_image_produces_expected_nearest_output():
    session = FakeSession(scale=2)
    model = make_model(scale=2)
    engine = make_engine(session, model)
    image = np.array(
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )

    output = engine.upscale_image(image, scale=2)
    expected = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

    assert output.shape == (4, 4, 3)
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output, expected)


def test_progress_callback_reports_every_tile():
    session = FakeSession(scale=2)
    model = make_model(scale=2, tile_size=3)
    engine = make_engine(session, model)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    reported = []

    engine.upscale_image(
        image,
        scale=2,
        progress_cb=lambda info: reported.append((info.current, info.total)),
    )

    assert len(reported) == 16
    assert [current for current, _ in reported] == list(range(1, 17))
    assert {total for _, total in reported} == {16}


def test_cancel_before_tile_raises_without_running_session():
    session = FakeSession(scale=2)
    model = make_model(scale=2)
    engine = make_engine(session, model)
    engine.cancel()

    with pytest.raises(SuperResCancelled):
        engine.upscale_image(np.zeros((2, 2, 3), dtype=np.uint8), scale=2)

    assert session.calls == 0


def test_scale_and_denoise_inputs_are_added_to_feeds():
    session = FakeSession(scale=2)
    model = make_model(scale=2, scale_input="scale", denoise_input="denoise")
    engine = make_engine(session, model)

    engine.upscale_image(np.zeros((2, 2, 3), dtype=np.uint8), scale=2)

    assert set(session.feeds[0]) == {"input", "scale", "denoise"}
    np.testing.assert_array_equal(session.feeds[0]["scale"], [2.0])
    np.testing.assert_array_equal(session.feeds[0]["denoise"], [0.0])


def test_oom_during_large_tile_falls_back_to_256():
    session = FakeSession(scale=2)
    session.fail_once_with_oom = True
    model = make_model(scale=2, tile_size=512)
    engine = make_engine(session, model)
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    output = engine.upscale_image(image, scale=2)

    assert output.shape == (600, 600, 3)
    assert session.calls == 5
    widths = [feed["input"].shape[3] for feed in session.feeds]
    assert widths == [300, 256, 54, 256, 54]
