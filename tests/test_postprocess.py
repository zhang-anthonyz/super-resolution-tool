import numpy as np
from PIL import Image, ImageFilter

from app.core.postprocess import finishing_touch, guided_denoise


def test_finishing_touch_returns_uint8_same_shape():
    image = np.full((8, 8, 3), 128, dtype=np.uint8)

    result = finishing_touch(image, denoise_strength=0.4, sharpen_strength=0.3)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_finishing_touch_is_noop_when_strengths_are_zero():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

    result = finishing_touch(image)

    np.testing.assert_array_equal(result, image)


def _noisy_step_image(size: int = 96, noise: float = 18.0, seed: int = 3) -> np.ndarray:
    """Half dark / half light with a vertical edge, plus gaussian noise."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 60.0)
    image[:, size // 2 :] = 200.0
    image += rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _noise_std(image: np.ndarray, flat_columns: slice) -> float:
    """Std-dev of a supposedly flat region (noise level proxy)."""
    patch = image[16:80, flat_columns].astype(np.float64)
    return float(patch.std())


def _edge_contrast(image: np.ndarray, size: int = 96) -> float:
    """Mean jump across the step edge (detail retention proxy)."""
    left = image[16:80, size // 2 - 4 : size // 2].astype(np.float64).mean()
    right = image[16:80, size // 2 : size // 2 + 4].astype(np.float64).mean()
    return right - left


def test_guided_denoise_removes_noise():
    noisy = _noisy_step_image()
    before_noise = _noise_std(noisy, slice(8, 40))

    denoised = guided_denoise(noisy, 1.0)

    after_noise = _noise_std(denoised, slice(8, 40))

    assert after_noise < before_noise * 0.5, (before_noise, after_noise)


def test_guided_denoise_keeps_edges_better_than_a_gaussian_blur():
    """The whole point of the rewrite: same noise, less edge damage."""
    noisy = _noisy_step_image()

    guided = guided_denoise(noisy, 1.0)
    blurred = np.asarray(
        Image.fromarray(noisy, "RGB").filter(ImageFilter.GaussianBlur(radius=2.0))
    )

    guided_edge = _edge_contrast(guided)
    blurred_edge = _edge_contrast(blurred)
    original_edge = _edge_contrast(noisy)

    assert guided_edge > blurred_edge * 1.15, (guided_edge, blurred_edge)
    # 并且保住原始对比度的绝大部分（旧 gaussian 路径约为六成）
    assert guided_edge > original_edge * 0.75, (guided_edge, original_edge)


def test_guided_denoise_strength_is_monotonic():
    noisy = _noisy_step_image()

    light = _noise_std(guided_denoise(noisy, 0.25), slice(8, 40))
    heavy = _noise_std(guided_denoise(noisy, 1.0), slice(8, 40))

    assert heavy <= light


def test_guided_denoise_handles_bands_for_tall_images():
    """Band-wise processing must match single-pass results on a tall image."""
    from app.core import postprocess

    rng = np.random.default_rng(7)
    image = np.clip(
        np.full((600, 64, 3), 128.0) + rng.normal(0, 15.0, (600, 64, 3)), 0, 255
    ).astype(np.uint8)

    original = postprocess._BAND_PIXELS
    try:
        postprocess._BAND_PIXELS = 10_000_000  # 单带
        single = guided_denoise(image, 0.8)
        postprocess._BAND_PIXELS = 8_000  # 多条带（600x64 = 38400 像素）
        banded = guided_denoise(image, 0.8)
    finally:
        postprocess._BAND_PIXELS = original

    np.testing.assert_array_equal(single, banded)


def test_guided_denoise_zero_strength_returns_input():
    image = _noisy_step_image(size=32)

    result = guided_denoise(image, 0.0)

    np.testing.assert_array_equal(result, image)
