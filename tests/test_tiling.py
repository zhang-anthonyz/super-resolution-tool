import numpy as np
import pytest

from app.core.tiling import (
    merge_tiles,
    postprocess_tensor,
    preprocess_image,
    split_tiles,
    tile_regions,
)


def test_preprocess_uint8_hwc_to_normalized_nchw():
    image = np.array(
        [
            [[0, 64, 128]],
            [[255, 128, 64]],
        ],
        dtype=np.uint8,
    )

    tensor = preprocess_image(image)

    assert tensor.shape == (1, 3, 2, 1)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(
        tensor[0, :, 0, 0],
        [0.0, 64.0 / 255.0, 128.0 / 255.0],
    )


def test_postprocess_chw_to_uint8_hwc():
    tensor = np.array([[[0.0]], [[0.5]], [[1.0]]], dtype=np.float32)

    image = postprocess_tensor(tensor)

    assert image.shape == (1, 1, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [0, 128, 255]


def test_tile_regions_stay_inside_image_and_clamp_final_edge():
    regions = tile_regions(height=5, width=5, tile_size=4, overlap=1)

    assert len(regions) == 4
    for region in regions:
        assert 0 <= region.top < region.bottom <= 5
        assert 0 <= region.left < region.right <= 5

    assert (regions[-1].bottom, regions[-1].right) == (5, 5)


def test_split_and_merge_reconstructs_original_tensor():
    tensor = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)

    tiles = split_tiles(tensor, tile_size=4, overlap=1)
    merged = merge_tiles(tiles, output_height=5, output_width=5, scale=1)

    assert merged.shape == (1, 1, 5, 5)
    np.testing.assert_array_equal(merged[0, 0], tensor[0, 0])


def test_merge_tiles_validates_scaled_tile_shape():
    region = tile_regions(height=2, width=2, tile_size=2, overlap=0)[0]
    bad_tile = np.zeros((1, 1, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="does not match"):
        merge_tiles([(region, bad_tile)], output_height=4, output_width=4, scale=2)
