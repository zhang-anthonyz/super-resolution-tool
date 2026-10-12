import numpy as np
import pytest

from app.core.tiling import (
    blend_tile,
    feather_ramp,
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


def test_preprocess_supports_fp16_dtype():
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)

    tensor = preprocess_image(image, dtype=np.float16)

    assert tensor.dtype == np.float16
    assert tensor.shape == (1, 3, 1, 1)
    np.testing.assert_allclose(
        tensor[0, :, 0, 0],
        [0.0, 128.0 / 255.0, 1.0],
        rtol=1e-2,
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


def test_blend_tile_copies_when_there_is_no_neighbour():
    dst = np.full((4, 4, 3), 10, dtype=np.uint8)
    src = np.full((4, 4, 3), 200, dtype=np.uint8)

    blend_tile(dst, src)

    np.testing.assert_array_equal(dst, src)


def test_blend_tile_ramps_across_the_left_band():
    """Pixel 0 must stay with the neighbour, the band end must be the new tile."""
    dst = np.full((1, 12, 3), 50, dtype=np.uint8)
    src = np.full((1, 12, 3), 200, dtype=np.uint8)

    blend_tile(dst, src, band_x=8)

    row = dst[0, :, 0].astype(int)
    # Untouched beyond the band.
    assert row[8] == 200 and row[11] == 200
    # Monotonic cross-fade inside it, starting near the neighbour's value.
    assert row[0] <= 60
    assert all(b >= a for a, b in zip(row[:8], row[1:8]))
    assert row[7] >= 190


def test_blend_tile_ramps_vertically():
    dst = np.full((12, 1, 3), 50, dtype=np.uint8)
    src = np.full((12, 1, 3), 200, dtype=np.uint8)

    blend_tile(dst, src, band_y=8)

    column = dst[:, 0, 0].astype(int)
    assert column[0] <= 60
    assert all(b >= a for a, b in zip(column[:8], column[1:8]))
    assert column[8] == 200


def test_blend_tile_fades_both_bands_in_the_corner():
    dst = np.full((10, 10, 3), 40, dtype=np.uint8)
    src = np.full((10, 10, 3), 240, dtype=np.uint8)

    blend_tile(dst, src, band_x=6, band_y=6)

    corner = int(dst[0, 0, 0])           # inside both bands: neighbour wins
    top_edge = int(dst[0, 9, 0])         # inside the top band only
    left_edge = int(dst[9, 0, 0])        # inside the left band only
    interior = int(dst[9, 9, 0])         # outside both bands: the new tile

    assert corner < top_edge and corner < left_edge
    assert top_edge == left_edge
    assert interior == 240
    assert dst.dtype == np.uint8


def test_feather_ramp_is_monotonic_and_normalised():
    ramp = feather_ramp(10)

    assert ramp.dtype == np.float32
    assert ramp.shape == (10,)
    assert np.all(np.diff(ramp) > 0)
    assert 0.0 < ramp[0] < 0.1 and 0.9 < ramp[-1] < 1.0
    assert feather_ramp(0).shape == (0,)
