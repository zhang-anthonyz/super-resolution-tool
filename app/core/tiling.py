"""Pure NumPy helpers for ONNX image pre/post-processing and tile merging."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileRegion:
    """Input-space rectangle used for one inference tile."""

    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


def axis_starts(length: int, tile_size: int = 512, overlap: int = 10) -> list[int]:
    """Return tile start offsets that cover ``length`` with a small overlap.

    The final tile is clamped to the image edge and may therefore be smaller
    than ``tile_size``. This mirrors the boundary requirement in the wave-2
    task while still averaging pixels that appear in more than one tile.
    """
    if length <= 0:
        return []
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if length <= tile_size:
        return [0]

    stride = max(1, tile_size - max(0, overlap))
    # Keep the final start as-is; the corresponding region is clamped to
    # ``length`` by callers. This avoids a nearly duplicate final tile.
    return list(range(0, length, stride))


def tile_regions(
    height: int,
    width: int,
    tile_size: int = 512,
    overlap: int = 10,
) -> list[TileRegion]:
    """Return all input-space tile rectangles in row-major order."""
    y_starts = axis_starts(height, tile_size, overlap)
    x_starts = axis_starts(width, tile_size, overlap)
    regions: list[TileRegion] = []
    for y in y_starts:
        bottom = min(y + tile_size, height)
        for x in x_starts:
            right = min(x + tile_size, width)
            regions.append(TileRegion(top=y, left=x, bottom=bottom, right=right))
    return regions


def split_tiles(
    tensor: np.ndarray,
    tile_size: int = 512,
    overlap: int = 10,
) -> list[tuple[TileRegion, np.ndarray]]:
    """Slice an NCHW tensor into overlapping tiles."""
    if tensor.ndim != 4:
        raise ValueError("tensor must have shape NCHW")

    _, _, height, width = tensor.shape
    tiles: list[tuple[TileRegion, np.ndarray]] = []
    for region in tile_regions(height, width, tile_size, overlap):
        tile = tensor[
            :,
            :,
            region.top : region.bottom,
            region.left : region.right,
        ]
        tiles.append((region, np.ascontiguousarray(tile)))
    return tiles


def merge_tiles(
    tiles: list[tuple[TileRegion, np.ndarray]],
    output_height: int,
    output_width: int,
    scale: int,
) -> np.ndarray:
    """Average overlapping output tiles into an NCHW output tensor."""
    if not tiles:
        raise ValueError("tiles must not be empty")

    sample = tiles[0][1]
    channels = sample.shape[1]
    accumulator = np.zeros(
        (1, channels, output_height, output_width),
        dtype=np.float32,
    )
    weights = np.zeros((1, 1, output_height, output_width), dtype=np.float32)

    for region, tile in tiles:
        out_top = region.top * scale
        out_left = region.left * scale
        out_bottom = region.bottom * scale
        out_right = region.right * scale

        expected_h = region.height * scale
        expected_w = region.width * scale
        if tile.shape[2:] != (expected_h, expected_w):
            raise ValueError(
                "tile output shape does not match the scaled input region: "
                f"{tile.shape[2:]} != {(expected_h, expected_w)}"
            )

        accumulator[
            :,
            :,
            out_top:out_bottom,
            out_left:out_right,
        ] += tile
        weights[
            :,
            :,
            out_top:out_bottom,
            out_left:out_right,
        ] += 1.0

    return accumulator / weights


def preprocess_image(
    image: np.ndarray,
    channel_order: str = "RGB",
    normalize: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Convert an RGB HWC uint8 image to normalized NCHW float32."""
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError("image must have shape HWC")

    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

    if channel_order.upper() == "BGR" and arr.shape[-1] == 3:
        arr = arr[..., ::-1]

    mean, std = normalize
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(chw[None, ...], dtype=np.float32)


def postprocess_tensor(tensor: np.ndarray) -> np.ndarray:
    """Convert a normalized CHW/NCHW float32 tensor to RGB HWC uint8."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("only a single batch element can be post-processed")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError("tensor must have shape CHW")

    arr = np.clip(tensor, 0.0, 1.0) * 255.0
    arr = np.rint(arr).astype(np.uint8)
    return np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))
