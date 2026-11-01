"""Shared image-loading helpers (EXIF orientation and transparency).

Both the processing pipeline and the UI read the same files, and they have to
agree on what the picture *looks like*: a phone photo stores its pixels
sideways plus an orientation tag, so a loader that ignores the tag shows the
image lying on its side. The pipeline bakes the orientation into the pixels
it writes, so every reader must apply the same transform to stay consistent.
"""
from __future__ import annotations

from typing import Optional

from PIL import Image, ImageOps

#: EXIF tag id of the orientation field (1 = upright, 3/6/8 = rotated).
ORIENTATION_TAG = 0x0112


def exif_orientation(image: Image.Image) -> int:
    """Return the EXIF orientation tag (1-8), or 1 when absent or unreadable."""
    try:
        return int(image.getexif().get(ORIENTATION_TAG, 1) or 1)
    except Exception:  # noqa: BLE001 - a broken EXIF block must not stop the job
        return 1


def orient_source(source: Image.Image) -> tuple[Image.Image, Optional[bytes]]:
    """Bake the EXIF orientation into the pixels.

    Returns the oriented image and the EXIF block to carry over. The
    orientation tag itself is dropped (the pixels are upright now), while the
    rest of the metadata survives.
    """
    if exif_orientation(source) == 1:
        # Nothing to transpose. ``exif_transpose`` would still copy the whole
        # image, which doubles peak memory on large photos.
        return source, source.info.get("exif")
    oriented = ImageOps.exif_transpose(source)
    return oriented, oriented.info.get("exif")


def extract_alpha(image: Image.Image) -> Optional[Image.Image]:
    """Return the alpha channel as an ``L`` image, or None when fully opaque.

    Transparency has to travel alongside the colour channels: dropping it
    turns a logo's background into opaque black.
    """
    if image.mode in ("RGBA", "LA", "PA"):
        alpha = image.getchannel("A")
    elif image.mode == "P" and "transparency" in image.info:
        alpha = image.convert("RGBA").getchannel("A")
    else:
        return None
    try:
        if alpha.getextrema()[0] >= 255:
            return None          # fully opaque: nothing worth carrying
    except ValueError:
        return None
    return alpha


def composite_over_white(image: Image.Image) -> Image.Image:
    """Flatten transparency onto a white background.

    Used for display: dropping the alpha channel instead would show the raw
    colour values stored underneath the transparent area, which is not what
    the picture looks like.
    """
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")
