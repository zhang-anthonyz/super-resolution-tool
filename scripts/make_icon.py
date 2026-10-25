# -*- coding: utf-8 -*-
"""Generate the application icon (assets/app.ico).

Design: STUDIO jade rounded-square badge + a white magnifier lens over a 2x2
pixel grid — "look closer, resolve more". The master is drawn at 1024px and
downscaled with Lanczos so every smaller size stays crisp. Re-run after any
tweak: ``python scripts/make_icon.py``.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "assets" / "app.ico"

MASTER = 1024
#: Brand jades from app/ui/theme.py: ACCENT family.
GRADIENT_TOP = (10, 107, 80)      # #0A6B50
GRADIENT_BOTTOM = (63, 175, 138)  # #3BAF8A
BADGE_RADIUS = 230
OFF_WHITE = (250, 250, 247)
PIXEL_DIM = 150  # ~55% alpha white


def _badge(size: int) -> Image.Image:
    """Rounded-square badge with a vertical jade gradient."""
    rows = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    gradient = GRADIENT_TOP + rows * (np.array(GRADIENT_BOTTOM) - GRADIENT_TOP)
    img = Image.fromarray(gradient.astype(np.uint8).repeat(size, axis=1), "RGB")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=int(BADGE_RADIUS * size / MASTER), fill=255
    )
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def _mark(master: Image.Image) -> None:
    """Draw the magnifier + pixel grid onto the 1024px master."""
    draw = ImageDraw.Draw(master)
    scale = master.width / MASTER

    def sx(value: float) -> int:
        return int(round(value * scale))

    lens_center = (sx(430), sx(430))
    lens_radius = sx(285)
    ring = sx(66)
    handle_length = sx(160)
    handle_width = sx(96)

    # Handle first so the lens ring overlaps its joint cleanly.
    import math

    angle = math.radians(45.0)
    start = (
        lens_center[0] + int((lens_radius - ring * 0.4) * math.cos(angle)),
        lens_center[1] + int((lens_radius - ring * 0.4) * math.sin(angle)),
    )
    end = (
        start[0] + int(handle_length * math.cos(angle)),
        start[1] + int(handle_length * math.sin(angle)),
    )
    draw.line([start, end], fill=OFF_WHITE, width=handle_width)
    for point in (start, end):
        r = handle_width // 2
        draw.ellipse(
            (point[0] - r, point[1] - r, point[0] + r, point[1] + r),
            fill=OFF_WHITE,
        )

    # Lens ring.
    draw.ellipse(
        (
            lens_center[0] - lens_radius,
            lens_center[1] - lens_radius,
            lens_center[0] + lens_radius,
            lens_center[1] + lens_radius,
        ),
        outline=OFF_WHITE,
        width=ring,
    )

    # 2x2 pixel grid centred in the lens: two solid, two dimmed.
    cell = sx(170)
    gap = sx(36)
    corner_radius = sx(26)
    top_left = (
        lens_center[0] - cell - gap // 2,
        lens_center[1] - cell - gap // 2,
    )
    for row in range(2):
        for col in range(2):
            x0 = top_left[0] + col * (cell + gap)
            y0 = top_left[1] + row * (cell + gap)
            solid = (row + col) % 2 == 0
            draw.rounded_rectangle(
                (x0, y0, x0 + cell, y0 + cell),
                radius=corner_radius,
                fill=OFF_WHITE if solid else OFF_WHITE + (150,),
            )


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    master = _badge(MASTER)
    _mark(master)

    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [master.resize((size, size), Image.LANCZOS) for size in sizes]
    frames[-1].save(
        TARGET,
        format="ICO",
        sizes=[(size, size) for size in sizes],
        append_images=frames[:-1],
    )
    print(f"wrote {TARGET} ({TARGET.stat().st_size} bytes), sizes={sizes}")


if __name__ == "__main__":
    main()
