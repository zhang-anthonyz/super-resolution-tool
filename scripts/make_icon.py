# -*- coding: utf-8 -*-
"""Generate the application icon (assets/app.ico).

Design: STUDIO jade rounded-square badge + a "resolution ladder" — a faint
2x2 source-pixel cluster at the bottom-left steps up through a mid-tone
square to one large solid square at the top-right. The size progression *is*
the message: upscale. The master is drawn at 1024px and downscaled with
Lanczos so every smaller size stays crisp. Re-run after any tweak:
``python scripts/make_icon.py``.
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
#: Ladder opacity progression: source pixels → step → upscaled output.
SOURCE_ALPHA = 140   # ~55% white
STEP_ALPHA = 190     # ~75% white
OUTPUT_ALPHA = 255   # solid


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
    """Draw the resolution ladder onto the 1024px master."""
    draw = ImageDraw.Draw(master)
    scale = master.width / MASTER

    def sx(value: float) -> int:
        return int(round(value * scale))

    def square(x: int, y: int, size: int, alpha: int, radius: int) -> None:
        draw.rounded_rectangle(
            (x, y, x + size, y + size),
            radius=radius,
            fill=OFF_WHITE + (alpha,),
        )

    # 2x2 source-pixel cluster, bottom-left.
    cell = sx(84)
    gap = sx(24)
    origin = (sx(96), sx(1024 - 96 - 2 * cell - gap))
    for row in range(2):
        for col in range(2):
            x = origin[0] + col * (cell + gap)
            y = origin[1] + row * (cell + gap)
            square(x, y, cell, SOURCE_ALPHA, sx(18))

    # Mid step, centre of the diagonal.
    step = sx(230)
    square(sx(330), sx(330), step, STEP_ALPHA, sx(44))

    # Upscaled output, top-right — the biggest, fully solid.
    output = sx(400)
    square(sx(560), sx(64), output, OUTPUT_ALPHA, sx(76))


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
