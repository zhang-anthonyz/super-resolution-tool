# -*- coding: utf-8 -*-
"""Generate the application icon (assets/app.ico).

Design: flat brand-jade rounded square with a bold white "SR-DX" wordmark —
the same mark shown in the sidebar's brand chip. The master is drawn at
1024px and downscaled with Lanczos so every smaller size stays crisp.
Re-run after any tweak: ``python scripts/make_icon.py``.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "assets" / "app.ico"

MASTER = 1024
#: Brand jade from app/ui/theme.py ACCENT; deep tone keeps white text legible.
BADGE_COLOR = (10, 107, 80)  # #0A6B50
BADGE_RADIUS = 230
TEXT = "SR-DX"
OFF_WHITE = (250, 250, 247)

#: First existing font wins; all ship with Windows.
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\segoeuib.ttf",   # Segoe UI Bold
    r"C:\Windows\Fonts\arialbd.ttf",    # Arial Bold
    r"C:\Windows\Fonts\arial.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, box: int) -> ImageFont.FreeTypeFont:
    """Largest font size whose text width fits inside ``box`` px."""
    size = box
    while size > 8:
        font = _load_font(size)
        if draw.textlength(TEXT, font=font) <= box:
            return font
        size -= 4
    return _load_font(8)


def main() -> None:
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    font = _fit_font(probe, box=int(MASTER * 0.72))

    img = Image.new("RGBA", (MASTER, MASTER), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, MASTER - 1, MASTER - 1), radius=BADGE_RADIUS, fill=BADGE_COLOR)

    left, top, right, bottom = draw.textbbox((0, 0), TEXT, font=font)
    draw.text(
        (
            (MASTER - (right - left)) / 2 - left,
            (MASTER - (bottom - top)) / 2 - top,
        ),
        TEXT,
        font=font,
        fill=OFF_WHITE,
    )

    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [img.resize((size, size), Image.LANCZOS) for size in sizes]
    frames[-1].save(
        TARGET,
        format="ICO",
        sizes=[(size, size) for size in sizes],
        append_images=frames[:-1],
    )
    print(f"wrote {TARGET} ({TARGET.stat().st_size} bytes), sizes={sizes}")


if __name__ == "__main__":
    main()
