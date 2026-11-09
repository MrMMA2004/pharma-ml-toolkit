#!/usr/bin/env python3
"""Build the GitHub social preview from the tracked, real web-console capture.

Run from the repository root after installing Pillow::

    python scripts/build_social_preview.py

The source screenshot is not synthesized or relabelled: it is the verified
v1.2.159 browser capture copied to ``pharma-ml-toolkit/docs/assets/``.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pharma-ml-toolkit/docs/assets/web-console-v1.2.159.png"
OUTPUT = ROOT / "pharma-ml-toolkit/docs/assets/github-social-preview.png"
ROBOTO = Path(os.environ.get("PHARMAML_ROBOTO_DIR", "/usr/share/fonts/truetype/roboto"))
DEJAVU = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Prefer Roboto, with a portable DejaVu fallback."""
    candidates = [
        ROBOTO / ("Roboto-Bold.ttf" if bold else "Roboto-Regular.ttf"),
        DEJAVU / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def rounded(image: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *image.size), radius=radius, fill=255)
    result = image.convert("RGBA")
    result.putalpha(mask)
    return result


def main() -> None:
    width, height = 1280, 640
    image = Image.new("RGB", (width, height), "#071b1d")
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            glow = max(0.0, 1.0 - ((x - 1060) ** 2 + (y - 60) ** 2) ** 0.5 / 900)
            pixels[x, y] = (
                int(7 + 5 * glow),
                int(27 + 24 * glow),
                int(29 + 22 * glow),
            )

    draw = ImageDraw.Draw(image)
    mint = "#45dfb2"
    pale = "#d7f5e9"
    muted = "#9abbb5"

    # Compact brand mark and wordmark.
    draw.rounded_rectangle((58, 49, 102, 93), radius=12, fill="#103f3b", outline="#267566", width=2)
    draw.line((75, 59, 85, 59, 85, 70, 94, 82, 88, 86, 72, 86, 66, 82, 75, 70, 75, 59), fill=mint, width=3)
    draw.line((72, 77, 88, 77), fill=mint, width=3)
    draw.text((118, 54), "PHARMAML TOOLKIT", font=font(25, bold=True), fill=pale)

    draw.rounded_rectangle((58, 122, 268, 156), radius=17, fill="#123e39")
    draw.text((76, 130), "OPEN SOURCE · OFFLINE FIRST", font=font(14, bold=True), fill=mint)

    draw.text((56, 183), "Pharmaceutical ML,", font=font(51, bold=True), fill="#f4faf8")
    draw.text((56, 246), "offline by default.", font=font(51, bold=True), fill="#f4faf8")
    draw.text((60, 329), "From compound tables to reproducible", font=font(23), fill=muted)
    draw.text((60, 362), "models, simulations and scientific reports.", font=font(23), fill=muted)

    draw.text((60, 429), "QSAR  ·  PK/PD  ·  DOCKING  ·  XAI", font=font(20, bold=True), fill=mint)

    for x, label, w in ((60, "PYTHON 3.10+", 158), (230, "17 STUDIO PAGES", 188), (430, "MIT", 70)):
        draw.rounded_rectangle((x, 522, x + w, 562), radius=12, fill="#102f30", outline="#28514c", width=1)
        draw.text((x + 14, 533), label, font=font(14, bold=True), fill=pale)
    draw.text((60, 590), "github.com/MrMMA2004/pharma-ml-toolkit", font=font(16), fill=muted)

    # Real application capture, framed as a browser card with a soft shadow.
    capture = Image.open(SOURCE).convert("RGB").resize((640, 400), Image.Resampling.LANCZOS)
    capture = rounded(capture, 20)
    shadow = Image.new("RGBA", (700, 460), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((20, 20, 680, 440), radius=24, fill=(0, 0, 0, 150))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    image.paste(shadow, (552, 95), shadow)
    image.paste(capture, (590, 112), capture)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((590, 112, 1230, 512), radius=20, outline="#3b6c63", width=2)
    draw.text((602, 532), "THE NEXT.JS EXPLORER · ACTUAL APPLICATION CAPTURE", font=font(14, bold=True), fill=muted)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({width}x{height})")


if __name__ == "__main__":
    main()
