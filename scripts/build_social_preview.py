#!/usr/bin/env python3
"""Build the version-neutral GitHub social preview.

From the repository root, after installing Pillow::

    python scripts/build_social_preview.py

The illustration is decorative, not an application screenshot. Upload the
result in the repository's Settings > General > Social preview to use it on
GitHub; committing the file does not change that setting.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
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


def main() -> None:
    width, height = 1280, 640
    image = Image.new("RGB", (width, height), "#0a2027")
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            glow = max(0.0, 1.0 - ((x - 1020) ** 2 + (y - 190) ** 2) ** 0.5 / 720)
            pixels[x, y] = (int(8 + 3 * glow), int(28 + 28 * glow), int(36 + 17 * glow))
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 40):
        draw.line((x, 0, x, height), fill="#102e35", width=1)
    for y in range(0, height, 40):
        draw.line((0, y, width, y), fill="#102e35", width=1)
    draw.rounded_rectangle((62, 56, 74, 68), radius=3, fill="#8de6ba")
    draw.text((90, 54), "P H A R M A M L   T O O L K I T", font=font(16, bold=True), fill="#a4dbc9")
    draw.text((60, 158), "From compound data", font=font(50, bold=True), fill="#f4fbf8")
    draw.text((60, 225), "to research insight.", font=font(50, bold=True), fill="#f4fbf8")
    draw.text((63, 315), "Machine learning / pharmacology / molecular science", font=font(18), fill="#bbd2cb")
    draw.line((64, 387, 116, 387), fill="#7ee0bc", width=4)
    draw.text((130, 377), "OPEN SOURCE  ·  OFFLINE FIRST", font=font(15, bold=True), fill="#96bdb0")
    draw.text((64, 560), "github.com/MrMMA2004/pharma-ml-toolkit", font=font(16), fill="#8caaa3")

    # A stylized molecule and data curve, not a product UI or a scientific result.
    draw.rounded_rectangle((760, 112, 1190, 500), radius=22, fill="#102d33", outline="#42665f", width=2)
    draw.text((787, 143), "A WORKFLOW FOR EXPLORATION", font=font(14, bold=True), fill="#9bbeb4")
    draw.line((788, 176, 1161, 176), fill="#2b4b4b", width=2)
    molecule = [(808, 252), (851, 227), (894, 252), (894, 302), (851, 327), (808, 302), (808, 252)]
    second = [(894, 252), (937, 227), (980, 252), (980, 302), (937, 327), (894, 302)]
    draw.line(molecule, fill="#8ed4c3", width=3, joint="curve")
    draw.line(second, fill="#8ed4c3", width=3, joint="curve")
    draw.line((851, 227, 851, 205), fill="#8ed4c3", width=3)
    draw.line((980, 252, 1002, 238), fill="#8ed4c3", width=3)
    for x, y in ((851, 205), (1002, 238), (937, 327)):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#8de6ba")
    draw.line((1018, 220, 1018, 357, 1156, 357), fill="#456560", width=2)
    draw.line([(1019, 338), (1035, 326), (1052, 296), (1069, 270), (1084, 264),
               (1099, 278), (1114, 303), (1130, 310), (1143, 291), (1156, 274)],
              fill="#9ce7af", width=4, joint="curve")
    for x, y in ((1084, 264), (1156, 274)):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#cef4bc")
    for x, label in ((788, "CHEMISTRY"), (910, "MODELS"), (1032, "PK / PD")):
        draw.rounded_rectangle((x, 430, x + 110, 460), radius=15, fill="#20453f")
        draw.text((x + 12, 438), label, font=font(12, bold=True), fill="#b6deca")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({width}x{height})")


if __name__ == "__main__":
    main()
