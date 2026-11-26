#!/usr/bin/env python3
"""Build the version-neutral GitHub social preview.

From the repository root, after installing Pillow::

    python scripts/build_social_preview.py

The card is an illustration, not an application screenshot or a scientific
result. Upload the generated PNG under repository Settings > General > Social
preview if you have repository-admin access; committing it alone does not
change GitHub's social-preview setting.
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


def bezier(
    start: tuple[float, float],
    control_a: tuple[float, float],
    control_b: tuple[float, float],
    end: tuple[float, float],
    steps: int = 48,
) -> list[tuple[int, int]]:
    """Sample a cubic curve for the illustrative chart."""
    points = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        x = u**3 * start[0] + 3 * u**2 * t * control_a[0] + 3 * u * t**2 * control_b[0] + t**3 * end[0]
        y = u**3 * start[1] + 3 * u**2 * t * control_a[1] + 3 * u * t**2 * control_b[1] + t**3 * end[1]
        points.append((round(x), round(y)))
    return points


def main() -> None:
    width, height = 1280, 640
    image = Image.new("RGB", (width, height), "#0a2027")
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            glow = max(0.0, 1.0 - (((x - 1020) ** 2 + (y - 190) ** 2) ** 0.5) / 760)
            pixels[x, y] = (
                int(7 + 4 * glow),
                int(25 + 25 * glow),
                int(34 + 18 * glow),
            )

    draw = ImageDraw.Draw(image)
    for x in range(0, width, 36):
        draw.line((x, 0, x, height), fill="#16333a", width=1)
    for y in range(0, height, 36):
        draw.line((0, y, width, y), fill="#16333a", width=1)

    draw.rounded_rectangle((64, 55, 76, 67), radius=3, fill="#9de5c4")
    draw.text((92, 50), "P H A R M A M L   T O O L K I T", font=font(16, bold=True), fill="#a8dacb")
    draw.text((62, 146), "Science, from data", font=font(50, bold=True), fill="#f4fbf8")
    draw.text((62, 211), "to insight.", font=font(50, bold=True), fill="#f4fbf8")
    draw.text((65, 288), "Molecular science  /  machine learning  /  PK–PD", font=font(17), fill="#c2d7d2")
    draw.line((66, 350, 112, 350), fill="#9de5c4", width=4)
    draw.text((128, 339), "OPEN SOURCE  ·  LOCAL-FIRST RESEARCH", font=font(14, bold=True), fill="#a9c5bd")
    draw.text((65, 566), "github.com/MrMMA2004/pharma-ml-toolkit", font=font(15), fill="#88aaa4")

    # A stylized molecule and curve; the chart is explicitly illustrative.
    left, top, right, bottom = 744, 74, 1218, 516
    draw.rounded_rectangle((left, top, right, bottom), radius=20, fill="#112f38", outline="#477a70", width=2)
    draw.text((left + 24, top + 25), "A RESEARCH WORKFLOW", font=font(13, bold=True), fill="#b3d8cb")
    label = "ILLUSTRATIVE"
    label_box = draw.textbbox((0, 0), label, font=font(11))
    draw.text((right - 24 - (label_box[2] - label_box[0]), top + 27), label, font=font(11), fill="#8eafa8")
    draw.line((left + 24, top + 52, right - 24, top + 52), fill="#35545a", width=1)

    draw.text((left + 26, top + 78), "MOLECULAR DATA", font=font(11, bold=True), fill="#a3c6bd")
    draw.text((left + 245, top + 78), "MODEL & EVALUATE", font=font(11, bold=True), fill="#a3c6bd")
    atom = "#95ddc8"
    stroke = "#95ddc8"
    molecule_a = [(left + 44, top + 163), (left + 74, top + 145), (left + 104, top + 163), (left + 104, top + 198), (left + 74, top + 216), (left + 44, top + 198), (left + 44, top + 163)]
    molecule_b = [(left + 104, top + 163), (left + 134, top + 145), (left + 164, top + 163), (left + 164, top + 198), (left + 134, top + 216), (left + 104, top + 198)]
    draw.line(molecule_a, fill=stroke, width=3, joint="curve")
    draw.line(molecule_b, fill=stroke, width=3, joint="curve")
    draw.line((left + 74, top + 145, left + 74, top + 125), fill=stroke, width=3)
    draw.line((left + 164, top + 163, left + 183, top + 152), fill=stroke, width=3)
    draw.line((left + 134, top + 216, left + 146, top + 236), fill=stroke, width=3)
    for x, y in ((left + 74, top + 125), (left + 183, top + 152), (left + 146, top + 236)):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=atom)

    divider_x = left + 220
    draw.line((divider_x, top + 82, divider_x, top + 310), fill="#35545a", width=1)
    chart_left, chart_top = left + 246, top + 128
    chart_right, chart_bottom = right - 26, top + 268
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill="#78958e", width=1)
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill="#78958e", width=1)
    curve = bezier(
        (chart_left + 4, chart_bottom - 12),
        (chart_left + 34, chart_bottom - 18),
        (chart_left + 46, chart_top + 30),
        (chart_left + 77, chart_top + 31),
    )
    curve += bezier(
        (chart_left + 77, chart_top + 31),
        (chart_left + 105, chart_top + 31),
        (chart_left + 101, chart_bottom - 1),
        (chart_left + 130, chart_top + 41),
    )[1:]
    curve += bezier(
        (chart_left + 130, chart_top + 41),
        (chart_left + 157, chart_top + 68),
        (chart_left + 158, chart_top + 10),
        (chart_right - 4, chart_top + 8),
    )[1:]
    draw.line(curve, fill="#9de5c4", width=4, joint="curve")
    for x, y in (curve[36], curve[95], curve[-1]):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#c5f3d0")

    draw.line((left + 24, top + 328, right - 24, top + 328), fill="#35545a", width=1)
    for x, label in ((left + 25, "EXPLORE"), (left + 145, "MODEL"), (left + 255, "VALIDATE"), (left + 373, "REPORT")):
        draw.text((x, top + 356), label, font=font(11, bold=True), fill="#b7dcd0")
    draw.text((65, 600), "TOOLS FOR EXPLORATION, EVALUATION AND REVIEW", font=font(11, bold=True), fill="#88aaa4")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({width}x{height})")


if __name__ == "__main__":
    main()
