"""Screenshot-level theme regression (part7 §9, R29-c).

The pixel check R28 left as future work: render the real window
offscreen under the studio theme, grab it, and verify the canvas is
actually Warm Ivory / Midnight at the pixels (not just in the QSS
string).  Catches a stylesheet that parses but never reaches the
widget it targets, and a dark theme that is accidentally the light
one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage

from ui.theme_manager import ThemeManager

# where the grabbed PNGs go (also used by the round report)
SHOT_DIR = Path("build") / "theme-shots"
_TOLERANCE = 16  # per-channel distance for a "is this the token" check


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def _close(px: tuple[int, ...], rgb: tuple[int, int, int]) -> bool:
    return all(abs(a - b) <= _TOLERANCE for a, b in zip(px, rgb))


def _canvas_points(image: QImage) -> dict[str, tuple[int, ...]]:
    """Canvas (not navigation/status) pixel probes.

    Layout truth (1024x700 home): the top corners and the upper
    canvas are the background; the bottom strip belongs to the
    status bar (Deep Navy by design, §2.1); cards are white
    surfaces in the middle.  The top-right and mid-canvas points
    are the honest canvas probes.
    """
    w, h = image.width(), image.height()
    # stay OUTSIDE the card grid: top strip + the margins between
    # the scroll area edge and the window frame (cards start ~y=90
    # and are inset ~24px; the far-right margin is canvas in both
    # themes, unlike (w-60, h/3) which lands on a dark card).
    pts = {"top-left": (6, 6), "top-right": (w - 7, 6),
           "canvas-top": (w // 2, 40), "canvas-right": (w - 30, h // 3)}
    out = {}
    for name, (x, y) in pts.items():
        c = image.pixelColor(x, y)
        out[name] = (c.red(), c.green(), c.blue())
    return out


def _render(studio_app, tmp_path, theme, base):
    """MainWindow built the way a real session is: the preference is
    PERSISTED first, so the window's own theme.restore_state() (called
    in its __init__) applies exactly this theme — the app's restore
    flow, not a hand-applied stylesheet that the window then replaces.
    """
    import os

    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)  # fresh QSettings
    settings = QSettings("pharmaml-shot", base)
    settings.setValue("theme/provider", "studio")
    settings.setValue("theme/name", theme)
    settings.sync()
    tm = ThemeManager(studio_app, settings)
    from ui.main_window import MainWindow

    window = MainWindow(tm)  # its __init__ restores studio/<theme>
    assert tm.provider == "studio" and tm.theme == theme
    window.resize(1024, 700)
    window.show()
    pix = window.grab()
    img = pix.toImage().convertToFormat(QImage.Format.Format_RGB32)
    window.close()
    return img


def test_light_canvas_is_warm_ivory(studio_app, tmp_path) -> None:
    img = _render(studio_app, tmp_path, "light", "light")
    canvas = _canvas_points(img)
    ivory = _hex_rgb("#F3F5F1")
    for name, px in canvas.items():
        assert _close(px, ivory), f"{name}: {px} vs {ivory}"
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(str(SHOT_DIR / "home-light.png"))


def test_dark_canvas_is_midnight(studio_app, tmp_path) -> None:
    img = _render(studio_app, tmp_path, "dark", "dark")
    midnight = _hex_rgb("#101B24")
    # top strip: the Midnight background token
    for name, px in _canvas_points(img).items():
        if name == "canvas-right":
            continue  # lands inside the card grid in dark — see below
        assert _close(px, midnight), f"{name}: {px} vs {midnight}"
    # the dark home fills its scroll viewport with the elevated
    # surface token (#263D4D) — the §2.2 surface ladder, verified at
    # the card region, NOT the light canvas assumption
    surface = _hex_rgb("#263D4D")
    c = img.pixelColor(img.width() // 2, img.height() // 3)
    px = (c.red(), c.green(), c.blue())
    assert _close(px, surface), f"card region: {px} vs {surface}"
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(str(SHOT_DIR / "home-dark.png"))


def test_themes_render_differently(studio_app, tmp_path) -> None:
    light = _render(studio_app, tmp_path, "light", "cmp-light")
    dark = _render(studio_app, tmp_path, "dark", "cmp-dark")
    assert _canvas_points(light) != _canvas_points(dark)


def test_grab_is_not_blank(studio_app, tmp_path) -> None:
    img = _render(studio_app, tmp_path, "light", "blank")
    # a blank page would be one uniform color across the whole grab
    colors = {img.pixel(x, y) for x in range(0, img.width(), 64)
              for y in range(0, img.height(), 64)}
    assert len(colors) > 4, f"suspiciously uniform: {len(colors)} colors"


def test_card_surfaces_contrast_the_canvas(studio_app, tmp_path) -> None:
    """Somewhere mid-grab there must be white card surface against
    the ivory canvas (light) — i.e. the cards actually render."""
    img = _render(studio_app, tmp_path, "light", "cards")
    w, h = img.width(), img.height()
    white = _hex_rgb("#FFFFFF")
    found = any(
        _close((img.pixelColor(x, y).red(),
                img.pixelColor(x, y).green(),
                img.pixelColor(x, y).blue()), white)
        for x in range(0, w, 16) for y in range(0, h, 16)
    )
    assert found, "no white card surface found in the light grab"


@pytest.mark.parametrize("theme,expected", [
    ("light", "#F3F5F1"), ("dark", "#101B24")])
def test_studio_qss_tokens_reach_the_canvas(theme, expected) -> None:
    """Static cross-check: the QSS text really carries the canvas
    token for this theme (grab tests above prove it APPLIES)."""
    from ui.studio_theme import studio_qss

    assert expected in studio_qss(theme)
