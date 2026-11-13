"""Application font loading (part7 §3: Inter + JetBrains Mono).

Bundles SIL-OFL fonts (redistribution permitted) and registers them
with Qt at startup; the fallback CHAINS still apply when a font is
missing, so nothing here is load-bearing for correctness.  §3.3:
"Do not assume that Inter or JetBrains Mono is already installed on
every user's computer."

``ui_font()``/``mono_font()`` are the single constructors pages and
widgets use — no page builds its own QFont with a hardcoded family.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase

from .design_tokens import SCALE, TYPEFACES

_FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"

# bundled, license-clean files (SIL OFL 1.1 — see fonts/*-OFL.txt)
_BUNDLED = (
    "Inter-Regular.ttf",
    "Inter-Medium.ttf",
    "Inter-SemiBold.ttf",
    "Inter-Bold.ttf",
    "JetBrainsMono-Regular.ttf",
)

_registered = False


def register_fonts() -> list[str]:
    """Register the bundled fonts with Qt (idempotent).

    Returns the preferred families whose bundle files registered in
    THIS call (or, on a repeat call, are already resolved) — an
    empty list means the bundle is absent and the fallback chains
    carry the UI, which callers must not treat as an error.
    """
    global _registered
    if _registered:
        return _families()
    _registered = True
    gained: set[str] = set()
    if _FONT_DIR.is_dir():
        for name in _BUNDLED:
            path = _FONT_DIR / name
            if path.is_file():
                font_id = QFontDatabase.addApplicationFont(str(path))
                if font_id < 0:
                    continue
                gained.update(QFontDatabase.applicationFontFamilies(font_id))
    if not gained:
        return []
    available = set(QFontDatabase.families())
    return [f for f in (*TYPEFACES.ui_families, *TYPEFACES.mono_families)
            if f in available]


def _families() -> list[str]:
    """The preferred families Qt can currently resolve."""
    available = set(QFontDatabase.families())
    return [f for f in (*TYPEFACES.ui_families, *TYPEFACES.mono_families)
            if f in available]


def has_inter() -> bool:
    return "Inter" in _families()


def has_mono() -> bool:
    return "JetBrains Mono" in _families()


def ui_font(point_size: int | None = None,
            weight: int = SCALE.WEIGHT_REGULAR) -> QFont:
    """The interface font with its full fallback chain."""
    font = QFont()
    font.setFamilies(list(TYPEFACES.ui_families))
    if point_size is not None:
        font.setPointSize(point_size)
    font.setWeight(_qt_weight(weight))
    return font


def mono_font(point_size: int | None = None,
              weight: int = SCALE.WEIGHT_REGULAR) -> QFont:
    """The technical font (SMILES, code, logs, identifiers)."""
    font = QFont()
    font.setFamilies(list(TYPEFACES.mono_families))
    if point_size is not None:
        font.setPointSize(point_size)
    font.setWeight(_qt_weight(weight))
    return font


def _qt_weight(token_weight: int) -> QFont.Weight:
    """Map the token weights (400/500/600/700) onto Qt's enum."""
    return {
        400: QFont.Weight.Normal,
        500: QFont.Weight.Medium,
        600: QFont.Weight.DemiBold,
        700: QFont.Weight.Bold,
    }.get(token_weight, QFont.Weight.Normal)


def set_mono(widget, point_size: int | None = None,
             weight: int = SCALE.WEIGHT_REGULAR) -> None:
    """Set a technical-content widget to the mono face (part7 §3.2).

    An EXPLICIT QFont beats the stylesheet, so SMILES readouts, log
    panes and identifier fields stay mono under every theme provider
    (qt-material and qdarkstyle ship no such rule).  Call on any
    QWidget; no-op for non-widget input.
    """
    if widget is None:
        return
    try:
        widget.setFont(mono_font(point_size, weight))
    except AttributeError:
        pass


__all__ = [
    "has_inter", "has_mono", "mono_font", "register_fonts", "set_mono",
    "ui_font",
]
