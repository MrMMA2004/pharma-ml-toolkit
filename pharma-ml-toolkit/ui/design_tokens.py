"""PharmaML Studio design tokens (part7 visual identity).

One module owns every color, typography, spacing and chart-palette
decision — the §6 rule "no duplicated hardcoded colors and no
independent styling definitions across individual pages".  The
identity: Deep Navy + Warm Ivory + Electric Lime (60/30/10), Inter
for the interface, JetBrains Mono for scientific/technical content.

Tokens are plain data (frozen dataclasses + module constants) so QSS
generation (studio theme provider), chart styling (ChartPalette)
and tests (WCAG contrast verification) all consume the SAME source.

Colors are 6-digit hex strings (QSS/pyqtgraph/matplotlib all read
them directly).  Any new visual decision belongs HERE, not inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# -- color tokens (part7 §2) ---------------------------------------------
@dataclass(frozen=True)
class ColorTokens:
    """The spec's token table for one theme (light or dark)."""

    name: str
    background: str          # neutral canvas
    surface: str             # cards, tables, dialogs
    primary: str             # navigation, structure
    accent: str              # interactive highlights (on dark/navy fills)
    accent_hover: str        # hovered primary actions
    selected_surface: str    # selected items, active states
    text: str                # primary text
    text_secondary: str      # supporting information
    border: str              # separators, boundaries
    disabled: str = ""       # disabled controls (dark: same as border)
    focus_ring: str = ""     # keyboard-focus indicator color; empty -> accent

    def hex_or(self, fallback: str) -> str:
        """A token or its stand-in when a theme has no value."""
        return self.disabled or fallback

    def focus_color(self) -> str:
        """The :focus indicator color (WCAG 1.4.11-safe per theme).

        Electric Lime is the identity accent and clears 3:1 against
        every dark-theme fill it sits on, but on light-theme white
        surfaces it lands at 1.40:1 — below the non-text UI-component
        floor.  Light theme therefore carries a dedicated olive-lime
        focus ring (5.6:1 on white, still the lime hue family);
        dark theme reuses the accent unchanged (8.1:1 on surface).
        """
        return self.focus_ring or self.accent


LIGHT = ColorTokens(
    name="light",
    background="#F3F5F1",      # Warm Ivory
    surface="#FFFFFF",         # White
    primary="#172F43",         # Deep Navy
    accent="#C6E84B",          # Electric Lime (fills with navy text)
    accent_hover="#ACD52F",    # Dark Lime
    selected_surface="#F0F6D9",  # Pale Lime
    text="#202B36",            # Charcoal
    text_secondary="#64748B",  # Slate
    border="#7B8B99",          # Steel Gray — 3.5:1 on white (WCAG 1.4.11)
    disabled="#7E8C99",        # Slate Gray — 3.44:1 on white
    focus_ring="#55701F",      # Olive Lime — 5.64:1 on white
)

DARK = ColorTokens(
    name="dark",
    background="#101B24",      # Midnight
    surface="#263D4D",         # Deep Blue Gray (elevated)
    primary="#203645",         # Slate Navy (panels/navigation)
    accent="#C6E84B",          # Electric Lime
    accent_hover="#ACD52F",    # Dark Lime
    selected_surface="#344C35",  # Forest Slate
    text="#EAF0F4",            # Soft White
    text_secondary="#A7B5C1",  # Cool Gray
    border="#344957",          # Muted Blue Gray
)

THEMES: dict[str, ColorTokens] = {"light": LIGHT, "dark": DARK}


def tokens(theme: str) -> ColorTokens:
    """The color tokens for ``theme`` (unknown → light)."""
    return THEMES.get(theme, LIGHT)


# -- typography tokens (part7 §3) -----------------------------------------
@dataclass(frozen=True)
class TypefaceTokens:
    """Interface + technical faces with their fallback chains."""

    ui_families: tuple[str, ...] = (
        "Inter", "Segoe UI", "Arial", "sans-serif")
    mono_families: tuple[str, ...] = (
        "JetBrains Mono", "Cascadia Code", "Consolas", "monospace")

    def ui_css(self) -> str:
        return ", ".join(self.ui_families)

    def mono_css(self) -> str:
        return ", ".join(self.mono_families)


TYPEFACES = TypefaceTokens()

# part7 §3.4 hierarchy (pt targets; Qt adapts to metrics/DPI)
@dataclass(frozen=True)
class TypeScale:
    main_title_pt: int = 24    # Bold 700
    card_title_pt: int = 15    # Semibold 600
    section_title_pt: int = 13  # Semibold 600
    body_pt: int = 10          # Regular 400
    table_pt: int = 10         # Regular 400
    metadata_pt: int = 9       # Regular 400
    code_pt: int = 9           # Regular 400 (mono)

    WEIGHT_BOLD = 700
    WEIGHT_SEMIBOLD = 600
    WEIGHT_MEDIUM = 500
    WEIGHT_REGULAR = 400


SCALE = TypeScale()


# -- spacing, borders, elevation (part7 §6) --------------------------------
@dataclass(frozen=True)
class LayoutTokens:
    """Spacing/radius/border rhythm shared by every component."""

    space_xs: int = 4
    space_sm: int = 8
    space_md: int = 12
    space_lg: int = 16
    space_xl: int = 24
    radius_sm: int = 4
    radius_md: int = 6
    radius_lg: int = 10
    border: int = 1
    card_min_width: int = 260
    card_min_height: int = 200


LAYOUT = LayoutTokens()


# -- scientific chart palette (part7 §5) -----------------------------------
# Okabe-Ito colorblind-conscious categorical palette — SEPARATE from
# the interface accent palette, and NOT defaulted to Electric Lime.
SCIENTIFIC_CATEGORICAL: tuple[str, ...] = (
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Purple
    "#D55E00",  # Vermilion
    "#56B4E9",  # Sky Blue
    "#F0E442",  # Yellow
    "#333333",  # Dark Gray
)

# Dark-theme variant: Okabe-Ito verbatim EXCEPT the Dark Gray, which
# vanishes on Midnight surfaces (1.4:1) — it becomes a light gray.
# The other seven hold up on dark surfaces (all ≥ 3:1 there).
SCIENTIFIC_CATEGORICAL_DARK: tuple[str, ...] = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#D55E00", "#56B4E9", "#F0E442", "#CCCCCC",
)

# §5 "do not communicate class identity through color alone": these
# colors fall below 3:1 against LIGHT surfaces, so series drawn in
# them MUST carry a non-color encoding (marker shape or dash pattern)
# in light theme.  Programmatic, not a style comment.
MARKER_DISCIPLINE_LIGHT: frozenset[str] = frozenset({
    "#E69F00",  # Orange — 2.1:1 on white
    "#CC79A7",  # Purple — 2.8:1 on white
    "#56B4E9",  # Sky Blue — 2.1:1 on white
    "#F0E442",  # Yellow — 1.1:1 on white
})


def scientific_categorical(theme: str) -> tuple[str, ...]:
    """The categorical series cycle for a theme."""
    return SCIENTIFIC_CATEGORICAL_DARK if theme == "dark" \
        else SCIENTIFIC_CATEGORICAL


def marker_discipline(theme: str) -> frozenset[str]:
    """Series colors that need a non-color encoding in this theme."""
    return frozenset() if theme == "dark" else MARKER_DISCIPLINE_LIGHT


def colors_are_distinct(palette: tuple[str, ...],
                        floor: float = 1.25) -> bool:
    """True when every pair of series colors differs meaningfully in
    luminance (contrast ≥ floor) OR in hue (≥ 40° apart) — 8 hues
    cannot all clear a luminance floor pairwise, so distinctness is
    delivered by hue with luminance as the tiebreaker."""
    import colorsys

    def hue_deg(c: str) -> float:
        r, g, b = (int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return colorsys.rgb_to_hsv(r, g, b)[0] * 360

    for i, a in enumerate(palette):
        for b in palette[i + 1:]:
            close = abs(hue_deg(a) - hue_deg(b))
            hue_gap = min(close, 360 - close)
            if contrast_ratio(a, b) < floor and hue_gap < 40:
                return False
    return True


# Perceptually ordered sequential maps (matplotlib names — resolved
# where they are consumed; listed here so the choice is centralized)
SEQUENTIAL_MAPS: tuple[str, ...] = ("viridis", "cividis", "plasma")
DIVERGING_MAPS: tuple[str, ...] = ("RdBu",)


# -- contrast verification (part7 §7) --------------------------------------
def _rel_luminance(hex_color: str) -> float:
    """WCAG relative luminance of a #rrggbb color."""
    def channel(v: int) -> float:
        c = v / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r = channel(int(hex_color[1:3], 16))
    g = channel(int(hex_color[3:5], 16))
    b = channel(int(hex_color[5:7], 16))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two #rrggbb colors."""
    l1 = _rel_luminance(foreground)
    l2 = _rel_luminance(background)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


__all__ = [
    "ColorTokens", "LIGHT", "DARK", "THEMES", "tokens",
    "TypefaceTokens", "TYPEFACES", "TypeScale", "SCALE",
    "LayoutTokens", "LAYOUT",
    "SCIENTIFIC_CATEGORICAL", "SEQUENTIAL_MAPS", "DIVERGING_MAPS",
    "contrast_ratio",
]
