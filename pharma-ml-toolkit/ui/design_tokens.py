"""PharmaML Studio design tokens (Mineral visual identity).

One module owns every color, typography, spacing and chart-palette
decision — the standing "no duplicated hardcoded colors and no
independent styling definitions across individual pages" rule, carried
from part7 into the Mineral design system ("A clearer path from
molecules to evidence").

The identity: cool neutral surfaces, Teal ``#076F73`` for primary
actions and focus, IBM Plex Sans for the interface, JetBrains Mono for
scientific/technical content.  Values are transcribed from the
"Design System" specification and its structured token export; the
status background tints and dark-theme action/pressed value are the
documented derived tokens (see docs/DESIGN_SYSTEM_MIGRATION.md).

The Design System's two ALTERNATIVE directions — 02/SPECTRAL (warm
paper, indigo actions, rounded panels) and 03/CATALYST (navy frame,
ivory surfaces, electric-lime selection, squared cards) — ship as
switchable studio themes alongside Mineral's light/dark.  §2 named them
alternatives and pinned Mineral as canonical; they remain available
for users who want them WITHOUT being the default (per the partX3
request), each meeting the same WCAG contract the Mineral themes are
pinned to.  All three directions share the §4.2 scientific data
palette ("Three directions. One scientific foundation.").

Tokens are plain data (frozen dataclasses + module constants) so QSS
generation (studio theme provider), chart styling (ChartPalette)
and tests (WCAG contrast verification) all consume the SAME source.

Colors are 6-digit hex strings (QSS/pyqtgraph/matplotlib all read
them directly).  Any new visual decision belongs HERE, not inline.
"""

from __future__ import annotations

from dataclasses import dataclass


# -- semantic color tokens (Mineral §4.1) ----------------------------------
@dataclass(frozen=True)
class ColorTokens:
    """The spec's semantic token table for one theme (light or dark)."""

    name: str
    background: str          # bg/canvas
    surface: str             # bg/surface (cards, tables, dialogs)
    primary: str             # action/primary (teal; dark: bright teal)
    accent: str              # action/primary for the QSS accent path
    accent_hover: str        # action/hover
    accent_pressed: str      # action/pressed
    on_primary: str          # action/on-primary (text on teal fills)
    selected_surface: str    # action/selected
    text: str                # text/primary
    text_secondary: str      # text/secondary
    text_muted: str          # text/muted
    border: str              # border/default
    border_subtle: str       # border/subtle
    elevated: str            # bg/elevated (dialogs, floating)
    subtle: str              # bg/subtle (inputs, hover fills)
    disabled: str            # state/disabled
    focus_ring: str          # focus/ring
    error: str               # status/error (foreground)
    error_bg: str            # status soft background
    warning: str             # status/warning (foreground)
    warning_bg: str          # status soft background
    success: str             # status/success (foreground)
    success_bg: str          # status soft background
    info: str                # status/info (foreground)
    info_bg: str             # status soft background
    shell: str = ""          # app-shell fills (top/status/nav); "" → background
    shell_text: str = ""     # text on the shell; "" → text

    def hex_or(self, fallback: str) -> str:
        """A token or its stand-in when a theme has no value."""
        return self.disabled or fallback

    def focus_color(self) -> str:
        """The :focus indicator color (focus/ring token)."""
        return self.focus_ring

    def shell_fill(self) -> str:
        """The app-shell (top bar / status / nav) background."""
        return self.shell or self.background

    def shell_foreground(self) -> str:
        """Text on the app shell."""
        return self.shell_text or self.text


LIGHT = ColorTokens(
    name="light",
    background="#F4F7F8",      # bg/canvas
    surface="#FFFFFF",         # bg/surface
    primary="#076F73",         # action/primary (Mineral teal)
    accent="#076F73",
    accent_hover="#05565B",    # action/hover
    accent_pressed="#06494E",  # action/pressed
    on_primary="#FFFFFF",      # action/on-primary
    selected_surface="#E3F3F1",  # action/selected
    text="#14232C",            # text/primary
    text_secondary="#4D6572",  # text/secondary
    text_muted="#697F8A",      # text/muted
    border="#CCD8DE",          # border/default
    border_subtle="#EAF0F2",   # border/subtle
    elevated="#FFFFFF",        # bg/elevated
    subtle="#EAF0F2",          # bg/subtle
    disabled="#EAF0F2",        # state/disabled
    focus_ring="#076F73",      # focus/ring
    error="#B33840",           # status/error
    error_bg="#FCECEE",        # status soft background
    warning="#805400",         # status/warning
    warning_bg="#FFF4DC",      # status soft background
    success="#246C48",         # status/success
    success_bg="#E8F4EC",      # status soft background
    info="#2468A0",            # status/info
    info_bg="#E9F2FA",         # status soft background
)

DARK = ColorTokens(
    name="dark",
    background="#0D1B23",      # bg/canvas
    surface="#14232C",         # bg/surface
    primary="#79D9CF",         # action/primary (bright teal)
    accent="#79D9CF",
    accent_hover="#A0E9DF",    # action/hover
    accent_pressed="#5EBFB4",  # action/pressed (derived, 4.5:1 on canvas)
    on_primary="#0D1B23",      # action/on-primary (dark text on bright teal)
    selected_surface="#06494E",  # action/selected
    text="#EAF0F2",            # text/primary
    text_secondary="#A9BCC6",  # text/secondary
    text_muted="#697F8A",      # text/muted
    border="#697F8A",          # border/default
    border_subtle="#263A45",   # border/subtle
    elevated="#263A45",        # bg/elevated
    subtle="#1B2C35",          # bg/subtle
    disabled="#263A45",        # state/disabled
    focus_ring="#79D9CF",      # focus/ring
    error="#FFADB1",           # status/error
    error_bg="#3A1F26",        # status soft background (derived tint)
    warning="#F3CB7A",         # status/warning
    warning_bg="#33290F",      # derived tint
    success="#80D6AC",         # status/success
    success_bg="#122B1D",      # derived tint
    info="#99CAEF",            # status/info
    info_bg="#12253A",         # derived tint
)


# -- 02 / SPECTRAL — warm paper, indigo actions, rounded panels -------------
# The Design System's alternative direction, transcribed from its preview
# card: "Warm paper, indigo actions, broader spacing and rounded panels. A
# more editorial, exploratory character."  Spec-pinned anchors: canvas
# #F5F2EC, action indigo #5048A5; the rest of the ramp is derived to the
# same WCAG contract as Mineral (see docs/DESIGN_SYSTEM_MIGRATION.md).
SPECTRAL = ColorTokens(
    name="spectral",
    background="#F5F2EC",      # warm paper canvas
    surface="#FFFFFF",         # white surfaces on paper
    primary="#5048A5",         # action/primary (indigo)
    accent="#5048A5",
    accent_hover="#403A8C",    # deeper indigo
    accent_pressed="#332E70",  # deepest indigo
    on_primary="#FFFFFF",
    selected_surface="#EAE8F6",  # indigo-tinted paper selection
    text="#2B2440",            # indigo-black ink
    text_secondary="#5C5470",
    text_muted="#857E97",
    border="#D8D2C4",          # warm border
    border_subtle="#EAE5D9",
    elevated="#FFFFFF",
    subtle="#EFEAE0",
    disabled="#EAE5D9",
    focus_ring="#5048A5",
    error="#A32833",
    error_bg="#F7E7E6",
    warning="#7A5300",
    warning_bg="#F7EBD4",
    success="#1F6845",
    success_bg="#E3EFE5",
    info="#235E9D",
    info_bg="#E5EEF8",
)

SPECTRAL_DARK = ColorTokens(
    name="spectral-dark",
    background="#151321",      # deep indigo-black canvas
    surface="#1E1B30",         # indigo-slate surface
    primary="#9D96E8",         # lavender indigo (bright)
    accent="#9D96E8",
    accent_hover="#B7B1F0",
    accent_pressed="#837CCB",
    on_primary="#151321",      # dark text on lavender
    selected_surface="#322D52",
    text="#ECEAF4",
    text_secondary="#B4AECB",
    text_muted="#837D9C",
    border="#6F688F",
    border_subtle="#2C2946",
    elevated="#2C2946",
    subtle="#241F3A",
    disabled="#2C2946",
    focus_ring="#9D96E8",
    error="#F2A9B0",
    error_bg="#3B2229",
    warning="#E9C583",
    warning_bg="#342B14",
    success="#85D4AE",
    success_bg="#16301F",
    info="#9CC8F0",
    info_bg="#152741",
)


# -- 03 / CATALYST — navy frame, ivory surfaces, lime selection -------------
# "Navy frame, ivory surfaces and electric lime selection. Squared cards,
# stronger contrast and familiar navigation."  Spec-pinned anchors: navy
# #172F43 (the FRAME — shell chrome, not the content canvas), ivory
# surfaces, lime #C6E84B for selection accents.  Actions on ivory stay a
# brighter navy so AA holds; lime carries selection/focus, and in the dark
# variant it becomes the action color (9.87:1 on navy).
CATALYST = ColorTokens(
    name="catalyst",
    background="#F3F5F1",      # ivory content canvas
    surface="#FFFFFF",         # ivory-white surfaces
    primary="#2E5F7A",         # brighter navy for actions on ivory
    accent="#2E5F7A",
    accent_hover="#1F4860",
    accent_pressed="#16374A",
    on_primary="#FFFFFF",
    selected_surface="#DDF0B8",  # lime-tinted ivory selection
    text="#16232E",
    text_secondary="#46586A",
    text_muted="#6E7F8E",
    border="#C9D2CC",
    border_subtle="#E4E9E3",
    elevated="#FFFFFF",
    subtle="#E9EDE7",
    disabled="#E4E9E3",
    focus_ring="#172F43",      # the navy frame color as focus ink
    shell="#172F43",           # the navy frame: top bar / status / nav
    shell_text="#E8EFEA",
    error="#A32833",
    error_bg="#F7E7E6",
    warning="#7A5300",
    warning_bg="#F7EBD4",
    success="#1F6845",
    success_bg="#E3EFE5",
    info="#235E9D",
    info_bg="#E5EEF8",
)

CATALYST_DARK = ColorTokens(
    name="catalyst-dark",
    background="#0E1D2B",      # deeper navy canvas
    surface="#172F43",         # navy surface (the frame color)
    primary="#C6E84B",         # electric lime action
    accent="#C6E84B",
    accent_hover="#D6F27A",
    accent_pressed="#A8CC33",
    on_primary="#0E1D2B",      # navy text on lime
    selected_surface="#2A4560",
    text="#E8EFEA",
    text_secondary="#A9BAC8",
    text_muted="#75889A",
    border="#5A7288",
    border_subtle="#223A50",
    elevated="#223A50",
    subtle="#1B3550",
    disabled="#223A50",
    focus_ring="#C6E84B",
    error="#F2A9B0",
    error_bg="#3B2229",
    warning="#E9C583",
    warning_bg="#342B14",
    success="#85D4AE",
    success_bg="#16301F",
    info="#9CC8F0",
    info_bg="#152741",
)

THEMES: dict[str, ColorTokens] = {
    "light": LIGHT, "dark": DARK,
    "spectral": SPECTRAL, "spectral-dark": SPECTRAL_DARK,
    "catalyst": CATALYST, "catalyst-dark": CATALYST_DARK,
}


def tokens(theme: str) -> ColorTokens:
    """The color tokens for ``theme`` (unknown → light)."""
    return THEMES.get(theme, LIGHT)


def is_dark_theme(theme: str) -> bool:
    """True when ``theme`` is one of the dark variants."""
    return theme in ("dark", "spectral-dark", "catalyst-dark")


# -- typography tokens (Mineral §5) -----------------------------------------
@dataclass(frozen=True)
class TypefaceTokens:
    """Interface + technical faces with their fallback chains."""

    ui_families: tuple[str, ...] = (
        "IBM Plex Sans", "Segoe UI", "Arial", "sans-serif")
    mono_families: tuple[str, ...] = (
        "JetBrains Mono", "Cascadia Code", "Consolas", "monospace")

    def ui_css(self) -> str:
        return ", ".join(self.ui_families)

    def mono_css(self) -> str:
        return ", ".join(self.mono_families)


TYPEFACES = TypefaceTokens()

# Mineral §5 size hierarchy (logical px at 100% scaling; pt targets =
# px × 72/96, Qt adapts to metrics/DPI).  Display 40 and PageTitle 26
# are reserved for the home hero and page headers; body stays at the
# dense scientific default (14 px = 10.5 pt).
@dataclass(frozen=True)
class TypeScale:
    display_pt: float = 30      # Display 40px (home hero only)
    page_title_pt: float = 19.5  # Page title 26px (workspace headers)
    section_title_pt: float = 13.5  # Section title 18px
    card_title_pt: float = 12   # Card title 16px
    body_pt: float = 10.5       # Body 14px
    label_pt: float = 9.75      # Label 13px
    table_pt: float = 9.75      # Table 13px
    metadata_pt: float = 9      # Metadata 12px
    code_pt: float = 9          # Technical 12px (mono)

    WEIGHT_BOLD = 700
    WEIGHT_SEMIBOLD = 600
    WEIGHT_MEDIUM = 500
    WEIGHT_REGULAR = 400


SCALE = TypeScale()


# -- spacing, borders, elevation (Mineral §6) --------------------------------
@dataclass(frozen=True)
class LayoutTokens:
    """Spacing/radius/border rhythm shared by every component."""

    space_xs: int = 4
    space_sm: int = 8
    space_md: int = 12
    space_lg: int = 16
    space_xl: int = 24
    radius_sm: int = 4          # default corner radius
    radius_md: int = 6          # buttons
    radius_lg: int = 8          # cards
    radius_xl: int = 12         # larger containers
    border: int = 1             # default border
    border_emph: int = 2        # emphasized border
    control_height_text: int = 28   # text-only controls
    control_height_compact: int = 32
    control_height_default: int = 36
    control_height_lg: int = 40
    table_row_compact: int = 28
    table_row_standard: int = 32
    table_row_comfortable: int = 40
    card_min_width: int = 260
    card_min_height: int = 200


LAYOUT = LayoutTokens()


# -- scientific visualization palette (Mineral §4.2) -------------------------
# A separate scientific data palette — NOT defaulted to the Mineral
# teal action color, and kept distinct from status colors.
SCIENTIFIC_CATEGORICAL: tuple[str, ...] = (
    "#0072B2",  # Blue
    "#D55E00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Purple
    "#E69F00",  # Yellow
    "#56B4E9",  # Sky
    "#444444",  # Ink
)

# Dark-theme variant from the structured token export: each series gets
# its lighter companion value (blue→69BDE8, orange→F49A6A, green→63D3AC,
# purple→E4A6CF, yellow→F0CC68, sky→8DD6AC, ink→E1E7EB).
SCIENTIFIC_CATEGORICAL_DARK: tuple[str, ...] = (
    "#69BDE8",  # Blue (dark)
    "#F49A6A",  # Orange (dark)
    "#63D3AC",  # Green (dark)
    "#E4A6CF",  # Purple (dark)
    "#F0CC68",  # Yellow (dark)
    "#8DD6AC",  # Sky (dark)
    "#E1E7EB",  # Ink (dark)
)

# "Do not communicate class identity through color alone": these light-
# theme series colors fall below 3:1 against white surfaces, so series
# drawn in them MUST carry a non-color encoding (marker shape or dash
# pattern).  Programmatic, not a style comment.
# Dark theme: the token-export companions mostly hold up, but green
# (#63D3AC) and sky (#8DD6AC) converge to near-identical mint tints
# (contrast 1.08, hue gap 14°) — pairwise-distinctness for that pair
# is delivered by the same non-color encoding, so both companions join
# the dark discipline set.
MARKER_DISCIPLINE_LIGHT: frozenset[str] = frozenset({
    "#E69F00",  # Yellow — 2.25:1 on white
    "#56B4E9",  # Sky — 2.31:1 on white
})
MARKER_DISCIPLINE_DARK: frozenset[str] = frozenset({
    "#63D3AC",  # Green companion — vs sky companion: 1.08:1, 14°
    "#8DD6AC",  # Sky companion — the pair needs non-color encoding
})


def scientific_categorical(theme: str) -> tuple[str, ...]:
    """The categorical series cycle for a theme."""
    return SCIENTIFIC_CATEGORICAL_DARK if is_dark_theme(theme) \
        else SCIENTIFIC_CATEGORICAL


def marker_discipline(theme: str) -> frozenset[str]:
    """Series colors that need a non-color encoding in this theme."""
    return MARKER_DISCIPLINE_DARK if is_dark_theme(theme) \
        else MARKER_DISCIPLINE_LIGHT


def colors_are_distinct(palette: tuple[str, ...],
                        floor: float = 1.25) -> bool:
    """True when every pair of series colors differs meaningfully in
    luminance (contrast ≥ floor) OR in hue (≥ 40° apart) — 7 hues
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


# -- contrast verification (standing WCAG guard) -----------------------------
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
    "CATALYST",
    "CATALYST_DARK",
    "DARK",
    "DIVERGING_MAPS",
    "LAYOUT",
    "LIGHT",
    "MARKER_DISCIPLINE_LIGHT",
    "SCALE",
    "SCIENTIFIC_CATEGORICAL",
    "SCIENTIFIC_CATEGORICAL_DARK",
    "SEQUENTIAL_MAPS",
    "SPECTRAL",
    "SPECTRAL_DARK",
    "THEMES",
    "TYPEFACES",
    "ColorTokens",
    "LayoutTokens",
    "TypefaceTokens",
    "TypeScale",
    "contrast_ratio",
    "is_dark_theme",
    "tokens",
]
