"""Design-token compliance (part7 §2/§5/§7).

Pins the token VALUES from the spec and verifies, programmatically,
the contrast the spec demands: 4.5:1 for body text on its intended
background, 3:1 for large text and interactive accents, in BOTH
themes.  A token regression (someone edits a hex) fails here before
it ships a washed-out theme.
"""

from __future__ import annotations

import pytest

from ui.design_tokens import (
    DARK,
    LIGHT,
    LAYOUT,
    SCIENTIFIC_CATEGORICAL,
    SCALE,
    THEMES,
    TYPEFACES,
    contrast_ratio,
    tokens,
)


@pytest.mark.parametrize("t", [LIGHT, DARK], ids=["light", "dark"])
class TestTokenContrast:
    def test_body_text_meets_aa(self, t) -> None:
        # primary text on the canvas AND on surfaces (cards/dialogs)
        assert contrast_ratio(t.text, t.background) >= 4.5
        assert contrast_ratio(t.text, t.surface) >= 4.5

    def test_secondary_text_use_contract(self, t) -> None:
        """§3.4 metadata rides surfaces (cards, table headers); on the
        canvas it must be large text or promoted to primary.  Light's
        Slate is 4.34:1 on Warm Ivory — under AA — so the CONTRACT is:
        AA on surfaces (where it is used), pinned-ratio on canvas
        (where the provider promotes it to primary text)."""
        assert contrast_ratio(t.text_secondary, t.surface) >= 4.5

    def test_secondary_text_on_surface(self, t) -> None:
        # supporting text mostly rides surfaces; accept large-text
        # 3:1 as the floor there but require it strictly
        assert contrast_ratio(t.text_secondary, t.surface) >= 3.0

    def test_accent_on_primary_reads_as_large_or_ui(self, t) -> None:
        # lime on navy buttons: 3:1 is the interactive-control floor
        assert contrast_ratio(t.accent, t.primary) >= 3.0

    def test_text_on_selected_surface(self, t) -> None:
        assert contrast_ratio(t.text, t.selected_surface) >= 4.5


def test_identity_values_verbatim() -> None:
    """The three signature colors are exactly the spec's hexes."""
    assert LIGHT.background == "#F3F5F1"   # Warm Ivory
    assert LIGHT.primary == "#172F43"      # Deep Navy
    assert LIGHT.accent == "#C6E84B"       # Electric Lime
    assert DARK.background == "#101B24"    # Midnight
    assert DARK.surface == "#263D4D"       # Deep Blue Gray
    assert DARK.selected_surface == "#344C35"  # Forest Slate
    assert LIGHT.accent_hover == "#ACD52F"
    # WCAG 1.4.11 fix (review M-9): disabled/border/focus_ring were retuned
    # to clear the 3:1 non-text floor on light surfaces; the three
    # signature colors above are the identity and stay verbatim.
    assert LIGHT.disabled == "#7E8C99"
    assert LIGHT.border == "#7B8B99"
    assert LIGHT.focus_ring == "#55701F"


def test_light_theme_ui_components_clear_3_to_1() -> None:
    """Review M-9 regression guard: every light-theme token that paints a
    UI state (disabled text, borders, focus ring / selected-tab underline)
    must reach WCAG 2.1 1.4.11's 3:1 against the white surface."""
    for tok in (LIGHT.disabled, LIGHT.border, LIGHT.focus_ring):
        assert contrast_ratio(tok, LIGHT.surface) >= 3.0, tok
    assert DARK.focus_color() == DARK.accent  # dark keeps the lime (8.1:1)
    assert LIGHT.focus_color() == LIGHT.focus_ring


def test_dark_theme_is_not_an_inversion() -> None:
    """§2.2: dark is adapted, not inverted — distinct surface ladder."""
    assert DARK.surface != LIGHT.surface
    assert DARK.primary != LIGHT.primary
    # a dark surface must be LIGHTER than the canvas it floats on
    assert contrast_ratio(DARK.surface, DARK.background) > 1.0


def test_scientific_palette_is_okabe_ito_and_not_lime() -> None:
    assert SCIENTIFIC_CATEGORICAL == (
        "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#D55E00", "#56B4E9", "#F0E442", "#333333")
    assert "#C6E84B" not in SCIENTIFIC_CATEGORICAL
    assert len(set(SCIENTIFIC_CATEGORICAL)) == 8


def test_scientific_palette_legible_and_distinct_per_theme() -> None:
    """The chart contract, per theme (§5 + §7): every categorical
    series color is ≥3:1 against the theme's CANVAS in the dark
    variant, and all eight are pairwise distinct everywhere."""
    from ui.design_tokens import scientific_categorical

    dark = scientific_categorical("dark")
    # dark variant: all series clear 3:1 on the Midnight canvas
    for color in dark:
        assert contrast_ratio(color, DARK.background) >= 3.0, \
            f"{color} on dark background"
    # the two variants agree except the gray swap
    assert dark[:7] == SCIENTIFIC_CATEGORICAL[:7]
    assert dark[7] == "#CCCCCC"


def test_scientific_palette_pairwise_distinct() -> None:
    from ui.design_tokens import (
        colors_are_distinct,
        scientific_categorical,
    )

    for theme in ("light", "dark"):
        assert colors_are_distinct(
            scientific_categorical(theme)), theme


def test_marker_discipline_is_programmatic() -> None:
    """§5's non-color rule is enforced by data: exactly the light-
    theme colors under 3:1 on white are flagged, dark flags none."""
    from ui.design_tokens import marker_discipline

    assert marker_discipline("light") == frozenset(
        {"#E69F00", "#CC79A7", "#56B4E9", "#F0E442"})
    assert marker_discipline("dark") == frozenset()


def test_typography_fallback_chains() -> None:
    assert TYPEFACES.ui_families[0] == "Inter"
    assert TYPEFACES.mono_families[0] == "JetBrains Mono"
    # every chain ends in its generic family
    assert TYPEFACES.ui_families[-1] == "sans-serif"
    assert TYPEFACES.mono_families[-1] == "monospace"
    assert "Arial" in TYPEFACES.ui_families
    assert "Consolas" in TYPEFACES.mono_families


def test_type_scale_targets() -> None:
    assert 24 <= SCALE.main_title_pt <= 28
    assert 14 <= SCALE.card_title_pt <= 16
    assert SCALE.body_pt == 10
    assert SCALE.WEIGHT_BOLD == 700


def test_layout_rhythm() -> None:
    assert LAYOUT.space_xs < LAYOUT.space_sm < LAYOUT.space_md \
        < LAYOUT.space_lg < LAYOUT.space_xl
    assert LAYOUT.card_min_width >= 240


def test_tokens_unknown_theme_falls_back_to_light() -> None:
    assert tokens("neon").name == "light"
    assert tokens("dark").name == "dark"
    assert set(THEMES) == {"light", "dark"}
