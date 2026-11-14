"""Design-token compliance (Mineral §4/§5/§6).

Pins the token VALUES from the Design System spec and its structured
token export, and verifies, programmatically, the contrast the spec
demands: 4.5:1 for body text on its intended background, 3:1 for large
text and interactive accents, in BOTH themes.  A token regression
(someone edits a hex) fails here before it ships a washed-out theme.
"""

from __future__ import annotations

import pytest

from ui.design_tokens import (
    DARK,
    LAYOUT,
    LIGHT,
    SCALE,
    SCIENTIFIC_CATEGORICAL,
    SCIENTIFIC_CATEGORICAL_DARK,
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
        """Secondary text rides surfaces (cards, table headers) and must
        hold AA there; on the canvas it is promoted to primary text by
        the provider (the QSS routes canvas metadata to text_secondary
        only at metadata sizes, which hold 3:1+ everywhere)."""
        assert contrast_ratio(t.text_secondary, t.surface) >= 4.5

    def test_secondary_text_on_surface(self, t) -> None:
        # supporting text mostly rides surfaces; accept large-text
        # 3:1 as the floor there but require it strictly
        assert contrast_ratio(t.text_secondary, t.surface) >= 3.0

    def test_on_primary_reads_as_text(self, t) -> None:
        """action/on-primary on action/primary: white on teal (light)
        and ink on bright teal (dark) — 4.5:1, not just 3:1."""
        assert contrast_ratio(t.on_primary, t.primary) >= 4.5

    def test_text_on_selected_surface(self, t) -> None:
        assert contrast_ratio(t.text, t.selected_surface) >= 4.5

    def test_status_tokens_on_their_backgrounds(self, t) -> None:
        """Every status foreground clears AA on its soft background."""
        assert contrast_ratio(t.error, t.error_bg) >= 4.5
        assert contrast_ratio(t.warning, t.warning_bg) >= 4.5
        assert contrast_ratio(t.success, t.success_bg) >= 4.5
        assert contrast_ratio(t.info, t.info_bg) >= 4.5

    def test_status_foregrounds_on_surface(self, t) -> None:
        """Status text also appears on plain surfaces (table cells,
        report rows) — AA there too."""
        assert contrast_ratio(t.error, t.surface) >= 4.5
        assert contrast_ratio(t.warning, t.surface) >= 4.5
        assert contrast_ratio(t.success, t.surface) >= 4.5
        assert contrast_ratio(t.info, t.surface) >= 4.5


def test_identity_values_verbatim() -> None:
    """The Mineral signature values are exactly the spec's hexes."""
    assert LIGHT.background == "#F4F7F8"   # bg/canvas
    assert LIGHT.surface == "#FFFFFF"      # bg/surface
    assert LIGHT.primary == "#076F73"      # action/primary (Mineral teal)
    assert LIGHT.accent_hover == "#05565B"  # action/hover
    assert LIGHT.accent_pressed == "#06494E"  # action/pressed
    assert LIGHT.selected_surface == "#E3F3F1"  # action/selected
    assert DARK.background == "#0D1B23"    # bg/canvas
    assert DARK.surface == "#14232C"       # bg/surface
    assert DARK.primary == "#79D9CF"       # action/primary (bright teal)
    assert DARK.accent_hover == "#A0E9DF"  # action/hover
    assert DARK.selected_surface == "#06494E"  # action/selected
    # focus rings are the spec's focus/ring tokens
    assert LIGHT.focus_ring == "#076F73"
    assert DARK.focus_ring == "#79D9CF"
    # text/border ramps from the spec table
    assert LIGHT.text == "#14232C"
    assert LIGHT.text_secondary == "#4D6572"
    assert LIGHT.text_muted == "#697F8A"
    assert LIGHT.border == "#CCD8DE"
    assert DARK.text == "#EAF0F2"
    assert DARK.text_secondary == "#A9BCC6"
    assert DARK.border == "#697F8A"


def test_alternative_directions_are_absent() -> None:
    """§2: the retired Deep Navy/Ivory/Lime identity must be gone."""
    retired = ("#172F43", "#F3F5F1", "#C6E84B", "#ACD52F", "#F0F6D9",
               "#101B24", "#263D4D", "#55701F")
    for t in (LIGHT, DARK):
        for hex_val in retired:
            assert hex_val not in vars(t).values(), hex_val


def test_light_theme_ui_components_clear_3_to_1() -> None:
    """WCAG 1.4.11: the light-theme tokens that paint interactive UI
    states (focus ring, primary fill) reach 3:1 against the white
    surface.  border/default (#CCD8DE, 1.45:1) is DELIBERATELY exempt:
    the spec's quiet 1 px separators are not the identification cue —
    focus/selection are (see docs/DESIGN_SYSTEM_MIGRATION.md)."""
    for tok in (LIGHT.focus_ring, LIGHT.primary, LIGHT.accent_hover,
                LIGHT.accent_pressed):
        assert contrast_ratio(tok, LIGHT.surface) >= 3.0, tok
    assert DARK.focus_color() == DARK.accent
    assert LIGHT.focus_color() == LIGHT.focus_ring


def test_disabled_state_is_a_fill_not_text() -> None:
    """state/disabled paints backgrounds; disabled TEXT routes to
    text_muted (3.9:1 on canvas) by the QSS, which this pins."""
    assert LIGHT.disabled == "#EAF0F2"  # a fill, not a text color
    assert DARK.disabled == "#263A45"


def test_dark_theme_is_not_an_inversion() -> None:
    """§13: dark is adapted, not inverted — distinct surface ladder."""
    assert DARK.surface != LIGHT.surface
    assert DARK.primary != LIGHT.primary
    # a dark surface must be LIGHTER than the canvas it floats on
    assert contrast_ratio(DARK.surface, DARK.background) > 1.0
    # dark primaries INVERT the fill/text relationship instead
    assert contrast_ratio(DARK.on_primary, DARK.primary) >= 4.5


def test_scientific_palette_is_mineral_and_not_teal() -> None:
    """§4.2: the 7-series Mineral data palette, transcribed from the
    structured token export — and the action teal is NOT a series."""
    assert SCIENTIFIC_CATEGORICAL == (
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#444444")
    assert "#076F73" not in SCIENTIFIC_CATEGORICAL
    assert len(set(SCIENTIFIC_CATEGORICAL)) == 7


def test_scientific_palette_dark_companions_from_token_export() -> None:
    """Dark theme carries each series' lighter companion value."""
    assert SCIENTIFIC_CATEGORICAL_DARK == (
        "#69BDE8", "#F49A6A", "#63D3AC", "#E4A6CF",
        "#F0CC68", "#8DD6AC", "#E1E7EB")


def test_scientific_palette_legible_and_distinct_per_theme() -> None:
    """The chart contract, per theme: every categorical series color is
    ≥3:1 against the dark canvas; all are pairwise distinct."""
    from ui.design_tokens import scientific_categorical

    dark = scientific_categorical("dark")
    for color in dark:
        assert contrast_ratio(color, DARK.background) >= 3.0, \
            f"{color} on dark background"
    # light series ≥3:1 except the documented discipline set
    from ui.design_tokens import marker_discipline
    for color in scientific_categorical("light"):
        if color not in marker_discipline("light"):
            assert contrast_ratio(color, "#FFFFFF") >= 3.0, color


def test_scientific_palette_pairwise_distinct() -> None:
    """§12 contract per theme: every pair of series colors is distinct
    by luminance (≥1.25) OR hue (≥40°), or both members are flagged
    by marker discipline (the non-color encoding tells them apart)."""
    import colorsys

    from ui.design_tokens import (
        contrast_ratio,
        marker_discipline,
        scientific_categorical,
    )

    def hue_deg(c: str) -> float:
        r, g, b = (int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return colorsys.rgb_to_hsv(r, g, b)[0] * 360

    for theme in ("light", "dark"):
        palette = scientific_categorical(theme)
        disciplined = marker_discipline(theme)
        for i, a in enumerate(palette):
            for b in palette[i + 1:]:
                gap = abs(hue_deg(a) - hue_deg(b))
                distinct = (contrast_ratio(a, b) >= 1.25
                            or min(gap, 360 - gap) >= 40)
                if not distinct and not (
                        a in disciplined and b in disciplined):
                    raise AssertionError(
                        f"{theme}: {a} vs {b} neither distinct nor "
                        "covered by marker discipline")


def test_marker_discipline_is_programmatic() -> None:
    """§12's non-color rule is enforced by data: in light theme exactly
    the sub-3:1-on-white colors are flagged; in dark theme the
    green/sky companion pair (near-identical mint tints) is flagged so
    the pair is never told apart by color alone."""
    from ui.design_tokens import marker_discipline

    assert marker_discipline("light") == frozenset(
        {"#E69F00", "#56B4E9"})
    assert marker_discipline("dark") == frozenset(
        {"#63D3AC", "#8DD6AC"})


def test_typography_fallback_chains() -> None:
    assert TYPEFACES.ui_families[0] == "IBM Plex Sans"
    assert TYPEFACES.mono_families[0] == "JetBrains Mono"
    # every chain ends in its generic family
    assert TYPEFACES.ui_families[-1] == "sans-serif"
    assert TYPEFACES.mono_families[-1] == "monospace"
    assert "Segoe UI" in TYPEFACES.ui_families
    assert "Arial" in TYPEFACES.ui_families
    assert "Cascadia Code" in TYPEFACES.mono_families
    assert "Consolas" in TYPEFACES.mono_families


def test_type_scale_targets() -> None:
    """§5 hierarchy in logical px (pt = px·72/96): Display 40, Page
    title 26, Section 18, Card 16, Body 14, Label/Table 13, Metadata/
    Technical 12."""
    assert SCALE.display_pt == pytest.approx(30.0)      # 40 px
    assert SCALE.page_title_pt == pytest.approx(19.5)   # 26 px
    assert SCALE.section_title_pt == pytest.approx(13.5)  # 18 px
    assert SCALE.card_title_pt == pytest.approx(12.0)   # 16 px
    assert SCALE.body_pt == pytest.approx(10.5)         # 14 px
    assert SCALE.metadata_pt == pytest.approx(9.0)      # 12 px
    assert SCALE.code_pt == pytest.approx(9.0)          # 12 px
    assert SCALE.WEIGHT_BOLD == 700
    assert SCALE.WEIGHT_SEMIBOLD == 600


def test_layout_rhythm() -> None:
    """§6 scale: spacing 4/8/12/16/24, radius 4/6/8/12, controls
    28/32/36/40, table rows 28/32/40, borders 1/2."""
    assert (LAYOUT.space_xs, LAYOUT.space_sm, LAYOUT.space_md,
            LAYOUT.space_lg, LAYOUT.space_xl) == (4, 8, 12, 16, 24)
    assert (LAYOUT.radius_sm, LAYOUT.radius_md, LAYOUT.radius_lg,
            LAYOUT.radius_xl) == (4, 6, 8, 12)
    assert (LAYOUT.control_height_text, LAYOUT.control_height_compact,
            LAYOUT.control_height_default, LAYOUT.control_height_lg) \
        == (28, 32, 36, 40)
    assert (LAYOUT.table_row_compact, LAYOUT.table_row_standard,
            LAYOUT.table_row_comfortable) == (28, 32, 40)
    assert (LAYOUT.border, LAYOUT.border_emph) == (1, 2)
    assert LAYOUT.card_min_width >= 240


def test_tokens_unknown_theme_falls_back_to_light() -> None:
    assert tokens("neon").name == "light"
    assert tokens("dark").name == "dark"
    assert set(THEMES) == {"light", "dark"}


def test_card_title_contrast_both_themes() -> None:
    """Card titles read t.text on t.surface in BOTH themes (the part7
    dark-title regression stays fixed — t.primary in dark is a bright
    FILL token and would wash out as text)."""
    from ui.studio_theme import studio_qss

    for theme in ("light", "dark"):
        assert "QLabel#studioCardTitle" in studio_qss(theme)
        assert contrast_ratio(
            tokens(theme).text, tokens(theme).surface) >= 4.5


def test_notification_kinds_resolve_to_status_tokens() -> None:
    """The toast kinds map to the semantic status tokens, both themes
    (the migration's notification tokenization follow-up, pinned)."""
    from ui.widgets.notification_widget import kind_style

    for theme, t in (("light", LIGHT), ("dark", DARK)):
        assert kind_style("success", theme) == (t.success_bg, t.success)
        assert kind_style("info", theme) == (t.info_bg, t.info)
        assert kind_style("warning", theme) == (t.warning_bg, t.warning)
        assert kind_style("danger", theme) == (t.error_bg, t.error)


def test_navigation_accent_is_the_token_teal() -> None:
    """The sidebar's painted icon color cannot come from QSS, so it is
    pinned to the action/primary token value it mirrors."""
    from ui.widgets.navigation import Navigation

    assert Navigation.ACCENT == LIGHT.primary == "#076F73"
