# Design System Implementation Audit — PharmaML Studio

**Round:** Mineral design-system integration (v1.2.202) · 2026-11-14
**Source of truth:** the uploaded "Design System" specification + `Design System.svg`

## 1. What was inspected

- `ui/design_tokens.py`, `ui/studio_theme.py`, `ui/theme_manager.py`,
  `ui/font_loader.py` — the part7 identity infrastructure (Deep Navy +
  Warm Ivory + Electric Lime, Inter + JetBrains Mono).
- All 22 `ui/pages/*.py` workspaces, `ui/widgets/` (21 modules),
  `ui/main_window.py`, `ui/nine_cards.py` — hex-literal sweep, QSS
  duplication, focus-rule coverage.
- Existing pinning suites: design_tokens, theme_manager, font_loader,
  theme_screenshots, focus_visibility, marker_discipline,
  mono_adoption, nine_cards, qt_doctor, chart_export.
- `Design System.svg` (8900×10012, outlined text): the structured
  token export transcribed programmatically (group-stack position
  resolution of the swatch rects) — see §3.

## 2. Findings (pre-migration)

| # | Finding | Disposition |
|---|---------|-------------|
| A-1 | The part7 identity is retired by the spec (§2: "must not override the selected Mineral design") | Full token migration |
| A-2 | `notification_widget.py` still carried Material-era hex pairs | Tokenized via `kind_style()` |
| A-3 | `navigation.py` ACCENT was qt-material teal `#00796B` | Mineral action token, test-pinned |
| A-4 | `empty_state.py` baked construction-time inline styles (light-only) | QSS objectName rules |
| A-5 | Nine-card breakpoints 880/560 vs spec 1200/860 | Corrected |
| A-6 | No font-status diagnostics (spec §5) | 2 doctor checks added |
| A-7 | Scientific palette: 8-color Okabe-Ito with a dark-gray swap | 7-series Mineral cycle with dark companions |
| A-8 | `test_chart_export` model_analysis failure pre-exists at R33 (M-3 labeled button) | Test corrected (design intent) |

## 3. SVG structured token export — transcription method

The SVG carries no `<text>` (all type is outlined paths), so the token
export was read from the **swatch rect geometry**: every `<rect fill>`
was resolved to absolute coordinates through the `<g transform>`
stack, then rows were reconstructed by position. The status triples
(foreground/soft-background/dark-foreground), the seven light→dark
data-color companion pairs, the primitive ramp, and the control-state
swatches (checkbox/toggle) were all recovered this way. Values are
pinned verbatim in `tests/ui/test_design_tokens.py`.

Derived (documented, not in the export): dark status soft backgrounds
(≈18% tint of the status foreground over the surface) and dark
`action/pressed` `#5EBFB4`.

## 4. Known limitations

- Light-theme `border/default` (#CCD8DE) is 1.45:1 on white — the
  spec's own quiet 1 px separator. Interactive identification is
  carried by focus rings (5.95:1) per WCAG 1.4.11; documented
  exemption in the token tests.
- The dark data companions for green (#63D3AC) and sky (#8DD6AC)
  converge (1.08:1, 14° hue gap). Kept verbatim from the export; the
  marker-discipline machinery differentiates the pair with symbols.
- The 18-section Advanced navigation, Job Center, search palette and
  settings page inherit the identity via the shared QSS; no page
  keeps a private stylesheet (verified: zero hex literals outside
  the token modules).
