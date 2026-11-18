# Design System Migration — part7 → Mineral

**Version:** v1.2.202 · 2026-11-14
**Scope:** the complete visual identity swap, on the existing token/QSS/provider architecture (no parallel theme system created).

## 1. Design-to-code mapping (spec §10)

| Spec component | Implementation |
|---|---|
| Semantic color tokens (§4.1) | `ui/design_tokens.py::ColorTokens` (light/dark) |
| Scientific palette (§4.2) | `SCIENTIFIC_CATEGORICAL(_DARK)` + `marker_discipline()` |
| Typography (§5) | `TypefaceTokens` (IBM Plex Sans / JetBrains Mono chains), `TypeScale` (40/26/18/16/14/13/12 px) |
| Spacing/shape/controls (§6) | `LayoutTokens` (4/8/12/16/24; radius 4/6/8/12; controls 28/32/36/40; rows 28/32/40; borders 1/2) |
| Application shell (§7) | `studio_theme.studio_qss` (status bar = bg/subtle strip; header/tabs/scrollbars restyled) |
| Nine-card home (§8) | `ui/nine_cards.py` breakpoints 1200/860; card QSS (radius 8, 1 px border/subtle) |
| Reusable components (§9) | Shared QSS rules by objectName (`nineCard`, `studioEmptyState`, `chartExportButton`…); buttons/fields/tables/tabs/progress all from tokens |
| Qt migration map (§10) | Unchanged widget classes; QSS regenerated, no widget replaced |
| Chart theme (§12) | `ThemeManager._palette_for("studio")`: grid from border/subtle, accent = action/primary, series = Mineral cycle |
| Light/dark (§13) | Two complete `ColorTokens` sets; switch re-emits ChartPalette; pixel-verified |
| DPI (§14) | Qt's own scaling (`QT_SCALE_FACTOR` ladder verified 100–200%; no manual scaling) |

## 2. What changed, file by file

- `ui/design_tokens.py` — rewritten to the Mineral semantic set (see
  the AUDIT for values and derived tokens).
- `ui/studio_theme.py` — QSS regenerated: flat surfaces, 1 px
  separators, subtle-fill table headers + border-bottom, bg/subtle
  status-bar strip, Mineral radii (buttons 6, cards 8), focus rings
  from focus/ring, empty-state objectName rules, checkbox indicators.
- `ui/font_loader.py` — Plex bundle + `has_plex()`; float point sizes.
- `ui/theme_manager.py` — studio palette grid/accent from tokens.
- `ui/widgets/notification_widget.py` — `kind_style(kind, theme)` →
  status tokens.
- `ui/widgets/navigation.py` — ACCENT = teal action token.
- `ui/widgets/empty_state.py` — objectName styling, no inline hex.
- `ui/widgets/plot_widget.py` — `_needs_marker` is theme-aware (dark
  mint-pair discipline).
- `ui/nine_cards.py` — breakpoints 1200/860.
- `ui/doctor.py` — 2 font checks (18 total) + the QGuiApplication
  probe fix.
- `ui/__init__.py` — font-registration comment updated.
- `fonts/` — IBM Plex Sans Regular/Medium/SemiBold/Bold + OFL; Inter
  bundle removed.
- Tests: `test_design_tokens`, `test_theme_manager`,
  `test_font_loader`, `test_theme_screenshots`, `test_focus_visibility`,
  `test_marker_discipline`, `test_qt_doctor`, `test_chart_export`,
  `test_nine_cards` — re-pinned per the migration.

## 3. Compatibility

- Public token API shape preserved (`LIGHT/DARK/THEMES/tokens()`,
  `TYPEFACES`, `SCALE`, `LAYOUT`, `contrast_ratio`). Field names
  changed where the spec's semantic roles demanded it (e.g.
  `accent_pressed`, `on_primary`, status fields); all in-repo
  consumers were migrated and pinned by tests. No external plugin
  surface depends on the old part7 names (the API was internal).
- Providers: `studio` remains the default; qt-material/qdarkstyle/
  native unchanged.
- Backend: untouched (no calculation, model or workflow change).

## 4. Rollback

Single revert of this round's commit restores the part7 identity;
the fonts are additive files and the doctor's new checks warn (never
fail) when a bundle is absent.
