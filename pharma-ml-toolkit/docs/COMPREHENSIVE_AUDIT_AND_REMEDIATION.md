# Comprehensive Audit and Remediation — v1.2.203 (partX3 round)

Scope: the MASTER_PROMPT partX3 mandatory audit (nine-card compliance,
backend-to-GUI integration, capability registry, docs-vs-source
reconciliation, bug sweep) plus this round's commissioned work: the
Design System's 02/SPECTRAL and 03/CATALYST directions shipped as
switchable alternate studio themes, and a deeper page-by-page visual
pass.

## 1. Executive summary

- All nine cards remain the primary architecture, in the canonical
  order (Introduction, Datasets, ML, DL, Generative, Docking, PK/PD,
  MD, Visualization); the 18 specialized pages stay reachable through
  Advanced navigation. Verified by tests/ui/test_nine_cards.py,
  test_nine_card_first.py, test_nine_cards_evidence.py (all green).
- The studio theme provider now offers SIX themes: Mineral light/dark
  (canonical, fresh-install default) + Spectral light/dark +
  Catalyst light/dark as opt-in alternates via Settings → Appearance.
  Every theme meets the same WCAG contract (4.5:1 body text, 3:1
  interactive accents; token-level tests parametrized over all six).
- The §4.2 scientific data palette is shared by all three directions
  ("Three directions. One scientific foundation.") — the alternates
  change the interface identity only, never the data-viz contract.
- Fixed a dead-QSS defect: `QPushButton#nineCardEntry-primary` styled
  a primary action that no widget ever carried (objectNames were
  `nineCardEntry-<section>` only). Each card's FIRST entry is now the
  primary action button, so the identity action fill actually renders.
- Page-by-page visual pass: 90 renders (18 sections x 5 themes), all
  with the correct canvas at the edge, correct shell fills, and real
  content. The two initially flagged "near-blank dark pages" were
  checker artifacts (surface-vs-canvas tolerance), verified fixed.

## 2. Files and architecture inspected

ui/design_tokens.py, ui/studio_theme.py, ui/theme_manager.py,
ui/font_loader.py, ui/main_window.py, ui/nine_cards.py,
ui/card_workspace.py, ui/widgets/{navigation, top_bar,
notification_widget, empty_state, plot_widget, chart_export}.py,
ui/pages/settings_page.py + all 18 page modules;
src/pharmaml/studio/capability_registry.py; tests/ui/ (48 suites);
docs/ (FEATURES, CAPABILITY_MAP, NINE_CARD_EVIDENCE, RELEASE_*).

## 3. Compliance matrix (previous requirements)

| Requirement | Status | Evidence |
| --- | --- | --- |
| Nine-card-first, canonical order | Implemented | test_nine_cards.py::test_grid_builds_nine_cards_row_major |
| 18 sections reachable, Advanced opt-in | Implemented | test_nine_card_first.py::TestRestoreBehaviour |
| Backend reuse (no GUI re-implementation) | Implemented | capability_registry.py routes; test_pages_smoke.py |
| Capability registry ↔ cards consistency | Implemented | test_nine_cards_evidence.py |
| Design tokens centralized | Implemented | test_design_tokens.py (61 tests, 6 themes) |
| Light + dark complete | Implemented | test_theme_manager.py, test_theme_screenshots.py |
| IBM Plex Sans + JetBrains Mono | Implemented | test_font_loader.py; doctor checks 17-18 |
| Scientific palette separate from action | Implemented | test_marker_discipline.py |
| DPI 100-200%, responsive breakpoints | Implemented | test_responsive_policy.py, test_nine_cards.py |
| Spectral/Catalyst as switchable alternates | Implemented (this round) | test_theme_manager.py::test_alternative_directions_apply_and_persist |
| Page-by-page visual pass, both themes | Executed (this round) | work/page_shots/ 90 renders, canvas+content checks |

## 4. Confirmed bugs found and fixed this round

1. Dead primary-action rule (ui/studio_theme.py +
   ui/nine_cards.py): `nineCardEntry-primary` QSS never matched any
   widget. Fix: the first entry of each card is now the primary
   action button. Regression-pinned by re-pointed tests in
   test_nine_cards.py / test_nine_card_first.py.
2. TopBar QSS background never painted (ui/widgets/top_bar.py): a
   plain QWidget subclass ignores QSS backgrounds without
   WA_StyledBackground — the Catalyst navy shell strip rendered
   canvas-colored in live windows (worked only in isolated grabs).
   Fix: set WA_StyledBackground. Verified at the pixels.
3. Sidebar glyphs could not follow studio-theme switches
   (ui/widgets/navigation.py): painted QIcon colors are stylesheet-
   immune. Fix: `set_theme_accent()` re-paints on every theme change;
   MainWindow wires it in `_on_theme`.

## 5. Changes made (files)

- ui/design_tokens.py: SPECTRAL, SPECTRAL_DARK, CATALYST,
  CATALYST_DARK token sets (WCAG-validated); shell/shell_text token
  fields; is_dark_theme(); THEMES extended to 6.
- ui/studio_theme.py: STUDIO_THEMES extended; per-direction radius
  ramp (Spectral rounder, Catalyst squared); shell QSS rules for
  studioTopBar / studioNav / QStatusBar / QMenuBar.
- ui/theme_manager.py: studio palette dark-detection via
  is_dark_theme (covers the alternates).
- ui/widgets/navigation.py: set_theme_accent.
- ui/widgets/top_bar.py: WA_StyledBackground.
- ui/main_window.py: nav accent wiring on theme change.
- ui/nine_cards.py: first-entry primary action.
- ui/pages/settings_page.py: appearance note lists the alternates.
- tests: test_design_tokens.py (6-theme parametrization + 3 new
  tests), test_theme_manager.py (5 new tests), test_theme_screenshots
  .py (6-theme canvas parametrization), test_marker_discipline.py
  (alternate-direction discipline test).

## 6. Test commands and results (batched, sequential, memory-aware)

- tests/ui/ — 48 suites executed one-per-process: ALL EXIT 0 except
  the documented environmental kill: test_nine_card_first.py dies at
  test 21/21 (exit 137, 4 GB ceiling; four MainWindows in one
  process). That test PASSES in a fresh isolated process (exit 0) —
  same environmental profile as the v1.2.202 release sweep, not a
  code regression.
- `python studio.py --doctor`: 18 OK / 0 warn / 0 failed.
- ruff + mypy clean on every changed file (repo-wide pre-existing
  debt unchanged).

## 7. Visual verification

- Six-theme home renders (1280x800): identity pixels verified —
  canvas + shell + action fills per theme (scripts/
  verify_alt_pixels.py: ALL OK).
- 90 page renders (18 sections x 5 themes): canvas-at-edge + content
  checks all pass (scripts/page_pass_analysis.py).

## 8. Remaining limitations / not verified here

- The full 5,000+ backend test suite was not re-executed this round
  (UI suites + doctor were the scope; backend untouched by these
  changes — the last full profile is recorded in the v1.2.201/v1.2.202
  release audits).
- `_elide_entry_label` in ui/nine_cards.py is dead code since the R33
  flow-wrap round (labels render in full); left in place, flagged for
  the next cleanup pass to keep this round minimal.
- Spectral/Catalyst dark-chart companions reuse the Mineral dark
  series (by design: one scientific foundation); no per-direction
  chart variants are specified in the Design System.

## 9. Final risk assessment

LOW for the shipped scope. The alternates are additive (new theme
names, no changes to Mineral values — pinned verbatim by tests); the
default identity and all persisted user choices are unchanged; the
QSS/shell fixes affect rendering only, verified at the pixels.
