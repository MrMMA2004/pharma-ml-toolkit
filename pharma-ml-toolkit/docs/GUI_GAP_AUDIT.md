# GUI / Visualization / Export Gap Audit (§12, part5-continuation round 4)

**Prepared:** 2026-11-12, `part5-continuation` round 4
**Method:** MASTER_PROMPT part5 §15 — every row below was confirmed against
source and focused tests *before* any change; rows marked *not a gap* cite
the file, the round that landed it, and the test that pins it. Only
source-confirmed gaps were fixed (this round: 3 rows).

§12's standing rules were honoured throughout: the PySide6 / PyQtGraph /
QThreadPool architecture is untouched; no font-size increase or label
clipping was used as a layout substitute; existing export formats and
behaviour were preserved and extended, never replaced.

## Audit table

| # | §12 demand | Source-confirmed? | Status | Where / test |
| --- | --- | --- | --- | --- |
| 1 | Image-specific pixel/channel inspection | yes — already implemented | **not a gap** (rounds 240/245/246) | `ui/widgets/image_panel.py::pixel_at` (row/col, per-channel RGB, NaN disclosure, axis labels, sub-pixel scale note); `image_metadata()` at-rest census. Pinned by `tests/ui/test_image_hover_probe.py`, `test_image_subpixel_245.py`, `test_image_1to1_246.py` |
| 2 | Per-sample metadata on scatter plots | yes — already implemented | **not a gap** (round 241) | `PlotPanel.scatter(..., meta=)` + `ChartInspector._SeriesRecord.metadata` (computed lazily, cached per point); providers on model_analysis / deep_learning / pkpd / responseai / generative pages. Pinned by `tests/ui/test_scatter_meta_241.py` (16 tests) |
| 3 | Consistent nearest-point selection | yes — already implemented | **not a gap** (round 232) | `ChartInspector.inspect_at`: view-range-normalised distance, whole-bar-body targets for bars, hidden series skipped, honest `None` outside the tolerance window. Pinned by `tests/ui/test_chart_inspector.py::TestHoverReadout/TestChartKinds` |
| 4 | Dense-plot inspection debounced | **gap found** — the search ran on EVERY mouse move | **FIXED this round** | `ChartInspector._queue_inspect`: crosshair tracks each move; the nearest-point search + readout rebuild coalesce behind a 15 ms single-shot `QTimer`, cancelled on leave/disable/clear. New pins: `test_mouse_move_burst_runs_one_search`, `test_dense_series_move_storm_runs_bounded_searches` (200 moves over a 100k-point series → exactly 1 search), updated `test_mouse_move_drives_crosshair` |
| 5 | Debounced expensive metadata calculations | yes (per-point cache) + improved (row 4) | **closed** | The §21.6 `_meta_cache` (compute-once-per-point) was already in place; the missing half was the move-storm coalescing, fixed in row 4 — a hover burst now triggers at most one metadata lookup per settle window |
| 6 | Adaptive tick density | yes — already implemented | **not a gap** (round 238) | `PlotPanel._apply_tick_density`: stride-based thinning, every kept tick on a stride multiple, full ticks below the floor. Pinned by `tests/ui/test_plot_tick_density.py` (9 tests) |
| 7 | Axis-label collision handling | yes — already implemented | **not a gap** (rounds 238/242) | tick-share-aware elision inside `_apply_tick_density`; the §242 title budget (`_apply_title_budget`) keeps header/axis text inside the panel. Pinned by `test_plot_tick_density.py::test_very_long_names_fit_their_share`, `test_plot_title_budget_242.py` |
| 8 | Legend overlap | yes — already implemented | **not a gap** (round 243) | `PlotPanel._fit_legend` + the §243 placement policy. Pinned by `tests/ui/test_legend_policy.py` (11), `test_plot_legend_placement_243.py` |
| 9 | Long category labels | yes — already implemented | **not a gap** (round 238) | category ticks elide to their per-tick pixel share; thinned ticks still surface the full name via hover metadata. Pinned by `test_plot_tick_density.py` |
| 10 | Dense heatmaps | yes — already implemented | **not a gap** (rounds 221+) | `ImagePanel` (HistogramLUT + ROI stats) is the dense-matrix surface; the Image Lab similarity tab uses it. Pinned by `tests/ui/test_image_lab_page.py` (18), `test_image_hover_probe.py` |
| 11 | Small-window usability | yes — already implemented | **not a gap** (rounds 242/243 + responsive) | `ui/widgets/responsive.py` (`LEGIBILITY_FLOOR`, splitter policy, reflow) and the §23.3 export-side floor re-render. Pinned by `tests/ui/test_responsive_policy.py`, `test_plot_title_budget_242.py` |
| 12 | Export readability | yes — already implemented | **not a gap** (round ~250) | `ui/widgets/image_export.py`: `min_size` re-renders a squeezed `QGraphicsView` from its **vector scene** at floor scale (aspect preserved, background brush filled) — never a bitmap stretch. Pinned by `tests/ui/test_export_gui.py` |
| 13 | Preserve PNG/SVG/TIFF/JPEG; lossless high-bit TIFF | yes — already implemented | **not a gap** (round 226 core §17) | `pharmaml/viz/export.py`: `uint16` 2-D stays `I;16` in TIFF; compression knob (none/lzw/defflate); `export_figure` keeps SVG vectorial. Pinned by `tests/test_export_images.py::test_uint16_tiff_preserves_16_bits`, `test_tiff_compression_knob`, `test_figure_tiff_and_svg` |
| 14 | JPEG never silently downcasts high-bit-depth | yes — already implemented | **not a gap** (round 226) | `_flatten_for_target`: `I;16` → JPEG **raises** unless `downcast_16bit=True`; RGBA composites onto white with a warning. Pinned by `test_uint16_jpeg_refuses_then_explicit_downcast`, `test_rgba_jpeg_composites_onto_background` |
| 15 | Exported images reopened + validated (raster paths) | yes — already implemented | **not a gap** | `ImageExportOptions.validate=True` default: every `export_image`/`export_widget_image` write is reopened, magic-sniffed, dimension-checked; `ExportResult.validated`. Pinned by `test_validation_*` (4 tests) + `tests/ui/test_export_gui.py` (7) |
| 16 | Exported figures reopened + validated (matplotlib paths) | **gap found** — `save_figure` returns the path; the MD page reported "Saved" without checking the bytes | **FIXED this round** | `MolecularDynamicsPage.export_path` now validates every save: magic-sniff for PNG/SVG, real `%PDF` header for PDF (no raster magic bytes), nonzero size, dimensions for rasters — an unreadable file is an honest `Export FAILED … <reason>` line. New pins: `test_export_reports_validated_format_and_dimensions`, `test_export_validation_refuses_a_corrupt_file`, `test_export_validation_catches_empty_file` |
| 17 | Export evidence for the §11 Visualization card | **gap found** — the MD figure record carried only name/path | **FIXED this round** | the MD `record_run` summary now carries `format`, `width`, `height`, `size_bytes`, `kind` read from the **validated** artifact. Pinned by `test_export_records_real_evidence_for_the_card` |

## What changed (the round-4 diff)

1. `ui/widgets/chart_inspector.py` — the §12 dense-plot coalescing: a
   `_settle_timer` (single-shot, 15 ms) + `_pending_view` slot; mouse
   moves update the crosshair immediately but queue the search;
   leave-outside / disable / `reset()` cancel the pending target.
   `inspect_at()` itself is unchanged and synchronous (tests, pins and
   programmatic callers keep exact behaviour).
2. `ui/pages/molecular_dynamics.py` — `export_path` validates after
   saving (new `_validate_export`: sniff / `%PDF` / size / reopen via
   Pillow, never raising), reports format + dimensions + bytes +
   "validated" on success, `Export FAILED … reason` on a bad file, and
   records the checked artifact for the Visualization card.
3. `tests/ui/test_chart_inspector.py` — two new pins (burst coalescing;
   100k-point storm → one search) and the wiring test updated to the
   deferred-readout contract.
4. `tests/ui/test_molecular_dynamics_page.py` — four new §12 pins
   (validated reporting, corrupt-file refusal, empty-file refusal,
   card evidence).

## Honest environment record

All GUI tests in this audit run offscreen (`QT_QPA_PLATFORM=offscreen`
via `ui/headless.py` + the committed EGL stub) under the pinned stack
from `requirements.txt`: PySide6 6.8.3, pyqtgraph 0.13.7,
**rdkit 2026.9.1**, scikit-learn 1.5.2, numpy 1.26.4, torch 2.15
(cpu) — plus the optional extras the pages probe for (tqdm, openpyxl,
python-docx).

**The round-4 drift chase (recorded for the next session).** The first
chunk battery showed 14 failures. Root-caused in three layers, none of
them round-4 regressions:

1. *Missing optional deps* (tqdm, openpyxl, python-docx) — the doctor
   and Excel/report suites fail honestly without them; installed.
2. *A concurrent pip install* (torch, mid-run of chunk 4e) — segfault
   at teardown + ModuleNotFoundError noise; the stable-venv rerun was
   clean.
3. *The real drift:* rdkit **2026.09.2** (what a fresh `pip install
   rdkit` resolves today) vs the requirements pin **2026.9.1**. The
   seeded-number probes in the generative/DL/unsupervised suites (the
   GA "276 uniques" family, pareto front, umap, mini-batch k-means
   silhouette) shift under the patch drift — `mutation="topology"`
   walks ring systems through the RDKit API, and the 2026.09.2 build
   validates a different mutation set (measured: 260 vs 276 uniques
   under otherwise-identical seeds). With the pin restored: **81/81
   passed** on the three drift suites. This also corrects round 3's
   attribution: what `docs/NINE_CARD_EVIDENCE.md` recorded as
   "torch 2.15/numpy 2.1.3 version drift" was in fact this rdkit
   patch drift (that round's sandbox ran both drifts at once).

Final chunk battery (project chunk boundaries, one process per chunk,
all on the pinned stack): **4a–4f all exit 0, 532/532 UI tests
passed, 0 failed / 0 skipped / 0 errors** (dot count reconciled
against the 532 collected). The §12-adjacent suites inside that
battery: chart inspector 17, MD 7 (4 new this round), export GUI 7,
scatter meta 16, tick density 9, legend 11, image lab 18, plus the
hover/subpixel/1:1 probes, responsive and title-budget pins. A
GUI-with-real-display profile was **not executed** (no X server in
this sandbox) — recorded as not-executed-with-cause, per the
round-1..3 convention. Engine-dependent paths (docking/vina) are
likewise out of scope for §12.
