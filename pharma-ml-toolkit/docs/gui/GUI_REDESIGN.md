# PharmaML Studio — the PySide6 GUI (directive #16, increment 1)

Round 206 delivers the first increment of the complete GUI redesign
mandated by `upload/MASTER_PROMPT part2.pdf` (the expanded 44-page
four-technology edition: **PySide6** application framework,
**PyQtGraph** scientific visualization, **Qt-Material** primary
Material Design theming, **QDarkStyleSheet** verified alternative).

## Launch

```bash
cd pharma-ml-toolkit
python studio.py                      # desktop session
QT_QPA_PLATFORM=offscreen python studio.py   # headless / CI
```

`ui.headless.ensure_headless()` runs first in both cases: on machines
without system EGL it preloads the committed no-op stub
(`ui/native/libEGL.so.1`, real EGL always probed first) and selects
the offscreen platform when no display exists. On a normal desktop all
of this is a no-op.

## Architecture (16.5/16.16)

Three layers, strictly separated:

1. **Presentation** — `ui/`: `main_window.py` (QMainWindow, sidebar +
   QStackedWidget + status bar), `pages/` (dashboard, data explorer,
   model analysis, settings), `widgets/` (navigation, data table,
   plot panel, progress panel, notifications), `theme_manager.py`,
   `app_settings.py` (QSettings persistence).
2. **Application logic** — the existing `pharmaml` package, reused
   as-is: `read_table` for CSV/TSV/XLSX, `MODEL_REGISTRY` +
   `train_model` for training, the v1.2.119 `JobCancelled`
   cooperative-stop channel for cancellation.
3. **Data** — the bundled `examples/data/` datasets (and any file the
   user opens); nothing is fabricated: dashboard census numbers are
   computed live from the installed package, plots carry real
   held-out predictions.

```
pharma-ml-toolkit/
├── studio.py            ← entry point
├── playground.py        ← the Tkinter GUI (kept until migration completes)
├── ui/
│   ├── headless.py      ← EGL-stub/offscreen bootstrap
│   ├── theme_manager.py ← provider switching + chart palettes
│   ├── app_settings.py  ← QSettings facade
│   ├── main_window.py
│   ├── native/          ← eglstub.c + libEGL.so.1 + build_stub.sh
│   ├── pages/  dashboard · data_explorer · model_analysis · settings_page
│   ├── widgets/ navigation · data_table · plot_widget · progress_panel · notification_widget
│   └── workers/ job_worker.py (QRunnable + cooperative stop)
└── tests/ui/            ← 36 offscreen tests (pytest-qt)
```

## Theming (16.9–16.11)

One active provider at a time, switched through the Settings page (or
`ThemeManager.apply`) without restarting:

- **qt-material** (primary): curated teal/cyan/amber pairs —
  `dark_teal.xml` default; density via the template `extra` dict
  (`density_scale` −1..+2).
- **qdarkstyle** (alternative): its single dark theme, verified on
  PySide6 6.8.3 through qtpy (binding warning is cosmetic).
- **native**: no stylesheet — the guaranteed fallback.

Every switch clears the previous stylesheet first (16.10.4) and emits
a `ChartPalette` that all `PlotPanel`s consume, so PyQtGraph plots
re-colour (background, axes, grid, series pens) to match the theme.

## Background execution (16.14)

`JobWorker` (QRunnable on the global QThreadPool) runs backend
callables off the GUI thread; progress/results/errors/cancellations
marshal back via queued signals. The progress callback's `False`
return is the cooperative-stop signal — convention-following backend
loops (v1.2.119) raise `JobCancelled`, which lands as a clean
cancellation, never a traceback. `JobRunner` refuses duplicate
submissions while busy.

## Dependencies (16.17)

Verified pins under the numpy 1.26.4 constraint (see
`requirements.txt`): PySide6 6.8.3 (Qt 6.8 LTS), pyqtgraph 0.13.7
(the last numpy-1.x line — 0.14 requires numpy ≥ 2.2), qt-material
2.17, qdarkstyle 3.2.3, pytest-qt 4.5.0. `scratch/venv_repair.sh`
carries the same pins (step 6b) so rollbacks cannot drop the GUI
stack.

## Testing (16.19)

```bash
cd pharma-ml-toolkit && python -m pytest tests/ui/ -q
```

36 tests: theme switching/conflict-prevention/persistence, navigation
and geometry persistence, table model/sort/copy/empty-state/truncation,
worker finish/fail/cancel/duplicate-guard/progress, page smokes on real
data (dashboard census, real CSV + Excel load, an end-to-end `ridge`
training run with held-out R²), and chart-palette sync.

## Migration status (16.18)

Increment 2 of N — shell, navigation, theme system, Data Explorer,
Chemistry Workbench, Model Analysis, Settings are live and tested.
The Chemistry Workbench (increment 2, v1.2.121) standardizes a SMILES
column through the playground's own `standardize_csv` batch front door
and computes multi-family features through `compute_feature_frame`
with the v1.2.119 batch guard carried over: a five-molecule probe
fixes the exact per-molecule column count, then the dense-matrix
estimate is checked against half of `MemAvailable` and the batch
refuses with its arithmetic before the heavy compute — both jobs in
QThreadPool workers with the cooperative Stop. Next increments
migrate the remaining playground sections (PK/PD per the audit's
usage order); the Tkinter playground is retired only after every
section has a verified Studio page. See `docs/gui/GUI_AUDIT.md` for
the full tab-by-tab map.
