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
│   ├── pages/  dashboard · data_explorer · chemistry · pkpd · biology · docking · model_analysis · unsupervised · deep_learning · generative · settings_page
│   ├── widgets/ navigation · data_table · plot_widget · progress_panel · notification_widget
│   └── workers/ job_worker.py (QRunnable + cooperative stop)
└── tests/ui/            ← 167 offscreen tests (pytest-qt)
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

Increment 11 of N — shell, navigation, theme system, Data Explorer,
Chemistry Workbench, PK/PD Workbench, Biology Workbench, Docking
Workbench, Model Analysis, Unsupervised Workbench, Deep Learning
Workbench, Generative Workbench, Foundations Workbench, Best-of ML
Workbench, Response AI Workbench, Settings are live and tested. The
Response AI Workbench (increment 11, audit row 12) gives
``pharmaml.ml.drugresponse`` — the module absorbed from
CaDRReS / Precily / DeepResponse with no prior GUI surface — its
first one: eight tabs, each wired to the module's public APIs.  The
Hill curve (``fit_hill_curve`` on the seeded 4PL demo — every fitted
parameter against the generating truth, r², the log-dose AUC both
ways, model-based vs closed-form), the response metrics
(``response_metrics`` on the seeded noisy pair or two Data Explorer
columns — Pearson/Spearman/RMSE/MAE with the observed-vs-predicted
scatter), the rank correlations (``rank_correlations`` — Pearson,
Spearman and Kendall tau-b in one call), the response model
(``DrugResponseModel`` — the CaDRReS bilinear factorisation on the
seeded 12x4 screen with a 15% NaN mask, full-batch gradient descent
over the observed pairs only, the honest in-sample pearson and the
predicted-response head), the cross-validation
(``cross_validate_response`` — folds split CELL LINES so no cell line
straddles a fold; the ~0 mean pearson reported honestly), the
informative regression (``InformativeRegression`` — per-feature prior
variances shrink non-informative weights harder, the shrunk
coefficients against the generating truth), the response VAE
(``ResponseVAE`` — the ELBO bound improving, with the
reconstruction / KL / response ledgers), and the multi-omics fusion
(``MultiOmicsFusion`` — z-scored blocks with the per-block total
variance equalised to 1.0, the audit table).  17 offscreen tests,
every asserted number engine-printed first by the committed probe
(``scratch/probe_response_218.py``, round 219 re-seeded per section
so each tab is a standalone seeded job). The Best-of ML Workbench
(increment 10, audit row 11) brings the playground Best-of ML tab's
clusters, each wired to the same public backend APIs the cards call:
the catalogue (``search_catalog`` / ``catalog_top`` over the embedded
best-of-ml-python snapshot — 924 projects, 35 sections, the top-10
star counts as the bars; the search keeps the catalogue's score
order), the coverage audit (``coverage_table`` — the honest map of
all 35 categories: 12 native / 6 partial / 4 external / 13
out-of-scope, with the verdict-count bars), the differential privacy
surface (``dp_statistics`` — count/sum/mean/histogram under one
epsilon budget with the worst-bin row, the data-derived-bounds
disclosure and the RDP accountant's DP-SGD sigma; the true-vs-DP bin
counts as grouped value-space bars), the recommender (the seeded
24x20 latent-factor matrix through ``evaluate_recommender`` —
leave-one-out hit-rate against the popularity baseline, which CAN win
and the report says so — plus user 0's ranked recommendations), the
vector search (``ann_benchmark`` — the LSH recall@5 vs
candidate-compression trade-off curve over plane counts 2/4/6/8/10
with multiprobe), the adversarial audit (logistic regression on the
loaded table or the seeded two-blob demo attacked by the zeroth-order
sign family — the broken-rate curve and the median adversarial
margin), the time-series analysis half (ACF as the lag bars / PACF,
the additive decomposition strengths, CUSUM changepoints, the
tsfresh-lite window table), and the association rules
(``apriori_pipeline`` over pasted transactions with the lift bars).
19 offscreen tests, every asserted number engine-printed first by the
committed probe (``scratch/probe_bestofml_217.py``). The Foundations
Workbench (increment 9, audit row 10)
brings the playground Foundations tab's clusters, each wired to the
same public backend APIs the cards call: the group comparison
(``compare_groups`` — descriptives, both normality verdicts, the
decision branch reported, the CIs and Cohen's d, with both group
histograms drawn on ONE shared axis; the synthetic demo mirrors the
card's rng-42 groups exactly, the columns mode reads the Data
Explorer's loaded table), the distribution sampler
(``sample_distribution`` over the course's five distributions with the
theoretical pdf/pmf drawn over the sample histogram in the SAME value
space — the round-215 ``PlotPanel.bars(x=…)``/``histogram(range=…)``
extension), the inferential tests (``welch_anova`` + BOTH pairwise
post-hoc tables, Tukey HSD in the data table and Games-Howell in the
report), and the table profiler (``profile_dataset`` — the type
census bar chart, the per-column table and the fg-data-profiling
alert rules). 13 offscreen tests, every asserted number
engine-printed first by the committed probe
(``scratch/probe_foundations_215.py``). The Generative
Workbench (increment 8) brings the playground Generative tab's
clusters, each wired to the same public backend APIs the cards call:
the evolution run (``build_objective``-built objectives — QED, the
LogP / MW / TPSA windows, the QED+LogP blend, the multi-objective
triple, similarity — over ``GeneticOptimizer`` /
``ReinventOptimizer`` / ``MctsOptimizer``, the CLI corpus idiom
mirrored exactly, with the ranked leaderboard and the
per-generation best/mean curves: the GA over the built-in corpus
lands best 0.927 over 271 uniques, the multi-objective seeded run
0.962 over 276), the Pareto front (``pareto_front`` over the last
run's breakdown — 9 of 276 non-dominated, the trade-off scatter
over the first two objectives, the deepest-front-member rank; a
single-objective run refuses honestly), molecule surgery
(``ring_topology_moves`` — piperidine's 3 mutants — and
``scaffold_hop_crossover`` breeding, every product scored by
``QedObjective``), and the GuacaMol scoring oracles
(``qed_score`` / ``sa_score`` / ``penalized_logp`` /
``rediscovery_score`` + the property filter). The Deep Learning
Workbench
(increment 7) brings the playground Deep Learning tab's four audit
clusters, each wired to the same public backend APIs the cards call:
the training run (``make_train_test_split`` over the numeric-feature
rule, then ``DeepMLP*``/``ResidualMLP*`` (torch) or ``NumpyMLP*``
(the toolkit's own numpy autograd engine when torch is absent),
fitted with per-epoch callbacks and early stopping, then
``evaluate_model`` — the parity scatter with the overfit line
(Pearson r, train R², the gap) for regression or the per-class F1
bars with the confusion table for classification: on the bundled
260-row QSAR regression table DeepMLPRegressor lands R² 0.891 with
an honest overfit gap of 0.061, and DeepMLPClassifier reaches
accuracy 0.952 over active/inactive/weak), the runboard
(``RunTrace`` over the run's ``history_`` with the raw +
EMA-smoothed (``smooth_series``, bias-corrected, TensorBoard's own
slider math) loss curves, and the weight histogram via
``numpy_layer_stack`` + ``histogram_event`` — the numpy engine's
layers snapshot, with a torch run refusing honestly by naming the
engine rule), the UMAP map (``umap_embed`` on the run's test rows or
the dataset's numeric matrix, the fuzzy k-NN graph and spectral init
with the Venna-Kaski ``trustworthiness`` score in the report), and
the fiftyone-style curation (``SampleSet``/``DataView`` stage
algebra — filter / sort_by / limit / identity reset with every stage
named — plus the four audits: ``find_duplicates``,
``find_outliers``, ``uniqueness`` and ``sort_by_similarity``). The
Unsupervised Workbench (increment 6, v1.2.126) brings the PDSH
roster — PCA variance ladder + loadings, k-means/mini-batch/GMM/hierarchical
clustering with the soft responsibilities, KDE density over the
histogram twin, and all five manifolds. The Docking Workbench (increment 5, v1.2.125) brings four of the playground
Docking tab's engines, each wired to the same public backend APIs the
cards call, none needing the external engine binaries: the campaign
summary (`check_docking_completion` + `scan_docking_directory` over
the bundled synthetic AutoDock4 campaign — 14 folders: 12 complete
runs, one missing log, one interrupted before clustering — then the
rescore card's arithmetic on the OK rows: RDKit heavy atoms from the
campaign's `ligands.csv`, `ligand_efficiency_series`, and
`consensus_rescore` over `best_lowest_energy` + `best_num_in_cluster`
+ `ligand_efficiency`, ranked best-first: metformin 0.782 top), the
cluster analysis (`parse_dlg_cluster_table` — the CLUSTERING
HISTOGRAM as a real table with the top-cluster and most-populated
story, the folder combo fed by the scan's cross-tab state), the
ligand library audit (`load_ligand_csv` — 8 valid + 2 deliberately
invalid rows whose reasons ride the table — plus the RDKit heavy-atom
and rotatable-bond columns the preparation card reports), and the
DOCKSTRING lab (`target_frame`, all 58 upstream Vina search boxes,
with `get_target` config previews and `parse_search_box` validation
of a pasted six-key config). The Biology Workbench
(increment 4, v1.2.124) brings four of the playground Cell Lines
tab's engines, each wired to the same public backend APIs the cards
call: the screen audit (`load_response_table` →
`audit_response_table` → `replicate_variability` →
`collapse_replicates` → `summarize_compounds` → `response_matrix`,
one job: the bundled 185-row GDSC-style screen loads, audits its
missing metric values and duplicate pairs, collapses the replicates
and pivots the 23×8 matrix the other tabs share), the sensitivity
matrix (`response_matrix` + `call_sensitive_resistant` quartile
calls — lower LN_IC50 means more sensitive, the card's convention),
the MAD gene ranking (`most_variable_genes` with the per-gene MAD
recomputed honestly from the frame), and biomarker discovery
(`normalize_expression` → `impute_expression` →
`sensitivity_associations` with BH correction → `top_biomarkers`,
with `direction_of_association` naming the top hit's story). The
PK/PD Workbench (increment 3, v1.2.122) brings
three of the playground PK/PD tab's engines, each wired to the same
public backend APIs the cards call: the exact two-compartment regimen
simulator (`TwoCompartment` matrix-exponential linCmt with explicit
repeated `DosingEvent` records — concentration curve, dose markers,
Cmax/Tmax, trapezoid AUC), the non-compartmental analysis
(`validate_concentration_series` → `nca_summary` plus the extended
`fit_terminal_phase`/`auc_linear_log` — linear-up/log-down AUC,
terminal λz window, Vd and CL when an IV dose is given, the observed
profile drawn with its AUC shaded), and the NONMEM/mrgsolve event
grammar (`ev`/`addl_until`/`realize_addl`/`ev_days` — the full realized
dose-event table with weekday schedules). All three run in QThreadPool
workers; the honest disclosure: they are single-shot engine calls
without checkpoint loops, so the cooperative Stop lands only for
checkpoint-wired jobs. The Chemistry Workbench (increment 2, v1.2.121)
standardizes a SMILES column through the playground's own
`standardize_csv` batch front door and computes multi-family features
through `compute_feature_frame` with the v1.2.119 batch guard carried
over: a five-molecule probe fixes the exact per-molecule column count,
then the dense-matrix estimate is checked against half of
`MemAvailable` and the batch refuses with its arithmetic before the
heavy compute. Next increments migrate the remaining playground
sections (Foundations per the audit's usage order); the Tkinter
playground is retired only after every section has a verified Studio
page. See `docs/gui/GUI_AUDIT.md` for the full tab-by-tab map.
