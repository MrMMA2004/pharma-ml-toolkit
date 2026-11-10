# Feature inventory — the truthful traceability matrix

This is the §4.5 deliverable of the MASTER_PROMPT part 4 specification:
**every major feature classified honestly, with a feature-to-backend-to-UI-to-test
traceability matrix.** All evidence on this page was collected live against the
working tree (imports grepped, pages/cards/routes enumerated, tests collected
with `pytest --collect-only`), not copied from prose. Matrix inputs were frozen
at v1.2.167 (`9130c6b`); the inventory itself ships with v1.2.168.

## Status legend (the six §4.5 classes)

| Code | Class |
| --- | --- |
| **A** | Implemented and tested |
| **B** | Implemented but incompletely tested |
| **C** | Backend available but not integrated into the GUI |
| **D** | Partially implemented |
| **E** | Not implemented |
| **F** | Blocked by an optional dependency or external engine |

The three honesty rules are applied literally:

1. **A GUI button is not proof of backend functionality** — every row's UI
   entry point was traced to the backend call it makes (Studio page imports,
   web route → bridge op → `pharmaml` function). CLI counts come from
   `python -m pharmaml --help` and `cli.py`'s 44 `add_parser` sites
   (8 groups / 36 subcommands).
2. **A backend function is not proof of GUI integration** — backend modules
   with no page, card, or route are listed as class **C** below, not hidden.
3. **A synthetic demonstration is not proof of real experimental or production
   data support** — features whose shipped demos are synthetic say so
   explicitly (§ "Synthetic-demo register").

## 1. The desktop Studio — 18 pages

Backend column = modules the page imports/calls (verified by import grep).
Tests column = dedicated page test + the backend suites that pin the
underlying calls. All pages additionally ride `tests/ui/test_pages_smoke.py`
(8 smoke tests) and the offscreen platform checks
(`test_headless_platform.py`, `test_qt_doctor.py`).

| Page (nav key) | Backend | UI tests | Backend tests | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Dashboard (`dashboard`) | `pharmaml` meta, `ml.models` registry | smoke only | — | **B** | Statistics/quick-launch panel. The §5 nine-card workflow dashboard is class **D** (below). |
| Data Explorer (`data`) | `utils.io` | `test_data_table.py` (6) | `test_io.py` (69), `test_data_loading.py`, `test_data_provenance.py` | **A** | Page-level depth is the data-table widget; the import/export library side carries 69 focused tests. |
| Chemistry Workbench (`chemistry`) | `chem.csv_standardize`, `chem.features`, `utils.io` | `test_chemistry_page.py` (13) | `test_chem_*`, `test_descriptors.py` (73), `test_csv_standardize.py` (57), `test_similarity.py`, `test_diversity.py` | **A** | RISM descriptors are opt-in (see §5). |
| PK/PD Workbench (`pkpd`) | `pharma.pk`, `pkpd.compartments/events/nca_ext/ode` | `test_pkpd_page.py` (18) | `test_pk.py` (49), `test_pd.py`, `test_pkpd_engines.py` (99), `test_dosage.py` (24) | **A** | |
| Biology Workbench (`biology`) | `biology` (incl. `load_response_table`) | `test_biology_page.py` (13) | `test_biology.py` (93), `test_cellline_examples.py`, `test_variable_genes` parity | **A** | Demo cell-line data is synthetic-but-realistic (see §6). |
| Docking Workbench (`docking`) | `docking` | `test_docking_page.py` (14) | `test_docking.py`, `test_docking_prep.py` (84), `test_pose_rmsd.py` (48) | **A** | Prep/parse/rescore only — no bundled docking engine (class **F** note in §5). |
| Model Analysis (`models`) | `ml.cv_results`, `ml.models`, `ml.training` | `test_model_analysis_page.py` (13) | `test_cv_results.py`, `test_ml_pipeline.py` (211) | **A** | |
| ML Workbench (`mlworkbench`) | `ml` core, `ml.evaluation`, `ml.persistence`, `ml.tuner`, `utils.checkpoints` | `test_ml_workbench_page.py` (19) | `test_ml_pipeline.py` (211), `test_tuner.py` (118), `test_checkpoints.py` | **A** | Stacking/voting exposed; boosting families optional-gated (§5). |
| Unsupervised (`unsupervised`) | `ml.manifold`, `ml.unsupervised`, `utils.io` | `test_unsupervised_page.py` (23) | `test_unsupervised.py`, `test_manifold.py`, `test_hierarchical.py` | **A** | |
| Deep Learning (`deep_learning`) | `ml.neural`, `ml.numpy_neural`, `ml.runboard`, `ml.curation` | `test_deep_learning_page.py` (27) | `test_neural.py` (95), `test_numpy_neural_checkpoints.py`, `test_runboard.py`, `test_umap.py`, `test_curation.py` | **A** | Torch optional-gated (§5). Pruning/quantization/GNN/harness are class **C** (§4). |
| Generative Workbench (`generative`) | `generative`, `chem.oracles`, `chem.sa_scoring` | `test_generative_page.py` (31) | `test_generative.py` (291), `test_mcts.py` (12), `test_sa_score.py`, `test_generative_sa_cli.py` | **A** | MCTS moves are Studio-only (no web twin). |
| Foundations (`foundations`) | `ml.stats`, `ml.inferential`, `profiling` | `test_foundations_page.py` (13) — end-to-end (`_start_stats()` runs the real Welch ANOVA / Tukey / Games-Howell / `profile_dataset`) | none dedicated | **B** | No focused library suite for `ml.stats`/`ml.inferential`/`profiling`; the 13 page tests are the only direct pins. |
| Best-of ML (`bestofml`) | `ml.adversarial/annsearch/apriori/catalog/privacy/recommend` | `test_bestofml_page.py` (19) | `test_campaign.py` (111), `test_catalog_foldins.py` (27), plus per-module suites | **A** | Six absorbed engines, four cards each. |
| Response AI (`responseai`) | `ml.drugresponse` | `test_responseai_page.py` (17) — end-to-end hill-fit/metrics/rank over the real module (the module's first tests rode the page) | none dedicated | **B** | Studio-only (no web twin); `ml.drugresponse` has no focused backend suite beyond the page tests. |
| Image Lab (`imagelab`) | `biology` imaging, `chem.descriptors`, `chem.similarity`, `viz.export` | `test_image_lab_page.py` (18) + `test_image_1to1_246.py` (16) + `test_image_subpixel_245.py` (16) + `test_image_hover_probe.py` (20) | `test_export_images.py`, `test_viz.py` (221) | **A** | Pixel-policy pins are first-class tests. |
| Molecular Dynamics (`molecular_dynamics`) | `md`, `viz` | `test_molecular_dynamics_page.py` (3) | `test_md_analysis.py` (63), `test_md_cv.py` (65), `test_md_plots.py` + `test_viz_md_plots.py` | **A** | Page-test depth is 3 but the feature's backend+web coverage is the deepest in the toolkit (5 bridge ops, 5 cards). Synthetic trajectories (§6). |
| Utilities (`utilities`) | `chem.converters`, `docking` collect, `pharma`, `utils.io`, `viz` | `test_utilities_page.py` (18) | `test_converters.py` (74), `test_format_converters.py` (50), `test_campaign.py` (111), `test_approvals.py` (15) | **A** | Campaign + approvals management live here. |
| Settings (`settings`) | (none — Qt/theme/app settings only) | smoke only | `test_theme_manager.py` (9), `test_app_settings` via main-window tests | **B** | No backend surface by design; theme/settings logic is tested, the page itself is smoke-only. |

## 2. The web console — 6 sub-tabs, 33 always-rendered cards, 27 API routes

Cards are the canonical `PLAYGROUND_CARDS` registry
(`src/components/explorer/playground/card-directory.ts`, 33 entries, pinned
by `TestCardFinderParity` + the E2E probe that walks every entry). Routes are
the live `src/app/api/**/route.ts` set. Backend columns trace through the
`scripts/*_bridge.py` processes to the real `pharmaml` functions.

| Area | Cards | Routes | Backend | Tests | Status |
| --- | --- | --- | --- | --- | --- |
| Calculators | `crcl`, `bsa`, `pediatric`, `loading`, `halflife`, `auc`, `pd-emax`, `units` (8) | — (client-side TS) | TS ports of `pharma.pk`/`pharma` formulas; `UnitConverterParity` pins the converter | `test_web_parity.py` (74, unit-converter + contracts), E2E ritual | **B** — TS cards render+interact under E2E; only the unit converter has an automated TS↔Python parity pin; the clinical formulas share the spec, not a test |
| PK / PD Explorer | `pk-regimen`, `pk-profile`, `pk-pd-effect`, `pk-nca` (4) | `ml/dataset`, `ml/dataset/csv` (data plumbing) | `pkpd.*` engines via TS simulation (`src/lib/pk-pd.ts`) | `test_pkpd_engines.py` (99) pins the engines; session-report contracts | **B** — same TS-parity structure as calculators |
| Machine Learning | `ml-setup`, `ml-calibration` (2) + post-run cards (results/leaderboard/comparison/tuning — deliberately not in the finder registry) | `ml/train`, `ml/compare`, `ml/tune`, `ml/ad`, `ml/shap`, `ml/learning-curve` (6) | `ml` core via bridges; SHAP via `scripts/shap_bridge.py` → `ml.explain` | `test_ml_pipeline.py` (211), `test_explain.py` (115), `TestApplicabilityDomainParity`, `TestLearningCurveContract`, `TestAcquisitionParity`, `TestHyperbandRungFold` | **A** |
| Biology | `bio-biomarkers` (1) + post-run matrix cards | `biology/biomarkers` (1) | `biology` biomarker discovery | `test_biology.py` (93), `TestVariableGenesParity` | **A** |
| Docking | `dock-parse`, `dock-torsion`, `dock-vina-scan`, `dock-rescore` (4) | `docking/parse`, `docking/prep`, `docking/rescore` (3) | `docking.parsing/prep/rescore` via bridges | `test_docking.py`, `test_docking_prep.py` (84), `test_pose_rmsd.py` (48), `TestDockingConfigsParity` | **A** |
| Cheminformatics | `feat-factory` (1) | `chem/features`, `chem/sa-score` (2) | `chem.features`, `chem.sa_scoring` via bridges | `test_features.py` (119), `test_sa_score.py` | **A** |
| Ligand prep | `ligand-prep`, `ligand-batch` (2) | (rides `docking/prep`) | `docking.ligands` GPF/DPF writers | `test_docking_prep.py` (84) | **A** |
| Reconciler | `ligand-sets` (1) | — | TS reconciliation over uploaded sets | session-report contracts + E2E | **B** — TS-side, E2E-verified only |
| Generative | `gen-topology`, `gen-breed`, `gen-sa-score` (3) | `gen/surgery` (1) | `generative` surgery via `scripts/surgery_bridge.py` | `test_generative.py` (291) subsets, `TestSurgeryBridge`, `TestSurgeryWebWiring` | **A** |
| Molecular Dynamics | `md-xvg`, `md-traj`, `md-hbond`, `md-saltbridge`, `md-cv`, `md-eco` (6) | `md/trajectory`, `md/hbond`, `md/saltbridge`, `md/cv`, `md/xvg` (5) | `md.*` via `scripts/md_bridge.py` (5 ops) | `test_md_analysis.py` (63), `test_md_cv.py` (65), `TestMdCvBridge`, `TestMdCvWebWiring` | **A** — the newest card (`md-cv`, r258) has the deepest bridge pins |
| Session report | `session-report` (1, above the tab strip) | `report/session` (1) | `viz`/report bundle via `scripts/report_bridge.py` | `TestSessionReport*` (7 classes) + preview-contents census | **A** |
| Toolkit internals | — (download panel, API reference, file tree) | `toolkit/download`, `toolkit/file`, `toolkit/reference`, `toolkit/tree`, `toolkit/zip-size` (5) + root health | zip homes, `toolkit-fs.ts` | zip endpoint exact:true pinned every round; `test_requirements.py` | **A** |

## 3. The backend library — module-by-module

Line/file counts are live `wc` over `src/pharmaml/<pkg>/*.py` at the freeze
commit. "GUI" = Studio page(s); "Web" = card(s)/route(s) from §2.

| Package | Size | Feature headline | GUI | Web | Tests | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `chem` | 22 files / 19,957 L | standardize, descriptors (incl. RISM opt-in), tables, similarity, diversity, SELFIES, SA scoring, oracles | Chemistry | `feat-factory`, `gen-sa-score` | `test_chem_*`, `test_descriptors.py` (73), `test_selfies_codec.py`, `test_sa_score.py` | **A** |
| `ml` | 69 files / 67,251 L | registry (30+ estimator families), train/evaluate, persistence, CV diagnostics, HPO/multiobjective, XAI, drugresponse, unsupervised/manifold, adversarial, privacy, recommend, apriori, annsearch, catalog, runboard, umap, curation | ML Workbench, Model Analysis, Foundations, Best-of ML, Response AI, Deep Learning, Unsupervised | `ml/*` cards + routes | `test_ml_pipeline.py` (211), `test_tuner.py` (118), `test_explain.py` (115), `test_hpo.py` (64) + 20 more suites | **A** (except the §4 C-list) |
| `viz` | 22 files / 16,570 L | publication plots, interactive (express-layer), MD plots, export bundle, PDF reports | every page; Utilities | session report | `test_viz.py` (221), `test_viz_md_plots.py`, `test_export_images.py` | **A** |
| `generative` | 16 files / 9,365 L | graph surgery, ring topology moves, breeding, MCTS search, SA-constrained generation | Generative | `gen-*` cards | `test_generative.py` (291), `test_mcts.py` (12) | **A** |
| `pharma` | 19 files / 12,545 L | PK summaries, dosing design, unit conversions, pharmaceutical calculators | PK/PD, Utilities | calculators (TS ports) | `test_pk.py` (49), `test_dosage.py` (24), `test_units.py` (25) | **A** |
| `docking` | 11 files / 7,538 L | ligand/protein prep (GPF/DPF), output parsing, torsion QA, consensus rescore, DOCKSTRING benchmark boxes | Docking, Utilities | `dock-*` cards | `test_docking*.py`, `test_pose_rmsd.py` (48) | **A** (engine: class **F**) |
| `pkpd` | 15 files / 7,201 L | compartment ODEs, events, NCA, PD engines | PK/PD | PK/PD Explorer | `test_pkpd_engines.py` (99) | **A** |
| `biology` | 6 files / 3,508 L | cell-line response tables, biomarker discovery, variable genes | Biology, Image Lab, Response AI | `bio-biomarkers` | `test_biology.py` (93) | **A** |
| `md` | 11 files / 3,683 L | synthetic trajectories, RMSD/RMSF, H-bond & salt-bridge timelines, PLUMED-style CVs, XVG parsing | Molecular Dynamics | 6 `md-*` cards, 5 routes | `test_md_analysis.py` (63), `test_md_cv.py` (65) | **A** (data: synthetic — §6) |
| `utils` | 7 files / 3,185 L | io, checkpoints, conversions | all | — | `test_io.py` (69), `test_checkpoints.py` | **A** |
| top-level | 9 files | `cli.py` (16 command groups), `data.py`/`data_profiling.py`/`profiling.py`, `config`, `exceptions`, `logger` | Foundations, Utilities | — | `test_cli_features.py` (96), `test_data_*`, `test_format_converters.py` (50) | **A** |

## 4. Backend-only modules — class C (no GUI integration, by honest grep)

These modules have real implementations and real tests, and **no** Studio
page, web card, or API route references them. They are importable library
surface (and CLI where noted) — listed here instead of being papered over:

| Module | Tests | What it does | Why no GUI |
| --- | --- | --- | --- |
| `ml/gnn.py` + `ml/gnnxai.py` | `test_gnn.py` (48) | graph neural network featuriser/trainer + its XAI registry | absorbed r2xx; no card built yet |
| `ml/geometric.py` | `test_geometric.py` (36) | geometric-deep-learning primitives | same |
| `ml/pruning.py` | `test_pruning.py` (48) | magnitude/structured network pruning | Deep Learning page stops at train/runboard/umap/curation/xai |
| `ml/quantization.py` | `test_quantization.py` (50) | int8/fp16-style quantisation arithmetic | same |
| `ml/harness.py` | `test_harness.py` (56) | offline sharded-training harness (ZeRO stages + early stopping) | same |

Internal plumbing that is tested but intentionally not user-facing(classified under its parent feature, listed for honesty):
`ml/split_categorical` (stratified splits), `ml/scoring_transforms`,
`ml/catalog` fold-ins, `ml/harness` early-stopping callbacks.

## 5. Optional-dependency and external-engine boundaries — class F

Everything in the pinned stack (`requirements.txt`) is installed in the
shipped environment and tested green; these rows are about what happens
**without** the dependency or engine, so users are never surprised:

| Feature | Gate | Behaviour when absent |
| --- | --- | --- |
| `xgboost` / `lightgbm` / `catboost` families | optional imports in `ml/models.py` | clean `OptionalDependencyError`; registry skips the family |
| Torch DNN / residual DNN | optional import in `ml/neural.py` | clean refusal; the numpy MLP remains available |
| SHAP explanations | `ml/explain.py` + `scripts/shap_bridge.py` | clean refusal; permutation importances remain |
| `meeko`/`gemmi` prep paths | `docking/receptor.py`, `chem/standardize.py` | fallback parsers; prep card reports the limitation |
| **Docking execution itself** | external Vina binary + receptor PDBQT (never bundled) | the toolkit ships prep/parse/rescore/consensus; **scores are user or external-engine inputs** — a target-table entry is not a docking result (the `dockstring.py` docstring's own contract) |
| **Real MD trajectories** | external GROMACS/PLUMED (never bundled) | analysis runs on the deterministic synthetic demos; XVG/COLVAR parsers accept real exports when the user supplies them |
| DOCKSTRING 260k-ligand dataset | external download | only the 58 target boxes + benchmark objectives ship |

## 6. Synthetic-demo register (honesty rule 3)

These features are fully implemented and tested, **and** their shipped
demonstrations are synthetic — none of them claims real experimental or
production data support:

- **MD trajectories** (`md/trajectory.py`): deterministic helix/ligand demos;
  the salt-bridge demo *is* the distance-only criterion, stated as such.
- **Cell-line response tables**: designed demo matrices, loadable real CSVs.
- **Docking scores**: user inputs throughout; the benchmark boxes carry no
  results.
- **Generative oracles**: in-house synthetic objectives (SA/QED-style),
  not wet-lab assays.

## 7. CLI surface (8 command groups, 36 subcommands, 96 tests)

`models`, `data inspect|split|convert`, `ml train|compare|check|tune|ad`,
`chem pic50|descriptors|rism|compare|merge`, `pharma pk|dose|approvals`,
`docking scan|check|prep-ligands|prep-protein|dpf|collect|torsion|gpf-scan|
vina-scan|rescore` (+ `campaign list|distribute|cleanup|delete|reindex`),
`biology audit|top`, `generative run|corpus` — pinned by
`test_cli_features.py`.

## 8. Notebooks

8 executed notebooks (7 `complete_*_workflow` + `pharmaml_walkthrough`) with
7 clean no-output twins, re-executed at every release (the r260 OOM ritual:
dev server down, foreground run, error-outputs=0, home-path-survivors=0).

## 9. Status summary

Counted over the matrix rows above (18 Studio pages, 12 web areas,
11 backend packages, 5 backend-only modules):

| Status | Count | What |
| --- | --- | --- |
| **A** | 34 | implemented and tested end-to-end |
| **B** | 7 | implemented, incompletely tested — dashboard + settings (smoke only), foundations + responseai (page-suite-only backends), TS calculators + PK/PD explorer + reconciler (no automated TS↔Python parity pins) |
| **C** | 5 | backend-only: gnn(+gnnxai), geometric, pruning, quantization, harness |
| **D** | 1 | the §5 nine-card workflow dashboard (a stat dashboard exists; the nine workflow-entry cards are the Phase-3 refinement) |
| **E** | 1 | conformal prediction + coverage diagnostics + split audits (PR #9 parked, unmerged — its arena branch carries the work) |
| **F** | 3 | docking execution (external Vina), real MD data (external GROMACS/PLUMED), DOCKSTRING dataset (external download) |

**Maintenance rule**: this file must be re-trued whenever a page, card,
route, or backend module is added or retired — the
`tests/test_feature_inventory.py` suite fails the release if the matrix
drifts from the live registries (18 pages, 33 cards, 27 routes, the C-list,
the six legend classes).
