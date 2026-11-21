# Feature inventory — the truthful traceability matrix

This is the §4.5 deliverable of the MASTER_PROMPT part 4 specification:
**every major feature classified honestly, with a feature-to-backend-to-UI-to-test
traceability matrix.** All evidence on this page was collected live against the
working tree (imports grepped, pages/cards/routes enumerated, tests collected
with `pytest --collect-only`), not copied from prose. This is a historical
inventory, not a current certification: counts and grades have not been
revalidated against every current UI and dependency profile. See
[Part 5 audit](PART5_PHASE0_AUDIT.md) and [test profiles](TEST_PROFILES.md)
for scoped evidence and engine limitations. Matrix inputs were frozen
at v1.2.167 (`9130c6b`); the inventory itself ships with v1.2.240 (the
r315 Phase-4 search/job-center/workspace rows and the Phase-5 glossary/
presets/doctor rows are registry-level additions, not matrix changes;
r312 added the `/api/v1/model-evaluation` conformal + coverage family and
the `/api/v1/dl/gnn` API lane — registry-level additions, not matrix
changes; r313 added the GNN GUI card in the Deep Learning workspace (the
frontend lane over the `/api/v1/dl/gnn` API) — the component-test count
moves with it, not the matrix;
r314 added the ADMET lane — the `/api/v1/admet` registry + scorecard
family over the vendored `ml/admet.py` backend plus the Machine Learning
workspace's "ADMET Profile" stage (both legs in one round, 8 route
tests + 7 component tests) — a registry-level addition, not a matrix
change;
r315 added the drug–target interaction lane — the `/api/v1/dta`
registry + paired-featurization family over the in-tree `ml/dta.py`
backend (DeepDTA's verbatim alphabets + the ECFP/AAC pairing) plus the
Machine Learning workspace's "Drug–Target Interaction" stage (both
legs in one round, 10 route tests + 8 component tests) — a
registry-level addition, not a matrix change;
r321 added the multi-objective optimization lane — the
`/api/v1/multiobjective` registry + optimize family over the in-tree
`ml/multiobjective.py` backend (the r319-gated API leg: the Pareto
front, the exact 2-D hypervolume indicator, NSGA-II's crowding, the
MOEA/D Tchebycheff decomposition driving both samplers) plus the
Machine Learning workspace's "Multi-objective Optimization" stage
(the r321 GUI leg, 8 component tests) — a registry-level addition,
not a matrix change;
r266 arena absorption landed the former E-row conformal/coverage/split-audit
suite from `arena/8696775e`; the r265 §24.3/24.4 rows were trued live;
the r271 Response AI web twin closed the last Studio-only row — census
35 cards / 29 routes; the r285/r286 rounds flipped the §11.5 Molecular
Docking and §11.6 PK/PD workspaces from shell to connected workflow
(7 of 9 rows) — registry flips, not matrix changes; the r272 round added the part9 Windows-installer
packaging suite under `packaging/windows/`, documented in its own
BUILD_REPORT, not as a GUI matrix row; the r273 round verified that
packaging policy for real by freezing the app on the Linux host —
`tools/freeze_equivalent.py` + `tools/freeze_smoke_driver.py`, the
evidence under `packaging/windows/evidence/`; the r289 round closed the
Phase Sixteen audit's one FAIL row — the §14.1 startup gate
(`services/api/startup.ts` + `app/StartupGate.tsx`: the whole shell
mounts inside the gate only after the handshake → health → WS-open
sequence, with 8 drift guards and the 12/12 no-injection browser
verification), an integration-level fix, not a matrix row; the r304 round completed the
Tk playground's true retirement — `playground.py` and its 753-test suite
left the shipped tree (the v1.2.141 declaration, finally executed) and
the r304 architecture-hygiene guard (`tests/test_architecture_hygiene.py`,
472 tests: the duplicate-symbol map, the library-surface import graph,
the mabel/noael parity grid, the SBOM artifact pins) re-trued the
collection to 8,362 — no matrix row changed: the retired surfaces were
never matrix rows). The §5
nine-card mapping (§10 below) was
derived live at v1.2.169.

For later incremental data-audit, uncertainty, validation, provenance, and
checkpoint-inspection entry points, see [Research workflows](RESEARCH_WORKFLOWS.md).
The matrix below remains a historical snapshot.

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
| Dashboard (`dashboard`) | `pharmaml` meta, `ml.models` registry, `ui.nine_cards`, `ui.widgets.workflow_guide` | `test_nine_cards.py` (26) + `test_workflow_guide.py` (26) + `test_workflow_guide_lint.py` (14) + `test_nine_card_first.py` (23) | — | **A** | Statistics panel + the §5 nine-card workflow-entry grid (exact directive order, live dependency probes, external-engine honesty, real `go_to` routing) + the Phase-3 polish since v1.2.170: the responsive 3/2/1 re-flow and the real screening-library SMILES preview on the Datasets card. |
| Data Explorer (`data`) | `utils.io` | `test_data_table.py` (6) | `test_io.py` (69), `test_data_loading.py`, `test_data_provenance.py` | **A** | Page-level depth is the data-table widget; the import/export library side carries 69 focused tests. |
| Chemistry Workbench (`chemistry`) | `chem.csv_standardize`, `chem.features`, `utils.io` | `test_chemistry_page.py` (13) | `test_chem_*`, `test_descriptors.py` (73), `test_csv_standardize.py` (57), `test_similarity.py`, `test_diversity.py` | **A** | RISM descriptors are opt-in (see §5). |
| PK/PD Workbench (`pkpd`) | `pharma.pk`, `pkpd.compartments/events/nca_ext/ode` | `test_pkpd_page.py` (18) | `test_pk.py` (49), `test_pd.py`, `test_pkpd_engines.py` (99), `test_dosage.py` (24) | **A** | |
| Biology Workbench (`biology`) | `biology` (incl. `load_response_table`) | `test_biology_page.py` (13) | `test_biology.py` (93), `test_cellline_examples.py`, `test_variable_genes` parity | **A** | Demo cell-line data is synthetic-but-realistic (see §6). |
| Docking Workbench (`docking`) | `docking` | `test_docking_page.py` (17) | `test_docking.py`, `test_docking_prep.py` (84), `test_docking_contacts.py` (55), `test_pose_rmsd.py` (48) | **A** | Prep/parse/rescore only — no bundled docking engine (class **F** note in §5). The Contacts & PLIF tab ships the pose-level reading: contact geometry + interaction fingerprints (r264). |
| Model Analysis (`models`) | `ml.cv_results`, `ml.models`, `ml.training` | `test_model_analysis_page.py` (13) + `test_njobs_field.py` (TestModelAnalysisNJobs, 3) | `test_cv_results.py`, `test_ml_pipeline.py` (211) | **A** | part8 round 4: the manual n_jobs field feeds train_model and learning_curve (previously hard-pinned serial); invalid input falls back to serial with a shell toast. |
| ML Workbench (`mlworkbench`) | `ml` core, `ml.evaluation`, `ml.persistence`, `ml.tuner`, `ml.conformal`, `ml.calibration`, `ml.decision`, `ml.drift`, `ml.comparison`, `ml.multiplicity`, `ml.power`, `ml.coverage_diagnostics`, `ml.subgroups`, `ml.scaffold_coverage` (the part13 §9.3 scaffold-aware coverage bridge, r311 — backend-side; the Statistical tools tab's group selector stays manual), `utils.checkpoints` | `test_ml_workbench_page.py` (19) + `test_ml_statistical_tools.py` (11) + `test_active_learning_tab.py` (8) + `test_ml_statistics_batch2.py` (13) | `test_ml_pipeline.py` (211), `test_tuner.py` (118), `test_conformal.py`, `test_calibration.py`, `test_decision.py`, `test_drift.py` | **A** | Stacking/voting exposed; part8 round 4: the family benchmark takes the manual n_jobs field (previously serial) with the serial-fallback toast; round 5 added the §14.1 Active learning tab (pool holdout → six query strategies → ranked candidates with rationale → CSV export → measured-label import → retrain, no fabricated labels); the v1.2.187–188 Statistical tools tab (conformal intervals, calibration report, decision curve, drift detection, comparison test, p-value correction, power report, coverage by group, subgroup audit) + the v1.2.188 model-card writer (model_card module); boosting families optional-gated (§5). r312: the `/api/v1/model-evaluation` family (part13 Phase 4's queued API connection CLOSED — `conformal-regression` + `conformal-classification` (split | label-conditional) + `coverage-diagnostics` (groups | scaffold-aware) over the session model store, sync-only per the light-compute precedent, 17 route tests). |
| Unsupervised (`unsupervised`) | `ml.manifold`, `ml.unsupervised`, `utils.io` | `test_unsupervised_page.py` (23) | `test_unsupervised.py`, `test_manifold.py`, `test_hierarchical.py` | **A** | |
| Deep Learning (`deep_learning`) | `ml.neural`, `ml.numpy_neural`, `ml.runboard`, `ml.curation`, `ml.pruning`, `ml.quantization`, `ml.harness`, `ml.geometric`, `ml.gnn`, `ml.gnnxai` | `test_deep_learning_page.py` (27) + `test_dl_efficiency_lab.py` (6) | `test_neural.py` (95), `test_numpy_neural_checkpoints.py`, `test_runboard.py`, `test_umap.py`, `test_curation.py`, `test_pruning.py` (48), `test_quantization.py` (50), `test_harness.py` (56), `test_geometric.py` (36), `test_gnn.py` (48), `tests/api/test_model_evaluation.py::TestGnnApiLane` (13, r312) | **A** | Torch optional-gated (§5). The v1.2.189 Efficiency lab (pruning + quantization + ZeRO planner) and GNN lab (motif train + explain) closed the Deep Learning cluster. r312: the GNN API lane — `POST /api/v1/dl/gnn-train` (inline graphs or the motif fixture, sync/job) + `POST /api/v1/dl/gnn-predict` (the graph models' own prediction path; `/api/v1/ml/predict` + `/ml/explain` refuse graph models up front with the pointer). r313: the GNN GUI card — the graph-lane section in the Deep Learning workspace over the same API (the live conv/pool vocabulary off the catalog, the synthetic/inline source switch with client-side validation, the job-mode COMPLETED relay, the predict lane with the class-probability columns; 7 component tests, `GnnLane.test.tsx`). |
| Generative Workbench (`generative`) | `generative`, `chem.oracles`, `chem.sa_scoring` | `test_generative_page.py` (31) | `test_generative.py` (291), `test_mcts.py` (12), `test_sa_score.py`, `test_generative_sa_cli.py` | **A** | The MCTS search rides the web twin too since v1.2.214 (the `gen-mcts` card over `scripts/mcts_bridge.py`). |
| Foundations (`foundations`) | `ml.stats`, `ml.inferential`, `profiling` | `test_foundations_page.py` (13) — end-to-end (`_start_stats()` runs the real Welch ANOVA / Tukey / Games-Howell / `profile_dataset`) | none dedicated | **B** | No focused library suite for `ml.stats`/`ml.inferential`/`profiling`; the 13 page tests are the only direct pins. |
| Best-of ML (`bestofml`) | `ml.adversarial/annsearch/apriori/catalog/privacy/recommend` | `test_bestofml_page.py` (19) | `test_campaign.py` (111), `test_catalog_foldins.py` (27), plus per-module suites | **A** | Six absorbed engines, four cards each. |
| Response AI (`responseai`) | `ml.drugresponse` | `test_responseai_page.py` (17) — end-to-end hill-fit/metrics/rank over the real module (the module's first tests rode the page) | `test_drugresponse.py` — the focused headless suite | **A** | The web twin rides since v1.2.215: the `ml-drugresponse` card over `scripts/drugresponse_bridge.py` (hill / metrics+rank / model / cv, the page's four headline tabs; the deeper diagnostics stay desktop-side). The dedicated backend suite closed the old B note. |
| Image Lab (`imagelab`) | `biology` imaging, `chem.descriptors`, `chem.similarity`, `viz.export` | `test_image_lab_page.py` (18) + `test_image_1to1_246.py` (16) + `test_image_subpixel_245.py` (16) + `test_image_hover_probe.py` (20) | `test_export_images.py`, `test_viz.py` (221) | **A** | Pixel-policy pins are first-class tests. |
| Molecular Dynamics (`molecular_dynamics`) | `md`, `viz` | `test_molecular_dynamics_page.py` (3) | `test_md_analysis.py` (63), `test_md_cv.py` (65), `test_md_plots.py` + `test_viz_md_plots.py` | **A** | Page-test depth is 3 but the feature's backend+web coverage is the deepest in the toolkit (5 bridge ops, 5 cards). Synthetic trajectories (§6). |
| Utilities (`utilities`) | `chem.converters`, `docking` collect, `pharma`, `utils.io`, `viz`, §17A/B `studio.pipelines` (workflow templates + sequential engine), §17C `studio.comparison`, §17D `studio.reports` | `test_utilities_page.py` (19), `test_design_tokens.py` (20), `test_font_loader.py` (8), `test_mono_adoption.py` (8), `test_marker_discipline.py` (9), `test_theme_screenshots.py` (7), `test_biology_expression_path.py` (8), `test_focus_visibility.py` (13), `test_chart_export.py` (12) + `test_workflow_templates.py` (13) + `test_experiment_compare.py` (4) + `test_run_report.py` (3) + `test_resource_monitor.py` (6) | `test_converters.py` (74), `test_format_converters.py` (50), `test_campaign.py` (111), `test_approvals.py` (15), `test_workflow_pipelines.py` (19), `test_experiment_comparison.py` (10), `test_run_reports.py` (8), `test_report_figures.py` (5), `test_workspace_gaps.py` (5) | **A** | Campaign + approvals management live here; round 24 added the §17A workflow-template tab (7 chains) and the §17B sequential pipeline engine; round 25 added the §17C experiment comparison (Job Center multi-select → compatibility groups, cross-condition pairs refused) and the §17D scientific run report (eight sections, units + sig-figs, no auto-conclusions); round 26 closed §17: the §17E live resource monitor in the Job Center, card-integrated forms for all seven templates (every chain runnable from the tab), and the §17G favorites/export-folder-recall/layout-prefs workspace extensions; round 27 embedded exported PNG figures into the §17D run reports (HTMLReport.add_image — the file's own bytes, missing files stay honest text lines), auto-listed the response screen's compound values, auto-detected SMILES columns from any CSV header, and restored the utilities last-opened tab from the layout preference.; round 28 landed the part7 visual identity: the studio theme provider (QSS generated from ui/design_tokens.py — Deep Navy + Warm Ivory + Electric Lime, light and dark, studio/light the fresh-install default with persisted choices preserved), the bundled Inter + JetBrains Mono (SIL OFL) with full fallback chains, and the Okabe-Ito scientific chart palette wired through ChartPalette.; round 29 completed the identity's adoption: set_mono() pins JetBrains Mono on the technical widgets (SMILES input, Vina box config, four audit/SAR log panes) with the three-provider font contract pinned in tests, PlotPanel enforces marker discipline (flagged low-contrast light-theme series gain symbols/shapes, keyed by a per-panel counter that resets on clear), screenshot regression verifies both themes at the pixel level (real offscreen grabs: ivory canvas + white cards + navy status in light; midnight + elevated surface in dark), and the biology page's MAD + biomarker tabs share one expression-matrix path row (bundled default, Browse, live preview, cache invalidated on change).; round 30 closed the loop: the Ligand Sets reconciliation now rides the session report (a ligands group with the bucket-count metrics card, a bar-trace comparison figure through the report bridge, and per-bucket name tables capped at 50 with an honest +N confession), keyboard focus is visible on every focusable widget class across all nine workspaces (16 accent-colored :focus rules in the studio QSS, audited by test), and every chart-bearing page carries an Export chart… button wired to the §17 raster engine at the 640×480 legibility floor. |
| Settings (`settings`) | (none — Qt/theme/app settings only) | smoke only | `test_theme_manager.py` (15), `test_app_settings` via main-window tests | **B** | No backend surface by design; theme/settings logic is tested, the page itself is smoke-only. |

## 2. The web console — 6 sub-tabs, 35 always-rendered cards, 29 API routes

Cards are the canonical `PLAYGROUND_CARDS` registry
(`src/components/explorer/playground/card-directory.ts`, 35 entries, pinned
by `TestCardFinderParity` + the E2E probe that walks every entry). Routes are
the live `src/app/api/**/route.ts` set. Backend columns trace through the
`scripts/*_bridge.py` processes to the real `pharmaml` functions.

| Area | Cards | Routes | Backend | Tests | Status |
| --- | --- | --- | --- | --- | --- |
| Calculators | `crcl`, `bsa`, `pediatric`, `loading`, `halflife`, `auc`, `pd-emax`, `units` (8) | — (client-side TS) | TS ports of `pharma.pk`/`pharma` formulas; `UnitConverterParity` pins the converter | `test_web_parity.py` (74, unit-converter + contracts), E2E ritual | **B** — TS cards render+interact under E2E; only the unit converter has an automated TS↔Python parity pin; the clinical formulas share the spec, not a test |
| PK / PD Explorer | `pk-regimen`, `pk-profile`, `pk-pd-effect`, `pk-nca` (4) | `ml/dataset`, `ml/dataset/csv` (data plumbing) | `pkpd.*` engines via TS simulation (`src/lib/pk-pd.ts`) | `test_pkpd_engines.py` (99) pins the engines; session-report contracts | **B** — same TS-parity structure as calculators |
| Machine Learning | `ml-setup`, `ml-calibration`, `ml-drugresponse` (3) + post-run cards (results/leaderboard/comparison/tuning — deliberately not in the finder registry) | `ml/train`, `ml/compare`, `ml/tune`, `ml/ad`, `ml/shap`, `ml/learning-curve`, `ml/drugresponse` (7) | `ml` core via bridges; SHAP via `scripts/shap_bridge.py` → `ml.explain`; drug-response learning via `scripts/drugresponse_bridge.py` → `ml.drugresponse` | `test_ml_pipeline.py` (211), `test_explain.py` (115), `test_drugresponse.py`, `TestApplicabilityDomainParity`, `TestLearningCurveContract`, `TestAcquisitionParity`, `TestHyperbandRungFold`, `TestDrugResponseBridge`, `TestDrugResponseWebWiring` | **A** |
| Biology | `bio-biomarkers` (1) + post-run matrix cards | `biology/biomarkers` (1) | `biology` biomarker discovery | `test_biology.py` (93), `TestVariableGenesParity` | **A** |
| Docking | `dock-parse`, `dock-torsion`, `dock-vina-scan`, `dock-rescore` (4) | `docking/parse`, `docking/prep`, `docking/rescore` (3) | `docking.parsing/prep/rescore` via bridges | `test_docking.py`, `test_docking_prep.py` (84), `test_pose_rmsd.py` (48), `TestDockingConfigsParity` | **A** |
| Cheminformatics | `feat-factory` (1) | `chem/features`, `chem/sa-score` (2) | `chem.features`, `chem.sa_scoring` via bridges | `test_features.py` (119), `test_sa_score.py` | **A** |
| Ligand prep | `ligand-prep`, `ligand-batch` (2) | (rides `docking/prep`) | `docking.ligands` GPF/DPF writers | `test_docking_prep.py` (84) | **A** |
| Reconciler | `ligand-sets` (1) | — | TS reconciliation over uploaded sets | session-report contracts + E2E | **B** — TS-side, E2E-verified only |
| Generative | `gen-topology`, `gen-breed`, `gen-sa-score`, `gen-mcts` (4) | `gen/surgery`, `gen/mcts` (2) | `generative` surgery + the ChemTSv2-style `MctsOptimizer` search via `scripts/surgery_bridge.py` + `scripts/mcts_bridge.py` (real RDKit + torch on the server) | `test_generative.py` (291) subsets, `test_mcts.py` (12), `TestSurgeryBridge`, `TestSurgeryWebWiring`, `TestMctsBridge`, `TestMctsWebWiring` | **A** |
| Molecular Dynamics | `md-xvg`, `md-traj`, `md-hbond`, `md-saltbridge`, `md-cv`, `md-eco` (6) | `md/trajectory`, `md/hbond`, `md/saltbridge`, `md/cv`, `md/xvg` (5) | `md.*` via `scripts/md_bridge.py` (5 ops) | `test_md_analysis.py` (63), `test_md_cv.py` (65), `TestMdCvBridge`, `TestMdCvWebWiring` | **A** — the newest card (`md-cv`, r258) has the deepest bridge pins |
| Session report | `session-report` (1, above the tab strip) | `report/session` (1) | `viz`/report bundle via `scripts/report_bridge.py` | `TestSessionReport*` (7 classes) + preview-contents census | **A** |
| Toolkit internals | — (download panel, API reference, file tree) | `toolkit/download`, `toolkit/file`, `toolkit/reference`, `toolkit/tree`, `toolkit/zip-size` (5) + root health | zip homes, `toolkit-fs.ts` | zip endpoint exact:true pinned every round; `test_requirements.py` | **A** |

## 3. The backend library — module-by-module

Line/file counts are live `wc` over `src/pharmaml/<pkg>/*.py` at the freeze
commit. "GUI" = Studio page(s); "Web" = card(s)/route(s) from §2.

| Package | Size | Feature headline | GUI | Web | Tests | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `chem` | 22 files / 19,957 L | standardize, descriptors (incl. RISM opt-in), tables, similarity, diversity, SELFIES, SA scoring, oracles, the r303 chirality toggle (`compute_fingerprints(include_chirality=…)` morgan-only, default OFF — byte-identical history; `FeatureConfig.include_chirality` recorded in metadata; the `/api/v1/chemistry/fingerprints` field + the Datasets fingerprint-card toggle) | Chemistry | `feat-factory`, `gen-sa-score` | `test_chem_*`, `test_descriptors.py` (73+8 chirality), `test_selfies_codec.py`, `test_sa_score.py` | **A** |
| `ml` | 69 files / 67,251 L + the r303 Williams lane (`ml/applicability.py::leverage_williams` — the hat-matrix leverage + standardized-residual band, regression-only, 23 unit tests; `/api/v1/applicability-domain/leverage-williams` job/sync + the plot_williams artifact; the React/Tauri ML workspace's AD stage carries the live Williams lane) | registry (30+ estimator families), train/evaluate, persistence, CV diagnostics, HPO/multiobjective, XAI, drugresponse, unsupervised/manifold, adversarial, privacy, recommend, apriori, annsearch, catalog, runboard, umap, curation | ML Workbench, Model Analysis, Foundations, Best-of ML, Response AI, Deep Learning, Unsupervised | `ml/*` cards + routes | `test_ml_pipeline.py` (211), `test_tuner.py` (118), `test_explain.py` (115), `test_hpo.py` (64) + 20 more suites | **A** (except the §4 C-list) |
| `viz` | 22 files / 16,936 L | publication plots, interactive (express-layer), MD plots, docking contact-map/PLIF twins, SA-score distributions, export bundle, PDF reports | every page; Utilities | session report | `test_viz.py` (229), `test_viz_md_plots.py`, `test_export_images.py`, `test_generative_result_sa.py` (viz rows) | **A** |
| `generative` | 16 files / 9,682 L | graph surgery, ring topology moves, breeding, MCTS search, SA-constrained generation, the raw-SA results lens (`sa_frame` / `filter_by_sa` / `sa_summary`) | Generative | `gen-*` cards | `test_generative.py` (291), `test_mcts.py` (12), `test_generative_result_sa.py` (31) | **A** |
| `pharma` | 19 files / 12,545 L | PK summaries, dosing design, unit conversions, pharmaceutical calculators | PK/PD, Utilities | calculators (TS ports) | `test_pk.py` (49), `test_dosage.py` (24), `test_units.py` (25) | **A** |
| `docking` | 12 files / 8,266 L | ligand/protein prep (GPF/DPF), output parsing, torsion QA, consensus rescore, DOCKSTRING benchmark boxes, contact geometry + PLIF fingerprints | Docking, Utilities | `dock-*` cards | `test_docking*.py`, `test_docking_contacts.py` (55), `test_pose_rmsd.py` (48) | **A** (engine: class **F**) |
| `pkpd` | 15 files / 7,201 L | compartment ODEs, events, NCA, PD engines | PK/PD | PK/PD Explorer | `test_pkpd_engines.py` (99) | **A** |
| `biology` | 6 files / 3,508 L | cell-line response tables, biomarker discovery, variable genes | Biology, Image Lab, Response AI | `bio-biomarkers` | `test_biology.py` (93) | **A** |
| `md` | 11 files / 3,683 L | synthetic trajectories, RMSD/RMSF, H-bond & salt-bridge timelines, PLUMED-style CVs, XVG parsing | Molecular Dynamics | 6 `md-*` cards, 5 routes | `test_md_analysis.py` (63), `test_md_cv.py` (65) | **A** (data: synthetic — §6) |
| `utils` | 7 files / 3,185 L | io, checkpoints, conversions | all | — | `test_io.py` (69), `test_checkpoints.py` | **A** |
| top-level | 9 files | `cli.py` (16 command groups), `data.py`/`data_profiling.py`/`profiling.py`, `config`, `exceptions`, `logger` | Foundations, Utilities | — | `test_cli_features.py` (96), `test_data_*`, `test_format_converters.py` (50) | **A** |

## 4. Backend-only modules — class C (no GUI integration, by honest grep)

These modules have real implementations and real tests, and **no** Studio
page, web card, or API route references them. They are importable library
surface (and CLI where noted) — listed here instead of being papered over.
**EMPTY since v1.2.190** — every backend module has a GUI workflow
(`ml/conformal.py` left in v1.2.187, the ML statistics cluster in
v1.2.188, the Deep Learning cluster in v1.2.189, and v1.2.190's batch-4
closer: the classification conformal card on the Workbench's Statistical
tools tab plus the split-overlap audit and SAR matched-pairs panels on
the Data Explorer).

| Module | Tests | What it does | Why no GUI |
| --- | --- | --- | --- |
| *(none — the class-C table is empty)* | | | |

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
| **Docking execution itself** | compatible Vina runtime + user-supplied receptor PDBQT | the source/ZIP bundles Windows engines; native builds or Wine may be needed; Studio focuses on prep/parse/rescore/consensus; **scores are user or engine-generated inputs** — a target-table entry is not a docking result (the `dockstring.py` docstring's own contract) |
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
11 backend packages, 0 backend-only modules):

| Status | Count | What |
| --- | --- | --- |
| **A** | 46 | implemented and tested end-to-end (the §5 nine-card workflow dashboard rides the Dashboard row — see §10; A grows by three in v1.2.190 as the last class-C modules ride the ML Workbench and Data Explorer rows) |
| **B** | 6 | implemented, incompletely tested — settings (smoke only), foundations + responseai (page-suite-only backends), TS calculators + PK/PD explorer + reconciler (no automated TS↔Python parity pins) |
| **C** | 0 | backend-only — none since v1.2.190 (see §4: the class-C table is empty) |
| **D** | 0 | nothing currently classified **D** — the nine-card workflow dashboard moved to **A** in v1.2.169 (§10) |
| **E** | 0 | nothing currently classified **E** — the conformal/coverage/split-audit suite landed in v1.2.173 (the `arena/8696775e` absorption, formerly "PR #9 parked") and moved to **C** |
| **F** | 3 | docking execution (external Vina), real MD data (external GROMACS/PLUMED), DOCKSTRING dataset (external download) |

## 10. The nine-card dashboard mapping (§5 of the directive)

Shipped v1.2.169: the Studio home renders the directive's nine
high-level cards as navigation and workflow entry points onto the
existing pages — never duplicated implementations. The single source
of truth is `ui/nine_cards.py` (`NINE_CARDS`);
`tests/ui/test_nine_cards.py` (26 tests) pins the exact order, the
mapping below, live coverage of every specialized page, the
exception-safe dependency probes, and the `go_to` routing through the
real MainWindow. The page stack stays 18 widgets.

Phase-3 polish (§8, v1.2.170): the grid is responsive —
`NineCardGrid` re-flows the SAME card widgets row-major across three,
two (below 880 logical px) or one (below 560) columns, always in the
§5 order, never rebuilding state — and the Datasets card carries a
real SMILES preview: the genuine first row of
`examples/data/screening_library.csv` (aspirin) rendered through the
Studio's own §22 `MoleculeRenderer`, with the honest text refusal
when rdkit or a QApplication is absent. No other card renders a
molecule ("where meaningful", not everywhere).

| # | Card | Entry points (nav keys) | Backend home | Tests |
| --- | --- | --- | --- | --- |
| 01 | Introduction | `dashboard` | `pharmaml` meta | `test_nine_cards.py` |
| 02 | Datasets | `data`, `chemistry`, `utilities` | `utils.io`, `chem.*` | `test_nine_cards.py`, `test_io.py` |
| 03 | Machine Learning | `models`, `mlworkbench`, `unsupervised`, `bestofml`, `foundations` | `ml.*` | `test_nine_cards.py`, `test_ml_pipeline.py` |
| 04 | Deep Learning | `deep_learning` | `ml.neural`, `ml.pruning`/`ml.quantization`/`ml.harness` (Efficiency lab), `ml.geometric`/`ml.gnn`/`ml.gnnxai` (GNN lab) | `test_nine_cards.py`, `test_dl_efficiency_lab.py` |
| 05 | Generative AI | `generative` | `generative.*` | `test_nine_cards.py`, `test_generative.py` |
| 06 | Molecular Docking | `docking` | `docking.*` (platform-dependent engine runtime) | `test_nine_cards.py`, `test_docking.py`, `test_docking_prep.py` |
| 07 | PK/PD | `pkpd`, `biology`, `responseai` | `pkpd.*`, `pharma.*` | `test_nine_cards.py`, `test_pkpd_engines.py` |
| 08 | Molecular Dynamics Analysis | `molecular_dynamics` | `md.*` (data external) | `test_nine_cards.py`, `test_md_analysis.py` |
| 09 | Visualization | `imagelab` | `viz.*` | `test_nine_cards.py`, `test_viz.py` |

The optional-dependency chips are computed live per card
(`probe_available` — one honest import probe, never a hardcoded
answer): `shap` (ML explanations), `torch` (DL + generative backends),
`rdkit` (generative scoring + docking prep), `matplotlib` (figure
export). The class-F boundaries are stated on the cards themselves:
docking execution needs a compatible engine runtime plus user-supplied receptor
PDBQT (source/ZIP includes Windows engines; native builds or Wine may be
needed, and scores are user or engine-generated inputs). MD
trajectories come from external GROMACS/PLUMED runs (the analysis
engines are bundled, the simulations are not).

**Maintenance rule**: this file must be re-trued whenever a page, card,
route, backend module, or §5 nine-card entry is added or retired — the
`tests/test_feature_inventory.py` suite fails the release if the matrix
drifts from the live registries (18 pages, 34 cards, 28 routes, the
9-card mapping, the C-list, the six legend classes).
