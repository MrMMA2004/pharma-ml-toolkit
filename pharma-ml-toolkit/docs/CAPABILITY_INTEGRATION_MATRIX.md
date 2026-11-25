# CAPABILITY_INTEGRATION_MATRIX — part11_1 §21 (Phase Sixteen)

**Baseline:** v1.2.228 (round 289) · **r290 addendum:** the ML
Explainability/Uncertainty queue lane CLOSED — `POST /api/v1/ml/explain`
(`ml.explain.shap_summary`) with the GUI stage, 8 API pins, and the
live browser run (evidence: `docs/gui/screenshots/r290-ml-explainability-*.png`).
· **r291 addendum:** the Plotly.js src adoption campaign CLOSED —
the shared `frontend/src/components/charts/PlotlyFigure.tsx` surface
now renders every interactive 2-D chart lane through real Plotly
traces (§22.2–§22.5: the ML parity scatter with the §22.5 click
inspector over `predict`'s new evaluation block — the bounded
aligned actual/predicted/SMILES sample, 5 API pins — the confusion
heatmap, the §24.2.B/C AD scatters, the mean-|SHAP| bar, the DL
learning curve, the PK/PD + MD curves); the row below gains the
chart column's live evidence (r291 browser run, 7 evidence shots:
`docs/gui/screenshots/r291-*.png`).
· **r356 addendum (part17 §3.1):** the dataset-deletion lane CLOSED —
the Datasets · Inspect stage now carries the row-level Delete action
behind the two-step confirmation (name + metadata + scope note) →
the existing typed `datasetsApi.remove` → `DELETE /api/v1/datasets/{id}`
with the honest 404 lane, the registry refresh, and the deleted-selection
clear (the vitest §3.1 flows + the `tests/test_frontend_datasets_delete.py`
drift guard; no duplicate client, no new endpoint).
· **r357 addendum (part17 §4, the P1 lane):** the application
penetration-testing program BORN — not a capability row (no new API
surface; the contract stays 117/73) but the QA program this matrix's
evidence culture feeds: `tests/api/test_penetration_security.py` (67
OWASP-mapped security tests over the existing surface), the four
program documents under `docs/QA/` (THREAT_MODEL,
PENETRATION_TEST_PLAN, PENETRATION_TEST_REPORT,
SECURITY_REMEDIATION_REGISTER), and the isolated ZAP lane
(`.github/workflows/penetration-test.yml` + `tools/zap_target.py` +
`tools/zap_gate.py` + the suppression channel). The round's own find
F-356-01 (the artifact-filename Content-Disposition injection) is
FIXED+PINNED in the register.
· **r358 addendum (part17 §5, the M1 lane):** the generic
Multi-Response / Multi-Output ML capability row is BORN —
`src/pharmaml/ml/multiresponse.py` (the strategy registry with each
mechanism STATED — native 2-D Y / MultiOutputRegressor wrapper /
RegressorChain feed, the split-first leakage-safe trainer, the
per-target evaluation surface, the §5.6 persistence contract, the
`PerTargetMultiOutput` masked multi-task wrapper) +
`src/pharmaml/api/routes/multiresponse.py` (the four-path quartet:
registry / train / predict / models — the contract 117→121 paths,
73→75 schemas, `api.d.ts` re-grown) + the ML-workspace section
(stage 20 inside the EXISTING Machine Learning workspace — no tenth
workspace) + the five typed service methods + the vitest lane (6
tests) + `tests/api/test_multiresponse.py` (38 tests: the mandated
families — PLS2 / MultiOutputRegressor / RegressorChain /
MultiTaskElasticNet / per-target missing / target-scaling leakage /
group-aware split / CV / aggregates / serialization-reload /
prediction ordering / artifact export / contract / e2e) + the
`tests/test_frontend_multiresponse.py` drift guard (12 pins).
**Spec:** MASTER_PROMPT part11_1 §21 —
"Complete Backend–GUI Integration Audit" — with part11_2 §12's status labels.
**Method:** every row below was derived live from the working tree — the 88
OpenAPI operations (`contracts/openapi.json`) were traced to their route
handlers and the real scientific functions they invoke (`src/pharmaml/…`),
the frontend consumers were grepped from `frontend/src/services/*` and the
nine workspace pages, and the test evidence was collected from the live
pytest suites (`tests/api/` = 261, the frontend drift guards = 123, the
legacy studio suite = 873) plus the scratch browser drivers (e2e280–289).
Nothing was copied from prose or a capability registry alone — §21's own
rule: "Do not assume that a feature is connected merely because it appears
in a capability registry."

Status labels: **PASS** (implemented and verified) · **PARTIAL**
(connected but incomplete/unverified in some §21 dimension) · **FAIL**
(missing or broken — none remain) · **BLOCKED** (external limitation) ·
**NOTAPPLICABLE** (genuinely not a user-facing GUI capability, with the
classification §21 demands).

---

## 1. The §21 verification chain

§21 requires fifteen facts per capability. How each was verified this
round:

| §21 item | Verification method | Verdict |
|---|---|---|
| 1. Backend implementation exists | the endpoint→function trace (below): every handler names the `src/pharmaml/…` callable it runs | PASS |
| 2. Input schema defined | Pydantic/Form schemas in the route modules; the 54 `components.schemas` in the committed OpenAPI contract | PASS |
| 3. API operation implemented | the 88 operations in `contracts/openapi.json`, drift-gated by `tests/api/test_contracts.py` (6) | PASS |
| 4. API invokes the actual backend function | the per-route trace below names module.attr for every operation; `tests/api/` runs the real app (no mocks) | PASS |
| 5. Heavy operations execute in a worker | the dual-mode `_*_body → job_manager.submit` pattern; `ThreadPoolExecutor(4)` + `Semaphore(2)` for expensive jobs (`api/jobs.py`) | PASS |
| 6. Job lifecycle managed | the QUEUED→RUNNING→COMPLETED/FAILED/CANCELLED machine, state transitions upserted to SQLite; `test_backend_service.py` 43 tests | PASS |
| 7. Progress reported | `job.progress`/`training.progress|metrics`/`iteration.update`/`descriptor.progress`/`file.progress` envelopes through the EventBus → WS; pinned by `test_dl_semi_generative.py` (WS-streamed, replay-verified) | PASS |
| 8. Results returned | inline JSON (sync lanes) or `/jobs/{id}/result` (job lanes; 409 until COMPLETED) — both exercised in `tests/api/` | PASS |
| 9. React client receives the result | the 13 domain clients in `frontend/src/services/` (typed off the generated `api.d.ts`); the browser drivers e2e280–289 prove receipt end-to-end | PASS |
| 10. React component displays the result | the per-workspace tables/cards below (49 result `<table>`s, TrainedModelCard, ADResultCard, DLModelCard, CurveFigure, Molecule3D…) | PASS |
| 11. Applicable plots work | the visualization lanes below (6 SVG chart components + 2 Three.js viewers + the backend-figure artifact lane) | PARTIAL — see §3.2 |
| 12. Applicable exports work | the export columns below (tables ×6 formats, figures ×4 formats, model checkpoints, PDBQT artifacts); digest-verified downloads | PASS |
| 13. Errors handled | `describeError` surfaces `ApiError.status`+`detail` verbatim per page + ErrorBanner + honest 422/404s pinned in `tests/api/` | PASS |
| 14. Integration tests pass | `tests/api/` 261/261, frontend guards 123/123, `test_windows_pipeline.py` 8/8 (this round's gates) | PASS |
| 15. Results persisted where required | the persistence column below (SQLite write-through + FileVault when `data_dir` is set; `test_persistence.py` 30 + `test_persistence_database.py` 37) | PASS |

The one §21 FAIL the audit originally filed — **the browser app never
wired its own boot sequence** (`configureBackend`/`stream.connect` had no
caller in `frontend/src`; only the Node e2e harnesses called them, so a
real browser session could not reach its own backend without an injected
shim) — is **closed this round** by the §14.1 `StartupGate`
(`src/app/StartupGate.tsx` + `src/services/api/startup.ts`), browser-verified
12/12 (the app boots itself, the health probe passes, the event stream opens
`open`, and a dataset uploads through the native path —
`scratch/browser_verify_startup_r289.sh`). "A button that does not execute a
real backend function is not integration" — now no page even mounts before
the handshake completes.

---

## 2. The integration matrix

Columns: **Capability** (backend function) · **Endpoint** (the versioned
REST surface) · **GUI workspace / UI control** (the React consumer) ·
**Worker** (J = JobManager lane, S = synchronous request lane) ·
**Result handler** (the component/state that consumes it) · **Viz** ·
**Export** · **Test** (the live evidence) · **Status**.

### 2.1 Datasets + Chemistry (§11.1; card 02)

| Capability | Endpoint | GUI workspace / control | W | Result handler | Viz | Export | Test | Status |
|---|---|---|---|---|---|---|---|---|
| `data.load_dataset` + `DatasetStore.register` | POST `/datasets` (multipart) | Datasets · Import stage: file input, target/task/mode | J/S | upload `JobRunCard` → dataset registry select | — | dataset table artifact (csv/tsv/xlsx) | `test_datasets_ml_artifacts.py` 56; e2e280 upload job; r289 browser 12/12 | PASS |
| `DatasetStore.list/get` | GET `/datasets`, `/datasets/{id}` | every workspace's dataset select | S | registry + view cards | — | — | same suite; contract paths pinned | PASS |
| `stores.frame_payload` (bounded preview) | GET `/datasets/{id}/preview`, `/column/{col}` (≤500) | Datasets · Inspect: `PreviewTable`, `ColumnChips` | S | paginated preview + column chips | — | — | preview/NaN-as-null pins | PASS |
| `data_profiling.profile_frame` | POST `/datasets/{id}/profile` | Datasets · Inspect profile button | S/J | profile summary card | — | — | profile tests | PASS |
| `data.split_table` | POST `/datasets/{id}/split` | Datasets · Split/Leakage stage: ratio/stratify/seed | S/J | split side tables + target balance | — | — | split tests | PASS |
| `DatasetStore.evict` | DELETE `/datasets/{id}` | Datasets · Inspect: per-row Delete action behind the two-step confirmation (name + metadata + scope note; `§3.1` r356) | S | registry refresh + the deleted-selection clear + the honest 404 note | — | — | route + evict tests; vitest §3.1 flows (15 over the page suite); `test_frontend_datasets_delete.py` 14 | PASS — r356 |
| `chem.descriptors.compute_descriptors` | POST `/chemistry/descriptors` | Datasets · Chemistry stage: descriptor-set select; Generative · properties | J/S | descriptor frame table (+ failed-set honesty) | — | — | `test_backend_service.py` chemistry (sync+job+config preservation) | PASS |
| `chem.descriptors.compute_fingerprints` | POST `/chemistry/fingerprints` | Datasets · fingerprint kind/radius/n-bits/counts form (≤500) | J/S | fingerprint matrix table | — | — | same suite | PASS |
| `viz.statplots.plot_*` (stat) | POST `/visualization/stat` | Datasets · Visualize stage (column select); Visualization lane | S | stat histogram artifact → digest-verified download | artifact figure | PNG (+Visualization: PNG/SVG) | `test_docking_pkpd_md_viz_ad.py` viz group | PASS |
| `utils.io.write_table` via `/artifacts/table` | POST `/artifacts/table` | Datasets · Export: format select | S | artifact download (sha-256 verified) | — | csv/tsv/xlsx | download contract pins (ETag/digest) | PASS |

### 2.2 Machine Learning + Semi-supervised + AD (§11.2; card 03)

| Capability | Endpoint | GUI / control | W | Result handler | Viz | Export | Test | Status |
|---|---|---|---|---|---|---|---|---|
| `ml.models.list_models` | GET `/ml/models?task&only_available` | ML · Algorithm stage: catalog radio (task-filtered, unavailable rendered honestly) | S | catalog card | — | — | catalog tests | PASS |
| `ml.training.train_model` (+ `resolve_model_name`) | POST `/ml/train` | ML · Hyperparameters: model-params JSON editor, test-size/folds/n-jobs/seed | J/S | `TrainedModelCard` (best_params, CV mean/std) | — | predictions table | train/tuned-CV pins | PASS |
| stored estimator `.predict/.predict_proba` | POST `/ml/predict` (dataset mode carries the §22.5 `evaluation` block — bounded aligned actual/predicted/SMILES rows, 5 pins) | ML/DL · Inspect Predictions: predict-proba toggle + the Plotly parity scatter / confusion heatmap with the click→sample inspector | S/J | predictions table + probabilities block + evaluation block | Plotly parity scatter (customdata §22.5) · confusion heatmap | csv/tsv/xlsx | inline/on-dataset/proba+evaluation pins; e2e283 + r291 browser (evidence: r291-ml-parity-*.png) | PASS |
| `ml.semi_supervised.train_semi_supervised` | POST `/semi-supervised/train` | ML · Semi-supervised stage: method/base-estimator/threshold/k-best/kernel panel | J/S | pseudo-label counts + confidence summary view | — | — | `test_dl_semi_generative.py` 30 (incl. predict via `/ml/predict`) | PASS |
| `ml.semi_supervised.evaluate_semi_supervised` | POST `/semi-supervised/evaluate` | ML · Semi-supervised stage: the /evaluate GUI consumer (landed r303 — its wiring exposed + fixed the job-mode TypeError; the COMPLETION pin) | S | the leakage-safe evaluation verdict view | — | — | evaluate pins incl. the job-mode COMPLETION (r303) | PASS — §3.4 (was PARTIAL; consumer landed r303) |
| `ml.applicability.applicability_domain` (+ bootstrap) | POST `/applicability-domain/analyze` | ML · AD stage: k/percentile/n-models/uncertainty/figure panel (feature cap 50) | J/S | `ADResultCard` (counts + per-row reliability frame) | AD figure artifact | figure + AD table | analyze + bootstrap telemetry pins | PASS |
| `ml.applicability.knn_applicability_domain` | POST `/applicability-domain/knn` | consumed internally by analyze; standalone route | S | — | — | — | knn pins (API-only) | NOTAPPLICABLE — internal utility |
| `ml.applicability.check_domain_box` | POST `/applicability-domain/domain-box` | metadata rule check | S | — | — | — | domain-box pins | NOTAPPLICABLE — internal utility |
| `ml.explain.shap_summary` | POST `/ml/explain` | ML · Explainability stage: trained-model select, bounded max_rows sample (route cap 500), figure toggle; non-tree estimators refused honestly (the r289 queue lane, closed r290) | S/J | `ExplainResultCard` (engine/model/task chips, full mean-\|SHAP\| ranking table, base value, bounded beeswarm preview strip) | shap-importance.png + shap-beeswarm.png figure artifacts | digest-verified downloads ×2 | explain pins 8 (sync/job/bounds/refusal/figures/404s) + the r290 browser run | PASS |
| AD catalog | GET `/applicability-domain/catalog` | defaults source for the ML AD panel | S | panel defaults | — | — | contract pin | NOTAPPLICABLE — config surface |

### 2.3 Deep Learning (§11.3; card 04)

| Capability | Endpoint | GUI / control | W | Result handler | Viz | Export | Test | Status |
|---|---|---|---|---|---|---|---|---|
| `ml.neural.{DeepMLP, ResidualMLP}×{Classifier, Regressor}` catalog | GET `/dl/networks` | DL · Architecture: catalog with live signature defaults | S | architecture cards | — | — | network catalog pins | PASS |
| torch `.fit(on_epoch=…)` | POST `/dl/train` | DL · Configure/Train: hyperparam editor constrained to `DL_HYPERPARAM_KEYS` | J | `DLModelCard` + live loss/val curve | **LossCurveChart** (inline SVG, WS-fed) | loss-curve figure ×4 formats; joblib checkpoint (`/artifacts/model`) | `test_dl_semi_generative.py`: per-epoch `training.progress|metrics` **WS-streamed live** | PASS |
| checkpoint `ml.persistence.save_model` | POST `/artifacts/model` | DL · Save Checkpoint | S | artifact download | — | joblib bundle | model-bundle round-trip (download → `load_model` → identical predictions) | PASS |

### 2.4 Generative AI (§11.4; card 05)

| Capability | Endpoint | GUI / control | W | Result handler | Viz | Export | Test | Status |
|---|---|---|---|---|---|---|---|---|
| `generative.scoring` objective catalog (9 kinds) | GET `/generative/objectives` | Generative · Constraints: objective kind + param-spec editor | S | objective cards | — | — | objectives pins | PASS |
| `generative.rnn.SmilesRnn.train_prior` | POST `/generative/rnn/train` | Generative · corpus textarea / dataset column; embedding/hidden/epochs… | J/S | prior card (losses, final_loss, corpus_source) | prior-loss figure artifact | figure ×4 | prior train + `training.metrics` pins | PASS |
| `SmilesRnn.sample` | POST `/generative/rnn/sample` | Generative · Sample: n/temperature/top-k/seed | S (sync-only by design) | sample list + validity verdicts | — | — | sample pins (corpus ≥5 guard) | PASS |
| `generative.genetic.GeneticOptimizer.optimize` | POST `/generative/optimize` | Generative · GA config: population/generations/rates/elitism | J/S | top-rows table (score breakdown) + iteration telemetry | rank-scatter figure artifact | figure ×4 + table ×6 | optimize + `iteration.update` per generation; `JobCancelled` → CANCELLED pin | PASS |
| prior registry | GET `/generative/priors[/{id}]` | Generative · prior select | S | prior view | — | — | prior pins | PASS |

### 2.5 Molecular Docking (§11.5; card 06)

| Capability | Endpoint | GUI / control | W | Result handler | Viz | Export | Test | Status |
|---|---|---|---|---|---|---|---|---|
| `docking.workflow` engine census | GET `/docking/engines` | Docking · Engine stage: catalog with `boundary_note` verbatim (Windows-PE/Wine honesty) | S | engine cards | — | — | honest-boundary pins | PASS |
| `docking.dockstring` targets/objective | GET `/docking/dockstring/targets[/{name}]`, POST `/dockstring/objective` | Docking · Binding-site: 58-target table + search | S | target config text | — | — | dockstring pins | PASS |
| `docking.receptor.prepare_protein` | POST `/docking/receptor/prepare` | Docking · Prepare Receptor: PDBQT textarea + options | S | receptor artifact row | **Molecule3D** receptor card (Three.js) | receptor PDBQT artifact | receptor pins + 3D browser-verify (r288 21/21) | PASS |
| `docking.preparation.smiles_to_pdbqt` | POST `/docking/ligand/pdbqt` | Docking · Prepare Ligands | S | ligand artifact row | **Molecule3D** ligand card (never-fabricated-pose note) | ligand PDBQT artifact | ligand pins + 3D verify | PASS |
| `smiles_docking.dock_smiles_array` | POST `/docking/dock` | Docking · Run: engine/exhaustiveness/seed; pasted logs parse lanes | J | dock rows table + per-ligand progress telemetry | — | poses table ×3 | dock + per-ligand `job.progress`; partial-results-cancel honesty | PASS |
| `docking.parsing.parse_vina_output` / `parse_dlg_*` | POST `/docking/parse/{vina,dlg}` | Docking · Inspect Poses: pasted Vina log / DLG textareas + samples | S | mode table / cluster histogram | DLG cluster histogram | — | parse pins | PASS |
| `docking.scoring.{consensus_breakdown, ligand_efficiency}` | POST `/docking/scoring/{consensus, ligand-efficiency}` | Docking · Analyze: consensus rows / affinity+heavy-atoms | S | breakdown + efficiency tables | — | — | scoring pins | PASS |
| `docking.contacts.{compute_contacts, plif_matrix}` | POST `/docking/contacts` | Docking · Interactions: ligand+receptor PDBQT | S | contacts table + PLIF matrix | — | — | contacts/PLIF pins | PASS |
| `docking.pose_rmsd.best_pose_rmsd` | POST `/docking/pose-rmsd` | Docking · pose pairs | S | symmetry-aware RMSD values | — | — | pose-rmsd pins | PASS |
| pose-affinity figure | POST `/artifacts/figure` | Docking · Visualize: format select | S | figure artifact | pose-affinity figure | ×4 formats | figure lane pins | PASS |

### 2.6 PK/PD (§11.6; card 07)

| Capability | Endpoint | GUI / control | W | Result handler | Viz | Export | Test | Status |
|---|---|---|---|---|---|---|---|---|
| catalog | GET `/pkpd/catalog` | PKPD · Select Model: 4 catalog groups | S | model cards | — | — | catalog pins | PASS |
| `pharma.pk.nca_summary` + `nca_ext.fit_terminal_phase` | POST `/pkpd/nca` | PKPD · Import + NCA: pasted conc-time series | S | NCA metrics dict (AUC/Cmax/Tmax/t½ + r² + span_low warning verbatim) | — | — | NCA pins | PASS |
| `nca_ext.steady_state_metrics` | POST `/pkpd/steady-state` | PKPD · interval metrics | S | steady-state card | — | — | steady-state pins | PASS |
| `pharma.pk.iv_bolus_concentration` | POST `/pkpd/iv-bolus` | PKPD · ParamInput (c0/ke) | S | — | **CurveFigure** (SVG concentration-time) | — | iv-bolus pins | PASS |
| `pkpd.compartments.{bateman, superposition}` | POST `/pkpd/{bateman, superposition}` | PKPD · ParamInput (ka/ke/dose/volume) | S | curve + superposition table | CurveFigure | — | compartment pins | PASS |
| `pkpd.pbpk.MinimalPBPK` | POST `/pkpd/pbpk` | PKPD · physiology set | S | PBPK curve | CurveFigure | — | pbpk pins | PASS |
| `pharma.pd.effect_table` | POST `/pkpd/pd` | PKPD · sigmoid-Emax (Emax/EC50/hill) | S | effect table | response CurveFigure | — | pd pins | PASS |
| `pkpd.population.simulate_population` | POST `/pkpd/population` | PKPD · Population: theta/omega/seed/batches | J | percentile table + per-batch telemetry | — | table ×6 | population job + cancel-never-failed pin; e2e286 browser | PASS |
| exposure/response figures | POST `/artifacts/figure` | PKPD · Visualize | S | figure artifacts | exposure + response figures | PNG (fixed format) | figure pins | PASS |

### 2.7 Molecular Dynamics Analysis (§11.7; card 08 — analysis, never simulation)

| Capability | Endpoint | GUI / control | W | Result handler | Viz | Export | Test | Status |
|---|---|---|---|---|---|---|---|---|
| catalog | GET `/md/catalog` | MD · Select Analysis: 5 groups | S | analysis cards | — | — | catalog pins | PASS |
| `md.trajectory.synthetic_trajectory` + `md.metrics` bundle | POST `/md/synthetic`, `/md/analyze` | MD · Load: seeded demo or pasted frames; `Trajectory`/`Topology` | S | metric bundle (RMSD/RMSF/Rg/native contacts) | **OverlayCurveFigure** (raw + moving avg) | — | synthetic/analyze pins | PASS |
| `md.xvg.parse_xvg` + `metric_stats` | POST `/md/xvg` | MD · pasted .xvg textarea | S | XVG stats | CurveFigure | — | xvg pins | PASS |
| `md.enm.anm_fluctuations` | POST `/md/enm` | MD · ENM coords + cutoff/gamma | S | fluctuation table | — | — | enm pins | PASS |
| `md.dimred.trajectory_pca` | POST `/md/pca` | MD · n-components | S | eigen/projection tables | — | — | pca pins | PASS |
| `md.study` (+ `plot_md_series` ×2) | POST `/md/study` | MD · Study config (n-residues/noise/drift/mode) | J | staged progress + metrics + figures | md-series figures | PNG | study job pins (real figures) | PASS |
| trajectory playback | (frontend lane over `/md/synthetic` frames) | MD · Validate: frame slider + play/pause | — | **Molecule3D trajectory viewer** (220 ms steps) | 3D animated | — | r288 browser 21/21 | PASS |

### 2.8 Visualization (§11.8; card 09) + the cross-workspace registry

| Capability | Endpoint | GUI / control | W | Result handler | Viz | Export | Test | Status |
|---|---|---|---|---|---|---|---|---|
| catalog (families + themes) | GET `/visualization/catalog` | Visualization · family select | S | family cards | — | — | catalog pins | PASS |
| `viz.statplots.*` | POST `/visualization/stat` | Visualization · stat lane (bins/whis/n-boot/seed) | S | stats dict + figure | figure artifact | PNG/SVG | stat pins | PASS |
| `viz.pharma_plots.plot_concentration_profile` | POST `/visualization/concentration-profile` | Visualization · time-conc pairs | S | figure | figure | ×4 formats | cp pins | PASS |
| `viz.pharma_plots.plot_dose_accumulation` | POST `/visualization/dose-accumulation` | Visualization · dose/tau/k/a | S | figure | figure | ×4 formats | da pins | PASS |
| `viz.report.export_table` | POST `/visualization/table` | every workspace's Export stage | S | digest-verified download | — | ×6 formats (5,000-row cap) | table pins | PASS |
| `viz.report.HTMLReport` | POST `/visualization/report` | Visualization · report theme select | J | report body view (live `file.progress`) | self-contained HTML | report artifact | report job pins | PASS |
| the artifact registry (cross-workspace) | GET `/artifacts` (+ download) | Visualization · the consumption hub (kind filter + side-by-side Compare) | S | registry table + compare view | — | all downloads | artifact lanes | PASS |

### 2.9 The control plane (§12/§13/§16 — not a workspace)

| Capability | Endpoint | GUI / control | W | Result handler | Test | Status |
|---|---|---|---|---|---|---|
| health probe | GET `/api/v1/health` | consumed by the **StartupGate** (this round: the app's own boot verifies it) | S | gate + banner | `test_backend_service.py` handshake group; r289 browser | PASS |
| capability discovery | GET `/capabilities[/{id}]` | **no React consumer** — the React side's source of truth is the generated `api.d.ts` (§12.2: generated, not duplicated) | S | — | capability pins | NOTAPPLICABLE — code-gen-time surface |
| job control plane | GET `/jobs`, `/jobs/{id}`, `/jobs/{id}/result`; POST `/jobs/{id}/cancel` | `JobRunCard` (status/progress/cancel/result) + `useJob` on every workflow page; `jobsApi.list` **no UI ledger panel** | S | JobTracker overlays WS + REST authority | the full §13 group (43 tests) | PASS (`list` PARTIAL — §3.4) |
| the typed event stream | WS `/api/v1/events` | the app's own boot path opens it (StartupGate → `stream.connect()`); the banner consumes `useEventStreamState`; per-page raw subscriptions (DL metrics, generative iterations, docking per-ligand, PKPD batches, MD stages, report sections) | — | tracker + telemetry lists | WS hello/replay/since/live-stream pins; r289 browser `open` | PASS |
| SQLite persistence surface | GET `/database{,/jobs,/preferences[/{key}]}`; PUT/DELETE preferences; POST `/database/{backup,restore}` | **no React consumer** (§16/§35 surfaces; the ship's data lives in `%LOCALAPPDATA%`) | S | — | `test_persistence.py` 30 + `test_persistence_database.py` 37 | PARTIAL — §3.4 |
| artifacts view/delete | GET/DELETE `/artifacts/{id}` | **no UI consumer** (list+download are the used lanes) | S | — | evict pins | PARTIAL — §3.4 |

---

## 3. Honest findings

§21: "A capability that is blocked by a genuine technical limitation must
be explicitly reported and approved; it cannot be silently marked
complete." The findings this audit filed, with their dispositions:

### 3.1 CLOSED this round — the startup bridge (was FAIL)

The browser app never wired its own §14.1 boot sequence:
`configureBackend` and `getEventStream().connect()` had **zero callers**
in `frontend/src` (only the Node e2e drivers called them), so a real
browser/Tauri session threw "the backend endpoint is not configured yet"
on every service call. **Fixed (r289):** `services/api/startup.ts` (the
Tauri `backend_endpoint` IPC → `configureBackend` → `/health` verify →
stream open; dev-only same-origin fallback through the Vite proxy with
`VITE_PHARMAML_TOKEN`), `app/StartupGate.tsx` (connecting/failed+retry/ready;
the WS-degrade banner is the first `useEventStreamState` consumer), the
proxy's WS lane (`ws: true`), `@tauri-apps/api` pinned, and
`tests/test_frontend_startup.py` (8 drift guards). Browser-verified 12/12
(`scratch/browser_verify_startup_r289.sh`): the app boots itself, the
gate reports `backend 1.2.228 ready via dev-same-origin`, the stream is
`open`, and a dataset uploads through the native path.

### 3.2 PARTIAL — Plotly.js is declared, chunked, and pinned — but not yet imported

> **r356 truing (part17 §3.3):** this section is the part11-era r289
> snapshot and its "zero `src/` files import it" claim was superseded
> by the r291 addendum above — `PlotlyFigure.tsx` is now imported by
> its owning workflows (the ML page's lanes, PK/PD, and their 12 test
> suites; 16 files reference it), the workflow-specific charts live in
> their owning workflows, and the Visualization workspace stays the
> genuinely general-purpose backend-figure lane. The paragraph below
> stands as the historical record; the live truth is the r291 addendum.
> Status: the Plotly adoption is VERIFIED_COMPLETE since r291.

`plotly.js-dist-min@3.1.1` is a runtime dependency with its own manual
chunk; the drift guards pin it as part of the mandated stack. **Zero
`src/` files import it today** — every 2D chart (loss curves,
concentration-time, MD overlays) is a hand-rolled inline SVG component.
The backend-figure lane (Matplotlib → artifact → digest-verified
download) covers the static scientific figures, and the SVG components
are interactive where it matters (hover, selection). §46.1's
"Plotly.js is the primary interactive scientific charting library"
therefore stands at **PARTIAL**: installed, budgeted, and honest — the
migrations of the remaining chart components to Plotly traces is the
next campaign row, tracked in the queue, never claimed as done.

### 3.3 PARTIAL — the WS channel is behaviorally tested but absent from the OpenAPI contract

`/api/v1/events` is fully pinned by 43+ API tests (hello/replay/since/
live-stream/shutdown) and now by the app's own boot path — but OpenAPI
3.1 has no standard way to declare the WebSocket channel, and the repo
carries no AsyncAPI companion. Recorded here rather than silently
absent; the frontend side is drift-guarded against
`pharmaml.api.events.EventType` by `test_frontend_services.py`.

### 3.4 PARTIAL — backend surfaces with no React consumer (the honest classification)

§21's rule cuts both ways: these are NOT counted as GUI integrations,
and they are not silently omitted either:

* `GET/DELETE /artifacts/{id}` (view/delete), `GET /jobs` (ledger)
  — service functions exist (`artifactsApi.view/delete`,
  `jobsApi.list`), API-tested, **no UI control**. Disposition:
  housekeeping-lane campaign row (the r356 part17 §3.2 queue).
  [`DELETE /datasets/{id}` left this list in r356 — see the §3.1
  addendum above.]
* `POST /semi-supervised/evaluate` — the leakage-safe evaluation
  protocol; the ML page renders the honest queue note instead of a
  fake button. Disposition: campaign row (the workflow's §20.3
  evaluation-protocol display).
* The `/database` surface (job history, preferences CRUD, backup,
  restore) — §16/§35 surfaces exercised by 67 tests; the ship's data
  directory is the backend's `%LOCALAPPDATA%` contract. Disposition:
  campaign row (a Settings panel).
* `GET /capabilities`, `/applicability-domain/{catalog,knn,
  domain-box}` standalone routes — code-gen/config/internal-utility
  surfaces: **NOTAPPLICABLE** as standalone GUI features (§21's
  internal-utility classification), each still API-tested.

### 3.5 The legacy PySide6 registry is not this matrix

`pharmaml.studio.capability_registry` (62 rows, all `integrated`) and
`tests/ui/` (873 tests) cover the **legacy studio GUI** — §28 Stage 1's
preserved reference frontend, still functional for parity comparison.
The React integration status lives in this document and in
`frontend/src/app/workspaces.ts` (`state: "workflow"` ×9). PySide6
retirement (§28 Stage 5) remains gated on the §28 Stage 4 parity
audit — the final §46.8 sweep tracks it.

### 3.6 No executable JavaScript test files

`frontend/` has zero `*.test.tsx` files and no vitest/jest config. The
React side is guarded by 123 Python drift guards (foundation 6,
services 16, web parity 101 — which parse the real backend route
modules and generated `api.d.ts`) plus the scratch browser drivers
(e2e280–289, run live each round). §39.3's "React frontend tests"
criterion is met by this battery at the contract level; component-level
JS tests remain a documented campaign row, not a silent gap.

---

## 4. Verdict

| Domain | Operations | PASS | PARTIAL | FAIL | BLOCKED | N/A |
|---|---|---|---|---|---|---|
| Datasets + Chemistry | 12 | 12 | 0 | 0 | 0 | 0 |
| ML + Semi-supervised + AD | 10 | 7 | 1 (evaluate) | 0 | 0 | 2 (internal) |
| Deep Learning | 3 | 3 | 0 | 0 | 0 | 0 |
| Generative AI | 5 | 5 | 0 | 0 | 0 | 0 |
| Molecular Docking | 11 | 11 | 0 | 0 | 0 | 0 |
| PK/PD | 10 | 10 | 0 | 0 | 0 | 0 |
| MD Analysis | 7 | 7 | 0 | 0 | 0 | 0 |
| Visualization + artifacts | 8 | 7 | 1 (view/delete) | 0 | 0 | 0 |
| Control plane (health/jobs/WS) | 7 | 4 | 3 (db surface, jobs list, capabilities) | 0 | 0 | 1 (codegen) |
| **Total** | **73 capability rows** | **66** | **5** | **0** | **0** | **3** |

Every user-facing workflow chain (UI control → service client → REST →
job/worker → real scientific function → result handler → visualization →
export → persistence) is **PASS** with live test evidence. The five
remaining PARTIAL rows are housekeeping/protocol/config surfaces with no UI
consumer — each classified, dispositioned to a campaign row, and never
counted as integration. The audit's original FAIL (the unwired startup
bridge) is closed and browser-verified this round.

*Next queue (the remaining part11_1 audit phases): Phase Twenty-Four
(§29 dependency & packaging readiness — see
`packaging/windows/DEPENDENCY_INVENTORY.md`) and the §46 final
acceptance sweep (see `docs/ACCEPTANCE_SWEEP.md`).*
