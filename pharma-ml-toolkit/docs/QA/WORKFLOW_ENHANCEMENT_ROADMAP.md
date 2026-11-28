# WORKFLOW ENHANCEMENT ROADMAP — the nine-workspace plan (part21 / Round 372)

The per-workspace enhancement backlog for the v1.2.244 tree. Every row below was
written by reading the actual implementation: the workspace `Page` component in
`frontend/src/features/<workspace>/`, its typed service client in
`frontend/src/services/<workspace>/client.ts`, the FastAPI route file it calls in
`src/pharmaml/api/routes/`, and the scientific module underneath
`src/pharmaml/`. Nothing here is speculative — each entry names the file that
carries the limitation today, and where a capability already exists (route,
engine, or both) the entry says so and proposes only the missing link. The
route-vs-consumer cross-check behind the "zero frontend consumer" claims was
done mechanically: every `@router` path in `routes/` was diffed against every
literal/template API path in `frontend/src/services/` (148 route handlers, 114
distinct consumed paths — the deltas are exactly the unconsumed surfaces named
below).

## The constraint rails every entry respects

- No second desktop architecture — all work extends the existing React shell +
  Python sidecar.
- No scientific computation moved into the React frontend — every number stays
  a backend verdict; the frontend renders payloads and coordinates inputs.
- No second persistence authority — new lanes register through the existing
  dataset/artifact/run stores, never a parallel database.
- Offline-capable — no CDN, no external network calls; any proposed dependency
  is an optional-tier pin shipped as an offline wheel, honestly degraded when
  absent (the `GET /api/v1/system/diagnostics` probe pattern).
- English-only UI, reuse of the existing `PlotlyFigure` integration
  (`frontend/src/components/charts/PlotlyFigure.tsx`) for any new chart.

## How to read an entry

Each recommendation is one compact table. Fields:

| Field | Meaning |
| --- | --- |
| ID / Feature | The stable ID (workspace prefix + number) and the feature in one line |
| Current status | Implemented / Partial / Missing / Not applicable — judged from source |
| Problem | The specific limitation, with the file(s) that carry it today |
| Proposed solution | The concrete implementation approach (route → client → page stage) |
| Scientific value | Why a computational-pharmacy user cares |
| Feasibility / Priority / Effort | High/Medium/Low · P0–P3 · Small/Medium/Large |
| Backend / Frontend changes | required or not |
| Dependencies | Existing (already pinned) or proposed optional |
| Performance impact | CPU / RAM / GPU / disk honestly stated |
| Related links | Canonical documentation, only where genuinely relevant |
| Acceptance criteria | The fail-first tests that close the row |

Priority scale: **P0** critical correctness/security/reliability, **P1**
high-value missing functionality, **P2** significant moderate, **P3**
optional/exploratory. Grouping per workspace: **Immediate** = an existing
backend capability can be exposed/reused with little or no backend science work;
**Medium** = focused development; **Long-term** = new methods, optional
dependencies, or scientific validation work.

---

## 1. Introduction

Current state: `frontend/src/features/introduction/IntroductionPage.tsx` (521
lines) renders the nine-workspace directory from the hardcoded
`frontend/src/app/workspaces.ts`, the global `DemoDataSection`, the live studio
database card (status + backup/restore + persisted job history over
`services/database/client.ts`), `DatabasePreferences`, `JobCenterCard`, and the
`DiagnosticsCard` (dependency probes, storage health, crash diagnostics, cache
cleanup). The backend also serves `GET /api/v1/capabilities`
(`routes/capabilities.py`, 2 paths) — and no frontend code calls it; it exists
only in the generated `types/api.d.ts`.

### Immediate improvements

**INT-1 — Live capability census card**

| Field | Entry |
| --- | --- |
| ID / Feature | INT-1 — Surface `GET /api/v1/capabilities` as a live capability census in the Introduction page |
| Current status | Partial (backend Implemented; frontend consumer Missing) |
| Problem | `routes/capabilities.py:18` projects the 62-row `pharmaml.studio.capability_registry` (status, tests, optional deps, engines per row), but `rg capabilities frontend/src` hits only `types/api.d.ts` — zero service client, zero UI. The page's workspace directory is the static `WORKSPACES` array in `app/workspaces.ts`, so what the backend actually ships and what the page claims can drift silently. |
| Proposed solution | Add `capabilitiesApi.list()` (+ optional `view(id)`) to a small service client; render a collapsible card beside the directory showing the by-status/by-card summary and the rows whose `status != "integrated"` (the honest exceptions), each row linking its named `tests` and `optional_deps`. Read-only; no second registry — the route stays the single source. |
| Scientific value | A user checking "is scaffold-aware splitting actually integrated or engine-only?" gets the backend's own answer instead of the page's static claim. |
| Feasibility / Priority / Effort | High · P2 · Small |
| Backend / Frontend changes | not required / required |
| Dependencies | existing (route + registry already shipped and test-referenced) |
| Performance impact | None — one read-only GET of a static in-memory registry |
| Related links | — |
| Acceptance criteria | A vitest lane over the new client (narrowed DTOs, malformed-payload throw) + an Introduction page test pinning the census card's by-status summary against a mocked 62-row payload + the Python drift guard asserting `routes/capabilities.py` still exports both paths. |

**INT-2 — Surface the `startup_recovery` report**

| Field | Entry |
| --- | --- |
| ID / Feature | INT-2 — Render the startup-recovery verdict inside the studio-database card |
| Current status | Partial (field typed in the client; never rendered) |
| Problem | `services/database/client.ts:44` types `startup_recovery?: Readonly<Record<string, unknown>>` on `DatabaseStatus`, documenting "the optional startup_recovery report" — but `IntroductionPage.tsx` renders only `database/journal_mode/integrity/tables` counts; the recovery report (what the persistence layer restored after the last unclean exit) is dropped on the floor. |
| Proposed solution | When `status.startup_recovery` is present, render a compact one-row notice ("last session recovered: N datasets, M jobs re-registered — from backup <ts>") under the Status card, with the full JSON behind the existing bounded-JSON pattern the Job Center uses for job results. |
| Scientific value | Reliability honesty: after a crash the user learns what survived without opening the diagnostics card. |
| Feasibility / Priority / Effort | High · P2 · Small |
| Backend / Frontend changes | not required / required |
| Dependencies | existing |
| Performance impact | None |
| Related links | — |
| Acceptance criteria | A vitest pair: presence renders the notice line, absence renders nothing; a Python drift guard pinning the field name on `routes/database.py`'s status payload. |

### Medium-term enhancements

**INT-3 — Contextual glossary tooltips over `GET /api/v1/glossary`**

| Field | Entry |
| --- | --- |
| ID / Feature | INT-3 — A scientific glossary/tooltip layer for the nine workspaces |
| Current status | Partial (engine Implemented; API route Missing; UI Missing) |
| Problem | `src/pharmaml/studio/glossary.py` ships the ~45-entry `GLOSSARY` dict plus `apply_help` (the legacy "intro.glossary" capability row in `capability_registry.py:113`), but no route in `routes/` serves it and the web frontend has no tooltip/help surface at all — `apply_help` is Qt-widget-shaped, so the data, not the function, is the reusable part. |
| Proposed solution | Add `GET /api/v1/glossary` returning `{term, short, long}` rows; a tiny frontend `HelpTerm` component (info-icon popover, English-only) used first on the PK/PD and ML pages' jargon-dense labels (AUC, lambda_z, AD, SHAP), sourced from the same route. Engine stays the single source; no terms re-declared client-side. |
| Scientific value | Domain vocabulary (e.g. what "terminal phase" or "leverage" means) is exactly where a non-pharmacometrician stops; the definitions already exist in-repo. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | Negligible CPU; one static GET cached per session |
| Related links | — |
| Acceptance criteria | API test pinning ≥40 rows and the three fields; vitest popover open/render test; drift guard that `GLOSSARY` keys stay unique. |

**INT-4 — Memory-aware execution presets wired to the settings store**

| Field | Entry |
| --- | --- |
| ID / Feature | INT-4 — Expose the `PRESETS` resource policies and prefill execution defaults |
| Current status | Partial (engine Implemented; API route Missing; settings UI exists but disconnected) |
| Problem | `src/pharmaml/utils/resources.py:154` defines `PRESETS` / `find_preset` / `preset_for_low_memory` / `stored_preset` (the "intro.presets" capability), and the frontend settings store already persists execution defaults (`services/settings/types.ts` "execution defaults — the ML form's n_jobs prefill"), but nothing connects them: the user cannot pick a Low/Balanced/High policy that prefills n_jobs/cv/epochs across workspaces from the machine's real memory. |
| Proposed solution | Extend `routes/system.py` (or the capabilities router) with `GET /api/v1/system/presets` (the preset table + the live `snapshot()` memory that `preset_for_low_memory` would resolve to); a Settings panel row picks a preset; the settings store maps it onto the existing execution-defaults fields per workspace. |
| Scientific value | Prevents the out-of-memory worker death the r344 fix had to translate — the user gets the clamp before the fit, not after. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (`ResourceSnapshot`/`snapshot` already feed `/system/resources`) |
| Performance impact | Negligible (reads `/proc` counters already read by the resources route) |
| Related links | — |
| Acceptance criteria | API test that the preset table matches `PRESETS` verbatim; vitest that selecting "low" prefills the ML form's n_jobs via the store; drift guard pinning `preset_for_low_memory`'s threshold constant. |

---

## 2. Datasets

Current state: `frontend/src/features/datasets/DatasetsPage.tsx` (1,540 lines)
runs the 8-stage workflow Import → Inspect → Validate → Clean/Standardize →
Descriptor/Fingerprint Generation → Split/Leakage Audit → Visualize → Export over
`services/datasets/client.ts` (upload/list/view/preview/column/remove/profile/
split) + `services/quality/client.ts` (the §15.4 data-quality report) +
`services/chemistry/client.ts` (descriptors/fingerprints). All 9 routes in
`routes/datasets.py` are consumed. The split route, however, forwards only
`stratify: bool`.

### Immediate improvements

**DS-1 — Scaffold-aware and group-aware splits at the Split stage**

| Field | Entry |
| --- | --- |
| ID / Feature | DS-1 — Expose the `split_indices` strategy vocabulary through `POST /datasets/{id}/split` |
| Current status | **Implemented (r373 / v1.2.246)** — the strategy lane + the group census + the Split-stage selector shipped; the engine, the route, the contract, the parser, and the page selector now carry ONE vocabulary (pinned by `tests/test_frontend_split_strategy.py`) |
| Problem | `src/pharmaml/ml/splits.py:368` `split_indices` implements `random` / `stratified` / `scaffold` (Murcko groups via `scaffold_groups`) / `group` (explicit labels) / `kennard_stone`, with `SPLIT_STRATEGIES` validation — but `routes/datasets.py:381` `POST /{id}/split` passes only `stratify: bool` into `data.py:596 split_table`, and the Split stage UI (`DatasetsPage.tsx` stage 6, "stratified, seeded, target-balanced") offers no strategy control. A QSAR split that ignores scaffold structure silently overstates generalization (near-duplicate scaffolds land on both sides). |
| Proposed solution | Add `strategy: Literal["random","stratified","scaffold","group","kennard_stone"]` (+ optional `smiles_column` for scaffold, `group_column` for group) to the split request; the route maps the frame to the existing engine and returns the same sides-view plus per-side scaffold-group counts (reuse `scaffold_groups` for the census). Datasets page stage 6 gains a strategy selector and a scaffold-census readout. |
| Scientific value | Scaffold splits are the accepted honest evaluation for QSAR (Bemis–Murcko groups held out whole); group splits serve the PK/PD and cell-line panels the toolkit already models. |
| Feasibility / Priority / Effort | High · **P0** (evaluation correctness) · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (RDKit for Murcko, scikit-learn — both pinned) |
| Performance impact | CPU: seconds (Murcko keys ~n·μs each); RAM: one frame copy |
| Related links | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html · https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html |
| Acceptance criteria | API tests: scaffold split keeps a shared scaffold wholly on one side (fail-first fixture); unknown strategy 422; the UI test pins the strategy selector and the group-census line; Python drift guard pins the strategy literal set. |

**DS-2 — A molecular standardization action (the "Clean" stage's missing verb)**

| Field | Entry |
| --- | --- |
| ID / Feature | DS-2 — `POST /api/v1/chemistry/standardize` over a dataset SMILES column |
| Current status | **Implemented (round 379, v1.2.251)** — the `POST /api/v1/chemistry/standardize` route over the real REINVENT-style engine served (the ops vocabulary over the engine's own callables, the derived dataset registered with the provenance block, the source never mutated, the honest rejected-rows accounting); the Clean stage renders the action + the before/after counts
| Problem | `src/pharmaml/chem/standardize.py` ships `reinvent_standardize_smiles`, `canonicalize_tautomer`, `reinvent_standardize_mol` (the "data.standardize" capability, `test_chem_standardize.py`), but `routes/chemistry.py` serves only `/descriptors` and `/fingerprints` — and the Datasets stage 4 "Clean / Standardize / Select Columns" (`DatasetsPage.tsx:1001`) is read-only: it displays the loader's column policy, it cannot standardize anything. |
| Proposed solution | Add `POST /api/v1/chemistry/standardize` taking `{dataset_id, column, ops: ["uncharge","canonical_tautomer","strip_salts", …]}` (job/sync modes); the body runs the existing engine and registers the standardized frame as a NEW dataset (provenance meta: source dataset_id + ops), never mutating the source — the §15.4 no-mutation rule preserved. Stage 4 gains the action and shows the before/after counts of changed SMILES. |
| Scientific value | Tautomer/charge/salt forms are the classic silent QSAR confounder; REINVENT-style standardization before descriptor computation is the field norm. |
| Feasibility / Priority / Effort | High · P1 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (RDKit pinned) |
| Performance impact | CPU seconds per 10k SMILES; disk: the new dataset record + frame |
| Related links | https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.html |
| Acceptance criteria | API test: salt-stripped output differs from input and source dataset bytes untouched (sha256 pinned); 422 on a non-SMILES column; vitest stage-4 action test; drift guard scanning that the route never writes to the source record. |

**DS-3 — Register computed descriptor/fingerprint frames as derived datasets**

| Field | Entry |
| --- | --- |
| ID / Feature | DS-3 — "Register as dataset" for the Descriptor/Fingerprint stage's result frames |
| Current status | Partial (chemistry route returns frames; the dataset-store registration Missing) |
| Problem | Stage 5 (`DatasetsPage.tsx:1042`) computes the real `rdkit_all` descriptor frame or fingerprint matrix through `POST /chemistry/descriptors|fingerprints`, but `handleChemResult` (`DatasetsPage.tsx:548`) only sets display state — the frames never enter the dataset store, so the ML/DL workspaces cannot train on the featurized matrix (their train routes read the registered dataset's own numeric columns; only the in-backend workflow templates compute descriptors internally). |
| Proposed solution | Extend the chemistry route response (or add `POST /chemistry/frames/register`) with an optional `register: bool` that loads the returned frame through the existing `DatasetStore` registration path (target column optional) with `source_kind: "derived-descriptors|fingerprints"` provenance; stage 5 gains a register button + a link "continue in Machine Learning". |
| Scientific value | Closes the classical QSAR loop inside the GUI: raw SMILES table → descriptors → model, which today only the backend template chains can do end-to-end. |
| Feasibility / Priority / Effort | High · P1 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (store + chemistry engine both in-tree) |
| Performance impact | CPU: one descriptor pass (already paid today); disk: the derived frame persisted |
| Related links | https://www.rdkit.org/docs/GettingStartedInPython.html#descriptor-calculation |
| Acceptance criteria | API test: `register: true` returns a real `dataset_id` whose view shows the descriptor columns and provenance meta; ML train over that id trains (smoke); vitest pins the register button + navigate hint. |

### Medium-term enhancements

**DS-4 — SDF / `.smi` molecular file ingestion**

| Field | Entry |
| --- | --- |
| ID / Feature | DS-4 — Accept SDF/MOL/SMI uploads in `POST /api/v1/datasets` |
| Current status | Partial (converter engines Implemented; upload route rejects the suffixes) |
| Problem | `routes/datasets.py:179` enforces `ALLOWED_UPLOAD_SUFFIXES` (CSV/TSV/TXT/XLSX) — an SDF is a 415. Yet `src/pharmaml/chem/converters.py` implements `sdf_to_table` / `table_to_sdf` (the "data.convert" capability, `test_format_converters.py`), i.e. the molecule-aware reader exists and is unused by the API. |
| Proposed solution | Extend the upload route: for `.sdf`/`.sdf.gz`/`.mol`/`.smi`, read through `sdf_to_table` (title → optional ID column, properties → columns, canonical SMILES column), then register with the same `DatasetRecord` path; the Import stage's accept/label list updates honestly. |
| Scientific value | SDF is the exchange format every compound collection ships in; today a user must convert externally, off-toolkit. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (RDKit) |
| Performance impact | CPU seconds per 10k records; RAM: the parsed frame |
| Related links | https://www.rdkit.org/docs/source/rdkit.Chem.SDMolSupplier.html |
| Acceptance criteria | API test round-trips a 3-record SDF into a queryable dataset (3 rows, expected SMILES column); the 415 stays for unknown suffixes; upload-size cap test unchanged. |

**DS-5 — The pIC50 endpoint-preparation lane**

| Field | Entry |
| --- | --- |
| ID / Feature | DS-5 — `prepare_pic50_dataset` as the potency-target preparation stage |
| Current status | Partial (engine Implemented; route Missing) |
| Problem | `chem/converters.py:549 prepare_pic50_dataset` (+ `add_pic50_column`, `ic50_to_pic50`, `ki_to_pki`, `EndpointDatasetReport`, `pic50_from_ic50_series`) implements the IC50/Ki → pIC50/pKi target transform with unit scaling and an honest per-molecule convertibility report — none of it is reachable from any route; a user with an IC50_nM column must hand-compute the log transform before upload. |
| Proposed solution | `POST /api/v1/datasets/{id}/endpoint` taking `{value_column, unit, out_col}`; registers the derived dataset (pIC50 column added, unconvertible rows reported, `EndpointDatasetReport` rendered in the response); the Datasets stage 4/6 area gains the lane. |
| Scientific value | pIC50 (not IC50) is the regression target every QSAR model expects; the report's convertibility accounting is exactly the honesty the §15.4 quality card established. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: milliseconds |
| Related links | https://www.rdkit.org/docs/rdkit.Chem.MolStandardize.html (unit-aware potency handling context) |
| Acceptance criteria | API test: nM IC50 1000 → pIC50 6.0 exactly (the engine's exact-decimal-powers guarantee), unconvertible row reported not dropped silently; vitest renders the report's counts. |

---

## 3. Machine Learning

Current state: `frontend/src/features/ml/MLPage.tsx` (8,626 lines) runs the
23-stage workflow over `services/ml/client.ts` (6,354 lines: catalog/train/
predict/explain, AD, semi-supervised, ADMET, DTA, repurposing, drug response,
pharmacogenomics, multi-objective, multi-response, RSM, workflow builder) plus
`UnsupervisedLearningLane.tsx` (DBSCAN/mean-shift/spectral + Isolation Forest
+ the experimental RL grid-world). Every route in `routes/ml.py`,
`routes/semi_supervised.py`, `routes/unsupervised.py`, `routes/reinforce.py`
and the satellite routers is consumed — except two whole families below.

### Immediate improvements

**ML-1 — Conformal intervals, conformal label sets, and coverage diagnostics in the UI**

| Field | Entry |
| --- | --- |
| ID / Feature | ML-1 — A client + "Uncertainty" panel for the `/api/v1/model-evaluation` family |
| Current status | **Implemented (round 376, v1.2.248)** — `modelEvaluationApi`'s four typed operations + the stage-19 conformal panel (the bounded column bridge, the interval/label-set readouts, the Wilson-bounded coverage table, the catalog's honesty notes) |
| Problem | `routes/model_evaluation.py` serves four routes — `/catalog`, `/conformal-regression` (split-conformal intervals + optional group/scaffold-wise coverage), `/conformal-classification` (marginal and label-conditional sets), `/coverage-diagnostics` (Wilson-bounded coverage report) — all sync-only over session-trained models. `rg model-evaluation frontend/src` matches only `types/api.d.ts`: no service function, no UI. The page's stage 19 "Explainability & Uncertainty" (`MLPage.tsx:5413`) therefore offers only the SHAP lane + the AD bootstrap, and its own caption admits the bootstrap is "the uncertainty surface that also runs today". |
| Proposed solution | Add `modelEvaluationApi` (the 4 typed operations, strict parsers) to the ML client; extend stage 19 with a conformal panel: pick a trained model + a held-out calibration slice (the bounded column bridge already used by the AD stage), α + (for classification) conditional mode; render the interval radius, per-row bounds, empirical coverage, and the group/scaffold coverage table. Optionally add `quantile_prediction_intervals` (`ml/applicability.py:721`, also un-routed) as a second engine behind the same panel. |
| Scientific value | Distribution-free prediction intervals/label sets are the honest answer to "how wrong might this prediction be" — the missing half of a point-prediction readout, without the bootstrap's refit cost. |
| Feasibility / Priority / Effort | High · P1 · Medium |
| Backend / Frontend changes | not required / required |
| Dependencies | existing (routes + engines + API tests already in-tree) |
| Performance impact | CPU: one predict pass + a quantile (sync routes by design); RAM: the calibration slice |
| Related links | https://mapie.readthedocs.io/en/latest/ · https://github.com/aangelopoulos/conformal-prediction |
| Acceptance criteria | Vitest lane over the new client (payload narrowing, malformed-throw); page test rendering interval bounds and the coverage table from a mocked route payload; drift guard pinning the four paths remain consumed (the cross-check this roadmap opened with). |

**ML-2 — The kNN-distance and domain-box AD lanes**

| Field | Entry |
| --- | --- |
| ID / Feature | ML-2 — Expose `POST /applicability-domain/knn` and `/domain-box` in the AD stage |
| Current status | Partial (backend Implemented; frontend consumer Missing) |
| Problem | `routes/applicability_domain.py` documents four entry points; the ML page's stage 10 consumes `/analyze` and `/leverage-williams` only. `/knn` (StandardScaler + kNN distance AD, three metrics, self-match excluded — usable on raw feature matrices without a trained model) and `/domain-box` (the persisted feature-range box checked against new rows) have zero frontend callers. |
| Proposed solution | Extend the AD stage with a "distance-only AD" sub-lane (two raw matrices + k/percentile/metric → inside/outside verdicts) and a box-check sub-lane (the trained model's persisted ranges vs pasted/bridged rows), both rendered beside the existing analyze/Williams cards. |
| Scientific value | The distance-only lane serves the unsupervised case (no trained model yet — is the query set inside the training space at all?); the box check is the cheap per-feature range verdict the full kNN analysis overkills. |
| Feasibility / Priority / Effort | High · P2 · Small |
| Backend / Frontend changes | not required / required |
| Dependencies | existing |
| Performance impact | CPU: O(n·k) neighbours over ≤1000×50 matrices (the route's own caps) |
| Related links | — |
| Acceptance criteria | Vitest pair over the two new client operations; page test that a matrix-only user can reach a verdict without a trained model; the honest-422 lane (missing k / empty rows) surfaced, not swallowed. |

### Medium-term enhancements

**ML-3 — Scaffold/group split strategy inside the Train lane**

| Field | Entry |
| --- | --- |
| ID / Feature | ML-3 — A `split` section on `TrainRequest` riding `ml/splits.py` |
| Current status | **Implemented (round 375, v1.2.247)** — the carve runs over the full registered frame before the fit, the sides + group census + `shared_groups` verdict ride the response and the §15.2 record at both exits |
| Problem | `routes/ml.py:96` `TrainRequest` exposes `cv: int` (2–10) and nothing else split-shaped; `_train_body` calls `train_model(... cv=cv ...)` whose validation is the stratified-k-fold inside `ml/preprocessing.py:282 make_train_test_split` (stratify=True default). The `ml/splits.py` vocabulary (scaffold/group/kennard-stone) is therefore unreachable from the GUI's training path, even though the Datasets-side DS-1 fix will produce scaffold-aware held-out frames the train lane cannot itself request. |
| Proposed solution | Add optional `split: {strategy, smiles_column?, group_column?}` to TrainRequest; `_train_body` slices the training frame with `split_indices` before the fit and reports the strategy + group counts in the train response (and into the §15.2 run record the route already writes at both exits). The Configure Split/CV stage (`MLPage.tsx:3249`) gains the selector. |
| Scientific value | Cross-validated scores under scaffold-separated folds are the trustworthy headline metric for a QSAR model; today the GUI can only report random-stratified folds. |
| Feasibility / Priority / Effort | High · **P1** (correctness of the headline metric) · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (DS-1's route work lands the shared engine wiring first) |
| Performance impact | CPU: the Murcko keying pass + the same fit cost |
| Related links | https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data |
| Acceptance criteria | API test: same seed, scaffold strategy puts a shared scaffold in one fold only (fail-first fixture); response/run-record fields pinned; page test for the selector; DS-1's drift guard extended to the train route. |

**ML-4 — The remaining unsupervised families (k-means, GMM, hierarchical, PCA, KDE)**

| Field | Entry |
| --- | --- |
| ID / Feature | ML-4 — Extend `/api/v1/unsupervised` beyond DBSCAN/mean-shift/spectral/IsolationForest |
| Current status | Partial (engines Implemented and tested; route covers 3 clustering + 1 anomaly method) |
| Problem | `src/pharmaml/ml/unsupervised.py` implements `pca_reduce`, `kmeans_cluster` (+ silhouette), `gmm_cluster` (+ BIC selection), `kde_fit`, `hierarchical_cluster` (`tests/test_unsupervised.py`); `ml/manifold.py` and `ml/umap.py` add the embedding families. `routes/unsupervised.py` wraps only `ml/clustering.py` (dbscan/meanshift/spectral) + `ml/anomaly.py`; the frontend lane (`UnsupervisedLearningLane.tsx:66` `ClusterMethod = "dbscan"|"meanshift"|"spectral"`) mirrors exactly that. |
| Proposed solution | Add `method` values `kmeans`/`gmm`/`hierarchical` (+ a `/unsupervised/pca` sibling for `pca_reduce`) to the cluster route, each returning the engine's own diagnostics (inertia/silhouette, BIC table, linkage tree order) with the existing honest-degenerate reporting; extend the lane's method chips and result cards. The umap/manifold engines ride the same route as a follow-on `method: "umap"` projection. |
| Scientific value | GMM/BIC and k-means/silhouette are the standard chemical-space segmentation tools (series vs scaffold clusters); PCA is the first plot every chemist asks for on a descriptor matrix. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (scikit-learn pinned; umap engine is in-house numpy, no new dep) |
| Performance impact | CPU: seconds at the route's bounded sizes; RAM: one similarity matrix for hierarchical |
| Related links | https://scikit-learn.org/stable/modules/clustering.html · https://umap-learn.readthedocs.io/ |
| Acceptance criteria | API tests pinning the new method responses incl. the GMM BIC table and a degenerate single-cluster honest answer; vitest chip/result-card tests; drift guard that the methods literal equals the engine list. |

**ML-5 — A data-drift lane over `ml/drift.py`**

| Field | Entry |
| --- | --- |
| ID / Feature | ML-5 — `POST /api/v1/ml/drift` comparing a training sketch against new data |
| Current status | Partial (engine + CLI Implemented; API route and UI Missing) |
| Problem | `src/pharmaml/ml/drift.py` `detect_drift` (per-feature PSI + the training-sketch machinery, `tests/test_data_drift_diagnostic.py`, and the CLI `pharmaml ml drift` at `cli.py:417`) has no route — the batch-training CLI even wires `--no-drift-check` — but a GUI user re-scoring a deployed model on fresh data has no drift readout at all. |
| Proposed solution | `POST /api/v1/ml/drift` taking `{model_id | dataset_id (reference), new_dataset_id}`; the body reuses the engine's per-feature sketches over the stored frames and returns the PSI table + the drifted-feature list; an ML Evaluate-stage card renders it ("score drift before you trust the metrics"). |
| Scientific value | PSI drift is the standard first-line model-monitoring verdict for production QSAR — the toolkit's own CLI treats it as mandatory for batch scoring, the GUI should match. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: histogram sketches, sub-second at session sizes |
| Related links | https://alibi-detect.readthedocs.io/en/latest/ (the canonical drift-detection reference) |
| Acceptance criteria | API test: a shifted column flags, an identical one doesn't (fail-first); page test renders the PSI table with the drifted rows highlighted; drift guard pins the threshold constant the engine exports. |

---

## 4. Deep Learning

Current state: `frontend/src/features/dl/DLPage.tsx` (3,178 lines) runs the
11-stage workflow (Select Dataset → Validate → Preprocess → Select Architecture
→ Configure Model → Configure Training → Train → Monitor → Evaluate → Visualize →
Save Checkpoint/Export) over `services/dl/client.ts`; the tabular torch
families (MLP/residual/transformer), the GNN lane, the sequence-CNN lane, and
the tabular autoencoder lane are all live. All 8 `routes/dl.py` paths are
consumed. The efficiency and explainability engines below are not.

### Medium-term enhancements

**DL-1 — Hyperparameter search for the tabular families**

| Field | Entry |
| --- | --- |
| ID / Feature | DL-1 — A `tune` lane for `POST /api/v1/dl/train` riding `ml/hpo.py` |
| Current status | Partial (HPO engine Implemented; DL train request has no search mode) |
| Problem | `routes/dl.py:129` `DLTrainRequest` takes a flat `hyperparams: dict` — one fixed configuration — while the classical ML lane's `tune` flag runs the in-tree TPE/Hyperband optimizer (`ml/hpo.py:646 tune_model`, `hpo_presets.py`, `tests/test_hpo_resumability.py`). The DL page's "Configure Training" stage (`DLPage.tsx:1689`) therefore edits single values by hand; there is no search over epochs/lr/dropout/hidden layers. |
| Proposed solution | Add `tune: bool` + a bounded search-space block (per-key low/high or choices over the same `_HYPERPARAM_SPEC` vocabulary the route already validates) to `DLTrainRequest`; the job body samples through the existing TPE/Hyperband classes per candidate fit and reports the trial table + the best configuration, mirroring the ML lane's TuneReport surface. |
| Scientific value | LR/dropout/depth dominate tabular-DL quality; a seeded TPE search beats hand-tuning on the same data in the same wall-clock budget. |
| Feasibility / Priority / Effort | Medium · P2 · Large |
| Backend / Frontend changes | required / required |
| Dependencies | existing (torch + the in-house HPO engine; no optuna dependency) |
| Performance impact | CPU/GPU: k× the single-fit cost (k = trials, bounded by the Hyperband brackets); RAM: one model at a time |
| Related links | https://optuna.readthedocs.io/en/stable/ (the canonical TPE reference) |
| Acceptance criteria | API test: tune=true returns a trial table whose best config is among the candidates and the refit metric ≥ the fixed-config baseline on a seeded fixture; cancellation between trials lands CANCELLED (the population-route precedent); page test for the search panel. |

**DL-2 — GNN explainability over trained graph models**

| Field | Entry |
| --- | --- |
| ID / Feature | DL-2 — `POST /api/v1/dl/gnn-explain` + an edge-attribution panel |
| Current status | Partial (engine Implemented and tested; route Missing; UI Missing) |
| Problem | `src/pharmaml/ml/gnnxai.py` ships `GNNExplainer`, `PGExplainer`, `AttentionExplainer`, `SubgraphX` and the fidelity/sparsity/accuracy-at-k metrics (`tests/test_gnn.py`); the DL GNN lane trains and predicts (`/dl/gnn-train|predict` consumed) but offers no explanation — a user cannot see which edges/atoms drove a graph prediction. |
| Proposed solution | Add `/dl/gnn-explain` taking `{model_id, graphs, method, target}`; the body runs the chosen explainer and returns the per-edge importance + the fidelity metrics; the GNN lane's result card gains the edge-importance table and (PlotlyFigure) a sized-edge scatter over the graph's 2-D layout the backend computes. |
| Scientific value | Atom/bond attribution is what makes a GNN verdict actionable for a medicinal chemist — the same role SHAP plays for the tabular models in stage 19. |
| Feasibility / Priority / Effort | Medium · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (torch) |
| Performance impact | CPU/GPU: seconds per explained graph (masked forward passes) |
| Related links | https://arxiv.org/abs/1905.13686 (GNNExplainer) |
| Acceptance criteria | API test: fidelity_plus/minus computed on a seeded two-graph fixture and the explained subgraph ≤ full graph size; 404/422 lanes honest; vitest renders the edge table + the Plotly edge-size map. |

**DL-3 — The efficiency lab: weight quantization + pruning reports**

| Field | Entry |
| --- | --- |
| ID / Feature | DL-3 — `POST /api/v1/dl/efficiency` over `ml/quantization.py` + `ml/pruning.py` |
| Current status | Partial (engines Implemented and tested; route Missing; UI Missing) |
| Problem | `ml/quantization.py` (NF4 pack with double quantization, int8 absmax, `llm_int8_matmul`, `quantization_report` — `tests/test_quantization.py`) and `ml/pruning.py` (`apply_magnitude_pruning`, `iterative_prune_and_retrain` — `tests/test_pruning.py`) are the toolkit's "DL efficiency lab" (the `dl.quantization`/`dl.pruning` capability rows), none reachable from a route: the DL page's stage 11 (`DLPage.tsx:2070`) persists the full-precision joblib bundle only. |
| Proposed solution | Add `/dl/efficiency` taking `{model_id, mode: "quantize_int8"|"quantize_nf4"|"prune", level}`; the body loads the persisted estimator's weight arrays through the engines and answers with the `quantization_report` (bits/weight, reconstruction error) or the sparsity + the metric delta after pruning; register the report as a table artifact. |
| Scientific value | Deployment footprint (int8/NF4) and pruned retraining are the standard efficiency levers for shipping a trained model on a modest desktop — the packaged-app context this toolkit ships in. |
| Feasibility / Priority / Effort | Medium · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (numpy-based engines; no new dep) |
| Performance impact | CPU: the quantization/prune pass is one matrix sweep; RAM: a weight copy |
| Related links | https://docs.pytorch.org/docs/stable/quantization.html · https://github.com/bitsandbytes-foundation/bitsandbytes (the NF4 precedent) |
| Acceptance criteria | API test: int8 report's dequantized error within the engine's own documented bound on a seeded matrix; prune at 50% sparsity returns exactly 50% zeros; vitest renders the report card. |

### Long-term research extensions

**DL-4 — The D-MPNN molecular property predictor**

| Field | Entry |
| --- | --- |
| ID / Feature | DL-4 — Expose `ml/mpnn.py` (numpy D-MPNN) as a SMILES-input training lane |
| Current status | Partial (engine Implemented; route Missing; the GNN lane needs hand-built graph specs) |
| Problem | `src/pharmaml/ml/mpnn.py` implements the chemprop-style directed-message-passing network over RDKit graphs (`build_batch` from SMILES, `DMPNN`, `gradient_check` — tested in-tree), but the only graph-model route (`/dl/gnn-train`) takes explicit `GnnGraphSpec` edge/node lists the GUI user must construct; there is no "give a SMILES column, get a property model" lane anywhere. |
| Proposed solution | `POST /api/v1/dl/mpnn-train` taking `{dataset_id, smiles_column, target_column, epochs, …}`: `build_batch` featurizes server-side, the DMPNN fits as a managed job, predictions ride `/dl/mpnn-predict` over raw SMILES. Requires scientific validation of the numpy trainer against a published MoleculeNet-scale result before the UI lands (the honest validation step is why this sits long-term). |
| Scientific value | D-MPNNs are the reference architecture for molecular property prediction from structure alone — the workflow the toolkit's GNN lane gestures at but never reaches from plain SMILES. |
| Feasibility / Priority / Effort | Medium · P3 · Large |
| Backend / Frontend changes | required / required |
| Dependencies | existing (RDKit + numpy; no torch needed) |
| Performance impact | CPU: minutes for small corpora (numpy trainer); RAM: the message arrays |
| Related links | https://github.com/chemprop/chemprop |
| Acceptance criteria | A held-out RMSE on a seeded in-repo dataset within the pre-registered tolerance + API/vitest lanes mirroring the GNN pair. |

---

## 5. Generative AI

Current state: `frontend/src/features/generative/GenerativePage.tsx` (2,896
lines) runs the 10-stage workflow (Select Method → Starting Molecules →
Constraints/Objectives → Generate → Validate → Properties → Filter → Rank →
Visualize → Export); every `routes/generative.py` path (RNN prior
train/sample, GA optimize, feature-space GAN/Diffusion lanes, priors,
objectives) is consumed. The molecular-RL, MCTS, MOSES-metrics, and advanced
operator surfaces below are not.

### Immediate improvements

**GEN-1 — The GA operator vocabulary (bioisostere / topology / scaffold-hop)**

| Field | Entry |
| --- | --- |
| ID / Feature | GEN-1 — Expose `mutation` / `crossover` modes on `POST /api/v1/generative/optimize` |
| Current status | **Implemented (round 377, v1.2.249)** — the mutation/crossover Literals + the `min_heavy_atoms` child gate ride `OptimizeRequest` and forward verbatim into `GeneticOptimizer`; the Configure stage carries the two selects + the conditional gate input |
| Problem | `generative/genetic.py` accepts `mutation="smiles"|"topology"|"bioisostere"` and crossover modes including `"scaffold_hop"` (BREED-style linker excision/splicing over `scaffold_hop_crossover`, `bioisostere_mutate`, `ring_topology_mutate` — wired in at `genetic.py:131-140`), but `routes/generative.py:550` builds `ga_kwargs` from exactly eight numeric knobs (`population_size, generations, mutation_rate, crossover_rate, elitism, tournament_k, max_smiles_length, seed`) — the operator modes are unreachable from the API and the page's Configure stage. |
| Proposed solution | Add `mutation: Literal[...]` and `crossover: Literal[...]` (+ the mode-specific rates the engine already validates) to `OptimizeRequest`; forward verbatim into `GeneticOptimizer`. The page's Constraints/Configure stage gains the two selects with the engine's own descriptions. |
| Scientific value | Scaffold-hop crossover and bioisostere mutation are the moves that actually explore chemotypes — plain SMILES-point mutation alone converges to the seed's neighborhood. |
| Feasibility / Priority / Effort | High · P1 · Small |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: unchanged per generation; RDKit perceive cost per mutation only |
| Related links | https://github.com/BenevolentAI/guacamol (the benchmark these moves target) |
| Acceptance criteria | API test: `crossover="scaffold_hop"` produces children whose Murcko scaffold differs from both parents on a seeded fixture (fail-first); unknown mode 422; vitest select-passthrough test. |

### Medium-term enhancements

**GEN-2 — MOSES generation metrics (validity / uniqueness / novelty / diversity)**

| Field | Entry |
| --- | --- |
| ID / Feature | GEN-2 — `POST /api/v1/generative/metrics` over `moses_metrics.generation_metrics` |
| Current status | **Implemented (round 378, v1.2.250)** — the full MOSES-lite audit served (validity, uniqueness, novelty vs an inline or dataset reference, the internal diversities, the property-shift table, the honest not-measured nulls); the Validate stage renders the metrics card |
| Problem | `src/pharmaml/generative/moses_metrics.py:470 generation_metrics` computes the MOSES suite — validity fraction, uniqueness, novelty vs a reference corpus, internal diversity, property means and shifts (`tests/test_generative.py`); `routes/generative.py` serves none of it. The page's Validate stage (`GenerativePage.tsx:2133`) infers validity from the descriptor engine's parse, and nothing anywhere computes uniqueness, novelty, or internal diversity. |
| Proposed solution | Add `/generative/metrics` taking `{generated: [...], reference: [...] | dataset_id+column}`; the response is the full metrics dict + the property-shift table; the Validate stage gains the metrics card (validity confirmed against the parse verdict, uniqueness, novelty vs the user's reference corpus). |
| Scientific value | Without novelty/internal-diversity a generator's "top 10" may be one scaffold printed ten ways — the exact failure mode the MOSES suite exists to catch. |
| Feasibility / Priority / Effort | High · P1 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (RDKit fingerprints + the reference-corpus resolution the prior-train lane already implements) |
| Performance impact | CPU: one Morgan bit matrix over both sets — seconds at the route's bounded sizes |
| Related links | https://github.com/molecularsets/moses |
| Acceptance criteria | API test: a corpus containing duplicates reports uniqueness < 1 and the generated-vs-reference novelty matches a hand-computed fixture; vitest renders all five headline metrics with honest nulls when no reference is given. |

**GEN-3 — The REINVENT-style molecular RL optimizer**

| Field | Entry |
| --- | --- |
| ID / Feature | GEN-3 — `POST /api/v1/generative/rl-optimize` over `run_rl_optimization` |
| Current status | Partial (engine Implemented; route Missing; method selector offers RNN prior + GA only) |
| Problem | `generative/rl.py` implements `ReinventOptimizer` + `run_rl_optimization` (policy-gradient fine-tuning of an RNN prior against the scoring function, with experience replay — `probe/scratch` history and engine tests in-tree), but the page's method cards (`GenerativePage.tsx:1303`) are exactly two ("the backend's two REAL engines": RNN prior + genetic), and no route serves the RL engine. (The `routes/reinforce.py` the frontend *does* consume is the experimental grid-world `ml.reinforce`, not molecular generation.) |
| Proposed solution | Add `/generative/rl-optimize` taking `{prior_id, objective, steps, batch_size, lr, …}` as a managed job with per-step telemetry (the RNN train lane's event pattern); the method selector gains a third card wired to a selected prior. |
| Scientific value | RL fine-tuning is the REINVENT-family recipe for steering a prior toward an objective — the standard de-novo optimization this workspace otherwise only reaches through the GA. |
| Feasibility / Priority / Effort | Medium · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: minutes per run at the route's bounds; RAM: the prior network |
| Related links | https://github.com/MolecularAI/Reinvent |
| Acceptance criteria | API test: score trajectory improves over the seeded steps on a QED-like fixture; cancellation mid-run lands CANCELLED; vitest pins the third method card + the step curve through PlotlyFigure. |

**GEN-4 — The MCTS molecule designer**

| Field | Entry |
| --- | --- |
| ID / Feature | GEN-4 — `POST /api/v1/generative/mcts-optimize` over `run_mcts_optimization` |
| Current status | Partial (engine Implemented, `tests/test_mcts.py`; route Missing) |
| Problem | `generative/mcts.py` (MctsOptimizer + `run_mcts_optimization`) has no route and no UI card; the workspace's exploration budget method count stays at (post-GEN-3) three while the backend ships a fourth. |
| Proposed solution | Same shape as GEN-3: a managed job over the tree search with per-iteration telemetry, a fourth method card, the top-molecules table reusing the existing rank stage components. |
| Scientific value | MCTS explores the fragment tree with explicit budget/UCB control — complementary to GA (population) and RL (gradient); its rollouts are measurable against the same objective catalog. |
| Feasibility / Priority / Effort | Medium · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: the rollout budget (bounded, seeded); RAM: the tree |
| Related links | https://github.com/BenevolentAI/guacamol (the MCTS baseline setting) |
| Acceptance criteria | API test: seeded determinism — same seed, same top molecule; node budget honored; vitest card + ranked-table test. |

### Long-term research extensions

**GEN-5 — A SELFIES representation mode for the generators**

| Field | Entry |
| --- | --- |
| ID / Feature | GEN-5 — SELFIES-encoded generation (valid-by-construction molecules) |
| Current status | Partial (codec Implemented; the RNN/GA operate on SMILES strings only) |
| Problem | `chem/selfies_codec.py` ships the SELFIES round-trip (the optional `selfies>=2.1,<3` pin, the `/system/diagnostics` round-trip probe), but every generative engine (RNN prior, GA mutate/crossover) operates on raw SMILES, where a single token edit can invalidate the string — the page's own Validate stage exists largely to count exactly those failures. |
| Proposed solution | A `representation: "smiles"|"selfies"` switch on the prior-train/optimize requests: the corpus is encoded through the existing codec before training, samples decoded for the working set; invalidity collapses toward zero and the Validate stage's parse verdict becomes the round-trip check. Needs a scientific validation pass (corpus-coverage statistics) before enabling by default — hence long-term. |
| Scientific value | SELFIES guarantees chemical validity under arbitrary edits; the invalid-fraction metric stops measuring "string fragility" and starts measuring "out-of-distribution only". |
| Feasibility / Priority / Effort | Medium · P3 · Large |
| Backend / Frontend changes | required / required |
| Dependencies | existing optional (`selfies` already pinned optional and probed) |
| Performance impact | CPU: codec pass over corpus/samples (pure Python, seconds per 10k); RAM unchanged |
| Related links | https://github.com/aspuru-guz-selfies/selfies |
| Acceptance criteria | API test: 10k random token edits of encoded strings → zero sanitization failures (the codec's own guarantee, pinned); vitest representation switch; the metrics card (GEN-2) reporting validity 1.0 in SELFIES mode. |

---

## 6. Molecular Docking

Current state: `frontend/src/features/docking/DockingPage.tsx` (3,258 lines)
runs the 10-stage workflow (Select Ligands → Prepare Ligands → Prepare Receptor
→ Binding Site/Grid [with the synchronized `GridBoxViewer` 3-D view] →
Configure Engine → Run Docking → Inspect Poses → Analyze Interactions →
Visualize → Export) over `services/docking/client.ts` (1,155 lines). All 13
`routes/docking.py` paths are consumed — the gaps below are engine-side
capabilities the route never forwards.

### Immediate improvements

**DOCK-1 — A grid-spacing control on `DockRequest`**

| Field | Entry |
| --- | --- |
| ID / Feature | DOCK-1 — Expose `spacing` (AutoDock4 grid quantization) in `POST /docking/dock` |
| Current status | Partial (engine parameter Implemented; request model omits it) |
| Problem | `dock_smiles_array(..., spacing=0.375, ...)` (`smiles_docking.py:183`) quantizes the AutoDock4 box to `npts = round(size/spacing)+1` — but `routes/docking.py:207` `DockRequest` carries `center` and `size` only, and the `GridBoxViewer` itself pins the honesty note "the backend DockRequest carries center + size only, NO spacing field — the engine's own default applies, stated rather than an imagined control" (`GridBoxViewer.tsx` module doc). The stated-not-controllable note is exactly the acknowledged gap. |
| Proposed solution | Add `spacing: float = 0.375` (ge/le bounds mirroring the engine's) to `DockRequest`, forwarded verbatim; the viewer's spacing note updates from "default applies" to the live value, and the Engine stage gains the input (Vina ignores it; that is stated). |
| Scientific value | AutoGrid span ≠ requested size unless spacing is deliberate (the engine's own docstring warns the bundled AutoGrid4 interprets even npts as intervals) — grid control is a real reproducibility knob, not chrome. |
| Feasibility / Priority / Effort | High · P2 · Small |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | None (a config scalar; grid memory scales with npts³ at fine spacing) |
| Related links | https://autodock-vina.readthedocs.io/en/latest/docking_python.html |
| Acceptance criteria | API test: spacing 0.375 vs 0.75 yields the documented different npts in the AutoDock4 campaign path (fixture-pinned); vitest pins the live value into the viewer's readout. |

### Medium-term enhancements

**DOCK-2 — Durable, resumable docking campaigns**

| Field | Entry |
| --- | --- |
| ID / Feature | DOCK-2 — A campaign route over `run_vina_batch` (skip-existing resume + pre-flight census) |
| Current status | Partial (engine Implemented and tested; route uses a throwaway temp dir) |
| Problem | `docking/campaign.py:726 run_vina_batch(skip_existing=True)` + `vina_batch_progress` (the pre-flight resume census) + the `campaign_launch/preview/status` modules exist for resumable multi-ligand campaigns — but `routes/docking.py:501` `_dock_body` runs `dock_smiles_array` inside `tempfile.TemporaryDirectory(...)` with `keep_folders=False` and no `checkpoint_dir`: cancel a 100-ligand job at ligand 90 and every completed pose is destroyed with the temp dir; re-running starts from zero. |
| Proposed solution | Add `POST /api/v1/docking/campaign` (job mode) writing to a durable per-campaign folder under the persistence data dir (the same layout the artifact vault uses), forwarding `checkpoint_dir/checkpoint_every/checkpoint_resume` and the progress census; a resume lane reads the census and reports `done/total/remaining`. The Run Docking stage gains the campaign submit + resume controls. |
| Scientific value | Ligand-library screens are hours-long; honest resume is the difference between a cancelled screen being a partial result (the engine already guarantees partial rows) and a total loss. |
| Feasibility / Priority / Effort | Medium · P2 · Large |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | Disk: poses + logs per ligand under the data dir (bounded by the campaign size); CPU unchanged |
| Related links | https://autodock-vina.readthedocs.io/en/latest/docking_python.html |
| Acceptance criteria | API test: cancel at a checkpoint, re-submit with resume, the census reports the completed ligands as skipped and total wall-time halves on the fixture; no orphan temp dirs; vitest resume-control test. |

**DOCK-3 — PDB-text receptor preparation (the Meeko lane)**

| Field | Entry |
| --- | --- |
| ID / Feature | DOCK-3 — Accept a raw PDB (not only PDBQT) in `/docking/receptor/prepare` |
| Current status | Partial (engine branch Implemented; route takes PDBQT text only) |
| Problem | `docking/receptor.py`'s `prepare_protein` "imports a protein from a PDB file (or an already-prepared PDBQT)", typing and charging via Meeko (`source: "meeko"`) — but `routes/docking.py:132` `ReceptorPrepareRequest` has `pdbqt_text` + `filename: str = "receptor.pdbqt"` only; a user with a banked PDB must convert externally before the toolkit can touch it, and the engine's PDB branch is dead code from the API's perspective. |
| Proposed solution | Add `format: Literal["pdbqt","pdb"]` to the request; for `pdb` the temp workspace writes the text as `.pdb` and the same `prepare_protein` call runs its Meeko branch (meeko is a pinned dependency, `pyproject.toml:49`); the response's `report.source` (`"meeko"`) tells the UI which path ran. The Prepare Receptor stage gains a format toggle. |
| Scientific value | Receptors arrive as PDB files; the Meeko branch preserves AD4 typing/charges rather than asking the user to hand-prepare. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (meeko>=0.5 pinned) |
| Performance impact | CPU: seconds per receptor; RAM: the typed molecule |
| Related links | https://meeko.readthedocs.io/en/latest/ |
| Acceptance criteria | API test over a small PDB fixture: `source: "meeko"`, output atoms > 0, AD4 types present; the PDBQT path byte-identical to today's (regression pin); vitest format-toggle test. |

**DOCK-4 — The docking figure families as registered artifacts**

| Field | Entry |
| --- | --- |
| ID / Feature | DOCK-4 — Serve `viz/docking_plots.py` (energy distribution, top ligands, contact map, PLIF heatmap) |
| Current status | Partial (figure engines Implemented; no route serves them; no UI) |
| Problem | `src/pharmaml/viz/docking_plots.py` implements seven families — `plot_receptor_atom_types`, `plot_energy_distribution`, `plot_top_ligands`, `plot_consensus_ranking`, `plot_contact_map`, `plot_plif_heatmap` (+ the input-checked `plot_*` contract) — while `routes/visualization.py` serves statplots/pharma/MD/table/report only and `routes/docking.py` serves no figure route: a docked campaign's result is a table in the UI, never a figure artifact. |
| Proposed solution | Add `POST /api/v1/docking/figures/{family}` (job/sync) taking the rows the dock/contacts routes already return (or their artifact ids), rendering through the engines + the shared `export_figure` four-format stack, registering the artifact; the Inspect Poses / Analyze Interactions stages gain "plot" buttons whose results ride the existing artifact-row download lane. |
| Scientific value | An energy-distribution / contact-map figure is the review artifact a docking report actually carries; today the user exports tables and plots externally. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (matplotlib export stack already pinned) |
| Performance impact | CPU: seconds; disk: one artifact per figure |
| Related links | https://autodock-vina.readthedocs.io/en/latest/docking_python.html |
| Acceptance criteria | API test: energy distribution over a 5-row frame registers a downloadable artifact with the digest contract; contact map over an empty contacts list 422s (the engine's own `_check_contacts`); vitest plot-button + artifact-row test. |

---

## 7. PK/PD

Current state: `frontend/src/features/pkpd/PKPDPage.tsx` (2,092 lines) runs
the 8-stage workflow (Import → Inspect → Select Model → Configure →
Fit/Simulate → Diagnostics → Exposure/Response → Export) over
`services/pkpd/client.ts`; all nine `routes/pkpd.py` paths (NCA,
steady-state, IV bolus, Bateman, superposition, PBPK, PD effect table,
population job, catalog) are consumed, with the concentration-time profile
drawn through the house `PlotlyFigure`. The import stage is paste-only.

### Immediate improvements

**PK-1 — The dataset bridge for concentration–time series**

| Field | Entry |
| --- | --- |
| ID / Feature | PK-1 — Import stage reads a registered dataset's time/conc columns |
| Current status | Partial (the bounded column bridge exists app-wide; this page never uses it) |
| Problem | `PKPDPage.tsx:1150` stage 1 is a paste textarea ("the pasted concentration-time series (time conc per line)") + two hardcoded samples — while `services/datasets/client.ts:204 column()` is the established bridge the ML/DL/Generative pages use to pull a registered column, and the shipped `examples/data/pk_profile.csv` loads as a dataset through the normal Datasets workflow. A user who imported their PK table as a dataset must retype it into the textarea. |
| Proposed solution | Add a dataset + column pair picker (times column, concentration column) to stage 1: the page pulls both columns through the existing `column()` bridge, parses to the same series view, and the textarea remains the free-form lane. Sample-data section already covers the file, so the demo path stays one-click. |
| Scientific value | Removes the retype lane between the toolkit's own dataset store and its pharmacometrics engine — the exact bridge every other workflow already has. |
| Feasibility / Priority / Effort | High · P1 · Small |
| Backend / Frontend changes | not required / required |
| Dependencies | existing |
| Performance impact | One bounded GET per column |
| Related links | — |
| Acceptance criteria | Vitest: selecting a dataset with `time`/`conc` columns populates the series view (row-count + t/C ranges asserted); the backend's 422 on a non-numeric column surfaced verbatim; the paste lane unchanged (regression test). |

### Medium-term enhancements

**PK-2 — The generic compartmental ODE simulation lane (event tables)**

| Field | Entry |
| --- | --- |
| ID / Feature | PK-2 — `POST /api/v1/pkpd/ode` over `pharma/odesim.py` + `eventtable.py` |
| Current status | Partial (engines Implemented; routes expose analytic 1-compartment forms + MinimalPBPK only) |
| Problem | `src/pharmaml/pharma/odesim.py` implements `ODEModel`/`EventSchedule`/`solve_ode`/`et()` grid helpers/`lin_cmt_1`/`lin_cmt_2`/matrix-exponential steps, and `eventtable.py` adds `expand_events`, `simulate_events`, `find_steady_state`, `chiou_elimination_rate`, `dose_grid` — while `routes/pkpd.py`'s simulation endpoints are the closed-form one-compartment family (iv-bolus, bateman, superposition) and the fixed MinimalPBPK. Multi-compartment models with arbitrary dosing events (oral + infusion + missed doses) are unreachable from the GUI. |
| Proposed solution | Add `/pkpd/ode` taking a compartment structure (ka/k descriptors or a matrix), an event list (dose, time, route, duration), and a time grid; the body runs `simulate_events`/`solve_ode` and returns the profile arrays + NCA-on-top (the existing `nca_summary` can be applied to the simulated series). The Fit/Simulate stage gains a "custom compartmental model" card. |
| Scientific value | Real regimens (multiple doses, infusions, lagged absorption) are exactly what closed-form Bateman cannot express; the event-table engine already implements the semantics. |
| Feasibility / Priority / Effort | High · P2 · Large |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: RK4 segments — milliseconds per profile; RAM: the state vector history |
| Related links | https://docs.open-systems-pharmacology.org |
| Acceptance criteria | API test: an event-table b.i.d. regimen's steady-state trough matches `find_steady_state` within tolerance on a seeded fixture; a missed-dose event is honored (the engine's own semantics pinned); vitest model-card test. |

**PK-3 — The DDI risk-assessment lane**

| Field | Entry |
| --- | --- |
| ID / Feature | PK-3 — `POST /api/v1/pkpd/ddi` over `pkpd/ddi.py` |
| Current status | Partial (engine Implemented; route Missing) |
| Problem | `src/pharmaml/pkpd/ddi.py` implements the DDI stack — P-gp/CYP substrate records, `competitive_inhibition_factor`, `chen_prusoff` Ki, AUC-ratio classification (`ddi_classification`), the Guest et al. limits (`guest_et_al_limits`, `within_guest`), the network summary and qualification metrics (`gmfe`) — none served by any route. |
| Proposed solution | Add `/pkpd/ddi` taking perpetrator/substrate parameters (Ki, [I], fm) or an observed/predicted AUC-ratio pair; the answer carries the predicted ratio, its Guest-window verdict, and the qualification metrics. A Diagnostics-stage card renders the verdict. |
| Scientific value | The AUC-ratio + Guest-limits verdict is the standard first-pass DDI risk call a pharmacist makes before any interaction study — with the engine's own validation metrics attached. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: microseconds (closed forms) |
| Related links | — |
| Acceptance criteria | API test: a strong-inhibition fixture classifies "high risk" and lands within the Guest window; an out-of-window pair flags `within_guest=False`; vitest verdict-card test. |

**PK-4 — The sensitivity / covariate lane**

| Field | Entry |
| --- | --- |
| ID / Feature | PK-4 — `POST /api/v1/pkpd/sensitivity` over `pkpd/sensitivity.py` |
| Current status | Partial (engine Implemented; route Missing) |
| Problem | `src/pharmaml/pkpd/sensitivity.py` implements `sensitivity_analysis` (+ `StandardPKParameter` enum, `pk_parameter_frame`, `IndividualCharacteristics`, `match_paths`) — the OSP-style "vary covariates, watch PK parameters move" engine — with no route and no UI. |
| Proposed solution | Add `/pkpd/sensitivity` taking a model + parameter/covariate ranges; the response is the sensitivity table + the standard-PK-parameter frame; a Configure-stage card renders the tornado-style ranking through the existing PlotlyFigure (bar traces from backend arrays). |
| Scientific value | Which covariate (weight, age, CrCL) drives exposure in a population is the question every pop-PK discussion resolves; the engine computes it, the GUI never asks it. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: one simulation per grid point — seconds at bounded grids |
| Related links | https://docs.open-systems-pharmacology.org |
| Acceptance criteria | API test: monotone parameter → monotone Cmax sensitivity row on a seeded fixture; the parameter frame enumerates the enum verbatim; vitest ranking chart test. |

### Long-term research extensions

**PK-5 — Population PK / NLME estimation**

| Field | Entry |
| --- | --- |
| ID / Feature | PK-5 — Estimation (not just simulation) of population PK models |
| Current status | Partial (helpers Implemented: `pharma/nlme.py`, `pharma/poppk.py` eta-shrinkage/condition-number; the `/pkpd/population` route simulates IIV only) |
| Problem | The population route "simulates" (`routes/pkpd.py:25` — `simulate_population` over a bateman model with IIV on ka); fitting observed multi-subject data (NLME) has only engine helpers (`nlme.py`'s one-cmt oral concentrations + `individual_params`, `poppk.py`'s diagnostics) with no estimation loop and no route — the toolkit's pharmacometrics stops at simulation + NCA. |
| Proposed solution | A staged plan: (1) route the `poppk` diagnostics over user-supplied parameter tables; (2) implement a bounded FOCE/FOCEI-style estimator over the existing ODE engine as a managed job; (3) only then a UI estimation card. Requires a scientific validation campaign against a published dataset before any "fitted" claim is shown — hence long-term. |
| Scientific value | NLME is the core of clinical pharmacometrics; the run/lineage + diagnostics surfaces the toolkit already has are the accountability layer such fits need. |
| Feasibility / Priority / Effort | Low · P3 · Large |
| Backend / Frontend changes | required / required |
| Dependencies | existing (numpy/scipy in-tree) |
| Performance impact | CPU: minutes per fit; RAM: the subject×parameter matrices |
| Related links | https://docs.open-systems-pharmacology.org |
| Acceptance criteria | Pre-registered: parameter recovery within tolerance on a simulated ground-truth population; eta-shrinkage and condition-number reported with the fit (the `poppk` helpers); no estimation lane ships without the recovery test. |

---

## 8. Molecular Dynamics Analysis

Current state: `frontend/src/features/md/MDPage.tsx` (2,084 lines) runs the
7-stage workflow (Load Topology/Trajectory → Validate → Select Analysis →
Execute → Inspect → Visualize → Export) over `services/md/client.ts`; all
seven `routes/md.py` paths (synthetic, analyze, xvg, enm, pca, study, catalog)
are consumed, with the trajectory playback 3-D viewer and PlotlyFigure series.
Sources are the seeded synthetic trajectory, inline coordinate stacks, or
pasted `.xvg` text. The metric bundle is rmsd/rmsf/Rg/contact-map/native
fraction (+ PCA, ENM); Kabsch superposition is already inside `rmsd_trajectory`
(`md/metrics.py:87` — "after optimal superposition", not a gap).

### Medium-term enhancements

**MD-1 — The hydrogen-bond / salt-bridge timeline lane**

| Field | Entry |
| --- | --- |
| ID / Feature | MD-1 — `POST /api/v1/md/hbond` over `md/hbond.py` |
| Current status | Partial (engine Implemented and tested; route Missing) |
| Problem | `src/pharmaml/md/hbond.py` implements `find_hbonds`, `hbond_timeline`, `salt_bridges`, `salt_bridge_timeline` (`tests/test_md_analysis.py`) — the interaction metrics the GROMACS user reaches for first — while `routes/md.py` serves only rmsd/rmsf/Rg/contacts/native-fraction, and the MD page's Select Analysis stage mirrors exactly that list. |
| Proposed solution | Add `/md/hbond` (job/sync) taking the same inline/synthetic trajectory sources the analyze route already accepts plus donor/acceptor selections; the response is the per-frame H-bond count, the residue-pair table, and the salt-bridge timeline; the Select Analysis stage gains the two checkboxes and the Inspect stage renders the timeline through PlotlyFigure. |
| Scientific value | H-bond occupancy is the stability readout for a protein–ligand complex; salt-bridge persistence is the folding/instability signal — both are trajectory analyses the workspace's name promises. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: geometric checks per frame pair — seconds at the route's 2000-frame cap; RAM unchanged |
| Related links | https://manual.gromacs.org/current/command-line-tools/gmx-hbond.html |
| Acceptance criteria | API test: a synthetic trajectory with a persistent geometric H-bond yields a full-occupancy pair and a broken one decays (fail-first fixture); donor/acceptor validation 422s; vitest timeline test. |

**MD-2 — The protein–ligand interaction fingerprint (PLIF) lane**

| Field | Entry |
| --- | --- |
| ID / Feature | MD-2 — `POST /api/v1/md/plif` over `md/plif.py` |
| Current status | Partial (engine Implemented; route Missing) |
| Problem | `src/pharmaml/md/plif.py` implements `interaction_fingerprint` (the typed interaction record with plane-normal geometry for π-stacking, the residue-level fingerprint) — unused by any route; the docking workspace's contacts route covers static poses, but a ligand-in-pocket *trajectory* fingerprint has no lane. |
| Proposed solution | Add `/md/plif` taking a ligand atom selection + the trajectory frames; the response is the per-frame fingerprint matrix + the per-residue interaction census; the Inspect stage renders the occupancy heatmap (PlotlyFigure heatmap trace over backend arrays). |
| Scientific value | The PLIF is the bridge between MD and med-chem: which interaction persists across the ensemble is what a modeller reports — and the engine's typed geometry beats a bare distance cutoff. |
| Feasibility / Priority / Effort | Medium · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: per-frame normal/centroid math — seconds; RAM: frames×residues matrix |
| Related links | https://github.com/pharmaai/plip (the PLIP precedent) |
| Acceptance criteria | API test: a fixture with a persistent H-bond + one aromatic stacking yields the expected fingerprint codes; degenerate ligand selection 422; vitest heatmap test. |

**MD-3 — The collective-variables lane (with PLUMED-style parsing)**

| Field | Entry |
| --- | --- |
| ID / Feature | MD-3 — `POST /api/v1/md/cv` over `md/cv.py` (incl. `parse_plumed_cvs`) |
| Current status | Partial (engine Implemented, `tests/test_md_cv.py`; route Missing) |
| Problem | `src/pharmaml/md/cv.py` implements `cv_distance/angle/dihedral/coordination/rmsd/gyration`, the `CVTable` collector, and `parse_plumed_cvs` (a PLUMED input-file CV parser) — none served by `routes/md.py`, although the MD page's xvg lane already proves the pasted-text input pattern works. |
| Proposed solution | Add `/md/cv` taking the trajectory source + either structured CV specs or pasted PLUMED-style definitions; the response is the CVTable (per-CV series + summary stats); the Select Analysis stage gains the CV sub-lane and the Visualize stage renders the series through PlotlyFigure. |
| Scientific value | Distances/angles/dihedrals/coordination numbers are the collective variables any enhanced-sampling or stability discussion reduces to; PLUMED-definition compatibility means an existing CV script's numbers can be reproduced exactly. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: vector math per frame — negligible at route bounds |
| Related links | https://www.plumed.org/ |
| Acceptance criteria | API test: a distance CV over the synthetic trajectory equals the direct pairwise computation on a frame (fail-first); malformed PLUMED definition → the engine's own error surfaced; vitest CV-table test. |

**MD-4 — RDF and dihedral distributions**

| Field | Entry |
| --- | --- |
| ID / Feature | MD-4 — Expose `radial_distribution_function` and `dihedral` from `md/metrics.py` |
| Current status | Partial (engines Implemented; not in the served metric bundle) |
| Problem | `md/metrics.py` exports `radial_distribution_function` (g(r) with normalization) and `dihedral` in its `__all__` (lines 61–65) — the analyze/study routes' bundle carries only rmsd/rmsf/Rg/contact-map/native-fraction/PCA; g(r) is the standard liquid/pocket structure readout and dihedral distributions the standard flexibility readout, both unreachable. |
| Proposed solution | Extend `/md/analyze` (and the study job) with optional `include: ["rdf", "dihedrals"]` selections; the response adds the g(r) curve and the per-dihedral histogram data; the Inspect stage renders both through PlotlyFigure. |
| Scientific value | g(r) is how solvent/pocket structure is reported; dihedral distributions are how a rotamer stability claim is made — both are figure-first analyses the workspace's Visualize stage is built for. |
| Feasibility / Priority / Effort | High · P2 · Small |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: histogram/binning sweeps — seconds at route bounds |
| Related links | https://manual.gromacs.org/current/command-line-tools/gmx-rdf.html |
| Acceptance criteria | API test: uniform random coordinates → flat g(r) within tolerance (fail-first); include-flag absent → byte-identical response today (regression); vitest curve tests. |

### Long-term research extensions

**MD-5 — Real trajectory file ingestion (XTC/DCD/PDB multi-model)**

| Field | Entry |
| --- | --- |
| ID / Feature | MD-5 — An upload lane for real MD trajectory files |
| Current status | Missing (deliberate today: "no file dependencies for the demo path" per `routes/md.py`'s docstring) |
| Problem | `routes/md.py` accepts only the synthetic generator, inline coordinate stacks (≤2000 frames × ≤300 atoms, 240k elements), or pasted `.xvg` text; `md/trajectory.py` is the in-memory container (the MDAnalysis/mdtraj *shape*, not the readers). A user's own XTC/DCD/multi-model PDB from GROMACS/AMBER/NAMD cannot enter the workflow without external conversion to inline text. |
| Proposed solution | Add `POST /api/v1/md/load` (file upload, temp workspace) behind an **optional** `MDAnalysis` dependency shipped as an offline wheel and probed by the existing `/system/diagnostics` optional-tier pattern: when absent, the lane answers the honest DependencyError (with the pip hint), never a silent 500. The parsed Topology/Trajectory feeds the same metric routes. |
| Scientific value | The workspace becomes analyzable over real production trajectories — the difference between an MD *demo* and an MD *tool*. |
| Feasibility / Priority / Effort | Medium · P3 · Large |
| Backend / Frontend changes | required / required |
| Dependencies | proposed optional: MDAnalysis (offline wheel; probed like `selfies`/`meeko` today) |
| Performance impact | CPU+RAM: reader pass scales with file size; disk: temp workspace bounded by upload cap (mirroring the datasets cap) |
| Related links | https://docs.mdanalysis.org/stable/index.html |
| Acceptance criteria | Dependency-gated API test (skipped-with-reason when the wheel is absent, executed in the full environment): the shipped `tests/data/1crn.pdb` loads, atom count matches, and RMSD-to-first-frame runs; the honest-degrade lane pinned; upload cap enforced. |

---

## 9. Visualization

Current state: `frontend/src/features/visualization/VisualizationPage.tsx`
(1,960 lines) runs the 7-stage workflow (Select Dataset/Analysis Result →
Select Visualization → Configure Variables → Generate → Inspect →
Filter/Compare → Export) over `services/visualization/client.ts`; all seven
`routes/visualization.py` families (stat box/histogram/ecdf/regression,
concentration-profile, dose-accumulation, md-series, table export, HTML
report) are consumed. Data sources are the session artifact registry plus
paste lanes — notably, this is the one workspace with no registered-dataset
bridge and no `PlotlyFigure` usage.

### Immediate improvements

**VIZ-1 — The dataset column bridge for the stat family**

| Field | Entry |
| --- | --- |
| ID / Feature | VIZ-1 — Plot any registered dataset's numeric column without pasting |
| Current status | Partial (bridge exists app-wide; this page is paste-only) |
| Problem | The stat family reads `valuesText` — a textarea seeded with `SAMPLE_VALUES` (`VisualizationPage.tsx:456`) — while the bounded `column()` bridge (`services/datasets/client.ts:204`) is the established pattern in the ML/DL/Generative/Datasets pages, and the Datasets workspace's own Visualize stage (`DatasetsPage.tsx:1344`) already computes this same statplots family from a picked column, its caption promising "the Plotly-interactive rendering lands with the Visualization workspace row". The bridge and the promise both exist; the wiring doesn't. |
| Proposed solution | Add a dataset + column picker to stage 3 that pulls the column through the existing bridge and feeds the same values array the paste lane produces (one code path, two entry points); the registry/paste lanes stay as-is. |
| Scientific value | The workspace's stage-1 name becomes true — "Select Dataset/Analysis Result" — instead of select-artifact-or-retype. |
| Feasibility / Priority / Effort | High · P1 · Small |
| Backend / Frontend changes | not required / required |
| Dependencies | existing |
| Performance impact | One bounded GET per column |
| Related links | — |
| Acceptance criteria | Vitest: dataset+column selection produces the identical stat request the paste lane does for the same numbers (one snapshot); 422 on a non-numeric column surfaced; paste-lane regression test. |

**VIZ-2 — The clustermap / pairgrid stat kinds**

| Field | Entry |
| --- | --- |
| ID / Feature | VIZ-2 — Add `clustermap` and `pairgrid` kinds to `POST /visualization/stat` |
| Current status | Partial (engines Implemented; route serves 4 of 6 statplots kinds) |
| Problem | `viz/statplots.py:755/794` implement `plot_clustermap` and `plot_pairgrid` (the DataFrame-wide structure plots — `tests/test_viz.py`), but `routes/visualization.py`'s stat family serves `box / histogram / ecdf / regression` only, and the page's kind chips (`VisualizationPage.tsx:1395`) mirror that same four. |
| Proposed solution | Extend the stat route's `kind` literal + the request (a column set instead of one column for pairgrid); the page's kind chips and Configure stage gain the two options with the same stats-dict contract. |
| Scientific value | A descriptor-matrix pairgrid is the one-glance correlation structure a QSAR dataset review starts with; clustermap is its unsupervised companion. |
| Feasibility / Priority / Effort | High · P2 · Small |
| Backend / Frontend changes | required / required |
| Dependencies | existing |
| Performance impact | CPU: O(n²) pair plots; RAM: the frame copy |
| Related links | https://scikit-learn.org/stable/modules/clustering.html |
| Acceptance criteria | API test: a 3-column fixture renders both kinds and registers the artifact; kind list pinned in a drift guard; vitest chip test. |

### Medium-term enhancements

**VIZ-3 — The chemical figure families (descriptor distributions, similarity heatmap)**

| Field | Entry |
| --- | --- |
| ID / Feature | VIZ-3 — `POST /api/v1/visualization/chem` over `viz/chem_plots.py` |
| Current status | Partial (engines Implemented; route Missing) |
| Problem | `viz/chem_plots.py` implements `plot_descriptor_distributions`, `plot_similarity_heatmap`, `plot_table_comparison`, `plot_selfies_robustness` (the capability-registry's "viz.charts" family beyond statplots) — served by no route, so a chemist cannot plot a descriptor distribution histogram or a Tanimoto heatmap of a registered set from the GUI. |
| Proposed solution | Add a `/visualization/chem` family route taking `{kind, dataset_id, columns | smiles_column}` through the existing store bridges; artifact registration through the shared `export_figure` stack; the family chips + catalog row update (the catalog route already lists families from the route source). |
| Scientific value | Descriptor distributions and similarity heatmaps are the two figures every compound-library review begins with. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (RDKit) |
| Performance impact | CPU: descriptor pass + heatmap — seconds; disk: artifact |
| Related links | https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html |
| Acceptance criteria | API test: the four kinds each register an artifact with the digest contract; a non-SMILES column 422s on the similarity kind; vitest family-chip + catalog-row tests. |

**VIZ-4 — Inline interactive previews of the generated figures**

| Field | Entry |
| --- | --- |
| ID / Feature | VIZ-4 — Render the plotted data through the house `PlotlyFigure` beside the downloadable artifact |
| Current status | Partial (PlotlyFigure Integrated in 4 other workspaces; this page renders downloads + stats tables only) |
| Problem | The stat/pharma/md families return the *stats* + the artifact; the page renders the stats table + an `ArtifactRow` download link (`VisualizationPage.tsx:1705-1777`) — the actual curve never appears on screen, even though `PlotlyFigure` (`components/charts/PlotlyFigure.tsx`) is the house chart surface already used by the ML/DL/MD/PKPD pages, and the backend responses already carry the plotted arrays (stat values; concentration time/conc; md-series per-series points). |
| Proposed solution | Have the routes include the plotted data arrays in the responses they already build (no new computation — the arrays are the plot inputs), and render each family's result through `PlotlyFigure` (scatter/bar traces from those arrays, hovermode "closest" per the §22.4 contract) beside the artifact row. The matplotlib artifact stays the archival lane. |
| Scientific value | Zoom/hover on a concentration profile or an ECDF is the readout step; a download-only figure forces the user into an external viewer. |
| Feasibility / Priority / Effort | High · P2 · Medium |
| Backend / Frontend changes | required (echo the plot inputs in the response) / required |
| Dependencies | existing (react-plotly already in the bundle) |
| Performance impact | CPU/RAM: the plotly chunk's lazy load (already the §22.2 manualChunks split); no backend cost |
| Related links | https://plotly.com/javascript/ |
| Acceptance criteria | API contract test: the stat/pharma/md responses now carry the arrays the figure was built from (field names pinned); vitest: the rendered PlotlyFigure receives exactly those arrays; the artifact download lane unchanged (regression). |

### Long-term research extensions

**VIZ-5 — Figure-as-data export (the offline interactive HTML lane)**

| Field | Entry |
| --- | --- |
| ID / Feature | VIZ-5 — Serve `viz/interactive.py`'s figure JSON + self-contained HTML export |
| Current status | Partial (engine Implemented; route Missing) |
| Problem | `viz/interactive.py` implements the plotly figure-as-data model (`Figure.to_json`, `save_interactive_html` — deterministic bytes, no CDN) and `dashlite.py` the headless component/callback engine — none reachable from a route; the user cannot export an interactive (pan/zoom) figure that survives offline, only static PNG/SVG/TIFF/JPEG. |
| Proposed solution | Add `POST /visualization/interactive` taking trace data + layout (or a session figure spec) and answering the figure JSON (for a PlotlyFigure preview, reusing VIZ-4's rendering) and the registered self-contained HTML artifact. Validation: the engine's own determinism (same figure → same sha256) becomes the acceptance pin. |
| Scientific value | A study's figure stays inspectable decades later without the vendor's CDN — the module's own stated purpose, currently stranded engine-side. |
| Feasibility / Priority / Effort | Medium · P3 · Medium |
| Backend / Frontend changes | required / required |
| Dependencies | existing (pure-Python engine, no JS generated by design) |
| Performance impact | CPU: serialization; disk: the HTML artifact |
| Related links | https://plotly.com/javascript/ · https://docs.bokeh.org/en/latest/ |
| Acceptance criteria | API test: byte-determinism of the HTML artifact across two identical requests (sha256 pinned); the figure JSON round-trips through `Figure.from_json`; vitest preview test. |

---

## The final implementation sequence (P0 first, one line each)

| # | ID | Priority | One-line rationale |
| --- | --- | --- | --- |
| 1 | DS-1 | **P0** | Scaffold/group-aware splits at the dataset stage — the engine (`ml/splits.py`) is shipped and tested; random-only splits are a correctness defect in every downstream headline metric. |
| 2 | ML-3 | **P1** | The same strategy vocabulary inside `TrainRequest`, so the GUI's *cross-validated* scores — not just its held-out frames — become scaffold-honest. |
| 3 | ML-1 | **P1** | Four finished `/model-evaluation` routes (conformal intervals, label sets, coverage) with zero consumers — pure wiring for the uncertainty readout QSAR reports owe their users. |
| 4 | GEN-1 | **P1** | The GA's bioisostere/topology/scaffold-hop operators exist in `GeneticOptimizer`; the route's eight numeric knobs make them unreachable. |
| 5 | GEN-2 | **P1** | MOSES validity/uniqueness/novelty/diversity metrics — without them the Rank stage can rank ten copies of one scaffold. |
| 6 | DS-2 | **P1** | The "Clean/Standardize" stage currently *describes* standardization; the REINVENT-style engine needs one route to *do* it (new derived dataset, source untouched). |
| 7 | DS-3 | **P1** | Descriptor frames computed in stage 5 die in display state; registering them closes the SMILES→descriptors→model loop inside the GUI. |
| 8 | PK-1 | **P1** | The paste-only PK/PD import vs the column bridge every other workflow already has — small, unblocks the shipped `pk_profile.csv` path. |
| 9 | VIZ-1 | **P1** | Same bridge for the Visualization stat family; the Datasets page's own caption already promises this row. |
| 10 | ML-2 | P2 | The `/knn` + `/domain-box` routes ship unconsumed; a matrix-only AD verdict completes stage 10. |
| 11 | ML-4 | P2 | k-means/GMM/hierarchical/PCA engines tested in-tree; the unsupervised route serves three methods. |
| 12 | ML-5 | P2 | `ml/drift.py` + CLI exist; the GUI has no drift readout before re-scoring. |
| 13 | DL-1 | P2 | The in-tree TPE/Hyperband engine exists; DL training takes one fixed config. |
| 14 | DL-2 | P2 | GNN explainers (GNNExplainer/PGExplainer/SubgraphX) tested; the GNN lane gives no attribution. |
| 15 | DL-3 | P2 | Quantization/pruning engines tested; stage 11 persists full precision only. |
| 16 | INT-1 | P2 | Live capability census — `/api/v1/capabilities` answers with zero consumers. |
| 17 | INT-2 | P2 | `startup_recovery` typed in the client, rendered nowhere. |
| 18 | INT-3 | P2 | The 45-term glossary engine, no route, no tooltip layer. |
| 19 | INT-4 | P2 | `PRESETS` engine disconnected from the execution-defaults settings. |
| 20 | DS-4 | P2 | SDF ingestion — the converter exists, the upload route 415s the suffix. |
| 21 | DS-5 | P2 | The pIC50/Ki→pKi preparation engine, unreachable from any route. |
| 22 | GEN-3 | P2 | REINVENT-style RL engine exists; the method selector offers two of (then) three engines. |
| 23 | GEN-4 | P2 | MCTS engine tested; a fourth exploration budget unreachable. |
| 24 | DOCK-1 | P2 | `spacing` exists in the engine; `DockRequest` omits it and the GridBoxViewer notes the omission honestly. |
| 25 | DOCK-2 | P2 | Resumable campaigns: the engine checkpoints; the route throws the temp dir away on cancel. |
| 26 | DOCK-3 | P2 | PDB receptor input: the Meeko branch of `prepare_protein` is dead code behind a PDBQT-only request model. |
| 27 | DOCK-4 | P2 | Seven docking figure families in `viz/docking_plots.py`, zero served. |
| 28 | PK-2 | P2 | Event-table ODE simulation engines exist; routes serve closed-form 1-compartment only. |
| 29 | PK-3 | P2 | The DDI risk engine (Guest limits, AUC classification) has no route. |
| 30 | PK-4 | P2 | The sensitivity/standard-PK-parameter engine has no route. |
| 31 | MD-1 | P2 | H-bond/salt-bridge timelines — the engine ships, the metric bundle omits it. |
| 32 | MD-2 | P2 | PLIF engine unused; trajectories have no interaction fingerprint lane. |
| 33 | MD-3 | P2 | CV table + PLUMED parsing engine unused. |
| 34 | MD-4 | P2 | g(r) and dihedral engines exported but outside the served bundle. |
| 35 | VIZ-2 | P2 | clustermap/pairgrid kinds — the statplots route serves four of six. |
| 36 | VIZ-3 | P2 | chem figure families (descriptor distributions, similarity heatmap) unrouted. |
| 37 | VIZ-4 | P2 | PlotlyFigure reuse — the one workspace without an inline figure preview. |
| 38 | DL-4 | P3 | D-MPNN exposure — needs scientific validation before any UI claim. |
| 39 | GEN-5 | P3 | SELFIES representation mode — validity-by-construction, validation-gated. |
| 40 | PK-5 | P3 | NLME estimation — the long pharmacometrics arc, recovery-test gated. |
| 41 | MD-5 | P3 | Real trajectory file loading behind an optional MDAnalysis wheel, honestly degraded. |
| 42 | VIZ-5 | P3 | Figure-as-data / offline interactive HTML export — determinism-pinned. |

— execution log: **row 1 (DS-1) executed in round 373 at v1.2.246** — the
strategy lane through `split_indices`, the two-pass group-honest val carve,
the `group_census` leakage verdict, the Split-stage selector + the group
column gate, 13 API tests + 7 drift-guard tests + 7 vitests; the next row is
ML-3 (the Train lane's `split` section over the same engine wiring).

— execution log: **row 2 (ML-3) executed in round 375 at v1.2.247** — the
`split` section on `TrainRequest` (`TrainSplitSpec`: strategy +
`smiles_column` auto-pick + never-guessed `group_column` + `test_size`), the
carve over the FULL registered frame through the same `split_indices`
engine before the fit (the estimator sees the carved train side only —
`n_train_rows` IS the carve), the sides + per-side group census + the
`shared_groups` leakage verdict in the response and the §15.2 run record's
`cross_validation.split` at BOTH exits (the refused carve records its echoed
config + error, the fit failure records the computed carve), the Configure
Split/CV selector ("none" keeps the legacy lane — the request carries the explicit `split: null`, the pydantic-equal default) +
the group-column gate + the carve readout with the honest-cv caveat line; 9
API tests + 14 drift-guard tests (the guard extended to the train lane) +
7 parser vitests + 3 page vitests, the contract re-exported (142 paths /
97 schemas). The next row is ML-1 (the four finished
`/model-evaluation` routes with zero consumers).

— execution log: **row 3 (ML-1) executed in round 376 at v1.2.248** — the
`modelEvaluationApi` client (the four typed operations over the finished
`/model-evaluation` family: catalog, conformal-regression,
conformal-classification, coverage-diagnostics — every answer narrowed
through strict runtime parsers), the Explainability & Uncertainty stage's
conformal panel (the calibration + test slices over the bounded column
bridge with the requested-ratio cap, alpha + the conditional selector +
the SMILES column pick, the interval bounds table + the empirical
coverage line + the Wilson-bounded scaffold coverage table, the family
catalog's honesty notes rendered beside the controls, the
finite-sample-overlap caveat), 17 vitests + the new drift guard
`tests/test_frontend_model_evaluation.py` (7 tests) + the ml-workspace
guard extended (the family joined the route-coverage tuple + the
four-path consumption pin); a carrier drift caught by the
diagnostics-version guard (frontend/src/lib/version.ts stuck at
1.2.246 across the r375 bump) trued in the 1.2.248 lockstep. The next
row is GEN-1 (the mutation/crossover operator modes on the
generative optimize route).

— execution log: **row 4 (GEN-1) executed in round 377 at v1.2.249** — the
`mutation`/`crossover` Literals (the engine's own vocabulary:
uniform/scaffold/topology/bioisostere + splice/scaffold_hop) + the
`min_heavy_atoms` child gate on `OptimizeRequest`, forwarded verbatim
into `GeneticOptimizer` through `ga_kwargs` (the engine's own init
validation is the guard, the unknown-literal 422 is pydantic's own);
the Configure stage's Evolution parameters card gains the two selects
(the engine's own descriptions as the option labels) + the
conditional gate input; the contract re-exported (142 paths / 98
schemas — the OptimizeRequest schema carries the new literals); 4
API tests over the corpus-free fixture (the metadata echo as the
fail-first wiring proof + the hopped-Murcko evidence + the honest
route-level scope of the gate test + the unchanged default lane) +
2 vitests (the select passthrough + the default lane) + the
generative-workspace mirror guard extended with the three new
fields. The next row is GEN-2 (the MOSES generation metrics route
over `moses_metrics.generation_metrics`).

— execution log: **row 5 (GEN-2) executed in round 378 at v1.2.250** —
the `POST /api/v1/generative/metrics` route over the real
`moses_metrics.generation_metrics` engine (the full MOSES-lite audit:
validity, uniqueness, novelty vs an inline-or-dataset reference with an
existence bound never a size bound, IntDiv1/2, scaffold diversity, the
property-shift table, the honest not-measured nulls — never a fake 1.0;
the both-at-once reference refusal is an honest 422); the Validate
stage's MOSES metrics card (the reference-corpus select, the five
headline cells, the property-shift table, the pairwise-truncation
flag, the honest "not measured (no reference)" lane); 6 API tests
(the duplicate collapse + the hand-computed novelty fixture + the
not-measured lane + the dataset column + the refusals) + 2 vitests +
the generative-workspace mirror guard extended with the fourth
request; the contract re-exported at 143 paths / 98 schemas. The next
row is DS-2 (the Clean/Standardize stage's own route over a dataset
SMILES column).

— execution log: **row 6 (DS-2) executed in round 379 at v1.2.251** —
the `POST /api/v1/chemistry/standardize` route over the real
REINVENT-style engine (`chem/standardize.py`: the
`canonical_tautomer` pass and the `reinvent` chain — largest
fragment, RemoveHs, salt removal, neutralisation with its own
documented toggle, cleanup, the unwanted-pattern/size/element gates —
applied in request order); the standardized frame registers as a NEW
derived dataset through the same load-and-register path every upload
takes, with the provenance block the store now carries on record
views (`derived`: kind, source dataset_id, column, ops, neutralize) —
the source record is never mutated (sha-256 pinned over the column
lane by the API tests; the no-write scan pinned by the new drift
guard); the non-SMILES column is an honest 422; rejected rows are
accounted (row, input, reason), never silently dropped; the Datasets
stage 4's Clean action (the pass checkboxes, the uncharge toggle,
the report card with the changed/unchanged/rejected counts + the
rejected table + the derived dataset's select action); 9 API tests +
2 vitests + the 10-test DS-2 drift guard (the ops vocabulary pinned
across route / contract / client, the no-write scan, the page +
parser pins, the provenance-on-the-store pins) + the chemistry
mirror guard extended with the third request; the six stale
contract-count pins the r378 round left behind its own 143/98 growth
caught and trued to 144/99; the contract re-exported at 144 paths /
99 schemas. The next row is DS-3 (the descriptor/fingerprint frames
registered as derived datasets).

— trued round 372 (part21 / task 2-c): written read-only against the v1.2.244
tree; every "zero frontend consumer" claim above was verified by diffing all
148 route handlers against the 114 distinct API paths the service clients
actually call; the unconsumed set (`/capabilities`, `/model-evaluation/*`,
`/applicability-domain/knn|domain-box`, plus the engine-only modules named per
row) is exactly the Immediate tier of this roadmap.
