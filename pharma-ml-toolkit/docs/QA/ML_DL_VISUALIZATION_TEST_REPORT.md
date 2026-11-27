# ML/DL Visualization Test Report — Round 372 (MASTER_PROMPT part21 §11/§12.D)

**Branch: `feat/shared-ml-dl-evaluation-visualizations` (base `e8b565f6`)**
**Date: 2026-11-28 · Toolkit version under test: 1.2.245 (the carriers bumped in lockstep this round)**

Every number below was executed and captured this round in the sandbox
(the repo root + the repaired venv — numpy 1.26.4 / torch 2.14.1+cpu /
pharmaml 1.2.245 editable). Blocked or
not-run lanes are named as such — never counted as passed.

## 1. Test commands and counts

| Gate | Command | Result |
| --- | --- | --- |
| **The new backend contract suite** | `the venv's pytest tests/api/test_ml_predict_evaluation_curves.py -q` | **7/7 passed, exit 0** (regression diagnostics attached + the §22.5 residual identity re-derived; classification honest-absence; binary curves + calibration; multiclass per-class curves + no-calibration; no-proba → no curves; inline rows → no evaluation; the argmax-position-alignment premise verified live) |
| **The shared-component suite** | `cd frontend && bunx vitest run src/components/evaluation/__tests__/EvaluationCharts.test.tsx` | **10/10 passed** (both chart families; the confusion raw/normalized toggle; the sample-bound note; the §22.5 click-through inspector identity; the no-evaluation + no-probability explanatory states; non-numeric labels verbatim; the export-row builders) |
| **The DL integration suite** | `cd frontend && bunx vitest run src/features/dl/__tests__/EvaluateStageCharts.test.tsx` | **2/2 passed** (the full DLPage drives tabular train → predict → the SHARED charts render in the DL Evaluate stage; the null-evaluation explanatory state, no fabricated panels) |
| **Full frontend suite** | `cd frontend && bunx vitest run` | **369/369 passed over 43 files** (the r371 baseline 357/41 — the +12 tests / +2 files are this round's battery; the ML page's own evaluation-stage tests run against the shared wiring) |
| **The predict/contract regression battery** | `pytest tests/api/test_datasets_ml_artifacts.py tests/api/test_contracts.py tests/test_documentation_consistency.py -q` | **87 passed, exit 0** — the existing §22.5 evaluation-block contract tests unchanged-green (the payload extension is purely additive), the OpenAPI runtime==contract identity re-held (142 paths / 149 ops — the response-dict extension changed no path/operationId), the ledger guard green |
| **The drift-guard family (16 files)** | `pytest tests/test_frontend_*.py -q` (16 files) | **202/202 passed, exit 0** (the MLPage/DLPage surgery verified by the existing page guards) |
| **Notebook freshness + carriers** | `pytest tests/test_notebook_variants.py tests/test_version_carriers.py -q` | **passed, exit 0** — all 8 notebooks re-executed at pharmaml 1.2.245 (229s, zero error-outputs, zero home-path survivors), the 7 twins re-stripped, the carrier set agrees at 1.2.245 |
| **TypeScript** | `cd frontend && bunx tsc --noEmit` | **0 errors** |
| **Frontend production build** | `cd frontend && bunx vite build` | **green, 24.03s** (the Plotly lazy-chunk strategy unchanged) |
| **Backend lint** | `ruff check src/pharmaml` (the CI's exact scope) | **clean** (two findings during development fixed: a PERF402 and a SIM102 in the new module; note: a full-tree `ruff check src tests` also reports 90 × EXE002 exec-bit artifacts of the sandbox's file-copy modes — fresh checkouts never see them, and the modes were normalized this round) |
| **Type discipline** | `mypy src/pharmaml` | **Success: no issues in 315 source files** (the new module included) |
| **Default-suite collection** | `pytest --collect-only -q` | **8,293 collected / 0 errors over 443 files** (the r371 pin 8,286/442 — the +7/+1 is this round's contract suite) |

## 2. Regression scenarios covered by the new battery (part21 §4's edge-case list)

ML regression evaluation (parity + histogram + Q–Q + the summary
line) ✓ · ML classification evaluation (confusion + metrics + ROC/PR/
calibration) ✓ · DL regression/classification evaluation (the same
component, the DL answers) ✓ · multiclass confusion matrices (the 3-class
route test + the metrics table) ✓ · missing probability data (no curves
+ the explanatory note) ✓ · missing evaluation blocks (the explanatory
state, no charts) ✓ · non-numeric class labels (verbatim rendering,
the route preserves the estimator's ordering) ✓ · imbalanced datasets
(the per-class prevalence baselines + the supports-exposed zero-
division metrics) ✓ · missing values (null residuals → no diagnostics,
never a fabricated bin) ✓ · empty chart inputs (the empty-hint states
ride PlotlyFigure's own contract) ✓ · large evaluation samples (the
500-row route bound reported on the surface + the 200-point curve
decimation, rendering-only) ✓ · sample selection and metadata (the
§22.5 click-through identity: the point's own row/SMILES/actual/
predicted/residual) ✓ · training-history missing values (the shared
LossCurveChart's honest <2-point waiting state, carried by the existing
Monitor tests) ✓ · delayed/missing WebSocket telemetry (the existing
DLPage telemetry tests, unchanged-green in the 369) ✓ · graph-model
evaluation limitations (the route's 422 guards + the roadmap's DL-2 —
no graph data ever enters the tabular contract) ✓ · export and download
behavior (the builder pair + the existing artifact-lane tests in the
369) ✓.

## 3. Build results

- The frontend production build: green (§1).
- The v1.2.245 zip: built by `scratch/build_zip_373.py` (the canonical
  r303..r371 recipe: the exclusion list, the §4.3 hygiene, the
  must-carry gate extended with this round's 22 proof paths, the
  composition guard, the leak scan, the version_info quadruple check,
  the six homes + the offsite mirror, testzip each) — the ship
  numbers record in the worklog R373 entry.
- **No installer build this round** (part21 §6: no release authorized —
  v1.2.244 remains the published installers; the next owner-gated
  cycle carries this delta).

## 4. Runtime limitations (honest)

- **No browser E2E leg this round** (the sandbox's recurring resource
  class; the component-level and full-page jsdom suites carry the
  interactivity evidence, and the r372 housekeeping round's render
  evidence stands for the web console).
- The live `/api/toolkit/tree` census walk sat on the documented
  Turbopack SST-persist wedge class in prior rounds; the offline
  census replica (the r366/r371 precedent) carries the re-true
  measurement when needed.
- Tauri/Rust tests: **NOT RUN — no cargo/rustc in this sandbox**
  (D-11/D-12's standing constraint; unchanged).
- The CI leg: green at the pre-work HEAD `e8b565f6` (runs
  46066722216 Push / 46066722332 CI, verified via the GitHub API);
  the branch's own CI rides the post-merge push (the merge commit
  triggers the standard 19-job ladder).
- The ZAP penetration lane run 46038496882: **in-limbo** (`in_progress`,
  no conclusion, last update 2026-11-27T17:01:51Z) — never counted
  green; the local 67-test penetration API suite is the in-sandbox
  green lane; the Monday 2026-11-30 deep-scan slot stands.

## 5. CI links

- CI at the pre-work HEAD: run 46066722332 (`CI`, completed/success,
  19/19 check-runs) and 46066722216 (`Push on main`, success).
- The post-merge push CI: fires on the merge to main (the standard
  ladder: 14 CI jobs + 5 CodeQL analyze jobs, per the r372
  revalidation's live check-runs census).

## 6. Verdict

The part21 §11.1 acceptance: **met** — ML renders its existing
evaluation charts (now through the shared layer, verified by the
unchanged-green ML guards), DL renders the SAME component over its
real predict answers (the integration test), the family selection
follows the task type, no duplicate parity/confusion implementation
remains (the grep-diff: the inline copies deleted, one shared module),
missing data renders explanatory states (tested both), the interactive
Plotly rendering stays lazy-loaded (the manualChunks split unchanged,
the production build green), the exports ride the backend artifact
contract (the builder pair + the existing digest-verified lane), and
the tests cover both workspaces plus the §4 edge-case list.
