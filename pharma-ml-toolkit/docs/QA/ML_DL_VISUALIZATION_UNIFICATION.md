# ML/DL Visualization Unification — Round 372 (MASTER_PROMPT part21 §3)

**Status: IMPLEMENTED and gate-green on branch `feat/shared-ml-dl-evaluation-visualizations`**
**Base commit: `e8b565f6` (main) — the r372 housekeeping close-out**
**Date: 2026-11-28**

---

## 1. Executive summary

The ML and DL workspaces now render evaluation results through **one
shared component layer** over **one backend contract**. The ML
workspace's proven §22.5 evaluation charts (the parity scatter, the
confusion-matrix heatmap, the click-through sample inspector) were
extracted from `MLPage.tsx` into the shared module
`frontend/src/components/evaluation/`, extended with the part21
§3.1 diagnostic families, and adopted verbatim by the DL workspace's
Evaluate stage — which previously rendered prediction tables only and
never read the `evaluation` block its own predict call was already
receiving. The backend grew **one smallest-correct extension**: the
predict pass's evaluation block now carries the sample-derived
diagnostics (residual histogram / summary stats / normal Q–Q for
regression; per-class metrics + one-vs-rest ROC/PR curves + the binary
reliability diagram for classification), all computed server-side from
the aligned rows and probabilities the pass already produced. **No
duplicate parity/confusion implementation survives anywhere in the
frontend.**

## 2. The audit (what existed before)

| Concern | The state at `e8b565f6` | Evidence |
|---|---|---|
| ML Evaluate stage | Parity scatter + confusion heatmap + row inspector — **file-private inside MLPage.tsx** | `frontend/src/features/ml/MLPage.tsx` (the §22.5 block, since round 2011's part11 §22.5) |
| DL Evaluate stage | Prediction + probability **tables only**; `predictResult.evaluation` never read | `frontend/src/features/dl/DLPage.tsx:1929–1995` (grep: zero `evaluation` consumers) |
| The shared predict path | ALREADY unified — DL's tabular trainers register torch estimators in the same session model store; DLPage calls the same `mlApi.predict` (`POST /api/v1/ml/predict`, dataset mode, sync); the route populates `evaluation` for every store model (dataset mode + aligned lengths) and 422-refuses graph/sequence/autoencoder models to their own lanes | `src/pharmaml/api/routes/ml.py` (`_predict_body`, the `predict` route); `src/pharmaml/api/routes/dl.py` (`_train_body` — `predict_url: "/api/v1/ml/predict"`) |
| The evaluation contract | Aligned rows only (`rows[{name, actual, predicted, residual, smiles}]`, bounded to the 500-row sample) — no curves, no per-class aggregates | `_evaluation_block`, ml.py |
| Graph models | Own predict lanes (`gnn-predict`, `cnn-predict`, `autoencoder-encode`) — **never** forced into the tabular contract | the `predict` route's three 422 guards |
| Training telemetry | The DL Monitor's live per-epoch WebSocket envelopes (`TrainingMetricsPoint`); LossCurveChart file-private in DLPage | `dl.py` `_training_progress_hook`; `DLPage.tsx` |
| Curve recipes | Full matplotlib families existed in `viz/ml_plots.py` (ROC/PR/calibration/residuals) — but **no route served them for session-trained models** and the frontend had none | `src/pharmaml/viz/ml_plots.py` |
| Export | Both pages had **identical private** `buildExportRows` copies flowing into the same `POST /api/v1/visualization/table` artifact lane | MLPage/DLPage |

**The verdict the audit produced:** no backend extension was needed for
DL to *receive* evaluation data — the contract already carried it and
DL simply never rendered it. What the contract did NOT carry: the
curve/aggregate data for the part21 §3.1 chart families. The
scientific-calculations-stay-in-the-backend rail (part21 §2/§15) made
the decision: **compute the curves server-side, extend the block —
never duplicate sklearn-family math in TypeScript.**

## 3. The design

### 3.1 Backend — `src/pharmaml/api/evaluation_curves.py` (new)

Pure-stdlib (statistics.NormalDist for the Q–Q theoretical quantiles)
functions computing, **from the evaluation block's own bounded sample
rows and the pass's own probability block**:

* **Regression**: `residual_diagnostics` — the 20-bin equal-width
  residual histogram (bin edges + counts), the summary stats
  (mean/median/std/min/max/n), and the normal Q–Q points (Blom
  plotting positions over the standardized ordered residuals;
  stdlib `NormalDist.inv_cdf`). Attached only when ≥4 numeric
  residuals exist; a zero-variance sample attaches no Q–Q (the honest
  null).
* **Classification**: `class_metrics` — per-class precision/recall/F1/
  support in the estimator's own `classes_` ordering (the class-
  ordering rail), zero-division metrics as 0.0 with supports exposed,
  plus macro averages and the sample accuracy. Attached whenever
  aligned rows exist — probabilities not required.
* **When the pass carried `probabilities`**: per-class **one-vs-rest ROC
  curves** (grouped-threshold step walk, tie-aware AUC by the step
  trapezoid — equal to the tie-aware Mann–Whitney U), per-class
  **PR curves** (with average precision by the recall-gain integral
  **and the class's own prevalence baseline** — the imbalance-honest
  reference, never ROC's 0.5), and — for exactly two classes — the
  **binary reliability diagram** (10 equal-width bins, mean predicted
  vs observed fraction + per-bin count; empty bins stay count-0 rows).
  A class with no positives or no negatives in the sample carries no
  curve for that class. A multiclass answer attaches **no calibration
  block** (the binary case's own lane — never a misleading aggregate).
* Position alignment premise: the evaluation rows are the FIRST
  `n_rows_evaluated` rows of the prediction set; the probability rows
  cover the whole set in the same X order (verified live by the test
  `test_probabilities_align_with_evaluation_rows` — argmax of each
  sampled probability row equals that row's own prediction).
* Wire-in: `_predict_body` calls `enrich_evaluation_block(payload
  ["evaluation"], payload.get("probabilities"))` after both blocks are
  built — rides **sync and job modes identically** (the job mode
  executes the same function). Rendering-only decimation bound
  (`_MAX_CURVE_POINTS = 200`) applies to curve point lists only.
* Every emitted float passes a finite-coercion guard (NaN/inf never
  leave the module).

### 3.2 Frontend — the shared module `frontend/src/components/evaluation/`

| File | Role |
|---|---|
| `EvaluationCharts.tsx` | The orchestrator both pages render: picks the family from `task` + the fields the backend actually attached; renders the sample-bound note; the §22.5 inspector rides the parity click |
| `ParityChart.tsx` | The extracted parity scatter (identity line, per-point `customdata`, click-through) |
| `ConfusionChart.tsx` | The extracted confusion heatmap **+ the raw/normalized toggle** (per-true-class shares; an all-zero actual-class row stays zero, never 0/0) |
| `EvalRowInspector.tsx` | The extracted sample inspector (the point's own metadata) |
| `ResidualHistogramChart.tsx` | The backend-computed histogram as bars (centers/widths from the edges; hover carries the bin interval + count) |
| `QqPlotChart.tsx` | The backend-computed Q–Q pairs against the identity anchor |
| `RocCurveChart.tsx` | Per-class OvR lines + the chance diagonal + AUC/counts in legend/hover |
| `PrCurveChart.tsx` | Per-class lines + **each class's own prevalence baseline** (dashed) + AP |
| `CalibrationChart.tsx` | The reliability diagram over occupied bins + the perfect-calibration anchor + the count-honesty footnote |
| `ClassMetricsTable.tsx` | The per-class P/R/F1/support table + macro row + the sample-accuracy note (the cv block stays the authoritative lane) |
| `exportRows.ts` | The ONE export-row builder pair: `buildPredictionExportRows` (the moved ML copy) + `buildEvaluationExportRows` (the aligned sample) — both pages' exports flow through it into the same `/api/v1/visualization/table` + digest-verified download lane |
| `formatValue.ts` | The moved §17.3 cell formatter (one implementation) |

`frontend/src/components/training/LossCurveChart.tsx` — the extracted
shared training-history chart (§3.3): renders exactly the signals the
backend's `TrainingMetricsPoint` publishes (train/val loss per epoch),
displays a series only when its real data is present, and carries the
honest <2-point waiting state. Additional series (accuracy,
learning-rate, epoch duration) attach the day their lanes genuinely
emit them — nothing is fabricated now.

### 3.3 Adoption

* **MLPage.tsx** — the inline §22.5 block (4 components + the local
  `formatValue` + the local `buildExportRows`, ~8,969 chars) deleted;
  the Evaluate stage renders the shared `EvaluationCharts` at the same
  call site; the export stage rides the shared row builder. Net
  −575 lines in the page; behavior superset (the same charts +
  the new diagnostics families).
* **DLPage.tsx** — the Evaluate stage now renders the shared
  `EvaluationCharts` over its own `predictResult` (the explanatory
  state for a null block — inline-row answers); the export stage
  gained the **source select** (predictions table | evaluation
  sample) riding the shared builders; the inline LossCurveChart +
  local buildExportRows copies removed. The GNN/CNN/autoencoder lanes
  are untouched (their own predict lanes, per the architecture rail).

### 3.4 The three-concern separation (§3.5, preserved)

Reusable charts stay embedded in the ML/DL workflows; the backend
artifact system stays the publication-quality lane; the Visualization
workspace stays a separate workspace with its own registry/report
flow. Nothing moved into it. The standardization that DID happen: one
export-row builder pair feeds the existing artifact lane for both
workspaces — no new endpoints, no duplicate download implementations.

## 4. The scientific rails honored (part21 §4)

* The curves describe the **same bounded sample** the parity chart
  plots — `n_rows_evaluated` of `n_rows_available` reported on the
  surface; the model's authoritative quality stays the train answer's
  cross-validated block (stated in the metrics table's own footnote).
* No fabricated values: absent data → absent fields → explanatory
  states in the UI (tested for the no-probability and no-evaluation
  cases).
* Class labels and ordering preserved (the estimator's `classes_`
  ordering in the metrics; the original label strings in every
  chart/table — non-numeric labels render verbatim, tested).
* Calibration appears only with real probability data and only for
  the binary case; ROC/PR only with real scores + ground truth (the
  backend never attaches them otherwise, and the frontend can't
  fabricate them — it renders only what the parser admits).
* The imbalance rail: PR baselines are per-class prevalences, never a
  flat 0.5; the macro/micro aggregates carry their supports.
* Missing vs zero: null residuals produce no diagnostics; count-0
  calibration bins render as honest gaps.
* The graph/sequence/autoencoder models never flow through this
  contract — the route's own 422 guards remain the enforcement point
  (and the graph evaluation contract stays a separate future item —
  see §7).

## 5. Test evidence (the full battery in `ML_DL_VISUALIZATION_TEST_REPORT.md`)

* **Backend**: `tests/api/test_ml_predict_evaluation_curves.py` — 7/7
  (regression diagnostics attached + the §22.5 residual identity
  re-derived; classification no-diagnostics honest absence; binary
  curves + alignment premise + calibration; multiclass per-class
  curves + no-calibration; no-proba → no curves; inline rows → no
  evaluation block).
* **Frontend**: `components/evaluation/__tests__/EvaluationCharts.test.tsx`
  — 10/10 (both families, the normalized toggle, the sample-bound note,
  the §22.5 click-through identity, the explanatory states, non-numeric
  labels, the export builders).
* **Integration**: `features/dl/__tests__/EvaluateStageCharts.test.tsx`
  — 2/2 (the full DLPage drives tabular train → predict → the shared
  charts render in the DL Evaluate stage; the null-evaluation
  explanatory state).
* **Regression**: the full vitest suite 369/369 over 43 files; the
  drift-guard battery 202/202; the predict/contract battery 87 green
  incl. the OpenAPI contract identity (the response payload extension
  changed no path/operationId); tsc clean; vite build green; ruff clean
  over `src/pharmaml`; mypy clean (315 files).

## 6. What was deliberately NOT done (honest scope)

* **Graph-model evaluation contract** (§3.2's GNN carve-out): GNN
  predictions keep their own lane; a separate graph evaluation
  contract (graph-metric aggregates, node/edge-level views) is a
  roadmap item (see `WORKFLOW_ENHANCEMENT_ROADMAP.md`, DL-2), not
  silently bolted onto the tabular block.
* **Matplotlib publication figures for the new families**: the
  `viz/ml_plots.py` recipes (ROC/PR/calibration/residuals) still have
  no session-model route — the shared interactive charts are the
  in-workflow lane; a `figure: true`-style artifact route for
  evaluation curves is the roadmap's VIZ-3 lane.
* **Training-telemetry extensions** (accuracy/LR/epoch-duration
  series): the component renders only the signals the backend
  currently publishes; extending the callbacks is the roadmap's
  DL-lane work, never a fabricated series.
* **The Visualization workspace**: untouched (the three-concern
  separation is a design rail, not a defect).

## 7. Remaining gaps

1. No graph-model evaluation contract (roadmap DL-2).
2. No backend figure-artifact route for the evaluation curve families
   (roadmap VIZ-3) — the interactive charts are the only lane today.
3. The calibration lane is binary-only by design; multiclass
   reliability diagrams (Brier-per-class, one-vs-rest reliability)
   are a candidate roadmap extension.
4. The DL export's evaluation-sample source caps at the same 500-row
   sample the charts plot (the full prediction set rides the
   predictions export) — a full-evaluation export would need the
   route to emit the unbounded block, which the §17.3 bound exists to
   prevent.

**Verification instruction for a reviewer:** run
`pytest tests/api/test_ml_predict_evaluation_curves.py -q` and
`cd frontend && npx vitest run src/components/evaluation src/features/dl/__tests__/EvaluateStageCharts.test.tsx`
— the twelve new frontend tests and seven new backend tests are the
contract's executable description.
