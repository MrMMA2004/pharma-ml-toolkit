# Response Surface Methodology (RSM) — part17 §6, the R1 lane

**Status:** VERIFIED_COMPLETE (round 359, 2026-11-25) — the backend module
`src/pharmaml/ml/rsm.py`, the API route `src/pharmaml/api/routes/rsm.py`
(seven paths), the ML-workspace section (stage 21), the API suite
`tests/api/test_rsm.py` (38/38), the vitest `RsmLane` suite (9/9), and the
drift guard `tests/test_frontend_rsm.py` (16/16) are connected and verified.
Every limitation below is stated, never hidden.

## What RSM is — and is not

RSM (Box & Wilson, 1951) is a **designed-experiment** modelling
discipline: plan a design whose factor levels make the second-order model
estimable, run the experiments, fit the full quadratic in coded factors,
inspect the ANOVA and diagnostics, visualise the surface, then optimise.
The part17 mandate's own distinction table holds verbatim here:

| RSM is NOT | because |
| --- | --- |
| the generic multi-response ML (§5, `ml/multiresponse.py`) | that lane *predicts*; this lane *explains* a designed experiment |
| multi-objective optimisation (`ml/multiobjective.py`) | the desirability scale answers "how acceptable is each response" before any Pareto reasoning |
| drug-response matrix factorisation (`ml/drugresponse.py`) | no cell-line panels — the responses are measured experiment outputs |
| an ordinary 3-D visualisation | the surfaces are fitted models with confidence geometry, not plotted raw data |

## The design generators (§6.3)

`build_design()` over declared factors (`FactorSpec`: name, low, high,
optional unit; 2–10 factors, low < high, unique names):

- **Central Composite (CCD)** — the 2^k factorial corners, 2k axial runs
  at ±alpha, replicated center points. `alpha_type`: `rotatable`
  (`alpha = (2k)^(1/4)` — **the axial runs sit OUTSIDE the declared
  factor ranges, disclosed in the design assumptions, never hidden**),
  `faced` (alpha = 1, stays inside), or `custom` (0.5 < alpha ≤ 4). The
  classic **two-block split** (factorial block + axial block, centers
  split between them) is supported; randomisation shuffles the run order
  WITHIN each block (the block order is an experimental-protocol
  decision, stated).
- **Box-Behnken** — each factor pair at its 2^2 corners with every other
  factor at the center; every level inside [-1, +1] (the design never
  leaves the declared ranges); needs ≥ 3 factors; blocking honestly
  REFUSED (a typed 422 — never faked).
- **Full 2^k factorial** — all corners plus centers, capped at 8 factors
  (2^9+ corners would exceed the 512-run bound).

Every run table carries **both coded and actual levels** (center =
(low+high)/2, unit = (high-low)/2), the run order, block, and run type;
the design answer states its own assumptions (the rotatable-alpha
disclosure, the aliasing warning when center_points = 0, the
no-replicates-no-LOF note). The 512-run bound, factor-range validation,
and seed bounds are typed 422s, never silent clamps.

## The second-order fit (§6.4)

`fit_response_surface()` over an observed frame (actual units; the coding
rides the declared ranges; rows missing any factor/response value are
dropped complete-case with the dropped count reported):

- The term registry: 1 intercept + k linear + k(k-1)/2 interaction + k
  quadratic terms, in a STABLE order (the bundle's coefficient vectors
  ride it).
- Least squares over the coded model matrix; under rank deficiency the
  minimum-norm solution is reported with the **aliasing disclosure**
  (rank, the aliased term set via QR pivoting, and the note that the
  aliased coefficients are not separately interpretable).
- The coefficient table: estimate, std error, t, p, and the confidence
  interval at the request's confidence level.
- The ANOVA partition: regression / residual / total, the per-term
  partial SS rows, and — only where genuinely replicated design points
  exist — pure error and lack-of-fit with the F test. **No replicates,
  no pure error, a null — never invented.** A noise-free pure error
  (zero variance) reports the F as an UNDEFINED null, stated.
- Hierarchy notes: a significant interaction or quadratic with an
  insignificant parent linear term is FLAGGED ("keep the parents
  anyway") — **never auto-removed**.
- Diagnostics: R2, adjusted R2, predicted R2 over the exact
  leave-one-out PRESS, RMSE, leverage, internal studentised residuals,
  Cook's distance with STATED thresholds (|t| > 3; Cook's > max(1, 4/n)),
  reported as flags — never silent removals. A machine-precision fit
  (rmse at the float noise floor — the noise-free demo class) reports
  the scaled diagnostics as exact zeros with the note, rather than
  flagging floating-point dust as outliers.

## Prediction (§6.5) — the zone vocabulary

`predict_surface()` predicts in ACTUAL units with the mean confidence
interval and the new-observation prediction interval, and flags every
point with the honest zone: `inside_ranges` / `axial_region` (outside the
ranges but inside the CCD's axial reach) / `outside_design` (outside the
design region entirely — a model extrapolation, flagged, never presented
as an interpolation).

## Visualisation (§6.5)

`contour_grid()` returns the two-factor grid payload for the Plotly
contour / 3-D surface lanes: x/y in coded AND actual units, z either one
response's prediction or the **combined desirability overlay** under the
request's goals, every other factor held at explicitly displayed fixed
values (in both units), correct axis labels with the factors' declared
units. The frontend renders it inside the Machine Learning workspace
(the §6.7 rule — NOT the Visualization workspace; a drift guard pins it).

## Multi-response optimisation (§6.6)

`optimize_desirability()` — the Derringer-Suich approach:

- Goals per response: `maximize`, `minimize`, `target`, `in-range`.
  Bounds the request omits fall back to the **observed design min/max**,
  the fallback stated per goal (`bounds_source`), never silent.
- Individual desirabilities with per-response weights (0, 10] and
  exponents [0.1, 10]; the combined desirability is the weighted
  geometric mean.
- The search: a deterministic multi-start sweep (the design's own
  points, the center, the corners capped at 64), each polished with a
  bounded gradient method (L-BFGS-B free of constraints; SLSQP
  honouring optional linear coded-factor constraints). **Confined to
  the coded box [-1, +1]^k — the declared factor ranges — so the
  optimum cannot extrapolate outside the experiment by construction.**
- The answer carries the candidate with per-response predictions and
  individual desirabilities, the per-factor sensitivity sweep (±0.05
  coded, stated), the curated Pareto alternatives (per-response
  individual optima + weight-tilted combined optima — a deterministic
  SAMPLE, honestly not the exhaustive front), and the feasible-region
  mask for the requested pair when constraints ride the request.

## The API (7 paths) + the persistence contract

`/api/v1/rsm/registry` (vocabulary), `/design`, `/fit`, `/predict`,
`/contour`, `/optimize`, `/models`. The fitted `ResponseSurfaceBundle`
(the factors, the term registry, per-response coefficient vectors, the
statistics, the coded design matrix and observed responses, the JSON-safe
contract) registers in the session model store, records a run in the §15.2
lineage as family `ml.rsm`, and exports through the REAL `save_model`
joblib artifact lane (the house `load_model` reloads it).

The demo experiment is SYNTHETIC — a 17-run rotatable CCD over
temperature/pH/substrate with two KNOWN noise-free quadratic surfaces,
labelled so wherever it travels; the known coefficients make the exact
recovery verification in the API suite possible.

## The tests

`tests/api/test_rsm.py` (38) covers the §6.8 families: CCD/BBD/FFD
generation counts and levels, the coded↔actual conversion, seeded
randomisation determinism, the two-block split, factor-bound + design
refusals, the term registry, exact coefficient recovery over the KNOWN
surfaces, the ANOVA partition with the noise-free nulls, the significant
lack-of-fit for a cubic surface, the complete-case dropped-row report,
the rank-deficiency disclosure, the hierarchy flag-never-remove, the run
lineage, the joblib export + reload, the diagnostics block, the
prediction CI/PI + zone vocabulary, the contour response and
desirability modes with refusals, the desirability arithmetic (all four
goals, exact), the fine-grid optimality check, the constrained
optimisation with the feasible mask, the infeasible refusal, determinism,
and the end-to-end design→observe→fit→predict→contour→optimise workflow.
The vitest `RsmLane` suite (9) and the drift guard (16) pin the frontend
contract. The contract export stands at 128 paths / 83 schemas.

## Limitations (stated, never hidden)

- No mixture/process-noise modelling, no non-polynomial surface
  families, no Bayesian RSM.
- Blocking is the classic two-block CCD split only (BBD/FFD refused with
  a typed error).
- Lack-of-fit needs genuinely replicated design points; nulls
  elsewhere.
- The Pareto alternatives are a curated deterministic sample, not the
  exhaustive front.
- The optimiser never leaves the declared factor ranges; extrapolated
  PREDICTIONS are flagged with the zone vocabulary instead.
