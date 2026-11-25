# Generic Multi-Response / Multi-Output ML — part17 §5, the M1 lane

**Status:** VERIFIED_COMPLETE (round 358, 2026-11-25; this document is the
part17 §8 documentation the M1 lane owed, delivered with the r359 RSM
lane) — the backend module `src/pharmaml/ml/multiresponse.py`, the API
route `src/pharmaml/api/routes/multiresponse.py` (four paths), the
ML-workspace section (stage 20), the API suite
`tests/api/test_multiresponse.py` (38/38), the vitest `MultiresponseLane`
suite (6/6), and the drift guard `tests/test_frontend_multiresponse.py`
(12/12) are connected and verified.

## What this lane is — and is not

The generic multi-target supervised-learning pipeline the domain lanes
(drug-response, response models, multi-objective HPO) never were. It is
NOT the Response Surface Methodology of part17 §6 (that lane explains a
designed experiment; this lane predicts) — the distinction table lives in
both documents.

## The strategy registry — the mechanism stated, never implied

`STRATEGIES` rides three honest mechanisms (the scikit-learn multioutput
guide's own taxonomy, verified against the pinned sklearn 1.5.2 APIs):

| mechanism | meaning | the strategies |
| --- | --- | --- |
| **native** | the estimator's own `fit` accepts a 2-D Y | PLS2 (`PLSRegression`), `MultiTaskElasticNet`, `MultiTaskLasso`, `Ridge`, OLS, `KNeighbors`, DecisionTree/RandomForest/ExtraTrees, `MLPRegressor` |
| **wrapper** | `MultiOutputRegressor` clones one single-output base per target — the ONLY honest route for `SVR`/`GradientBoosting` whose fit enforces 1-D y | seven base choices |
| **chain** | `RegressorChain` feeds each target's predictions as features to the next (order matters; error propagation is the price) | the explicit order param |

The sample-weight capability is stated per strategy (the wrapper
forwards it; the chain/PLS2/MultiTask/MLP fits accept none — a
capability-honest 422 elsewhere).

## The masked multi-task missing lane

`PerTargetMultiOutput` — the mandate's "a compatible masked multi-task
approach, implemented and tested": each target's clone is fit on that
target's own non-missing rows (one target's gaps never erase another's).
The `per_target` missing strategy rides the wrapper mechanism ONLY; a
native/chain estimator needs the complete Y block — a typed refusal,
tested.

## The leakage discipline

The split fires FIRST (GroupShuffleSplit when group-aware, disjointness
asserted); the feature pipeline (median impute + standardize) and the
per-target scalars are fit on the TRAIN partition only. The
contract-level proof pins the scaler means equal the train means, never
the full-frame means; the metrics read in raw units after per-target
inverse scaling.

## The evaluation surface (§5.5)

Per target: R2 / RMSE / MAE / MSE / explained variance, each with its own
test-row count (per-target non-missing test masks — the anti-silent-drop
rule). The uniform and user-weighted **aggregates are reported ALONGSIDE,
never instead** — a weak target can never hide behind a strong aggregate.
Plus: the bounded residual samples (≤ 500, the bound reported), the
complete-case Pearson target correlation matrix, the k-fold CV over the
same full procedure (the leakage rule holds inside every fold; GroupKFold
when groups exist), and the weak-target-never-hides proof.

## The persistence contract (§5.6)

The `MultiResponseBundle`: the fitted estimator, the feature pipeline,
the per-target scalers, and the JSON-safe contract (target names/order/
units, scaler metadata, feature names, preprocessing, the estimator
strategy + hyperparameters, the split strategy, the seed, the per-target
metrics, the environment metadata). Pickles through the session model
store and reloads byte-parity; exports through the REAL `save_model`
joblib artifact lane.

## The API (4 paths)

`/api/v1/multiresponse/registry` (vocabulary), `/train` (dataset mode over
the REAL typed store or the seeded SYNTHETIC demo frame — labelled
synthetic wherever it travels), `/predict` (the STABLE contract target
order, per-target inverse scaling), `/models` (the comparison surface).
The runs record in the §15.2 lineage as family `ml.multiresponse`.
Sync-only light compute with the 5000-row bound 422'd up front.

## The tests

`tests/api/test_multiresponse.py` (38): the registry vocabulary, dataset
validation, two- and three-output round trips, PLS2 / MultiOutputRegressor
(with a GBR base) / RegressorChain / MultiTaskElasticNet, every-strategy
training, complete_case loss reporting vs per_target row retention, the
masked wrapper unit, the target-scaler leakage discipline, group-aware
split disjointness, exact per-target metrics + aggregate arithmetic, the
correlation matrix, CV summaries, the weak-target-never-hides, prediction
column ordering + stability, the typed 422/404 lanes, serialization
reload parity, the contract completeness, the artifact export with the
house `load_model` reload, the run lineage, and the end-to-end workflow.
The vitest suite (6) and the drift guard (12) pin the frontend contract.

## Limitations (stated, never hidden)

- No per-target loss aggregation inside the fit (the aggregate lives in
  the evaluation, not the objective).
- No native cross-target covariance modelling beyond what
  PLS2/MultiTask penalties themselves carry.
- Sync-only light compute (≤ 5000 rows); heavier training rides the
  other ML lanes.
- The demo frame is SYNTHETIC — every number it produces is a model
  estimate over generated data, never an experimental fact.
