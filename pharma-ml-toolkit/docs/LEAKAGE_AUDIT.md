# Data-Leakage Audit — the rar, the toolkit, the web mirror

Round 100x-101. Scope, per request: *"check the entire zip file for
data leakage"* — every machine-learning file inside the original
`کدهای ماشین لرنینگ.rar` archive (118 recovered files), plus the
toolkit that upgrades it and the web playground that mirrors it.
The archive's files were audited one by one; the findings below name
the exact pattern and the exact lines.

## What "data leakage" means here

**Train/test contamination**: any statistic of the held-out test rows
influencing what the model sees during training. The classic QSAR
incarnation — and the archive's signature bug — is fitting the
imputer/scaler on the **full** table *before* `train_test_split`:

```python
X_scaled = scaler.fit_transform(X)                    # sees ALL rows
X_train, X_test, ... = train_test_split(X_scaled, y)  # split after
```

The test set's mean/std/median are then baked into the training
features. Held-out metrics come back optimistically biased — the
model has, subtly, already met the test rows.

## A. The archive — 27 scripts + 1 notebook leak

**112 `.py` files scanned; 38 contain a train/test split.**

| Pattern | Files | Verdict |
| --- | --- | --- |
| **A1 — fit-before-split** (imputer + scaler `fit_transform` on the full table inside `preprocess()`, split called afterwards) | 27 | **leaky** |
| **A2 — pipeline-based, split first** (`Pipeline([('imputer', ...), ('scaler', ...), ('clf', ...)])` fitted on `X_train` only) | 11 | clean |
| **A3 — notebook flow** | 1 | **leaky** (see below) |
| No split at all (descriptors, converters, docking, readers) | 74 | n/a |

The 27 leaky scripts (both tasks, every estimator flavour):

```
Classification/  ADAboost, ANN, CATboost, Decision tree, Elastic net, KNN,
                 Lasso, LightGBM, Logistic regression, RF, Ridge, SVM,
                 Stacking, Voting, XGboost
Regression/      ADAboost, ANN, ANN2, CATboost, KNN, LightGBM,
                 Linear regression, RF, Ridge, Stacking, Voting, XGboost
```

Representative anatomy — `Classification/RF.py`:

- `preprocess()` (lines 36–51): `SimpleImputer(strategy="median").fit_transform(X)`
  then `StandardScaler().fit_transform(X)` — **on the full frame**;
- `split_data()` (lines 54–60): `train_test_split` — afterwards;
- `main()`: `preprocess(df)` → `split_data(X_scaled, y)` — the leak is
  the call order.

The clean 11 (the archive's own best practice — `Huber.py`, `Lasso.py`,
`Elastic net.py`, `SVM.py`, the `*-class.py` wrappers): `split_data(df)`
returns raw splits; the imputer/scaler live **inside** the `Pipeline`
that `GridSearchCV` fits on `X_train` only. This is exactly the design
the toolkit generalised.

**A3 — `Classification/BBB/BBB_Prediction_model.ipynb`** (cell 0):

- line 113/115: `fillna(df[col].mean())` / `mode()` — imputation
  statistics from the **full** table;
- line 137–138: `scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)`;
- line 143: `train_test_split(X_scaled_df, y, ...)` — after;
- line 217: `GridSearchCV(...).fit` — hyperparameters chosen on the
  already-contaminated features.

Two aggravating details: (1) the tuned model is saved as
`tuned_random_forest_bbb_prediction_model.pkl` — a **bare**
`RandomForestClassifier` with `feature_names_in_` (117 RDKit descriptor
names); **the scaler is never persisted**, which is why cell 6 refits it
"on the original training data" at deployment — the pipeline's missing
half; (2) the `.pkl` itself embeds **no training rows** (fitted trees
store split thresholds, not data — verified by disassembling the
pickle: only estimator parameters and feature names). The leakage is in
the flow, not the artifact.

## B. The toolkit — immune by construction, now pinned by tests

`pharmaml.ml` was designed against this exact pattern:

- `make_train_test_split` (preprocessing.py) splits **raw** features —
  no transform is applied inside or before it;
- `train_model` (training.py) builds `Pipeline([("prep",
  build_preprocessor(...)), (model)])` and fits it **on `X_train`
  only** — the imputer's median and the scaler's mean/std come from
  the train rows alone;
- `GridSearchCV`/`compare_models`/the tuner receive the pipeline, so
  every CV fold refits the preprocessing on the fold's training part;
- `persistence.save_model` stores the **whole pipeline** — the scaler
  travels with the model (the BBB notebook's missing half).

**New regression guards** (`tests/test_ml_pipeline.py ::
TestLeakageGuards`, 4 tests, this release):

1. `test_split_returns_raw_features` — the split hands back the
   original units, not a pre-centred column;
2. `test_preprocessor_statistics_are_train_only` — train block ~10,
   test block ~±1e6: the fitted `StandardScaler.mean_` equals the
   train mean to 1e-12, and the extreme test rows produce honest
   extreme extrapolations (a leaky scaler would have absorbed the
   outliers into the train regime — extremeness is the no-leak
   signature);
3. `test_imputer_median_is_train_only` — one NaN in train: the
   imputer's median is the train median;
4. `test_evaluate_on_foreign_test_rows_is_inert` — after training,
   blowing up the test rows cannot move the model's train predictions
   by a bit.

## C. The web mirror — same disease, surgically removed

The browser playground (`/api/ml/train|tune|compare`) had inherited the
rar's pattern in its CSV paths:

- `train/route.ts` `runRegression`: `xs = zscore(csv.x)` — full column
  — before `shuffledSplit`;
- `runClassification`: `zscore(x1)`, `zscore(x2)` — full columns —
  before the split;
- `ml-data.ts` (shared by tune + compare): same full-column z-scores
  feeding the CV folds and the held-out refit.

Why it mattered even though every fitter standardizes internally with
`fitStandardizer` on the rows it is given: the **outside** transform
contaminated the metric geometry. kNN distances and RBF kernels live in
the double-standardized space whose per-axis scale was set by
**full-data** statistics — the test rows' spread literally resized the
kernel. (For the linear-in-features families the affine composition
cancels in the final predictions; for kNN/svm_rbf it does not.)

**The fix (this release)**: the outside z-score calls are deleted —
CSV features reach the fitters in **raw units**, exactly like the
generator paths always did. The fitters' internal train-statistics
standardisation is now the only scaling, which also makes every CV
fold per-fold rigorous automatically (each fold's fitter standardizes
on its own training part — something a single precomputed transform
cannot even express).

Consequences, verified by request:

- generator-data responses are **bit-identical** to before (the
  generator paths never pre-scaled — pinned: PLS train R² 0.9121 /
  RMSE 0.4103 / CV 0.92 ± 0.0132 / coefficients 0.4582 + 1.387);
- CSV-data responses change **honestly** (charts now plot the raw
  column units; `datasetMeta.features` drops the `(standardized)`
  suffix — the coefficient cards keep their `x̂ = standardized x`
  legend, which is truer than ever: the standardization is
  train-fitted);
- the LEAKAGE RULE is now documented at the top of `ml-data.ts` for
  the next surface that joins.

## D. The downloadable ZIP — audited, then scrubbed clean

The `pharma-ml-toolkit-v<version>.zip` stream (toolkit/download route)
packages the toolkit tree with caches, virtualenvs, `.git`, and
generated `artifacts/` excluded. The archive's own dataset-specific
`.pkl` weights were never imported into the toolkit (COVERAGE.md:
"intentionally not shipped"); the example tables are deterministic
synthetic data regenerated by `tools/build_example_data.py`; no
`.env`, credentials or user data exist anywhere in the tree.

The one-by-one content scan (every text entry, 305 total) found a
single leakage channel: **build-machine paths**. Library deprecation
warnings inside the executed notebooks' committed outputs printed
their source as `~/.venv/lib/python3.12/site-packages/...`, and
20 `scratch/` driver scripts carried `~/...` run instructions.
Both are scrubbed as of 1.2.44: the scratch scripts' run hints are now
portable (`python scratch/foo.py`), the two affected notebooks' outputs
rewrite the site-packages prefix to the neutral `site-packages/`, and
`reexecute_all_notebooks_64.py` performs the same rewrite automatically
after every execution (the prefix computed from `sysconfig`, never
hardcoded). A full-archive scan for absolute home-directory prefixes
(Unix ``~user`` roots and the superuser root) now returns **zero** matches — the
shipped ZIP is build-machine-clean.

## E. Coverage consequence — the notebooks now tour every family

The same audit asked whether the notebooks and `playground.py` cover
the entire content. The playground already did (its model combobox
lists `list_models(task, only_available=True)` dynamically). The
notebooks did not: 10 of 24 families had any demonstration. The ML
workflow gains **§11 "The full registry — every family, one tour"**:
`list_models(task, only_available=False)` for both tasks, every
available family fitted on the notebook's own splits and scored on the
same held-out rows, unavailable families listed with their missing
requirement. A family added tomorrow joins the tour without an edit —
and the tour doubles as a full-registry smoke test (37 registry
entries, one table; ~7 s).

## Verdict

- **Archive**: 28 of the 38 splitting files leak (27 scripts + the BBB
  notebook). The toolkit is the fix: one pipeline, fitted on train
  rows only, scaler persisted with the model.
- **Toolkit**: leakage-free by construction; now pinned by 4
  regression tests that make the contract un-regressable.
- **Web mirror**: the inherited full-column pre-scaling is removed;
  generator responses bit-identical, CSV responses honest.
- **ZIP**: no sensitive-data or training-data leakage channels.
