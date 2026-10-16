# Task 26-R6 — full-stack-developer (playground model library rebuild)

## Task
Rebuild the 17-model ML Playground library (10 regression + 7 classification) lost to a sandbox reset. Writable: `src/lib/ml-math.ts`, `src/app/api/ml/train/route.ts`, `src/components/explorer/playground/ml-playground.tsx`, `src/lib/toolkit-types.ts`, `pharma-ml-toolkit/scratch/probe_playground_models_26.py`. No store.ts edits allowed → model+hyperparameter selection is component-local state, re-derived from the persisted training result (model name + echoed params) on remount/reload/restore.

## Key decisions
- Byte-compat pins captured before edits (sha256 prefixes): linear `030c5a3d…`, polynomial `220116b3…`, ridge `1767d3d1…` (r2 = 0.9122895655345933 — spec pin OK), logistic `b97ec92b…` in `/tmp/task26/base_*.json`.
- Response honesty: `coefficients` only for linear-family+logistic (legacy shape unchanged); trees/ensembles → `featureImportance` (normalized impurity reduction); knn/MLP/NB/SVM → `modelParams` echo. No fabricated coefficients.
- Seeds: split = `mulberry32(effSeed)` (unchanged); CV folds = `mulberry32(effSeed + 7919)` (unchanged); model internals (bagging / MLP init) = `mulberry32(effSeed + 104729 + 101·(fold+1))`, fold −1 = main run.
- sklearn-parity choices: lasso/enet use sklearn's objective scaling `(1/2n)·SSE + α·(l1·‖w‖₁ + ½(1−l1)·‖w‖²)`; NB uses sklearn's `var_smoothing=1e-9·max var` epsilon; SVM mirrors liblinear's L2-loss squared-hinge with the bias treated as a regularized constant-1 feature (LinearSVC semantics); trees use first-strict-max tie-breaking over features in order.
- zod bounds: k odd 1–25 (k=2 → 400), epochs 10–2000 (9999 → 400), nEstimators 3–200 (500 → 400), maxDepth 1–12, rounds 5–200, learningRate 0.005–1, hidden 2–64, lr 0.001–1, l1Ratio 0–1, C 0.01–100; bogus model name → clean 400 from per-task registry.
- UI: grouped Select (Linear family / Trees & ensembles / Instance-based / Probabilistic / Neural) with honest one-line hints; contextual hyperparameter controls (sliders via calc-ui ParamSlider, alpha/l2 inputs reuse the store's legacy string fields); Learned-parameters card is model-kind aware (coefficients / importance bars / hyperparameter echo with graceful post-reload fallback notes).

## Status
See final worklog entry (Task ID 26-R6) for verification evidence.
