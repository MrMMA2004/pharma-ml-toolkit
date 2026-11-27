# ROUND 368 SCIENTIFIC VALIDATION REPORT (part19, Phase 5)

Scope: the existing algorithm surface audited as-is (no algorithms added merely to grow counts). All
claims below were verified against the actual source (file:line evidence held in the working notes)
and, where marked *executed*, by live test runs this round on the pinned stack (sklearn 1.5.2 /
torch 2.14.1+cpu / rdkit 2026.9.1).

## A. Classical supervised — verified

- **Registry task compatibility: zero mismatches.** `MODEL_REGISTRY` (`ml/models.py:326–683`) holds 33
  families: 16 both-task, 9 classification-only (logistic, ridge-classifier, SGD-classifier,
  lasso/elastic-net logistic, GP-classifier, gaussian/multinomial/bernoulli NB), 8 regression-only
  (lasso, ridge, elastic-net, linear, huber, PLS, GP, HSVR). Every tag matches the estimator's real
  sklearn capability; `build_model` (`models.py:1066–1070`) raises the typed refusal on cross-task
  construction — *executed live*: every buildable pair builds a genuine estimator, every cross-task
  pair raises (`Model 'linear_regression' supports 'regression', not 'classification'`).
- Input + hyperparameter validation: `build_model` validates params against `get_params() ∪` the
  constructor signature (incl. the CatBoost under-reporting fix); `train_model` validates params,
  `cv ≥ 2`, and rows-vs-folds (`training.py:326–410`).
- Preprocessing: median imputation → scaling (numeric) or most-frequent + one-hot
  (categorical auto-detected) inside the sklearn Pipeline — **fitted on the train partition only**
  (pinned by `test_neural.py:840–854` and `api/test_multiresponse.py::
  test_target_scaler_uses_train_partition_only` — note: the file named `test_leakage.py` is the
  build-machine home-path scrub, not preprocessing leakage; the correct citations are the two above).
- Class imbalance: `class_weight` knob in HPO presets, stratified splitting with `min_class_count`,
  balanced-accuracy CV fallback for proba-less estimators; **no SMOTE/imblearn** (imbalanced-learn
  catalogued external — documented limitation, §L).
- CV + HPO: GridSearchCV with ROC-AUC-OvO scorer (balanced-accuracy fallback); from-scratch
  RandomSearch / TPE (real Parzen kernels) / Hyperband / pruners / parameter importances
  (`ml/tuner.py`) — pinned by 201 HPO tests.
- Reproducibility: `random_state=42` in every factory; torch RNG snapshot/restore around fit;
  joblib bundles carry sklearn version + UTC timestamp + input-report sidecars (`ml/persistence.py`).
- Metrics: regression RMSE/MAE/R²/max-error/Pearson-r (+ train mirrors + r2_gap); classification
  accuracy/balanced/precision/recall/F1(macro+weighted)/MCC/ROC-AUC/per-class table/confusion
  (`ml/evaluation.py`); learning curves (`viz/ml_plots.py`, `viz/diagnostics.py`).
- XGBoost string labels solved by the `_IntegerLabelClassifier` encoder round-trip.

## B. Unsupervised — verified, distinctions enforced

- Implemented: KMeans, Agglomerative (4 linkages), GMM, KDE, DBSCAN (+ k-distance eps),
  MeanShift (+ bandwidth estimate), **SpectralClustering** (`ml/clustering.py:474`),
  PCA, t-SNE, Isomap, LLE, MDS, **SpectralEmbedding** (`ml/manifold.py:239–244`),
  IsolationForest (`ml/anomaly.py:175`), from-scratch UMAP (`ml/umap.py`).
- **Spectral Embedding is never classified as Spectral Clustering**: the two modules cross-document
  the distinction ("this object assigns cluster labels; that one returns coordinates",
  `clustering.py:507–510`); a live source inspection confirms disjoint constructor use.
- The API surface (`routes/unsupervised.py`) separates clustering / anomaly routes and states
  "no target column is consulted, no train/test split applies"; degenerate outputs (all-noise DBSCAN,
  single-cluster mean-shift) are returned honestly, never fabricated.
- O(n²) ceilings: t-SNE > 6,000 rows and MDS > 2,500 rows are refused (documented).
- HDBSCAN is catalogued data only (external verdict), never claimed implemented.

## C. Deep Learning — verified with the honesty constraints intact

- `DeepMLP` / `ResidualMLP` (`ml/neural.py`): batchnorm-needs-batch≥2, 2-D/finite input guard,
  feature-name mismatch guard, stratified validation split (every class keeps ≥1 train row), scaler
  fit on train only, early stopping with best-weights restore, device `auto`→CPU fallback with
  fail-closed explicit-cuda-without-device, portable NumPy weight dicts (pickles load without torch),
  NaN-divergence refusal, seeded determinism.
- **Tabular Transformer** (`ml/transformer.py`): FT-Transformer-spirit feature-token attention
  **trained from scratch** — the module header and the `/dl` route both state no pretrained molecular
  Transformer is claimed anywhere (pinned by doctests + route tests).
- **Sequence CNN** (`ml/cnn.py`): 3-D-only sequence contract — *executed live*: feeding a 2-D tabular
  matrix raises exactly `X must be 3-dimensional … a 2-D tabular matrix belongs to the DeepMLP /
  Tabular Transformer families`; mixed-length sequences are a typed error; the module codifies
  "never reshape arbitrary tabular descriptors into artificial images".
- **Tabular Autoencoder** (`ml/autoencoder.py`): reconstruction/representation/dimensionality/
  anomaly-*ranking* (the error is "a heuristic outlier score, never ground truth"); the supervised
  `/ml/predict` lane refuses it with a 422 pointing at `/dl/autoencoder-encode` (route-test pinned).
- **GNN/MPNN** (`ml/gnn.py`): `GraphClassifier` **only** — graph classification; no GNN regression
  exists and no claim of it appears anywhere (grep-verified zero hits); the honesty header states
  teaching-scale CPU float64, node-level scatter, no dataloaders/hetero graphs.
- Route hygiene: `/dl/networks` carries per-network task tags; tabular predict refuses
  GraphClassifier/SequenceCNN models with 422s routing to their own lanes — *executed* in
  `api/test_dl_architectures.py` (27 tests, green this round).

## D. Generative AI — verified with the feature/molecular separation intact

- Molecular: char-level GRU `SmilesRnn` (torch lazily imported, seeded sampling), SELFIES codec with
  roundtrip audit + robustness experiment + valid-by-construction samplers + STONED-style mutants;
  GA with character/scaffold/topology/bioisostere mutations + splice crossover, invalid mutants
  discarded via re-sanitisation; objectives: QED (exact Bickerton constants), Ertl SA, penalized
  LogP, bands, similarity, substructure, novelty, multi; MOSES-lite distribution metrics
  (valid/unique/novel, IntDiv1/2, scaffold diversity, property shifts) with a deterministic
  pairwise cap.
- **Feature-space GAN/DDPM are never presented as molecular generation**: module headers state
  "NOT molecular generation: no SMILES, no sanitisation, no drug-likeness claim"; the prior store
  records `kind="feature_gan"|"feature_diffusion"` with `"molecular": False`; cross-kind sampling
  is a 422; route tests assert the "NOT molecules" notes verbatim (executed green).
- Feature-space samples are never registered as molecules (store separation verified).

## E. Reinforcement Learning — real implementations, honestly scoped

`ml/reinforce.py` implements, NumPy-only (no core-install RL dependency): **Q-Learning** (off-policy
TD(0)), **SARSA**, a compact **DQN** (replay buffer + frozen target net + hand-derived gradients),
**REINFORCE** with moving-average baseline, tabular **Actor-Critic**, explicit **MDPSpec** tables,
and a Gymnasium-shaped `GridWorldEnv` without the Gymnasium dependency; bounded episode/steps caps;
seed=42. The §6 separation from the supervised registry is **test-enforced**
(`test_reinforce.py:188–204`: RL keys absent from `MODEL_REGISTRY`, the module must say
"Experimental reinforcement learning"); routes label everything `experimental`. Honest scope:
deterministic toy grids + MDP tables only — "no pharmaceutical RL environment is invented here."
*Executed*: `test_reinforce.py` green this round.

## F. Multi-Response ML and RSM — preserved and verified

- `ml/multiresponse.py`: PLS2 (native), MultiTaskElasticNet/Lasso, ridge, linear, KNN, tree, forest,
  extra-trees, MLP, MultiOutputRegressor wrapper (incl. GBR with the documented 1-D-y reason),
  RegressorChain — mechanism tags per entry; per-target metrics (r2/rmse/mae/mse/explained variance)
  with train-partition-only scalers. *Executed*: `api/test_multiresponse.py` (36) green.
- `ml/rsm.py`: CCD (rotatable α / faced / custom), Box-Behnken (≥3-factor guard), full factorial,
  quadratic coded models, **ANOVA with pure-error + lack-of-fit partition** (honest nulled when
  centers are insufficient), alias disclosure, contour grids, Derringer-Suich desirability
  multi-response optimisation with multi-start + polish. *Executed*: `api/test_rsm.py` (38) green.

## G. Scientific validation machinery — verified

- Splits (`ml/splits.py`): random / stratified / **scaffold-aware (Bemis–Murcko on the own
  molgraph engine)** / group / Kennard–Stone / temporal; plus butina/cold splits in benchmark.
  Integrity pinned by `test_splits_integrity.py`, `test_cv_scaffold_integrity.py`,
  `test_split_overlap_audit.py` — *executed* green this round.
- No preprocessing leakage (§A); identifier-leakage audits: `ml/split_audit.py`, `ml/file_split_audit.py`.
- Conformal: split-conformal regression intervals + split and **label-conditional** classifier sets
  — *executed* (`test_conformal.py` green).
- Applicability domain: kNN-distance AD (with self-distance exclusion fix), bootstrap uncertainty,
  quantile intervals; deep ensembles; GP σ via `predict_with_uncertainty`.
- Provenance: environment snapshots, file fingerprints, experiment manifests, model cards, and the
  run store's `ExperimentRunRecord` (dataset id, feature config, preprocessing, algorithm + toolkit
  version, requested AND tuned hyperparameters, seed, CV config, metrics, environment, artifacts,
  job ref — failed runs recorded too), SQLite-durable.

## H. Executed evidence this round (Phase 5 focused set — all green, 194 tests)

`test_reinforce.py` (RL §6 separation) · `test_conformal.py` · `test_splits_integrity.py` ·
`api/test_dl_architectures.py` (27: route task-mismatch 422s + feature-space kind/note asserts) ·
`api/test_multiresponse.py` (36) · `api/test_rsm.py` (38) · `tests/test_ml_catalog.py` (6 — **new
this round**: the best-of-ML catalogue snapshot invariants — 924 projects / 35 categories /
license distribution / category-count agreement / per-category score ordering / verdict-vocabulary
coverage — previously doctest-only). The full-suite chunk run rides Phase 12.

## I. Defects found and fixed this round

- **D-368-06 (fixed)**: the best-of-ML catalogue had no non-doctest regression coverage →
  `tests/test_ml_catalog.py` added (6 pins, green).
- No scientific mislabel, task-compatibility mismatch, or false capability claim was found on any
  audited surface.

## J. Honest-limitation register (documented in source, retained deliberately)

No pretrained Transformer weights are claimed (trained-from-scratch contract) · CNN is sequence-only
(2-D refused) · autoencoder anomaly output is a heuristic ranking, "never ground truth" · GNN is
teaching-scale CPU float64, node-level scatter only · feature-space GAN/DDPM produce synthetic
feature rows, NOT molecules, with no experimental-validity claim · MOSES-lite lacks FCD/SNN/fragment
similarity + distribution-learning plots (external weights needed); CRT torch generator absent ·
SA score degrades offline to complexity-only (fpscores.pkl is a Dataverse download); ML-model
oracles (drd2/gsk3b/…), docking/retrosynthesis oracles, FCD, PAINS catalogs stay out · RL is toy
environments only, labelled experimental, kept out of the supervised registry by test ·
`train_model` does not pre-validate y for NaN/±inf (sklearn rejects raw NaN; split path rejects NaN
but passes ±inf — historical contract) · SGD `early_stopping` deliberately not default (small-screen
crash hazard) · MultinomialNB keeps counts unscaled (negatives refused up-front) · proba-less
estimators use the balanced-accuracy CV fallback · spectral clustering has no honest auto-k ·
t-SNE/MDS row ceilings · no native class-imbalance resampling (imbalanced-learn external) ·
multiresponse has no per-target loss aggregation inside fit; per-target missing handling is
wrapper-mechanism only · GBR multi-output rides the wrapper lane (1-D y) · GPC multiclass is
sklearn's internal OvR of binary Laplace GPs · RSM rotatable-CCD axial runs sit outside factor ranges
(stated assumption) · best-of-ML coverage verdicts: image external; audio/geospatial/financial
out-of-scope; DL engines external by design.

## Verdict

The scientific surface is **validated against its correct input and task contracts** with the
honesty constraints intact and test-pinned; the register above is disclosed, not hidden.
