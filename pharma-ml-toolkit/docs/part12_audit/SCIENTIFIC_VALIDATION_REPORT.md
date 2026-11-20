# Scientific Validation Report (part12)

Method-level audit of the scientific engines against part12's focus areas.
Statuses cite implementations verified in source this session; numerical
claims cite the repo's own regression suites where applicable.

## Applicability Domain (part12 PART 11)

| Method | Implemented | Definition & threshold | GUI |
| --- | --- | --- | --- |
| kNN distance | **YES** (`ml/applicability.py`, `knn_applicability_domain`) | mean k-NN distance in feature space, percentile threshold (self-distance excluded) | ML AD stage plot + sample verdicts (WIRED) |
| Bootstrap uncertainty | **YES** (`bootstrap_uncertainty`) | seeded clone-ensemble SD (regression) / probability SD (binary) | uncertainty plot (WIRED) |
| DomainBox (metadata box) | **YES** (`DomainBox`, `check_domain_box`) | per-feature min/max box audit | NON-GUI (documented) |
| Leverage / Williams | **YES** (CLOSED r303) — `ml/applicability.py::leverage_williams` standalone: hat-matrix leverage, h\* = 3(p+1)/n, standardized-residual band, regression-only guards | ML workspace Williams lane (two-panel Plotly + per-row tables) — WIRED |

- **Uncertainty is NOT conflated with AD**: the routes and plots keep them
  separate (`/analyze` returns both; two distinct plots).
- The uploaded reference `Applicability_domain_analysis.py` was absorbed
  into `ml/applicability.py` (docstring cites it).
- Documented decision frame (part12 says "Leverage / Williams plot **where
  implemented**"): implemented as of r303 — the standalone method + the
  Williams scatter (h\* and residual±3σ lines) + the GUI lane landed
  together (the FEATURE_GAP item closed with 23 unit + 3 figure + 7 route
  tests; the scikit-learn/RDKit references recorded; re-verified in-tree
  at the r307 reconciliation).

## Semi-supervised learning (part12 PART 9) — verified complete

- `ml/semi_supervised.py` (1,152 lines): SelfTrainingClassifier
  (threshold/k_best/max_iter), LabelPropagation, LabelSpreading
  (knn/rbf, n_neighbors, gamma).
- Unlabeled convention: `-1` sentinel, type-aware validation.
- Leakage safety: labeled-only stratified split for evaluation; pseudo-labels
  never treated as ground truth; original-vs-pseudo provenance recorded.
- GUI: `/semi-supervised/train` wired to the ML stage; the `/evaluate`
  GUI consumer landed r303 — its wiring exposed and fixed the job-mode
  TypeError (the COMPLETION pin rides RELEASE_READINESS).
- Graph-method cost warning: knn kernel options exist precisely to avoid
  the dense-graph blowup; the rbf path documents the memory caveat.

## Chemistry Lab (part12 PART 8)

- Descriptor families: rdkit 2D suite, Morgan (fixed radius, counts),
  Mordred full suite (Error/Missing→NaN + aggregate warning), 3D shape
  (ETKDGv3 + MMFF94s→UFF, 9 columns, seed 42). The Datasets workspace
  exposes set-level selection with job-mode progress.
- The 19–20 family feature factory (`chem/features.py`) covers
  atompair/ecfp/avalon/rdkit-fp/morgan-full/morgan-scaffold/maccs/fcfp/
  topological/pattern/layered/estate/erg/shape/rism with ranking and
  two-layer selection — exposed through the web feature-factory card AND
  the API.
- Fingerprint parameters: nBits (0, 65536], radius ≥ 0, count-vs-bit
  modes — validated; irrelevant parameters are not shown for algorithms
  that don't use them. **Chirality toggle CLOSED r303**: `include_chirality`
  (morgan-only, default OFF — B-5 resolved with the byte-identical default
  history preserved).
- Invalid-molecule/missing-value reporting: Mordred NaN policy + invalid
  SMILES surface in standardize/dataset audit lanes.

## ML/DL algorithms (part12 PART 10)

- 33-family MODEL_REGISTRY (24 sklearn classical + 3 boosting
  [xgboost/lightgbm/catboost] + DL + hierarchical SVR + voting/stacking).
- Validation: stratified + group + scaffold splits (near-duplicate
  rationale documented), nested-CV, conformal (regression + classification
  sets), calibration, SHAP explainability, permutation importance.
- The absorbed-research layer (~50 modules: ebm, bayesopt, metalearn,
  privacy, quantization, pruning, drift, subgroups…) is library-surface
  with doctests — tiered honestly in the matrix, not counted as GUI-wired.
- Optuna deliberately NOT adopted (existing tuner + hyperband lanes cover
  the workflows; no measured advantage — recorded decision).

## PK/PD and MD (honest boundary)

- PK/PD: NCA (linear+log AUC, terminal-phase regression with r²/span),
  steady-state, iv-bolus, bateman, superposition, PBPK (5-compartment +
  IVIVE + DDI Guest limits), PD Emax/sigmoid, population NLME (FOCEi-lite),
  diagnostics (VPC/bootstrap), units validation (positive doses, ordered
  series).
- MD: **trajectory analysis only** — RMSD/RMSF/Rg, H-bonds, SASA-adjacent
  contacts, PCA/tICA, ENM (ANM/GNM), PLUMED-style CVs, xvg parsing. The
  repo does NOT claim MD simulation — verified wording in README/docs.

## Data quality / reproducibility (part12 PART 14)

- Missingness, duplicates, constant/high-cardinality features, Cramér's V,
  outliers (z/IQR), class imbalance, target distribution — via the
  profiling layer + dataset workspace validation.
- Leakage: split audits (exact-key overlap), scaffold overlap via
  scaffold-split strategy, group leakage via group splits, 4 pinning
  regression tests (raw split, train-only scaler/imputer stats, inert
  foreign rows); 28 leaky rar-era files documented in LEAKAGE_AUDIT.md.
- Determinism: seeded generators, pinned stack
  (numpy 1.26.4/torch 2.14.1+cpu/rdkit 2026.9.1/shap 0.54.1), example data
  regenerating byte-identically (DATA_AUDIT 526/526), notebook freshness
  guards, BLAS-rounding provenance policy (6-decimal float64 rounding in
  release artifacts — r298/r299).

## Verdict

Scientific core: **PASS** across the audited areas; the two gaps this
audit carried (B-4 leverage/Williams standalone, B-5 chirality toggle)
were CLOSED by r303 — re-verified in-tree at the r307 reconciliation —
and one honest non-claim stands (no MD simulation). No mock/synthetic result masquerades
as a real computation in any GUI lane audited.
