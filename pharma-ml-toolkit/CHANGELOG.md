## [1.2.92] — 2026-10-27

### Fixed / Added — the ninth engineering review (100x-154): the round-36 web twin audited adversarially, a fake-zero RED closed in BOTH correlation twins, and the LAST unabsorbed D2.py gap (SHAP) shipped

The instruction: *"Review the current codebase and implementation,
troubleshoot potential issues, and continue improving the engineering
details."* The round audited v1.2.91's web interpretability twin with
adversarial numerics probes the round-36 parity grid never asked —
and closed every verified finding in BOTH languages, then shipped the
upload's final missing block as a first-class optional-gated surface:

* **The fake-zero RED (both twins)**: a CONSTANT 1e300-scale column
  produced **a fabricated r = 0.0** — the web `pearsonCorrelation`
  (the mean's ULP-scale rounding error ~1e284 makes the "zero"
  deviations nonzero, their squares overflow to Infinity,
  `num/Infinity = 0`) AND the desktop `signed_feature_correlations`
  (`np.corrcoef`, the identical arithmetic) — while scipy honestly
  returns NaN. Live in the route: ridge + a 1e300-constant CSV column
  rendered "big +0.0000 (weak)" — the exact fake zero both twins'
  docstrings promise never happens. The fix is the **offset-then-scale
  recipe** in both languages (re-centre on the first element — exact
  under Sterbenz, so a constant column collapses to exact zeros at
  ANY scale; re-scale by the max deviation — every product finite at
  any scale; then the two-pass WITH the scaled-mean subtraction — the
  centering is not optional, the offset alone leaves the
  `n(x̄−x₀)(ȳ−y₀)` cross term).
* **The lost-signal REDs (both twins)**: extreme-scale VARYING columns
  (1e300 overflow / 1e-300 underflow) were silently DROPPED (NaN
  masquerading as "undefined", with the misleading "constant" log
  wording) where scipy reports a defined r. The recipe keeps them —
  and matches scipy bit-for-bit (`huge_magnitude` / `tiny_values` at
  2.8e-17 in the new parity pair).
* **The honest tolerance story**: `huge_offset` (a 1e15-offset column
  whose jitter is an exact linear function of the target's — analytic
  r = 1.0) has an irreducible float64 input-quantization floor (the
  input itself is quantized at construction, ULP(1e15) = 0.125 vs
  jitter ~1): the parity contract widens to 5e-3 vs scipy and pins
  |recipe − 1.0| ≤ 2e-3 — the recipe is STRICTLY closer to truth than
  scipy itself (0.9990 vs 0.9968). `near_constant` is scipy's own
  `NearConstantInputWarning` territory (~1e-7 of a 1e-12 jitter); the
  recipe is exact and is pinned against the ANALYTIC truth at 1e-12.
* **NEW parity pair** `scratch/verify_corr_parity_154.py` + `.ts`
  (14-case adversarial grid: extreme scales, huge offsets,
  near-constant, exact cancellation, n=100 000, the mirrored zero,
  the constant-column fake-zero pins, mismatch/one-point/NaN) wired
  into the suite as `TestCorrelationParityAdversarial` — both sides
  run the SHIPPED code, worst |dr| 2.78e-17.
* **Classification runs get the correlation bundle** (the parity gap):
  the desktop twin serves them (no task guard — a 0/1 label is a
  numeric target, the point-biserial r); the web route now builds the
  bundle over the run's OWN held-out rows in the SAME request that
  trained the model, and the card notes the point-biserial read.
  Live-verified: the moons bundle (x2 −0.76 / x1 +0.54).
* **The GUI heatmap join's ndarray fallback requires exact length**
  (the last amber): the old `values[: len(frame)]` silently truncated
  a LONGER `y_test` — a partial pairing in the one fallback path where
  misalignment is real. Pinned end-to-end via the library's typed
  length guard (defense in depth: the ranking refuses the mismatch
  first, the join refuses any non-exact length).
* **SHAP — the LAST unabsorbed D2.py gap (its lines 472-528) — shipped
  as `pharmaml.ml.explain`** (the round-33 audit's standing note
  finally paid): `shap_summary` (TreeExplainer over a fitted tree
  model; LightGBM / XGBoost / CatBoost / sklearn trees) returns the
  typed `ShapSummary` record (names, mean |SHAP|, the full values
  matrix, the feature values the beeswarm colours by, the base value);
  `plot_shap_importance` (the upload's own top-20 bar) and
  `plot_shap_beeswarm` (**pure matplotlib** — jittered (sample,
  feature) points coloured by the feature's own value on the house
  amber→teal ramp, seeded jitter, the base-value caption — the plotting
  layer never needs shap installed). Optional-gated exactly as
  predicted: `pharmaml[explain]` (`shap>=0.45,<0.60`, runtime-verified
  on the numpy 1.26.4 stack); the `DependencyError` carries the exact
  pip command, non-tree estimators get the typed error naming the
  `permutation_feature_importance` sibling (never a silent
  KernelExplainer fallback), multiclass is refused with the same
  honesty. The Results card gained the fourth row (SHAP beeswarm /
  SHAP top-20 bar) over the run's OWN held-out rows — the D2.py
  upload is now FULLY absorbed, every block a first-class surface.

## [1.2.91] — 2026-10-27

### Fixed / Added — the eighth engineering review (100x-153): the round-35 additions audited, one red + three ambers closed, the interpretability bundle gained its web twin

The instruction: *"Review the current codebase and implementation,
troubleshoot potential issues, and continue improving the engineering
details."* The round audited the round-35 additions (v1.2.90's
interpret.py, the heatmap twins, `filter_low_variance`, the GA
bioisostere wiring, the train-side overfit gap, `cross_model_importances`,
the GUI's third button row) with six adversarial probes — and shipped
every verified finding's fix plus the standing "web twin" candidate:

* **The GUI correlation heatmap's target join is now SWAP-PROOF**
  (playground.py, the audit's one RED): the handler previously joined
  the target column from the *session frame* (`session_frame.loc[...]`)
  even though the run tuple already carries the honest `y_test` — a
  same-shaped swapped dataset (RangeIndex covering the split's
  indices) silently passed the "rows align" guard and **paired the new
  dataset's y with the old run's features** (the audit's probe read
  a-y = +0.17, noise, where the run's own target gives +1.00). The
  join now takes the run's OWN `y_test` (aligned by construction,
  swap-proof, and no longer racing a concurrent dataset load); a GUI
  regression test reproduces the exact natural swap flow and pins the
  +1.00 cell.
* **`filter_low_variance` cuts inf-poisoned columns** (the audit's
  second amber): after the `fillna(0)` convention a NaN variance can
  only arise from inf/-inf cells — a column every downstream estimator
  rejects with a typed error. Such columns were silently KEPT (no
  ledger entry, no warning); they are now CUT, ledgered as NaN
  (undefined — the same verdict `rank_features` already reports, so
  the twins stay in agreement) and named in a warning.
* **`filter_low_variance`'s threshold guard rejects non-numbers**
  (third amber): `float(True)` is `1.0` and `float('0.01')` is `0.01`,
  and the old `threshold = float(threshold)` coercion accepted both —
  the guard now fires BEFORE the coercion, the house family
  (interpret's radius/bit, viz's `top_n`) rule.
* **`signed_feature_correlations` skips complex-dtype columns** (fourth
  amber): pandas counts complex128 "numeric" and the dtype-float cast
  silently discards the imaginary part (an unread ComplexWarning) — a
  half-honest correlation. Complex columns now ride the non-numeric
  skip path with the naming warning.
* **The web interpretability twin** (the standing candidate): the ML
  playground's results column gained a **Feature–target correlations**
  card — signed Pearson-r bars (teal pulls the target up, rose pulls
  it down, undefined dropped never zeroed) plus the compact top-N
  correlation matrix with a rose→white→teal diverging scale, both
  computed over the run's OWN held-out split *in the same request
  that trained the model* — inherently swap-proof (the desktop round's
  red finding, designed out on the web from the start). The pure core
  is a new shipped `pearsonCorrelation` in `src/lib/ml-math.ts`,
  pinned against `scipy.stats.pearsonr` by a paired parity probe
  (11-case seeded grid, worst |Δr| = 8.3e-17, NaN-agreement on every
  undefined case including the length-mismatch case where the TS must
  not silently truncate to the common prefix) — the 100x-153 parity
  pair, wired into the suite's web-parity guards.

The audit also *disproved* ~20 suspicions (recorded in
`scratch/audit_153.txt`): the GA's attempt-based operator bookkeeping,
the permutation-importance seed discipline, every typed guard probed,
and the signed-bars twin's honesty all held.

## [1.2.90] — 2026-10-27

### Added — the seventh engineering review (100x-152): the D2.py interpretability bundle + the overfit gap, four audit gaps closed

The instruction: *"Review the current codebase and implementation,
troubleshoot potential issues, and continue improving the engineering
details."* The round audited the round-34 additions line by line (the
quantile intervals feature verified clean — the real risk lived
elsewhere), repaired the environment after its twelfth reset, and
closed four of the D2.py audit's standing gaps — the upload's best
remaining interpretability ideas, each upgraded to the toolkit's
typed-record conventions with every original defect documented
honestly in the docstrings:

* **`decode_morgan_bit`** (`pharmaml.chem.interpret`) — a Morgan
  fingerprint bit finally *means* something: the bit-index →
  atom-environment decoding the upload wrote to
  `morgan_atom_environments.txt`. Returns a typed `MorganBitInfo`
  record (every `(atom, radius)` hit the bit fires on, each decoded
  with **the hit's own radius** — the original passed the fingerprint
  radius and thus drew the wrong environment for any bit that fired
  below it, a real decode bug, fixed and pinned; radius-0 hits are
  KEPT, the original silently dropped them all; the `DrawMorganBit`
  SVG legend rides along as `.drawing`). The multi-molecule example
  hunt is deliberately a caller-side loop — one call decodes one
  molecule completely instead of three molecules partially.
* **`signed_feature_correlations`** (`pharmaml.chem.interpret`) — the
  computation half of the upload's correlation bundle: every numeric
  column correlated with the target, sorted by |r| descending with the
  sign preserved. The original correlated only the first 1000 features
  "for speed" and appended a **fake `0.0`** wherever `pearsonr`
  raised — a failed column masquerading as "no correlation"; this port
  correlates everything and *drops* undefined correlations (logged,
  never zeroed). Pearson and Spearman both supported.
* **`plot_signed_correlations` + `plot_correlation_heatmap`**
  (`pharmaml.viz`) — the bundle's figure halves: the top-N signed bar
  (teal pulls the target up, the warm negative draws it down — the
  permutation-importance colour convention) and the pure-matplotlib
  feature × feature heatmap (the upload called seaborn; the toolkit's
  viz stays matplotlib-only) with the diverging house palette, a
  zero-centred scale, grey undefined cells and annotations only while
  matrices stay readable (≤ 12 columns).
* **`cross_model_importances` + `plot_permutation_heatmap`**
  (`pharmaml.ml` / `pharmaml.viz`) — the audit's GAP 3: the
  features × models matrix the upload drew after looping
  `permutation_importance` over its base models and meta-learner.
  The runner keys one `PermutationImportance` record per fitted model
  (same seed for every model, so cross-model differences reflect the
  *models*, not the shuffle luck — pinned by test), and the heatmap
  renders the top-N features by cross-model mean |importance| with a
  grey "no vote" cell for a feature a model never saw — honestly
  absent instead of a fake zero.
* **`filter_low_variance`** (`pharmaml.chem.features`) — the audit's
  GAP 6, the `xxx.ipynb` filtering cell's `VarianceThreshold(0.01)`:
  `rank_features` *ranks* by variance, this *cuts* by it, with the
  same `fillna(0).var()` definition so the two functions can never
  disagree about a column (pinned by test). Returns a typed
  `VarianceFilterResult` (the kept-column frame, the kept list, and
  the dropped ledger — every cut carries its own variance as
  evidence). Non-numeric columns pass through untouched with a
  warning; a mid-round test caught the first cut silently *dropping*
  the very carrier columns it promised to pass through — fixed before
  ship.
* **`RegressionReport`'s train side + `r2_gap`** (`pharmaml.ml`) — the
  audit's GAP 7: `evaluate_model(..., X_train=, y_train=)` fills
  `pearson_r` (the `fak_ml.ipynb` per-fold metric), `train_rmse`,
  `train_r2`, `train_pearson_r` and the `r2_gap` property — the
  overfit gap as one signed number instead of a vibe. The pair is
  all-or-nothing (half-passed raises) and regression-only (a
  classification train side has no rmse to mirror). The docstring
  keeps the honest caveat: train-side metrics are an overfit
  *diagnostic*, never a quality claim.

### Fixed — the audit's untracked-files finding (the red-suite bomb)

`src/pharmaml/generative/bioisostere.py` had shipped in the ZIPs since
its round but was **never committed to git and never wired into the
package** — imported by nothing, its own doctest importing
`from pharmaml.generative import bioisostere_moves` and failing, a
live failure waiting in every bare `pytest` run (the audit's Finding
1). The module itself is sound (all five doctest claims verified live
with RDKit), so this round *wired it up* instead of deleting it: both
functions export from `pharmaml.generative`, and the genetic
optimiser gains the **`mutation="bioisostere"`** mode the module was
designed for — a single-atom heteroatom exchange drawn per offspring
(ester O → amide N, ether O → thioether S, ring CH2 → N/O/S), with
the adaptive operator-selection book seeing the two-operator mutation
family exactly as topology mode does, and `None` draws falling back
to the character mutation so evolution never stalls. `__main__.py`
(the `python -m pharmaml` entry) joins the repo as well; the web-side
orphan `data-file-preview.tsx` was left untracked deliberately (zero
importers, superseded by the committed `data-file-view.tsx`).

### The GUI (playground.py, +782 lines)

Four first-class surfaces: the regression results now render the
**Pearson r / train R² / R² gap (train−test)** line (tolerant of
older persisted reports); the Results card's third button row carries
**Correlations…**, **Correlation heatmap** and **Decode ECFP bit…**
(the defaults derived from the last importance run and the loaded
dataset); the leaderboard card gained **Cross-model importance…**
(the trained model panel through the new runner + heatmap); and the
feature factory card gained the **Filter low variance…** row with a
threshold entry, the kept/dropped ledger, and frame adoption through
the EC50 → pEC50 prep card's pattern.

named `pharma-ml-toolkit-v<version>.zip`.

## [1.2.89] — 2026-10-26

### Added — the sixth engineering review (100x-151): the D2.py prediction intervals, ported honestly

The instruction: *"Review the current codebase and implementation,
troubleshoot potential issues, and continue improving the engineering
details."* The round audited the round-33 additions line by line (the
least-reviewed code in the tree — 3 real defects found, 17 suspected
findings disproven and recorded in `scratch/audit_151.txt`), fixed all
three, and shipped the one upload capability that had been the standing
open candidate since round 30:

* **`quantile_prediction_intervals`** (`pharmaml.ml`) — the
  `D2.py` uncertainty block, ported honestly. The upload's
  `generate_uncertainty_quantification` trained THREE plain
  `RandomForestRegressor`s on the same data with no quantile loss, so
  the three forests were bit-for-bit IDENTICAL: interval width 0.0 on
  every row, empirical coverage 0.0%, and the "90% Prediction
  Interval" its plot legend advertised did not exist
  (`RandomForestRegressor` has no quantile mode in scikit-learn at
  all — probed). This round fits what the upload *meant*:
  `GradientBoostingRegressor(loss="quantile")` at the `α/2` / `0.5` /
  `1−α/2` quantiles, returning a typed `QuantileIntervals` record
  (bounds, median fit, empirical coverage against the held-out truth,
  mean width, and quantile crossings COUNTED and repaired by swapping
  — never hidden). Deterministic by default (`subsample=1.0` carries
  no stochastic element; a stochastic configuration is seeded through
  `random_state` — the seed-honesty pinned by test). The
  ensemble-std sibling `bootstrap_uncertainty` answers the other
  question (model wobble, not residual noise) — cross-referenced in
  both docstrings.
* **`plot_prediction_intervals`** (`pharmaml.viz`) — the GP ±σ plot's
  twin for bracket-shaped uncertainty: parity scatter of the median
  fit, the band drawn through the test rows sorted by actual value
  (the upload's deliberate calibration-style choice, kept), the
  coverage + mean-width box, and a crossed-bounds guard (raw crossed
  input reaching the plot is a caller bug — the record repairs
  crossings).
* **The GUI twin** — the ML tab Results card's **Prediction
  intervals** button (regression-only gate, α entry defaulting to
  0.1, the honest coverage/width/crossing note, the figure replacing
  the parity plot until the next train) plus **Save intervals…** (the
  five-format table export, retiring on the next train — the
  permutation card's 100x-146 twin). Live smoke: 52 held-out rows,
  84.6% coverage, mean width 6.26, 0 crossings.

### Fixed — the round-33 audit's three findings

* **The board's "Save table…" twin promised four fillers; only three
  fed it** (RISK). The hyperparameter search's render filled the
  board TREE + status but never assigned `_board_frame` — right after
  a search, clicking "Save table…" either lied ("Nothing to save…
  or a hyperparameter search first" with the trial ranking plainly
  visible) or exported the PREVIOUS run's leaderboard as "the numbers
  behind the bars" (a cross-KIND stale export). The render now builds
  the trials frame (full configuration text, float scores, ranked
  1..n, failed trials last) — the fourth filler the button, its
  tooltip, its guard message and the 1.2.88 CHANGELOG bullet all
  promised. Pinned behaviorally (the frame, the stale-leaderboard
  replacement) and at the source level for all four fillers.
* **`pic50_to_ic50`'s overflow fix missed the lex side** — `float()`
  on a Python bigint (`10**400`, reachable through an object-dtype
  column) raises `OverflowError` that escaped the
  `(TypeError, ValueError)` net: the same absurd-cell class the
  round fixed on the pow side. `OverflowError` joined the net (the
  value is unparseable-as-float → `None`).
* **`out_col` label validation stopped one step short** —
  `out_col=None`/`""`/whitespace passed all three siblings' collision
  guards and pandas appended a `None`-named column (no clobber, but a
  label that breaks CSV round-trips and lookups). All three
  (`add_ic50_column`, `add_pic50_column`, `prepare_pic50_dataset`)
  now refuse blank labels with the same typed error.
* **By-catch (latent round-33 ordering defect)**: the notebooks'
  committed outputs printed 1.2.87 against a 1.2.88 tree — the
  re-execution ritual had run *before* the version bump. The ritual
  now ran after the 1.2.89 bump; all 119 variant tests green.

### Verified

* 3018 collected = **3012 passed + 6 skipped / 0 failures** (651 s,
  junit151.xml — the 8 first-run failures were all the stale-notebook
  version tests, fixed by the ritual above); +47 tests over 1.2.88
  (18 quantile intervals, 8 interval plots, 10 converter guards, 3
  board-filler regressions, 8 GUI). ruff clean (`.`); mypy clean (66
  files); applicability doctests green; `--self-test` green on Xvfb;
  leakage guard green (the audit's scratch notes leak-scanned clean).

## [1.2.88] — 2026-10-26

### Fixed — the fifth engineering review (100x-150): three audit findings, the board's Save-table twin

The instruction: *"Review the current codebase and implementation,
troubleshoot potential issues, and continue improving the engineering
details."* A line-by-line static audit of the round-31 additions (the
code with the least review mileage — that round's subagent worklog
entries were lost to context deadlines) found three real defects; all
three are fixed and test-pinned, plus the round's one GUI feature:

* **`benchmark_feature_families` promised a raise it could never
  deliver** — the docstring's "fewer aligned rows than `cv` folds"
  `ModelError` lived *inside* the per-family `try:` whose blanket
  `except Exception` swallowed it into a NaN board row, so a 2-row
  dataset with `cv=5` degraded into a 100%-NaN leaderboard instead of
  the actionable error (the exact 27-A degradation `compare_models`
  guards against up front). The guard is now hoisted above the loop
  and fires before any featurisation work is spent; a family that
  drops rows below `cv` only *after* featurisation (bad SMILES)
  honestly keeps its NaN row.
* **`pic50_to_ic50` crashed with a raw `OverflowError` on a finite
  extreme pIC50** — `10 ** (exponent − pIC50)` overflows for pIC50
  ≲ −299 (nM), and one absurd cell in a prediction sheet killed the
  whole `add_ic50_column` job with a raw traceback out of a
  `float | None | NaN` contract. The fix returns the mathematical
  limit `inf` (probed: `10**(9−(−299.26))` raises; `−299.25` is
  finite), mirroring the underflow end where `10**(9−400)` silently
  yields `0.0` — and `inf` correctly *fails* the GUI potency triage
  (an impossibly weak compound is never shortlisted as a hit).
* **`add_ic50_column` / `add_pic50_column` silently clobbered on
  out_col collision** — `out_col="pIC50"` replaced the source column,
  `out_col="SMILES"` destroyed the identifier column, no warning; the
  round-146 guard fixed exactly this in `prepare_pic50_dataset` but
  stopped one function short of the two table twins. Both now refuse
  with `DataValidationError` ("appends, never overwrites"), and the
  GUI's two call sites turn a re-click into an explicit recompute
  (the stale column is dropped first) instead of an error dialog.

### Added

* **The shared leaderboard's "Save table…" button** — the board that
  `compare_models`, `benchmark_feature_families`, the OOF stacking
  card and hyperparameter searches all fill finally has the export
  twin the permutation card got in 100x-146: the numbers behind the
  bars through the five-format dialog (CSV/TSV/Markdown/JSON/styled
  HTML), ready for a report without a re-run.
* **The environment-repair round's by-catch**: the venv's 11th reset
  (rdkit/torch/mordred gone) was repaired in the foreground (the
  sandbox reaps background pip installs mid-download), and the file
  modes the reset corrupted (every tracked file turned executable,
  115 `EXE002` ruff errors) were restored from the git index — the
  `ruff check .` green convention holds again.

### Verified

* 2971 collected = **2965 passed + 6 skipped / 0 failures** (627 s,
  junit150.xml, home-paths scrubbed); +17 tests over 1.2.87 (9
  converter guards/overflow, 2 benchmark guards, 6 GUI pins). ruff
  clean (`.` — modes restored); mypy clean (66 files); doctests green
  (converters); notebooks re-executed (203 s) + 119 variant tests;
  `--self-test` green on Xvfb; the leakage guard caught and scrubbed
  the audit's own scratch notes before the archive shipped.

## [1.2.87] — 2026-10-26

### Added — the third uploaded notebook (100x-149): `look.ipynb`'s OOF meta-stacking shipped

The instruction: *"Check the contents of this files one by one and
update the playground.py file."* The third freshly uploaded notebook —
**`look.ipynb`** (a single 13 KB cell: the pIC50 stacking regressor —
IC50 cleaning, descriptors + three fingerprints, univariate-selection
union, seven tuned base models + three boosted, out-of-fold
meta-stacking with four tuned meta-learners, Ridge residual
correction) — was read line by line. The cell-by-cell gap analysis
found the descriptor/fingerprint/model vocabulary already covered by
1.2.86's twenty-family factory; this round ships the notebook's
genuinely NEW machinery — the parts no uploaded notebook had asked
for before:

* **`stack_oof_regressors`** (`pharmaml.ml.stacking`) — the notebook's
  steps 10–16 as one deterministic call: KFold(shuffled, seeded)
  out-of-fold predictions from registry base models build the honest
  meta-feature matrix (no blender ever scores a row it was trained
  on — the notebook's own leakage fix, kept), the bases refit on the
  full train set for the test meta-features, meta-learners compete on
  a `test_r2`-ranked leaderboard (a failing blender keeps a NaN row
  and sinks — the `compare_models` convention), and the winner gets a
  `Ridge` residual-corrector fitted on its OOF residuals; the refined
  prediction is blender + corrector. `OOFStackingResult` carries the
  OOF frame, the leaderboard, both scores and the best blender's test
  predictions for the parity scatter.
* **`target_transform="quantile"`** — the notebook's
  `QuantileTransformer(output_distribution='normal')` threaded through
  the TARGET, fitted on the training target only. HONEST SCALE NOTE
  (pinned in the docstring and the GUI note): every score is then on
  the TRANSFORMED scale — the notebook's own convention (it never
  inverse-transforms); raw-scale R² comparisons would be apples to
  oranges.
* **`select_kbest_union`** (`pharmaml.ml.preprocessing`) — the
  notebook's step-6 selector pair: one `SelectKBest` per scoring
  function (`f_regression`, `mutual_info_regression` — the MI side
  pinned at seed 42), each keeping its own top-k, the two kept-sets
  UNIONed in original column order. Both selectors fit on the train
  frame only and merely subset the test frame — the notebook's
  "selection on training data only" fix as the contract.

**Honest review notes — the notebook's own bugs, FIXED not ported**
(documented in the module docstring): the uploaded cell cannot run —
three syntax slips (`odel.predict` missing its `m`,
`tuned_meta_models].predict` broken, `in(...)` where `min(...)`
belongs) — and its descriptor block double-counts MolWt / LogP / TPSA
/ HBA / HBD / ROTB, which are already inside
`Descriptors.descList` (six duplicate column names). The toolkit port
drops the duplicates and parses.

**The playground GUI (the card the notebook asked for):**

* The ML tab gained **"Stacking leaderboard (OOF + residual)"**:
  base-model and meta-learner entries (comma-separated registry
  names), OOF folds (2–10), test holdout percent (10–50), a
  "Quantile target" checkbutton (the transform note in the board
  status line), a "Residual correction" checkbutton (ticked by
  default), a Run button that fills the shared leaderboard board with
  the meta rows plus the `+residual` row, and the card's own figure
  host carrying the notebook's step-18 parity scatter (best blender's
  test predictions, the y=x diagonal). Regression only — the notebook
  predicts pIC50, not a class.

### Verified

* New tests: `TestStackOofRegressors` (8: end-to-end synthetic,
  determinism, quantile path, residual off, defaults, alias collapse,
  failing-meta NaN row, all-fail guard), `TestStackingGuards` (9:
  empty/unknown bases, unknown meta, splits below 2 / beyond rows,
  bad transform, NaN target, row/column mismatches),
  `TestOnStepCallback` (2: payload coverage, raising callback
  swallowed), `TestSelectKbestUnion` (10: union size bounds,
  determinism, column parity, leakage contract, guards) and
  `TestReview149Gui` (5: source pins, widgets, session persistence,
  no-dataset and regression-only guards); module doctests green;
  ruff clean; mypy clean on the new files.
* GUI end-to-end under Xvfb: the card's round-trip screenshot
  (`download/pg149-ml-stack.png`) — the board fills with the
  meta-learner rows and the parity scatter draws into the card host.

## [1.2.86] — 2026-10-26

### Added — the one-by-one notebook review round (100x-148): both uploaded notebooks' missing halves shipped

The instruction this round: *"Check the contents of this files one by
one and update the playground.py file."* Both freshly uploaded
notebooks were read cell by cell — **`xxx.ipynb`** (35 cells: the
3-class molecular classification pipeline — featurization, filtering,
selection, 13 GridSearch classifiers, ensembles, evaluation) and
**`fak_ml.ipynb`** (183 cells: the BindingDB SYK pIC50 regression —
descriptor + 16-fingerprint benchmarks, hypertuning, external drug
prediction, hit triage). The cell-by-cell gap analysis against the
toolkit found ~70% coverage; this round ships the missing 30%.

**The feature factory grew from 9 to TWENTY families:**

* The ten `fak_ml.ipynb` fingerprint families (cell 40's datamol
  benchmark vocabulary), each verified **byte-exact against
  `dm.to_fp`** on real molecules: `maccs` (167 structure-key bits),
  `fcfp` / `fcfp_count` (feature-invariant Morgan, 2048), `topological`
  / `topological_count` (the 4-atom TORSION generator with count
  simulation — datamol's own `fp.py` maps 'topological' there, NOT to
  the path fingerprint a first draft confused it with), `rdkit_count`
  (the path generator at `numBitsPerFeature=1`, datamol's count
  default — a naive generator double-counts: aspirin 602 vs 301),
  `pattern` (2048 substructure-screening bits), `layered` (2048),
  `estate` (the 79 E-State atom-type COUNTS) and `erg` (the 315-float
  reduced-graph fingerprint, ±inf → NaN). Honest build note pinned in
  the tests: datamol's own `fp_to_array` crashes on ethanol's empty
  torsion-count sparse vector — OUR zero vector is the right answer.
* **`shape_3d`** — `xxx.ipynb` cell 6's 3D block as a family:
  `compute_3d_descriptors` (ETKDGv3 conformers at the fixed seed 42,
  MMFF94s relaxation with the notebook's UFF fallback, also triggered
  by MMFF typing failures, lowest-energy conformer selection) feeding
  the nine rdMolDescriptors shape descriptors (Asphericity,
  Eccentricity, PMI1-3, RadiusOfGyration, SpherocityIndex, NPR1/2). A
  failed 3D pipeline keeps the ROW as NaN — the notebook's
  failed-molecule behaviour.
* The full matrix is now ~27,700 columns (was ~12,860); every prefix
  stays unique (the exact-inverse `family_of_column` contract), and
  the GUI picker grew to twenty ticked-by-default checkboxes with the
  honest widths.

**The library ports:**

* **`benchmark_feature_families`** — `fak_ml.ipynb` cells 40-47's
  `test_fp` loop (one LGBM per datamol fingerprint) as a
  `compare_models`-convention leaderboard over the 18 2D families
  (the two 3D families stay opt-in — conformer embedding would
  dominate the run): per-family CV score/std, fit time, feature and
  sample counts, progress callback, NaN rows sink honestly.
* **`pic50_to_ic50` / `add_ic50_column`** — the notebook's
  `dm.molar.log_to_molar` inverse, with exponent arithmetic so integer
  pIC50s land on exact decimal powers (datamol's own product returns
  999.9999999999999 for pIC50 6; this returns 1000.0 — probed).
* **`plot_pr_curves`** — `xxx.ipynb` cell 28's OvR Precision-Recall
  curves (previously missing from the toolkit entirely), with each
  class's average precision in the legend AND the honest no-skill
  baseline at the class prevalence (a line the notebook's own plot
  omitted).
* **`ClassificationReport.per_class`** — the notebook's
  `precision_recall_fscore_support(average=None)` table as a
  `str(label) → {precision, recall, f1, support}` dict (backward
  compatible: direct constructions keep `None`).
* **`sgd_classifier`** — the notebook's 13th classifier in the
  registry (hinge / log_loss / modified_huber grid; the notebook's
  `early_stopping=True` deliberately NOT the factory default — it
  reserves a validation split that raises on screens smaller than
  2 × n_classes rows).

**The playground GUI (every notebook flow now has a button):**

* The Results card grew live **"ROC curves"** and **"PR curve"**
  buttons (the ROC figure previously rode only the saved report
  bundle; the PR curves existed nowhere).
* The classification metrics render the **per-class line** (P · R ·
  F1 · n per label) under the macro aggregates.
* The Predict card gained **"pIC50 → IC50 nM"** (auto-detects the
  prediction column) and the **potency-threshold triage** (≤ 100 / 300
  / 500 / 1000 nM — the notebook's `query('IC50 < 300')` shortlist;
  the tree shows the hits, Save CSV still writes the full table).
* The Leaderboard gained **"Benchmark feature families"** (model +
  folds entries, the board fills with the per-family ranking).
* The model picker lists `sgd_classifier` automatically; the module
  docstring and the factory subtitle tell the twenty-family story.

### Fixed

* The first-draft `topological` family used the path generator
  (byte-identical to `rdkit_fingerprint` — a silent duplicate);
  corrected to the torsion generator with datamol's exact defaults
  after reading datamol's own `fp.py` mapping table (the parity test
  now pins all ten families against `dm.to_fp`).
* `test_features_docstring_count_honest` and the width-label pins
  updated for the vocabulary growth (9 → 20 families, ~12,860 →
  ~27,700 columns).

### Verified

* `pytest` (bare, testpaths): full suite green — the new tests:
  `TestFakmlFamilies` (10: datamol parity, prefix uniqueness,
  shape_3d determinism/NaN/degenerate, ethanol torsion zero-vector),
  `TestPic50ToIc50` (13) + `TestAddIc50Column` (6),
  `TestPrCurves` (5), `TestPerClassMetrics` (5), `TestSgdClassifier`
  (6), `TestBenchmarkFeatureFamilies` (11), `TestReview148Gui` (15),
  plus the updated contract pins; ruff clean, mypy clean (zero
  errors, was 65-baseline-clean before and after), doctests green on
  features/converters/descriptors/training.
* GUI end-to-end under Xvfb: the twenty-family factory run, the
  per-class line after a multiclass train, the PR button drawing into
  the Results host, the IC50 conversion + triage round-trip
  (1/100/1000/10000 nM), and the 18-family benchmark board.

## [1.2.85] — 2026-10-26

### Added — the fourth engineering-details round (100x-146): the indexed progressive reader, the chunk-boundary guard, the loader phase notes, and the permutation Save-CSV twin

The instruction this round: *"Review the current codebase and
implementation, troubleshoot potential issues, and continue improving
the engineering details."* The fresh one-by-one audit (the round-145
additions end to end plus the never-audited module tail) harvested one
real parity break, one stale-state risk and four polish items — every
finding probe-verified, ten candidate concerns disproven with reasons
(the standing ledger).

**The chunk-boundary parity guard (the audit's headline finding):**

* pandas' own low-memory chunked C parser silently TRUNCATES a row
  that carries extra fields and lands exactly on a chunk boundary —
  the plain reader raises `ParserError` for the same bytes, so
  `read_table_with_progress` could hand back different DATA than
  `read_table` (the one hole in the round-144 parity promise, caught
  by the audit's boundary-aligned probe). **`_guard_ragged_boundaries`**
  scans the physical lines once (bytes only, no parsing) and rejects
  such files up front with the sync reader's own verdict; it bows out
  for quote-bearing and interior-blank-line files (where lines cannot
  be aligned to records — pandas' own mid-chunk raises still cover
  those) and needs no rule for SHORT rows (both parsers pad them with
  `NaN`). The guard re-runs for the semicolon retry in the separator
  ladder, and its boundary arithmetic holds for `chunk_rows=1`.
* **`read_indexed_table_with_progress`** is the new public indexed
  twin: the expression-matrix loader's plain
  `pd.read_csv(path, index_col=...)` pair — preferred label first,
  positional fallback on the missing-label `ValueError` — with the
  determinate rows/ETA bar of the round-144 reader. The index is
  preserved through the chunked concat, the label fallback fires only
  while no chunk has landed (a LATER `ValueError` propagates for its
  honest verdict instead of silently re-reading), no separator
  sniffing happens (comma on both paths — a semicolon table is one
  mangled column below the threshold AND above it), and ragged
  boundary rows ride the same guard.
* **The Cell Lines expression-matrix loader** — the last marquee in
  the loader story, standing "by design" since round 28 — now streams
  through the new reader: a wide matrix carries a determinate bar
  instead of the open-ended marquee, byte-for-byte the plain pair on
  small files.
* **The Cell Lines screen loader's AUDIT/COLLAPSE phases now speak:**
  after the streaming read completes, the status bar carries
  "auditing rows" and (when the box is ticked) "collapsing replicates"
  counter-style notes — a slow in-memory pass on a huge screen is
  never a silently frozen bar.

**The round's smaller features and fixes:**

* **The permutation-importance Save-CSV twin:** the Results card's
  "Save CSV…" writes the last run's table (feature,
  importance_mean, importance_std — the ±1 std spread the whiskers
  drew) while "Save figure…" keeps the picture; the table retires
  when a new model trains (the old table names the old model's
  features).
* **Loading a new dataset retires the previous one's derived
  tables** (the audit's stale-state finding): the Clean card's and
  the prep card's Save buttons used to write the PREVIOUS dataset's
  table while claiming to describe the new one —
  `_ml_adopt_frame` clears both frames now.
* **`prepare_pic50_dataset` refuses a colliding `out_col`** (the
  audit's duplicate-key finding): `out_col="SMILES"` used to silently
  clobber the SMILES column in the result dict — the three reserved
  names are rejected up front with a typed error.
* The module docstring's stale pre-ecfp3 family count ("8-family,
  10,000+-column") is now honest ("9-family, ~12,860-column"), as is
  `selection_metadata`'s ("the seven fingerprint families"); the
  Save-figure docstring and empty hint name the permutation figure;
  the prep card gates pandas like its Clean-card sibling.

**Verification:** suite **2809 collected = 2804 passed + 5 skipped /
0 failures** (+37: `TestGuardRaggedBoundaries` ×8,
`TestReadIndexedTableWithProgress` ×9, the out_col collision ×3,
`TestReview146Gui` ×17 — with the round-143 collapse-capture pin
updated where the round deliberately changed its line); ruff clean;
mypy `src/` clean (65 files); `--self-test` green on Xvfb; doctests
green on all three touched modules (the io module gained the guard
line and the full indexed-reader section); docs/INDEX regenerated
(286 symbols); notebooks re-executed at 1.2.85 (208 s, twins
regenerated, 119 variant tests green).

## [1.2.84] — 2026-10-26

### Added — the "کد علی" upload audit (100x-145): D2.py's data prep, its permutation importance, and the radius-3 ECFP family

The instruction this round: *"Check the contents of this file one by
one and update the playground.py file"* over the freshly uploaded
`کد علی.zip` (8 files). The one-by-one verdict: the docking quartet
was already covered — `pdbqtgeneration.py` is byte-identical to the
audited `niggaqt.py` (the MGLTools PDB → PDBQT batch card),
`pdbfromsmile.py` is the stricter older twin of `nigga.py` (the
LIGAND STRUCTURES card implements the more tolerant column handling),
`mismatchcheck.py` rides the grid/torsion/completion audit card (its
`check_torsion_consistency` even auto-detects the MGLTools-vs-Meeko
convention), and `niggavinamulti.py` is the thread-pool Vina batch
with resume the config-batch card already runs — leaving the `Ml/`
pair as the round's genuine work.

**The EC50 → pEC50 dataset prep (`prepare_pic50_dataset`):**

* **`pharmaml.chem.prepare_pic50_dataset`** upgrades `Ml/D2.py`'s
  `process_data`: the original hard-coded `Ligand SMILES` /
  `EC50 (nM)` columns, silently stripped `<`/`>` qualifiers, a
  silent `0 < EC50 ≤ 1e9` range filter and a silent per-SMILES median
  collapse — here every step is a named, counted, reversible choice.
  The columns auto-detect from the upload's own spellings (or ride
  explicit overrides), censored magnitudes are kept AND counted, the
  duplicate rule is a parameter (`median` — D2.py's own — `mean`,
  `first`, `keep`), and the conversion rides `ic50_to_pic50` so any
  unit works. The typed `EndpointDatasetReport` carries the full
  accounting identity (every input row lands in exactly one bucket).
* **The Data & ML Dataset card gained the "EC50 → pEC50 prep" row**
  with the duplicate-rule combo: the audit lands in the status line,
  one click adopts the prepared set as the active dataset with pEC50
  as the target column, and "Save prep CSV…" writes the table
  (SMILES, n_measurements, endpoint_median, pEC50).

**The permutation importance (D2.py's interpretability step):**

* **`pharmaml.ml.permutation_feature_importance`** wraps
  `sklearn.inspection.permutation_importance` with the toolkit's
  validation conventions and returns a typed `PermutationImportance`
  record (names, means, stds, scoring, repeats, n) with
  `as_frame()` / `ranking()` helpers — the spread the original
  threw away. Works with bare estimators AND pipelines (the
  preprocessing reruns on every shuffle), defaults D2.py's own
  choices (`r2`/`accuracy` scoring, 5 repeats, seed 42), and pins
  `n_jobs=1` (the nested-parallelism trap the original hit).
* **`pharmaml.viz.plot_permutation_importance`** draws the top-N
  signed bars with ±1 std whiskers — negative means (a shuffle that
  HELPED — a noise feature) render in red so the eye cannot miss
  them. Accepts the record or its `as_frame()` twin.
* **The Results card gained the "Permutation importance" button**
  with a repeats knob: it runs on the last trained model's held-out
  rows, replaces the parity figure, and rides the card's existing
  Save-figure export.

**The ecfp3 family (the upload's R3 fingerprint half):**

* **`pharmaml.chem.features` gained the `ecfp3` family** — radius-3
  Morgan at the toolkit's 2048-bit width (`ECFP3_BITS`,
  `ECFP3_RADIUS`), `Ml/D2.py`'s own R3 fingerprint: the wider atom
  environments radius 2 cannot see, as a first-class family in the
  canonical order (prefix `ecfp3_bit_`). The honest divergence note:
  D2.py's R2 half ran 4096 bits where the toolkit's `ecfp` runs
  2048 — combining `ecfp + ecfp3` covers the dual-radius story.
* **The feature factory gained the checkbox** ("ECFP3 bits — radius 3
  (2048)") — ticked by default like every family, riding the
  width-preview and the session's families map for free.

### Fixed

* `read_table_with_progress`'s docstring now shows the
  ragged-semicolon parity verdict the tests pinned (the sniffed `;`
  separator choking on a ragged body delegates to the sync reader's
  own mangled-frame verdict — round 29's open candidate c).
* The working tree's stale executable bits (an artifact of the
  original zip extraction) are cleared — `ruff`'s EXE002 noise over
  66 files is gone and the tree matches git's recorded 100644 modes.

### Tests

* +37: `TestPreparePic50Dataset` (12 — the D2-spelling end to end
  with the accounting identity, the four aggregate modes, explicit
  columns/units, the IC50 fallback, the inclusive cap, every error
  path, the report's `__str__`/`as_dict`, no-mutation),
  `TestPermutationImportance` (7 — ranking, DataFrame names, the
  pipeline path, classification scoring, the record helpers, the six
  validation errors, seed determinism), the five viz tests (record,
  frame, red-negative, the four contract errors, end-to-end with a
  fitted model), `TestEcfp3Family` (4 — radius-3 sees more
  environments than radius-2 on aspirin, prefixes/round-trip,
  default-run inclusion, canonical order), and `TestReview145Gui`
  (15 — the docstring pins, the prep card's source/combobox/needs-
  dataset/end-to-end-with-adoption/declined-adoption/save-guard
  contract, the permutation button's source/needs-run/after-training/
  fractional-and-zero-repeats contract, and the ecfp3 factory pins).
* Standing contract updates: the four family-list pins in
  `test_features.py` carry `ecfp3` (canonical order, prefixes, bit
  constants, the full-run family counts at 12,800 + RISM +
  descriptors).
* docs/INDEX.md regenerated (278 → 285 symbols).

## [1.2.83] — 2026-10-26

### Added — the round-29 engineering review (100x-144): multi-file picks ride the session, large CSVs load with a determinate progress bar, and the audit's parity/normaliser fixes

The instruction this round: *"Review the current codebase and
implementation, troubleshoot potential issues, and continue improving
the engineering details."* A read-only audit covered the regions no
round had audited line-by-line (the module header, the round-28 loader
twins themselves, and this round's fresh additions); its findings are
fixed below, alongside the three open candidates round 28's own
handover listed.

**The multi-file session picks (the handover's item a):**

* **The three Utilities list pickers persist across restarts.** The
  merge & split tables, the approval exports and the distribute-files
  list (`askopenfilenames` selections) now ride the session file as a
  new always-present `paths_multi` key with the single paths' own
  discipline: every file must still exist on this machine (a vanished
  file drops out, the survivors keep their picked order), a live pick
  is never overwritten, and only allowlisted attributes ever land.
  `dump_session` refuses to char-split a stray non-list value (an
  entry drops instead), `load_session` sanitises foreign payloads to
  shorter honest lists, and the approval card's inline label
  re-renders after a restore in its picker's exact "first three +
  … (+N)" format (the merge/tools labels ride their existing
  mode-aware helpers).

**The determinate loader bar (the handover's item c):**

* **`read_table_with_progress`** (new public reader in
  `pharmaml.utils.io`): CSV-like files at/above 8 MiB stream in
  `chunk_rows` chunks with `(rows_done, rows_estimated, "")` pushed
  per chunk; the estimate comes from the file's own bytes-per-line
  average (sampled from the first 64 KiB) and degrades to an honest
  `None` for a giant first line. Small files and Excel inputs pass
  straight through `read_table` — no reports, no behavioural
  difference. Column hygiene (`_finalise_columns`) is ONE shared rule
  for both readers, and ragged edge cases delegate to the sync
  reader's own verdict so the two can never disagree (the audit's §3
  parity break: a sniffed `;` separator choking on a ragged body now
  returns the same mangled frame `read_table` returns, and a truly
  malformed file raises the same clean `DataValidationError`).
* **The four read_table loaders report.** Data & ML, Deep Learning,
  the feature factory and the Cell Lines screen
  (`load_response_table(..., report=...)`, a new optional callback a
  DataFrame caller never fires) all stream their reads — a large CSV
  now carries a determinate bar with "rows done / estimated · ETA"
  instead of the open-ended marquee (proven live on a 22 MB / 1.5M-row
  load: `loading wide_library.csv — 900000/1499898 · ~1 s left`).
  The expression-matrix loader keeps the marquee by design (its
  indexed read has its own fallback semantics).
* **The chained-job guard (the audit's §2):** `_poll_jobs` clears
  `_on_done` only when the app is idle — a render callback that chains
  straight into a follow-up job keeps its callback instead of having
  it silently wiped (round 27's fix was one line short of its own
  promise; no current handler chains, but the guard is now pinned).

**Engineering details:**

* The Apply-best relay's frame/path invariant is stated where it lives
  (the handover's item b): `_ml_adopt_frame` sets frame and path
  TOGETHER and nothing ever clears the frame alone, so the relay's
  `elif _ml_path` branch is a safety net for programmatic plants, not
  a state a normal load can reach.
* A header-only CSV's single empty chunk no longer emits a spurious
  0-rows progress blip; the shortcut-hints comment confesses its seven
  entries; `load_response_table`'s docstring names the streaming
  reader; a stale test pin (the Cell Lines collapse capture) follows
  the loader's new one-argument `work` signature.

**Verification:** suite **2726 collected = 2721 passed + 5 skipped /
0 failures** (+27: `TestReadTableWithProgress` ×11, the biology
report-forwarding pair, `TestReview144Gui` ×14); ruff clean; mypy
`src/` clean (65 files); `--self-test` green on Xvfb; doctests green
on both touched modules; docs/INDEX regenerated (278 symbols);
notebooks re-executed at 1.2.83; VLM review PASS on all three desktop
shots (the mid-job determinate bar, the restored merge list, the
restored approval list).

## [1.2.82] — 2026-10-26

### Fixed — the round-28 engineering review (100x-143): every file-dialog CSV loader off the main thread, the Utilities restored-path labels, and three audit findings

The instruction this round: *"Review the current codebase and
implementation, troubleshoot potential issues, and continue improving
the engineering details."* A fresh one-by-one audit covered the regions
round 27 never reached line-by-line (Home, Chemistry, PK/PD, Cell
Lines, the Generative tail and Utilities — every candidate verified by
grep or an Xvfb probe before reporting); its findings plus the loader
port the previous round itself flagged are all fixed.

**The loader port (the audit's §4, every tab):**

* **Five file-dialog loaders now ride the worker pool.** Data & ML,
  Deep Learning, the feature factory, the Cell Lines screen (load +
  audit + replicate collapse) and its expression matrix ALL ran
  `read_table` / `pd.read_csv` on the main thread — a
  multi-hundred-megabyte CSV froze the whole window while every other
  long operation already showed a progress bar. Each tab gained a
  `_load_file` job twin (`_start_job("loading {name}")`, the Docking
  tab's own proven pattern); the bundled examples stay synchronous
  (shipped and tiny), and each tab's sync core and job render share
  ONE adoption body (`_ml_adopt_frame`, `_dl_load_frame`,
  `_feat_adopt_frame`, `_bio_adopt_screen`, `_bio_adopt_expr_frame`)
  so the two paths can never drift apart. The Cell Lines collapse
  checkbox is captured on the MAIN thread (a Tk variable is never read
  from a worker), and the busy guard inside `_start_job` is the
  Ctrl+O guard's twin — a load can never swap a frame under a running
  trainer.

**The new audit's findings, fixed:**

* **The Utilities restored-path labels (the round's one real bug).**
  Seven picked paths ride the session across merge & split / table
  compare / campaign file tools / reindex, but
  `_refresh_restored_path_labels()` walked only the Docking cards and
  the converter pair — the four Utilities labels kept their launch
  placeholders after a restore while the run silently used the
  restored paths. The refresh now runs the three cards' own
  mode-aware helpers (`_util_merge_mode_changed`,
  `_util_cmp_refresh_info`, `_util_tool_mode_changed`) and writes the
  reindex picker's full-path text — round 27's "every restored path
  re-renders its picker's info label" promise, finally kept.
* **"SDF → table" demanded an output file it never writes.** The
  converter's run guard fired for every non-pIC50 mode, but the SDF
  branch returns an in-memory table (tree + "Save table CSV…") and
  ignores the output path entirely — the pIC50 mode's own exemption,
  extended to its in-memory twin, plus the same in-memory notice when
  picking an output for either mode.
* **The MACCS mixed-parse label clobbered itself.** With a list where
  some molecules parsed and some failed, the dropped-molecules message
  overwrote the computed summary AND lied ("no MACCS keys" is false
  when molecules parsed). The summary now wins with the dropped count
  riding alongside; the full clobber is reserved for the
  nothing-parsed case where it is actually true.
* **Two advisory missing-input dialogs used `showerror`** where the
  app convention is `showwarning` (the descriptor card's own "Empty
  SMILES" pattern): the surgery card's "Type a SMILES in the molecule
  row first" and the rescore card's "name at least one evidence
  column" are now warnings — parse errors keep their `showerror`.

**The notebook tour (the round's documentation candidate):**
`complete_docking_workflow.ipynb` §12 gained the census + workers>1
cell — `vina_batch_progress` demonstrated on a three-ligand tree (one
docked, one crashed with a zero-byte log, one untouched → "1/3 already
docked · 2 to go"), and `run_vina_batch(..., max_parallel=2,
skip_existing=True)` showing the honest engine gate; the recap table
gained the Resume census row. The web finder's localStorage memory was
audited against the desktop's round-27 "tab:title" discipline and
verified already stronger (keys on unique catalog ids, tab label
always shown) — no change needed.

### Verification

* 2699 collected = 2694 passed + 5 skipped / 0 failures (bare pytest,
  dev server stopped — the OOM lesson; +21 tests this round:
  TestReview143Gui covering all five job loaders end-to-end, the
  empty-CSV warning via the render path, the busy guard, the
  synchronous-example design pin, the main-thread collapse capture,
  the Utilities label re-render, the SDF guard exemption + notice +
  still-firing twin, both MACCS label verdicts, and the two
  showwarning convention pins).
* Doctests green; ruff clean; mypy `src/` clean (65 files);
  playground.py at the 61-error pre-existing baseline; `--self-test`
  green on Xvfb; the notebook re-executed with nbclient and the
  `_no_output` twin regenerated (4 source pins + the census output +
  the home-path leak gate).

## [1.2.81] — 2026-10-26

### Added — the round-27 engineering review (100x-142): a one-by-one audit of every tab, nine latent bugs fixed, and three workflow features

The instruction this round: *"Review the current codebase and
implementation, troubleshoot potential issues, and continue improving
the engineering details."* A full one-by-one audit of playground.py's
tabs (module helpers, chrome, session machinery, Data & ML, Deep
Learning, Generative — the Docking tab reviewed in parallel) found one
real bug, five risks and three polish items; all nine are fixed, plus
three features shipped.

**The audit's findings, fixed:**

* **The text-map drift (the round's one real bug).** Four separate
  maps disagreed about which text boxes ride the session: the restore
  path knew all seven keys, the per-tab reset map carried five, the
  copy-inputs map a *different* five, and the ownership walk only
  four. Net effect: the Chemistry tab's MACCS box and both
  similarity-scan boxes escaped "Reset this tab to defaults" and
  "Copy inputs as JSON" while riding the session file all along. One
  shared `_session_text_widgets()` map now feeds all four consumers —
  no drift possible.
* **The Ctrl+O busy guard + the main-thread frame captures.**
  File ▸ Open CSV stayed live during a running job and swapped
  `self._ml_frame` synchronously while the worker thread was still
  reading it (a narrow but real race: KeyError on the swapped target,
  or silent training against the new frame). The menu accelerator now
  obeys the busy rule, and `_ml_train` / `_ens_train` / `_ml_tune` /
  `_ml_compare` capture the frame on the MAIN thread
  (`_ml_clean_frame`'s own pattern) before the job starts.
* **The `_poll_jobs` ordering.** The poller invoked the render
  callback and only then cleared `self._on_done` — a render that
  chained into `_start_job` would have its own callback silently
  wiped. Capture-and-clear now precedes the invocation.
* **The finder's title-only recency keys.** Five card titles are
  duplicated ACROSS tabs (Dataset ×2, Preview ×2, Results ×3), so a
  "Recently jumped" chip recorded from the Generative *Results* card
  jumped to the Data & ML *Results* card (build order wins). The
  memory now keys on `"tab:title"`; legacy bare titles from older
  session files keep a first-match fallback, and a duplicated title's
  chip appends the tab label so it confesses WHICH card it opens.
* **The fractional-input truncation.** `parse_int` turned a "cv
  folds" of `2.5` into a quiet 2. Fractional input is now rejected
  with the field name in the error; whole-valued float spellings
  (`2.0`, `1e2`) still parse and `inf`/`nan` are honestly refused.
* **The AD k-guard.** The friendly too-few-training-rows warning
  fired only when the bootstrap model count was zero; every other
  short-train case fell through to the library's mid-job
  `ModelError` dialog. The guard now covers all cases.
* **`_ml_tune`'s shadowed worker parameter** (`report` was both the
  progress reporter and the `evaluate_model` binding) — renamed to
  `notify`, the DL tab's own documented convention.
* **`_require`'s unguarded `next()`** — a module name outside the
  dependency table raised `StopIteration` inside the Tk callback
  (surfacing as the generic "Unexpected error" dialog). The guarded
  fallback keeps the gate honest for any future dependency name.
* **`_copy` clobbered a running job's status line** — a copy during a
  job now skips the label flash (the clipboard write, the user's real
  intent, still happens).

**The features:**

* **The picked file/folder PATHS ride the session file.** ~27 pure
  picker attributes (`_SESSION_PATH_ATTRS`) — the Docking tab's
  receptor / ligand / campaign / Vina / vbatch / gen / MGL / audit
  pickers, the conversion card's pair, the Utilities pickers — now
  persist and restore, so a restart hands the folders back instead of
  demanding eleven re-picks. Dataset paths that need an in-memory
  reload (the CSV loaders) stay out on purpose: a path without its
  frame would be a lie the label tells. A path comes back only when
  it still EXISTS on this machine (a mailed session or a moved folder
  degrades to the honest un-picked state), and a live pick is never
  overwritten. `_refresh_restored_path_labels()` re-renders every
  picker's info label after the restore.
* **The generate → MGLTools relay.** The ligand-structures card's
  "Send PDBs → MGLTools" button hands the structure output folder to
  the converter card as its PDB source, pre-fills the PDBQT output
  folder BESIDE it (never inside it, never over a live pick),
  refreshes the card's info label, and jumps there with the finder's
  own flash — the nigga.py → niggaqt.py relay without one re-pick.
* **The pre-flight resume census.** `vina_batch_progress(ligands_dir,
  out_dir) -> VinaBatchProgress(total, docked, remaining, collisions)`
  counts what a resumed Vina batch would skip, with the SAME skip
  rule AND the same collision-disambiguated output names the run
  itself uses (the stem logic lives in one shared
  `_batch_output_names` helper — the census and the run can never
  disagree). The batch card's "Count remaining" button reports
  "N/M already docked · K to go" and confesses name collisions;
  toggling the resume checkbox with both folders picked runs the
  census on its own.

### Fixed

* `_ml_tune_apply` → dnn reloaded the ORIGINAL CSV from `_ml_path`
  into the Deep Learning tab, resurrecting every row the Clean card
  had removed — the winner was then trained on data it was never
  tuned on. The new `_dl_load_frame()` adopts the LIVE (cleaned)
  frame; `_dl_load_path` is now a thin wrapper over it.

### Verification

* **2678 collected = 2673 passed + 5 skipped / 0 failures** (bare
  pytest, dev server stopped — the OOM lesson; +41 tests this round:
  7 for `vina_batch_progress` incl. the run-agreement census, 33 for
  the round's GUI surface incl. the live Chemistry reset/copy
  confessions, the session-path round-trips, the bridge, the census
  button + auto-census trace, and the source-level ordering pins).
* Doctests green on the touched modules (workflow.py gained the
  `VinaBatchProgress` example tour); ruff clean (two RUF100
  auto-fixes); mypy `src/` clean (65 files); playground.py at the
  61-error pre-existing baseline; `--self-test` green on Xvfb.
* docs/INDEX.md regenerated (277 symbols) after the docs-index symbol
  test caught the two new public names — the standing trap.

## [1.2.80] — 2026-10-26

### Added — the round-26 review (100x-141): the "new folder.zip" upload audited file by file — the nigga.py ligand-structure pipeline + the niggaqt.py MGLTools batch converter + the Vina batch's RESUME

The user delivered a new upload — `new folder.zip` (1.0 MB, 22
entries) — with the instruction *"Check the contents of this file one
by one and update the playground.py file."* Every file was checked
one by one against the toolkit's surfaces:

* **ligand generation/** — `nigga.py` (the CSV → canonical tautomer →
  50 ETKDG conformers → MMFF-with-UFF-fallback → lowest-energy
  conformer as PDB + OpenBabel/Gasteiger MOL2 pipeline, parallel
  worker pool, unsupported-atom skip list) had **no dedicated
  surface**: `prepare_ligand_library` covers the CSV → PDBQT campaign
  twin, but the PDB/MOL2 structure pipeline itself did not exist.
  `niggaqt.py` (the MGLTools `prepare_ligand4.py` batch converter
  with its 30-second timeout, numbered sub-folders and conversion
  log) had been *absorbed* as "MGLTools deprecated → Meeko path" —
  the re-delivery says the user wants it. `mismatch2.py` /
  `niggamatch.py` were already covered (`check_torsion_consistency` /
  `scan_torsion_consistency`, the audit card's torsion mode).
* **Vina/** — `vina.conf` + `log.txt` were already covered
  (`parse_vina_config`, the grid box / config batch cards,
  `parse_vina_output`).
* **Vina_contol/** — `adder.py` (receptor distribution),
  `vinar.py`, `vinar_multiple.py`, `vina_reader.py`, `conf.txt` were
  already covered (`distribute_files` / `_dock_distribute_receptor`,
  `run_vina` / `run_vina_batch` / `scan_vina_directory`); `batch.py`
  and `niggavina.py` contributed one genuinely missing behaviour —
  the **resume** (skip already-docked ligands; a zero-byte log marks
  a crashed job).

**The ligand structures card (nigga.py, whole).**
`pharmaml.docking.generate_ligand_structures(library, output_root,
*, num_confs=50, default_b_factor=20.0, write_mol2=True, workers=1)`
runs the original pipeline per library row: RDKit tautomer
canonicalisation (best-effort — a failure keeps the input form and
confesses in the row), the metal-centre skip list
(`UNSUPPORTED_LIGAND_ATOMS` = Fe/Zn/Cu/Mn/Co/Ni/Mo/W, per-row reason,
never batch-fatal), `EmbedMultipleConfs` ETKDG, MMFF minimisation
over every conformer with a UFF fallback when the molecule lacks MMFF
parameters, the **lowest-energy** conformer kept, and two outputs: a
one-model PDB written by the original's own writer (HEADER / COMPND /
ATOM / **CONECT** / END — the connectivity MGLTools'
`prepare_ligand4.py` reader wants; RDKit's own `MolToPDBBlock` drops
it) and, when OpenBabel is installed, the Gasteiger-charged MOL2 twin
(without OpenBabel the PDB still lands and the row records why). The
`workers` knob mirrors the original's `cpu_count() - 1` process pool
through a `ProcessPoolExecutor` — with the **spawn** context, because
the playground runs it inside a worker thread and Python 3.12+ warns
correctly that forking a multi-threaded process can deadlock. The
report is a DataFrame (`ligand`, `smiles`, `tautomer`, `pdb`, `mol2`,
`forcefield`, `energy`, `conformers`, `atoms`, `ok`, `error`), with
invalid-SMILES rows riding along at the end exactly like
`prepare_ligand_library`'s reports. The Docking tab's new **Ligand
structures (CSV → PDB + MOL2)** card drives it over the already-loaded
library with conformers / workers / MOL2 knobs, a live report tree and
the report CSV export.

**The MGLTools PDB → PDBQT batch card (niggaqt.py, whole).**
`pharmaml.docking.convert_pdb_folder_to_pdbqt(source_dir, output_dir,
mgl_python, adt_script, *, timeout_s=30.0, add_hydrogens=True)`
drives `prepare_ligand4.py -l <pdb> -o <pdbqt> -A hydrogens` over
every `.pdb` of the source folder into the numbered-sub-folder tree
(`1/`, `2/`, `3/` … — the `output_pdbqt/<n>/<name>.pdbqt` layout the
campaign readers and the torsion audit expect), each subprocess
guarded by the original's 30-second timeout (the manual
`while proc.poll()` busy-wait became one `subprocess.run(...,
timeout=...)` call — same verdict, no polling), and writes the
original's `conversion_log.txt` (SUCCESSFUL CONVERSIONS / TIMED OUT
FILES / FAILED CONVERSIONS, timestamped) beside the tree. Both the
interpreter and the script paths are checked **before** the first
subprocess fires (a typo'd path raises a typed `DockingError` naming
it, instead of leaving a half-converted tree). The Docking tab's new
**PDB → PDBQT batch (MGLTools)** card drives it with the source /
output pickers, the two MGLTools paths pre-filled with the original's
Windows defaults, the timeout knob, the live report tree and the
report CSV export.

**The Vina batch's RESUME (batch.py + niggavina.py).**
`run_vina_batch(..., skip_existing=False)` gained the upload
controllers' resume behaviour: when ticked, a ligand whose
`<stem>_out.pdbqt` **and** non-empty `<stem>.log` already exist is
not started again (the zero-byte log marks a crashed job —
`niggavina.py`'s empty-log cleanup rule — and is re-docked). Skipped
ligands return `DockingRunResult(skipped=True, ok=True, seconds=0.0)`
so an interrupted campaign re-runs in seconds, docking only the
remainder; the results table's status column shows `skip` and the
summary confesses the count. The Vina config batch card grew the
**skip already-docked (resume)** checkbox wired straight through.

The coverage probe's uploads axis now carries all 13 script/config/
log files of the zip (the `vina.exe` / `vina_split.exe` / license
`.rtf` payloads are external engines, absorbed like the rar's own);
the verdict reads **119 rar + 14 upload files, zero-hit 0**.

### Verified

Bare `python -m pytest`: **2637 collected = 2632 passed + 5 skipped /
0 failures** — 30 new tests (TestLigandStructureGeneration ×6,
TestMglToolsConversion ×6, TestVinaBatchResume ×4, TestReview141Gui
×14; the docs-index test re-synced via `scratch/build_docs_index.py`
— 275 symbols). Doctests green on the three touched modules; ruff
clean on the docking package; `playground.py --self-test` green on
Xvfb :99 ("UI built, every tab visited … no errors"); the per-file
coverage probe green (119 rar + 14 upload, zero-hit 0). Version
chain: VERSION + pyproject + `pharmaml.__version__` + web
`SHIPPED_TOOLKIT_VERSION` → 1.2.80.

## [1.2.79] — 2026-10-26

### Added — the round-25 full review (100x-140): the forward links' POINTER confession + the census's KEYBOARD path + the confession's in-dialog RESET

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a twenty-fifth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–138's documented floor — the axis holds; no new gaps;
playground.py itself parses clean at 17,6xx lines with its full
structure intact). Round 139's handover named the round's work: (a)
the spotlight could grow a KEYBOARD path (the legs are buttons so
Tab+Enter already works — but a digit quick-switch could ride the
dialog's keydown), (b) the strip's confession could name the RESET as
an action inside the DIALOG too (today the way out is the card's chip
behind the dialog), and (c) the HTML twin's Contents lines could grow
the same title tooltip the return link gained (the forward links
confess nothing to the pointer world either — the round's own lesson,
applied symmetrically). All three shipped.

**The forward links' POINTER confession.** Every Contents line in
`HTMLReport`'s output now carries
`title='Jump to this page in the report'` — the same tooltip
discipline the return link gained in 1.2.78, applied symmetrically to
the other leg: the aria world always knew where the jump goes (the
line's own text names the page), a hovering pointer gets told now.
The tooltip is minted at the ONE place the lines are minted (the
render join), so href and tooltip can never drift — the lockstep
discipline the ids and names have always had. No caller change: the
desktop's HTML twin, `save_visual_bundle`, and the web bridge's
`/api/report/session` all gained it automatically; the PDF twin's ink
stays frozen, its lines stay plain text.

**The census's KEYBOARD path.** The web preview's spotlight gained a
keyboard twin: digits **1/2/3** toggle the census line's
1st/2nd/3rd leg (figure / table / structure grid — the digits count
the line's legs, clickable or plain, so the mapping is stable whatever
the session lists), **0** clears, every other key passes untouched
(Tab, Escape — already the pin-then-close layering's own — and
anything a modifier rides: the browser keeps those). The map is PURE
(`spotlightKeyAction` in `src/lib/session-report.ts`, probe-pinned)
and the dialog's keydown consumes the SAME verdict a click consumes —
the two paths cannot disagree. A zero-count leg's digit is a
swallowed no-op (the plain-text rendering's own honesty — nothing to
spotlight); the legs carry `aria-keyshortcuts` with their digit, and
the census paragraph's aria-label and title grew the keyboard
confession ("keys 1-3 spotlight the kinds, 0 clears").

**The confession's in-dialog RESET.** The strip's order confession
grew its way out INSIDE the dialog — a "Reset now" link beside the
sentence restores the canonical walking order without closing the
preview (the strip, the section list and the download all take it
immediately, through the SAME single `resetOrder` source the card's
dashed chip uses; the card's chip stays the other way out, for when
the dialog is closed). The confession retires honestly on reset —
the canonical order confesses nothing, as always.

### Verified

Bare `python -m pytest`: **2605 collected = 2600 passed + 5 skipped /
0 failures** (junit140.xml, scrubbed after the final run; the dev
server stopped for the run; both notebook drivers re-executed with
1.2.79 — 207 s / "26-R7 driver OK"; the re-execution's home-path leak
in two notebook outputs caught and scrubbed before the suite ran).
ruff clean after the standing EXE002 chmod sweep (113 files — the
git-index flip fired again); mypy fully clean (zero errors across all
65 source files); web `bunx tsc --noEmit` clean in src/ + `bun run
lint` clean. GUI round-trip on Xvfb :99: SMOKE140_OK + SHOOT140_OK
(pg140-picklist-anchor.png / pg140-forward-tooltip-report.html), VLM
review PASS on every shot (the picklist's numbered rows + confession
+ in-dialog Reset button verified legible, the web keyboard/reset
shots verified, mobile 390px no cut-off). PROBE140_OK (the keyboard
map's every verdict pinned). agent-browser E2E green — the KEYBOARD
golden path (pressing "1" spotlights figures while the table line
dims to 0.4; pressing "0" clears; non-digit keys pass; the spotlight
still resets across a dialog close), the RESET golden path (persisting
a custom order, then "Reset now" inside the dialog retires the
confession and flips the badge to "default order" without closing),
the API probe (POST `/api/report/session` 200 — the downloaded HTML
carries the forward tooltip on EVERY Contents line plus the return
tooltip on every wrapper), mobile 390×844 (scrollWidth ==
clientWidth == 390, dialog 358, legs + Reset now visible), the sticky
footer both ways, zero console/page errors.

## [1.2.78] — 2026-10-26

### Added — the round-24 full review (100x-139): the return link's POINTER confession + the census's CLICKABLE legs + the strip's ORDER confession

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a twenty-fourth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–137's documented floor — the axis holds; no new gaps;
playground.py itself parses clean at 17,6xx lines with its full
structure intact). Round 138's handover named the round's work: (a)
the HTML twin's return link could gain a title tooltip confessing the
jump target (the aria world already knows — the pointer world does
not), (b) the web preview's census could grow a click-to-spotlight
(clicking "2 figures" dims the non-figure lines), and (c) the
picklist's confession could surface in the WEB preview too ("your
last saved order" vs the canonical walking order). All three shipped.

**The return link's POINTER confession.** The small "↑ Contents" link
every page wrapper ends in now carries
`title='Back to the contents block at the top of the report'` — the
pointer world gets the same confession the aria world always had (the
link's own text names the destination; a hovering pointer had no
clue). One string, minted once (`_BACK_CONTENTS_LINK` in
`pharmaml.viz.report`), so the table and figure wrappers can never
drift apart — the same lockstep discipline the page ids follow. No
caller change: the desktop's HTML twin, `save_visual_bundle`, and the
web bridge's `/api/report/session` all gained it automatically; the
PDF twin's ink stays frozen.

**The census's CLICKABLE legs.** The web preview's census line became
interactive — each kind's count ("2 figures", "1 table", "1 structure
grid") is a button that SPOTLIGHTS that kind's lines in the strip:
the other kinds dim to 40% opacity (never hidden — every dimmed line
stays exactly as clickable as before, jumps included), the active leg
carries `aria-pressed` with accent + dotted-underline styling, and
clicking the active leg again clears the spotlight. A kind with zero
lines renders as plain text (nothing to spotlight — an honest no-op
avoided); the legs' labels are the same text the plain line prints
(`contentsCensusLegs`, pure, probe-pinned — the visual census can
never drift from the aria one); the spotlight resets when the dialog
closes, so a fresh open starts fresh.

**The strip's ORDER confession.** The web preview's Contents strip now
confesses the order's provenance — the web twin of the desktop
picklist's memory label: when the browser's persisted walking order is
in force, the strip's confession line appends "Your last saved walking
order was applied — Reset walking order forgets it." (the pure
`orderConfessionText`, probe-pinned); the canonical order confesses
NOTHING — the desktop's own noise rule ("nothing remembered: stay
quiet"), mirrored: there is no memory to name when none was applied.

### Verified

Bare `python -m pytest`: **2594 collected = 2589 passed + 5 skipped /
0 failures** (junit139.xml, scrubbed after the final run; the dev
server stopped for the run; both notebook drivers re-executed with
1.2.78 — 212 s / "26-R7 driver OK"; the re-execution's home-path leak
in two notebook outputs caught and scrubbed before the suite ran).
ruff clean; mypy **fully clean** (the toolchain rotation retired the
old 61-error baseline — zero errors across all 65 source files); web
`bunx tsc --noEmit` clean in src/ + `bun run lint` clean. GUI
round-trip on Xvfb :99: SMOKE139_OK + SHOOT139_OK
(pg139-anchor-confession.png / pg139-tooltip-report.html), VLM review
PASS on every shot (the anchor wording legible, the web confession +
census legs + badge verified, mobile 390px no cut-off). PROBE139_OK
(the legs + confession derivations pinned). agent-browser E2E green —
the SPOTLIGHT golden path (clicking "2 figures" dims the table line to
0.4 while both figure lines stay at 1; the dimmed line's jump still
lands; clicking the active leg clears it; the spotlight resets across
a dialog close), the confession golden path (persisting a custom order
appends "Your last saved walking order was applied", the canonical
order stays quiet), the API probe (POST `/api/report/session` 200 —
the downloaded HTML carries the tooltip on BOTH wrappers), mobile
390×844 (scrollWidth == clientWidth == 390, dialog 358, legs
visible), the sticky footer both ways, zero console/page errors.

## [1.2.77] — 2026-10-25

### Added — the round-23 full review (100x-138): the report's RETURN leg + the strip's KIND CENSUS + the picklist confession's PROVENANCE

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a twenty-third time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–135's documented floor — the axis holds; no new gaps;
playground.py itself parses clean at 17,5xx lines with its full
structure intact). Round 137's handover named the round's work: (a)
the HTML twin's Contents could grow a "back to contents" link on each
page wrapper (the anchor jumps one way; a reader deep in a long report
has no one-click return), (b) the web preview's Contents strip could
show the kind badges' counts (the census in the strip), and (c) the
desktop's imported order could confess its PROVENANCE in the picklist
itself ("imported from \<file\>" vs "your last saved order"). All
three shipped.

**The report's RETURN leg.** Every table/figure page wrapper in
`HTMLReport`'s output ends in a small "↑ Contents" link back to the
header's block — the nav carries `id='contents'` and its own
`tabindex='-1'` so the jump lands focus there, and it flashes the same
accent ring the jumped-to pages do (`:target`, the scroll-margin
breathing room included). The link is minted at ADD time with the
wrapper: a page wrapper exists exactly when at least one Contents line
does, so the back-link's target can never be missing, and a prose-only
report renders neither the nav nor the links (the noise rule's other
half). Fragment links only — the self-contained contract holds; the
PDF twin's ink stays frozen, its pages carry no return. The web
preview's strip needs no twin here: it is already sticky, which IS the
always-available return.

**The strip's KIND CENSUS.** The web preview's Contents strip header
now carries the census line — "2 figures · 1 table · 1 structure
grid" — derived from the strip's OWN entries (`sessionContentsCensus`
+ `contentsCensusText`, pure, probe-pinned), so metrics are excluded
by construction: the footer's own census counts metric cards, this
one never does — the strip confesses only its own lines. The wording
rides the footer's singular/plural discipline; a census with no images
drops the leg entirely; a metrics-only payload hides the whole strip
as before.

**The confession's PROVENANCE.** An order that arrived through File ▸
Import remembered order now says so in the picklist itself — "The
imported order from \<file\> was applied — Reset to walking order
forgets it" (the basename, truncated the honest way for very long
names) — until the reader completes their own Save (the order becomes
theirs; the wording returns to "Your last saved order was applied"),
resets, or clears. The empty import confesses nothing (an empty memory
is nobody's import); a foreign file never touches the live memory OR
its provenance; and a fresh LAUNCH restores from the session file —
which IS the last saved order — so a restart never claims an import
(the honest restart contract, test-pinned).

### Verified

Bare `python -m pytest`: **2586 collected = 2581 passed + 5 skipped /
0 failures** (junit138.xml; the dev server stopped for the run; both
notebook drivers re-executed with 1.2.77 — 206 s / "26-R7 driver OK";
the re-execution's home-path leak in two notebook outputs caught by
the round-137 guard and scrubbed before the clean re-run). ruff clean
after the EXE002 chmod sweep (the git-index flip fired again — 114
files); mypy at the 61-error pre-existing baseline, zero new. GUI
round-trip on Xvfb :99: SMOKE138_OK + SHOOT138_OK
(pg138-imported-confession.png / pg138-own-order.png /
pg138-return-leg-report.html / pg138-shared-order.json), VLM review
PASS on both shots. The web twin: `bunx tsc --noEmit` clean in src/,
`bun run lint` clean, PROBE138_OK (the census derivation pinned),
agent-browser E2E green (the census visible in the strip, the
download's HTML carrying the return links, mobile 390px, sticky
footer, zero console errors).

## [1.2.76] — 2026-10-25

### Added — the round-22 full review (100x-137): the order memory's export/import pair + the HTML twin's clickable Contents anchors + the web preview's Contents twin

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a twenty-second time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–135's documented floor — the axis holds; no new gaps).
Round 136's handover named the round's work: (a) the web preview's
dialog could show its own numbered-contents line (the file's Contents
block now exists on both twins; the preview's header could mirror it),
(b) the desktop picklist's memory could grow a cross-session
export/import (ride the report_page_order key the session already
serialises), and (c) the HTML twin's Contents lines could become
CLICKABLE anchors (ids on the page wrappers — the PDF twin has no
links, but HTML naturally does). All three shipped — plus CRON 473854,
which hit the exec-limits disable again (run_count 1), deleted and
recreated fresh as 473930.

**The order memory's SHARE pair.** The memory lived in exactly two
places — the process (100x-134) and the auto-saved session file
(100x-135, which a full Export session already carries) — so sharing
an order meant sharing the WHOLE session: every input, every pasted
corpus, every path. File ▸ Export remembered order writes the
standalone artifact instead: a tiny JSON carrying the format marker,
the kind tag and the two label lists (`dump_report_order` normalises —
the preference without the inputs). The honest edges: a memory that
never existed gets "Nothing to share yet" (never a silent no-op,
never an empty file pretending to be one), and a failed write confesses
on the job label. Import remembered order adopts a colleague's file
through `load_report_order` (a foreign or corrupt file — missing
marker, missing kind tag, unreadable JSON — is rejected with an honest
error and the live memory is NEVER touched), replaces the in-process
memory, and writes the session file through immediately (the Clear
menu item's discipline, mirrored for adoption — the imported order
survives a crash and the next launch re-opens pre-arranged exactly
where the colleague left it). A valid but EMPTY order imports as the
honest reset; `persist_session=False` names its scope. The next
picklist opens pre-arranged in the imported order with its confession
showing — proven end-to-end in scratch/shoot137.py across two fresh
apps (a planted memory exported, a fresh HOME imported it back).

**The HTML twin's clickable anchors.** Every table/figure page now
rides inside its own id-carrying wrapper — `<section class='page'
id='pg-N' tabindex='-1'>` for tables, `<figure class='page' id='pg-N'
tabindex='-1'>` for figures (the figcaption's legitimate parent at
last) — and every Contents line links to it
(`<li><a href='#pg-N'>name</a></li>`), so a click jumps the reader
straight to the page. The ids are minted at add time in lockstep with
the names, so href and target can never drift. The links inherit the
list's ink until the reader points at them (hover/focus → accent +
underline), the jumped-to page flashes an accent ring via `:target`,
`scroll-margin-top` gives the landing breathing room, and `tabindex`
lets the fragment jump land focus for keyboard readers. The fragment
links keep the self-contained contract (no network, no companion
files); the PDF twin's ink is frozen at write time, so its Contents
lines stay plain text. No caller change — `HTMLReport` grew it, so
the playground's session report and `save_visual_bundle`'s
report.html gained the anchors automatically.

**The web preview's Contents twin.** The preview dialog gained the
file's header block mirrored as a STICKY strip above the section list:
exactly the lines the downloaded file's Contents prints (every
table/figure/image section's title in walking order — metrics are
title-page material and never listed, the file twin's discipline; a
metrics-only report hides the strip entirely, the noise rule). Each
line is a labelled button — the section's overall number (the SAME
coordinates the numbered entries below carry), its kind icon, its
title — that JUMPS the dialog to its section: smooth scroll, a brief
accent ring on the arrival (the HTML twin's `:target` ring, mirrored
inside the dialog; it fades on its own timer), and `scroll-mt` so the
landing clears the pinned strip. The derivation lives in
`sessionReportContents` (src/lib/session-report.ts — pure, probe-pinned
by scratch/probe_session_report_137.ts: walking order, metrics never
listed, overall indices with gaps, kinds carried, the noise rule, the
file-twin parity). The strip stays pinned while the list scrolls — a
dialog is small, a contents list that scrolls away is a contents list
you can only use once.

**The build-machine-path leak, found and fixed.** The round's own
ZIP verification re-ran the 1.2.44 full-archive scan
(LEAKAGE_AUDIT.md's documented contract: "zero /home/<user>/ matches")
and found **42 entries leaking** — the 1.2.44 scrub had regressed
silently: rounds 120+ added scratch drivers whose run hints, sys.path
inserts and OUT paths reintroduced the absolute prefixes one round at
a time, and the later verification chains had dropped the scan (it
was not in the standing bump checklist). The fix: scratch/
scrub_home_paths_137.py neutralises every form (the venv python run
hints → `python`, the toolkit-root sys.path inserts → the documented
cwd, the absolute download/bridge targets → file-relative `Path`
computations, the junit xmls' recorded skip locations → the same
site-packages-neutral rewrite the notebook driver performs), and
**tests/test_leakage.py fails the suite forever after** if any
absolute home prefix returns — the pattern assembled at runtime so
the guard's own source never matches, with the scrub tool itself
excluded as a pattern-documenter (the LEAKAGE_AUDIT quoting
convention). All scrubbed drivers re-verified (SMOKE134/136/137_OK;
venv_repair_9.sh now rides $HOME so it still repairs THIS machine);
ruff clean after the sweep.

### Verified

- EVIDENCE CHAIN BEFORE WORK (all green): `--self-test` ("UI built,
  every tab visited…"); probe_coverage_111_perfile.py 118 rar (115
  covered + 3 absorbed) + 1 upload = 119/119, zero-hit 0; the
  notebooks probe green; audit_round7_122.py TOTAL 246 symbols /
  43 zero-hit — byte-identical to rounds 127-135. playground.py
  itself: AST parse OK, no truncation markers — the completeness
  verdict stands.
- TESTS: 22 new — TestReview137Gui (12, tests/test_playground.py: the
  dump/load normalisers incl. the foreign rejections and the honest
  empty; the export's standalone artifact (marker + kind tag + the two
  lists, NONE of the session's inputs) + the nothing-to-share honesty;
  the import's adopt-and-write-through with the picklist re-open
  pre-arranged + the foreign rejection that never touches the live
  memory + the empty-order reset + the persist-off scope; the
  docstring pins; the File-menu wiring via inspect.getsource) +
  TestPreviewContentsTwin (3, tests/test_web_parity.py: the TS probe's
  derivation verdicts pinned — walking order, metrics never listed,
  overall indices [1,2,3,5], kinds, the noise rule, the file-twin
  parity; the collector's export surface; the component's sticky
  strip + jump handler + arrival ring + scroll margin + confession
  + labelled buttons) + 2 anchor tests in tests/test_viz.py (every
  href resolves to a real id, fragment-only links, the wrappers with
  tabindex; the anchor CSS rides the theme variables) + TestLeakage
  (5, tests/test_leakage.py: the shipped tree carries zero absolute
  home paths; the scrub tool's idempotence documentation; the
  pattern catches the venv/repo-root/foreign-user forms). Standing
  pins updated where the feature evolved: the Contents `<li>` pins
  across test_viz/test_playground/smoke136 (now `<li><a href='#pg-N'>…`),
  and the bundle test's anchor resolution.
- DESKTOP ARTIFACTS: scratch/smoke137.py (SMOKE137_OK: the full story
  — normalisers, export, import, pre-arranged re-open, the anchors,
  the docstring pins) + scratch/shoot137.py (SHOOT137_OK: the File
  menu posted with BOTH new items between their siblings —
  pg137-file-menu.png; the imported-order picklist with its confession
  — pg137-imported-order.png; the artifacts pg137-shared-order.json +
  pg137-anchors-report.html with the clickable lines).
- VLM REVIEW on both shots: every QA point PASS — the File menu
  (dropdown visible, all three remembered-order items legible,
  Export/Import sitting immediately after Clear, zero defects), the
  imported-order dialog (numbered rows in the imported order, the
  "last saved order was applied" confession present, arrows visible,
  zero defects).

## [1.2.75] — 2026-10-25

### Added — the round-21 full review (100x-136): the File menu's Clear remembered order + the HTML twin's Contents block + the web preview's persistent pin

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a twenty-first time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–135's documented floor — the axis holds; no new gaps).
Round 135's handover named the round's work: (a) the HTML twin's
header could grow the same CONTENTS block the PDF title page gained
(its numbered headings already made one possible — HTMLReport would
need to collect its add_* headings), (b) the web preview's pin could
PERSIST per browser (localStorage, mirroring the desktop's new
session-file key — today the pin died with the dialog), and (c) the
desktop picklist could offer a "Clear remembered order" affordance in
the FILE menu (today only the dialog's Reset forgot, and only while a
drift existed). All three shipped — plus CRON 473787, which hit the
exec-limits disable again (run_count 1), deleted and recreated fresh
as 473854.

**The File menu's always-available forget.** The dialog's Reset
button only exists while the picklist is open, and only works while
the order has drifted from the walk order (it disables itself at the
walk order, the boundary arrows' honesty) — a reader who remembered
an order had no way to forget it without dialog gymnastics. File ▸
Clear remembered report order (sitting right after its forgetting
sibling "Clear saved session…") drops BOTH kinds and writes the
session file through IMMEDIATELY — the Reset's forget rides the
close-time auto-save, this one does not wait, so the forget survives
a crash. The honest messages cover every case: a memory that never
existed gets "Nothing to forget yet" (never a silent no-op),
persist_session=False names its scope ("the memory never reached a
file"), and a failed file write confesses the consequence ("the next
launch restores the remembered order after all"). The job label
carries the round's own status line.

**The HTML twin's Contents block.** HTMLReport now collects every
table and figure page name in the walking order the file writes them
(each line the SAME string the page itself prints — the table's
`<h2>` heading / the figure's `<figcaption>` caption; a heading-less
block lists its positional fallback `table N` / `figure N`, the
identity the alt already carries) and renders a `<nav
class='contents' aria-label='Contents'>` panel right under the
subtitle — the SAME mini table-of-contents the PDF title page gained
in 100x-135, themed through the same CSS variables both palettes
ride. A report with no table/figure pages skips the block entirely
(a contents list of nothing is noise, the PDF twin's rule).
save_visual_bundle's report.html gains it automatically (no caller
change), and the playground's HTML twin lists its count-prefixed
pages exactly the way the picklist's numbered rows read — the saved
message says so.

**The web preview's persistent pin.** The pin (Enter on a focused
trace group, 100x-135) died with the dialog; now it rides a small
external store over localStorage (`src/lib/pinned-trace-store.ts`,
keyed by the figure section's title — the web twin of the desktop's
report_page_order session-file key) wrapped as a React external
store: useSyncExternalStore subscribes, the snapshot reads an
in-memory mirror hydrated once from the key, writes update the key
AND the mirror and notify every subscriber, and SSR hydrates against
the null server snapshot — no restore effect anywhere, so the
react-hooks/set-state-in-effect rule stays clean. A stored pin is
validated against the live trace count at READ time (a stale index
pins nothing, the honest fallback), a restored pin confesses its
provenance ("pinned from your last visit"), the caption hints the
persistence ("a pin is remembered per browser"), and an empty map
REMOVES the whole key (the walking-order key's minimal-payload
discipline). The Escape layering (the 135 E2E catch) holds for
restored pins exactly as for fresh ones.

### Verified

- Full suite: **2544 collected = 2539 passed + 5 skipped / 0
  failures** (bare `python -m pytest`, junit136.xml; 19 new tests —
  TestReview136Gui ×11, TestHTMLReportContents ×5,
  TestSessionReportPinStore ×3); ruff clean; mypy at the 61-error
  pre-existing baseline (zero new, none in the round's regions).
- The web side: `bun run lint` clean, `bunx tsc --noEmit` clean.
- Notebooks re-executed with 1.2.75 via BOTH drivers (64: TOTAL
  208 s; 26: "26-R7 driver OK").
- The round's evidence: scratch/smoke136.py (SMOKE136_OK), the
  pg136-file-menu.png / pg136-after-clear.png shots +
  pg136-contents-report.html artifact (scratch/shoot136.py,
  SHOOT136_OK), the TS probe scratch/probe_session_report_136.ts
  (PROBE136_OK), and the agent-browser E2E (home renders v1.2.75 ·
  2539 tests; the preview's pin persists across a dialog
  close/reopen; zero console errors).

## [1.2.74] — 2026-10-25

### Added — the round-20 full review (100x-135): the picklist's order memory across restarts + the PDF title page's Contents + the web preview's keyboard inspect

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a twentieth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–134's documented floor — the axis holds; no new gaps).
Round 134's handover named the round's work: (a) the desktop
picklist's memory could PERSIST across sessions (today it was
per-session — a session-file key would mirror the web twin's
localStorage exactly), (b) the web preview's inspect could grow a
KEYBOARD path (focus a trace group + Enter shows the numbers — the
tooltip was pointer-only), and (c) the PDF twin's title page could
list the numbered page names as a mini table-of-contents (the count
prefix made one possible). All three shipped — plus CRON 473765,
which hit the exec-limits disable again (run_count 1), deleted and
recreated fresh as 473787.

**The order memory now survives RESTARTS.** The web twin persists
its walking order per browser (localStorage, since 100x-132); the
desktop twin's `_report_page_order` now rides the auto-saved
session file too — a `report_page_order` text key carrying one JSON
dict of kind → label list (the same JSON-in-a-text trick as the
finder's recency row, 100x-130). `_collect_session` serialises only
the two real kinds with labels coerced to str (never a half-typed
payload); `_restore_session` replants it before the first picklist
ever opens, parsed with the SAME tolerance — a corrupt or foreign
value resets per kind (a broken kind drops, a valid one in the same
payload still lands, exactly the way families restore), never the
restore. The Reset button's forget is symmetric: it empties the
attribute, the close-time auto-save writes the empty state, and the
next LAUNCH starts at the walk order — the web twin's localStorage
delete, mirrored exactly. `persist_session=False` keeps the
per-process memory and never writes (the test harness's world).

**The PDF's title page carries a CONTENTS section.**
`save_pdf_report` now writes a mini table-of-contents under the
subtitle: every page name in the walking order — the table pages
first, then the figure pages, exactly the sequence the document
writes them — each line the SAME string the page's own heading
prints (a shared `_page_heading` helper, so the list can never drift
from what the pages say). The playground's count-prefixed names
("figure N · <caption>" / "table N · <caption>", 100x-134) make the
list read like the picklist's numbered rows, so the reader can find
any page by the number they ticked; direct callers' names read
title-cased, the way their page headings always have. A report with
no tables and no figures stays a pure title page (a contents list of
nothing is noise, skipped entirely); a long list simply flows (auto
page break — never truncated, never capped); and a name shared by a
table and a figure is listed TWICE, honestly, because it is two
pages (a dict merge would silently drop one).

**The web preview's inspect grew a KEYBOARD path.** The hover
INSPECT (100x-134) was pointer-only; now every trace group in the
preview's FigureSketch is focusable — Tab reaches it, `role="group"`
carries the inspect text as its aria-label (screen readers announce
the same honest numbers), and focus thickens the trace exactly like
hover does (`group-focus-visible:stroke-2` /
`group-focus-visible:fill-primary/25`, mirroring the hover emphasis
stroke-for-stroke). Because the native SVG `<title>` tooltip never
appears on keyboard focus, a readout strip renders under the sketch
carrying the FULL inspect text (a polite `role="status"` live
region, so the numbers are announced as they land) — and **Enter
PINS it** (Space too — a toggle, so Enter again unpins; Escape
clears; only one trace pinned at a time; a pin outlives the focus
that set it, `pinnedTrace ?? focusedTrace`). The caption's
confession now names both paths: "hover a trace (or Tab to it) to
inspect its numbers, Enter pins the readout".

### Verified

- `playground.py --self-test` green; the GUI axis re-proven **119/119**
  (118 rar + 1 upload) before the round's edits; the library axis
  holds at **246 public symbols / 43 zero-hit** (byte-identical
  membership to rounds 127–134).
- 18 new tests: `TestReview135Gui` (9 — the collect path's JSON key
  with foreign kinds dropped; the end-to-end restart: completed Save
  → close-time auto-save → fresh launch replants; the Reset forget
  riding the file (the next launch starts at the walk order); the
  launch-path proof: a fresh boot from a remembering file opens its
  FIRST picklist pre-arranged with the confession rendered and the
  Reset armed; five poison payloads resetting per kind with the
  restore itself landing; persist_session=False keeping the
  in-process memory; three docstring/source pins) +
  `TestSessionReportKeyboard` (6 — every trace group focusable and
  labelled with guarded blur; the pin contract as a toggle with the
  readout precedence; the readout strip's live-region + confession;
  the focus emphasis mirroring hover; the type import pin; the E2E-caught Escape layering pin — Radix closes the dialog on Escape at document-CAPTURE, so the DialogContent itself holds the door while a pin lives: the first Escape clears the pin, the dialog stays, only a second closes it) + 3 new
  `TestSavePdfReport` cases (the Contents section's walking order
  with lines matching the page headings; the empty report's skip; a
  shared name listed twice). The two round-133/134 caption pins
  updated to the round's dual-path wording.
- The desktop smoke (`scratch/smoke135.py`): SMOKE135_OK — the
  restart story walked live (store → file → replant; the Reset
  forget across a restart; three poison payloads), the Contents
  section read back off a real PDF via pypdf (walking order, the
  shared-name double listing), the docstring pins.
- The screenshots (`scratch/shoot135.py`): SHOOT135_OK — two full
  app lifecycles (App #1 saves the order and closes; App #2 launches
  from the file and opens its first picklist pre-arranged with the
  confession + armed Reset), the PDF saved from the remembered order
  with its Contents page rasterised via pdftoppm
  (`download/pg135-contents-page-1.png`, the PDF riding along as
  `pg135-contents-report.pdf`), and the Reset press on the replanted
  memory (`pg135-restart-remembered.png`,
  `pg135-restart-reset.png`); VLM review of the shots — every QA
  point PASS.
- ruff clean (after the standing EXE002 chmod sweep); mypy unchanged
  (61 pre-existing baseline errors, all in `cli.py`, zero new); web
  `bun run lint` exit 0 and `tsc --noEmit` zero errors in `src/`
  (the examples/skills errors are pre-existing, outside `src/`).
- Suite: **2525 collected = 2520 passed + 5 skipped / 0 failures**
  (bare `python -m pytest` — the testpaths trap; junit135.xml; the
  dev server stopped for the run); the notebooks re-executed with
  1.2.74 via BOTH drivers first — 64: TOTAL 212 s; 26: "26-R7
  driver OK".

## [1.2.73] — 2026-10-25

### Added — the round-19 full review (100x-134): the picklist's order memory + both twins' count-prefixed page names + the web preview's hover INSPECT

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a nineteenth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–133's documented floor — the axis holds; no new gaps).
Round 133's handover named the round's work: (a) the web preview's
sketches could grow a hover INSPECT, (b) the desktop picklist could
remember the last custom order per session, and (c) the PDF twin
could ride the caption feed further. All three shipped — plus CRON
473637, which hit the exec-limits disable again, recreated fresh as
473703.

**The picklist's order now REMEMBERS.** The web twin persists its
walking order per browser (localStorage, since 100x-132); the desktop
twin now keeps the last SAVED dialog order per session: a completed
Save stores the dialog's per-kind label order
(`_report_page_order` — ALL labels, kept or not, so an unchecked page
keeps its place for the next visit), and the next Save's picklist
re-opens with each kind's pages already pre-arranged in that order
(labels the walk no longer finds are skipped — a re-rendered card, a
cleared table; a NEW page rides after the remembered ones, in walk
order). A muted confession line under the header says so ("Your last
saved order was applied — Reset to walking order forgets it."),
rendered only when a remembered order actually applied. The dialog's
own **"Reset to walking order"** button (the web twin's dashed Reset
chip, mirrored on the desktop) restores the walk order AND forgets
the memory — the next dialog opens in the walk order too — and its
enabled state rides the same refresh as the boundary arrows: enabled
exactly while the dialog's order has DRIFTED from the walk order (a
remembered order on arrival, or a manual ▲ / ▼ move), disabled at the
walk order itself. Cancel never touches the memory; only a completed
Save rewrites it.

**Both twins' page names carry the count.** The PDF's figure pages
are now named "figure N · <caption>" and its table pages
"table N · <caption>", and the HTML twin's figure captions and
table headings read the same — numbered in the picklist's own walking
order, so the page headings in both files read exactly the way the
picklist's numbered rows did (round 133 numbered the rows; round 134
makes the files agree). The "·" separator is latin-1-safe and a plain
file-name character, so the library's page-name validator and the
temp-png walk accept it untouched; `unique_name` stays as the
pathological guard. The HTML twin's 100x-124 " (2)" caption dedupe
retired with it — the position prefix already separates two
same-titled pages, and reads truer to the checklist.

**The web preview's sketches grew a hover INSPECT.** Every trace
group in the preview dialog's inline SVG sketch now carries a native
`<title>` — the honest numbers behind the approximation, zero JS:
the trace's name, point count, and x/y ranges (fmtCell-compacted,
the exponential branch included); a band reports its CENTER mean
(the polygon's edges are the ± spread, the mean line is what reads);
a classified scatter reports its class split ("31 class 1 / 9 class
0"); the empties confess ("no points" / "no finite points"); a
degenerate axis reads its single value (not "v…v"). The trace under
the pointer thickens while hovered (group-hover stroke/fill
emphasis) so WHICH trace the tooltip describes is never ambiguous,
the cursor confesses the affordance (`cursor-help`), and the caption
under the sketch says it out loud ("hover a trace to inspect its
numbers"). The wording lives in a pure exported helper
(`traceInspectText`) so the probe pins it verbatim.

### Verification

- `playground.py --self-test` green (UI built, every tab visited);
  `probe_coverage_111_perfile.py` 118 rar (115 covered + 3 documented
  absorptions) + 1 upload = **119/119, zero-hit 0**;
  `probe_coverage_113_notebooks.py` green;
  `audit_round7_122.py` **TOTAL 246 symbols / 43 zero-hit** (the
  documented floor, byte-identical membership).
- `scratch/smoke134.py` — SMOKE134_OK (the memory store + the
  count-prefix captions, the pre-arranged re-open + confession +
  cancel-keeps-memory, the stale-skip + new-append, the Reset's
  walk-order restore + forget + drift-honesty, the docstring pins).
- `scratch/shoot134.py` — SHOOT134_OK (`pg134-picklist-first-save.png`,
  `pg134-picklist-remembered.png`, `pg134-picklist-reset.png`, the
  `pg134-remembered-report.html` artifact with count-prefixed
  captions in the dialog's order).
- New tests: `TestReview134Gui` (9 — the save-stores/reopen-applies
  contract with the confession + enabled Reset, the stale-skip +
  new-append, the Reset's restore + forget + manager-state collapse,
  the drift-honest Reset through manual moves, the count-prefix HTML
  artifact, both menus' source pins, the docstring pins) +
  `TestSessionReportInspect` (3 — the TS probe pins over crafted
  traces: the line/band/scatter wording, the honest empties, the
  degenerate spans, the exponential branch; the collector's export;
  the card's inspect surfaces); the standing index pins updated for
  the count prefix (TestReview123's PDF page name,
  TestReview124's retired dedupe, TestReview132's three table-heading
  pins, the preview caption's hover confession).
- Full suite: **2507 collected = 2502 passed + 5 skipped / 0
  failures** (foreground, 443 s; junit134.xml); notebooks re-executed
  with BOTH drivers (64: TOTAL 206 s; 26: "26-R7 driver OK";
  `test_notebook_variants` 119/119); ruff clean + mypy at the
  61-error pre-existing baseline (zero new) after the EXE002 chmod
  sweep; web `bun run lint` clean + `bunx tsc --noEmit` clean in
  src/ (the examples/skills errors are pre-existing, outside src/).

## [1.2.72] — 2026-10-25

### Added — the round-18 full review (100x-133): the picklist's per-page reorder + the registry's captions + the web twin's PREVIEW

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited an eighteenth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–131's documented floor — the axis holds; no new gaps).
Round 132's handover named the round's work: (a) the desktop picklist
could gain per-page REORDER, (c) the surgery feed could reach the
DESKTOP picklist's labels, and (b) the web report could grow a
PREVIEW. All three shipped.

**The picklist's pages reorder.** The desktop twin of the web report
card's draggable chips: every row of the pre-save "Session report
contents" dialog now carries ▲ / ▼ buttons that move the page among
its OWN kind's neighbours — a figure never crosses into the table
block, because both walkers (`save_pdf_report` and the HTML menu's
builder) write the kinds as separate blocks. The rows themselves now
ride in the WALKING order — the table pages first, then the figure
pages, exactly the sequence the walkers ship them — so what the
checklist reads is what the file gets; a muted note under the header
says so ("This list is the report's walking order…"), the numbering
rewrites itself after every move ("figure 1 · …" follows the row, not
its birth order), the boundary buttons disable honestly (the first
row's ▲ is a no-op), and the returned `(kept_live, kept_tables)`
lists carry the dialog's order — the saved report walks its pages
the way the checklist read them (pinned end-to-end: the moved
figure's caption appears in its new position in the written HTML).

**The registry's labels carry counts.** `_render_figure` takes an
optional `caption` that overrides the running job's label in the
session-figure registry, and the three generative structure grids
pass theirs: "Ring topology moves — 6 mutant(s)",
"Scaffold-hop litter — 4/6 viable children",
"Top 12 molecules — the last run's leaderboard" — so the picklist's
rows (and the reports' figure captions) say what each page actually
holds instead of the card's job name.

**The web report card grew a PREVIEW.** A "Preview order" button
(Eye affordance, honest when nothing would ship) opens a dialog that
renders the ordered sections BEFORE any file is written — the exact
payload the download would POST (same options, same walking order):
numbered entries with kind badges (metrics / table / figure /
structure grid), metric cards as label/value chips, tables capped at
6 rows with the honest "+N more rows in the download", figures as
inline SVG SKETCHES drawn from the same trace data the bridge hands
matplotlib (lines → polylines, bands → filled polygons with the mean
line, class scatters → stride-capped dots — the dialog says the
sketch is an approximation), and the surgery grids' real PNGs (the
data URIs, direct). The dialog footer carries the census line and
its own download button; the collector exports the sketch's pure
helpers (`figureSketchBounds` — band edges participate, degenerate
axes pad ±1, empty sets fall back to the unit box — plus the
`PREVIEW_TABLE_ROWS` / `PREVIEW_SCATTER_POINTS` caps) so the probe
pins the arithmetic.

### Fixed

- `playground.py`'s venv reset for the sixteenth time (rdkit, meeko,
  gemmi, mordred, torch, ruff, mypy, the editable install — the
  proven repair recipe, run in foreground chunks after the background
  attempt died); `/tmp` wiped again (the rar audit re-extracted, 118
  files, the standing 2-failure cosmetic exit); Xvfb :99 respawned.
- CRON 473611 (round 132's 15-minute reviewer) hit the exec-limits
  disable again → deleted + re-created fresh as 473637.

### Verification

- `playground.py --self-test` green (UI built, every tab visited);
  `probe_coverage_111_perfile.py` 118 rar (115 covered + 3 documented
  absorptions) + 1 upload = **119/119, zero-hit 0**;
  `probe_coverage_113_notebooks.py` green;
  `audit_round7_122.py` **TOTAL 246 symbols / 43 zero-hit** (the
  documented floor, byte-identical membership).
- `scratch/smoke133.py` — SMOKE133_OK (the walking-order rows, the ▼
  move + renumber + the saved file following, the boundary disables,
  the caption feed, the docstring pins).
- `scratch/shoot133.py` — SHOOT133_OK (`pg133-picklist-walking-order.png`,
  `pg133-picklist-reordered.png`, the `pg133-reordered-report.html`
  artifact — beta genuinely first).
- New tests: `TestReview133Gui` (9 — the reorder's return order, the
  boundary disables, the saved-report order, the caption kwarg, the
  three grids' captions, the note, the docstring pins) +
  `TestSessionReportPreview` (4 — the TS probe pins over the fixed
  131 fixture: the bounds arithmetic, the caps, the section stream;
  the collector's exports; the card's preview surfaces); the
  TestReview132Gui index pins updated for the walking-order rows.
- Full suite: **2495 collected = 2490 passed + 5 skipped / 0
  failures** (foreground, 420 s; junit133.xml); notebooks re-executed
  with BOTH drivers (64: TOTAL 210 s; 26: "26-R7 driver OK";
  `test_notebook_variants` 119/119); ruff clean + mypy at the
  61-error pre-existing baseline (zero new) after the EXE002 chmod
  sweep (tests/ + src/ + playground.py — the bits had been reset
  again); web `bun run lint` clean; `bunx tsc --noEmit` clean for the
  touched files.

## [1.2.71] — 2026-10-25

### Added — the round-17 full review (100x-132): the report order — the desktop picklist + the web twin's reorderable chips

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a seventeenth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–131's documented floor — the axis holds; no new gaps).
Round 131's handover had named the round's work: (a) the desktop
session report's per-section control, (c) the web report's walking
order — both twins shipped — and (b) the surgery-feeding question was
settled by inspection (every surgery grid already renders through
`_render_figure`, so the desktop report already walks them; nothing
to build).

**Both desktop report menus consult a pre-save picklist.** The web
twin's report card gained its per-section control in 1.2.70; the
desktop's groups differ — it walks LIVE pages (one figure page per
card render plus one text page per filled table) — so the desktop
twin is a "Session report contents" dialog between the guards and
the file dialog: every page the walk found is offered with a
checkbox, all on by default, grouped and numbered ("figure 1 · …",
"table 1 · …"), and only the kept pages ship. A LIVE census footer
("N of M pages will ship") recounts on every flip; at zero pages it
says "Nothing would ship — tick at least one page." and the Save
button disables itself — and the confirm command re-checks the
census anyway (a keyboard/programmatic activation of the disabled
button stays a no-op). Cancel aborts with no save started at all.
Unchecking a page genuinely drops it from the written report — the
round's tests pin the filtered HTML byte-level (the unchecked
caption absent, the kept captions and tables present, the save
message counting the truth).

**The web report's section chips reorder.** Drag a chip (or ride its
hover/focus arrows) and the report walks its groups in that order —
the reader's narrative, not the code's. The collector was refactored
into six pure per-group builders assembled under a sanitized walking
order: unknown ids drop, duplicates collapse (first occurrence wins),
and missing groups still ship (appended canonical-last — a stored
order that predates a new group never silently drops it). The order
persists per browser (`pharmaml.reportSectionOrder`, tolerant
restore; equality with the canonical order removes the key entirely),
a dashed "Reset walking order" chip appears only while the order
drifted (the finders' Clear-chip convention), a GripVertical handle
plus a primary drop-caret cue the drag, each chip carries labelled
micro-arrows (the keyboard/touch path, with extended hit areas), and
the census line confesses "in your chosen walking order". Reordering
changes how the report reads, never what ships: the census is
order-invariant (pinned byte-identical to 1.2.70's baseline), and a
disabled group stays disabled wherever it sits.

### Verified

- `playground.py --self-test` green; the GUI axis re-proven **119/119**
  (118 rar + 1 upload) before the round's edits; the library axis
  holds at **246 public symbols / 43 zero-hit** (byte-identical
  membership to rounds 127–131).
- 11 new tests: `TestReview132Gui` (7 — every page offered and
  numbered; cancel aborts with no save; the unchecked figure/table
  never ships, the save message counts the truth; the census recounts
  live, Save disables at zero and the guarded confirm stays a no-op;
  both menus' wiring + the docstring pins) + `TestSessionReportOrder`
  (4 — the collector's order semantics over the fixed fixture: the
  sanitizer's unknown/duplicate/partial/empty contracts, whole-group
  moves that never re-cut a group's internals, the order-invariant
  census; the collector's exports; the card's drag/arrows/reset/
  persistence/a11y wiring pins; the desktop picklist's source pins).
  The six pre-132 report-menu test classes gained the
  `picklist_accept_all` bypass (their original whole-session
  semantics preserved).
- The desktop smoke (`scratch/smoke132.py`): every page offered, the
  filtered report written with the unchecked page genuinely absent,
  the census/Save/guard contract walked live, both menus' wiring —
  all green under Xvfb.
- The screenshots (`scratch/shoot132.py`): the picklist all-on and
  mid-filter states + a real filtered report artifact
  (`download/pg132-filtered-report.html`); VLM review of both shots —
  every QA point PASS.
- The TS probe (`scratch/probe_session_report_132.ts`): the 131
  census reproduced byte-identically under the refactored collector
  (no regression), the reorder semantics pinned, the sanitizer's
  corrupt-payload contracts pinned.
- ruff clean; mypy unchanged (61 pre-existing baseline errors, zero
  new); web `bun run lint` exit 0 and `tsc --noEmit` zero errors in
  `src/`.
- Suite: **2482 collected = 2477 passed + 5 skipped / 0 failures**
  (foreground chunks; the notebooks re-executed with 1.2.71 via BOTH
  drivers first — 64: TOTAL 190 s; 26: "26-R7 driver OK").

## [1.2.70] — 2026-10-25

### Added — the round-16 full review (100x-131): the web session report's per-section control + the surgery feed; both finders' clear-recent

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a sixteenth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–130's documented floor — the axis holds; no new gaps). The
round's genuine work was the two longest-standing handover items —
round 129's (a) the web session report's per-section toggle, and
round 130's (b) the surgery cards feeding that report — plus the
finders' (d) clear-recent affordance, both taken to completion.

**The web session report gained per-section control.** Every group
the session actually has content for now renders a toggle chip on
the report card ("Include in report"): ML training, learning curve,
PK / PD, NCA data, run history and — new this round — surgery. A
group with nothing to ship renders no chip (nothing to skip), so a
fresh session shows only PK / PD and the chips appear as the session
fills. The census line ("Ready to ship: … figures · structure grids
· tables · metric cards") and the download itself walk the SAME
filtered collector, so "what will ship" is always the truth; a fully
deselected session disables the button and says so. Skipped groups
persist in localStorage (`pharmaml.reportSections`, tolerant restore
— a corrupt payload resets to everything-on, never a broken report).
Skip the PK figures when only the ML run matters; ship only the
surgery grid; whatever the session needs.

**The surgery cards now feed the session report.** A run on either
Surgery card (ring topology moves or breed two molecules) persists
to the session store — `surgeryResult`, sessionStorage-backed with a
tolerant mode-discriminated validation, the latest run winning
exactly like `mlResult` does — and the report collector walks it:
the run's metrics card (source or parents, mutants or viable
children, seed, compute time, RDKit version), its SMILES list as a
real text table (one row per mutant or per draw, a failed crossover
an honest "(no viable crossover)" cell), and the captioned 2D
structure grid itself. The grid rides through a new **image section
kind** on the report bridge: the payload carries the PNG data URI
(the same raster the web card shows), the bridge validates it
(base64-decoded, PNG magic-checked, 2 MB decoded cap), re-composes
it as a full-bleed matplotlib figure sized to the raster's own
aspect (zero letterboxing, no re-scaling artefacts) and embeds it
through the SAME public `HTMLReport.add_figure` contract every
chart rides — caption included, self-contained file preserved. The
route's zod schema accepts the kind; the bridge census counts the
grid among the figures (that is what the reader sees).

**Both finders' recency rows gained a "Clear" affordance.** One
click on the dashed chip forgets every remembered jump: the desktop
empties the session-persisted `finder_recent` key and collapses the
row in place (the honest empty, no re-open needed); the web twin
clears its localStorage and hides the strip live — and because the
memory is gone, the next open renders no row at all (the standing
honest-empty rule). No row, no Clear button: a session without
recency never shows the affordance.

### Verified

- `playground.py --self-test` green (every tab visited, no errors);
  the GUI axis re-proven **119/119** (118 rar + 1 upload) before and
  after the round's edits; the library axis holds at **246 public
  symbols / 43 zero-hit** (byte-identical membership to rounds
  127–130).
- 15 new tests: `TestReview131Gui` (7 — the desktop Clear button:
  rides the row, empties the memory and collapses it in place,
  persists through the session payload, the next open renders no
  row, no button without recency, the round's docstring pins) +
  `TestSessionReportSections` (8 — the collector's option semantics
  over the fixed fixture: all-on/no-PKPD/no-surgery/surgery-only
  censuses, `sessionGroupsAvailable`'s honest empty map, the image
  section's data-URI contract, the piperidine mutant rows; the
  bridge's surgery-image round trip: 5 embedded figures incl. the
  grid, the caption, self-containment; the broken-image rejections;
  the store/cards/route/report-card/finder wiring pins).
- The bridge smoke: a real 72×36 matplotlib PNG round trips through
  the image kind 1:1 (the full-bleed axes re-embeds it at its own
  aspect); bad base64, a jpeg mime and a non-PNG payload all exit 2
  with the contract's own error text.
- ruff + mypy clean; web `bun run lint` exit 0 and `tsc --noEmit`
  zero errors in `src/` (the examples/skills errors are the
  pre-existing baseline outside `src/`).

## [1.2.69] — 2026-10-25

### Added — the round-15 full review (100x-130): the two-sided finder recency row + the web twin's Surgery sub-tab

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a fifteenth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **246 public symbols, 43 zero-hit**, byte-identical to
rounds 127–129's documented floor — the axis holds; no new gaps). The
round's genuine work was round 129's handover items (c) and (d), both
taken to completion.

**The Ctrl+F card finder gained a "Recently jumped" recency row on
BOTH sides.** Every jump the finder makes is now remembered —
most-recent-first, de-duplicated, capped at six (`_FINDER_RECENT_MAX`)
— and the dialog offers one-click chips back to those cards, so the
cards you keep visiting stay one click away without retyping. On the
desktop the memory persists with the session (the `finder_recent`
text key, JSON-encoded; the restore parses tolerantly — a corrupt
value resets the memory, never the restore), and the chips are
filtered through `_live_cards()` so a remembered title whose shell
vanished is honestly omitted, never a dead button. The web twin's
Ctrl+Shift+F finder gained the same memory (localStorage-backed
`pharmaml.cardFinder.recent`, capped at six): only a jump that
actually **landed** is remembered — a failed resolve surfaces the
honest "could not find" toast and never pollutes the row — and the
chips carry their sub-tab tag, ride under the query in a command-
palette-styled strip, and survive reloads.

**The web twin's Playground gained the Surgery sub-tab** — the mirror
of the desktop Generative tab's two molecule-surgery cards (100x-122):
*Ring topology moves* (every distinct one-step ring-size mutant of one
molecule, with the captioned 2D structure grid) and *Breed two
molecules* (the seeded scaffold-hop crossover, honest null rows for
failed draws, parents-and-children grid). One POST per card runs the
REAL `pharmaml.generative` operators — `ring_topology_moves` /
`scaffold_hop_crossover` — with real RDKit on the server through the
repo-root `scripts/surgery_bridge.py` (the same stdin→stdout JSON
bridge pattern as the feature factory and the report bridge; the
zod-validated route at `/api/gen/surgery` guards the caps: SMILES ≤
300 chars, children 1–8, seed 0–999999, ≤24 grid panels with an
honest truncation note). The bridge composes the SAME captioned-panel
structure grids the desktop's "Draw mutants" / "Draw litter" buttons
build (an RDKit `MolToImage` raster per panel, "(unparseable SMILES)"
slots, the classic "nothing to draw" line), rendered to a base64 PNG
data URI through `pharmaml.viz.figure_to_base64` — and each card
carries a **Download PNG** button (the web twin of the desktop cards'
per-card "Save figure…": a figure leaves the app from the card that
shows it). The card finder's catalog grew to 26 entries (the two
Surgery cards joined, `generative` tab), and the drift probe +
`TestCardFinderParity` pin them like every other entry.

### Verified

- `tests/test_playground.py::TestReview130Gui` (13 new): the recency
  memory (most-recent-first, de-dup, the six-cap), the dialog's chip
  row (live titles only, chip click jumps + closes + re-remembers,
  no row without memory), the session JSON round trip (incl. the
  tolerant restore), and the documentation pins.
- `tests/test_web_parity.py::TestSurgeryBridge` (5 new): the shipped
  bridge over fixed requests — the piperidine mutant set pinned
  (`C1CCCC1, C1CCCNCC1, C1CCNC1`), the acyclic honest-empty with its
  "nothing to draw" grid, the seeded litter (one row per draw, nulls
  included; parents + viable panels), per-seed determinism, and the
  client-error exits (unparseable SMILES, bad children/seed, bad
  mode → exit 2).
- `tests/test_web_parity.py::TestSurgeryWebWiring` (5 new): the
  sub-tab strip, the cards' route + honest-null row + Download PNG +
  the desktop defaults, the route's discriminated-union validation,
  the catalog's two new entries, and the web finder's recency wiring
  (`pharmaml.cardFinder.recent`, cap 6, land-only remembering).
- Suite: **2456 collected = 2451 passed + 5 skipped / 0 failures**;
  `probe_gui_e2e.py` gained the 100x-130 section (PROBE_OK);
  `probe_card_directory_129.ts` reports 26/26 with zero problems;
  the card-finder walk re-ran with the two Surgery entries;
  `smoke130.py` SMOKE130_OK; `shoot130.py` SHOOT130_OK (VLM-reviewed
  screenshots: the desktop chip row, the desktop breed card with the
  litter grid, the web Surgery tab, the web finder's recency strip —
  every QA point PASS, the one full-window "clipping" claim disproved
  by the precise crop, the round-129 lesson standing).
- All 8 notebooks re-executed with 1.2.69 under BOTH drivers (the
  standing trap honoured).

## [1.2.68] — 2026-10-25

### Added — the round-14 full review (100x-129): the PDF session report's dark-ink variant + the web twin's card finder

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a fourteenth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **245 public symbols, 43 zero-hit**, byte-identical to
rounds 127/128's documented floor — the axis holds; no new gaps). The
round's genuine work was round 128's handover items (a) and (d), both
taken to completion.

**The PDF session report gained its dark-ink variant, riding the View ▸
theme radios** the same way the HTML report's palette does (1.2.60):
save while the app is dark and the PDF comes out in the app's own dark
palette — near-black pages, light ink and accent headings, every figure
on a white plate (the bright-window look of the dark HTML report);
save while light and it stays the classic white paper,
**byte-identical to every prior release** (the light path never touches
a colour operator — pinned by test). `auto` resolves through
`_resolve_auto_theme` at save time, because a PDF's ink is frozen at
write time — there is no reading-time toggle like the HTML twin's, and
the library's new `PDF_THEMES = ("light", "dark")` contract says so on
purpose (exported from `pharmaml.viz`, alongside `REPORT_THEMES`).
`save_pdf_report(..., theme=...)` is the library surface
(`src/pharmaml/viz/report.py`): an fpdf2 `set_page_background` page
fill that follows the auto page break (a long dark table never bleeds
white), explicit per-write text colours (ink / muted notes / accent
headings — fpdf colours stick, so every dark-mode write passes its
colour explicitly), and the white figure plate sized from the saved
PNG's real pixel aspect (PIL — matplotlib's own hard dependency — with
a missing-PIL fallback that drops the plate, never the figure). The
playground wires it through `_menu_save_session_report`: the theme
radio is read on the main thread (the HTML report's own pattern),
`auto` is resolved first, the guard falls back to `light` if the radios
ever drift from `PDF_THEMES`, the subtitle names the variant, the save
dialog tells the user which ink they got and how to get the other one
("Flip the View ▸ theme radio before saving"), and a quiet disabled
caption under the radios — "(both session reports save in this
theme)" — makes the double duty discoverable in the menu itself.

**The web twin gained the card finder** — the Ctrl+F finder's own
mirror (round 1.2.59 shipped the desktop one): `Ctrl/Cmd+Shift+F`
(not plain Ctrl+F — the browser owns that one, and the dialog's footer
says so) opens a fuzzy-filtered dialog over every always-rendered
Playground card — 24 entries across all eight sub-tabs plus the
session-report strip, each with its title, blurb and tab tag. Jumping
switches the top tab (opening the finder from elsewhere lands you in
the Playground first), the sub-tab and — for the PK / PD Explorer —
the inner tab, then scrolls the card into view and flashes it
(`pg-card-flash`: a primary ring + soft wash that settles out, the
desktop finder's flash-and-jump semantics). The catalog lives in
`card-directory.ts` (the hand-maintained twin of the desktop's
`_CARD_INDEX`); the jump resolves each entry's match string against
the live `[data-slot="card"]` elements — **title-first, panel-scoped**,
and gated on the target tab actually being mounted, because the first
draft's naive resolve hit a card from the tab you were LEAVING (an
Overview blurb contains "Torsion QA" — the exit-animation race, caught
by the E2E probe and fixed before shipping). A renamed card surfaces
honestly as a "Card not found" toast instead of a silent wrong jump;
cards that only appear after a run (ML results, the biology matrix)
are deliberately absent — the finder lists what is on screen, the same
honesty rule as the desktop finder.

### Verified

- **Library**: `PDF_THEMES` export + the `ValueError` guard (message
  pins the frozen-at-write-time contract); a real dark vs light PDF
  pair over a figure + table + metrics — the light output carries ZERO
  colour operators (byte-classic), the dark output carries the page
  fill `0.0824 0.102 0.0941 rg`, the accent `0.0863 0.6627 0.5412 rg`,
  the light ink `0.9059 0.9255 0.9137 rg` and the plate `1 1 1 rg`
  (asserted on the zlib-decompressed page streams — fpdf2 compresses).
- **GUI**: `TestReview129Gui` (9 tests) — the library contract, the
  dark/light/auto rides through the REAL menu (spy on
  `save_pdf_report`'s kwargs + the written file's colour operators),
  the View-menu caption pin, and the module/method docstring pins.
- **Web**: `TestCardFinderParity` (2 tests) — the catalog cannot drift
  (a bun probe imports the shipped `card-directory.ts` and checks
  every match string + title against the component sources, JSX
  entities normalized) and the wiring pins (page-level mount,
  sub-tab-strip trigger, the hotkey contract, the flash keyframes).
- **E2E**: `probe_gui_e2e.py` + the 100x-129 section (a REAL IV plot →
  dark report: page fill + plate + light ink + the dialog hint; light
  report: zero colour ops + different bytes; auto resolves) →
  PROBE_OK; `probe_card_finder_129.sh` walks **all 24 catalog entries
  through the live finder UI** — every jump lands on the right card,
  the right sub-tab (inner PK tabs included), in viewport, from a
  foreign top tab, 24/24 → PROBE129_OK; `smoke129.py` → SMOKE129_OK;
  `shoot129.py` → the dark/light app screenshots + the round's own
  `pg129-session-dark.pdf` / `pg129-session-light.pdf` artifacts, VLM
  review of the exact figure-host crop: COMPLETE (the full-window
  "clipped label"/"overlapping characters" claims were VLM misreads
  of the en-dash in "Cockcroft–Gault" and the "…" in "Save figure…"
  at small scale — the precise crop shows both axis titles fully
  visible).

## [1.2.67] — 2026-10-25

### Added — the round-13 full review (100x-128): the per-card figure export closes app-wide

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a thirteenth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: **245 public symbols, 43 zero-hit**, byte-identical to
round 127's documented floor — every member a previously verified
source absorption: `check_torsion_consistency`/`TorsionCheck` are the
engine behind the ligand-audit card's folder walk, `PKSummary`/
`aucm_trapezoid` are what the PK card reads (`summary.aucm`),
`correlate_with_response` is absorbed inside `sensitivity_associations`,
and so on; no new library-symbol gaps). The round's genuine work was
round 127's handover item (b), taken to completion:

**Every remaining figure card in the playground now carries its own
"Save figure…" export.** Rounds 1.2.65/1.2.66 shipped the pattern for
the Generative cards; 1.2.67 closes the set — one button per card,
each saving the figure the card's **own** host shows (never the global
`Ctrl+S` last-drawn-anywhere figure) through the library's
`save_figure` with the menu path's PNG/SVG/PDF filetypes:

| Tab | Cards gaining the per-card export |
| --- | --- |
| Data & ML | **Results** (parity / calibration / learning curve, whichever drew last — one host, many figures) · **Leaderboard** (the cross-validated bars) · **Applicability domain** (the two-panel reliability picture) |
| Deep Learning | **Results** (parity or confusion matrix — the card had no button row of its own until now) · **Training history** (a TWO-host card: loss curves + learning curve; the newest render stamp wins, the same semantics the Generative Results card uses) |
| PK / PD | **IV bolus profile** · **Emax pharmacodynamics** · **NCA** (the concentration-time profile with its AUC) · **infusion accumulation** · **pharmacology family** |
| Chemistry | **Feature factory** (the descriptor histogram grid) · **similarity scan** (the heat-map) |
| Docking | **receptor card** (the atom-type census bars) · **results card** (the top-ligands energy bars) |
| Utilities | **table reconciliation** (the three-way overlap figure) |

- The honest guards ride everywhere: a card with nothing drawn yet
  answers with the per-card hint ("Run … first — this saves the figure
  the card is currently showing"), never a crash and never the global
  shortcut's silent wrong-figure race.
- The DL Results card's button row is new UI (the train buttons live
  on the Network card); every other button joined its card's existing
  row. Verified in-browser-class GUI E2E: a real IV profile drawn by
  the card's own Plot button, exported by the card's own Save button
  over a decoy global figure; the DL history card's two hosts racing
  with the newest stamp winning both directions.
- Tests: `TestReview128Gui` (20) — the per-tab button census (15 new +
  4 Generative = 19 app-wide), parametrized guards over all 15 cards,
  the NCA decoy semantics, the DL two-host newest-wins, and the
  docstring/home-guide pins. Probe/smoke/shoot sections re-run clean;
  VLM review of the IV / NCA / ML card screenshots: every QA point
  clean (buttons visible in the rows, figures fully visible, labels
  readable, no defects).
- Notebooks re-executed with 1.2.67 via BOTH drivers (the 26-driver's
  first run hit the box's 4 GB-cgroup OOM killer mid-kernel — dmesg
  pinned it; the documented re-run was clean).

## [1.2.66] — 2026-10-24

### Added — the round-12 full review (100x-127): the per-card figure export on every Generative figure card, and the web twin's session report

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a twelfth time (the
rar/upload contracts re-proven 119/119 first — 118 rar files: 115
covered + 3 documented absorptions, 1 upload — then the library-axis
audit re-run: **245 public symbols, 44 zero-hit**; the +1 over round
125's floor is `REPORT_THEMES` itself, exported with the round-126
theme engine and until now consumed only inside the library — this
round gives the GUI a real caller for it, below). The round's genuine
work sits where round 126's handover pointed:

| Surface | What 1.2.66 adds |
| --- | --- |
| Playground ▸ Generative ▸ Results card | **Save figure…** — the per-card export the Pareto card shipped in 1.2.65, now on the one card that holds **two** figure hosts (the leaderboard structure grid and the evolution history). `_render_figure` stamps every host with a monotonic `_pg_figure_seq`; the save exports whichever of the card's own figures drew **last** — still strictly narrower than the global Ctrl+S, which spans every card in the app |
| Playground ▸ Generative ▸ Ring topology moves card | **Save figure…** — the mutant structure grid leaves the app through the card's own button, not the global last-drawn-anywhere shortcut |
| Playground ▸ Generative ▸ Breed two molecules card | **Save figure…** — the parents-and-children grid, same honest contract (guard path included on all three: nothing on the card → the honest info box) |
| `pharmaml.viz.REPORT_THEMES` | **A real GUI caller.** The HTML session report's bake now validates the app's theme against the library's own public contract — a drifted/invalid value falls back to `auto` instead of letting `HTMLReport` raise mid-job (the audit's only new zero-hit becomes a wiring, not a constant) |
| Web ▸ Playground | **The session report twin** — the web mirror of the desktop's `File ▸ Save session report (HTML)…`. A "Session report" card rides above the eight sub-tabs with a live census ("4 figures · 2 tables · 2 metric cards ready"); one click collects the persisted session state (the last ML training result with its parity/decision series, the learning-curve run, the always-live PK/PD regimen, the NCA rows, the run history) and downloads a self-contained HTML file |
| `/api/report/session` + `scripts/report_bridge.py` | The server side of the twin: the collector's section payload (line / band / class-scatter traces with the web boundary-dot opacity semantics) is composed into **real matplotlib figures** and assembled through the **same `HTMLReport` the desktop saves** — theme baked from the app's resolved palette, the per-reader toggle included, figures base64-embedded, zero external references |

- The learning-curve result moved into the store (`lcResult`,
  sessionStorage-persisted with the same light-structural validation
  `mlResult` gets), so the collector walks it and the card survives
  reloads identically.
- Parity guard: `tests/test_web_parity.py::TestSessionReportContract`
  runs the round trip from inside the suite — the SHIPPED TS collector
  over a fixed fixture snapshot (census pinned: 4 figures / 2 tables /
  2 metric cards; the parity figure's four-trace contract pinned in
  order), its payload piped through the shipped bridge, and the
  composed document asserted against the desktop report's own contract
  (theme attribute, toggle + localStorage, 4 embedded PNGs, 2 tables,
  no `<link>`, no external src/href). The bridge's 400 paths (invalid
  theme, malformed sections, non-finite coordinates) fail cleanly with
  one-line JSON errors.
- 12 new tests: `TestReview127Gui` (10 — the four buttons pin, the
  per-card guards, the topology save over a decoy `_last_figure`, the
  two-host newest-wins semantics with the stamp ordering, the
  REPORT_THEMES fallback to `auto`) + the 2 contract tests above.
  Suite: **2402 collected = 2397 passed + 5 skipped / 0 failures**.
- Browser-verified end-to-end (agent-browser): the card renders the
  live census, the golden path (train → learning curve → download)
  fires POST /api/report/session → 200, the downloaded HTML opens with
  the baked theme + toggle + embedded figures, and the VLM review of
  the three GUI screenshots (topology / breed / Results cards, each
  with the new button over a real drawn grid) passed all QA points.

## [1.2.65] — 2026-10-24

### Added — the round-11 full review (100x-126): the HTML report's theme engine and the Pareto card's per-card figure export

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited an eleventh time (the
rar/upload contracts re-proven 119/119 first — 118 rar files:
115 covered + 3 documented absorptions, 1 upload — then the
library-axis audit re-run: **244 public symbols, 43 zero-hit at the
documented floor**, byte-identical to round 125's final set; the two
fresh-eye suspects were re-verified as absorptions in source:
`check_torsion_consistency`/`TorsionCheck` are the single-file engine
behind the ligand-audit card's `scan_torsion_consistency` folder walk
(playground.py:9693), and `PKSummary`/`aucm_trapezoid` are the return
type + internal integrator behind the PK card's `summary.aucm` readouts).
The round's genuine work sits where round 125's handover pointed —
both UI improvements, one on each side of the report path:

| Surface | What 1.2.65 adds |
| --- | --- |
| `pharmaml.viz.HTMLReport` | **A theme engine.** The constructor gains `theme="auto" \| "light" \| "dark"` (validated, default `auto`), baked as the `<html data-theme=…>` attribute; the stylesheet was rebuilt on CSS custom properties so the dark palette applies via `html[data-theme='dark']` (forced) *or* the original `prefers-color-scheme` media query under `data-theme='auto'` — **the auto behaviour is preserved palette-for-palette** (muted greys stay fixed across themes exactly as the shipped auto-dark always rendered them; the one deliberate change: the footer's border joins the themed borders instead of staying light-on-dark) |
| Every generated report | **A per-reader toggle button** rides the top-right: one click cycles auto → light → dark, the label (◐ / ☀ / ☽) follows, and the choice is remembered per browser via `localStorage` — the sender pins a starting palette, the reader overrides it for themselves. The script is inline: the self-contained contract (zero external resources, survives being moved or e-mailed) holds |
| Playground ▸ File ▸ Save session report (HTML)… | The report now **opens in the app's own View ▸ theme setting** — the theme radios do double duty (they tint the playground AND bake the report's initial `data-theme`, captured on the main thread before the worker starts). The completion dialog tells the reader about the toggle |
| Playground ▸ Generative ▸ Pareto front card | **Save figure…** — a per-card export of whatever the card currently shows (the colour-coded structure grid or the objective scatter, whichever drew last), through the library's `save_figure` with the same PNG/SVG/PDF filetypes as the menu path. The honest distinction from Ctrl+S: the global shortcut saves whatever figure drew last **anywhere** — run any other calculation in between and the front is no longer what gets saved; this button reads the host's `_pg_figure`, the exact object the card shows. Guard path included (nothing on the card → the honest info box) |

- The theme engine is verified end-to-end in a real browser (agent-browser):
  forced dark computes `rgb(11, 18, 32)` (#0B1220), the toggle cycles
  dark → auto → light → dark with the background following at every
  step, `localStorage` persists, and **a saved `light` overrides a
  baked `dark` on reload** — the reader's choice beats the sender's
  pin, exactly the designed semantics.
- The browser-verified toggle screenshots passed VLM review (all QA
  points clean: button visible, palette correct, metric cards and the
  embedded figure readable, no layout defects).
- `REPORT_THEMES` joins the public API (`pharmaml.viz.REPORT_THEMES`)
  and `docs/INDEX.md` (regenerated; the docs-index test guard caught
  the symbol the moment the suite ran — the guard works).

### Fixed

- Round 126's first test draft hit the round-123 modal-dialog trap
  AGAIN (an unpatched success `showinfo` hung 400 s until the
  faulthandler traceback pinned it at the exact line) — the test now
  patches every dialog up front; recorded so the next round patches
  dialogs BEFORE the first run, not after the hang.

## [1.2.64] — 2026-10-24

### Added — the round-10 full review (100x-125): the Pareto front's colour-coded structure grid and the web learning curve

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a tenth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: 43 zero-hit symbols at the documented floor — the
round-124 set, every absorption re-verified in source this round:
`bootstrap_uncertainty` behind the AD card's sensitivity engine,
`ModelInputReport` behind `validate_bundle_input` at the bundle audit
card, `TrialResult` behind the tuner's trials CSV export,
`ring_topology_mutate` inside the GA's topology mutation operator).
The two genuine gaps closed this round sit where round 124 pointed
them: the front's molecules were still invisible, and the web twin
still lacked the desktop's learning-curve figure:

| Surface | What 1.2.64 adds |
| --- | --- |
| Generative ▸ Pareto front card | **Draw front** — the front's members as 2D structure panels (top 12 in leaderboard order), each panel's score row carrying the member's objective values, and NSGA-II's crowding verdict encoded in colour: **boundary corners (best/worst on some objective, crowding +inf) accented teal, interior trade-offs muted neutral** — the accent speaks twice (caption colour + a visible panel border) so the split survives grayscale printing and colour-vision differences alike. `Plot front` brings the scatter back (the established figure-replaces-figure pattern) |
| `molecule_grid_figure` | An optional **colours row**: one accent colour per entry (caption + border). The same pad/truncate contract as the scores row — a shorter list pads with classic panels, a longer one is truncated, so a caller's off-by-one can never shift the grid |
| Web ▸ ML Training tab | **Learning curve — is more data still buying accuracy?** — the web twin of the desktop card's `plot_learning_curve` figure. New `POST /api/ml/learning-curve` re-runs the parked train request (synthetic OR CSV) at the library's default 10 % … 100 % 8-step schedule with k-fold CV (3/5/10) at every size; both curves render with mean ± 1 std fold-spread bands (the GP chart's stacked-area recipe), the gap verdict strip, and the three-reading hint. Regression GP with unset kernel slots clones the run's LML-learned kernel (ONE full-data fit, then fixed for every refit — the sklearn clone semantics) |
| Web ▸ ML Training tab | `src/lib/learning-curve.ts` — the pure module the route imports: the numpy-mirroring `linspace(0.1, 1.0, 8)` schedule, the 0.15 gap threshold, the byte-mirrored verdict strings, and the fold-reduction engine |

- The contract parity guard `scratch/probe_lc_parity_125.{py,ts}` +
  `TestLearningCurveContract`: the fractions match **bit-for-bit**
  (the TS linspace reproduces numpy's arithmetic exactly), the gap
  threshold agrees (0.15, read from the shipped Python source), and
  the verdict strings compare **byte-for-byte** — the Python side
  reads them off REAL figures the shipped library produced (an
  overfitting deep tree, a converging OLS), the TS side formats the
  SAME parsed gaps. The reduction engine is pinned on the TS side
  against a hand-computed fixture (prefix subsampling of each fold's
  train part, 1-based fold slots, population std).
- The same one-message verdict convention as the desktop: the gap
  strip colours rose over 0.15, emerald under, exactly where the
  verdict text flips.

### Fixed

- The environment's /tmp wipe re-flipped executable bits across the
  whole tree (112 EXE002 ruff hits) — the de-execute walk ran again.
- A stale display artifact (`[m` swallowed by the terminal) briefly
  looked like a corrupted type-ignore comment; hexdump proved the file
  correct — no change needed, recorded so the next round does not
  chase the same ghost.

## [1.2.63] — 2026-10-24

### Added — the round-9 full review (100x-124): the HTML session report and the scored leaderboard grid

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a ninth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: 44 zero-hit symbols, every claimed absorption
re-verified in source — `correlate_with_response` inside
`sensitivity_associations`, `aucm_trapezoid` inside `pk_summary`,
`check_torsion_consistency` behind `scan_torsion_consistency`). The
audit's genuine find: **`pharmaml.viz.HTMLReport` had NO caller
anywhere** — a documented, user-facing, self-contained HTML report
builder shipped dark. This round gives it its surface, and the
generative Results card gains the structure grid its leaderboard
always deserved:

| Surface | What 1.2.63 adds |
| --- | --- |
| File ▸ Save session report (HTML)… | **The portable twin** of the PDF report: `HTMLReport`'s first caller anywhere. The exact same two axes (every live session figure + every filled table, walked by the same registry + table walk), plus the builder's own **metric cards** (`add_metrics` — the session census: figures, tables, cards on screen, pharmaml version — which not even the library exercised) and an `add_text` provenance block. One `.html` file: figures base64-embedded, zero external resources, renders identically offline, survives being moved or e-mailed. No fpdf dependency |
| Generative ▸ Results card | **Draw top molecules** — a 2D structure panel per leaderboard row (top 12, rank 1 first). The ranked molecules become visible, not just tabulated |
| `molecule_grid_figure` | An optional **scores row**: one score line per entry rendered as the panel's xlabel, a distinct row under every structure (a ranked leaderboard reads at a glance). Shorter lists pad, longer ones truncate — a caller's off-by-one can never shift the grid. NB: `axis("off")` would hide the xlabel machinery (`axison` skips the whole xaxis at draw time), so the panels hide frame + ticks instead |
| Web ▸ Cell Line tab | **Rank variable genes (MAD)** — the desktop's MAD pre-filter mirrored as a disclosure on the web biomarker card: `src/lib/variable-genes.ts` (the TS twin of `most_variable_genes`: same median/MAD conventions, MAD descending, ties by gene name ascending) computed over the SAME expression matrix in the biomarkers route, parity-guarded row-by-row |

- The HTML report's captions dedupe the way the PDF's page names do
  (two cards carrying the same job label get numbered captions), the
  save message counts both axes, and the guard covers both axes
  ("Nothing to report yet" fires only when neither a live figure nor
  a filled table exists).
- The web parity guard (`TestVariableGenesParity`) runs both sides on
  two hand-crafted matrices (a 9-line odd panel and an 8-line even
  panel, each with MAD ties, a constant gene, negatives and decimals)
  and asserts row-by-row agreement — gene order exact, MAD values to
  1e-12; the fixtures also pin the semantics (the runaway-spread gene
  leads, the constant gene is last, the MAD ties rank alphabetically).
- Tests: 12 new in `TestReview124Gui` (the grid's scores-row contracts
  incl. pad/truncate/None, the draw-top guard + full real-evolution
  path, the button pin, the File-menu pin, the HTML report's guard +
  real-file write with embedded figures/tables/metric cards +
  caption dedupe, the docstring/home-guide pins) + 1 web parity test
  → **2363 passed + 5 skipped / 0 failures** (2368 collected).

## [1.2.62] — 2026-10-24

### Added — the round-8 full review (100x-123): the report's table axis and the surgery structure grids

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited an eighth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: the 44 remaining zero-hit symbols re-examined and every
claimed absorption re-verified in source — the GA/RL thin wrappers,
`SmilesRnn` behind `ReinventOptimizer`, the RISM constant family
behind the RISM card's meanings). The library axis stays at its
documented floor; this round closed the two genuine CONTENT gaps on
the GUI's own surfaces:

| Surface | What 1.2.62 adds |
| --- | --- |
| File ▸ Save session report (PDF)… | The **table axis**: every filled table on screen — leaderboards, prediction sheets, descriptor panels, docking results, any card's table with rows — is walked the same way the figure registry walks hosts and shipped through `save_pdf_report(tables=…)` as REAL text pages (searchable, selectable, the library's own 38-row truncation note). The PDF is now the session's whole record: pictures AND numbers |
| Ring topology moves card | **Draw mutants** — a 2D structure panel per enumerated mutant (RDKit `MolToImage` rasters in a matplotlib grid); the ring-size contraction/expansion becomes visible instead of a line of text |
| Breed two molecules card | **Draw litter** — parents and viable children side by side, one 2D panel each; the bridge / ring-merge / linker-transplant geometry of every crossover draw becomes visible |

- **The table walk** (`_live_session_tables`) discovers every live
  `ttk.Treeview` with rows and derives each table's provenance by
  climbing the widget tree to the nearest registered card shell (the
  `Ctrl+F` card index), so no card needed per-site registration;
  trees outside any card (the finder dialog) and empty trees are
  skipped honestly. Hierarchical trees (`show="tree headings"`, e.g.
  the sensitivity matrix) contribute their `#0` labels as the first
  column exactly the way users read them.
- **The structure grids** (`molecule_grid_figure`) never lie: an
  unparseable SMILES keeps its panel with an honest note instead of
  silently vanishing, and an empty litter draws the "nothing to draw"
  line. Captions carry the label AND the truncated SMILES, so a grid
  pasted into the session report stays self-describing.
- The report guard now covers both axes ("Nothing to report yet" only
  when there is neither a figure nor a filled table), the save
  message and the PDF subtitle count both page kinds, and page names
  reuse the plain-file-name contract (`"1 of 7"`, never `"1/7"`).
- Tests: 17 new GUI tests (the grid helper's panels/edge cases, both
  draw buttons' full job paths and no-data guards, the card-title
  climb, the walk's empty/hierarchical/real-job contracts, three real
  PDF writes incl. the tables-only and combined-axes reports, and the
  docstring/home-guide pins). Full suite: 2355 collected, 0 failures.

## [1.2.61] — 2026-10-24

### Added — the round-7 full review (100x-122): the reference objectives, molecule surgery, the session PDF report

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a seventh time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: 51 zero-hit symbols re-examined one by one, every
claimed absorption verified in source). Five genuine user-facing gaps
surfaced; all closed in this release:

| Library capability | playground.py surface since 1.2.61 |
| --- | --- |
| `generative.SimilarityObjective` | Generative → Objective & evolution — the combobox's new *Similarity to reference SMILES* entry (Tanimoto/Morgan to the reference molecule); the swapped-in reference row carries the seed SMILES |
| `generative.NoveltyObjective` | Generative → Objective & evolution — *Novelty vs corpus*: 1 − the best Tanimoto against the card's own corpus (the library's built-in 244-molecule pool when the text is empty) |
| `generative.SubstructureObjective` | Generative → Objective & evolution — *Contains substructure (SMARTS)*: the hard pharmacophore constraint; the swapped-in row becomes a SMARTS entry |
| `generative.ring_topology_moves` | Generative → **Ring topology moves** card: one SMILES in, every distinct one-step ring-size mutant out (contraction + expansion over each saturated ring; fused/spiro never touched) — the what-if tool behind the GA's topology mutation |
| `generative.scaffold_hop_crossover` | Generative → **Breed two molecules** card: two parent SMILES + children count + seed; bridge / ring-merge / linker-transplant cuts compete per draw; failed draws say so honestly |

- **The objective combobox finally carries every bundled objective**
  (nine entries): the three reference-driven ones above join the six
  window/blend objectives, each with its hint text, and the reference
  row swaps in exactly like the GA/RL parameter blocks
  (`grid_remove` collapse) — label "reference" or "SMARTS". The
  window values are honestly ignored where irrelevant (the hints say
  so).
- **The UI gained its reporting layer**: `File ▸ Save session report
  (PDF)…` composes `viz.save_pdf_report` over every live figure on
  screen — a title page with the census and timestamp, one page per
  figure, `close=False` keeps the on-screen figures alive. The
  registry (`_render_figure` feeds it) keeps one entry per figure
  host: a re-rendered card replaces its entry instead of accumulating
  dead references, and page titles dedupe ("parity plot (2)").
- **A latent session-restore bug fixed**: a restored RL method
  selection used to leave the GA parameter block visible (and
  vice-versa); the restore now re-syncs the swapped GA/RL frames AND
  the reference row after the comboboxes hold their restored values —
  the same discipline the ML tab's hyperparameter panel already had.

### Tests

- 25 new GUI tests (`TestReview122Gui`): the objective tuple/combo
  pin (nine kinds, mirrored), the factory contracts for all three
  reference objectives (real `score()` calls: aspirin similarity 1.0,
  corpus member novelty 0.0, benzene SMARTS hit), the reference row's
  hidden-by-default / swap / re-hide states (via `grid_info()`
  emptiness — the honest `grid_remove` check), the missing-reference
  and missing-SMARTS input errors, both surgery cards (piperidine's
  three mutants named, the aromatic no-move verdict, the 4/4 viable
  aspirin × ibuprofen litter, the parent/count guards), the session
  report (File-menu label, the no-figure info guard, registry
  tracking + in-place replacement, a real two-figure PDF written with
  the figures kept alive, title dedupe), the restore re-sync (the
  latent RL/GA bug fix pinned), and the docstring/home-guide round
  pins.
- The E2E probe gained the 100x-122 section (objective row swap,
  both surgery jobs, the guard + registry + real-PDF report path) —
  the live-card census is now 68.

## [1.2.60] — 2026-10-24

### Added — the round-6 full review (100x-121): UnitConverter + export_table + the deep learning curve

The standing complaint — "playground.py content is incomplete. Also
improve its user interface." — was re-audited a sixth time (the
rar/upload contracts re-proven 119/119 first, then the library-axis
audit re-run: 38 zero-hit symbols re-examined one by one). Two more
genuine user-facing gaps surfaced, plus three content/UI polish items
from the 120 hand-over list:

| Library capability | playground.py surface since 1.2.60 |
| --- | --- |
| `pharma.UnitConverter` | PK / PD → **Unit converter** card (row 7): dimension picker (mass / volume / amount / concentration / time) loads both comboboxes, value entry, `Convert` reports the result + dimension + conversion factor; the entries stay editable so an unknown or cross-dimension unit surfaces the library's own `UnitConversionError` — which names every supported unit by dimension |
| `viz.export_table` | **Every table save in the window** now routes through one shared dialog offering five formats — `.csv`, `.tsv`, GitHub-ready `.md`, records `.json`, standalone styled `.html` — the format follows the picked extension. All 11 save sites migrated (descriptors, RISM-KH, feature selection, NCA summary, ligand preparation report, docking results, ML predictions, DL history, GA leaderboard, Pareto front, and the shared `_save_frame_dialog` callers) |

- **Deep Learning tab — the size-vs-score learning curve**: the
  Training history card gained a *Learning curve…* button + CV-folds
  entry + its own figure host — `plot_learning_curve` over the FULL
  dataset (train + test re-joined), the estimator cloned per fold
  exactly like the sklearn registry models ("should we assay more
  compounds?" answered with a curve; the deep twin of the ML Results
  button from 1.2.59).
- **Docking — the parsed DLG cluster summary rides the verdict line**:
  after a successful *AutoDock4 single folder* run, the summary now
  carries `parse_dlg_summary`'s numbers — cluster count, the top
  cluster's lowest energy and size, and the best-populated cluster's
  (the columns the original rar exported to Excel).
- **The Ctrl+F card finder remembers its last query** — prefilled on
  the next dialog open AND persisted across restarts (the session
  file's new `finder` text key; older sessions simply lack it).
- **The home guide gained its missing Deep Learning row** — the "What
  lives where" card listed seven of the eight content tabs; the DL
  blurb (architectures, loss curves, the new learning curve, the CSV
  audit) completes it, and the PK / PD row now mentions the unit
  converter.

### Tests

- 21 new GUI tests (`TestReview121Gui`): the unit converter's default
  conversion / dimension swap / time defaults / unknown-unit /
  cross-dimension / bad-value contracts, the five-format save dialog
  (filetypes + real `.md`/`.html` writes + routing from the ML
  predictions, NCA and RISM saves + the no-frame guard), the DL
  learning curve (no-network guard + full job + bad-folds), the DLG
  summary verdict line (a real `CLUSTERING HISTOGRAM` table), the
  finder memory (session payload + restore + prefilled dialog), the
  home-guide Deep Learning row and the docstring round pin. The DL
  session-input pin grew 15 → 16 (the CV-folds entry).
- The E2E probe gained the 100x-121 section (unit converter + error
  contract, four export formats on disk, the DL curve embedded,
  finder memory, the DLG verdict) — the live-card census is now 66.

## [1.2.59] — 2026-10-24

### Added — the round-5 full review (100x-120): the last library stragglers gain GUI surfaces

The user's standing complaint — "playground.py content is incomplete"
— was re-audited a fifth time (the library-axis audit re-run: every
public symbol of every `pharmaml` module grepped against
`playground.py`). Seven user-facing capabilities were still without a
GUI surface; all closed in this release:

| Library capability | playground.py surface since 1.2.59 |
| --- | --- |
| `viz.plot_descriptor_distributions` | Chemistry → Feature factory → *Plot distributions* (histogram grid over the computed frame, title carries rows × columns) |
| `viz.plot_learning_curve` | Data & ML → Results → *Learning curve* (train-set size vs CV score over the last run's train+test rows; CV-folds knob 2–10; replaces the parity figure until the next train) |
| `viz.plot_table_comparison` | Utilities → Table compare → *Plot reconciliation* (the three-way overlap figure, labels from the picked table names) |
| `docking.parse_vina_config` | Docking → Vina config batch → *Parse config* (key→value dump in the card; centre/size/exhaustiveness copied into the Campaign runner's single-ligand form) |
| `biology.most_variable_genes` | Cell Lines → Expression table → *Rank variable genes (MAD)* (top-N MAD-ranked list with values, inspectable without a screen) |
| `pharma.effect_table` | PK / PD → PD model explorer — the sigmoid table now renders `effect_table`'s own parallel lists (parity with `sigmoid_emax_effect` pinned to the exact same values) |
| `docking.run_autodock4` | Docking → Campaign runner — the new **AutoDock4 single folder** engine mode (one ligand folder instead of the whole campaign; skip-completed resume checkbox exposed; engine-aware folder picker label) |

The remaining zero-hit symbols (38, down from 46) are the documented
non-goals: dataclass result containers riding inside composed calls
(`PKSummary`, `TableComparison`, `TrialResult`, …), internal plumbing
(`build_preprocessor`, `split_features_target`,
`correlate_with_response` inside `sensitivity_associations`,
`check_torsion_consistency` inside the audit scan), wrappers absorbed
by the class-based paths (`run_genetic_optimization`, the objective
classes behind `build_objective`) and `viz` export plumbing
(`save_pdf_report` inside `save_visual_bundle`).

### Added — the Ctrl+F card finder (the UI round)

A 65-card window needs navigation beyond tab hunting: **Ctrl+F** (or
`View ▸ Find a card…`) opens a live finder — type words from any
card's title or subtitle, Enter (or double-click) selects the card's
tab, scrolls the card into view and flashes its accent border for
1.2 s. Every card shell registers itself in a module-level index at
build time (`make_card` appends to `_CARD_INDEX`); the finder filters
the index live and resolves tabs through the live notebook so stale
entries never match. The shortcut is bound, advertised in the hint row
and pinned by tests together with every other accelerator.

### Fixed

- **Version drift**: `VERSION`/`pyproject.toml` said 1.2.58 while
  `pharmaml/__init__.py` and the web app's
  `SHIPPED_TOOLKIT_VERSION` still said 1.2.57 — all four carriers now
  agree on 1.2.59 (the web badge and download panel had been showing
  a stale version since the 1.2.58 bump).

### Tests

- 18 new GUI tests (`TestReview120Gui`): every new button's
  no-data-first guard and its full job path (figure embedded, table
  filled), the `effect_table` parity pins, the single-folder engine's
  registration/label/warning, the card finder's live-card census
  (≥ 55), dialog population, jump-select-flash-unflash, and the
  Ctrl+F hint-row + binding pins (the accelerator contract grew to
  six sequences).
- The E2E probe (`scratch/probe_gui_e2e.py`) gained the 100x-120
  section: learning curve, descriptor distributions, MAD ranking, the
  finder dialog + jump + flash, the single-folder engine label, the
  Vina-config parse round-trip. Suite: 2291 collected, 0 failures
  (2286 passed + 5 skipped).

## [1.2.58] — 2026-10-23

### Added — the full-review completeness pass (100x-119): the library's own surface, fully in the GUI

The user's standing complaint — "playground.py content is incomplete" —
was re-audited from the fourth axis: not rar-file coverage (119/119,
re-proven) but **library coverage** — the toolkit had grown to 1.2.57
while the GUI mapped an older library. The audit (every public symbol
of every module grepped against playground.py) found ~30 user-facing
capabilities with no GUI surface. All closed in this release:

* **PK / PD tab — four new cards:**
  * *Non-compartmental analysis (NCA)* — paste a concentration-time
    series: `validate_concentration_series` + `nca_summary` (AUC /
    AUMC / MRT / Cmax / Tmax / terminal t½), `vd_from_dose_auc` + CL
    when a dose is given, the shaded profile figure
    (`plot_concentration_profile`) and a summary CSV export.
  * *IV infusion & distribution* — `infusion_rate` (rate = CL·Css),
    `iv_infusion_steady_state`, `volume_of_distribution` (CL/ke),
    `bioavailability` (EV/IV AUC pairs) and the repeated-dosing
    approach-to-steady-state figure (`plot_dose_accumulation`).
  * *TDM trough adjustment* — `tdm_trough_adjustment` with the
    optional minimum-dose floor.
  * *PD model explorer* — the family beyond plain Emax:
    `sigmoid_emax_effect` (Hill slope), `inhibitory_emax_effect`,
    `linear_effect`; effect table + curve, and the reverse question via
    `ec50_for_target_effect` ("which EC50 reaches a target effect?").
* **Chemistry tab — two new cards + a figure:**
  * *Tautomer & 3D structure* — `canonicalize_tautomer` plus the
    multi-conformer 3D PDB writer (`smiles_to_pdb`, ETKDG + MMFF/UFF
    with its `PdbLigandReport`).
  * *MACCS key fingerprint* — `compute_maccs`: the 167 structural keys
    per pasted molecule.
  * The similarity scan card gained *Plot heatmap* — the labelled
    square heat-map (`plot_similarity_heatmap`) over references +
    library combined.
* **Data & ML tab — two new figures:**
  * *Calibration curve* — `plot_calibration_curve` (the reliability
    diagram) for the last trained classifier, with a bins knob.
  * *Plot leaderboard* — `plot_model_comparison` bars over the
    compare-models ranking.
* **Cell Lines tab** — the screen card gained *Replicate variability*
  (`replicate_variability`, always over the raw pre-collapse table)
  and *Compound summary* (`summarize_compounds`).
* **Docking tab — two new figures:** *Plot atom types*
  (`plot_receptor_atom_types` over the prepared receptor's census) and
  *Plot top ligands* (`plot_top_ligands` over the campaign scan, with
  a friendly "campaign scans only" guard for DLG/Vina tables).

### Changed — UI polish (the same review's second half)

* Every card title now carries a 3px accent strip (`make_card`),
  recoloured with the theme via `_retint_plain_widgets`.
* The home tab's hero offers quick-jump chips straight to every tab
  (the new `Chip.TButton` style).
* The status bar names the finished job and its duration
  ("training ridge — done in 3 s") instead of reverting to "idle".
* `Ctrl+PgDn` / `Ctrl+PgUp` cycle the notebook tabs; the hint row
  lists the new accelerator.
* The home guide's blurbs were rewritten to describe the grown tabs.

### Tests

* `tests/test_playground.py` +11 (217 total): the NCA card (default,
  with dose, mismatch guard), infusion (rate/Css/Vd + the F pair), TDM
  adjustment, the PD family table + reverse EC50, tautomer + MACCS,
  quick-jump + tab cycling, the accent strips, the new shortcut hint.
* The two status-bar tests were updated for the "done in" label.
* `scratch/probe_gui_e2e.py` grew a 100x-119 section exercising every
  new card end-to-end under Xvfb (figures asserted embedded).
* Full suite: **2192 passed / 2 skipped / 0 failed**; coverage probe
  verdict unchanged: 118 rar + 1 upload = 119/119 GUI surfaces;
  `ruff` and `mypy src/` clean.

## [1.2.57] — 2026-10-23

### Added — applicability domain analysis (the 2026-10 user upload, file #119)

The user's uploaded `Applicability_domain_analysis.py` — "is this QSAR
prediction trustworthy?" — is now first-class toolkit surface, the
round's answer to "playground.py content is incomplete":

* `pharmaml.ml.applicability` — `knn_applicability_domain` (the distance
  half: standardised descriptor space, mean kNN distance, percentile
  threshold), `bootstrap_uncertainty` (the ensemble half) and the
  composed `applicability_domain` returning an
  `ApplicabilityDomainResult` whose `to_frame()` is the upload's
  reliability table (`name, predicted, prediction_sd, knn_distance,
  inside_ad, low_uncertainty, reliable`).
* `plot_applicability_domain` (`pharmaml.viz`) — the upload's combined
  two-panel figure: the training distance density with its threshold
  and the distance-vs-prediction scatter colour-coded inside/outside
  the domain.
* CLI `ml ad` — both flows: `--data` (split, train, judge the held-out
  rows) and `--bundle + --train-data + --test-data` (the upload's
  verbatim flow), with `--out` writing the reliability table and
  `--plot` the figure.
* The playground's **Data & ML** tab gains the "Applicability domain &
  prediction reliability" card: model source (the last trained run's
  own split, or a loaded persistence bundle + its original CSVs), k /
  percentile / bootstrap-model knobs, per-member progress in the status
  bar, the SMILES-named reliability table beside the combined figure,
  and the upload's `Applicability_Domain_Results.csv` save button.
* The ML workflow notebook gains **§14 Trust** — the analysis on the §9
  regression split with one wild extrapolation, the verdict counts and
  the figure.

### Changed — four deliberate upgrades over the upload's own math

* **Estimator-agnostic**: the bootstrap ensemble is `clone(model)` of
  whatever fitted pipeline the caller hands in — the upload hard-coded
  30 fresh `CatBoostRegressor`s, so only CatBoost models could be
  audited; ridge, random forests, stacking ensembles, GP pipelines and
  binary classifiers (via the positive-class probability spread) all
  work now.
* **Half the fits**: every bootstrap member predicts on the query AND
  training rows in one pass — the upload ran two separate 30-model
  rounds (60 fits) to get the training-side threshold.
* **Deterministic**: resample `i` is seeded with `random_state + i` —
  the upload sampled unseeded, so two runs of the "same" analysis
  produced different reliability verdicts.
* **Self-distance excluded**: the training kNN query now drops each
  point's own zero first-neighbour distance — the upload's thresholds
  were systematically slightly too tight.

### Testing — the coverage contract extends to uploads (119 files)

The user's standing rule ("check that playground.py covers the entire
contents of the zip and rar files") now includes every file handed
over, not just the two archives: `scratch/probe_coverage_111_perfile.py`
gains an **uploads axis** — the upload maps to the AD card's GUI
symbols (all 8 hit) — and `scratch/probe_coverage_113_notebooks.py`
requires its notebook surface too. 40 new unit tests
(`tests/test_applicability.py`) pin the math, the error contracts and
the plot; 6 new GUI tests (`tests/test_playground.py` ::
`TestApplicabilityDomainGui`) drive the card end-to-end under Xvfb
including the bundle round-trip; `scratch/probe_ad_e2e_118.py` is the
one-shot full-workflow probe (last-run mode, kNN-only pass, bundle
mode, save, guard rails).

## [1.2.56] — 2026-10-21

### Added — the Hyperband bracket-best fold as a probed shipped module

The per-bracket best fold (1.2.55) lived inline in the 5.8k-line
`ml-playground.tsx`; it now ships as the pure module
`src/lib/hyperband-rungs.ts` (`buildRungPoints` + `RungPoint`),
following the `bayes-acquisition.ts` precedent: the component imports
it back, and the probe runs the code the browser runs, not a mirror.
`scratch/probe_rung_fold_117.ts` (bun) pins the whole contract —
winner-first on the API's ranked trials, exactly one bracket-best dot
per bracket owning the bracket max, ties keeping the first occurrence,
the winner legitimately carrying both flags, filter defaults, and
row-parity with the original quadratic algorithm on a 241-trial seeded
grid — and `tests/test_web_parity.py` runs it from inside the suite
(`TestHyperbandRungFold`), so a regression now fails CI.

### Fixed — NaN resources no longer draw a dot at x = NaN

The fold's filter asked `typeof resource === "number"`, which lets
`NaN` through (`typeof NaN === "number"`); a Hyperband trial scored at
a non-finite budget would have rendered a broken dot. Caught by the new
probe on its first run, fixed in the shipped module (the filter now
requires a finite resource), and pinned as an explicit contract —
while extracting the fold, the quadratic `find`-in-loop was also
replaced by an O(n) single pass (behavior-identical on the seeded grid,
proven row-for-row).

### Added — the walkthrough §9 heat-map figure and the docs/ guides join the failing contracts

Two extensions of the established guard families:
`tests/test_notebook_variants.py` gains
`test_walkthrough_similarity_heatmap_is_committed` — the §9 similarity
heat-map (1.2.53) must carry its committed `image/png` output, decoded
and magic-byte-checked with a minimum size, so the figure can never
silently vanish or become a stub; and
`tests/test_data_provenance.py`'s documentation-surface leg now scans
every `docs/*.md` guide (not just `cli.py`/README) for
`examples/data/...` promises — currently empty by design, so a
synthetic pattern-rot guard keeps the scanner honest.

## [1.2.55] — 2026-10-21

### Added — per-bracket best-CV chips on the Hyperband facet panels

The "Per bracket" facet view (1.2.54) now answers "which bracket found
it" at a glance instead of only flagging the panel that holds the global
winner: every facet caption carries a `best CV <score>` chip — emerald
tinted with the winner diamond when that bracket owns the global best,
stone tinted otherwise — and the bracket's own best dot wears an emerald
ring in BOTH views (the overlay legend gains a "bracket best" entry, the
tooltip explains "this bracket's own winner"). Pure fold over the trial
rows; deterministic tie-breaking on the first occurrence.

### Added — the provenance contract extended to the CLI and README

The notebook-source scan (1.2.54) now covers the other two documentation
surfaces that promise data files (`tests/test_data_provenance.py`, 3 new
tests): every `examples/data/...` path mentioned by the CLI's help
text/docstrings (`src/pharmaml/cli.py`) and by the README — prose
promises ("ships as ...") bind exactly like code blocks — must resolve
on disk AND be builder-owned. A stray file or a stale promise after a
rename now fails the suite.

### Added — drift lines name their owning builder function

`scratch/build_example_data.py --check` drift reports are actionable
pointers now, not just file names: every drift line carries
`(owner: build_qsar_multiclass_and_regression)`-style attribution, and
the new `STATS["owners"]` maps all 44 owned paths to their 11 builder
functions. The `@_owned` decorator tracks the running builder; the
planted-drift pytest guard pins the attribution on both the byte-flip
and the missing-file branches.

## [1.2.54] — 2026-10-21

### Added — the data-provenance round-trip check (the standing 100x-104 item)

"Every CSV the notebooks load == the file the builder writes" is now a
failing check, not a silent fix. Three legs, all pinned in
`tests/test_data_provenance.py` (5 tests) and re-runnable as one probe
(`scratch/probe_provenance_roundtrip_115.py`):

1. **Notebook-source scan** — every `DATA_DIR / "<name>"` reference in
   the eight executed notebooks' code cells resolves to a top-level
   `examples/data` entry (15/15 today); a stale path fails loudly.
2. **No orphans** — every shipped top-level entry is loaded by at least
   one notebook, so the inventory can never grow a file the demos
   forgot.
3. **Builder `--check`** — `scratch/build_example_data.py --check` is
   the non-mutating drift audit: byte-identity of every owned file (42
   re-generated in memory), the missing-file branch, an orphan scan
   (`disk == owned`, 43 files), the PDB size pins, and the full
   `verify()` content pins — exit 1 with the drift list. The
   planted-drift guard proves the gate fires on exactly one flipped
   byte without ever repairing it.

### Added — web-mirror parity guards inside the pytest suite

The TS-mirror probe pairs now fail CI, not just the probes
(`tests/test_web_parity.py`, 3 tests): `bayes-acquisition.ts` vs
scipy.stats.norm (both twins run — the python emitter, then `bun` on
the shipped module, then the cross-language verdict with the twins' own
tolerances restated) and `docking-configs.ts` vs `pharmaml.docking`
(the byte-parity probe plus the library twin). Skipped honestly in the
zip-only layout or without bun.

### Added — Hyperband bracket facet rows (web)

The rung scatter gains a "All brackets / Per bracket" view switch
(appears when > 1 bracket): the small-multiples edition renders one
mini panel per bracket on **shared axes** — the breadth-vs-depth trade
the η schedule makes becomes readable at a glance (bracket 0: many
configs at full budget; the last: few configs, cheap rungs only). Each
panel header carries config/promoted counts and the winner marker;
dashed full-budget + defaults references per panel; ARIA labels carry
the mode.

## [1.2.53] — 2026-10-21

### Fixed — the GPC/GP/HSVR refit-determinism pins flaked on exact-bit equality

`test_deterministic_refit` (GaussianProcessClassifier) failed once in
the 100x-114 baseline under parallel ruff+mypy load — unreproduced in
20+ re-runs loaded and unloaded. Root-cause class: `assert_array_equal`
demands bit-equality between two independent BLAS-backed fits, while
thread-scheduling may vary the reduction order by ~1 ULP. The three
sklearn-family refit pins (GP regression, GPC, HierarchicalSVR) now
assert round-off tolerance (`rtol=0, atol=1e-8`) — still catching any
real seed/state leak (an O(1) shift) — with the provenance comment and
`scratch/probe_gpc_flake_114.py` (40 paired refits, plain and under
load) documenting the hunt.

### Added — EI/UCB acquisition parity: the shipped web math vs scipy

The 100x-113 GP-band overlay's Bayesian-optimisation math moved from
the 5.3k-line component into `src/lib/bayes-acquisition.ts` (pure,
self-contained — and its Φ doc comment now correctly credits
Abramowitz–Stegun **26.2.17**, the direct polynomial, not 7.1.26 erf).
New paired probes `pharma-ml-toolkit/scratch/verify_acquisition_
parity_114.py` + `scratch/verify_acquisition_parity_114.ts` (the TS
twin imports the SHIPPED lib module) verify against scipy.stats.norm:
Φ ≤ 7.294e-8 (inside the 7.5e-8 AS bound claim), φ ≤ 5.6e-17,
EI ≤ 3.5e-7, UCB exact — guard branch included.

### Added — Hyperband per-trial rung scatter + the walkthrough §9 heat-map

- The tuner result card grows the rung scatter the leaderboard table
  could not show: every Hyperband evaluation as a budget-vs-score dot,
  coloured by rung (stone → amber → teal → emerald), promoted dots
  solid with a ring, culled dots faint, the winner a haloed diamond,
  dashed references at the defaults CV and the full budget; custom
  tooltip + ARIA summary (the 100x-102 candidate, closed).
- Walkthrough §9's similarity scan now ends in the PICTURE:
  `compute_fingerprints` → `tanimoto_matrix` → `plot_similarity_heatmap`
  over the two reference drugs plus aspirin's three nearest library
  hits (the standing "tables only" gap, closed).

## [1.2.52] — 2026-10-21

### Fixed — the ensemble builders were documented as `ml.*` but never exported from the package

`docs/COVERAGE.md` rows say `ml.build_stacking('classification', [...])`
and `ml.build_voting(...)` — the same "documented as `pkg.*`, importable
only via the submodule" defect the chem package had for
`similarity_scan`/`top_matches` (fixed in 100x-110). Both builders now
export from `pharmaml.ml` (imports + `__all__`), with a regression test
pinning the package-level identity (`test_ensemble_builders_exported_
from_package`). `docs/INDEX.md` regenerated: 260 → **262 symbols**.

### Added — the notebook axis of the per-file coverage contract

The 100x-111 audit greps `playground.py` per rar file (the GUI axis, the
binding contract); the notebook axis was documented separately without
per-file evidence. New `scratch/probe_coverage_113_notebooks.py` reuses
the canonical FILE_MAP and greps every executed notebook's **code cells**
(markdown prose does not count — an API tour means import/call surface).
First verdict: 108/118 covered, 7 absent — the entire Stacking/Voting
family. Both gaps closed:

- **`complete_ml_workflow.ipynb` §11** gains a "Meta-ensembles" subsection
  that builds, fits and evaluates `build_voting` (soft votes, weighted)
  and `build_stacking` (out-of-fold + logistic/ridge blender) on the
  splits the notebook already trained single models on — classification
  *and* regression, the direct upgrade of the 7 ensemble rar files.
- **`pharmaml_walkthrough.ipynb` §9 "Data utilities"** tours the every-day
  chores on shipped demo data: `table_to_sdf`/`sdf_to_table`/`smi_to_csv`
  round-trips, `merge_tables`/`compare_tables`/`annotate_with_smiles`,
  `split_table_into_chunks` + `merge_table_folder` (split ↔ merge-back
  round-trip), `clean_numeric_frame`, `list_file_names`,
  `filter_approved_drugs`, `similarity_scan`/`top_matches`,
  `read_gpf`/`write_config_batch` (Ni receptor types included) and
  `scan_vina_directory` + the honest `run_vina_batch` engine gate.
  After: **118/118 — notebook-covered 115, absorbed 3, absent 0.**

### Added — pytest guards for the 100x-110/111 GUI surfaces (the three-round-standing open item)

The scratch probes (`probe_coverage_110/111.py`) drove the Utilities tab
cards and the docking coverage surfaces manually for three rounds;
`tests/test_playground.py` now pins them as
`TestUtilitiesCoverageGui` (9 tests: merge/reorder, split, compare/
annotate, approved filter, collect/list/distribute/cleanup, reindex,
converters incl. censored pIC50 honesty, similarity matrix/threshold/
top-k, clean-numeric-frame) and `TestDockingCoverageGui` (5 tests: GPF/
torsion/completion audit, single SMILES→PDBQT, Vina config batch with a
fake engine, batch GPF/DPF campaign + Ni receptor types both paths, Vina
results-folder scan) — **14 GUI regression tests**, all driven the way a
mouse drives it (picker stubs, real callbacks, drained job queue).

## [1.2.51] — 2026-10-21

### Added — the per-file audit's two closing surfaces: batch campaign configs and the Vina results-folder scan

The user asked for the coverage check **per file** ("each file of the
zip and rar"). The fresh per-file grep audit (118 rows,
`scratch/probe_coverage_111_perfile.py`) confirmed every file had a
playground surface — except two behaviours the family-level audit of
1.2.50 had absorbed too coarsely:

- **`docking.write_config_batch`** (new, campaign) — the batch loop of
  `others/Docking/gpfer.py`, `gpferreceptorni.py` and `dpfer.py`
  upgraded: one `r.gpf` + `r.dpf` pair per campaign sub-folder, the
  grid box computed from each folder's own ligand (padding / npts /
  spacing forwarded to `compute_grid_box`), a custom
  `receptor_types` override (the Ni-bearing `"A C HD N Ni OA SA"`
  string `gpferreceptorni.py` hard-coded), `skip_existing` for
  idempotent resumes, and honest per-folder statuses (`ok`,
  `skipped (r.gpf exists)`, `no ligand PDBQT`, `failed: …`) — one bad
  folder never sinks the batch. Exported from `pharmaml.docking`;
  7 tests in `tests/test_campaign.py::TestWriteConfigBatch`.
- **playground.py — Grid box & docking configs card** — new *Batch
  GPF/DPF for campaign…* button (per-folder report with counts and
  box centres) plus a *receptor types (blank = default)* field that
  also feeds the single-ligand path; the card subtitle/docstring and
  the home guide updated.
- **playground.py — Docking results card** — new *Scan Vina results
  folder…* button riding `docking.scan_vina_directory` (the
  `vina_reader.py` folder walk: best affinity per log, strongest
  binder first, crashed logs keep NaN rows and are reported, not
  fatal). The card's subtitle now counts four sources.

The ZIP side of the audit is re-verified standing: the downloaded
archive's `playground.py` is byte-identical to the working tree, zero
rar-content leakage (no Persian-named paths, no `ml-codes`/
`extracted` entries), `testzip` clean.

## [1.2.50] — 2026-10-21

### Added — full RAR coverage in playground.py: the Utilities tab and five closing cards

The one-by-one coverage audit of the original `کدهای ماشین لرنینگ.rar`
(118 files) had verified the **library/CLI** surface, but the **GUI**
still hid ten workflow families behind no card at all. This release
closes the gap — every script of the archive now has a playground
surface, and `docs/COVERAGE.md` gains the standing per-file playground
matrix so future rounds keep the guarantee:

- **Utilities tab** (new, five cards):
  - *Table merge & split* — `merge_tables` (the `CSV_merger.py`
    column-ordering contract), `merge_table_folder`
    (`filemerger.py`, always-write) and `split_table_into_chunks`
    (`Split CSV file.py`, streaming, flat memory).
  - *Table compare & annotate* — `compare_tables` (the
    `common_between_two_excel.py` / `notcommon.py` three-way
    reconciliation, name-normalised) and `annotate_with_smiles`
    (`smilesgetter.py`, `<column>_SMILES` companions).
  - *Approved drugs → report* — `filter_approved_drugs` over picked
    TSV/CSV exports with the CSV **and** `export_docx` Word report
    (the rar's `approved.py` bare .docx name list, upgraded to a
    bordered grid table; python-docx probed with an install hint).
  - *Campaign file tools* — `collect_files` (name ∪ pattern union,
    folder-named destinations), `list_file_names`, `distribute_files`
    (the `adder.py` bin/receptor push), `delete_named_files` and
    `cleanup_tree` (keep-list ∪ keep-pattern allowlist) — every
    destructive path dry-run by default with a counted confirmation
    before the real pass.
  - *Reindex by number* — `reindex_by_number` (all three
    `Prettier_extracted_dlgs.py` copies): sparse per-ligand rows
    left-joined onto the dense number range, gaps visible.
- **Chemistry tab**:
  - *Structure & table converters* — `table_to_sdf` (3D embed + UFF,
    every column an SD tag), `smi_to_csv`, `sdf_to_table`,
    `table_to_smi` and the batch `add_pic50_column` (censored
    `<10`-style entries respected, never fabricated) — the rar's five
    converter scripts in one combobox card.
  - *Library similarity scan* — the `Tanimoto calculator.py` /
    `Tanimoto-edited.py` pair as three modes: the reference × library
    matrix (`tanimoto_matrix`), the pairs-above-threshold scan
    (`similarity_scan`) and top-k (`top_matches`); pasted or
    CSV-loaded library, four fingerprint kinds.
- **Docking tab**:
  - *Grid, torsion & completion audit* — `read_gpf` (typed
    parameters of one GPF), `scan_gpf_ligand_types`
    (`gpfreader.py`), `scan_torsion_consistency` (`niggamatch.py` /
    `mismatch2.py`: TORSDOF vs active REMARKs vs BRANCH counts) and
    `check_docking_completion` (`checker.py`: OK / INCOMPLETE /
    MISSING_LOG per folder).
  - *Vina config batch* — `run_vina_batch` (the `vinar_multiple.py`
    story): one config docks every `.pdbqt` under a tree
    simultaneously across workers, per-ligand results table.
  - *SMILES → PDBQT (single ligand)* — `smiles_to_pdbqt`
    (`smilestopdbqt.py` / `smiqt.py`) with the ETKDG + MMFF + Meeko
    path and AD4 atom-type repair.
- **Data & ML tab** — *Drop non-numeric columns* + *Save cleaned
  CSV…* (`clean_numeric_frame`, the rar's `Non-numerical data
  remover.py`): target + keep-list survive, dropped names reported,
  one confirmation makes the cleaned frame the active dataset.

### Fixed

- **`chem.similarity_scan` / `chem.top_matches` are now exported from
  `pharmaml.chem`** — the functions existed (and were documented as
  `chem.*`) but the package `__init__` never re-exported them; the
  playground's new scan card found the gap.
- **The invalid naproxen SMILES** (`COc1cc2nc(...)`, unkekulizable)
  in the playground's default arrays and the docking trivial-name
  table is now the correct `COc1ccc2cc(ccc2c1)C(C)C(=O)O` — the
  SMILES-array default silently dropped one of its five drugs and the
  similarity-scan library one of its seven entries.

### Verified

- `playground.py --self-test` under Xvfb :99 — 10 tabs, no errors.
- `scratch/probe_coverage_110.py` — headless mouse-path E2E over
  every new card: 26 checks, all PASS (merge 5 rows + reorder, split
  3 parts, compare 1/2/1, annotate, approved 2 names, collect
  union 3, dry-run safety, distribute 3 copies, cleanup allowlist,
  reindex dense 4, SDF 3355 B round-trip, .smi BOM CSV, SDF→table 3
  rows, pIC50 2/3 censored-honest, matrix (2, 8) with the identity
  cell at 1.0, threshold scan 2 identity pairs, top-k sorted, GPF 9
  params, torsion 1 ligand, completion INCOMPLETE/OK/MISSING_LOG,
  single PDBQT written, fake-vina batch 2 outs + 2 rows, clean frame
  keeping SMILES).

## [1.2.49] — 2026-10-21

### Added — the playground round the screenshot asked for: SMILES-array docking, RISM-KH, ensembles, model bundles

Six user-visible upgrades landed together — the docking error of the
reported screenshot fixed at the root, the batch docking the request
described, the descriptor and ensemble families the request named,
the model save/reuse loop, and the RL advisory that turns a silent
empty leaderboard into an actionable message:

- **`prepare_protein(allow_bad_res=True)`** — when Meeko's templates
  cannot match residues (broken electron density, non-standard
  chemistry — the `PolymerCreationError: Template matching failed for
  ['A:972', ...]` of the screenshot), the offending residues are now
  deleted automatically and the build retried (up to six rounds), the
  library twin of Meeko CLI's `--delete_bad_res`. The strict path's
  `DockingError` names the residues and the fix; the report counts
  them (`bad_residues_removed`, `bad_residue_ids`) and the playground
  ships a ticked-by-default checkbox.
- **`dock_smiles_array(...)`** (`pharmaml.docking`) — the whole pasted
  SMILES array docked in one call: every SMILES embedded to 3D PDBQT
  (Meeko + RDKit, deterministic seed) into its own campaign folder,
  then Vina or the full AutoDock4 stack (receptor distributed, GPF +
  DPF written, autogrid4 + autodock4) run **simultaneously** across a
  `ThreadPoolExecutor` worker pool (subprocesses release the GIL);
  best affinities parsed from each engine's own output (Vina REMARK
  poses / the DLG cluster table) into one ranked DataFrame with a
  per-ligand progress callback. The Docking tab grows the matching
  card: paste the array, pick the engine, box, exhaustiveness and
  worker count, run — the ranked table lands in the results card and
  the session remembers the array.
- **RISM-KH solvation in the playground** — the Chemistry tab gains a
  dedicated card for the ten 3D-RISM-KH-inspired descriptors (SPC /
  TIP3P site models, temperature knob, per-descriptor plain-language
  meanings, CSV export) and the feature factory's 8th family
  checkbox: `rism` rides the same ranked top-K pipeline as the other
  seven families.
- **Voting + stacking are registry models** — `voting` and `stacking`
  join `MODEL_REGISTRY` (dual-task, RF + SVM + KNN defaults, weights /
  passthrough / cv grids, `vote`/`stack` aliases), so the model
  combobox, the leaderboard and the CLI all see them; the Data & ML
  tab additionally grows an **ensemble builder card** — tick any
  members, pick soft/hard voting or the stacking meta-learner, train —
  with the same held-out metrics, figure and report plumbing as every
  other model.
- **Model persistence in the playground** — a save half writes the
  last trained model (or ensemble) as the provenance bundle
  (`save_model`, now also recording the training **target** column so
  inference audits can drop it automatically); a load half reopens
  any bundle, shows its provenance (versions, timestamp, feature
  contract) and predicts on any CSV after the `validate_bundle_input`
  audit — with classifier probabilities and a predictions CSV export.
- **The RL honesty pass** — a `prior epochs < 6` run that samples no
  valid SMILES now explains itself (raise the epochs, 8+ is the CLI
  default) instead of showing an empty leaderboard; the spinbox
  default moves to the library default 8.

### Fixed

- `parse_typed_value` in the playground now parses comma-separated
  integer lists as tuples for `None`-defaulted parameters — the voting
  ensemble's `weights` knob is typeable by hand.
- The Docking tab's engine-binary pre-flight understands the
  SMILES-array card's `vina` / `autodock4` tokens (the same friendly
  install hint as the campaign runner).

## [1.2.48] — 2026-10-21

### Added — manual hyperparameters + live progress/ETA: hands on the knobs, eyes on the clock

The playground's Data & ML tab trained every model at its registry
defaults, and every long job (a hyperparameter search, a leaderboard
sweep, a deep-network fit) showed the same silent marquee. Both gaps
close together — the feature the user asked for and the visibility it
needs:

- **`train_model(model_params=...)`** — manual hyperparameters, pinned
  as-is on the estimator before anything is fitted. Keys are raw
  estimator parameter names; an unknown key raises `ModelError` naming
  the valid ones. **Pinned keys are excluded from any grid search**, so
  `tune=True` searches only the slots the user left blank — pin
  `alpha`, search `l1_ratio` around it — and with every slot pinned the
  search degrades gracefully to a single fit at the pinned values
  (`best_params` reports exactly what was fitted, pinned and found).
- **Playground — the "Manual hyperparameters" card**: pick a concrete
  model and its *registry grid knobs* appear as editable fields
  (defaults from the freshly built estimator, the searched grid shown
  beside each). **Blank keeps the default** — only edited keys pin —
  and the values persist per model with the session (one JSON
  variable, so switching back and forth never loses an edit). A
  pre-flight parse names the offending field before any job starts,
  tuple slots (`hidden_layer_sizes`) parse `(128, 64)` properly, and
  **Apply best** now prefills the card with the search winner. A "cv
  folds" spinner joins the train card (2–10, the fold count Train
  threads into `train_model`).
- **Progress + ETA plumbing**: a `work` function that accepts an
  argument now receives a progress reporter — the status bar switches
  from the marquee to a **determinate bar** with a live
  **"~ 42 s left"** ETA (elapsed/fraction extrapolation) and a note
  (`tuning ann — 3/8 · ~45 s left · best 0.712`). Three library hooks
  feed it: **`RandomSearch`/`Hyperband` `on_trial`** (one payload per
  trial; Hyperband's `total` is the exact bracket-schedule count,
  mirrored by `_hyperband_total_trials`), **`DeepMLP*.fit(on_epoch=...)`**
  (keyword-only, one payload per epoch — early stopping ends the stream
  early; not a hyperparameter, so `clone`/`get_params` never see it),
  and **`compare_models(on_model=...)`** (one call per leaderboard row,
  the longest job in the tab). Zero-argument jobs keep the historical
  marquee; the bar resets to it when the job ends.
- **Tests**: 25 new cases — 6 on the pin contract (untuned pin, unknown
  key, search-around-the-pin, all-slots-pinned degradation, `None`
  parity, compare's per-row callback), 3 on the tuner callbacks
  (payload shape, Hyperband total == real count, the schedule mirror),
  2 on the epoch stream (ordering/keys/early-stop parity, clone
  cleanliness), 14 on the playground (typed parsing incl. tuples,
  duration formatting, the injection rule, the panel's
  build/pin/switch/reset/bad-value flows, determinate-bar + ETA
  rendering, post-job reset, a live tune search reporting progress,
  Apply-best prefill). The `--self-test` now asserts all of it
  headlessly too.

## [1.2.47] — 2026-10-20

### Added — gaussian_process_classifier: the Laplace GP classifier, the 26th registry family

The classification half of the GP family ships one day after the
regression half closed its own release: a latent **σf²·RBF(ℓ) GP squashed
through the logistic likelihood**, posterior approximated by **Laplace's
method** (Newton mode-finding on the B = I + W^(1/2)KW^(1/2) Cholesky —
GPML Algorithm 3.1, sklearn-exact down to the iteration structure), and
predictive probabilities by the **Williams–Barber 5-erf sigmoid integral**.
No white-noise term: the logistic likelihood already absorbs the label
noise a WhiteKernel would model on the regression side.

- **Registry**: `gaussian_process_classifier` (task=`classification`,
  aliases `gpc` / `gp_classifier` / `gp_classification`), kernel-shape grid
  (RBF / Matern / RationalQuadratic) for tuning, binary labels on the
  single latent + sklearn's internal one-vs-rest for multiclass.
  `gaussian_process` itself stays regression-only (pinned by test).
- **Web mirror, numerically gated**: `fitGaussianProcessClassification`
  in `ml-math.ts` — the Newton loop is **bit-exact against sklearn**
  (Laplace lml Δ 0.0 on the parity dataset; the numpy reference itself
  matches sklearn to 0.0 in `scratch/verify_gpc_parity.py`), and the
  predictive probabilities land within **6.3e-8** of sklearn
  (`scratch/verify_gpc_parity.ts` + `verify_gpc_parity_ts.py`), the gap
  being erf's own A&S 7.1.26 approximation (|ε| ≤ 1.5e-7). Two subtle
  traps the gate caught: the predictive variance needs a **forward
  substitution only** (Σv² = rhsᵀB⁻¹rhs — cho_solve there squares the
  B⁻¹), and the parity harness must standardize features on train
  statistics before fitting sklearn (the web fitter's no-leakage
  contract puts ℓ in standardized-feature units).
- **Web playground**: GP classification in the model picker (kernel
  group), the task-aware **Learn kernel (Laplace lml)** toggle (grid
  search over (ℓ, σf) when the request leaves them unset), the honest
  probabilistic scores on the results card — **ROC AUC / log-loss /
  Brier** from the 5-erf probabilities — and a decision-boundary chart
  whose **dot opacity carries |p − 0.5|**: the classifier's own
  uncertainty surface, fading exactly where it hesitates.
- **Tests**: 7 new cases (registry wiring + regression refusal + the
  gaussian_process regression-only contract re-pinned, accuracy/AUC on a
  deterministic binary set, predict_proba well-formedness, deterministic
  refit, far-from-data hedging toward 0.5, kernel-grid tuning);
  registry count guards 25 → 26 (torch-hidden 23 → 24).

## [1.2.46] — 2026-10-20

### Fixed — the database audit: two real data errors cured, the whole inventory pinned

The shipped example-data inventory (13 CSVs + the docking campaign + 2 PDB
structures, 43 files) audited file-by-file with a **526-check independent
auditor** (`scratch/audit_database_103.py`) — gaps (missing cells, ragged
rows, coverage holes) and errors (invalid chemistry, impossible values,
duplicate identities, drifted pins) — plus **20 permanent regression
guards** (`tests/test_database_audit.py`, `TestStructuralIntegrity` /
`TestChemicalIntegrity` / `TestSemanticIntegrity` / `TestDockingCampaign` /
`TestPdbFiles`). Full report: `docs/DATA_AUDIT.md`.

- **`walkthrough_compounds.csv` — one invalid SMILES in all 180 rows,
  cured.** The metadata column carried the constant string `CCO(C)C` — a
  trivalent-oxygen structure (chemically impossible: RDKit refuses it) with
  zero diversity in the very column that demos "a user file's shape". The
  SMILES column now resamples the toolkit's own **12-molecule seed
  universe** (aspirin … lidocaine), drawn *after* the activity draw so the
  descriptor bytes and the label census stay byte-pinned; the builder gains
  a vocabulary pin (set equality + RDKit validity on every row).
- **`qsar_regression.csv` — the impossible endpoint, clipped.** The pIC50
  target ranged **-4.3 .. 14.7** — IC50 from ~21 *molar* to ~20
  *femtommolar*; 15 rows claimed IC50 above 1 M and 2 below 10 fM.
  `make_regression_table` now clips to the **assay-plausible window
  [3, 11]** (IC50 1 mM .. 10 pM), the encoding a real plate produces at its
  concentration limits ("reported as > max / < min, encoded at the
  boundary") — 58 rows clipped at 3, 22 at 11; the rationale lives in the
  generator docstring, and the window is pinned on both the file and the
  builder. Downstream notebooks re-executed: the ML/tuning/deep-learning
  regression sections now train on the honest endpoint.
- **`pk_profile.csv` — the venv-reset repr drift, repaired.** One file on
  disk had drifted one last-digit from its generator (the standing trap's
  residue; caught by the idempotent builder's write-if-changed count) —
  rebuilt byte-identical, AUC pins verified through the package (linear
  17.9607 / log-trapezoid 17.2536).
- **Designed quirks documented, not "fixed"** (the notebooks' audit/repair
  sections demo them; the tests pin their exact counts): the 2 planted
  invalid SMILES in `screening_library.csv`, the invalid + empty rows in
  `ligand_library.csv`, the duplicate assay replicate + missing LN_IC50 in
  `cellline_screen.csv`, the campaign's `08_benzocaine` MISSING_LOG +
  `10_nicotine` INCOMPLETE cases, and the real sub-100-Da molecules in
  `library_with_descriptors.csv` (pyrrole 67.09 — not an error).
- **The SQLite side checked too**: the web app's `db/custom.db` (Prisma
  scaffold, User/Post, 0 rows) is unused by any route — the playground is
  client-side by design; no gap, no error, documented here for the record.

## [1.2.45] — 2026-10-20

### Added — the 25th family ships with its uncertainty: gaussian_process, VIP readout, the search trajectory

- **`gaussian_process` (kriging) — the active-learning workhorse, both
  sides on the same day.** Library: `ConstantKernel·RBF + WhiteKernel`
  start kernel, hyperparameters tuned by marginal likelihood at fit
  time (one deterministic optimiser run, `n_restarts_optimizer=0`,
  `normalize_y=True`), `param_grid` offering the Matérn ν=1.5 and
  RationalQuadratic alternatives, aliases `gp` / `gpr` / `kriging`,
  regression-only by contract (the classification request fails with
  the house `ModelError`). **`predict_with_uncertainty`**
  (`pharmaml.ml.evaluation`) — the mean **and a per-row σ** for the GP
  (and any estimator whose `predict` accepts `return_std=True`); σ
  grows away from the training data, the honest extrapolation readout
  no point estimator can give. **`plot_gp_uncertainty`** (`pharmaml.viz`)
  — the parity scatter with ±1.96σ error bars and the band-coverage
  annotation; the desktop playground routes GP runs to it
  automatically. Web: `fitGaussianProcess` in `ml-math.ts` — the
  train-set standardisation, a hand-rolled Cholesky solve, the
  posterior mean and σ **with the observation-noise floor**
  (`Var(y*) = Var(f*) + σn²`), the log marginal likelihood, and a
  deterministic mini LML grid search when the request leaves the
  kernel hyperparameters unset (the mirror of sklearn's
  marginal-likelihood tuning); the train response gains a `gp` block
  (tuned `lengthScale`/`signalStd`/`noiseStd`, `logMarginalLikelihood`,
  `nSupport`, `kernelTuned`, `coverage95`, `medianSigma`); the fit
  chart becomes an **uncertainty band** (stacked-area ±1.96σ, custom
  tooltip), multivariate runs get ŷ ± 1.96σ whiskers on the parity
  chart. **Numerical gate vs sklearn** (fixed kernel, same spaces):
  predictions Δ 8.1e-10, σ Δ 4.8e-10, LML Δ 4.9e-8 — all within
  sklearn's default `alpha=1e-10` jitter (`scratch/verify_gp_parity.ts`
  + `.py`; the direct-numpy posterior agrees with the web fitter to
  1e-15). 14 new tests (`TestGaussianProcessFamily` — registry wiring,
  aliases, the classification guard, determinism, σ-grows-away-from-data,
  coverage calibration, non-GP rejection; `TestGaussianProcessUncertaintyPlot`
  — the viz block).
- **VIP scores surface** — the classic QSAR readout: the PLS fitter
  has computed VIP (mean-square-1 convention, VIP > 1 = influential
  descriptor) since 1.2.43, but the train route dropped them. The
  response now carries `vip { features, values, threshold: 1 }` and
  the importance card renders every descriptor's VIP bar with the
  **VIP > 1 threshold line** and influential/weak colouring.
- **RandomSearch per-trial trajectory** (the tune panel's missing
  visual): every drawn configuration as a dot (x = trial number in
  draw order, y = its mean CV score), the **running-best step line**
  in emerald (how fast the sampling converges), the winner emphasised,
  and the default-configuration CV as a dashed amber reference line —
  the honest bar every trial is judged against. Custom tooltip: trial,
  score, best-so-far, Δ defaults, the configuration. Hyperband keeps
  its ladder; RandomSearch finally has its form.
- **§11 registry-tour cumulative budget** — the ML workflow's tour
  gains a `cum fit_s` running column and one `(total)` row, with the
  economics print: the tour's 13.5 s fit budget, what a 5-fold CV
  sweep of the same families would spend (~68 s), and the pointer to
  the tuning notebook's Hyperband ladder as the machinery that cuts
  such multi-config budgets. The tour itself grew to **38 registry
  entries** (gaussian_process joined with zero edits — the
  self-updating guarantee holding for the third release running).

### Fixed

- Registry count guards caught up with the 25th family
  (`test_neural.py`: 24→25 entries, torch-hidden available count
  22→23) — they were the honest tripwire that flagged the interrupted
  landing.
- Two recharts typing drifts in the playground: the trajectory step
  line uses `type="stepAfter"` (this recharts version's prop), and the
  GP parity `ErrorBar` drops `isAnimationActive` (not in this
  version's typings).

## [1.2.44] — 2026-10-20

### Added — the data-leakage audit: guards, docs, and the registry tour

- **`docs/LEAKAGE_AUDIT.md`** — the one-by-one audit of every
  machine-learning file in the original rar (112 `.py` + 2 `.ipynb`):
  **27 scripts + the BBB notebook leak** (imputer + scaler
  `fit_transform` on the full table *before* `train_test_split` —
  `Classification/RF.py` lines 36–51 vs 54–60 is the representative
  anatomy), 11 scripts are clean (the archive's own pipeline-based
  best practice: `Huber.py`, `Lasso.py`, the `*-class.py` wrappers),
  and the BBB `.pkl` embeds no training rows (pickle disassembled) but
  never persisted its scaler — the deployment gap cell 6 patches by
  refitting "on the original training data". The doc also records the
  toolkit's immunity (pipeline fitted on train rows only) and the web
  mirror's fix (below).
- **`tests/test_ml_pipeline.py :: TestLeakageGuards`** — 4 regression
  tests pinning the no-leak contract: the split returns raw features;
  the fitted `StandardScaler.mean_` equals the train mean to 1e-12
  while extreme test rows (±1e6) produce honest extreme extrapolations
  (a leaky scaler would have absorbed the outliers — extremeness is
  the no-leak signature); the imputer's median is the train median;
  and blowing up the test rows after training cannot move the model's
  train predictions by a bit.
- **ML workflow §11 "The full registry — every family, one tour"** —
  the notebook-coverage fix the audit demanded: 13 of 24 registry
  families had no notebook demonstration (adaboost, catboost,
  elastic_net_logistic, extra_trees, gradient_boosting, lasso,
  lasso_logistic, lightgbm, pls, ridge, ridge_classifier, svm,
  xgboost). The tour walks `list_models(task, only_available=False)`
  for **both tasks**, fits every available family (37 entries) on the
  notebook's own §5/§9 splits via `train_model(..., tune=False)` (~7 s
  total), scores each on the same held-out rows, and lists unavailable
  families with their missing requirement — nothing vanishes silently.
  Self-updating: a family added tomorrow joins the tour without an
  edit; it doubles as a full-registry smoke test inside the executed
  notebook. Downstream sections renumbered (Interpret→§12 …
  Recap→§15), ToC and recap table updated.
- **`docs/COVERAGE.md`** gains the second audit axis — "do the
  notebooks and playground.py cover the toolkit": the playground's
  model combobox lists `list_models(task, only_available=True)`
  dynamically (all 24 families, zero maintenance), and the notebooks'
  family coverage is the §11 tour's guarantee.

### Fixed — xgboost refused string class labels (the registry tour's find)

- Fitting `train_model('xgboost', 'classification', ...)` on string
  targets (`'active'/'inactive'/'weak'`) died inside XGBClassifier with
  `Invalid classes inferred from unique values of y` — xgboost is the
  ONE registry family that requires integer labels (lightgbm, catboost
  and every sklearn family accept any dtype). The notebook §11 tour
  found it on its first execution: the demo classification table uses
  string labels.
- **The fix**: `_xgb_cls()` now wraps XGBClassifier in
  `_IntegerLabelClassifier` — a ClassifierMixin that fits a sorted
  LabelEncoder on y, fits the wrapped estimator on the encoded labels,
  and decodes `predict` output back to the ORIGINAL label space (so
  `evaluate_model`, reports and persistence see real labels).
  Hyperparameter grids address the wrapped estimator directly
  (`clf__n_estimators` keeps working — get_params/set_params forward
  transparently), and the mixin marks the pipeline as a classifier so
  the ROC-AUC scorer recognises it (without it every tuned fold scored
  NaN). 3 new tests: string-label fit/evaluate with decoded
  predictions, tuned scoring without NaN folds, and the historical
  integer-label path untouched. `docs/COVERAGE.md`'s stale
  "LabelEncoder handling inside make_train_test_split" claim is
  corrected to name the real mechanism.

### Fixed — web mirror: the rar's signature leak, removed from the CSV paths

- The browser playground's CSV paths pre-scaled features with
  **full-column** z-scores before the split
  (`train/route.ts` `runRegression`/`runClassification`,
  `ml-data.ts` for tune + compare) — the same train/test contamination
  the audit found in the rar, inherited into the mirror. Every fitter
  already standardizes internally on the rows it is given
  (`fitStandardizer(X)`), so the outside transform's only effect was
  contamination: kNN distances and RBF kernels lived in a space whose
  per-axis scale was set by full-data statistics — held-out rows
  literally resized the kernel.
- **The fix**: the outside z-score calls are deleted; CSV features
  reach the fitters in raw units, exactly like the generator paths
  always did. Generator-data responses are **bit-identical** (pinned:
  PLS R² 0.9121 / RMSE 0.4103 / CV 0.92 ± 0.0132); CSV responses
  change honestly — charts plot the raw column units,
  `datasetMeta.features` drops the `(standardized)` suffix (the
  coefficient cards keep `x̂ = standardized x`, truer than ever), and
  every CV fold now standardizes on its own training part (per-fold
  rigor a precomputed transform cannot express). The LEAKAGE RULE is
  documented at the top of `ml-data.ts` for future surfaces.

### Fixed — the downloadable ZIP is now build-machine-clean

- The one-by-one archive scan (every text entry of the 305-entry ZIP)
  found one leakage channel: **absolute build-machine paths**. Library
  deprecation warnings inside the executed notebooks' committed outputs
  printed their source as
  `~/.venv/lib/python3.12/site-packages/...`, and 20 `scratch/`
  driver scripts carried `~/...` run instructions. The scratch
  hints are now portable (`python scratch/foo.py`), the notebook
  outputs rewrite the site-packages prefix to the neutral
  `site-packages/`, and `reexecute_all_notebooks_64.py` performs the
  same rewrite automatically after every execution (the prefix computed
  from `sysconfig`, never hardcoded). A full-archive regex for
  `/home/<user>/` now returns zero matches.

### Changed

- Version carriers bumped to **1.2.44** (VERSION, pyproject,
  `pharmaml.__version__`, web `toolkit-version.ts`); all 8 executed
  notebooks re-executed and print 1.2.44; twins regenerated.

## [1.2.43] — 2026-10-20

### Added — `pls`: partial least squares regression, the QSAR staple the registry was missing

- **A 24th registry family** (`pls`, aliases `plsr` / `pls_regression` /
  `partial_least_squares`): sklearn's `PLSRegression` (NIPALS),
  **regression only** — the registry refuses the classification side
  with the standard typed error instead of failing inside fit/predict.
  Latent directions are chosen to maximize X–y *covariance* (not X
  variance alone — the difference from PCA regression), which is
  exactly why PLS is the classic first model for wide, collinear
  descriptor blocks with few observations, where OLS strains and even
  ridge only regularizes what PCA regression would mis-rotate. Param
  grid: `n_components` `[1, 2, 3, 5, 8]`; factory default 2 components
  with sklearn's internal `scale=True` z-scoring (mirrored by the web
  playground, which standardizes before its own NIPALS solve). The same
  registry/tuning/persistence pipeline as every other family
  (`train_model('pls', ...)`, `compare_models(models=[..., 'pls'])`,
  `pharmaml ml train --model pls` all work through the alias).
- **The web playground gains the mirror on the same day** (no mirror
  gap this time): a 16th regression family `fitPlsRegression` in
  `src/lib/ml-math.ts` — hand-rolled NIPALS PLS1 with rank-safe early
  stop, coefficients recovered exactly by evaluating the (linear)
  sequential score projection on the basis vectors, VIP scores
  computed for the record, `nComponents` slider + RandomSearch grid
  `[1..6]` + the honest "components (used)" echo (the count actually
  extracted — a single-feature matrix caps at one). **Verified
  numerically against sklearn on a shared collinear dataset:
  predictions match to 2.2e-15, coefficients (converted to the same
  space) to 1.1e-15.** PLS stays out of Hyperband deliberately —
  `n_components` is not a monotone resource slot — so the tune route
  rejects Hyperband mode for it with the same actionable message as
  every other resource-less family.
- Tests: 4 new cases in `tests/test_ml_pipeline.py` (registry wiring +
  notes, factory defaults + `scale=True`, fit quality on a collinear
  block where a 1-component model must already explain ≥ 0.85, aliases
  + param passthrough + the classification task guard); the
  registry-count guards moved 23 → 24 (`test_neural.py`, both the
  full-count and the torch-hidden available-count 21 → 22).

## [1.2.42] — 2026-10-20

### Added — `gradient_boosting`: Friedman's classic GBM, the mirror-gap closer

- **A 23rd registry family** (`gradient_boosting`, aliases `gb` /
  `gradientboosting` / `friedman_gbm` / `sk_gbm`): sklearn's
  `GradientBoostingClassifier` / `GradientBoostingRegressor` with
  `random_state=42` — shallow depth-3 trees fit the residual gradient,
  learning-rate shrunk, one stage at a time. Classification **and**
  regression, the same registry/tuning/persistence pipeline as every
  other family (`train_model('gb', ...)` works through the alias).
  Param grid: `n_estimators` / `learning_rate` / `max_depth`. The
  **web playground has spoken this family since its earliest rounds**
  (the JS fitter, the tune grids, the Hyperband "boosting rounds"
  resource) — the LIBRARY side was the gap; with 1.2.42 the
  mirror closes and the boosting set is complete on both sides:
  adaptive (AdaBoost, reweighted stumps) + gradient (this family) +
  the three engineered implementations (xgboost / lightgbm / catboost
  when installed). The `gbm` alias remains lightgbm's, as it has
  always been.
- Tests: 4 new cases in `tests/test_ml_pipeline.py` (registry wiring,
  seeded factories + Friedman defaults, fit quality on a nonlinear
  target, aliases + param passthrough + the `gbm`→lightgbm guard), the
  registry-count guard moved 22 → 23 in `tests/test_neural.py`, and a
  `compare_models` determinism pin (two identical calls → byte-identical
  metric columns and ordering; only `fit_time_s` — wall-clock — may
  differ; the same contract the web route pins for its JS mirror).

## [1.2.41] — 2026-10-20

### Added — `extra_trees`: extremely randomized trees, the classic tree family the rar never shipped

- **A 22nd registry family** (`extra_trees`, aliases `et` / `ert` /
  `extratrees` / `extremely_randomized_trees`): sklearn's
  `ExtraTreesClassifier` / `ExtraTreesRegressor` with
  `random_state=42, n_jobs=-1` — random thresholds on the FULL sample
  (`bootstrap=False`), classification **and** regression, the same
  registry/tuning/persistence pipeline as every other family
  (`train_model('et', ...)` works through the alias). Param grid
  mirrors the forest's (`n_estimators` / `max_depth` /
  `min_samples_split`). It is deliberately NOT an "upgrade" of any rar
  file — the coverage matrix marks it as the one classic ensemble the
  original collection was missing; bagging (RF), boosting (AdaBoost)
  and randomization (ET) are now all three shippable from one
  registry.
- **The web playground mirrors it on both tabs**: a 15th regression
  and 10th classification family, fitter included in
  `src/lib/ml-math.ts` (seeded `randomVarianceSplit` /
  `randomGiniSplit`: one uniform random threshold per feature, best-gain
  candidate wins — the honest mirror of sklearn's RandomSplitter),
  `extra_trees` tune grids in both RandomSearch spaces, and an
  Hyperband resource slot (`nEstimators`, "estimators") so successive
  halving can promote ET rung by rung. Verified live: reg R² 0.904 /
  CV 0.918 on the toy seed 7, cls acc 0.76 / CV 0.817; Hyperband
  ladder 10→50→150 trees.
- Tests: 4 new cases (registry wiring, bootstrap=False signature,
  fit quality, aliases + param passthrough) and the registry-count guards
  moved 21→22 (20 without torch); the suite is now 2,013 passed + 2
  skipped. Coverage matrix summary updated (22 registry models); README
  registry list mentions the family and its aliases.

## [1.2.40] — 2026-10-21

### Added — the verified-stack `requirements.txt` (install every library at the version the QA chain actually runs)

- **`requirements.txt` is now a pinned snapshot of the verified stack** —
  every core dependency ships as `name==exact.version` (numpy 1.26.4,
  pandas 2.2.3, scikit-learn 1.5.2, scipy 1.14.1, joblib 1.5.3, openpyxl
  3.1.5, tqdm 4.67.1), and every optional feature set is a commented
  block of exact pins: cheminformatics (rdkit 2026.09.1, mordred 1.2.0),
  docking preparation (meeko 0.8.0, gemmi 0.7.5), gradient boosting
  (xgboost 2.1.3, lightgbm 4.5.0, catboost 1.2.8), deep learning
  (torch 2.14.1, with the `--index-url https://download.pytorch.org/whl/cpu`
  line for the CPU wheel spelled out inline), reports (matplotlib 3.9.2,
  seaborn 0.13.2, fpdf2 2.8.5, python-docx 1.2.0) and the dev/QA stack
  (pytest 9.0.2, pytest-cov 7.0.0, mypy 2.4.0, ruff 0.17.1). The header
  explains the contract: `pyproject.toml` keeps the loose compatibility
  ranges, `requirements.txt` is the exact environment the 2,0xx-test QA
  chain runs green under (verified under Python 3.12).
- **`tests/test_requirements.py` (new)** — the pin file can never silently
  drift from the package metadata again: the core block is parsed and
  checked to (a) pin every dependency declared in `pyproject.toml` (and
  declare nothing extra), (b) satisfy each declared range, and (c) use
  `==` pins only. The optional blocks are checked to map one-to-one onto
  the `[project.optional-dependencies]` extras (every extra's pinned
  names cover its declared names).
- **README** — the installation section now leads with the pinned-stack
  recipe (`pip install -r requirements.txt` before/instead of the bare
  editable install for reproducible environments) and the download panel
  on the web explorer mirrors the same layered commands with the version
  numbers.

## [1.2.39] — 2026-10-20

### Added — the desktop Tune card speaks Hyperband too: the resource ladder from the family's own grid

- **`pharmaml.ml.tuner.resource_bounds` (new public API)** — the declared
  `[min, max]` grid of a hypermodel's resource slot, exported from
  `pharmaml.ml` and documented with doctests. The Hyperband ladder for a
  *shipped* hypermodel: `max_resource` is the grid's largest value (the full
  budget a plain Train runs at), `min_resource` the smallest — every rung
  stays inside the declared space and the final rung is (up to integer
  division) the full budget. Raises the actionable `ModelError` when the
  hypermodel declares no such slot (hsvr) or a non-`hp.Int` slot.
- **`playground.py` — the Tune card's Hyperband mode**: a `searcher` combobox
  (random | hyperband — the CLI's own tokens) plus an η spinbox (2–4,
  default 3). Hyperband successive-halves the family's resource slot —
  `max_iter` for the sklearn `ann`, `epochs` for the torch `dnn` — with the
  rung ladder sized from the family's own declared grid via
  `resource_bounds` (no hardcoded budgets; the trials spinbox is ignored and
  says so in its tooltip). The leaderboard grows a **resource** column
  (hyperband mode only) and the status line carries the epoch-economics
  story the notebook §5 and the web card tell: bracket count, the ladder
  (`resource max_iter 200–1000`), the honest budget line (`budget 3998 of a
  naive 6000 max_iter (33% saved)`) and the promotion rule. **Apply best**
  carries the winner's rung resource along (dnn prefills the Deep Learning
  tab's epochs from the winning rung); **Save trials CSV…** already carried
  the resource column (the CLI `--out` story). Guards: `hsvr` + hyperband
  pre-flights with the actionable message (no resource slot to halve — pick
  ann or dnn, or switch back to random) before any bracket starts; η bounds
  pre-flighted with the other settings.
- **Web — the HSVR trust-radius card's blend-weight overlay**: a “Blend
  weights” toggle (single-feature models only) draws each cluster's trust
  fade w(z) = 1 − |z − centroid| / R as a tent polyline on a secondary
  right-hand 0–1 axis — the linear blend the prediction actually computes,
  finally visible (full local weight at the centroid, the dashed band's edge
  is exactly where the local voice falls silent). Dots' hover titles gain
  their exact local weight (1-D for single-feature, 2-D when the plot is the
  whole feature space; projected models keep honest coordinates-only
  titles); the legend gains a tent glyph; multi-feature models get the
  honest “the fade depends on the whole feature vector” note instead of a
  misleading curve.
- Tests +8: `test_tuner.py` +5 (`TestResourceBounds`: the four shipped
  families' pinned grids, the missing-slot error, the non-Int rejection, a
  real Hyperband sized by the declared grid — the 333/999/1000 schedule —
  and the package export) and `test_playground.py` +3 (the hsvr+hyperband
  pre-flight, the ann hyperband end-to-end — 6 trials, the resource column,
  the budget line, the winner's `max_iter` — and the dnn hyperband → Deep
  Learning tab bridge) plus the searcher/eta defaults pinned into the
  existing defaults test. `+1` doctest (the `resource_bounds` examples).

### Changed

- The Tune card is retitled “Hyperparameter search” (was “… (random)”) with
  a subtitle covering both searchers; the session file carries the searcher
  and η inputs for free (the generic widget walk).

## [1.2.38] — 2026-10-19

### Added — the desktop playground joins the four-feature story: the Tune card + the batchnorm checkbox

- **`playground.py` — Data & ML tab: the “Hyperparameter search (random)”
  card** (open since 100x-87): the `ml tune --searcher random` story, mouse
  edition.  A keras-tuner-style `RandomSearch` over the same hypermodel
  families the CLI resolves (`ann` / `hsvr` / `dnn` — pinned to the CLI's own
  registry by a test so the two surfaces cannot drift).  Every trial is scored
  by mean CV on the SAME shuffled folds; the family's **defaults are scored
  too** under those exact folds (the number the search has to beat); the
  winner is refit and scored on the held-out 20 %.  The leaderboard card fills
  with the trial ranking (rank / trial / configuration / score / ± std, best
  first, failed trials last) and a one-line summary carries best vs defaults
  (the gain), the held-out verdict and the winning configuration.  **Apply
  best** carries the winner forward: `ann`/`hsvr` point the model combobox at
  the family; `dnn` prefills the WHOLE Deep Learning tab from the winning
  parameters (layers, dropout, learning rate, activation, **batchnorm**,
  epochs — plus the dataset and the target column) and switches to it, so the
  loss curves and the report pipeline run on the exact winner.  **Save
  trials CSV…** writes the full trials table (the CLI `--out` story).  Guards:
  hsvr on a classification target explains itself and starts no job; trials /
  cv bounds are pre-flighted; the zero-numeric-feature rule fires before any
  search starts (the 100x-62 convention).
- **`playground.py` — Deep Learning tab: the batchnorm checkbox** (the second
  half of the open item): `batchnorm=True` on all four torch estimators from
  one tick — with the estimator's own batch-size ≥ 2 guard surfaced as a
  pre-flight dialog, and the session file carrying the new input for free
  (the widget-tree ownership map picked it up automatically).
- **`pharmaml.ml.tuner.make_cv_splitter` (new public API)** — the tuners' fold
  protocol (`KFold(shuffle, random_state=0)`; stratified for classes, plain
  KFold fallback when stratification is impossible) exposed so a baseline can
  be scored under the EXACT folds the search used — the apples-to-apples
  contract the notebook §4, the web tune route and now the playground's Tune
  card all rely on; previously each surface re-implemented (or hardcoded) the
  rule, the classic drift trap.  Both tuners' `search()` now route through it
  (one implementation, zero behaviour change); exported from `pharmaml.ml`
  and documented with doctests.
- Tests +15: `test_tuner.py` +3 (the public splitter: delegation to the
  tuners' own protocol across regression / stratified / singleton-fallback /
  auto cases, argument guards, the package export) and `test_playground.py`
  +12 (the config-text renderer, the CLI-registry pin, the card defaults, the
  ann search end-to-end incl. Apply best, the hsvr classification guard, the
  before-any-search guards, the bad-settings pre-flight, the trials CSV
  export, the dnn → Deep-Learning-tab bridge, the batchnorm run + the
  singleton-batch pre-flight, and the numeric-feature guard for the search).

### Changed

- The Data & ML tab's leaderboard card is now titled “Leaderboard” with a
  dynamic status line that names what it currently shows (the registry CV
  ranking or the trial ranking) — two stories, one table.
- The Deep Learning session round-trip pin grows to 15 saved inputs (the
  batchnorm checkbox rides the generic save/restore machinery for free).

## [1.2.37] — 2026-10-19

### Added — the executed-notebook face of the four-feature round: `complete_tuning_workflow.ipynb`

- **`examples/complete_tuning_workflow.ipynb`** (plus its no-output twin): the
  seventh workflow notebook, walking the four 1.2.36 features in the order a
  modeller meets them — §1 data entry (the shipped `qsar_regression.csv`) →
  §2 **3D-RISM-KH descriptors** on the shipped 12-drug set (`seed_molecules.csv`),
  the ten water-scale columns with the surrogate contract told honestly
  (features for QSAR, not RISM-calibrated energies; the Gsolv = LJ + Coulomb
  decomposition, the dipole ranking) → §3 **`HierarchicalSVR` vs the global
  SVR** under the same split and seed (R² 0.38 → 0.62 on this table), the
  honest per-cluster diagnostics (three potency regimes: mean pIC50 7.6/3.9/5.1
  with their trust radii) and two parity scatters → §4 **`RandomSearch`** over
  `build_ann_regressor`: the HyperParameters collection/trial protocol, one
  shared seeded fold split for every trial, the apples-to-apples baseline
  (the hypermodel's defaults scored under the tuner's own folds: 0.623 →
  0.820, **gain +0.197 R²**), the leaderboard as a `plot_model_comparison`
  figure → §5 **`Hyperband`** over `build_deep_regressor` (`resource="epochs"`,
  η=3, max 27): the resource ladder committed straight from the trials frame
  (mean CV R² 0.251 → 0.539 → 0.841 across the 3/9/27-epoch rungs) and the
  epoch economics — 783 epoch-budgets spent of a naive 1863, **58 % saved** →
  §6 **batchnorm A/B** — same split, same seed, same capacity, only the flag
  differs; the honest verdict on a small standardised table (plain 0.941 vs
  batchnorm 0.916 — per-batch statistics add noise here; that is exactly why
  §5's hypermodel makes the flag searchable, and its winner indeed chose off)
  → §7 recap with the CLI/web cross-references.  Executed in ~20 s on a
  laptop CPU; every number is produced by the cells (seeded, deterministic);
  RDKit powers §2 and PyTorch §5-§6, both optional extras.
- **Guard test** (`test_tuning_workflow_quartet_is_committed`): pins the
  committed verdicts — the RISM table, the hsvr/global R² pair, the tuner
  gain line, Hyperband's savings line, the two A/B lines — so the demo can
  never silently become stale or un-executed (the §12/§9 convention, third
  member).
- **Registrations**: the strip/reexecute drivers and the
  `test_notebook_variants` guard family now cover eight executed notebooks
  (72 no-fabrication assertions); the README notebook tour gains the seventh
  workflow section and the tree/eight-count lines.

### Fixed — the sandbox venv had been reset between rounds

- The toolkit virtualenv lost its optional-dependency layer between rounds
  (the bridge answered `pharmaml.chem.features unavailable`): reinstalled
  `rdkit 2026.09.1`, `torch 2.14.1+cpu`, `meeko` (+ `gemmi`) and the editable
  `pharmaml` install; the web `/api/chem/features` bridge (the `rism_` family
  included) verified end-to-end afterwards.

## [1.2.36] — 2026-10-19

### Added — the four-request round: 3D-RISM-KH descriptors, HSVR, the keras-tuner, batch normalization

- **3D-RISM-KH-inspired solvation descriptors** (`pharmaml.chem.rism`): the
  ten quantities a 3D-RISM-KH analysis reports — solvation free energy
  decomposed into Lennard-Jones and Coulomb parts, water O/H site
  distribution peaks, first-shell coordination numbers, a hydration-
  exclusion volume, the Gasteiger dipole and the charge-hydration
  asymmetry — computed from a documented *pairwise tempered-KH surrogate*
  that runs in milliseconds: ETKDGv3 embedding (fixed seeds, MMFF94→UFF
  relaxation) + Gasteiger charges + Lorentz-Berthelot site potentials
  with a smooth dielectric ramp (1 at contact → 78.4 far away — the
  mean-field stand-in for the OZ-solve screening the surrogate cannot
  express) + the KH closure branches with the attractive Boltzmann
  exponent tempered at K=2.0 (bare point-charge H-bond contacts would
  reach e**33; real RISM water site peaks live at ~5-10 = e**K).  Site
  densities respect water stoichiometry (1 O + 2 H per molecule — a
  naive equal-density solvent is net-charged and its 1/d tails blew
  ethanol's Gcoul to -28,000 kcal/mol); the shell window anchors at the
  pair's g peak (the first g>=1 crossing sits in the 1.0-1.2 band on the
  LJ wall and closed the window instantly — benzene's Hw_cn1 was 0.0);
  the excluded-volume onset uses the steric (LJ-only) distribution
  (like-charge pairs keep the total-potential g < 0.5 out into the
  screened tail).  SPC and TIP3P site models; fully deterministic
  (repeated calls bit-identical); the Gasteiger dipole of ethanol lands
  at 1.62 D vs the experimental 1.69 D.  The same ten columns are the
  **8th feature-factory family** (`rism_` prefix) riding the
  10,000-column ranking/selection pipeline.  CLI twin: `chem rism
  --data mols.csv [--solvent tip3p] [--temperature 310]`.
- **HierarchicalSVR** (`pharmaml.ml.hierarchical`, registry key `hsvr`,
  aliases `hierarchical_svr`/`h_svr`): the two-level regression
  estimator for heterogeneous QSAR data — k-means local specialists
  (seeded, standardised space) blended with a global SVR, the local
  weight fading linearly from `alpha` at the centroid to 0 at the
  cluster's trust radius so far queries fall back to the global model.
  Full sklearn contract (clone/GridSearchCV/pickle), constant-target
  clusters answered by a stand-in constant (SVR's dual has no support
  vectors for a single distinct value), DataValidationError/ModelError
  guards throughout, `train_model("hsvr", ...)` and `compare_models`
  wired.
- **Keras-tuner-style hyperparameter search** (`pharmaml.ml.tuner`):
  `HyperParameters` with the `Int`/`Float`/`Choice`/`Boolean`/`Fixed`
  methods called *inside* the hypermodel (space-collection mode returns
  defaults and registers slots; trial mode returns the oracle's values),
  `RandomSearch` (seeded sampling, cross-validated scoring via any
  sklearn scorer or callable, failed trials recorded not fatal, the
  winner refitted on the full data as `results.best_estimator`) and
  `Hyperband` (the classic Li et al. brackets — ceil((s_max+1)·eta^s/(s+1))
  configs at max_resource/eta^s, top ceil(n/eta) promoted per rung, the
  resource hyperparameter overridden and clamped per rung).  Ready-made
  hypermodels for the whole neural axis: `build_ann_regressor`/
  `build_ann_classifier` (sklearn MLP, `max_iter` resource),
  `build_deep_regressor`/`build_deep_classifier` (PyTorch, `epochs`
  resource, `batchnorm` a `hp.Boolean` slot) and `build_hsvr`.  The
  `ml tune` CLI runs either searcher with `--out` trials table and
  `--save-best` bundle; `--task auto` flips non-integer float targets
  with > 10 distinct values to regression (the loader's class-count
  heuristic calls small numeric targets classification, and scoring
  'accuracy' on singleton float classes crashes every trial).  The
  StratifiedKFold splitter falls back to plain KFold when the smallest
  class is smaller than the fold count.
- **Batch normalization on the deep estimators**: `batchnorm=True` on
  all four torch estimators (`DeepMLP*`, `ResidualMLP*`) inserts a
  `BatchNorm1d` between every hidden linear map and its activation
  (residual blocks: fc1→bn1→act→drop→fc2→bn2, BN before the residual
  add); running statistics ride along in the portable NumPy weight dict
  and the rebuild-on-predict path reconstructs the layers from the
  architecture snapshot.  `batch_size=1` is rejected at fit time and
  singleton tail batches are skipped (BatchNorm1d statistics are
  undefined on single samples); the ANN registry note points BN users
  at the torch estimators (sklearn's MLP has no BN layers).
- Tests +108 (new `tests/test_rism.py` / `test_hierarchical.py` /
  `test_tuner.py`, a `TestBatchNormalization` class in `test_neural.py`,
  `TestChemRismCommand` + `TestMLTuneCommand` in `test_cli_features.py`,
  the 8-family updates in `test_features.py`; registry count pins
  20 → 21).  README gains the four feature sections + CLI lines; counts
  1853 → 1962.

## [1.2.35] — 2026-10-19

### Added — the rescore report story reaches the GUI (`Save report…`)

- **The playground's Consensus rescore card gains a "Save report…"
  button + an "include report.pdf" checkbox** (the `_ml_save_report`
  pattern, transplanted to the docking axis): the last rescore is
  stored as a 5-tuple (rescored frame, evidence columns, weights,
  direction overrides, source label) and the button rebuilds the
  EXACT `docking rescore --plots` bundle from it —
  `consensus_ranking.png` (always), `energy_distribution.png` (scan-
  shaped tables only, the CLI's skip pattern), the
  `consensus_breakdown` table, `report.html`, and the multi-page
  `report.pdf` when the checkbox is ticked. The guard chain mirrors
  the ML twin: require pharmaml + matplotlib → "No rescore yet"
  warning naming the remedy → the fpdf gate with its install hint →
  the folder picker; figures build under the forced-Agg backend inside
  the worker thread, never touching Tk. The bundle's subtitle carries
  the CLI-identical recipe line plus the source table's label.
- Tests +3 (`tests/test_playground.py`, a new
  `TestDockingRescoreReport` class live under Xvfb): the no-rescore
  guard (warning names the explainable leaderboard), the fpdf gate
  (the install hint, gated before the folder dialog), and the
  end-to-end flow on the example campaign — campaign scan → rescore →
  save with the PDF ticked → all five artifacts verified (both
  figures ride along because the scan table is folder-shaped), the
  "Report saved" dialog, the job label, the PDF's title page naming
  the evidence — plus the unticked default (a second save ships HTML
  only).
- README's playground Docking tour names the report button; test
  counts 1850 → 1853.

## [1.2.34] — 2026-10-19

### Added — the rescore joins the report story (`--plots` + the ranking figure)

- **`plot_consensus_ranking`** (`pharmaml.viz.docking_plots`, exported
  from `pharmaml.viz`) — the figure twin of
  `consensus_breakdown`'s leaderboard: horizontal bars sorted so the
  strongest 0–1 consensus sits on top, the value annotated outside the
  bar tip, identity labels from the frame's first column by default
  (`folder` for scans, `ligand` for enriched tables, `rank` for cluster
  tables — overridable via `identity_column`). NaN consensus rows
  cannot rank and are dropped; the validation contract mirrors
  `plot_top_ligands` (typed `DataValidationError` naming the missing
  column and the available set; the shared `top_n` guard; first-
  declared-wins ties via the stable sort).
- **`pharmaml docking rescore --plots [--pdf-report] [--out-dir]`** —
  the rescore command joins the `ml train --plots` story: the visual
  bundle (`report.html` + figures + tables) now ships for the consensus
  ranking too. The bundle always carries `consensus_ranking.png` (the
  new figure over the rescored table) and the `consensus_breakdown`
  table (csv + markdown, the explainable units riding along); scan-
  shaped tables (a `folder` column + energies) additionally get
  `energy_distribution.png`, while consensus-only tables skip it (the
  `feature_importance` skip pattern). `--pdf-report` implies `--plots`
  and writes the multi-page `report.pdf`; the printed recipe line rides
  along as the bundle's subtitle.
- Tests +6: `tests/test_viz.py` +3 (sorted order with NaN dropped +
  the axis label contract; `top_n` cut + default/explicit identity
  column; the four validation guards incl. the shared `top_n`
  contract) and `tests/test_cli_features.py` +3 (the ligand-identity
  bundle without the energy figure, the scan-shaped bundle with both
  figures, and `--pdf-report` implying `--plots` with a ≥3-page
  document whose title page names the evidence).
- README: the CLI quick-start gains the `--plots` rescore line; the
  viz section imports and demonstrates `plot_consensus_ranking`; test
  counts 1844 → 1850. `docs/INDEX.md` regenerated (237 → 238 symbols).

## [1.2.33] — 2026-10-19

### Added — the consensus breakdown reaches the last two surfaces: the GUIs

- **The playground's Docking tab gains a "Consensus rescore" card** (the
  desktop axis of the consensus story, closing the arc library
  1.2.31 → CLI 1.2.31 → notebook 1.2.32 → GUI): "Rescore table" runs the
  real `consensus_breakdown` over the loaded results table (campaign
  scan, DLG, Vina or any summary CSV opened through the card's own
  **Open summary CSV…** button — the CLI's `--data` twin) and replaces
  the results tree with the explainable leaderboard: the identity
  column, one 0–1 `<column>_norm` unit per evidence column and the
  consensus, sorted descending with NaN rows last. The verdict panel
  prints the CLI-identical recipe line (every column's direction and
  weight) plus the driving-evidence verdict for the winner — the
  per-row argmax over the `_norm` units. Evidence columns, weights and
  higher-is-better overrides are typed in three entries, prefilled from
  each loaded table (the toolkit's canonical evidence names in
  preference order: ligand efficiency first, then energy, then cluster
  population). The rescored frame becomes the results table, so **Save
  results CSV…** writes exactly the CLI's `*_rescored.csv` content.
- Tests +5 (`tests/test_playground.py`, live under Xvfb): the example
  campaign rescore (14 rows, `13_coumarin` wins at 0.684 on the cluster
  unit, the two incomplete folders keep NaN and sort last, the full
  rescored frame behind the save button), the DLG + ligand-efficiency
  rescore (rank 4 wins at 0.860 — the LE/energy units carry it, not
  rank 1), the weights recipe line (1,3 lifts the winner to 0.842 with
  both weights named), the three input gates (no table / empty columns
  / malformed weights, each naming its remedy before any job starts),
  and the summary-CSV round trip (the `--data` twin, reproducing §9's
  metformin verdict on a three-ligand enriched table).
- The verdict computation reads the winning row column-wise
  (`.iloc[valid.idxmax()]` per column): `.loc[scalar]` on the mixed-dtype
  frame upcasts the row Series to float64 and would render integer
  identities ("rank 4") as "4.0".
- README's playground Docking tour names the rescore card; test counts
  1839 → 1844.

## [1.2.32] — 2026-10-19

### Added — the consensus breakdown reaches the executed notebook (§9 demo)

- **`examples/complete_docking_workflow.ipynb` §9** now demonstrates
  `consensus_breakdown` alongside `consensus_rescore` (the notebook axis
  of the 1.2.31 round, mirroring how 1.2.28 demonstrated
  `validate_bundle_input` in the ML workflow): four new cells — the
  introspection story (the consensus says *who* wins; the breakdown says
  *why* — one 0–1 `<column>_norm` unit per evidence column next to the
  score), the explainable table (all 12 ligands, sorted by consensus,
  computed from the notebook's own §9 rescoring frame), the
  driving-evidence verdict for the top-ranked ligand (07_metformin tops
  the table at consensus 0.782 with `ligand_efficiency_norm = 1.000` —
  not the best energy in the table, which is the point), and the
  `pharmaml docking rescore` CLI-twin markdown with the recipe line
  naming every column's direction and weight. Landed via
  `scratch/patch_docking_notebook_83.py` (nbclient execution, MPLBACKEND
  popped; committed outputs verified by source pins + output pins + the
  fabrication gate + an AST audit that the new names bind nowhere else);
  §9's intro paragraph and the §12 recap's Rescore row now name
  `consensus_breakdown` too.
- Tests +1: `test_docking_workflow_consensus_breakdown_is_committed` in
  `tests/test_notebook_variants.py` (the 100x-78 guard pattern) — pins
  the demo's source shapes (the import, the `breakdown = …` call with
  the notebook's own three evidence columns, the per-row argmax) and the
  committed verdicts (the explainable-table header, the top ligand's row
  of units, the driving-evidence line); the module docstring's
  guard-family list gains its fifth bullet.
- README's docking-notebook tour now names the §9 breakdown demo
  ("ligand efficiency + consensus + the explainable breakdown — why the
  top ligand wins"); test counts 1838 → 1839.

## [1.2.31] — 2026-10-19

### Added — the consensus ranking becomes explainable (`docking rescore`)

- **`consensus_breakdown`** (`pharmaml.docking.scoring`) — the
  introspection twin of `consensus_rescore`: identical validation and
  normalisation (the shared `_normalised_evidence` core, so the two can
  never drift apart), but instead of collapsing the evidence into one
  Series it returns one `<column>_norm` unit column per evidence column
  plus the `consensus` — read a row across the `_norm` columns to see
  *which* evidence drove a ligand's rank. `consensus_rescore` itself is
  now a thin wrapper over the same core (byte-identical behavior).
- **`pharmaml docking rescore`** — the CLI surface the consensus story
  never had: reads a summary CSV (a `scan --out` or `vina-scan` export),
  combines the requested evidence columns into the 0–1 consensus, prints
  the leaderboard with the per-column `_norm` units plus a recipe line
  naming every column's direction and weight, and writes the rescored
  table (default `<stem>_rescored.csv`). `--weights`,
  `--higher`/`--lower` direction overrides and `--top` are supported;
  errors (missing column, non-numeric weights) go through the standard
  CLI error wrapper with exit 1.
- Tests +8: four library tests in `TestScoring` (per-column units +
  consensus values, agreement with `consensus_rescore` across weights/
  directions/NaN policy, direction override + zero-variance NaN
  preservation, the shared validation contract) and four CLI tests in
  `TestDockingRescoreCommand` (leaderboard + written CSV + recipe line,
  both direction overrides with effective ranking pins, `--top`
  truncation, error paths with exit 1).
- README's docking section gains the `consensus_breakdown` line and the
  CLI quick-start gains the `rescore` example.

## [1.2.30] — 2026-10-19

### Added — the persistence audit reaches the Deep Learning tab

- **Deep Learning tab, "Check CSV…" button** — the deep twin of 1.2.29's
  Data & ML audit: the PyTorch estimators (`DeepMLP*` / `ResidualMLP*`)
  satisfy the same sklearn contract — `n_features_in_` and
  `feature_names_in_` recorded at fit, joblib-serialisable — so
  `validate_bundle_input` audits them exactly like the registry models.
  After a training run, pick any CSV and the handler audits it against
  the last network before you predict: missing features, extra columns,
  reordering and version drift, every difference named, in a dialog
  (plus the `ml check` CLI twin line, target flag included).
- The training label is dropped automatically when the audited CSV
  still carries it (the GUI's advantage, same as 1.2.29), the bundle is
  a throwaway inside a temp directory that deletes itself, and the
  audit state (`_dl_last_run`) is set before the matplotlib guard so
  Check CSV also works when plotting is unavailable.
- Tests +4 (`TestDLCheckCSV`, the `TestMLCheckCSV` pattern): the no-run
  guard (remedy named), the cancelled picker (no job started), the OK
  verdict end-to-end on `qsar_regression.csv` (dialog title/body/
  CLI-twin/job-label pinned, `--target pIC50` and the
  `DeepMLPRegressor bundle` named), and the FAIL verdict on a drifted
  copy (missing 2 / extra 1, all named). All four passed on the first
  run, after a manual Xvfb smoke of both paths.
- README's Deep Learning bullet gains the audit sentence; the web
  download panel's playground card mentions both tabs.

## [1.2.29] — 2026-10-19

### Added — the persistence audit reaches the playground (Check CSV…)

- **Data & ML tab, "Check CSV…" button** — the GUI axis of the round-trip
  audit 1.2.27 shipped (`validate_bundle_input` / `ml check`) and 1.2.28
  demonstrated in the executed notebook: after a training run, pick any
  CSV and the handler audits it against the last model before you
  predict — missing features, extra columns, reordering and version
  drift, every difference named, in a dialog (plus the `ml check` CLI
  twin line, target flag included).
- The GUI's advantage over the CLI: it *knows* the training target
  column's name, so the label is dropped automatically when the audited
  CSV still carries it — re-auditing the very CSV the model was trained
  on works out of the box (the CLI needs `--target` for that).
- The bundle itself is a throwaway: `save_model` writes it into a temp
  directory inside the worker thread, `validate_bundle_input` reads the
  verdict, and the temp directory deletes itself — nothing touches the
  user's tree. `_ml_last_run` gains the training target as a sixth
  element (the Save-report flow unchanged).
- Tests +4 (`TestMLCheckCSV`): the no-run guard (remedy named), the
  cancelled picker (no job started), the OK verdict end-to-end on
  `qsar_regression.csv` (dialog title/body/CLI-twin/job-label pinned,
  `--target pIC50` named), and the FAIL verdict on a drifted copy
  (feature dropped + feature renamed → `missing 2 feature(s): descr_000,
  descr_001` / `1 extra column(s) … descr_1`, label auto-dropped). All
  four passed on the first run, after a manual Xvfb smoke of both paths.
- README's playground bullet and the web download panel's playground
  card gain the Check-CSV sentence.

## [1.2.28] — 2026-10-19

### Added — the persistence audit reaches the executed notebook (§12)

- **`examples/complete_ml_workflow.ipynb` §12** now *demonstrates* the
  round-trip audit 1.2.27 shipped (library + `ml check` CLI) instead of
  just linking it: after the save/load/score cells, four new cells run
  `validate_bundle_input` against the very bundle the notebook just
  wrote —
  - the **OK verdict** on the training frame itself (the contract holds);
  - the **drift demo** on a derived copy of the notebook's own shipped
    CSV (one feature dropped, one renamed, one brand-new column — the
    three ways real data drifts), with the FAIL verdict naming every
    difference: `missing 2 feature(s): descr_00, descr_01` /
    `2 extra column(s) … renamed_feature, batch_id`;
  - the **reorder trap** — the same names in a different order, sklearn's
    subtlest rejection, caught before `predict()` ever runs;
  - the **CLI twin** in markdown: `pharmaml ml check … --target activity`,
    exit-1-for-CI and the `--target` workflow note.
- All outputs are **committed** (the notebook is executed top-to-bottom
  with the shipped code) and **pinned** by a new drift-guard test
  (`test_ml_workflow_persistence_audit_is_committed` in
  `tests/test_notebook_variants.py`): a cell edit nobody re-executed, or
  a verdict that changed shape, fails the suite.
- The §12 intro markdown and the §14 recap's Deploy row now name
  `validate_bundle_input`; README's audit paragraph points at the
  notebook demo. The no-output twin regenerated
  (`scratch/strip_notebook_outputs.py`), sources pinned identical.

## [1.2.27] — 2026-10-19

### Added — the model round-trip audit (`validate_bundle_input` / `ml check`)

- **`validate_bundle_input(path, frame)`** (`pharmaml.ml.persistence`): the
  production failure mode this closes — a model saved months ago, a
  regenerated CSV, and a crash deep inside sklearn with a cryptic
  feature-count error.  The bundle's metadata already recorded the
  ordered feature names; the audit now checks a new frame against that
  contract *up front* and names every difference: missing features,
  extra columns the model has never seen, **reordering** (sklearn
  rejects a same-set-different-order frame even when the names match),
  the feature-count contract when names were never recorded
  (`n_features_in_` vs frame width), and **version drift** — the
  sklearn/pharmaml mismatches `load_model` only logged, surfaced in the
  returned `ModelInputReport` (frozen dataclass; `ok` + the column
  buckets + human-readable `messages`; drift is always a warning, it
  never flips `ok`).  `str(report)` renders exactly what the CLI
  prints.
- **`pharmaml ml check --model x.joblib --data new.csv`**: the CLI twin
  — prints the bundle's provenance block (name, task, trained-at,
  pharmaml/sklearn versions, the feature preview) and the audit
  verdict; **exit 1 on a failed audit** (CI-friendly: run it between a
  regenerated CSV and a predict step).  `--target <col>` drops the
  training label first, so re-auditing the very CSV the model was
  trained on works out of the box — and when the audit fails with
  extra columns, the output ends with a hint pointing at that flag.

## [1.2.26] — 2026-10-19

### Added — the report bundle reaches the playground (the PDF arc closes)

- **Data & ML ▸ "Save report…"**: the GUI twin of the CLI's
  `ml train --plots` / `--pdf-report` pair.  After any training run,
  the button rebuilds the *same* visual bundle the CLI writes —
  `report.html`, the confusion/ROC/class-distribution (or
  parity/residual) figures, CV fold scores, feature importances and
  the held-out metrics tables — into a folder picked with a native
  dialog; the *include report.pdf* checkbox adds the multi-page
  document (title/metrics page, one page per table, one page per
  figure).  Figures are built inside the worker thread under the
  package's forced-Agg backend, so the window stays responsive and no
  Tk canvas is ever touched from off the main thread.  Guards follow
  the house pattern: no run yet → "Train a model first"; a ticked PDF
  without fpdf → the exact `pip install 'pharmaml[reports]'` hint
  (new `fpdf_available()` probe helper); the success dialog + the
  status bar echo the file count and the target folder.

## [1.2.25] — 2026-10-19

### Added — the PDF document reaches the command line

- **`ml train --pdf-report`** and **`ml compare --pdf-report`**: the
  1.2.24 `save_pdf_report` surface, exposed where users actually meet
  the toolkit.  The flag *implies* the visual bundle (`--plots` /
  `--plot`) — `pharmaml ml train --data qsar.csv --pdf-report` writes
  `report.html` + figures + tables + **`report.pdf`** (title/metrics
  page, one page per table, one page per figure) into
  `<out>/report/` without a second flag; the stdout report gains a
  dedicated `pdf:` line pointing at the document.  Same story for
  the leaderboard: `ml compare --pdf-report` produces exactly three
  pages (title + leaderboard table + chart) next to the existing
  bundle.  Without the flag the output is byte-identical to 1.2.24
  (pinned by test).  A missing fpdf degrades through the standard
  CLI error path — `error: ... install 'pharmaml[reports]'`, exit 1.

## [1.2.24] — 2026-10-19

### Added — what the ZIP was missing (the follow-up to the coverage audit)

- **`save_pdf_report`** (`pharmaml.viz.report`): the multi-page **PDF
  document** — a title page with the metrics, one page per table
  (header row + rows, capped at 38 rows so the page layout holds),
  and one page per figure (rendered headlessly through Agg into a
  temporary png, embedded full-width).  This is the upgrade of the
  original archive's *signature output*: every model script ended by
  assembling an fpdf report with the classification report and the
  confusion/ROC images — and `fpdf>=1.7` has been declared in the
  `reports` extra since the first release, waiting for the function.
  Non-latin-1 punctuation is transliterated (`—` → `-`, `≥` → `>=`)
  because the built-in Helvetica only encodes latin-1 — a unicode
  metric name never crashes the report.  `save_visual_bundle` gains
  `pdf_report=True`, which writes `report.pdf` as the bundle's second
  artifact (after `report.html`); the ML notebook §13 now turns it on
  and lists the PDF in its artifact table.
- **`export_docx`** (`pharmaml.viz.report`): any DataFrame as a real
  Word `.docx` document — a bordered grid table with a bold header
  row, cells formatted like every other export (compact floats,
  `NaN`/`inf` never leak as raw `nan`).  Upgrades the legacy
  `approved.py`, which wrote a bare list of approved-drug names to
  .docx; `python-docx>=1.1` joins the `reports` extra.
- **`LICENSE`** (MIT): the pyproject declared
  `license = { text = "MIT" }` and the MIT classifier since 1.0 — the
  file itself never existed.  A "production-ready" package now ships
  its licence text.
- **`py.typed`** (PEP 561): the package is fully typed (mypy clean
  across 60 files) but downstream mypy users got no type info because
  the marker was missing.  Now `[tool.setuptools.package-data]`
  ships it, and `import pharmaml` type-checks for library users.

## [1.2.23] — 2026-10-19

### Added — the one-by-one RAR coverage audit (closes the last two gaps)

- **`docs/COVERAGE.md`**: every one of the 118 recovered files from
  `کدهای ماشین لرنینگ.rar` audited one by one and mapped to the
  PharmaML surface that upgrades it — 112 `.py`, 2 `.ipynb`, 1 `.pkl`,
  3 `.txt`. The audit surfaced exactly two functional gaps, both closed
  in this release; the remaining deltas are intentional absorptions
  (dataset-specific weights, literal `.docx`/`.mol2` outputs, the
  deprecated MGLTools bridge, empty placeholders), each with its
  reason stated in the matrix.
- **`lasso_logistic`** (ml registry, classification): the archive's
  `Classification/Lasso.py` was an *L1-penalized logistic classifier*
  (`penalty='l1'`, liblinear) — the registry previously covered only
  the regression flavour under that name. Grid: `C`.
- **`elastic_net_logistic`** (ml registry, classification): the
  archive's `Classification/Elastic net.py` was an elastic-net
  *logistic* classifier (`penalty='elasticnet'`, saga, `l1_ratio`
  grid) — same story. The registry is now 18 classical + 2 deep
  families.
- **Avalon fingerprints** (`chem.compute_fingerprints(kind='avalon')`):
  the archive's `Descriptor classification.py` / `Descriptor
  regression.py` feature unions folded 512-bit Avalon fingerprints
  alongside ECFP/RDKit/Morgan blocks; the kind is now first-class —
  bit (`GetAvalonFP`) and count (`GetAvalonCountFP`) flavours, the
  count path hashing straight to `n_bits` where the archive folded by
  hand. `as_counts=True` is now valid for morgan **and** avalon (the
  only two count-capable kinds); the rejection message for other kinds
  names both.
- The ML notebook §2 demonstrates both additions: the fingerprint cell
  computes 512-bit Avalon bits + counts and the §2 bullet documents
  the count-capable kinds.

### Added — replicate variability reaches the cell-line notebook (100x-72, completed)

- `examples/complete_cellline_workflow.ipynb` gains §2b: the
  `[1.0, 3.0]` vs `[1.9, 2.1]` teaching pair (same median 2.0,
  ten-fold different noise → `cv_pct` 70.710678 vs 7.071068), the
  shipped screen's single duplicate (MCF-7 × crizotinib,
  byte-identical rows → CV 0.0, collapsed median 3.712163), and the
  `add_cv=True` companion-column table (`LN_IC50_replicate_cv_pct`,
  NaN for singletons). The §10 recap gains the row. Drift-guard pins
  hold in `tests/test_viz.py` (source symbols + the five output
  pins).

## [1.2.22] — 2026-10-19

### Added — replicate variability surfaces the noise the median collapse discards (100x-71)

- **`replicate_variability(screen)`** (biology/response): one row per
  duplicated (cell line, compound) pair *per metric* — replicate count
  (valid values only; NaN entries never inflate it), median, mean, std
  (ddof=1), spread and `cv_pct` (std/|mean| × 100). The motivating
  defect: `[1.0, 3.0]` and `[1.9, 2.1]` both collapse to the *same*
  median `2.0` — until now the difference was invisible. `cv_pct` is
  NaN (never a fabricated `inf`) for single observations and zero
  means; NaN-keyed duplicate pairs are counted, mirroring the audit.
  A screen without duplicates returns an empty frame with the full
  column set.
- **`collapse_replicates(screen, add_cv=True)`** keeps the noise next
  to the answer: each metric column gains a
  `{metric}_replicate_cv_pct` companion (placed immediately to its
  right — the column-order bug caught during bring-up: positions must
  be recomputed against the *current* frame as earlier inserts shift
  later metrics). The companions are annotations, *not* metrics, so
  `response_matrix` / `summarize_compounds` consumers are unaffected;
  a pre-existing column with the CV name raises `DataValidationError`
  instead of silently overwriting. Default `False` — byte-identical
  behaviour to 1.2.21.
- **CLI**: `pharmaml biology response-audit --replicate-cv` prints the
  pre-collapse variability table between the audit and the
  per-compound summary (and a one-liner when the screen holds no
  duplicates). The planted demo pair is a good advertisement:
  LN_IC50 replicates 0.39/2.61 → CV ≈ 396 %, byte-identical AUC
  replicates → CV 0.0.
- Tests +11 (3 `TestCollapseReplicates` additions incl. the collision
  guard, 5 `TestReplicateVariability`, 3 CLI); a doctest pins the
  `[1.0, 3.0]` → 70.7 % CV example.
- Environment note (sandbox repair, no shipped-tree change): the venv
  had been partially reset — torch/rdkit/meeko/gemmi/mordred and the
  editable install were restored, numpy pinned back to 2.1.3 after
  mordred's resolver downgraded it, and mordred's numpy-2 incompatibilities
  (`from numpy import product`, `np.float`) were patched in
  site-packages to resurrect the six full-deps descriptor tests.

## [1.2.21] — 2026-10-19

### Added — the prune story closes in the flagship notebook (100x-70)

- **The generative workflow notebook gains the crowding-distance
  shortlist section** (§4, right after the Pareto-front cells): a
  markdown intro (the unwieldy-front problem — the measured 46 of
  8,108; why rank-truncation drops the corners), a code cell
  (`front.crowding_distance()` with the corner census, `prune(8)`'s
  honest audit, the 8-row table with the crowding column, and the
  crowding-encoded scatter), and a markdown verdict. The teachable
  pinned outcome: the front's strongest dominator (557 dominated
  molecules) does **not** make the shortlist — it sits in a crowded
  region of objective space; dominating many molecules and being an
  isolated trade-off anchor are different virtues.
- **§10 publish** now bundles the encoded figure (`pareto_pruned` —
  6 → 7 figures); **§11 recap** carries the prune sentence.
- The patch driver (scratch/patch_pareto_prune_70.py) follows the 59
  convention: insert-after-anchor + string patches + nbclient
  execution + twin regeneration + 24 source / 18 output pins + the
  fabrication gate; new bindings (distances / shortlist /
  fig_pruned) collide with nothing (AST-audited).
- test_viz notebook pins gain `crowding_distance`, `front.prune(8)`
  and the encoded-figure title.

## [1.2.20] — 2026-10-19

### Added — the crowding distance reaches the Pareto figure (100x-69)

- **`plot_pareto_front` encodes the NSGA-II crowding distance visually**
  (the viz axis 100x-67 shipped without): the trade-off curve is now
  pure connective tissue (no markers), interior front members render
  with marker size proportional to their isolation over the plotted
  pair (`34 + 140 x crowding` — a sparse-region anchor stands large, a
  crowded middle small), and the `+inf` boundary corners are amber
  diamonds that a `prune` selection always keeps. The legend names all
  four traces (`dominated` / `Pareto front (N)` / `interior — size =
  crowding` / `corner — +inf crowding`); degenerate pairs (no spread
  on either objective) fall to the floor size by construction. The
  CLI `--pareto-plot` artifact and the playground's Pareto card figure
  pick the encoding up unchanged (both call the same function).
- **Direct viz tests for `plot_pareto_front`** — the function was only
  exercised through the CLI's PNG-magic check before this round.
  `TestParetoFrontPlot` (5 tests, synthetic breakdown — no RDKit):
  the four-entry legend + trace structure (dominated 2 / interior 3 /
  corners 2 over the hand-derived fixture), the exact interior sizes
  ([160, 202, 188] — the pinned crowdings in x order), the diamond
  path + non-black facecolor of the corner trace, the two-member
  front (corners only, no interior trace), and the left-anchored
  title + objective-pair axes.

## [1.2.19] — 2026-10-19

### Added — the RL optimiser reaches the playground GUI (100x-68)

- **The Generative tab gains an optimiser selector** — "genetic
  algorithm" or "reinforcement learning (RL)". The GA parameter rows
  (population / generations / mutation / crossover / operator
  selection) and the RL rows (iterations / batch size / prior epochs /
  anchor weight) live in their own frames and swap via
  `grid_remove`/`grid` (the remembered grid position returns; the card
  never reshuffles). The random seed, objective and corpus controls are
  shared.
- **RL runs fill the same leaderboard card** — the ranked molecules,
  the `best … — SMILES` summary (tagged `RL · built-in corpus` / `RL ·
  seeded` / `RL · custom corpus`), and the Pareto front card on
  multi-objective recipes (RL fills the same per-objective breakdown —
  the CLI parity). Seeded mode anchors the on-the-fly prior's corpus
  with the user's molecules ahead of `DEFAULT_CORPUS` (the CLI idiom).
- **The four-panel prior-drift introspection figure** (100x-66's
  `plot_rl_introspection`) renders in the results card after every RL
  run — the GUI axis the round shipped without — plus a first→last
  history table (`_format_rl_history`: best/mean score, valid
  fraction, the drift with the teaching note, entropy, both sequence
  log-probs) replacing the GA operator statistics.
- **Dependency gating follows the card idiom**: RL requires torch (the
  install-hint dialog names it), validated inputs raise the standard
  Input dialog (negative anchor weight, non-integer iterations).
- Tests +6: the method-combo block swap (grid_info pinning), the
  negative-anchor input error, the pinned RL run (summary, leaderboard,
  history table with the exact iteration-0 zero, the 4-axis
  introspection figure, no dialogs), the RL multi-objective Pareto
  parity, and two `_format_rl_history` unit tests.

## [1.2.18] — 2026-10-19

### Added — Pareto pruning, NSGA-II crowding distance (100x-67)

- **`ParetoFront.crowding_distance()`** (from
  `pharmaml.generative`) — the NSGA-II crowding distance for every
  front member: per objective, members are sorted by value, the two
  boundary positions mark their members +inf (the trade-off corners
  NSGA-II always preserves), and interiors accumulate the normalised
  neighbour gap. A no-spread objective (max == min) contributes 0 to
  everyone — crowding measures *spread*, and a lone member or a fully
  degenerate front returns 0.0s. Deterministic (value, index) sorts;
  a pure function of the front.
- **`ParetoFront.prune(keep)`** — the `keep` most isolated members,
  ranked by crowding distance (corners first, ties broken by
  leaderboard rank) and re-emitted in leaderboard order. The
  selection is monotonic in `keep` (a bigger keep extends, never
  reshuffles); `keep >= len(front)` returns the front itself.
  A pruned front carries `n_front` (the un-pruned size), keeps every
  member's original `dominates` count (pairwise dominance cannot
  change with membership), and its `__str__` honestly reports the
  original non-dominated count plus the cut. Motivation (measured):
  a real three-objective GA over a 300-molecule population for 30
  generations leaves 46 of 8,108 molecules non-dominated — and
  rank-truncation would drop the very corners the front exists to
  show.
- **`ParetoFront.to_dataframe()` gains the `crowding` column** —
  the isolation ranking sits next to the domination counts, so the
  reader sees who earned their seat (+inf = a corner).
- **CLI `--pareto-keep N`** (with `--pareto`) — prunes the printed
  front to its most isolated members; the summary reports the
  original front size plus the cut (`pruned to the 8 most isolated
  of 46 by crowding distance`). Actionable typed errors for a
  missing `--pareto` and for `keep < 1`, following the `--pareto-plot`
  convention.
- Tests +17: `TestParetoCrowding` (14: boundary/interior/degenerate/
  identical/empty/singleton hand-derived pins, prune corners +
  monotonicity + audit carry-over + honest `__str__`, the pinned real
  GA run with its 5 inf corners surviving `prune(5)`) and 3 CLI
  tests (pruned table + honest summary, missing `--pareto`, `keep
  < 1`). The `to_dataframe` columns pin updated for `crowding`.

## [1.2.17] — 2026-10-19

### Added — RL prior introspection (100x-66)

- **`result.history` records the prior-drift diagnostics** every RL
  iteration: `drift` (the squared agent-vs-prior sequence
  log-probability gap — the exact term the anchor penalty is built
  from), `entropy` (the mean per-step sampling entropy), and both mean
  sequence log-probabilities (`agent_logprob`, `prior_logprob`).
  Iteration 0's drift is the float-exact `0.0` — the agent starts as a
  bit-exact copy of the frozen prior and takes its first gradient step
  only after the diagnostics are computed (pinned by test).
  The diagnostic is independent of the penalty: `anchor_weight=0`
  (vanilla REINFORCE) still records the drift it no longer pays for.
- **`plot_rl_introspection(result)`** (from `pharmaml.viz`) — the RL
  history figure API twin of `plot_operator_trajectory`: four panels
  over the shared iteration axis (score trajectory, agent-vs-prior
  log-probs, the drift with the anchor weight annotated in the
  y-label, the exploration entropy).  A `GeneticOptimizer` history
  (no drift fields) raises the actionable typed error.
- **CLI: `--plot` on `--method rl`** now saves the introspection
  figure — the flag's GA-only restriction is retired the same round
  the diagnostics land (`pharmaml generative run --method rl --recipe
  qed --plot drift.png`).  The GA path is unchanged.
- The mode-collapse brake is now measurable: probed at seed 7 over 8
  iterations, final drift 38.8 unanchored vs 0.74 with
  `anchor_weight=1.0` — the `test_heavy_anchor_brakes_the_drift`
  regression pins the ordering.
- Tests +8 (4 introspection invariants, 4 figure/CLI checks — one of
  them replacing the retired GA-only `--plot` error pin — plus the
  updated history-shape field pin); README bullet + generative/viz
  package docstrings; INDEX regenerated (+1 symbol).

## [1.2.16] — 2026-10-19

### Added / Fixed — the keyboard-shortcut hint row + a dead accelerator (100x-65)

- **A quiet hint row above the status bar** lists every global
  accelerator (Ctrl+O open · Ctrl+S save figure · Ctrl+Shift+S save
  session · Ctrl+Shift+R reset this tab · Ctrl+Q quit), so the
  bindings are discoverable without opening a menu.
  ``View ▸ Shortcut hints`` (a new checkbutton) or the row's own
  ``Hide`` button toggles it; the preference is a plain
  ``tk.Variable`` on the app, so the generic session machinery saves
  and restores it like any other input (``_restore_session``
  re-applies visibility after the variables load).
- **Bug fix: Ctrl+Shift+R did nothing.**  The File menu (and the
  module docstring) had advertised it as the "Reset current tab to
  defaults" accelerator since the feature landed (100x-54), but no
  ``bind_all`` ever existed.  Bound now — and the new
  ``TestShortcutHints`` suite pins all five advertised accelerators to
  real bindings, so an advertised-but-unbound shortcut can never ship
  again.
- 5 new tests: bindings pinned, hint text == SHORTCUT_HINTS, row
  geometry above the status bar, View-toggle/Hide-button visibility
  flips (pack_info based, no mapping needed), and the preference
  surviving a session round trip.

## [1.2.15] — 2026-10-19

### Fixed — notebook freshness: committed outputs now match the shipped tree (100x-64)

- **The staleness audit**: five notebooks' committed outputs still
  printed ``pharmaml 1.0.0`` (executed during the round-26 conversion)
  and the generative one ``1.2.9`` (round 44), while the shipped tree
  was 1.2.14 — the committed evidence predated the code that ships
  with it.  All seven notebooks re-executed top-to-bottom against
  current code (~3 min total); the substantive numbers proved stable
  (the ML notebook's diff was version lines, tmpdir names, timings and
  a timestamp — CV ROC-AUC 0.916 ± 0.019 identical), so the refresh
  is an honesty fix, not a numbers change.
- **A freshness guard keeps it that way**:
  ``test_executed_outputs_print_shipped_version`` (parametrized over
  all seven executed notebooks) pins the ``pharmaml <version>`` print
  in the committed outputs to the *shipped* ``__version__`` — a
  version bump now requires re-executing the notebooks
  (``scratch/reexecute_all_notebooks_64.py``, the new one-command
  driver that re-runs all seven and regenerates the no-output twins),
  so outputs can never silently predate the tree again.
- The cell line notebook's setup cell now prints its version like
  every other notebook (it was the only one that didn't — the guard
  requires it).
- README: the FDR claims tightened to the executed precision
  (rho=+0.978, FDR≈2.4e-13); the deterministic-examples bullet
  documents the freshness guard + driver convention.
- +7 tests.

## [1.2.14] — 2026-10-19

### Added — the deep learning workflow notebook (100x-63)

- **`examples/complete_deep_learning_workflow.ipynb`** — the sixth
  workflow notebook and the executed-notebook face of the Deep
  Learning tab (1.2.12): 26 cells, executed with all outputs and six
  inline figures committed (~15 s on a laptop CPU).  Where the cell
  line notebook only meets `dnn` through the `compare_models`
  leaderboard, this one walks the **direct estimator API** of
  ``pharmaml.ml.neural``: data entry → split → `DeepMLPRegressor`
  (the sklearn contract) → the per-epoch history record plotted by
  ``plot_training_history`` (best-validation marker + the
  early-stopping verdict) → parity/residuals → plain-vs-residual
  architecture comparison (same split, same seed) → the registry path
  ``train_model("dnn", ...)`` (with the wrapped-vs-direct R²
  difference explained) → `DeepMLPClassifier` + confusion matrix →
  history export as CSV → recap.
- The **no-output twin**
  (`complete_deep_learning_workflow_no_output.ipynb`) derives via
  `scratch/strip_notebook_outputs.py`, which gained the notebook in
  its tuple; the drift-guard in `tests/test_notebook_variants.py`
  grew with it (`VARIANTS` + the executed-notebook fabrication guard —
  7 notebooks × 9 banned tokens = 63 assertions, zero allowlist
  entries).
- Built by `scratch/build_deep_learning_notebook_63.py` (nbclient
  under the single-threaded env; `MPLBACKEND` popped first so inline
  PNGs survive — the 100x-43 lesson).  +14 tests.

## [1.2.13] — 2026-10-19

### Fixed — the numeric-feature pre-flight + the session round-trip pin (100x-62)

- **Zero-numeric-feature datasets now fail BEFORE a job starts.**  A
  CSV with only string columns (molecule, series, label) used to run
  the progress bar briefly and die mid-job with the estimator's
  ``X has 0 feature columns``; all three train flows (Data & ML train,
  the CV compare, Deep Learning) now pre-flight the dataset and warn
  up front — naming the dropped non-numeric columns and pointing at
  the Chemistry tab's feature factory as the remedy.  New shared pure
  helper ``numeric_feature_columns`` (the numeric/dropped split rule
  the work closures also use, so pre-flight and training can never
  drift apart).
- **The Deep Learning tab's inputs are pinned through a session
  round-trip**: all 14 ``_dl_*`` inputs save, and 12 mutated values
  restore verbatim onto a freshly built window — the generic
  save/restore machinery covers the tab for free, now provably.
- Probed and cleared: ``plot_roc_curves`` with a 420-class (ID-column)
  target renders in 1.3 s / +23 MB — no resource guard needed, unlike
  the confusion-matrix case fixed in 1.2.12.
- 5 new tests (``TestNumericFeatureGuard`` + the session round-trip).

## [1.2.12] — 2026-10-19

### Added — the Deep Learning tab: the PyTorch estimators in the playground (100x-61)

- **A new "Deep Learning" tab** (between Data & ML and Generative) puts
  the four `pharmaml.ml.neural` estimators behind one card of inputs:
  architecture (feed-forward `DeepMLP` / residual `ResidualMLP`),
  comma-separated hidden widths, activation, dropout, Adam's learning
  rate and weight decay, early-stopping patience, validation fraction
  and seed.  The worker thread trains on an 80/20 split and fills three
  cards: the metrics grid (R²/RMSE/MAE or accuracy/balanced-accuracy/
  MCC, epochs run, best validation loss, fit time), the parity/confusion
  figure, and the epoch-by-epoch loss curves with the early-stopping
  verdict on the title — plus "Save history as CSV…" exporting the exact
  per-epoch records.  15 new tests (`TestDeepLearningTab` +
  `TestDefaultTargetColumn`).
- **`plot_training_history`** (`pharmaml.viz`): the library figure
  behind the tab — training/validation loss curves, a dashed marker at
  the best-validation epoch (the weights actually restored) and the
  verdict line ("early stopped N epoch(s) after the best" / "still
  improving — more epochs may pay off" / "no validation split").
  Accepts the estimator's `history_` record list or an equivalent
  DataFrame; `None` validation losses plot as gaps.  5 tests +
  doctests.
- **Smart target-column default** (`default_target_column`): both the
  Data & ML and Deep Learning dataset cards now preselect the most
  likely target by walking from the right and skipping the SMILES
  column and unique-per-row string ID columns — loading
  `qsar_classification.csv` used to preselect `compound_id` (420 unique
  row IDs), silently training a 420-class classifier on row IDs.
- **`plot_confusion_matrix` large-matrix guard**: the figure size no
  longer grows linearly with the class count (a 420-class matrix once
  requested a 464-inch canvas) and per-cell annotations stop at 50
  classes — beyond that the matrix renders as a plain heat-map with a
  title note, instead of allocating 176k text artists and OOM-killing
  the process.  Regression test with a 120-class matrix.

## [1.2.11] — 2026-10-19

### Added — "Copy inputs as JSON", the footer's second affordance (100x-60)

- **Every content tab's footer strip gains a quiet "Copy inputs as
  JSON" button** next to the 100x-55 reset button: one click puts the
  tab's LIVE inputs on the clipboard as a small JSON snippet
  (`{tab, variables, families, texts}` — the session file's own field
  vocabulary, scoped to one tab), with the item count in the status
  line.  A share-lite for issue reports and quick hand-offs (the full
  File ▸ Export/Import session flow stays for whole-window state).
  4 new tests in `tests/test_playground.py` (live-value reflection,
  corpus-text inclusion, index-explicit isolation, field typing) plus
  the 100x-55 census test extended to both footer buttons.

## [1.2.10] — 2026-10-19

### Added — the Pareto front in the executed notebook (100x-59)

- **`examples/complete_generative_workflow.ipynb` gains the Pareto-front
  section** (three cells after the §4 de novo leaderboard): the dominance
  definition, the one-line audit + the full 26-row front table + the
  `qed × similarity` trade-off scatter for the pinned 613-molecule run,
  and the interpretation — rank 1 dominates 557 (as it must), the
  **deepest front member sits at leaderboard rank 197**, and the scatter
  title's "8 of 613" is the documented 2-objective projection of the
  3-objective front.  The §10 visual bundle ships the figure as its 6th
  entry (`pareto_front`) and the §11 recap bullet tells the rank-197
  story.  Driver: `scratch/patch_pareto_notebook_59.py` (insert +
  patch + nbclient execution + twins + 30 pins + fabrication gate);
  the no-output twin regenerated.  With this round the Pareto axis
  spans every surface: library (1.2.7), CLI (1.2.8), playground
  (1.2.9), notebook (1.2.10).

## [1.2.9] — 2026-10-19

### Added — the Pareto front in the playground (100x-58)

- **The Generative tab's new "Pareto front" card**: every multi-objective
  run (the new objective kind **"Multi-objective (QED × MW × TPSA)"** —
  the CLI's `druglike` recipe — plus the existing "QED + LogP blend")
  fills the audit line, the non-dominated table (rank / smiles /
  per-objective scores / domination counts), the deepest front member's
  leaderboard rank, an objective-pair scatter (grey dominated dots, the
  front highlighted and connected; the x/y pair is switchable and
  re-plottable) and a **Save front as CSV** export.  Single-objective
  runs get an explainer instead of an empty card.  On the pinned seed-42
  demo (population 12, 4 generations): "7 of 36 molecules non-dominated",
  deepest rank 20.  7 new tests in `tests/test_playground.py`
  (deterministic run, single-objective explainer, objective-pair
  re-plot, CSV export, no-run guards, objective-factory coverage).

## [1.2.8] — 2026-10-19

### Added — the Pareto front on the CLI (100x-57)

- **`pharmaml generative run --pareto`** (multi-objective recipes
  `druglike` / `novel` / `similar`, GA or RL — both fill the per-objective
  breakdown) prints the front after the leaderboard: the one-line audit,
  the deepest front member's leaderboard rank ("a trade-off the blended
  ranking hides" — on the seed-42 druglike demo the deepest sits at rank
  149 of 159), the front table (per-objective scores + domination counts,
  truncated at `--top` with an explicit note), and
  **`--pareto-plot PATH`** saves the trade-off scatter over the recipe's
  first two objectives.  Actionable typed errors follow the `--plot`
  convention: single-objective recipes (`qed` / `logp`) are rejected with
  the multi-objective recipe list, `--pareto-plot` without `--pareto` is
  rejected with the fix.  The epilog fence and README gained the example
  `pharmaml generative run --recipe druglike --pareto --pareto-plot front.png`.

## [1.2.7] — 2026-10-19

### Added — multi-objective Pareto front report (100x-56)

- **`pharmaml.generative.pareto_front(result)`** re-reads a
  `MultiObjective` run's per-objective breakdown and extracts the
  **non-dominated set** — the molecules where improving one objective
  necessarily sacrifices another.  The weighted-geometric-mean leaderboard
  hides these trade-offs: on the demo run (QED × novelty × LogP band,
  population 24, 8 generations, seed 11) 23 of 159 molecules are
  Pareto-optimal and the deepest sits at leaderboard rank 154.  The
  returned `ParetoFront` carries the front in leaderboard order,
  per-member domination counts, the examined/excluded tallies (NaN / ±inf
  rows are excluded — dominance is undefined for them),
  `to_dataframe()` and a one-line audit `__str__`.  A `objectives=`
  keyword projects the front onto a subset of the objectives (the
  2-objective front is never larger than the 3-objective one).
- **`pharmaml.viz.plot_pareto_front(result, x=..., y=...)`** renders the
  trade-off scatter: dominated molecules as muted dots, the 2-objective
  front over the plotted pair as a connected accent line, front size in
  the title.  Re-exported from `pharmaml.viz` (INDEX regenerated).
- 12 new tests (`TestParetoFront`): hand-built dominance contracts, the
  rank-1-is-always-on-the-front invariant over a 25-case random sweep,
  projection semantics, NaN/±inf exclusion, validation errors, a pinned
  real-GA run (23/159, deterministic twin), dataframe/audit contracts.

## [1.2.6] — 2026-10-19

### Added — per-tab reset buttons pinned in every tab footer (100x-55)

- Every content tab (Chemistry, PK/PD, Cell Lines, Docking, Data & ML,
  Generative) now carries its own quiet **"Reset this tab to defaults"**
  button in a NEW non-scrolling footer strip below the scrollable cards
  (`ScrollFrame.footer`), so the reset is always one click away without
  scrolling back up — Home, which owns no inputs, deliberately carries
  none.  The button is wired index-explicitly through the new
  `_reset_tab(index)` method (the `File ▸ Reset current tab to defaults`
  menu command and Ctrl+Shift+R now delegate to it too), so a tab can be
  reset even while another one is visible.  The quiet "Ghost.TButton"
  style is flat on the page background, muted until hovered, and follows
  the live theme like every other widget.

## [1.2.5] — 2026-10-19

### Added — shareable playground sessions + per-tab reset (100x-54)

- **`File ▸ Export session`** writes the exact session payload (the
  same format as the auto-saved file) to any path — mail it to a
  colleague; **`File ▸ Import session`** applies such a file onto the
  live window, including a theme switch when the imported session
  carries a different one.  Invalid files are rejected with a clear
  error dialog and never touch the live state; the two pure helpers
  (`export_session` / `read_json_file`) never raise.
- **`File ▸ Reset current tab to defaults`** (Ctrl+Shift+R) restores
  just the ACTIVE tab's inputs to their freshly-launched values,
  leaving every other tab untouched.  Ownership is decided by the
  widget tree (entries/spinboxes/comboboxes/checkbuttons/radiobuttons
  mapped to their linked variables; the feature-family checkboxes and
  the two pasted corpora included, the theme radio excluded), and the
  defaults are snapshotted right after the tabs are built — before any
  session restore can overwrite either side of the mapping.
- Eight new tests: the file round-trip, unwritable-path and
  junk-file contracts, export/import/reset live under Xvfb, and the
  ownership-map invariants.  Fail-first demonstrated via a file-copy
  round-trip (8/8 red on the pristine source).

## [1.2.4] — 2026-10-19

### Added — opt-in minimum-heavy-atom sanity gate for the BREED crossover (100x-53)

- **`scaffold_hop_crossover(..., min_heavy_atoms=3)`** rejects the
  chemically comic child class: at certain draw states aspirin x
  naproxen breeds `'OO'` (hydrogen peroxide) or `'CC'` (ethane) as
  bridge children — single-fragment, sanitisable, under the length
  cap, i.e. contract-conformant but with no medchem meaning (98 of
  the first 400 seeds, ~24%).  The gate counts heavy atoms (atomic
  number > 1) after sanitisation in both child assemblers, so every
  one of the seven recombination geometries is covered; a gated-out
  child is not an error — the bridge pairing loop tries the next
  pairing (seed 25: `'OO'` becomes a real 33-atom hybrid) and the
  operator returns `None` when nothing survives.
- **Default 1 = gate off**: every historical GA trajectory stays
  byte-identical (the full determinism suite re-baselined green).
  `GeneticOptimizer(min_heavy_atoms=...)` threads the gate into both
  crossover call sites (uniform and adaptive modes); the documented
  scope is the crossover offspring — the character-mutation fallback
  can still mint short strings by deletion, and selection pressure
  handles those.
- Plain and numpy integers are accepted; bools and values below 1
  raise (`ValueError` at the operator level, `DataValidationError`
  at the optimizer level).  Nine new regression tests pin the comic,
  the no-op default, both assembler gates, the validation surface and
  the end-to-end optimizer plumbing (a length-minimising objective
  converges to 22 tiny members by default, far fewer with the gate).

## [1.2.3] — 2026-10-19

### Fixed — the mordred descriptor frame was unusable (100x-52)

- **`compute_descriptors(descriptor_set="mordred")` now returns a
  frame you can actually use**: column names are mordred's own string
  descriptor names (`"MW"`, `"nH"`, ...) instead of mordred
  `Descriptor` *instances*. The old frame's column labels were
  callable objects, so `frame["MW"]` raised `KeyError` and even
  iterating `frame.columns` crashed inside pandas (pandas calls
  callable column labels *on the frame*).
- **Failed cells arrive as NaN**: mordred reports per-molecule
  failures as `Missing`/`Error` sentinel objects; they are now
  converted to NaN (mordred's own `fill_missing` convention) with one
  aggregate warning naming the cell counts, the Error/Missing split
  and a few example descriptor names — instead of object-dtype
  columns full of `Error` objects that no downstream ML path could
  consume.
- All-NaN descriptor columns stay that way on purpose (a real
  mordred verdict; the ML pipeline's SimpleImputer drops them), and
  the docstring contract was rewritten accordingly. Six new
  regression tests pin string column names, name-based access with
  pinned benzene values, the NaN conversion, ML-readiness of a mixed
  batch, and the once-per-call aggregate warning.

## [1.2.2] — 2026-10-18

### Added — playground session persistence + dark theme (100x-51)

- **Every playground input now survives a restart**: on close the
  window writes `~/.pharmaml/playground-session.json` (all entries,
  toggles, feature-family checkboxes, pasted SMILES corpora, the
  active tab, the window geometry and the theme) and restores it on
  the next launch. `File ▸ Save session now` (Ctrl+Shift+S) and
  `File ▸ Clear saved session` control it manually; a "session
  restored" note appears in the status bar.
- **`View ▸ Follow system / Light / Dark`** recolours the whole
  window: two named palettes drive every shade, the choice persists
  with the session, and clam's light state maps (readonly combobox
  fields, focus borders, scrollbar troughs) are pinned to the active
  palette so dark mode has no bright stragglers. Embedded figures
  keep the toolkit's white published-figure background on purpose.
- Sessions recorded on a different screen restore their size but drop
  an off-screen position; `--self-test` never reads or writes the
  session file (deterministic CI runs).

## [1.2.1] — 2026-10-18

### Fixed — viz contract guards and ML contracts (100x-49)

- **`top_n` is now one contract everywhere**: `plot_top_ligands` and
  `plot_feature_importance` accept `top_n: int | None` — `None` keeps
  every row, a positive integer keeps that many, and zero/negative or
  non-integer values raise `DataValidationError` instead of the old
  inconsistent behaviours (all rows / empty plot / all-but-one /
  raw `TypeError`).
- **An explicit `energy_column` that is missing from the scan frame now
  raises** a `DataValidationError` naming the requested column and the
  frame's actual columns — previously it was silently ignored and the
  default column was used.
- **Six raw-error corners now raise typed errors**: `plot_table_comparison`
  label/frame arity, `plot_similarity_heatmap` on non-numeric data,
  non-positive/non-integer `bins` in the two histogram plotters,
  non-numeric metric cells in `plot_model_comparison`, junk receptor
  atom-type mappings in `docking_plots`, and ROC/calibration plots on
  stub estimators without `classes_` (`ModelError`).
- **Duplicated column names are rejected early** by
  `plot_descriptor_distributions` and `plot_model_comparison` with a
  message naming the duplicates (previously: a misleading "no numeric
  columns to plot" / every ytick reading "model").
- **`ml.training` contracts documented and the silent all-NaN guard
  added**: all-NaN `cv_scores` now log a loud warning naming the fold
  scores (return values unchanged); the single-sample-fold `r2` NaN and
  the NaN/±inf-target behaviour of `train_model` are documented
  contracts.
- **`chem.descriptors`**: the mordred `BCUT2D` NaN-preservation contract
  is now documented on `descriptor_set`/Returns.

## [1.2.0] — 2026-10-18

### Added — the feature factory and the Docking tab

- **`pharmaml.chem.features`** (new module): the seven-family feature
  factory from the user's own ML pipeline, now a public tested API —
  RDKit property descriptors, hashed atom-pair counts, ECFP bits,
  Avalon counts, RDKit fingerprint bits, Morgan-full bits and
  Morgan-on-Murcko-scaffold bits: **10,800+ columns per molecule**.
  `compute_feature_frame` builds the matrix (dropping and reporting
  unparseable SMILES), `rank_features` scores every column (Random
  Forest / LightGBM-if-installed / variance when no target), and
  `select_top_features` applies the two-layer selection — per-family
  median-importance prefilter, then the top-K you ask for — with
  `selection_metadata` emitting the companion JSON.
- **Playground Chemistry tab — Feature factory card**: load a CSV (or
  paste SMILES, or use an example), choose the families, the ranker
  and **how many features to keep**, then export the selected-feature
  CSV (SMILES + target + features) and the metadata JSON — the exact
  artefacts the standalone scripts produced.
- **Playground Docking tab**: ligand-library → PDBQT batch preparation
  with a per-ligand audit tree; receptor preparation (PDB/CIF/PDBQT →
  cleaned receptor PDBQT) plus one-click distribution into every
  campaign folder; grid-box computation with GPF/DPF writing for a
  reference ligand; DLG / Vina / whole-campaign result parsing with
  ligand-efficiency columns; and an AutoDock4/Vina campaign runner
  that checks for the external binaries before spending a second.

## [1.1.0] — 2026-10-18

### Added — the graphical playground

- **`playground.py`** at the toolkit root: a single-file, tabbed Tkinter
  GUI that opens with `python playground.py` — no command-line knowledge
  required. Tabs cover every domain: chemistry converters, RDKit
  descriptor tables and Tanimoto similarity, PK/PD dosing calculators
  with plotted IV/Emax curves, dataset-agnostic ML training with parity
  and confusion figures, and the generative GA (adaptive runs render the
  operator-mix learning curve). Heavy dependencies are probed at launch;
  every feature that is missing one explains its exact `pip install`
  line. Long calculations run in a worker thread. `--probe`,
  `--self-test` and `--version` helpers support headless checks.
- **Cell Lines tab**: GDSC/CCLE-style drug-response screens — load a
  long-format (cell line, compound, metric) table, audit it, collapse
  replicates, pivot the sensitivity matrix, summarise per-compound
  potency, call sensitive/resistant cell lines, and run the full
  biomarker-discovery pipeline (normalise expression → impute →
  per-gene correlation with Benjamini-Hochberg FDR control) with the
  planted EGFR biomarker recoverable from the bundled example data in
  two clicks. Includes the `LN_IC50 ↔ pIC50` converter.

### Added — versioned downloads

- The download archive is now **version-stamped**: the file downloads as
  `pharma-ml-toolkit-v1.1.0.zip`, carries a `VERSION` file plus this
  changelog, and the web explorer's Download tab shows the live version.
- **Updating an existing copy needs no uninstall**: unzip the new
  archive over your old folder (`unzip -o pharma-ml-toolkit-v1.1.0.zip
  -d <your-folder>`). Because the install is editable
  (`pip install -e .`), the package picks up the new files immediately.
  Verify what you have with `cat VERSION` or
  `python playground.py --version`.

### Changed — adaptive GA operator trajectory (generative flagship arc)

- Adaptive genetic-algorithm runs record their per-generation operator
  mix in metadata (`operator_trajectory`); the new
  `plot_operator_trajectory` figure renders the learning curve; the
  executed notebook demonstrates it (published as
  `aos_trajectory.png/.svg`); `pharmaml generative run --plot <path>`
  saves it from the CLI; and the demo script gained an adaptive section.

### Fixed

- viz: duplicate-index boolean masks, empty-input guards, `t_half ≥ 1e17`
  underflow, `kind="stable"` tie handling, and bundle-name path escapes;
  SVG/PDF exports are now byte-reproducible across processes.
- ml: NaN/±inf guards in evaluation and preprocessing; classification
  targets with ±inf fail with actionable messages.
- chem: `[H]`-only SMILES rows and boolean descriptor weights rejected
  with clear errors instead of raising raw exceptions.

## [1.0.0] — 2026-06-13

Initial public release.

### The package (`pharmaml`)

- **chem** — unit/endpoint conversion (IC50/Ki ⇄ pIC50/pKi), RDKit
  descriptor and fingerprint tables, Tanimoto similarity.
- **pharma** — pharmacokinetics (NCA, half-life, accumulation), dosage
  design (loading/maintenance, BSA/weight, pediatric rules, renal and
  TDM adjustment), pharmacodynamics (Emax models).
- **biology** — cell-line drug-response screens (GDSC/CCLE style),
  omics normalisation and biomarker discovery with FDR control.
- **ml** — QSAR pipeline: preprocessing, model registry, tuned training
  with cross-validation, evaluation, persistence, leaderboards.
- **generative** — genetic-algorithm molecular optimisation with
  adaptive operator selection and a REINVENT-style RL optimiser.
- **docking** — ligand/receptor preparation (Meeko), AutoDock4/Vina
  batch execution, result parsing, rescoring and consensus ranking.
- **viz** — domain figures (parity, confusion, ROC, PK profiles,
  similarity heat-maps, docking rankings), table export and
  self-contained HTML reports.
- **cli** — `pharmaml` command-line interface covering every domain.

### The distribution

- 1,400+ pytest cases with doctest contracts, mypy-strict source,
  ruff-pinned lint, GitHub Actions CI (Python 3.10–3.12).
- Executed Jupyter notebooks (ML, docking, PK/PD, generative, cell-line
  biology and the every-domain walkthrough) plus no-output twins.
- Dockerfile, example datasets, and this explorer web app.
