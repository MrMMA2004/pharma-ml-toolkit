# Changelog

All notable changes to the PharmaML toolkit are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/) — **minor**
bumps add features, **patch** bumps ship fixes.

The current version is also in [`VERSION`](VERSION) (one line, no
newline-stripping surprises: `cat VERSION`) and in
`pyproject.toml` → `[project] version`. The downloadable archive is
named `pharma-ml-toolkit-v<version>.zip`.

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
