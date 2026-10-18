# Changelog

All notable changes to the PharmaML toolkit are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/) — **minor**
bumps add features, **patch** bumps ship fixes.

The current version is also in [`VERSION`](VERSION) (one line, no
newline-stripping surprises: `cat VERSION`) and in
`pyproject.toml` → `[project] version`. The downloadable archive is
named `pharma-ml-toolkit-v<version>.zip`.

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
