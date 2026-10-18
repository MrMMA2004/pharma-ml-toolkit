# Changelog

All notable changes to the PharmaML toolkit are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/) — **minor**
bumps add features, **patch** bumps ship fixes.

The current version is also in [`VERSION`](VERSION) (one line, no
newline-stripping surprises: `cat VERSION`) and in
`pyproject.toml` → `[project] version`. The downloadable archive is
named `pharma-ml-toolkit-v<version>.zip`.

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
