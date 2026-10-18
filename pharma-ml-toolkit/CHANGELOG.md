# Changelog

All notable changes to the PharmaML toolkit are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/) — **minor**
bumps add features, **patch** bumps ship fixes.

The current version is also in [`VERSION`](VERSION) (one line, no
newline-stripping surprises: `cat VERSION`) and in
`pyproject.toml` → `[project] version`. The downloadable archive is
named `pharma-ml-toolkit-v<version>.zip`.

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
