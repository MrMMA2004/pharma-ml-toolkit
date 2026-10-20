# PharmaML Toolkit

A **production-ready, typed, tested** Python toolkit for computational pharmacy, consolidating three
domains into one coherent package:

| Domain | What it does |
|---|---|
| **`pharmaml.data`** | **Bring your own data** — load user CSV/TSV/Excel files, auto-detect the target column, infer classification (2+ classes) vs regression, audit data quality |
| **`pharmaml.ml`** | QSAR machine learning — preprocessing, model registry (18 classical + **2 deep-learning families** + **hierarchical SVR**), cross-validated training (binary **and multiclass**), evaluation, persistence, keras-tuner-style hyperparameter search (`RandomSearch` / `Hyperband` over `HyperParameters` spaces, with the tuners' own fold protocol `make_cv_splitter` public for apples-to-apples baselines and `resource_bounds` sizing Hyperband ladders from the family's own declared grid) |
| **`pharmaml.biology`** | **Cell line data analysis** — GDSC/CCLE-style drug-response screens (load, audit, replicate collapse, LN_IC50↔pIC50), response matrices, sensitive/resistant calling, expression normalization, **biomarker discovery with BH-FDR** |
| **`pharmaml.pharma`** | Pharmaceutical calculations — unit conversion, pharmacokinetics (PK), pharmacodynamics (PD), dosage design, regulatory-approval filtering (DGIdb TSV) |
| **`pharmaml.docking`** | Molecular docking — ligand CSV import + protein/receptor preparation (PDB & mmCIF), grids **and DPF generation**, AutoDock4 & Vina batch workflows (incl. config-driven batches), result parsing, scoring, torsion QA, campaign file management |
| **`pharmaml.chem`** | Cheminformatics helpers — IC50/Ki ↔ pIC50/pKi conversion, RDKit descriptors, fingerprints, Tanimoto similarity, the eight-family **feature factory** (`compute_feature_frame` → 10,000+ columns per molecule, `rank_features`, `select_top_features` two-layer top-K selection; the 8th family is the **3D-RISM-KH solvation descriptor** set `compute_rism_descriptors`), SMI/CSV/SDF format conversion, tautomer standardization + 3D ligand generation (SMILES→PDB), ligand-table reconciliation & merging |

This package is the upgraded, re-engineered version of a flat collection of research scripts:
every module now has **type hints, Google-style docstrings, structured logging, custom
exceptions, input validation, and pytest coverage** (2022 tests).

---

## Design assumptions

* **Python ≥ 3.10** (union syntax `X | None` used throughout).
* **Core deps only**: `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib` — everything else
  (RDKit, Mordred, XGBoost/LightGBM/CatBoost, Meeko, gemmi) is **optional and lazily imported**;
  missing packages raise an informative `DependencyError` instead of crashing at import time.
* **Runs anywhere**: local, Docker (`Dockerfile` included), or CI (GitHub Actions workflow included).
* **Deterministic examples**: the executed notebooks run entirely on the shipped
  `examples/data/` inventory (43 files — every input table, the demo receptor and the demo
  docking campaign; see the tree below), regenerated deterministically by
  `scratch/build_example_data.py`; the example scripts keep their fixed seeds — no external
  datasets required. A pytest freshness guard also pins every executed notebook's
  committed outputs to the *shipped* version (the setup cell prints
  `pharmaml <version>`), so the committed evidence can never silently predate the
  tree that ships it — after a version bump, `scratch/reexecute_all_notebooks_64.py`
  re-runs all eight notebooks (~3 min) and regenerates the no-output twins.
* **Safety**: pharmaceutical functions validate every input (positive doses, ordered time series,
  bounded fractions) and raise `DataValidationError` on bad values rather than returning nonsense.
* **Docs stay in sync with code**: `docs/INDEX.md` is auto-generated from each module's `__all__`
  and docstrings (`python scratch/build_docs_index.py` from the repo root); a drift-guard test
  (`tests/test_docs_index.py`) fails whenever the index and the live package diverge, so the
  documented symbol list can never silently rot.
* **Explorer dataset parity**: the companion web app's dataset preview route (`/api/ml/dataset`)
  uses the *exact* same seeded generators as its training route (`/api/ml/train`) — identical
  `mulberry32` PRNG, `make_regression_data`, and `makeMoons` implementations — so previewed rows,
  statistics, and scatter plots are guaranteed to match the data a model is actually trained on.
* **Shareable line references**: the web app's code explorer copies `path#LNN` references on any
  line click and accepts them back in its search box (validated against the file tree, then
  jumped to with a highlight), so a doc review, PR comment, or chat message can point peers at
  an exact line of the package.

---

## Installation

**Reproducible route (recommended):** `requirements.txt` pins the exact
stack the QA chain runs green under — install it first, then the package.

```bash
# 1. every core library at the verified version
#    (numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, scipy 1.14.1, ...)
pip install -r requirements.txt

# 2. the toolkit itself (editable — `import pharmaml` works immediately)
pip install -e .
```

Optional feature sets come as commented pinned blocks inside
`requirements.txt` (cheminformatics, docking prep, boosting, deep
learning with the torch CPU-wheel index line, reports, dev/QA) —
uncomment what you need, or use the loose-range extras instead:

```bash
# with cheminformatics extras (RDKit descriptors/fingerprints/similarity)
pip install -e ".[chem]"

# with docking preparation extras (RDKit + Meeko: SMILES -> 3D PDBQT)
pip install -e ".[docking]"

# with gradient-boosting model families
pip install -e ".[boosting]"

# development (pytest, mypy, ruff)
pip install -e ".[dev]"
```

Pick one philosophy per environment — the pinned file or the loose
extras — and don't mix them. `pyproject.toml` stays the compatibility
contract; `requirements.txt` is the verified snapshot (Python 3.12).

Quick test run (no install needed — `pyproject.toml` maps `src/` onto the import path):

```bash
python -m pytest
```

---

## Graphical playground

The toolkit ships a single-file graphical front end — open it instead of a terminal
whenever you prefer pointing and clicking:

```bash
python playground.py
```

A tabbed window opens with every calculation domain: **Chemistry** (IC50/Ki →
pIC50/pKi, molar mass, the ~200-descriptor table for any SMILES, Tanimoto
similarity, and the **feature factory** — compute the full 10,000+-column
seven-family feature matrix for a CSV or pasted SMILES set, rank every
column and keep the top K you choose, exporting the selected-features CSV
and its metadata JSON), **PK/PD** (half-life/accumulation, plotted IV bolus decay and Emax
curves, oral Cmax, Cockcroft–Gault with renal adjustment, loading/maintenance
and pediatric dosing), **Cell Lines** (GDSC/CCLE-style drug-response screens:
load and audit a (cell line, compound, metric) table, collapse replicates,
pivot the sensitivity matrix, call sensitive/resistant lines, and run the full
biomarker-discovery pipeline with Benjamini-Hochberg FDR control — the bundled
example recovers the planted EGFR biomarker for erlotinib in two clicks, plus
the LN_IC50 ↔ pIC50 converter), **Docking** (prepare a whole ligand library
as PDBQT with a per-ligand audit, turn a PDB/CIF receptor into a clean
receptor PDBQT and distribute it into every campaign folder, compute the
grid box and write the matching GPF/DPF pair for a reference ligand, parse
DLG/Vina results — or scan an entire campaign — with ligand-efficiency
columns, run AutoDock4/Vina when the binaries are on your PATH, and
**rescore any loaded table — or an external summary CSV — into the
explainable consensus**: one 0–1 `<column>_norm` unit per evidence column
plus the consensus, with the evidence recipe and the driving-evidence
verdict — then write the whole ranking as the one-click report bundle
(`report.html` + the consensus-ranking figure + the breakdown table,
`report.pdf` when the checkbox is ticked; the `docking rescore
--plots`/`--pdf-report` story, mouse edition)),
**Data & ML** (open any CSV or a bundled example, train a
registry model, read the metrics, browse a cross-validated leaderboard,
and write the whole run as a one-click report bundle — `report.html` +
figures + tables, with the multi-page `report.pdf` when the checkbox is
ticked; the CLI `--plots`/`--pdf-report` story, mouse edition — run a
**hyperparameter search**: `ml tune --searcher random`'s RandomSearch
over the ann / hsvr / dnn hypermodel families, every trial scored on the
same shuffled folds, the family's defaults scored too (the number the
search has to beat), the winner refit and scored on the held-out 20 %,
an apply-best bridge (ann/hsvr point the model combobox; dnn prefills
the whole Deep Learning tab) and a trials CSV export — or `--searcher
hyperband`'s successive halving over the family's resource budget
(ann's max_iter, dnn's epochs; the ladder is the family's own declared
grid via `resource_bounds`, the leaderboard gains a resource column and
the status line carries the honest budget economics) — and audit any
new CSV against the last trained model with **Check CSV…**:
missing features, extra columns, reordering and version drift, every
difference named, the training label dropped automatically; the `ml
check` story, mouse edition),
**Deep Learning** (the PyTorch estimators behind one card of inputs —
pick the architecture (feed-forward or residual MLP), hidden widths,
dropout, activation, Adam's learning rate and weight decay, the
early-stopping patience and validation fraction, a **batch-normalisation
checkbox** (`batchnorm=True`, a `BatchNorm1d` between every hidden
linear map and its activation), then read the held-out
metrics, the parity/confusion figure and the epoch-by-epoch loss curves
with the early-stopping verdict, exportable per-epoch as CSV — and the
same **Check CSV…** audit runs against the last trained network: the
estimators carry the sklearn feature contract, so missing features,
extra columns, reordering and version drift are named identically) and
**Generative** (evolve molecules against QED/LogP objectives with the
genetic algorithm, or fine-tune a SMILES RNN prior with reinforcement
learning — the optimiser combo swaps the parameter block, and RL runs
render the four-panel prior-drift introspection plus a first→last
history table; adaptive GA runs render the operator-mix learning curve;
multi-objective runs fill the Pareto front card — audit line,
non-dominated table with the crowding-distance column and the deepest
hidden trade-off rank, an objective-pair scatter and a CSV export of
the front).

Engineering notes:

* **No installation required** — like `pytest` above, the playground falls back
  to the bundled `src/` tree, so it runs straight after unzipping the archive.
* **Nothing is mandatory** — heavy dependencies are probed at launch and each
  feature names the exact `pip install` line it wants when missing; missing
  pieces never crash the window.
* **Responsive by design** — long calculations (descriptor batches, model
  training, evolution) run in a worker thread with a live progress bar; the
  figures reuse the package's own plotting helpers, so the window matches the
  published figure style (with an automatic PNG fallback for exotic Tcl/Tk
  builds).
* **Scriptable** — `python playground.py --probe` prints a dependency report,
  `--self-test` builds the whole UI headlessly (used by the test suite under
  Xvfb), `--version` prints the playground, package and toolkit versions.
* **Sessions persist** — on close the window writes
  `~/.pharmaml/playground-session.json` (every entry, toggle, pasted corpus,
  the active tab, the geometry, the theme) and restores it on the next
  launch; `File ▸ Save session now` / `Clear saved session` control it
  manually, and `--self-test` never touches the file.
* **Sessions are shareable** — `File ▸ Export session` writes the payload
  to any path (mail it to a colleague), `File ▸ Import session` applies
  such a file onto the live window (theme included), and
  `File ▸ Reset current tab to defaults` (Ctrl+Shift+R) restores just the
  active tab's inputs while every other tab keeps its live values. Every
  content tab also carries its own quiet **"Reset this tab to defaults"**
  button pinned in a non-scrolling footer strip below the cards, so the
  reset is always one click away (Home, which owns no inputs, has none) —
  with a second footer affordance next to it, **"Copy inputs as JSON"**:
  the same tab's live inputs on the clipboard as a small JSON snippet,
  a share-lite for issue reports and quick hand-offs.
* **Light and dark** — `View ▸ Follow system / Light / Dark` recolours the
  whole window from two named palettes; the choice travels with the session
  and embedded figures keep the published white plot background.
* **Keyboard-first** — every global accelerator is bound and pinned by
  tests: Ctrl+O open · Ctrl+S save figure · Ctrl+Shift+S save session ·
  Ctrl+Shift+R reset this tab · Ctrl+Q quit. A quiet **shortcut hint row**
  above the status bar lists them all (the Ctrl+Shift+R binding itself was
  a 100x-65 bug fix: the menu had advertised it since the reset feature
  landed, but no binding existed); `View ▸ Shortcut hints` or the row's own
  **Hide** button toggles it, and the preference rides the session file.

---

## Versioned downloads & updating

The downloadable archive is version-stamped: it downloads as
`pharma-ml-toolkit-v1.2.0.zip`, carries its version in the `VERSION` file
(and in `pyproject.toml` → `[project] version`), and the web explorer's
Download tab shows the live version next to the size. `CHANGELOG.md` at the
archive root documents every release; versions follow semver — minor bumps
add features, patch bumps ship fixes.

**Updating an existing copy needs no uninstall.** Unzip the new archive over
your current folder — `-o` overwrites without prompting:

```bash
unzip -o pharma-ml-toolkit-v1.2.0.zip -d pharmaml   # your existing folder
cat VERSION                                         # what you have now
```

Because the install is editable (`pip install -e .`), Python picks up the
replaced files on the next run — nothing to uninstall or reinstall, unless
`pyproject.toml` gained new dependencies (then re-run `pip install -e .`).
`python playground.py --version` prints the playground, package and toolkit
versions in one place. Files removed between versions are simply left behind
(harmless); delete the folder first only for a truly clean slate.

---

## Package layout

```
pharma-ml-toolkit/
├── src/pharmaml/
│   ├── __init__.py          # public exceptions + version
│   ├── exceptions.py        # PharmaMLError hierarchy
│   ├── logger.py            # configured logging (console + optional file)
│   ├── config.py            # ToolConfig: paths, random seed, verbosity
│   ├── data.py              # bring your own data: load_dataset, infer_task, summarize
│   ├── cli.py               # `pharmaml` command-line interface
│   ├── chem/                # converters, descriptors, similarity, tables, standardize
│   ├── ml/                  # preprocessing, models (20: incl. dnn + dnn_residual PyTorch), neural.py, training, evaluation, persistence
│   ├── generative/          # NEW: molecule generation — corpus, objective library, SMILES GA, RNN prior, RL
│   ├── biology/             # NEW: response screens, omics, biomarkers — cell line analysis
│   ├── pharma/              # units, pk, pd, dosage, approvals
│   ├── docking/             # ligands + receptor (CSV/PDB/mmCIF in), grids + GPF/DPF, workflow, parsing, scoring, campaign
│   ├── viz/                # figures for every domain + HTML reports + multi-format table export
│   └── utils/               # io: read_table, write_table, safe_filename · validation: validate_smiles_syntax, require_columns, coerce_frame, looks_continuous
├── playground.py           # NEW: the graphical front end — `python playground.py` opens a tabbed
│                            # Tkinter window over every domain (chemistry · PK/PD · cell lines · QSAR · generative);
│                            # probes its dependencies at launch, degrades gracefully, headlessly
│                            # self-testable (`--self-test`, used by tests/test_playground.py under Xvfb)
├── VERSION                 # the release version (1.2.0) — `cat VERSION` after extracting
├── CHANGELOG.md            # what changed in each versioned release
├── LICENSE                 # MIT — the pyproject declaration, materialised
├── docs/COVERAGE.md        # the one-by-one RAR audit: every original file → its PharmaML surface
├── tests/                   # 2022 pytest cases across all domains (incl. test_docking_prep.py, test_campaign.py, test_biology.py, test_neural.py, test_cellline_examples.py, test_descriptors.py, test_rism.py, test_hierarchical.py, test_tuner.py, test_format_converters.py, test_features.py, test_playground.py + data/ fixtures)
├── examples/                # runnable demos + eight executed notebooks (complete ML · complete docking · complete PK/PD · complete cell line · complete molecule generation · complete deep learning · complete tuning & hierarchical · walkthrough) + seven no-output twins
│   └── data/                # the shipped demo inventory — 43 files, ~718 KB, every executed notebook
│                            #   loads its inputs from here (never fabricates sample data);
│                            #   regenerate deterministically via scratch/build_example_data.py
│       ├── qsar_classification.csv        # ML §1      420 × 11 (active 118 / weak 155 / inactive 147)
│       ├── screening_library.csv          # ML §2       79-row SMILES library (77 valid + 2 planted invalid)
│       ├── library_with_descriptors.csv   # ML §3       77 × 10 (8 key descriptors + activity)
│       ├── qsar_multiclass.csv            # ML §8      360 × 21 (inactive 125 / weak 102 / moderate 79 / strong 54)
│       ├── qsar_regression.csv            # ML §9      260 × 21, continuous pIC50 endpoint
│       ├── cellline_screen.csv            # cellline §1 185 rows: 23 cell lines × 8 compounds across 11 tissues
│       │                                  #   (+1 duplicate pair, +1 missing value to audit)
│       ├── cellline_expression.csv        # cellline §6 23 × 300 expression matrix (EGFR/KRT8 markers planted)
│       ├── pk_profile.csv                 # PK §1       49 samples, t = 0..24 h (AUC lin 17.9607 / log 17.2536)
│       ├── ligand_library.csv             # docking §1  10 rows (8 valid + 2 planted invalid)
│       ├── lysozyme_1aki.pdb              # docking §2  real lysozyme, RCSB entry 1AKI (public domain)
│       ├── docking_campaign/              # docking §6  14 folders: 12 complete runs (r.dlg cluster table
│       │                                  #   + r.glg each) + 2 planted audit cases + ligands.csv manifest
│       │       ├── 08_benzocaine/         #   MISSING_LOG case — README only, the log was lost
│       │       ├── 10_nicotine/           #   INCOMPLETE case — interrupted before clustering
│       │       └── ligands.csv            #   14-row (name, SMILES) manifest mapping folders to structures
│       ├── seed_molecules.csv             # generative §1b  12 named drugs for seeding (443 B, pinned)
│       ├── walkthrough_compounds.csv      # walkthrough  180 × 8 (active 39 / inactive 88 / weak 53)
│       ├── walkthrough_classification.csv # walkthrough  300 × 9 binary target
│       └── walkthrough_multiclass.csv     # walkthrough  300 × 9, 3 classes
│                            # Provenance: every file is byte-identical to what the corresponding
│                            #   notebook cell originally generated (same seeds, same code path — see
│                            #   the map in scratch/build_example_data.py). PDBs come from RCSB entries
│                            #   1AKI/1CRN (public domain; tests/data/1crn.pdb is the second one) and
│                            #   are re-downloaded only when missing.
├── docs/                    # INDEX.md — auto-generated module/API index
├── .github/workflows/       # CI matrix (Python 3.10–3.12) + full-deps job
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### The complete ML process in one notebook

`examples/complete_ml_workflow.ipynb` is the **zero-to-one-hundred** walkthrough —
every stage of a QSAR project in the order you would actually run it, with all
outputs committed (numbers *and* inline figures):

> §1 data entry (your own file) → §2 **feature calculation** (SMILES → 217 RDKit
> descriptors, Morgan/RDKit/atom-pair/Avalon fingerprints, MACCS keys, Tanimoto
> similarity heat-map, structures → model demo) → §3 EDA → §4 IC50→pIC50 →
> §5 split → §6 grid-searched training → §7 evaluation (confusion, ROC) →
> §8 multiclass → §9 regression (parity, residuals, outlier-robust Huber demo) → §10 leaderboard →
> §11 feature importance → §12 save/load & scoring new compounds → §13
> one-click HTML report + png/svg/pdf + csv/md/json.

Open it, replace the demo file with your own CSV (and the §2 library with your
own SMILES column), and re-run all cells. §2 needs RDKit —
`pip install 'pharmaml[chem]'`.

### The complete docking process in one notebook

`examples/complete_docking_workflow.ipynb` is the docking **zero-to-one-hundred**
walkthrough — a full virtual-screening campaign that now starts from the two
data-entry stages (CSV and PDB in), executed end-to-end without a docking engine
(the demo campaign tree uses the real AutoDock4 `.dlg` format):

> §1 ligand-library import (`load_ligand_csv` on the shipped
> `examples/data/ligand_library.csv` — 10 rows, 8 valid + 2 planted invalid;
> SMILES column auto-detected, RDKit validation, invalid rows exported to
> `failed_smiles.csv`, then `prepare_ligand_library` builds the folder-per-ligand
> campaign tree) → §2 protein import & preparation (`prepare_protein` on a real
> lysozyme PDB — waters/HETATM/altlocs cleaned, Meeko-typed + charged receptor,
> `distribute_receptor` copies it into every ligand folder) → §3 ligand
> preparation (SMILES → 3D PDBQT via Meeko) → §4 grid box + GPF/DPF → §5 running
> the engines (AutoDock4 / Vina, guarded failures) → §6 parsing (cluster tables,
> completion audit, campaign scan, idempotent batch re-run — on a scratch copy of
> the shipped 14-folder `examples/data/docking_campaign/` demo) → §7 torsion QA &
> campaign housekeeping (tree-wide TORSDOF/BRANCH audit with one deliberate
> mismatch, plus dry-run collect/cleanup/distribute/delete previews) → §8 Vina
> pose tables → §9 rescoring (ligand efficiency + consensus + the explainable
> breakdown — why the top ligand wins) → §10
> energy-distribution & top-ligand figures → §11 one-click HTML report.

Point `DOCKING_DIR` at your real campaign folder and re-run — parsers, rescoring
and plots work unchanged on real `.dlg` trees. §1–§3 need
`pip install 'pharmaml[docking]'`; a ready-to-use demo receptor ships at
`examples/data/lysozyme_1aki.pdb`.

### The complete PK/PD process in one notebook

`examples/complete_pk_workflow.ipynb` completes the trilogy — the pharma
zero-to-one-hundred walkthrough the legacy archive never had, executed with all
outputs committed (numbers *and* inline figures):

> §1 data entry (your own concentration–time file, validated on the way in) →
> §2 NCA (AUC/AUMC/MRT/Cmax/Tmax/terminal t½) → §3 the AUC math exposed
> (linear vs log trapezoid) → §4 derived parameters (CL, CL/F, Vd — with an
> honest look at what Dose/AUC really is) → §5 analytic 1-compartment checks →
> §6 IV routes (bolus curve, infusion steady state) → §7 repeated dosing &
> accumulation → §8 PD (Emax family, EC80 inversion) → §9 dosage design
> (loading/maintenance/infusion, BSA & weight rules, Clark's & Young's) → §10
> patient-specific adjustments (Cockcroft-Gault, renal, TDM trough) → §11 units
> & molar mass → §12 one-click HTML report + multi-format tables.

Replace the demo profile with your own CSV (`time_h`, `conc_ng_ml` columns) and
re-run; the CLI equivalent is `pharmaml pharma pk --data profile.csv`.

(`examples/pharmaml_walkthrough.ipynb` remains the every-domain tour including
PK, docking and chem conversions.)

Each of the seven workflow notebooks also ships a **no-output twin**
(`*_no_output.ipynb`) — identical cells with every output stripped, 4–30 % of
the executed size (the ML twin is 46 KB vs 1.1 MB), for slow connections; run
it top-to-bottom to produce the results yourself. A pytest drift-guard pins
the twins' cell sources to the executed originals.

### The complete cell line process in one notebook

`examples/complete_cellline_workflow.ipynb` — the cell line zero-to-one-hundred
walkthrough, executed with all outputs committed (23 cell lines across 11 tissues,
8 compounds, 300 genes — planted structure that the analysis genuinely recovers;
the screen and the expression matrix ship as `examples/data/cellline_screen.csv`
and `examples/data/cellline_expression.csv`):

> §1 data entry (GDSC-style response screen, validated on the way in) → §2 QC
> audit (missing values, duplicate pairs, replicate collapse, **§2b replicate
> variability** — the CV the median discards, `replicate_variability` +
> `add_cv=True` companion columns) → §3 normalization
> (LN_IC50 → pIC50, distributions) → §4 the cell × compound potency matrix +
> per-compound ranking → §5 sensitive/resistant calling by quantiles → §6 omics
> features (log2 + z, MAD gene selection) → §7 **biomarker discovery**
> (per-gene Spearman vs response, BH-FDR — the planted EGFR → erlotinib
> association is recovered at rho=+0.978, FDR≈2.4e-13) → §8 classical response
> model → §9 **deep learning** (`dnn` / `dnn_residual` in the same
> `compare_models` leaderboard — where `dnn` tops the table, see the
> target-standardisation note under *Deep learning*) → §10 recap.

Two runnable scripts back the notebook: `examples/make_synthetic_cellline_data.py`
generates the demo tables (24 cell lines × 8 compounds with the planted EGFR
biomarker, optional `--clean` to skip the hygiene issues), and
`examples/run_cellline_analysis.py` runs the whole flow on synthetic or user
data — audit → collapse → pIC50 ranking → sensitivity calls → biomarkers →
classical (+ `--deep`) response model:

```bash
python examples/run_cellline_analysis.py                 # synthetic data
python examples/run_cellline_analysis.py --deep           # + dnn / dnn_residual
python examples/run_cellline_analysis.py --screen my_screen.csv \
    --expression my_expr.csv --compound cisplatin        # your own tables
```

Point §1 at your own GDSC/CCLE export (same columns) and §6 at a CCLE
expression matrix; the CLI equivalents are `pharmaml biology response-audit`
and `pharmaml biology top-biomarkers`.

### The complete molecule generation process in one notebook

`examples/complete_generative_workflow.ipynb` is the molecule-building
**zero-to-one-hundred** walkthrough — the fifth workflow notebook, executed
with all outputs committed (numbers *and* inline figures) in ~20 s on a
laptop CPU:

> §1 the bundled 244-SMILES corpus + its property landscape (217 RDKit
> descriptors) → §2 the objective library (QED, LogP/TPSA/MW bands,
> similarity, pharmacophore, novelty — scored on a probe set) → §3
> **MultiObjective** (weighted geometric mean; one zero vetoes the molecule,
> demonstrated live) → §4 **de novo GA** (leaderboard with free breakdown
> columns + convergence figure) → §5 **seeded GA** from your own molecules →
> §6 **scaffold-preserving mutation** (Murcko-retention A/B: 100 % on seed
> frameworks vs 50.5 % for uniform mutation) → §6b **scaffold-hop
> crossover** (BREED-style homologous cuts: aspirin × naproxen → a hybrid
> carrying one fragment of each; ring-bond cuts — rings merged, or a linker
> transplanted: piperidine × an ether oxygen → a 7-ring oxazepane, piperidine ×
> diethylene glycol → an 8-ring via the whole `-CH2O-` path; the structural
> mode's leaderboard) →
> §7 **RL** (char-GRU prior loss curve + REINVENT-style fine-tuning with
> early stop) → §8 the **novelty** anti-rediscovery recipe (honest
> strange-chemistry trade-off) → §9 the four-arm **GA-vs-RL head-to-head**
> (de novo, seeded, structural-mode GA and RL — on this objective the
> structural arm's framework recombinations win, 0.749 vs 0.699 seeded)
> → §10 leaderboard
> CSV + one-click visual bundle → §11 recap.

The 12-drug seed table ships as `examples/data/seed_molecules.csv` — point
`SEEDS_CSV` at your own CSV and every cell below it runs unchanged. §1–§6 need
RDKit (`pip install 'pharmaml[chem]'`); §7 additionally needs PyTorch
(`pip install 'pharmaml[deep]'`).

### The complete deep learning process in one notebook

`examples/complete_deep_learning_workflow.ipynb` is the sixth workflow
notebook — the executed-notebook face of the PyTorch estimators behind the
playground's *Deep Learning* tab (100x-61), executed with all outputs and
six inline figures committed in ~15 s on a laptop CPU. Where the cell line
notebook only meets `dnn` through the `compare_models` leaderboard, this one
walks the **direct estimator API** of `pharmaml.ml.neural`:

> §1 data entry (the shipped `qsar_regression.csv`, audited on the way in) →
> §2 the seed-reproducible train/test split → §3 **`DeepMLPRegressor`** (the
> sklearn contract: fit/predict/score, minibatch Adam, per-feature
> standardisation, early stopping on a held-out validation fraction) → §4
> **the training story** — the per-epoch history record plotted by
> `plot_training_history`, dashed best-validation marker, the
> early-stopping verdict in the title → §5 evaluation (parity scatter,
> residuals, R²/RMSE/MAE) → §6 **architecture comparison** — plain
> `(16, 8)` vs `ResidualMLPRegressor (16, 16, 16)` under the same split,
> same seed, same cap → §7 the **registry path**
> (`train_model("dnn", ...)` — impute + scale wrapped; why its R² differs
> from the direct path, explained) → §8 **`DeepMLPClassifier`** (3-level
> activity, 91 % accuracy, confusion matrix) → §9 the history record
> exported as CSV → §10 recap with the playground cross-reference.

Point §1 at your own descriptor table and every cell below runs unchanged
(same story as the other notebooks). PyTorch is the only extra:
`pip install 'pharmaml[deep]'`.

### The complete tuning & hierarchical workflow in one notebook

`examples/complete_tuning_workflow.ipynb` is the seventh workflow
notebook — the executed-notebook face of the four 1.2.36 features (the
user's request round): 3D-RISM-KH solvation descriptors, `HierarchicalSVR`,
the keras-tuner-style search and batch normalization, executed with all
outputs and five inline figures committed in ~20 s on a laptop CPU:

> §1 data entry (the shipped `qsar_regression.csv`) → §2 **3D-RISM-KH
descriptors** — the ten water-scale quantities for the shipped 12-drug set,
the surrogate contract told honestly (features, not RISM-calibrated
energies) → §3 **`HierarchicalSVR` vs the global SVR** — the three potency
regimes k-means finds, the per-cluster diagnostics table, two parity
scatters → §4 **`RandomSearch`** over an ANN hypermodel — the
hyperparameters-space protocol, one shared seeded fold split for every
trial, the defaults-vs-best gain line, the leaderboard figure → §5
**`Hyperband`** successive halving over the deep hypermodel — the resource
ladder (mean CV R² climbing rung by rung) and the epoch economics → §6
**batchnorm A/B** — same split, same seed, only the flag differs; the
honest verdict on a small standardised table → §7 recap with the CLI/web
cross-references (`pharmaml chem rism`, `pharmaml ml tune`, the playground's
Hierarchical group).

A pytest guard pins the committed verdicts (the hsvr/global R² pair, the
tuner's gain line, Hyperband's savings, the A/B lines) so the demo can
never silently become stale. RDKit powers §2 and PyTorch §5-§6 — both
optional extras.

### Bringing your own data (CSV / TSV / Excel)

`pharmaml.data` is the single entry point for **your own files** — the legacy scripts
hardcoded dataset filenames in ~30 places, so using a different file meant editing source
code. Point `load_dataset()` at any file instead:

```python
from pharmaml.data import load_dataset

# 1. Load the file (CSV/TSV/TXT/XLSX/XLS; delimiter sniffed automatically,
#    semicolon- and tab-separated European exports included).
ds = load_dataset("my_compounds.csv", target="activity")

# 2. Inspect before training.
print(ds.summary())
# source        : my_compounds.csv
# shape         : 350 rows x 41 columns (40 numeric features)
# target        : activity
# task          : classification
# classes (3)   : 'active': 120, 'inactive': 190, 'partial': 40
# dropped cols  : SMILES, Name  [non-numeric]

# 3. Train — ds.X / ds.y are ready for any pharmaml.ml function.
from pharmaml.ml import train_model, evaluate_model
from pharmaml.ml.preprocessing import make_train_test_split

X_train, X_test, y_train, y_test = make_train_test_split(ds.X, ds.y)
result = train_model("random_forest", ds.task, X_train, y_train)
report = evaluate_model(result.model, X_test, y_test, ds.task)
```

What it does for you:

* **Target auto-detection** — when `target` is omitted, common column names
  (`Target`, `class`, `label`, `activity`, `pIC50`, `y`, …) are tried, case-insensitively.
* **Task inference** — `infer_task()` treats any non-numeric target (or a numeric target
  with 2–`max_class_count` distinct values) as classification and continuous targets as
  regression; pass `task=` explicitly to override.
* **Multiclass out of the box** — 2, 3, 10 classes: string or integer labels both work.
* **Visible cleanup** — rows with a missing target are dropped (logged), non-numeric
  feature columns (SMILES, IDs) are excluded and reported in `dropped_columns`.
* **Categorical features kept on demand** — `load_dataset(...,
  keep_categorical=True)` keeps string feature columns (assay, cell line,
  formulation code) in `X`, lists them in `categorical_columns`, and
  `train_model` / `compare_models` one-hot encode them automatically
  (`handle_unknown="ignore"`, so unseen categories at scoring time don't
  crash). Metadata like SMILES should still go through `drop_cols`.

Stratified splits (upgrading the legacy `Split CSV file.py`, which cut the
file in half without class balance):

```python
from pharmaml.data import split_table

parts = split_table("my_compounds.csv", test_size=0.2, val_size=0.1)
parts.stratified        # True when class balance was preserved
parts.train, parts.val, parts.test
parts.write("splits/", prefix="qsar")   # qsar_train/val/test.csv
```

Rare classes trigger a logged fallback to a plain random split instead of
crashing; `--no-stratify` opts out entirely.

No-code alternative via the CLI:

```bash
pharmaml data inspect --data my_compounds.csv --target activity --head 5
pharmaml data split  --data my_compounds.csv --test-size 0.2 --val-size 0.1
pharmaml data convert --data ligands.csv --to sdf          # CSV -> 3D-embedded SDF
pharmaml data convert --data library.smi --to csv          # streaming .smi -> CSV
pharmaml ml train  --data my_compounds.csv --target activity --task auto
pharmaml ml train  --data my_compounds.csv --model rf --plots --keep-categorical
pharmaml ml train  --data my_compounds.csv --pdf-report          # bundle + report.pdf
pharmaml ml compare --data my_compounds.csv --target activity --plot
pharmaml ml compare --data my_compounds.csv --pdf-report        # leaderboard + report.pdf
pharmaml ml check   --model artifacts/rf.joblib --data new.csv --target activity
```

`--task auto` (the default) infers classification vs regression from the target column, so
the same command works for a binary, multiclass or pIC50 endpoint file.

### Import your docking inputs (ligands CSV + protein PDB)

Docking campaigns now start at data entry too — point the package at your ligand
table and your protein file instead of the legacy `smilestopdbqt.py` /
`adder.py` scripts:

```python
from pharmaml.docking import (
    load_ligand_csv, prepare_ligand_library,
    prepare_protein, distribute_receptor,
)

lib = load_ligand_csv("ligands.csv")            # SMILES col auto-detected, bad rows -> failed_smiles.csv
prepare_ligand_library(lib, "ligand_folders")   # upgrades smilestopdbqt.py: folder-per-ligand + l.pdbqt
receptor = prepare_protein("protein.pdb", "r.pdbqt")   # waters/HETATM/altlocs cleaned, AD4 types + charges
distribute_receptor(receptor, "ligand_folders")        # upgrades adder.py: r.pdbqt into every folder
```

Modern `.cif`/`.mmcif` entries work too — `prepare_protein("protein.cif",
"r.pdbqt")` converts them via gemmi (`pip install gemmi`) before the same
cleaning pipeline; the receptor's AutoDock atom-type census is one command away
with `pharmaml.viz.plot_receptor_atom_types(receptor)`.

Try it with the shipped demo receptor `examples/data/lysozyme_1aki.pdb`, or go
no-code: `pharmaml docking prep-ligands` / `pharmaml docking prep-protein`.

### Machine learning (QSAR)

```python
import pandas as pd
from pharmaml.data import load_dataset
from pharmaml.ml import (
    clean_numeric_frame, make_train_test_split, train_model,
    evaluate_model, compare_models, save_model, load_model,
)

ds = load_dataset("dataset.csv", target="Target")   # or build frames yourself:
frame = ds.frame                                        # clean_numeric_frame etc.
X_train, X_test, y_train, y_test = make_train_test_split(
    ds.X, ds.y, test_size=0.2
)

result = train_model("random_forest", ds.task, X_train, y_train, cv=5, tune=True)
print(f"CV ROC-AUC {result.cv_mean:.3f} ± {result.cv_std:.3f} in {result.fit_seconds:.1f}s")
print("Best params:", result.best_params)

report = evaluate_model(result.model, X_test, y_test, ds.task)
print(f"accuracy={report.accuracy:.3f}  f1={report.f1_macro:.3f}  roc_auc={report.roc_auc:.3f}")

save_model(result.model, "artifacts/rf.joblib", name="random_forest",
           task="classification", metrics={"accuracy": report.accuracy})
model, metadata = load_model("artifacts/rf.joblib")

from pharmaml.ml import validate_bundle_input
audit = validate_bundle_input("artifacts/rf.joblib", new_frame)
print(audit)   # 'audit: OK' or every difference named

leaderboard = compare_models(ds.X, ds.y, ds.task, cv=5)
print(leaderboard.head())   # one row per model, sorted by CV metric
```

`validate_bundle_input` is the round-trip audit: before predicting on a new
dataset with a model saved months ago, it checks the frame against the
bundle's recorded feature contract (names *and* order — sklearn rejects a
reordered frame even when the set matches) and names every difference:
missing features, extra columns, reordering, the feature-count contract when
names were never recorded, plus sklearn/pharmaml version drift as warnings.
The CLI twin is `pharmaml ml check --model artifacts/rf.joblib --data new.csv`
(exit 1 on a failed audit — CI-friendly; `--target <col>` drops the training
label first so re-auditing the training CSV works out of the box). The ML
workflow notebook demonstrates the audit end-to-end on a drifted copy of its
own CSV — the OK verdict, the named drift and the reorder trap, all with
committed outputs (`examples/complete_ml_workflow.ipynb`, §12).

A fully **executed Jupyter notebook** covering all four domains lives at
`examples/pharmaml_walkthrough.ipynb` (every cell has its output committed — run it
top-to-bottom to see training metrics, NCA results and docking tables).

Registry models: `logistic_regression`, `ridge_classifier`, `svm`, `knn`, `decision_tree`,
`random_forest`, `extra_trees` (**extremely randomized trees** — random thresholds on the
full sample, `bootstrap=False`: the extra variance reduction when bagging alone plateaus;
aliases `et` / `ert`), `adaboost`, `gradient_boosting` (**Friedman's classic GBM** —
shallow depth-3 trees fit the residual gradient, learning-rate shrunk; the sklearn
ancestor behind `XGboost.py` / `LightGBM.py`; aliases `gb` / `sk_gbm`, while `gbm`
remains lightgbm's short name), `ann`, `lasso`, `ridge`, `elastic_net`,
`linear_regression`, `huber_regression` (outlier-robust), `pls` (**partial least squares
regression** — NIPALS latent directions that maximize X–y covariance, the QSAR staple for
wide, collinear descriptor blocks with few observations; regression only, aliases
`plsr` / `partial_least_squares`), `hsvr` (**hierarchical SVR** —
k-means local
specialists blended with a global SVR, regression only), plus `xgboost` / `lightgbm` /
`catboost` when the optional packages are installed — and two **deep-learning families**:
`dnn` (deep feed-forward MLP) and `dnn_residual` (residual net with learned skip
projections), both classification **and** regression, powered by PyTorch
(`pip install 'pharmaml[deep]'`).
They are full sklearn-API citizens — `train_model("dnn", …)`, `compare_models(models=[…,
"dnn"])` and grid search work unchanged; without torch installed they simply report
unavailable instead of breaking anything.
`list_models()` shows what is available in *your* environment.

#### Deep learning (`dnn` / `dnn_residual` — PyTorch)

```python
from pharmaml.ml import train_model, evaluate_model

# A deep feed-forward net — same call as every classical model.
result = train_model("dnn", "classification", X_train, y_train, tune=True)
report = evaluate_model(result.model, X_test, y_test, "classification")

# Or the residual architecture (skip connections, deeper stacks):
result = train_model("dnn_residual", "regression", X_train, y_train)

# Or side-by-side in one leaderboard, classical vs deep:
board = compare_models(X, y, task="regression",
                       models=["elastic_net", "random_forest", "ann", "dnn", "dnn_residual"])
```

Both families: minibatch Adam with weight decay, dropout, **optional batch
normalisation** (`batchnorm=True` inserts a `BatchNorm1d` between every hidden linear
map and its activation — running statistics ride along in the portable weight dict,
`batch_size=1` is rejected and singleton tail batches are skipped), early stopping on a
validation split, per-feature standardization fitted inside `fit`, deterministic under
`random_state`, multiclass softmax, and sklearn `clone`/`GridSearchCV`/pickle compatible.
Aliases: `deep_mlp` → `dnn`, `resnet` / `residual_dnn` → `dnn_residual`.
(sklearn's own MLP behind `ann` has no BN layers — the registry note points
batch-normalisation users at the torch estimators.)

**Target standardisation for regression.** Regression targets are z-scaled inside
`fit` (on the training rows only) and mapped back at `predict`. Pharmaceutical
endpoints live far from zero — pIC50 ≈ 5, LN_IC50 ≈ 2–4, AUC ∈ [0, 1] — and on a
raw-scale MSE the output bias crawls toward the target mean at roughly one
learning-rate step per epoch, so early stopping fires long before the intercept
is learned (a plain `MLPRegressor`-style net then loses to linear regression by
an order of magnitude). With internal scaling the deep models are competitive
from the first epochs: in the cell line notebook's leaderboard `dnn` posts the
best CV RMSE (0.47 vs 0.49 for the tuned elastic net).

#### Hierarchical SVR (`hsvr`) — local specialists for heterogeneous data

One global kernel SVR underfits when different regions of chemical space carry
different input–output structure. `HierarchicalSVR` (registry key `hsvr`, aliases
`hierarchical_svr` / `h_svr`) keeps the global SVR as the upper level and adds
k-means local specialists below it; predictions blend the nearest cluster's local
model with the global one, the local weight fading from `alpha` at the centroid to
0 at the cluster's trust radius — far-from-training queries fall back to the global
model instead of extrapolating through an unfamiliar local kernel:

```python
from pharmaml.ml import HierarchicalSVR, train_model

result = train_model("hsvr", "regression", X_train, y_train, tune=True)
hsvr = result.model
hsvr.cluster_sizes_    # rows per specialist
hsvr.radii_            # per-cluster trust radii (standardised space)

# Or standalone, fully sklearn-compatible (clone/GridSearchCV/pickle):
hsvr = HierarchicalSVR(n_clusters=3, alpha=0.8, radius_factor=2.0,
                       random_state=42).fit(X_train, y_train)
```

#### Keras-tuner-style hyperparameter search (`ml.tuner`)

The `HyperParameters` / `RandomSearch` / `Hyperband` programming model from
[Keras Tuner](https://keras.io/keras_tuner/), re-implemented over ANY
scikit-learn-compatible estimator — no TensorFlow requirement (the toolkit's deep
nets are PyTorch, the legacy ANN is sklearn's MLP). The hypermodel declares its
space by calling `hp.Int(...)`/`hp.Float(...)`/`hp.Choice(...)`/`hp.Boolean(...)`
inside the builder; the tuner samples, cross-validates and refits the winner:

```python
from pharmaml.ml import RandomSearch, Hyperband, build_ann_regressor, build_deep_regressor

# Any sklearn scorer name (or a callable(estimator, X, y) -> float); higher wins.
results = RandomSearch(build_ann_regressor, objective="r2",
                       max_trials=12, cv=3, seed=42).search(X, y)
results.best_params, results.best_score, results.best_estimator
print(results.summary())            # the leaderboard
results.trials_frame().to_csv("trials.csv", index=False)

# Successive halving over the resource hyperparameter (epochs for the deep nets,
# max_iter for the sklearn ANN; batchnorm is itself searchable):
results = Hyperband(build_deep_regressor, objective="r2",
                    resource="epochs", max_resource=81).search(X, y)

# Size the ladder from the family's OWN declared grid -- the shipped
# hypermodels' resource slots (the playground's Tune card does exactly this):
from pharmaml.ml import resource_bounds
r_min, r_max = resource_bounds(build_ann_regressor, "max_iter")   # (200, 1000)
results = Hyperband(build_ann_regressor, resource="max_iter",
                    max_resource=r_max, min_resource=r_min, eta=3).search(X, y)

# Score a baseline under the tuner's OWN folds -- apples to apples:
from sklearn.model_selection import cross_val_score
from pharmaml.ml import HyperParameters, make_cv_splitter

baseline = cross_val_score(build_ann_regressor(HyperParameters()), X, y,
                           scoring="r2", cv=make_cv_splitter("regression", y, 3))
```

Ready-made hypermodels: `build_ann_regressor` / `build_ann_classifier` (sklearn
MLP), `build_deep_regressor` / `build_deep_classifier` (PyTorch, with
`batchnorm` a `hp.Boolean` slot) and `build_hsvr`. The CLI twin is
`pharmaml ml tune` (random or hyperband, leaderboard + trials table +
`--save-best` bundle) and the desktop playground's **Hyperparameter
search** card runs the same search in either flavour — RandomSearch, or
Hyperband with the rung ladder sized from the family's own declared grid
via `resource_bounds` (a resource column in the leaderboard, the honest
budget line: “budget 3998 of a naive 6000 max_iter (33% saved)”) — with
the defaults baseline, an apply-best bridge into the Deep Learning tab
(the winner's rung resource rides along) and a trials CSV export (mouse
edition). Failed trials are recorded, not fatal; sampling is seeded, so
the same seed + data always reproduce the same search.

#### 3D-RISM-KH solvation descriptors (`chem.rism`)

Ten solvation-theory descriptors shaped after the quantities a 3D-RISM-KH
analysis reports — site distribution peaks (`RISM_Ow_gmax` / `RISM_Hw_gmax`),
first-shell coordination numbers (`RISM_*_cn1`), an excluded-volume estimate,
the Gasteiger dipole, charge hydration asymmetry, and a solvation free energy
decomposed into Lennard-Jones and Coulomb parts — computed from a documented
*pairwise tempered-KH surrogate* (ETKDG 3D embedding + Gasteiger charges +
screened site potentials) that runs in milliseconds instead of grid-hours.
Fully deterministic; intended as QSAR features for solubility/permeability
endpoints, not as a production RISM calculation:

```python
from pharmaml.chem import compute_rism_descriptors

frame = compute_rism_descriptors(["CCO", "c1ccccc1"], solvent="spc")  # or "tip3p"
# RISM_Gsolv_kcalmol, RISM_Glj_kcalmol, RISM_Gcoul_kcalmol, RISM_Ow_gmax,
# RISM_Hw_gmax, RISM_Ow_cn1, RISM_Hw_cn1, RISM_Ow_exclvol_A3,
# RISM_dipole_D, RISM_q_neg_frac
```

The same ten columns are the **8th feature-factory family** (`rism_` prefix —
`compute_feature_frame(smiles, families=("descriptors", "rism", ...))`), so the
solvation descriptors ride along the 10,000-column ranking/selection pipeline.
The CLI twin is `pharmaml chem rism --data mols.csv [--solvent tip3p]`.

#### Classification — two recipes: exactly 2 classes, or K classes of your choosing

**Recipe A — binary classification (two classes).** The classic
active/inactive screen; any target column with exactly 2 distinct values
(`0/1`, `Y/N`, `active/inactive`, …) takes this path:

```python
from pharmaml.data import load_dataset
from pharmaml.ml import train_model, evaluate_model
from pharmaml.ml.preprocessing import make_train_test_split

# Your file: a target column with exactly 2 distinct values.
ds = load_dataset("my_compounds.csv", target="activity")   # e.g. active/inactive
X_train, X_test, y_train, y_test = make_train_test_split(ds.X, ds.y)   # stratified

result = train_model("random_forest", "classification", X_train, y_train)
report = evaluate_model(result.model, X_test, y_test, "classification")
print(report.roc_auc)     # single ROC-AUC for the positive class
print(report.confusion)   # 2 x 2 confusion matrix
```

**Recipe B — multiclass classification (you select the number of classes).**
Set `N_CLASSES` to any value ≥ 2 — 3 (`low/medium/high`), 4, 5, … — and
every component adapts automatically:

```python
from pharmaml.data import load_dataset
from pharmaml.ml import train_model, evaluate_model
from pharmaml.ml.preprocessing import make_train_test_split

N_CLASSES = 4   # <-- select the class count: 2, 3, 4, 5, ...

# From your own file: a target with K distinct values is automatically
# multiclass. Or generate a demo table with exactly K classes:
#   from make_synthetic_qsar_data import make_classification_table
#   demo = make_classification_table(n_classes=N_CLASSES,
#                                    class_labels=["low", "medium", "high", "very_high"])
ds = load_dataset("my_compounds.csv", target="activity")   # K distinct values
X_train, X_test, y_train, y_test = make_train_test_split(ds.X, ds.y)   # every class stratified

result = train_model("random_forest", "classification", X_train, y_train)
report = evaluate_model(result.model, X_test, y_test, "classification")
print(report.confusion)   # K x K confusion matrix
print(report.roc_auc)     # one-vs-rest multiclass AUC
```

* Both recipes share one pipeline — the only difference is the number of
  distinct values in the target column (`load_dataset` reports it as
  `classes (K): …`).
* Splits are **stratified**, so every class appears in train *and* test folds.
* Metrics are macro/weighted-averaged (`f1_macro`, `precision_macro`, …) plus a k×k
  confusion matrix and one-vs-rest ROC-AUC; MCC and balanced accuracy are multiclass-native.
* `compare_models()` scores with one-vs-one ROC-AUC — and falls back to balanced accuracy
  when a compared model lacks `predict_proba` (e.g. `ridge_classifier`), so leaderboards
  never silently fill with NaN.
* `build_voting(..., voting="hard")` works for any classifier and any class count;
  `voting="soft"` validates that every member exposes `predict_proba` and tells you which
  ones don’t.
* Generate demo data with a selected class count:
  `python examples/make_synthetic_qsar_data.py --classes 4` or
  `python examples/run_classification.py --classes 4` (a cached synthetic table is
  regenerated automatically when its class count differs from the requested one).

### Molecule generation (from 0 to 100)

Can the toolkit **invent** molecules, not just analyse them? Yes — the `generative`
subpackage ships the whole story on a laptop CPU: a bundled curated corpus
(`DEFAULT_CORPUS`, 244 canonical drug-like SMILES — aspirin, paracetamol, ibuprofen,
caffeine, … plus a heterocycle/functional-group tour), an objective library, and two
complementary optimisers. A runnable narrative lives in
`examples/run_generative_design.py` (~15 s end to end, the GA running
denovo / seeded / structural), and the fully
**executed notebook** `examples/complete_generative_workflow.ipynb` walks
every stage with committed outputs (see *Quickstart* above).

```python
from pharmaml.generative import (
    DEFAULT_CORPUS, QedObjective, LogPBandObjective, SimilarityObjective,
    NoveltyObjective, MultiObjective, GeneticOptimizer, ReinventOptimizer, SmilesRnn,
)

# 1) Define what a "good" molecule means: drug-like + LogP window + aspirin-ish.
#    One zero score vetoes the whole molecule (weighted geometric mean).
objective = MultiObjective(
    [QedObjective(), LogPBandObjective(1.0, 3.0), SimilarityObjective("CC(=O)Oc1ccccc1C(=O)O")],
    weights=[0.5, 0.25, 0.25],
)

# 2) Genetic algorithm: character-level mutation + splice crossover over canonical
#    SMILES, tournament selection, elitism. CPU-strong, torch not required.
#    mutation="scaffold" keeps each parent's Murcko framework and resamples only
#    its decorations (lead-optimisation mode); crossover="scaffold_hop" cuts two
#    parents at homologous single bonds and recombines the fragments (BREED-style
#    scaffold hopping).
ga = GeneticOptimizer(objective, population_size=48, generations=20, seed=42)
de_novo = ga.optimize()                                   # corpus-initialised
seeded  = ga.optimize(seed_smiles=["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"])
leadopt = GeneticOptimizer(
    MultiObjective([QedObjective(), NoveltyObjective()]),   # drug-like AND novel
    population_size=48, generations=20, seed=42, mutation="scaffold",
).optimize(seed_smiles=["CC(=O)Oc1ccccc1C(=O)O"], corpus=["CC(=O)Oc1ccccc1C(=O)O"])
hopped = GeneticOptimizer(
    objective, population_size=48, generations=20, seed=42,
    crossover="scaffold_hop",                             # hybrids of two parents
).optimize(seed_smiles=["CC(=O)Oc1ccccc1C(=O)O"])

# 3) Reinforcement learning (REINVENT-style): train a char-level GRU prior on the
#    corpus, then fine-tune it by policy gradient toward the same objective
#    (running baseline + entropy bonus + prior anchor + experience replay).
prior = SmilesRnn(hidden_dim=96, seed=42)
prior.train_prior(DEFAULT_CORPUS, epochs=8)               # < 1 s
rl = ReinventOptimizer(objective, prior=prior, iterations=15, seed=42).optimize()

de_novo.to_dataframe().head(10)   # ranked leaderboard: rank, smiles, score, breakdown
```

* Both optimisers are **deterministic given `seed`** (GA: dedicated `random.Random`;
  RNN/RL: `torch.Generator` + per-iteration sampling seeds) and never crash on invalid
  SMILES — invalid candidates score 0.0 and are counted.
* **Novelty as an objective:** `NoveltyObjective` scores a candidate as
  `1 − max Tanimoto` against a reference corpus (default: the bundled corpus — a
  corpus member scores 0, a structurally distant molecule scores high). Inside a
  `MultiObjective` it acts as an anti-rediscovery constraint: one corpus look-alike
  vetoes the whole molecule.
* **Scaffold-preserving GA mode (`mutation="scaffold"`):** lead-optimisation style
  evolution — each offspring keeps its parent's Murcko framework verbatim while a
  random subset of decorations is resampled from a corpus-derived decoration
  vocabulary (built once per corpus, re-assembled with `Chem.molzip`). Acyclic or
  bare-ring parents (and any RDKit failure) fall back to the standard character
  mutation; the character splice crossover is bypassed in this mode (it destroys
  ring systems) — the scaffold-hop crossover is not. Honest scope: it is decoration
  resampling on a fixed framework, **not** a full scaffold-decorator generative model
  like REINVENT 2.0 (no learned attachment grammar).
* **Scaffold-hop crossover (`crossover="scaffold_hop"`):** BREED-style structural
  recombination — both parents are cut at a pair of *homologous* single bonds
  (join atoms match on element, degree, aromaticity, charge) and recombined
  with `Chem.molzip`. Six geometries, completing the {excised piece} × {cut
  type} matrix: classic **bridge cuts** swap fragments
  cross-wise (ring systems survive by construction; the offspring's framework
  differs from both parents whenever the parents' frameworks span different
  fragments — biaryls, diaryl ethers, sulfonyl bridges); **ring merges** zip
  two opened rings into one combined ring holding atoms from both parents;
  **linker transplants** excise a two-ended linker — a single atom (an ether
  O, a methylene, a secondary amine N) or a contiguous acyclic path of up
  to four atoms (an ethylene bridge, an oxyethylene `-OCH2-` segment) — from
  one parent and zip it into a cut ring
  bond of the other — ring homologation by one atom or a whole chain:
  piperidine + diethyl ether → `C1CNCCOC1` (a 7-ring oxazepane), piperidine
  + hexane → an 8-ring azocane via the transplanted `-CH2CH2-`;
  **fused-ring splices** (the full Nicolaou move set) excise a whole ring —
  any ring attached to its molecule through exactly two single anchor bonds,
  aromatic ones included — and splice it across the other parent's opened
  ring bond, giving fused / bridged bicyclic offspring that hold one ring
  from each parent: decalin × piperidine → `C1CCC2CCNCCCC2C1` (an azadecalin
  core), indane × piperidine → `c1ccc2c(c1)CCCCNC2` (a benzo-fused azocane —
  the aromatic donor ring survives intact, only its two aryl-CH2 anchor
  bonds are cut);
  **ring inserts** splice that same excised ring INTO the other parent's cut
  *acyclic* bond — both bridge halves re-join *through* the ring, so it
  becomes the spacer: ring-as-linker (decalin × hexane → `CCCC1CCCCC1CCC`,
  a dipropylcyclohexane; indane × heptane → `CCCCCc1ccccc1CC`, a
  dialkylbenzene);
  **segment swaps** (two-point crossover, the graph analogue of classic
  two-point GA recombination) exchange one parent's bounded acyclic
  *middle* — a single atom or a contiguous path of up to four atoms — for
  the other parent's homologous middle: the only size-conserving geometry
  and the medchem linker / bioisostere exchange — aspirin × paracetamol →
  `CC(=O)Nc1ccccc1C(=O)O` (salicylamide: the ester O swapped for the
  amide N), an ether O for a thioether S. Across the 244-molecule
  corpus the swap scan admits 7,328 candidates over 295 distinct
  linker chemistries — 1,738 of them heteroatom bioisostere exchanges
  (O↔N the classic ester/amide class).
  Aromatic ring bonds are never cut (the
  re-zip would have to invent bond orders). Also usable standalone via
  `scaffold_hop_crossover(parent_a, parent_b, rng)`; deterministic; `None`
  (fall back) when no viable cut exists. Combine with `mutation="scaffold"`
  (decoration resampling) or `mutation="topology"` (ring-size moves) for
  the full structural mode. Pinned demo: aspirin × naproxen →
  `COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1` (one fragment per parent).
* **Ring-topology mutations** (`ring_topology_moves` /
  `ring_topology_mutate`, the GA's `mutation="topology"` mode): the
  single-parent medchem moves — one atom removed from a saturated ring
  with its neighbours rejoined (**contraction**: piperidine →
  pyrrolidine, morpholine → pyrrolidine / tetrahydrofuran, decalin →
  hydrindane) or one methylene inserted into a saturated single ring
  bond (**expansion**: pyrrolidine → piperidine, piperidine → azepane).
  Aromatic and unsaturated rings are never touched; fused/spiro atoms
  and shared bonds stay put; the ring-size window is 5–8. Acyclic or
  aromatic-only parents fall back to character mutation — never an
  error. Pinned: piperidine's full move set is exactly
  `{cyclopentane, pyrrolidine, azepane}` (contract the N, contract a
  CH2, expand); 30 of the 244 corpus molecules carry viable moves (133
  distinct one-step mutants). The executed notebook runs the
  head-to-head on a ring-rich objective (7-membered-ring band × QED ×
  lead-size MW band): the topology arm reaches best 0.928 in generation
  2 while uniform mutation needs seven generations for 0.872, with the
  population mean ahead in every generation and ~4× the offspring
  survival.
* **Adaptive operator selection** (`GeneticOptimizer(..., operator_selection="adaptive")`,
  CLI `--adaptive-operators`): the GA learns *in-run* which operators pay
  off. Three biasable levels — crossover-vs-clone, the seven scaffold-hop
  crossover geometries (via `scaffold_hop_crossover(..., geometry_weights=...)`
  for manual biasing), and the mutation family — each keep a sliding-window
  credit book (`reward = (valid offspring ? score : 0) − mid-parent`, failed
  operators are charged, not forgiven) and draw through a softmax-with-floor
  rule (window 20, temperature 0.1, floor 1/(k·5): operators are demoted,
  never abandoned). Fully seeded-deterministic; the default
  `"uniform"` path is bit-identical to before; per-operator statistics
  (draws, successes, credit, final probability, best offspring) land in
  `result.metadata["operator_stats"]`. On the ring-rich objective from the
  executed notebook's A/B (7-membered-ring band × QED × lead-size MW,
  topology mode, seed 42): adaptive closes at population mean 0.808 vs
  uniform's 0.765 and *discovers* the topology family on its own (final
  draw probability 0.896 vs 0.104 for character mutation). The learning
  curve itself is one call away: `plot_operator_trajectory(result)` (from
  `pharmaml.viz`) stacks the run's score trajectory with one panel per
  adaptive level, drawing `result.metadata["operator_trajectory"]` — one
  full-precision record per generation (record 0 is the exactly-fair
  prior, the final record the closing mix).
* **Multi-objective trade-offs (Pareto front):** a `MultiObjective` run
  ranks by one blended number, which hides the trade-offs —
  `pareto_front(result)` (from `pharmaml.generative`) re-reads the run's
  per-objective breakdown and extracts the non-dominated set: the
  molecules where improving one objective necessarily sacrifices
  another. Front members can sit arbitrarily deep in the leaderboard
  (a real run below: 23 of 159 molecules non-dominated, the deepest at
  rank 154); `plot_pareto_front(result, x="qed", y="novelty")` renders
  the trade-off scatter with the front highlighted, and the returned
  `ParetoFront` carries domination counts, a `to_dataframe()` table and
  a one-line audit `__str__`.
* **Pareto pruning (NSGA-II crowding distance):** real fronts can outgrow
  the report itself — a three-objective GA over a 300-molecule population
  for 30 generations leaves 46 of 8,108 molecules non-dominated, and
  rank-truncating that table would drop the very corners the front exists
  to show. `front.crowding_distance()` ranks every member by isolation in
  objective space (+inf for the trade-off corners, low values for the
  crowded middle), `front.prune(8)` keeps the most isolated members with
  the corners guaranteed to survive (the selection is monotonic in `keep`
  and deterministic), and `to_dataframe()` carries the `crowding` column
  so the reader sees who earned their seat. `plot_pareto_front` encodes
  the same ranking visually: interior markers scale with isolation and
  the +inf corners are amber diamonds. The CLI twin is
  `--pareto-keep N` (e.g. `pharmaml generative run --recipe druglike
  --pareto --pareto-keep 8`): the summary honestly reports the original
  front size plus the cut — `pruned to the 8 most isolated of 46 by
  crowding distance`.
* **RL prior introspection** — the RL loop always *computed* the prior-drift
  diagnostics for its anchor loss; now it records them instead of discarding
  them. Every `result.history` row carries `drift` (the squared
  agent-vs-prior sequence log-probability gap — the exact quantity the
  `anchor_weight * mean((logP_agent - logP_prior)²)` penalty is built from),
  `entropy` (the mean per-step sampling entropy — the exploration meter), and
  both mean sequence log-probabilities. Two invariants make the story
  self-explaining: **iteration 0's drift is exactly `0.0`** (the agent starts
  as a bit-exact copy of the frozen prior, taking its first gradient step only
  after the diagnostics are computed — pinned by test), and the drift curve is
  the mode-collapse brake made measurable — probed at seed 7 over 8
  iterations, the final drift is 38.8 unanchored vs **0.74 with
  `anchor_weight=1.0`**. `plot_rl_introspection(result)` (from `pharmaml.viz`)
  stacks the four panels — score trajectory, agent-vs-prior log-probs (the two
  curves whose gap *is* the fine-tuning), the drift with the anchor weight
  annotated, and the entropy — and the CLI twin is
  `pharmaml generative run --method rl --recipe qed --plot drift.png` (the
  `--plot` flag, previously GA-only, now saves each method's own learning
  curve).
* **Method trade-offs, honestly:** the GA explores wider (hundreds of unique molecules
  per run, no extra dependency, seeded *and* de novo modes) while RL fine-tunes a learned
  prior and needs fewer evaluations to focus on high-scoring neighbourhoods — but costs a
  torch install and yields fewer unique hits at this scale. Run both, compare leaderboards.
* CLI twins: `pharmaml generative run --method ga|rl --recipe qed|logp|druglike|similar|novel`
  (with `--seeds examples/data/seed_molecules.csv` for the "with some sample molecules"
  mode; `--scaffold` switches the GA to scaffold-preserving mutations,
  `--scaffold-hop` to the BREED-style crossover and `--topology` to the
  ring-size mutation mode — e.g.
  `pharmaml generative run --method ga --recipe novel --scaffold --scaffold-hop --seed 42`
  is the full structural mode; with `--adaptive-operators`, `--plot <path>`
  saves the run's operator-trajectory figure in png/svg/pdf; with
  `--pareto` (multi-objective recipes) the front is printed after the
  leaderboard, `--pareto-plot <path>` saves its trade-off scatter, and
  `--pareto-keep N` prunes an unwieldy front to its most isolated members
  — e.g. `pharmaml generative run --recipe druglike --pareto
  --pareto-keep 8 --pareto-plot front.png`) and
  `pharmaml generative corpus`.
* Requires `pip install "pharmaml[chem]"`; the RNN/RL half additionally needs
  `pip install "pharmaml[deep]"` (PyTorch is imported lazily — the package works without it).

### Cell line data analysis (from 0 to 100)

A complete workflow for **cell line pharmacology** — GDSC/CCLE-style drug-response
screens paired with expression data — in the same "Zero to One Hundred" style as the
other domains. A fully executed notebook walks every stage:
`examples/complete_cellline_workflow.ipynb` (24 cell lines, 8 compounds, planted
EGFR → erlotinib biomarker recovered with rho=+0.978, FDR≈2.4e-13).

```python
from pharmaml.biology import (
    load_response_table, audit_response_table, replicate_variability,
    collapse_replicates, ln_ic50_to_pic50, response_matrix,
    summarize_compounds,
    call_sensitive_resistant, normalize_expression, most_variable_genes,
    sensitivity_associations, top_biomarkers, direction_of_association,
)

# 1) Load + audit a GDSC-style screen (long format: cell line, compound, LN_IC50).
screen = load_response_table("screen.csv", tissue_column="tissue")
print(screen.summary())
audit_response_table(screen)          # missing values, duplicate pairs, ...
print(replicate_variability(screen))  # pre-collapse spread of the duplicates (CV!)
clean = collapse_replicates(screen)   # median per (cell line, compound)
#    or keep the noise visible: collapse_replicates(screen, add_cv=True)
#    appends a {metric}_replicate_cv_pct column next to every median.

# 2) Normalize: GDSC LN_IC50 (ln of µM) -> pIC50 (bigger = more potent).
clean.frame["pIC50"] = ln_ic50_to_pic50(clean.frame["LN_IC50"])

# 3) The cell x compound potency landscape + per-compound ranking.
matrix = response_matrix(clean, "LN_IC50")       # cell lines x compounds
summary = summarize_compounds(clean, "LN_IC50")  # median/min/max/spread per drug

# 4) Call sensitive vs resistant cell lines by response quantiles.
labels = call_sensitive_resistant(matrix["erlotinib"], lower_is_sensitive=True)

# 5) Omics: expression matrix (rows = cell lines, columns = genes), log2 + z,
#    then pick the MAD-most-variable genes.
expr = normalize_expression(expression_frame)
genes = most_variable_genes(expr, n=40)

# 6) Biomarker discovery: every gene vs one drug's response, BH-FDR adjusted.
hits = sensitivity_associations(expr, matrix["erlotinib"].dropna(), method="spearman")
top = top_biomarkers(hits, n=10)               # gene, rho, p_value, fdr, significant
print(direction_of_association(top.iloc[0]["rho"], lower_is_sensitive=True))
```

* `ResponseScreen` carries the screen (cell lines, compounds, metrics, tissue);
  `response_matrix` pivots it for heat-map-style analyses.
* **Replicate noise stays visible**: `replicate_variability()` reports every
  duplicated (cell line, compound) pair's pre-collapse spread (n, median, mean,
  std, spread, CV% = std/|mean| — NaN for single observations or zero means,
  never a fabricated inf), and `collapse_replicates(..., add_cv=True)` keeps a
  `{metric}_replicate_cv_pct` companion column next to each collapsed median
  (annotations, not metrics — pivoting and summaries are unaffected). Two
  replicates `[1.0, 3.0]` and `[1.9, 2.1]` collapse to the *same* median 2.0;
  the CV tells you which one to trust.
* Sensitive/resistant calling is **direction-aware** (LN_IC50: lower = sensitive;
  pIC50: higher = sensitive) and documented per function.
* `sensitivity_associations` is the omics → biomarker step: per-gene Spearman/Pearson
  correlation against one drug's response vector, Benjamini–Hochberg FDR, sorted by
  effect size — the notebook's planted EGFR biomarker survives FDR≈2.4e-13.
* CLI twins: `pharmaml biology response-audit` (add `--replicate-cv` for the
  pre-collapse variability table) and `pharmaml biology top-biomarkers`.
* The notebook closes by predicting erlotinib response from expression with
  classical ML **and the `dnn` / `dnn_residual` deep-learning models in one
  `compare_models` leaderboard** — cell line data is where deep learning earns its
  keep.

### Pharmaceutical calculations

```python
import numpy as np
from pharmaml.pharma import (
    auc_trapezoid, clearance, creatinine_clearance,
    bsa_mosteller, dose_by_bsa, loading_dose, maintenance_dose,
)

# Non-compartmental PK from plasma samples
t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
c = np.array([0.0, 8.2, 6.1, 3.4, 1.1, 0.35])
auc = auc_trapezoid(t, c)                     # µg·h/mL

# Renal function & dosing
crcl = creatinine_clearance(age_years=68, weight_kg=70, serum_cr_mg_dl=1.1, sex="male")
bsa  = bsa_mosteller(height_cm=175, weight_kg=70)             # m²
dose = dose_by_bsa(dose_per_m2=250, height_cm=175, weight_kg=70)  # mg, optional max_dose cap

# Regimen design
ld  = loading_dose(vd=45.0, target_concentration=12.0)      # mg
md  = maintenance_dose(clearance_=3.2, target_css=6.0, tau=8.0)  # mg per interval
```

### Molecular docking

```python
import pandas as pd
from pharmaml.docking import (
    compute_grid_box, write_gpf, parse_vina_output,
    parse_dlg_cluster_table, ligand_efficiency, consensus_rescore,
    consensus_breakdown,
)

box = compute_grid_box("reference_ligand.pdbqt", padding=4.0)   # grid box around the co-crystal ligand
write_gpf("ligand.pdbqt", receptor_file="receptor.pdbqt", box=box, output_path="grid.gpf")

poses = parse_vina_output(open("vina.out").read())    # ranked poses (affinity, RMSD l.b./u.b.)
clusters = parse_dlg_cluster_table(open("r.dlg").read())  # AutoDock4 cluster histogram

le = ligand_efficiency(affinity_kcal_mol=-9.6, num_heavy_atoms=32)  # 0.30 kcal/mol per atom

scores = pd.DataFrame({"vina": [...], "ad4": [...]})  # one row per ligand
consensus = consensus_rescore(scores, columns=["vina", "ad4"])     # 0–1 combined score
breakdown = consensus_breakdown(scores, columns=["vina", "ad4"])   # + per-column _norm units — the explainable ranking
```

### Visualization & reports (every domain)

```python
from pharmaml.viz import (
    plot_confusion_matrix, plot_roc_curves, plot_predicted_vs_actual,
    plot_residuals, plot_model_comparison, plot_class_distribution,
    plot_feature_importance, plot_cv_scores,
    plot_calibration_curve, plot_learning_curve,
    plot_concentration_profile, plot_dose_accumulation,
    plot_descriptor_distributions, plot_similarity_heatmap,
    plot_table_comparison,
    plot_energy_distribution, plot_top_ligands, plot_consensus_ranking,
    HTMLReport, export_table, metrics_to_dataframe, save_visual_bundle, save_figure,
    export_docx, save_pdf_report,
)

fig = plot_confusion_matrix(report)          # ClassificationReport -> heat-map (2+ classes)
save_figure(fig, "confusion.png")            # png / svg / pdf by extension
fig2 = plot_roc_curves(model, X_test, y_test)   # binary or per-class one-vs-rest
fig3 = plot_calibration_curve(model, X_test, y_test)  # reliability + Brier scores
fig4 = plot_learning_curve(model, X, y, cv=5)  # is more data still buying accuracy?
fig5 = plot_table_comparison(comparison)  # ligand-set overlap (compare_tables result)
fig6 = plot_consensus_ranking(rescored)   # the explainable leaderboard as ranked bars

export_table(leaderboard, "leaderboard.md")  # csv / tsv / markdown / json / html
export_docx(leaderboard, "leaderboard.docx")  # a real Word document (python-docx)

save_visual_bundle(                          # one call -> report.html + files
    figures={"confusion": fig, "roc": fig2},
    tables={"leaderboard": leaderboard},
    metrics=report,
    out_dir="reports",
    pdf_report=True,     # + report.pdf: the multi-page document (the legacy
                         #   scripts' fpdf output, upgraded — title + metrics,
                         #   one page per table, one page per figure)
)

save_pdf_report(                             # or standalone, without the bundle
    {"confusion": fig}, tables={"leaderboard": leaderboard},
    metrics=report, out_path="model_report.pdf",
)
```

matplotlib is optional (``pip install 'pharmaml[reports]'``); the Agg backend is
forced automatically outside notebooks, so plotting works on headless servers
and CI — and renders inline inside Jupyter.

---

## Command-line interface

```
pharmaml --version
pharmaml models                       # list available registry models
pharmaml data inspect --data my.csv --target activity --head 5
pharmaml data split --data my.csv --test-size 0.2 --val-size 0.1
pharmaml data convert --data ligands.csv --to sdf          # CSV/TSV/Excel -> SDF (3D + UFF, SD tags)
pharmaml data convert --data ligands.csv --to sdf --no-3d  # fast 2D write
pharmaml data convert --data library.smi --to csv          # streaming .smi -> CSV (Excel BOM)
pharmaml data convert --data compounds.sdf --to csv        # SDF -> table (SMILES + SD tags)
pharmaml data convert --data ligands.csv --to smi          # table -> .smi lines
pharmaml ml train --data my.csv --target activity --task auto
pharmaml ml train --data my.csv --model rf --cv 5 --plots --keep-categorical
pharmaml ml compare --data my.csv --target activity --out board.csv --plot
pharmaml ml tune --data my.csv --estimator ann --max-trials 12 --objective r2    # keras-tuner-style search
pharmaml ml tune --data my.csv --estimator dnn --searcher hyperband              # successive halving over epochs
pharmaml ml tune --data my.csv --estimator hsvr --save-best tuned.joblib         # hierarchical SVR, refit winner
pharmaml chem pic50 --data data.csv --column "IC50 (nM)" --unit nM
pharmaml chem descriptors --data data.csv --smiles-col SMILES
pharmaml chem rism --data data.csv --smiles-col SMILES --solvent spc             # 3D-RISM-KH solvation descriptors
pharmaml chem compare --data-a d_summary.csv --data-b ligands.csv --column Ligand --figure overlap.png
pharmaml chem merge --dir batches/ --first-columns SMILES,pIC50
pharmaml pharma pk --data conc.csv --time time --conc conc
pharmaml pharma dose --cl 3.2 --vd 45 --target-css 6.0 --tau 8 --f 1.0
pharmaml pharma approvals --data 1.tsv --data 2.tsv --out approved.csv
pharmaml docking prep-ligands --ligands ligands.csv --out ligand_folders/
pharmaml docking prep-protein --protein protein.pdb --out r.pdbqt
pharmaml docking dpf --ligand ligand_folders/1/l.pdbqt --ga-runs 100   # AutoDock4 r.dpf
pharmaml docking scan --directory runs/ --log-name r.dlg
pharmaml docking check --directory runs/
pharmaml docking torsion --dir ligand_folders/    # TORSDOF/REMARK/BRANCH consistency
pharmaml docking gpf-scan --dir runs/             # ligand_types census per r.gpf
pharmaml docking vina-scan --dir vina_results/    # best affinity per .log
pharmaml docking rescore --data summary.csv --columns best_lowest_energy,cluster_size --weights 1,3    # explainable 0–1 consensus (per-column _norm units)
pharmaml docking rescore --data summary.csv --columns best_lowest_energy,cluster_size --plots    # + report.html with the consensus-ranking figure & breakdown table
pharmaml docking collect --dir runs/ --pattern '^\\d+\\.pdbqt$'  # gather artifacts
pharmaml docking campaign list --dir ligand_folders/ --ext .pdbqt --out names.csv
pharmaml docking campaign distribute --dir ligand_folders/ --file receptor.pdbqt
pharmaml docking campaign cleanup --dir runs/ --keep r.dlg --remove-empty --dry-run
pharmaml docking campaign delete-named --dir runs/ --name r.gpf --dry-run
pharmaml docking campaign reindex --data results.csv --column name --start 1 --stop 300  # gap audit + reindex
pharmaml biology response-audit --data screen.csv --tissue-column tissue --metric LN_IC50
pharmaml biology top-biomarkers --expression expr.csv --response resp.csv --response-column pIC50 --compound erlotinib --top 15
pharmaml generative run --method ga --recipe qed --iterations 20 --seed 42
pharmaml generative run --method ga --recipe qed --adaptive-operators --plot operators.png
pharmaml generative run --method rl --recipe druglike --iterations 15 --out molecules.csv
pharmaml generative run --method ga --recipe similar --seeds examples/data/seed_molecules.csv
pharmaml generative corpus --out corpus.csv
```

`--model` accepts registry names and aliases (`rf`, `logistic`, `xgb`,
`mlp`, `svm`, … — see `pharmaml.ml.MODEL_ALIASES`).

`ml train --plots` emits the full visual bundle — `report.html` plus
confusion matrix, ROC curves, class distribution, CV fold scores and
feature-importance figures (parity + residual diagnostics for regression)
— into `<out>/report/`, no Python required. Adding `--pdf-report`
(implies `--plots`) also writes `report.pdf` — the multi-page document
with a title/metrics page, one page per table and one page per figure;
`ml compare --pdf-report` does the same for the leaderboard bundle
(title + leaderboard + chart, three pages). `ml compare --plot` saves the
leaderboard bar chart and `--out` exports the table in any of
csv/tsv/md/json/html; `generative run --adaptive-operators --plot <path>`
saves the adaptive GA's operator-trajectory figure, and
`generative run --recipe druglike|novel|similar --pareto` prints the
multi-objective Pareto front after the leaderboard (with
`--pareto-plot <path>` saving the trade-off scatter). All plotting flags
require the `reports` extra (`pip install 'pharmaml[reports]'`).

---

## Error handling & logging

```python
from pharmaml import DataValidationError, DependencyError, PharmaMLError
from pharmaml.logger import get_logger

log = get_logger("my-analysis", level="INFO")   # console; add log_file="run.log" for a file copy
```

* Every public function validates inputs and raises a **domain-specific exception**
  (`DataValidationError`, `UnitConversionError`, `DockingError`, `DockingParseError`,
  `DependencyError`) — all subclasses of `PharmaMLError`, so one `except` clause catches everything.
* Workflows log progress (folds, grid-search steps, docking runs) through the shared logger —
  never bare `print()`.

---

## Testing

```bash
python -m pytest                 # full suite (1539 tests incl. doctests, skips RDKit-only tests if absent)
python -m pytest tests/test_pk.py -v
python -m pytest --cov=pharmaml  # with pytest-cov
```

The suite covers: PK/PD/dosage math (against hand-computed references), unit conversion
round-trips, ML pipeline behaviour (splits, metrics, persistence round-trip), **user-data
loading (CSV/Excel, delimiter sniffing, target auto-detection, multiclass and regression
inference)**, **multiclass end-to-end flows (3–4 classes, scorer fallbacks, ensembles)**,
**every plotting function / export format / HTML report path (headless Agg CI proof,
plus a guard that the shipped notebooks stay executed and error-free, and a
no-inline-fabrication guard: 6 executed notebooks × 9 banned fabrication tokens = 54
assertions with zero allowlist entries, so the notebooks can never go back to making
their own sample data instead of loading `examples/data/`)**, docking
preparation & parsers (ligand CSV import, receptor prep incl. the mmCIF/gemmi path,
synthetic DLG/Vina/PDBQT fixtures), scoring functions, converter edge cases, and the
SMI/CSV/SDF format-conversion round-trips.

## CI

`.github/workflows/ci.yml` runs the matrix (Python 3.10–3.12): install → ruff → mypy (0-error baseline, blocking) → pytest (doctests included).

## Docker

```bash
docker build -t pharmaml .
docker run --rm -v "$PWD:/work" pharmaml ml train --data /work/data.csv
```

---

## Design choices & trade-offs

1. **scikit-learn as the ML backbone** instead of raw PyTorch — QSAR tabular data is the sweet
   spot for sklearn; `ann` (MLPClassifier) covers the neural case, and the registry design makes
   adding a PyTorch model a single file change. Trade-off: no GPU training for tabular models
   (rarely needed at this data scale).
2. **Lazily-imported optional dependencies** — the package imports cleanly on a bare Python +
   numpy/pandas/sklearn environment; missing extras fail *at call time* with install hints.
   Trade-off: a slightly more complex import path, worth it for portability.
3. **Pure-Python docking parsers** — DLG/Vina/PDBQT parsing has no external dependency, so
   result analysis works even on machines without AutoDock installed. Execution wrappers
   (`run_autodock4`, `run_vina`, `batch_run_directories`) locate binaries via `shutil.which`
   and raise `DockingError` with guidance instead of `shell=True` surprises.
4. **Explicit unit conventions over a unit-library** — PK functions document their units in the
   docstring instead of pulling in `pint`. Trade-off: caller responsibility, but zero-dependency
   math functions are trivially testable and fast.
5. **One shared validation layer** (`pharmaml.utils.validation`) — every domain raises the same
   `DataValidationError` with the parameter name in the message, keeping error messages uniform.

## License

MIT.
