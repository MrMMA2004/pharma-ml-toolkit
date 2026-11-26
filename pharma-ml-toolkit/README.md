# PharmaML Toolkit

A **research-oriented Python toolkit** for computational pharmacy, with typed APIs for data handling, machine learning, biology, molecular docking, PK/PD and cheminformatics. It is designed to support exploration and reproducible workflows; its presence or test coverage does not establish scientific or clinical validity.

| Area | Scope |
|---|---|
| **`pharmaml.data`** | Load CSV/TSV/Excel tables, inspect columns and targets, prepare splits, and check exact-key overlap. |
| **`pharmaml.ml`** | QSAR and tabular-model workflows: preprocessing, training, evaluation, comparison, tuning, and selected uncertainty or interpretation tools. |
| **`pharmaml.chem`** | Optional RDKit/Mordred-backed descriptors, fingerprints, similarity, standardization, and molecular-file conversion. |
| **`pharmaml.docking`** | Ligand/receptor preparation, campaign-file handling, result parsing, and scoring; engine execution requires compatible external tools. |
| **`pharmaml.pharma` / `pharmaml.pkpd`** | Pharmaceutical calculations, PK/PD analysis, and simulation; verify units, model assumptions, and intended use. |
| **`pharmaml.biology`** | Drug-response table and biomarker workflows, including cell-line data; these are not patient-response evidence. |
| **`pharmaml.generative`** | Research workflows for molecular generation and objective-based exploration; outputs need chemical and experimental validation. |
| **`pharmaml.viz`** | Scientific plots and report/export helpers; optional dependencies apply. |

Optional packages and workflow coverage vary by task. See [`docs/RESEARCH_WORKFLOWS.md`](docs/RESEARCH_WORKFLOWS.md), the [API index](docs/INDEX.md), and the relevant tests for detailed assumptions and evidence.

The package is a structured alternative to a collection of one-off research scripts. Public modules use type annotations, validation and consistent error handling where implemented; test coverage varies by workflow. Tests check software behavior, not the validity of a scientific method or result.

---

## Design assumptions

* **Python ≥ 3.10** (union syntax `X | None` used throughout).
* **Core deps only**: `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib` — everything else
  (RDKit, Mordred, XGBoost/LightGBM/CatBoost, Meeko, gemmi) is **optional and lazily imported**;
  missing packages raise an informative `DependencyError` instead of crashing at import time.
* **Local-first**: the package is designed for local use and has Docker/CI support; optional dependencies and native engines vary by platform and workflow.
* **Deterministic examples**: the executed notebooks run entirely on the shipped
  `examples/data/` inventory (45 files recursively — every input table, the demo receptor and
  the demo docking campaign; see the tree below), regenerated deterministically by
  `tools/build_example_data.py`; the example scripts keep their fixed seeds — no external
  datasets required. A pytest freshness guard also pins every executed notebook's
  committed outputs to the *shipped* version (the setup cell prints
  `pharmaml <version>`), so the committed evidence can never silently predate the
  tree that ships it — after a version bump, `tools/reexec_notebooks.py`
  re-runs all eight notebooks (runtime depends on the environment); then run
  `python tools/strip_notebook_outputs.py` to regenerate the output-free variants.
* **Input checks**: many public functions validate units, ranges and required columns and raise domain-specific errors for invalid inputs. Review each function's documented assumptions; validation does not make a calculation patient-specific or clinically suitable.
* **Research-catalog provenance**: `docs/GITHUB_RESEARCH_CATALOG_AUDIT.md` records the live, one-by-one disposition of all 140 catalog repositories, including absorbed library/API work, interface surfaces and explicit non-parity boundaries.
* **Docs stay in sync with code**: `docs/INDEX.md` is auto-generated from each module's `__all__`
  and docstrings (`python tools/build_docs_index.py` from the toolkit root); a drift-guard test
  (`tests/test_docs_index.py`) fails whenever the index and the live package diverge, so the
  documented symbol list can never silently rot.
* **Separate Next.js console — dataset parity**: its dataset-preview route (`/api/ml/dataset`)
  uses the same seeded generators as its training route (`/api/ml/train`) — identical
  `mulberry32` PRNG, `make_regression_data`, and `makeMoons` implementations — so previewed rows,
  statistics, and scatter plots match the data used by that console's training route.
* **Separate Next.js console — shareable line references**: its code explorer copies `path#LNN`
  references on line clicks and accepts them in its search box (validated against the file tree,
  then highlighted), so a doc review, PR comment, or chat message can point peers at an exact
  line of the package.
* **Separate Next.js console demonstration**: the root project includes its own ML Playground.
  A parity test compares its Pearson-correlation helper with `scipy.stats.pearsonr`
  (`tests/test_web_parity.py`, worst |Δr| = 8.3e-17 over an 11-case seeded grid). These facts
  do not imply feature parity with the Python package or React/Tauri Studio.

---

## Installation

PharmaML is distributed through its source repository and GitHub releases; it is not
currently published on PyPI. Run the commands below from this directory, the inner
`pharma-ml-toolkit/` package root.

**Pinned-core route (recommended):** `requirements.txt` pins the tested
core stack — install it first, then install the checked-out package without
re-resolving its dependencies.

```bash
# 1. every core library at the verified version
#    (numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, scipy 1.14.1, ...)
python -m pip install -r requirements.txt

# 2. the toolkit itself (editable — `import pharmaml` works immediately)
python -m pip install --no-deps -e .
```

`requirements.txt` contains the pinned Python 3.12 core snapshot; optional
extras in `pyproject.toml` remain range-resolved. Commented pins for selected
optional stacks are provided as a reference, but are not automatically
activated. You can uncomment the pins you need and install with `-r` for a
pinned stack, or use the source-local extras below with `requirements.txt` as a
constraints file to preserve the core pins while resolving the extra within its
declared compatibility range:

```bash
# with cheminformatics extras (RDKit descriptors/fingerprints/similarity)
python -m pip install -c requirements.txt -e '.[chem]'

# with docking preparation extras (RDKit + Meeko: SMILES -> 3D PDBQT)
python -m pip install -c requirements.txt -e '.[docking]'

# with gradient-boosting model families
python -m pip install -c requirements.txt -e '.[boosting]'

# with PyTorch estimators and molecular-generation models
python -m pip install -c requirements.txt -e '.[deep]'

# with plotting, HTML/PDF/Word reports, and image export
python -m pip install -c requirements.txt -e '.[reports]'

# with the SELFIES codec
python -m pip install -c requirements.txt -e '.[selfies]'

# with SHAP-based tree explanations
python -m pip install -c requirements.txt -e '.[explain]'

# development (pytest, mypy, ruff)
python -m pip install -c requirements.txt -e '.[dev]'
```

Extras in `pyproject.toml` use compatibility ranges; `-c requirements.txt`
preserves the tested Python 3.12 core pins while resolving them. Optional
versions themselves remain range-resolved. For a fully pinned feature stack,
uncomment its optional pins in `requirements.txt` and install with `-r`.

> The `studio` extra in `pyproject.toml` is retained for the retired
> PySide6 source only. It does not install the supported desktop app;
> use the React/Tauri development instructions below instead.

A quick test run after installing the development dependencies is a starting
point, not full release validation. Use the [named test profiles](docs/TEST_PROFILES.md)
and record their results separately.

```bash
python -m pytest
```

---

## Graphical interface — PharmaML Studio

The supported desktop application is the **React + TypeScript frontend** in `frontend/`, hosted by **Tauri 2** in `src-tauri/`. It provides nine workspace areas and calls the local FastAPI sidecar for scientific operations. The workspaces expose documented workflows; algorithm coverage and optional-dependency requirements vary by task. The Python package and CLI can also be used without the desktop app.

For frontend development and tests, start from this toolkit directory:

```bash
cd frontend
npm ci
npm test
npm run typecheck
npm run build
```

A browser session with live scientific workflows also needs a local backend and the development-only Vite proxy settings; see [`frontend/README.md`](frontend/README.md). The native Tauri shell requires a compatible frozen Python sidecar staged under `src-tauri/resources/backend/`; that generated Windows resource is not included in a fresh source checkout, so `cargo tauri dev` is not a turnkey launch command. The [Windows release workflow](../.github/workflows/windows-release.yml) stages the sidecar before packaging.

The former PySide6 launcher is **retired and unsupported**; `legacy/studio.py` is kept for archival and reproducibility only and is excluded from production packaging. Do not use `python studio.py` or the `.[studio]` extra as a route to the supported desktop app. The repository's Next.js console is a separate web application, not Studio.

## Versioned downloads & updating

The downloadable archive is version-stamped: it downloads as
`pharma-ml-toolkit-v<version>.zip`, carries its version in the `VERSION` file
(and in `pyproject.toml` → `[project] version`), and the separate Next.js
console's Download tab shows the live version next to the size. `CHANGELOG.md` at the
archive root documents every release; versions follow semver — minor bumps
add features, patch bumps ship fixes.

**Updating an existing copy needs no uninstall.** Unzip the new archive over
your current folder — `-o` overwrites without prompting:

```bash
unzip -o pharma-ml-toolkit-v1.2.242.zip -d pharmaml   # your existing folder
cat VERSION                                         # what you have now
```

Because the install is editable (`python -m pip install --no-deps -e .`), Python picks up replaced files on the next run. If core dependencies changed, install the updated `requirements.txt` first and reinstall the package with `--no-deps`; for an added optional extra, use `python -m pip install -c requirements.txt -e '.[extra]'` from this directory.
`cat VERSION` prints the toolkit version in one place. Files removed
between versions are simply left behind (harmless — v1.2.141's retirement
of `playground.py` is the example: delete it if you unzip over an older
copy); delete the folder first only for a truly clean slate.

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
├── legacy/studio.py        # RETIRED legacy GUI launcher (part18 §10): quarantined,
│                            # unsupported, archival only. The retired GUI (directive #16
│                            # pages): sidebar navigation, PyQtGraph scientific plots,
│                            # studio-token theming (Qt-Material/QDarkStyle optional), QThreadPool workers with
│                            # the cooperative Stop; runs offscreen for CI (`ui/headless.py`
│                            # preloads the committed EGL stub when the box has none); the
│                            # legacy Tkinter playground.py (declared retired v1.2.141) left the
│                            # shipped tree in round 304 — 30,047 lines, git history keeps it
├── ui/                      # retired PySide6 UI implementation used by legacy/studio.py; archival only
│                            # do not confuse this source with the supported React/Tauri Studio in frontend/.
├── VERSION                 # the release version — `cat VERSION` after extracting (kept in
│                            # sync with pyproject + src/pharmaml/__init__ by the ship ritual)
├── CHANGELOG.md            # what changed in each versioned release
├── LICENSE                 # MIT — the pyproject declaration, materialised
├── docs/COVERAGE.md        # the one-by-one RAR audit: every original file → its PharmaML surface
├── docs/LEAKAGE_AUDIT.md   # the data-leakage audit: the rar's 28 leaky files, the toolkit's guards, the web mirror's fix
├── docs/DATA_AUDIT.md     # the database audit: every shipped data file checked for gaps and errors (2 real ones fixed)
├── docs/RELEASE_READINESS.md # the Phase-6 evidence report (part6 §16): dependency matrix, capability coverage, §14.2-chunkened suite results, carrier agreement, archive audit — regenerate via tools/release_readiness.py
├── tests/                   # Python/library tests, API tests, and notebook→builder guards; tests/ui/ are for the retired PySide6 implementation, while frontend tests live under frontend/.
├── examples/                # runnable demos + eight executed notebooks (complete ML · complete docking · complete PK/PD · complete cell line · complete molecule generation · complete deep learning · complete tuning & hierarchical · walkthrough) + seven no-output twins
│   └── data/                # the shipped demo inventory — 43 files, ~720 KB (audited 526-check clean, docs/DATA_AUDIT.md), every executed notebook
│                            #   loads its inputs from here (never fabricates sample data);
│                            #   regenerate deterministically via tools/build_example_data.py
│       ├── qsar_classification.csv        # ML §1      420 × 11 (active 118 / weak 155 / inactive 147)
│       ├── screening_library.csv          # ML §2       79-row SMILES library (77 valid + 2 planted invalid)
│       ├── library_with_descriptors.csv   # ML §3       77 × 10 (8 key descriptors + activity)
│       ├── qsar_multiclass.csv            # ML §8      360 × 21 (inactive 125 / weak 102 / moderate 79 / strong 54)
│       ├── qsar_regression.csv            # ML §9      260 × 21, pIC50 clipped to the assay window [3, 11]
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
│       ├── walkthrough_compounds.csv      # walkthrough  180 × 8 (active 39 / inactive 88 / weak 53; SMILES = the 12-molecule seed vocab)
│       ├── walkthrough_classification.csv # walkthrough  300 × 9 binary target
│       └── walkthrough_multiclass.csv     # walkthrough  300 × 9, 3 classes
│                            # Provenance: every file is byte-identical to what the corresponding
│                            #   notebook cell originally generated (same seeds, same code path — see
│                            #   the map in tools/build_example_data.py). PDBs come from RCSB entries
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
> §11 **the full registry tour** (all 26 families fitted on the same splits, held-out
> scores, unavailable engines listed with their missing requirement — the
> self-updating coverage guarantee) → §12 feature importance → §13 save/load &
> scoring new compounds → §14 one-click HTML report + png/svg/pdf + csv/md/json.

Open it, replace the demo file with your own CSV (and the §2 library with your
own SMILES column), and re-run all cells. §2 needs RDKit —
`python -m pip install -e '.[chem]'`.

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
`python -m pip install -e '.[docking]'`; a ready-to-use demo receptor ships at
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

Each workflow notebook also has an **output-free variant**
(`*_no_output.ipynb`) with the same cell sources and saved outputs removed,
for smaller downloads. Run it top-to-bottom to produce the results yourself.
A pytest drift guard checks that each variant's cell sources match its
executed notebook.

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
RDKit (`python -m pip install -e '.[chem]'`); §7 additionally needs PyTorch
(`python -m pip install -e '.[deep]'`).

### The complete deep learning process in one notebook

`examples/complete_deep_learning_workflow.ipynb` is the sixth workflow
notebook — a Python API walkthrough of the PyTorch estimators in
`pharmaml.ml.neural`, executed with all outputs and six inline figures
committed in ~15 s on a laptop CPU. Where the cell-line notebook only meets
`dnn` through the `compare_models` leaderboard, this one walks the **direct
estimator API**. It does not claim feature parity with PharmaML Studio or
the separate Next.js console:

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
> exported as CSV → §10 recap with a Python/CLI cross-reference.

Point §1 at your own descriptor table and every cell below runs unchanged
(same story as the other notebooks). PyTorch is the only extra:
`python -m pip install -e '.[deep]'`.

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
honest verdict on a small standardised table → §7 recap with the CLI
commands (`pharmaml chem rism`, `pharmaml ml tune`). Interface coverage is
documented separately; this notebook does not claim desktop feature parity.

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
# regression? pass the train split too and the report gains the
# overfit diagnosis: pearson_r, train_r2 and r2_gap (train − test).
report = evaluate_model(result.model, X_test, y_test, ds.task,
                        X_train=X_train, y_train=y_train)  # regression only
```

What it does for you:

* **Target auto-detection** — when `target` is omitted, common column names
  (`Target`, `class`, `label`, `activity`, `pIC50`, `y`, …) are tried, case-insensitively.
* **Task inference** — `infer_task()` treats any non-numeric target (or a numeric target
  with 2–`max_class_count` distinct values) as classification and continuous targets as
  regression; pass `task=` explicitly to override.
* **The overfit gap as a number** — for regression, `evaluate_model`
  accepts the training rows (`X_train=` / `y_train=`) and fills the
  report's `pearson_r`, `train_r2`, `train_pearson_r` and `r2_gap`
  (train R² − test R², signed) — an overfit *diagnostic*, never a
  quality claim, and the pair is all-or-nothing and regression-only
  (a classification train side has no rmse to mirror).
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
pharmaml ml ad     --data my_compounds.csv --model ridge --out Applicability_Domain_Results.csv
pharmaml ml ad     --bundle artifacts/rf.joblib --train-data train.csv --test-data test.csv --plot
```

`--task auto` (the default) infers classification vs regression from the target column, so
the same command works for a binary, multiclass or pIC50 endpoint file. `ml ad` is the
uploaded `Applicability_domain_analysis.py` workflow: the kNN-distance domain (is the query
inside the training descriptor space?), the bootstrap-ensemble spread (how sure is the
model?) and one reliable/not-reliable verdict per compound — `--bundle` reproduces the
upload's train/test-table flow verbatim, `--data` trains and judges the held-out rows.

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
"r.pdbqt")` converts them via gemmi (included in the source-local `.[docking]`
extra, installed with `python -m pip install -e '.[docking]'` from the toolkit
package root) before the same cleaning pipeline; the receptor's AutoDock atom-type census is one command away
with `pharmaml.viz.plot_receptor_atom_types(receptor)`. The pose-level reading is
one command further: pair a pose with the receptor and `compute_contacts`
classifies every atom pair in the 5 Å shell (salt bridge / H-bond /
hydrophobic / vdw distance bands), `plot_contact_map` draws the residue ×
ligand-atom grid and `plif_matrix` answers the per-residue interaction
counts — the shipped demo pair (real `lysozyme_1aki.pdb` + the synthetic
`contacts_demo_ligand.pdbqt`) reads 91 contacts over 25 residues.

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
`plsr` / `partial_least_squares`), `gaussian_process` (**kriging** — a
kernel-likelihood surrogate whose every prediction carries its own σ
(`predict_with_uncertainty`), the active-learning workhorse for scarce
observations; regression only, aliases `gp` / `gpr` / `kriging`),
`gaussian_process_classifier` (**Laplace GP classifier** — a latent GP
squashed through the logistic likelihood, Newton-mode posterior, calibrated
5-erf probabilities that hedge toward 0.5 away from the data; aliases
`gpc` / `gp_classifier`), `hsvr`
(**hierarchical SVR** — k-means local
specialists blended with a global SVR, regression only), plus `xgboost` / `lightgbm` /
`catboost` when the optional packages are installed — and two **deep-learning families**:
`dnn` (deep feed-forward MLP) and `dnn_residual` (residual net with learned skip
projections), both classification **and** regression, powered by PyTorch
(`python -m pip install -e '.[deep]'`).
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

#### Out-of-fold stacking (`stack_oof_regressors`) — the look.ipynb blender

When one model is not enough, stack them *honestly*: base models'
shuffled-KFold out-of-fold predictions build the meta-feature matrix (a
blender never scores a row it was trained on), the bases refit on the full
train set for test meta-features, and the meta-learners compete on a
`test_r2`-ranked leaderboard — the winner gets a Ridge residual-corrector
fitted on its own OOF residuals:

```python
from sklearn.model_selection import train_test_split
from pharmaml.ml import stack_oof_regressors, select_kbest_union

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
# optional: the univariate union selector (fit on train only)
X_tr, X_te, kept = select_kbest_union(X_tr, y_tr, X_te, k=256)

stack = stack_oof_regressors(
    X_tr, y_tr, X_te, y_te,
    base_models=["ridge", "random_forest", "gradient_boosting", "extra_trees"],
    meta_models=["ridge", "huber_regression", "lightgbm"],   # None → that default
    n_splits=5, residual=True,
)
stack.leaderboard          # model / oof_r2 / test_r2 (+ '<best>+residual' row)
stack.best_meta, stack.best_test_r2, stack.residual_test_r2
```

`target_transform="quantile"` threads the notebook's
`QuantileTransformer(output_distribution='normal')` through the target
(fitted on the training target only); every score is then on the
TRANSFORMED scale — the notebook's own convention, stated honestly in the
docstring. Deterministic end to end (seeded KFold, seeded MI, seeded
transformer); a failing meta-learner keeps a NaN row and sinks instead of
killing the leaderboard. Upgrades `look.ipynb` — whose
uploaded cell carried three syntax slips and six duplicate descriptor
columns (MolWt et al. are already inside `Descriptors.descList`), fixed
here, not ported.

#### Prediction intervals (`quantile_prediction_intervals`) — the D2.py uncertainty block, ported honestly

How wide a bracket does a prediction deserve? The quantile-regression
answer: fit gradient-boosted quantile losses at the `α/2`, `0.5` and
`1−α/2` quantiles, and report the empirical coverage of the bracket
against the held-out truth:

```python
from pharmaml.ml import quantile_prediction_intervals

result = quantile_prediction_intervals(X_train, y_train, X_test, y_test, alpha=0.1)
result.coverage          # e.g. 0.9 — share of test rows inside the bracket
result.mean_width        # mean of (upper - lower)
result.median            # the middle fit's point predictions
result.n_crossing        # quantile crossings, counted and repaired (swapped)
result.as_frame(y_test)  # actual / pred_median / pred_lower / pred_upper / interval_width
```

The companion figure is `pharmaml.viz.plot_prediction_intervals`
(a parity scatter with the sorted-by-actual band and the coverage box —
the GP ±σ plot's twin for bracket-shaped uncertainty).
Deterministic by default (`subsample=1.0` has no stochastic element; a
stochastic configuration is seeded through `random_state`).
The ensemble-std sibling `bootstrap_uncertainty` answers a different
question — how much the *model* wobbles under resampled training data,
not where the *residual noise* puts the truth.
Upgrades `D2.py` — whose `generate_uncertainty_quantification` trained
three IDENTICAL plain `RandomForestRegressor`s (no quantile loss:
`RandomForestRegressor` has no quantile mode in scikit-learn at all),
so its advertised "90% Prediction Interval" had width 0.0 on every row
and empirical coverage 0.0% — fixed here, not ported.

#### The interpretability bundle (`chem.interpret`) — decode_morgan_bit + signed correlations

A fingerprint column finally *means* something. The D2.py upload's
best remaining interpretability idea, closed as four first-class
surfaces:

```python
from pharmaml.chem import decode_morgan_bit, signed_feature_correlations, filter_low_variance
from pharmaml.viz import plot_signed_correlations, plot_correlation_heatmap

# 1. Which atoms make this ECFP bit fire?
record = decode_morgan_bit("CC(=O)Oc1ccccc1C(=O)O", 73)   # aspirin
record.hits               # every (atom, radius) environment the bit fires on
record.hits[0].environment_smiles   # the readable chemistry
record.drawing            # the RDKit bit legend as an SVG string

# 2. Which features pull the target up, which drag it down?
corr = signed_feature_correlations(ds.X, ds.y)    # |r| descending, sign kept
plot_signed_correlations(corr, top=10)            # teal up, warm down
plot_correlation_heatmap(ds.X, columns=corr.head(10).index)

# 3. Cut the columns that never move (the xxx.ipynb filter cell).
kept = filter_low_variance(ds.X, threshold=0.01)
kept.frame, kept.dropped    # the survivors + every cut's variance as evidence
```

Upgrades `D2.py` honestly: the original's decode passed the
FINGERPRINT radius to `FindAtomEnvironmentOfRadiusN` while discarding
the hit's own radius (for a bit that fired at radius 1 it drew the
radius-2 environment — a real decode bug, fixed and pinned) and
silently dropped every radius-0 hit; its correlation bundle
correlated only the first 1000 features "for speed" and appended a
fake `0.0` wherever `pearsonr` raised (a failed column masquerading
as "no correlation" — dropped, never zeroed, here). The cross-model
twin `cross_model_importances` + `plot_permutation_heatmap`
(`pharmaml.ml` / `pharmaml.viz`) renders the features × models matrix
the upload drew over its base models and meta-learner — same seed per
model so differences reflect the *models*, not the shuffle luck, and
a feature a model never saw renders grey ("no vote"), not a fake
zero.

#### Unsupervised analysis (`ml.unsupervised` + `ml.manifold`) — the PDSH handbook's chapters 05.09–05.13

The Python Data Science Handbook's unsupervised block, reviewed
notebook by notebook and absorbed as typed, seeded, guard-railed
surfaces (the per-notebook disposition table: `docs/COVERAGE.md`):

```python
from pharmaml.ml import pca_reduce, kmeans_cluster, gmm_cluster, kde_fit
from pharmaml.ml import manifold_embedding
from pharmaml.viz import (plot_pca_variance, plot_kmeans_diagnostics,
                          plot_gmm_selection, plot_kde_density,
                          plot_manifold_embedding, plot_hexbin_density)

pca = pca_reduce(descriptors)                     # variance ladder + loadings
km = kmeans_cluster(descriptors)                  # k=None: silhouette sweep picks k
gmm = gmm_cluster(descriptors)                    # n=None: BIC sweep picks components
kde = kde_fit(frame["pIC50"])                     # bandwidth by CV log-likelihood
emb = manifold_embedding(descriptors, "isomap")   # n_neighbors=10, seeded
plot_pca_variance(pca)                            # every record has its figure twin
```

Every call threads `random_state=42`; every bad input (NaN rows,
non-numeric columns, violated perplexity/neighbourhood rules, the
O(n²) sample ceilings for MDS/t-SNE) is refused with the typed
`DataValidationError` naming the problem; every record carries
`as_frame()` (the CSV twin) and a `__str__` summary. The naive-Bayes
family from the same handbook (`gaussian_nb` / `multinomial_nb` /
`bernoulli_nb`) joins `MODEL_REGISTRY` — with `multinomial_nb` trained
WITHOUT the StandardScaler (counts must stay counts), enforced by the
`NO_SCALE_MODELS` set.

#### SHAP (`ml.explain`) — the TreeExplainer surface, five figures deep

The upload's LAST unabsorbed block (its lines 472-528), shipped as the
optional-gated surface the round-33 audit predicted — then extended by
the round-43 shap/shap repository review with the plot gallery's
remaining signature figures (all pure matplotlib; only the computation
needs shap):

```python
from pharmaml.ml import group_difference, shap_interactions, shap_summary
from pharmaml.viz import (plot_shap_beeswarm, plot_shap_dependence,
                          plot_shap_force, plot_shap_group_difference,
                          plot_shap_heatmap, plot_shap_importance,
                          plot_shap_waterfall)

summary = shap_summary(fitted_gbr, x_test)   # needs: python -m pip install -e '.[explain]'
summary.ranking(top_n=3)                     # mean |SHAP| per feature, descending
plot_shap_beeswarm(summary)                  # every (sample, feature) point,
                                            # coloured by the feature's own value
plot_shap_importance(summary)                # the compact top-20 bar
plot_shap_waterfall(summary, sample=0)       # the LOCAL explanation: the
                                            # base value -> prediction walk
plot_shap_force(summary, sample=0)           # the static additive bar twin
plot_shap_dependence(summary, "MolWt")       # value vs SHAP, coloured by the
                                            # strongest INTERACTING feature
plot_shap_heatmap(summary)                   # instances x features, rows
                                            # clustered over |SHAP|

# the cohort decomposition -- WHY does one cohort score differently?
import numpy as np
mask = np.asarray(y_test) > np.median(y_test)   # high- vs low-target rows
difference = group_difference(summary, mask, label="high target")
difference.ranking(top_n=3)                     # per-feature mean SHAP gap,
                                                # 5-95% bootstrap bounds
plot_shap_group_difference(difference)          # the bars + whiskers figure

# the exact pairwise interaction cube (same optional gate):
interactions = shap_interactions(fitted_gbr, x_test)
interactions.ranking(top_n=3)   # (feature_a, feature_b, mean |interaction|)
```

`TreeExplainer` only (LightGBM / XGBoost / CatBoost / sklearn trees —
the upload's own choice); anything else gets the typed error naming
the model-agnostic `permutation_feature_importance` sibling instead
of a slow `KernelExplainer` approximation, and multiclass is refused
with the same honesty. The beeswarm is pure matplotlib — only the
computation needs shap, the drawing never does.

#### Offline attribution & attention (`ml.xai`) — captum · tf-keras-vis · PAIR-code/saliency · bertviz, no engines

Round 185 checked, one by one, the contents of four interpretability
repositories and rebuilt what survives without an engine dependency —
**pytorch/captum** (the attribution protocol itself: baselines, the
integrated-gradients `convergence_delta` completeness residual, the
Saliency / InputXGradient / NoiseTunnel-SmoothGrad / Occlusion family,
the positive/negative sign split `visualize_image_attr` paints with,
LayerActivation), **keisen/tf-keras-vis** (the explain-one-network-on-
one-input card shape), **PAIR-code/saliency** (the grayscale and
diverging mask palettes) and **jessevig/bertviz** (the layer × head
attention grid + the query→key token-pair digest, on the toolkit's own
numpy multi-head attention):

```python
import numpy as np

from pharmaml.ml.xai import (
    attribution_methods, input_x_gradient, integrated_gradients,
    occlusion_attributions, smoothgrad_attributions,   # captum's family
    positive_negative_split, diverging_hex, grayscale_hex,
    layer_activation_trace, layer_activation_summary,  # + PAIR-code
    demo_attention_layers, token_pairs,                 # bertviz
)

attribution_methods()          # the registered attribution methods
ig = integrated_gradients(numpy_net, row, steps=50)
ig.attributions                 # per-feature attribution values
ig.convergence_delta            # f(x) - f(baseline) - sum(attributions):
                                # captum's completeness residual, the
                                # honesty number for the step count
sg = smoothgrad_attributions(numpy_net, row, n_samples=32,
                             stdev_ratio=0.2)            # NoiseTunnel
occ = occlusion_attributions(any_net, row)   # the per-feature score drop
                                # under replacement -- predict-only, so
                                # it runs on EVERY engine, torch included
pos, neg = positive_negative_split(ig.attributions)   # the sign split

trace = layer_activation_trace(numpy_net, row)   # hidden units, numpy
layer_activation_summary(trace)  # units, min/mean/max, dead-unit share

layers = demo_attention_layers(["the", "kinase", "inhibitor", "binds"])
pairs = token_pairs(layers[0].weights[0],
                    ["the", "kinase", "inhibitor", "binds"])
                                # ranked (weight, query, key) triples,
                                # bertviz's digest of one head
diverging_hex(ig.attributions[0], np.abs(ig.attributions).max())
grayscale_hex(ig.attributions[0], ig.attributions)
                                # scalar colour helpers for attribution values
```

The gradient trio is exact analytic backprop through the numpy
engine's layer stack (`numpy_layer_stack` reads
`NumpyMLPRegressor`/`NumpyMLPClassifier` — pinned against finite
differences in the test suite, so the numbers never depend on torch);
occlusion needs only `predict`/`predict_proba`. No torch, no
tensorflow, no network — and each registry entry credits its origin
repo (`ATTRIBUTION_METHODS`). These are Python APIs and do not imply
that either graphical application exposes every method.

The registry's three deferred fold-ins are compact offline surfaces in
the same module and card:

```python
from pharmaml.ml.xai import (
    model_activation_trace, tabular_anchor, evaluate_attribution_metrics,
)

# torchlens-style nodes: parents, shapes, parameters, bytes and activations
nodes = model_activation_trace(numpy_net, row)

# alibi-style numeric quantile-bin conjunction, measured on reference rows
anchor = tabular_anchor(any_net, row, reference_rows, threshold=0.95)
print(anchor.rule, anchor.precision, anchor.coverage, anchor.success)

# Quantus-style complexity + faithfulness + max-sensitivity.  The callback
# recomputes the SAME attribution method for every seeded noisy row.
metrics = evaluate_attribution_metrics(
    any_net,
    row,
    occ.attributions,
    explain_fn=lambda values: occlusion_attributions(any_net, values).attributions,
    n_samples=12,
)
```

The graph tracer supports the toolkit's numpy `LayerStack`; it does not claim
arbitrary torch/JAX/TensorFlow interception. The anchor search is numeric,
deterministic and empirical on the supplied reference frame (no KL-LUCB/PAC
confidence claim), and returns `success=False` when its precision target is
unreachable. Quantus complexity uses attribution magnitudes; faithfulness
correlates grouped attribution mass with prediction drops; max-sensitivity is
the maximum relative explanation change over seeded perturbations. Undefined
correlations remain NaN with a reason rather than becoming fabricated zeroes.

#### The run board, UMAP, curation views & express figures (`ml.runboard` · `ml.umap` · `ml.curation` · `viz.interactive`) — tensorboard · umap · fiftyone · plotly.py, offline

Round 186 checked four more repositories file by file and lifted what
survives without their engines (no TensorFlow graph, no numba, no
MongoDB, no plotly.js):

```python
import numpy as np
import pandas as pd
from pharmaml.ml.runboard import (
    RunTrace, RunBoard, smooth_series, reservoir_sample, nearest_neighbours,
)
from pharmaml.ml.umap import (
    find_ab_params, fuzzy_simplicial_set, umap_embed, trustworthiness,
)
from pharmaml.ml.curation import (
    SampleSet, find_duplicates, find_outliers, uniqueness,
    sort_by_similarity, evaluate_classification,
)
from pharmaml.viz.interactive import (
    express_scatter, express_histogram, apply_template, TEMPLATES,
    save_interactive_html,
)

# tensorboard: the writer API + the accumulator's reservoir + the
# dashboard's EMA smoothing (bias-corrected)
rng = np.random.default_rng(42)
trace = RunTrace("fit-01")
trace.add_scalar("loss/train", 0.9, step=0)
trace.add_histogram("weights/layer1", rng.normal(size=64), step=0)
trace.add_hparams({"lr": 0.01}, {"val_loss": 0.35})
smooth_series([1.0, 0.5, 0.2], beta=0.6)      # the slider's formula
reservoir_sample(list(range(1000)), 10, seed=42)  # Vitter + keep-last
nearest_neighbours(matrix, row=0, k=5)        # the projector's click

# umap: the algorithm's own machinery, gradient for gradient
a, b = find_ab_params(1.0, 0.1)               # (1.577, 0.895) -- umap's own fit
record = umap_embed(features, n_neighbors=15, min_dist=0.1, n_epochs=200)
record.transform(new_rows)                    # weighted-average init + frozen SGD
trustworthiness(features, record.embedding_array, n_neighbors=10)  # == sklearn's

# fiftyone: views are immutable stage chains, never mutations
samples = SampleSet.from_frame(frame)
view = samples.view().filter("ic50", ">", 5.0).sort_by("ic50").limit(20)
view.summary()          # '420 samples -> filter(ic50 > 5.0) -> ... -> 20 samples'
find_duplicates(samples, fields=["smiles"]).groups
find_outliers(samples, method="iqr", threshold=1.5).flagged
uniqueness(samples, fields=["x", "y"])        # 1 - mean k-NN cosine
evaluate_classification(samples, "cls", "pred").confusion

# plotly.express: one-liners over the figure-as-data model + hover HTML
figure = express_scatter(frame, "dose", "response", color="arm", hover=["pid"])
express_histogram(frame, "dose")              # Freedman-Diaconis bins, numpy-side
apply_template(figure, "ggplot2")             # the named themes
save_interactive_html(figure, "map.html")     # <title> tooltips, no CDN, no JS
```

The honest scope, stated once: tensorboard's served dashboards, event
files and graph viewer; umap's numba/nndescent acceleration, densMAP,
parametric/aligned variants; fiftyone's MongoDB/ODM layer, image-only
constructs (patches/frames/clips) and annotation integrations; plotly's
JavaScript/WebGL renderer and kaleido static export — none of it is
vendored. What transfers is the event model, the algorithm, the view
algebra and the express API, each pinned by its own test suite
(`tests/test_runboard.py`, `tests/test_umap.py`,
`tests/test_curation.py`, `tests/test_interactive.py` — the
trustworthiness port is pinned against sklearn's own implementation,
and the find_ab_params fit against umap's published defaults).

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

#### Hyperparameter search (`ml.tuner`) — Keras Tuner's model, Optuna's samplers and pruners

The `HyperParameters` / `RandomSearch` / `Hyperband` programming model from
[Keras Tuner](https://keras.io/keras_tuner/), re-implemented over ANY
scikit-learn-compatible estimator — no TensorFlow requirement (the toolkit's deep
nets are PyTorch, the legacy ANN is sklearn's MLP) — then extended by the
round-43 optuna/optuna repository review with **`TPESearch`** (the Tree-structured
Parzen Estimator sampler: below/above KDE mixtures, gamma = min(ceil(0.1n), 25),
`n_startup_trials` uniform bootstrap, `n_ei_candidates` acquisition),
**`PercentilePruner`/`MedianPruner`** (the fold-loop pruner: after each CV fold
the running mean is reported; a trial below the percentile of the completed
trials' same-fold values is cut short — state `pruned`, partial `fold_scores`
kept), Optuna-style trial **states + durations + intermediates**, log-scale
`hp.Float(..., log=True)`, **`param_importances`** (the MDI random-forest
attribution) and the seven **study figures** (`plot_optimization_history`,
`plot_slice`, `plot_parallel_coordinate`, `plot_param_importances`, `plot_edf`,
`plot_timeline`, `plot_intermediate_values`). The hypermodel declares its
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

# Optuna's TPE sampler -- same contract, Parzen-guided sampling:
from pharmaml.ml import TPESearch
results = TPESearch(build_ann_regressor, objective="r2",
                    max_trials=12, cv=3, seed=42).search(X, y)

# with the median pruner cutting hopeless trials after each fold:
from pharmaml.ml import MedianPruner
results = RandomSearch(build_ann_regressor, objective="r2", max_trials=12,
                       cv=3, seed=42, pruner=MedianPruner()).search(X, y)
pruned = [t for t in results.trials if t.pruned]   # partial fold scores kept

# which hyperparameter actually mattered?  the MDI forest:
from pharmaml.ml import param_importances
param_importances(results).ranking(top_n=3)   # normalized importances

# the study figures (any TunerResults, pure matplotlib):
from pharmaml.viz import plot_optimization_history, plot_slice
plot_optimization_history(results)   # dots + best-so-far + pruned markers

# Successive halving over the resource hyperparameter (epochs for the deep nets,
# max_iter for the sklearn ANN; batchnorm is itself searchable):
results = Hyperband(build_deep_regressor, objective="r2",
                    resource="epochs", max_resource=81).search(X, y)

# Size the ladder from the family's OWN declared grid -- the shipped
# hypermodels' explicitly declared resource slots:
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
`batchnorm` a `hp.Boolean` slot) and `build_hsvr`. The Python API supports
RandomSearch, **TPESearch (Optuna's Parzen sampler)** and Hyperband; the CLI
command `pharmaml ml tune` provides a leaderboard, trials table and optional
`--save-best` bundle. Hyperband sizes its rung ladder from the family's own
declared grid via `resource_bounds`, records resource usage and budget
comparisons, and supports median pruning. Trial states and partial scores
are retained explicitly; the API also provides seven study figures
(optimization history / slice / parallel coordinates / parameter
importances / EDF / timeline / intermediates). Failed and pruned trials
are recorded, not fatal; sampling is seeded, so the same seed + data
reproduce the same search. The React/Tauri Studio and separate Next.js
console have independently documented workflows; this Python guide does
not imply feature parity.

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
caffeine, … plus a heterocycle/functional-group tour), an objective library, and three
complementary optimisers. A runnable narrative lives in
`examples/run_generative_design.py` (~15 s end to end, the GA running
denovo / seeded / structural), and the fully
**executed notebook** `examples/complete_generative_workflow.ipynb` walks
every stage with committed outputs (see *Quickstart* above).

```python
from pharmaml.generative import (
    DEFAULT_CORPUS, QedObjective, LogPBandObjective, SimilarityObjective,
    NoveltyObjective, MultiObjective, GeneticOptimizer, ReinventOptimizer, SmilesRnn,
    MctsOptimizer,
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

# 3b) Monte Carlo tree search (the ChemTSv2 model, Ishida et al. 2023): the
#     moves are SMILES characters, the policy is the RNN's next-character
#     distribution, the reward is the objective -- the search spends its
#     budget where policy and reward disagree. Extension mode grows only a
#     chosen prefix's continuations (ChemTSv2's --input_smiles).
mcts = MctsOptimizer(QedObjective(), iterations=16, seed=42).optimize()
grown = MctsOptimizer(QedObjective(), seed=42).optimize(prefix="CC(=O)O")

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
  torch install and yields fewer unique hits at this scale. MCTS sits between them: it
  searches *through* the same learned prior without fine-tuning it (tree search over the
  prior's own distribution), spends its budget adaptively (UCB1/PUCT exploration vs the
  cumulative-probability expansion) and ships the extension mode (`optimize(prefix=...)`)
  the other two lack — but it also needs torch for the default policy and yields fewer
  unique molecules per run than the GA. Run all three, compare leaderboards.
* **MCTS mechanics (100x-163, the ChemTSv2 absorption):** UCB1 exactly as ChemTSv2 v2
  computes it (`Q/N + c_val·√(2·ln N_parent / N)`) or ChemTS v1's PUCT
  (`Q/N + puct_c·prior·√N_parent/(1+N)`, `tree_policy="puct"`); the
  cumulative-probability expansion (children = the policy's top characters until their
  probabilities sum to `expansion_threshold`, default 0.995, never fewer than two);
  `simulation_num` rollouts per expanded child per cycle; unparseable rollouts score 0.0
  (the toolkit's [0,1] convention — ChemTSv2's −1 penalty mapped onto it); a dedup
  cache scores every distinct canonical molecule once; `iterations` cycles is a
  deterministic budget (seeded runs are byte-reproducible). The per-cycle search
  statistics (reward story, tree growth, cache economy) render via
  `plot_mcts_introspection(result)` (from `pharmaml.viz`). A custom policy can be
  injected (`MctsOptimizer(objective, policy=callable)`) — any prefix → distribution
  mapping works, no torch needed.
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
* Requires `python -m pip install -e '.[chem]'`; the RNN/RL half additionally needs
  `python -m pip install -e '.[deep]'` (PyTorch is imported lazily — the package works without it).

### SELFIES: the robust molecular string (from 0 to 100)

SMILES has one structural flaw a generative pipeline keeps paying for: an arbitrary
string of SMILES characters is *usually not a molecule*. SELFIES (SELF-referencIng
Embedded Strings, Krenn et al. 2020; `the-matter-lab/selfies`, Apache-2.0 — an optional
`python -m pip install -e '.[selfies]'`, pure-Python with zero transitive dependencies)
fixes that by construction: its grammar clamps every bond, branch and ring to the
remaining bonding capacity, so **any sequence of SELFIES symbols decodes to a valid
molecule**. The `chem.selfies_codec` module is the toolkit's typed surface over it:

```python
from pharmaml.chem.selfies_codec import (
    smiles_to_selfies, selfies_to_smiles, batch_smiles_to_selfies,
    selfies_roundtrip_report, selfies_robustness_experiment,
    selfies_alphabet, selfies_feature_matrix,
    random_selfies_molecules, selfies_mutants, semantic_constraints,
)

# 1) Codec: both directions, symbol by symbol.
selfies = smiles_to_selfies("CC(=O)Oc1ccccc1C(=O)O")     # aspirin, 19 symbols
smiles_to_selfies("CCO"), selfies_to_smiles("[C][C][O]")  # ('[C][C][O]', 'CCO')

# 2) Audit a whole dataset: the SELFIES repo's own QA methodology —
#    encode → decode → RDKit canonical-SMILES equality, row-preserving.
report = selfies_roundtrip_report(["CCO", "CCN", "c1ccccc1"])
report.fidelity, report.alphabet_size, report.summary()

# 3) The corruption experiment (the paper's Figure 3, on your molecules):
#    k random SELFIES *symbols* vs k random SMILES *characters* — the
#    SELFIES curve stays at ~100% valid while the SMILES curve collapses.
experiment = selfies_robustness_experiment(
    ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"], mutations=(1, 2, 4, 8), seed=42)
experiment.selfies_validity, experiment.smiles_validity   # ([1.0, ...], [0.6, ...])
plot_selfies_robustness(experiment)                      # from pharmaml.viz

# 4) Vocabulary + ML features: the label / one-hot encodings of the
#    SELFIES VAE literature, straight into scikit-learn shape.
matrix, vocab = selfies_feature_matrix(["CCO", "CCN", "CCS"])        # (3, 3) labels
hot, _ = selfies_feature_matrix(["CCO", "CCN"], encoding="one_hot")  # (2, 3*V)

# 5) Generation with validity guaranteed: uniform random SELFIES strings
#    (every sample valid by construction) and STONED-style symbol mutants.
random_selfies_molecules(6, 12, seed=42)     # 6 valid molecules, 12 symbols each
selfies_mutants("CCO", n=8, seed=42)         # the symbol-level neighbourhood

# 6) Semantic constraints are PROCESS-GLOBAL in the package — the context
#    manager puts the previous table back, even on exceptions.
with semantic_constraints({"C": 1, "O": 1, "N": 1, "?": 1}):
    selfies_to_smiles("[C][#C]")             # 'CC' — the triple bond clamped
selfies_to_smiles("[C][#C]")                 # 'C#C' — restored, nothing leaked
```

* Honest scope: the decoder writes aromatic rings in kekulé form (`C1=CC=CC=C1`) —
  chemically identical, and canonical-equal under RDKit, which is exactly the
  round-trip audit's criterion.
* Requires `python -m pip install -e '.[selfies]'`; the audit / robustness / feature
  halves additionally need RDKit. The package is imported lazily everywhere.

### REINVENT4's scoring DSL & diversity filters (from 0 to 100)

REINVENT4 (`MolecularAI/REINVENT4`, Apache-2.0) is AstraZeneca's de novo design
framework: a plugin scoring system over transforms, scaffold-memory diversity
filters and a standardization chain. The toolkit absorbed the programming model
in typed library surfaces (everything runs on RDKit + numpy — no torch, no
plugin discovery, no external CLIs):

```python
from pharmaml.generative import (
    GeneticOptimizer, make_transform, RawComponentObjective,
    CustomAlertsObjective, DiversityFilteredObjective, MultiObjective,
    QedObjective, weighted_arithmetic_mean, weighted_geometric_mean,
)
from pharmaml.generative.scoring_transforms import DoubleSigmoidTransform

# 1) The eight transforms with REINVENT4's exact formulas: sigmoid /
#    reverse sigmoid (the numerically stable base-10 form, the k
#    rescaling 10k/(high-low)), double sigmoid (the band pass), the
#    three steps, the rectified exponential decay, the value mapping.
#    make_transform("double-sigmoid", low=150, high=450, k=100) resolves
#    exactly like REINVENT4's get_transform name normalisation.
mw = RawComponentObjective("molecular_weight",
                           DoubleSigmoidTransform(low=150, high=450, k=100.0))
mw.score("CC(=O)Oc1ccccc1C(=O)O")      # aspirin (180 Da): inside the band

# 2) The filter/penalty semantics: CustomAlerts vetoes blocklist hits
#    (0.0 -> the geometric mean zeroes the molecule); MatchingSubstructure
#    carries REINVENT4's 0.5*(1+match) penalty product.
alerts = CustomAlertsObjective(["[N+](=O)[O-]", "C(F)(F)F"])

# 3) The NaN-aware weighted aggregators (a NaN component drops out of
#    numerator AND denominator — REINVENT4's means.py semantics).
weighted_arithmetic_mean([([0.2, 0.8], 1.0), ([0.6, 0.4], 3.0)])

# 4) The diversity filters: wrap ANY objective — the GA / RL / MCTS score
#    through it unchanged, and the scaffold memory accumulates across the
#    run exactly like REINVENT4's RL memories (bucket counter, minscore
#    gate, the five penalty curves step/linear/sigmoid/tanh/erf).
objective = DiversityFilteredObjective(
    MultiObjective([QedObjective(), alerts]),
    "identical_murcko_scaffold", bucket_size=25, minscore=0.4)
result = GeneticOptimizer(objective, population_size=24, seed=42).optimize()
objective.scaffold_stats()   # scaffolds, full buckets, zeroed scores, ...
```

* Honest scope: the components that shell out to external CLIs or licensed SDKs
  (SynthSense, ROCS/OpenEye, ChemProp, Qptuna, Maize, DockStream) did not land.
  The Ertl SA score DID land (Part-3 §24, round 230) — see
  "Synthetic accessibility scoring" below — because the RDKit wheel ships
  Ertl's own fragment database (no vendored pickle required).

### DiffDock's pose analysis (from 0 to 100)

DiffDock (`gcorso/DiffDock`, MIT) is the diffusion-based docker; the diffusion
models themselves (torch + e3nn + ESMFold + trained checkpoints) cannot land in
an RDKit/numpy toolkit — every piece of pose mathematics *around* them did:

```python
import numpy as np
from pharmaml.docking.pose_rmsd import (
    kabsch_rmsd, qcp_rmsd, hungarian_rmsd,
    symmetry_corrected_rmsd, best_pose_rmsd, rmsd_wrapper,
    torsion_bonds, rotatable_masks, dihedrals, perturb_torsions,
    randomize_pose, pose_evaluation_report, pose_clusters,
    extract_pocket, crop_beyond,
)
from rdkit import Chem

# 1) The alignment mathematics: Kabsch (DiffDock's rigid_transform_Kabsch_3D),
#    QCP (Theobald's quaternion characteristic polynomial, spyrmsd/qcp.py),
#    Hungarian (the ordering-free per-element assignment).
kabsch_rmsd(square, square @ rotation.T)      # 0.0 — rotations align away

# 2) The symmetry-corrected RMSD — the metric DiffDock's own evaluation uses
#    (spyrmsd's symmrmsd): every automorphism of the molecular graph, the
#    minimum RMSD over the permutations, cached across a pose list.
ethane = Chem.MolFromSmiles("CC")
a = np.array([[0.0, 0, 0], [1.54, 0, 0]])
b = np.array([[1.54, 0, 0], [0.0, 0, 0]])     # the two carbons swapped
symmetry_corrected_rmsd(ethane, a, b)          # 0.0 — the SAME molecule

# 3) The torsion machinery: DiffDock's own graph-disconnect rotatable-bond
#    rule, the masks, the dihedrals, torsion perturbation and
#    randomize_pose (the diffusion initial state: random torsions + a
#    random rotation, seeded).
torsion_bonds(Chem.MolFromSmiles("CCCC"))     # [(1, 2)] — terminals never rotate

# 4) The evaluate.py metrics on a reference vs N poses: mean/median/
#    percentiles, the <2 A / <5 A success fractions, centroid distances,
#    best-of-N, the self-clash fraction and greedy pose clusters.
report = pose_evaluation_report(mol, reference, poses)
report.rmsd_below_2_fraction, report.summary()

# 5) The pocket rules: MOAD's chain_cutoff residue selection (whole chains
#    or pocket residues only) and DiffDock's 20 A crop_beyond.
pocket = extract_pocket(protein_pdb_text, ligand_coords, residue_cutoff=10.0)
pocket.residues, pocket.pdb_text, pocket.summary()
```

* Honest scope: no diffusion model, no confidence model, no ESM embeddings —
  the campaign docking stays AutoDock4/Vina, now with DiffDock-grade pose QA.

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
    plot_contact_map, plot_plif_heatmap,
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

#### First-class image export: PNG, SVG, TIFF, JPEG (Part-3 §17)

```python
import numpy as np
from pharmaml.viz import (
    export_image, export_matrix, export_figure, ExportManager, sniff_format,
)

export_image(matrix_u8, "panel.png")         # .png / .tif / .tiff / .jpg / .jpeg
export_image(matrix_u16, "assay.tiff",       # 16-bit data stays 16-bit (I;16),
    compression="tiff_lzw")                  # lossless: tiff_lzw / tiff_deflate / none
export_matrix(page0, "stack.tiff",           # multi-page TIFF (image stacks),
    pages=[page1, page2])                    # every page bit-depth-checked
export_image(photo, "slide.jpg",             # configurable quality 1-95; a 16-bit
    jpeg_quality=92)                         # source REFUSES silent quantisation
export_figure(fig, "figure.tiff", dpi=300)   # matplotlib fig -> same rules
                                             # (.svg stays vectorial)

manager = ExportManager("figures", "{name}.tiff")   # batch + JSON manifest
manager.export(heatmap, "heatmap")
manager.export(parity, "parity")             # each write reopened + magic-sniffed

sniff_format("figures/heatmap.tiff")         # "TIFF" from the magic bytes —
                                             # never the extension's claim
```

The scientific-integrity rules are the feature: 16-bit matrices export
as genuine `I;16` TIFFs (never silently quantised to 8-bit); float
matrices refuse without an explicit `normalize_float=True`; JPEG is
honest about being lossy (explicit `downcast_16bit=True` required for
16-bit sources, alpha composited onto a chosen background); and every
exported file is reopened, magic-byte-sniffed and dimension-checked
before success is reported — an extension never lies about content
(§17.4). Pillow is optional exactly like matplotlib
(`python -m pip install -e '.[reports]'`). This export engine is part of
the Python package; front-end export controls are documented separately
and are not assumed to expose identical options.

matplotlib is optional (``python -m pip install -e '.[reports]'``); the Agg backend is
forced automatically outside notebooks, so plotting works on headless servers
and CI — and renders inline inside Jupyter.

#### Durable job checkpoints: pause, resume, stop without losing work (Part-3 §20)

Long jobs checkpoint.  `pharmaml.utils.checkpoints.CheckpointManager`
writes a small self-verifying JSON file (SHA-256 over its canonical
content) after every completed unit of work — one feature family, one
docking batch, one generative chunk — atomically (`fsync` + rename:
crash-safe), rotated (only the newest N survive) and validated on read
(a truncated or tampered newest checkpoint is quarantined to
`*.json.corrupt` and the resume falls back to the previous one instead
of failing).  The family benchmark (`benchmark_feature_families`) is
wired end-to-end: pass `checkpoint_dir=` and a run stopped by the
cooperative-stop contract leaves its finished families on disk; the
next call with the same signature (dataset, folds, metric, model,
family list — all hashed together) resumes from the first unscored
family instead of recomputing, and any mismatch starts fresh — a
resumed board is only ever assembled from rows computed on exactly
that request.  Checkpoint write failures degrade to log warnings; they
never destroy a 40-minute sweep. These guarantees apply to the Python
family-benchmark API; they do not establish that the current desktop app
or separate Next.js console can resume arbitrary jobs.

#### Synthetic accessibility scoring in the generative loop (Part-3 §24)

Every generated molecule can now carry the Ertl–Schuffenhauer
**synthetic accessibility (SA) score** — the reference implementation
(Ertl & Schuffenhauer, *J. Cheminformatics* 2009) whose fragment
database ships inside the RDKit wheel itself
(`Contrib/SA_Score/fpscores.pkl.gz`), so the score is the *real*
fragment model, not an approximation: aspirin 1.58, caffeine 2.30,
the conventional 1–10 scale where **lower = estimated easier to
synthesise**.  `pharmaml.chem.sa_scoring` exposes the batch API
(`sa_scores`, `filter_by_sa`, `sa_score_summary`): per-molecule rows
keep the original SMILES, the canonical form, the score, an
`easy/moderate/hard` band and the parse error when a row is invalid —
invalid SMILES never crash the batch.  The optional max-SA bound only
*flags* rejects (the §24.4 contract: never silently discard
candidates); the summary reports mean ± std, range and the honest
caveat (a computational estimate, not a route plan).

The score is available in three Python surfaces: the **oracle book**
(`chem.oracles.sa_score` — the fragment table is loaded lazily and a
Contrib-less RDKit build degrades honestly to the complexity-only
ranking, and `penalized_logp` now uses the real SA too), the
generative **evolution objective** (`SAScoreObjective`, where the
window bounds become the easy/hard reward anchors — §24.5: SA as ONE
signal in a multi-objective loop, never the sole target), the
**results container itself** (round 265: `GenerationResult.sa_frame()`
appends the raw `sa_score` column to any run's leaderboard — sortable
`"easiest"`/`"hardest"`, a run that optimised the SA reward keeps it
as a distinct `sa_score_reward` column — while
`filter_by_sa(min_sa, max_sa)` preserves the rejected rows with
reasons (`kept + rejected == total`, always), `sa_summary()` reports
the valid-subset statistics with the drug-like-band count, and
`plot_sa_distribution()` draws the exact-edge histogram with the
conventional 1–4 band shaded).

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
require the `reports` extra (`python -m pip install -e '.[reports]'`).

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
python -m pytest                 # full suite (~6,960 tests at 1.2.201 with the pinned stack incl.
                                 # doctests; skips RDKit/torch-only tests when those extras
                                 # are absent — run `pytest --co -q` for the exact count)
python -m pytest tests/test_pk.py -v
python -m pytest --cov=pharmaml  # with pytest-cov
```

The suite covers: PK/PD/dosage math (against hand-computed references), unit conversion
round-trips, ML pipeline behaviour (splits, metrics, persistence round-trip, **the
no-leakage guards — the split returns raw features and the fitted preprocessor's
statistics provably come from the train rows only**; see
`docs/LEAKAGE_AUDIT.md`), **user-data
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
