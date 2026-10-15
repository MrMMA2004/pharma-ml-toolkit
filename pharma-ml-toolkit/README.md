# PharmaML Toolkit

A **production-ready, typed, tested** Python toolkit for computational pharmacy, consolidating three
domains into one coherent package:

| Domain | What it does |
|---|---|
| **`pharmaml.data`** | **Bring your own data** — load user CSV/TSV/Excel files, auto-detect the target column, infer classification (2+ classes) vs regression, audit data quality |
| **`pharmaml.ml`** | QSAR machine learning — preprocessing, model registry, cross-validated training (binary **and multiclass**), evaluation, persistence |
| **`pharmaml.pharma`** | Pharmaceutical calculations — unit conversion, pharmacokinetics (PK), pharmacodynamics (PD), dosage design, regulatory-approval filtering (DGIdb TSV) |
| **`pharmaml.docking`** | Molecular docking — ligand CSV import + protein/receptor preparation (PDB & mmCIF), grids **and DPF generation**, AutoDock4 & Vina batch workflows (incl. config-driven batches), result parsing, scoring, torsion QA, campaign file management |
| **`pharmaml.chem`** | Cheminformatics helpers — IC50/Ki ↔ pIC50/pKi conversion, RDKit descriptors, fingerprints, Tanimoto similarity, SMI/CSV/SDF format conversion, tautomer standardization + 3D ligand generation (SMILES→PDB), ligand-table reconciliation & merging |

This package is the upgraded, re-engineered version of a flat collection of research scripts:
every module now has **type hints, Google-style docstrings, structured logging, custom
exceptions, input validation, and pytest coverage** (626 tests).

---

## Design assumptions

* **Python ≥ 3.10** (union syntax `X | None` used throughout).
* **Core deps only**: `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib` — everything else
  (RDKit, Mordred, XGBoost/LightGBM/CatBoost, Meeko, gemmi) is **optional and lazily imported**;
  missing packages raise an informative `DependencyError` instead of crashing at import time.
* **Runs anywhere**: local, Docker (`Dockerfile` included), or CI (GitHub Actions workflow included).
* **Deterministic examples**: all example scripts run on synthetic data with fixed seeds — no
  external datasets required.
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

```bash
# core (ML + pharma + docking parsing/scoring)
pip install -e .

# with cheminformatics extras (RDKit descriptors/fingerprints/similarity)
pip install -e ".[chem]"

# with docking preparation extras (RDKit + Meeko: SMILES -> 3D PDBQT)
pip install -e ".[docking]"

# with gradient-boosting model families
pip install -e ".[boosting]"

# development (pytest, mypy, ruff)
pip install -e ".[dev]"
```

Quick test run (no install needed — `pyproject.toml` maps `src/` onto the import path):

```bash
python -m pytest
```

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
│   ├── ml/                  # preprocessing, models (16 incl. Huber), training, evaluation, persistence
│   ├── pharma/              # units, pk, pd, dosage, approvals
│   ├── docking/             # ligands + receptor (CSV/PDB/mmCIF in), grids + GPF/DPF, workflow, parsing, scoring, campaign
│   ├── viz/                # figures for every domain + HTML reports + multi-format table export
│   └── utils/               # io: read_table, write_table, safe_filename · validation: validate_smiles_syntax, require_columns, coerce_frame
├── tests/                   # 724 pytest cases across all domains (incl. test_docking_prep.py, test_campaign.py, test_format_converters.py + data/ fixtures)
├── examples/                # runnable demos + four executed notebooks (complete ML · complete docking · complete PK/PD · walkthrough) + three no-output twins
│   └── data/                # lysozyme_1aki.pdb — ready-to-use demo receptor input (real lysozyme, public domain)
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
> descriptors, Morgan/RDKit/atom-pair fingerprints, MACCS keys, Tanimoto
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

> §1 ligand-library import (`load_ligand_csv` on a 10-row demo `ligands.csv` —
> SMILES column auto-detected, RDKit validation, invalid rows exported to
> `failed_smiles.csv`, then `prepare_ligand_library` builds the folder-per-ligand
> campaign tree) → §2 protein import & preparation (`prepare_protein` on a real
> lysozyme PDB — waters/HETATM/altlocs cleaned, Meeko-typed + charged receptor,
> `distribute_receptor` copies it into every ligand folder) → §3 ligand
> preparation (SMILES → 3D PDBQT via Meeko) → §4 grid box + GPF/DPF → §5 running
> the engines (AutoDock4 / Vina, guarded failures) → §6 parsing (cluster tables,
> completion audit, campaign scan, idempotent batch re-run) → §7 torsion QA &
> campaign housekeeping (tree-wide TORSDOF/BRANCH audit with one deliberate
> mismatch, plus dry-run collect/cleanup/distribute/delete previews) → §8 Vina
> pose tables → §9 rescoring (ligand efficiency + consensus) → §10
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

Each of the three workflow notebooks also ships a **no-output twin**
(`*_no_output.ipynb`) — identical cells with every output stripped, 4–30 % of
the executed size (the ML twin is 46 KB vs 1.1 MB), for slow connections; run
it top-to-bottom to produce the results yourself. A pytest drift-guard pins
the twins' cell sources to the executed originals.

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
pharmaml ml compare --data my_compounds.csv --target activity --plot
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

leaderboard = compare_models(ds.X, ds.y, ds.task, cv=5)
print(leaderboard.head())   # one row per model, sorted by CV metric
```

A fully **executed Jupyter notebook** covering all four domains lives at
`examples/pharmaml_walkthrough.ipynb` (every cell has its output committed — run it
top-to-bottom to see training metrics, NCA results and docking tables).

Registry models: `logistic_regression`, `ridge_classifier`, `svm`, `knn`, `decision_tree`,
`random_forest`, `adaboost`, `ann`, `lasso`, `ridge`, `elastic_net`, `linear_regression`,
`huber_regression` (outlier-robust), plus `xgboost` / `lightgbm` / `catboost` when the
optional packages are installed.
`list_models()` shows what is available in *your* environment.

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
)

box = compute_grid_box("reference_ligand.pdbqt", padding=4.0)   # grid box around the co-crystal ligand
write_gpf("ligand.pdbqt", receptor_file="receptor.pdbqt", box=box, output_path="grid.gpf")

poses = parse_vina_output(open("vina.out").read())    # ranked poses (affinity, RMSD l.b./u.b.)
clusters = parse_dlg_cluster_table(open("r.dlg").read())  # AutoDock4 cluster histogram

le = ligand_efficiency(affinity_kcal_mol=-9.6, num_heavy_atoms=32)  # 0.30 kcal/mol per atom

scores = pd.DataFrame({"vina": [...], "ad4": [...]})  # one row per ligand
consensus = consensus_rescore(scores, columns=["vina", "ad4"])     # 0–1 combined score
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
    plot_energy_distribution, plot_top_ligands,
    HTMLReport, export_table, metrics_to_dataframe, save_visual_bundle, save_figure,
)

fig = plot_confusion_matrix(report)          # ClassificationReport -> heat-map (2+ classes)
save_figure(fig, "confusion.png")            # png / svg / pdf by extension
fig2 = plot_roc_curves(model, X_test, y_test)   # binary or per-class one-vs-rest
fig3 = plot_calibration_curve(model, X_test, y_test)  # reliability + Brier scores
fig4 = plot_learning_curve(model, X, y, cv=5)  # is more data still buying accuracy?
fig5 = plot_table_comparison(comparison)  # ligand-set overlap (compare_tables result)

export_table(leaderboard, "leaderboard.md")  # csv / tsv / markdown / json / html

save_visual_bundle(                          # one call -> report.html + files
    figures={"confusion": fig, "roc": fig2},
    tables={"leaderboard": leaderboard},
    metrics=report,
    out_dir="reports",
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
pharmaml chem pic50 --data data.csv --column "IC50 (nM)" --unit nM
pharmaml chem descriptors --data data.csv --smiles-col SMILES
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
pharmaml docking collect --dir runs/ --pattern '^\\d+\\.pdbqt$'  # gather artifacts
pharmaml docking campaign list --dir ligand_folders/ --ext .pdbqt --out names.csv
pharmaml docking campaign distribute --dir ligand_folders/ --file receptor.pdbqt
pharmaml docking campaign cleanup --dir runs/ --keep r.dlg --remove-empty --dry-run
pharmaml docking campaign delete-named --dir runs/ --name r.gpf --dry-run
pharmaml docking campaign reindex --data results.csv --column name --start 1 --stop 300  # gap audit + reindex
```

`--model` accepts registry names and aliases (`rf`, `logistic`, `xgb`,
`mlp`, `svm`, … — see `pharmaml.ml.MODEL_ALIASES`).

`ml train --plots` emits the full visual bundle — `report.html` plus
confusion matrix, ROC curves, class distribution, CV fold scores and
feature-importance figures (parity + residual diagnostics for regression)
— into `<out>/report/`, no Python required. `ml compare --plot` saves the
leaderboard bar chart and `--out` exports the table in any of
csv/tsv/md/json/html. All plotting flags require the `reports` extra
(`pip install 'pharmaml[reports]'`).

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
python -m pytest                 # full suite (626 tests, skips RDKit-only tests if absent)
python -m pytest tests/test_pk.py -v
python -m pytest --cov=pharmaml  # with pytest-cov
```

The suite covers: PK/PD/dosage math (against hand-computed references), unit conversion
round-trips, ML pipeline behaviour (splits, metrics, persistence round-trip), **user-data
loading (CSV/Excel, delimiter sniffing, target auto-detection, multiclass and regression
inference)**, **multiclass end-to-end flows (3–4 classes, scorer fallbacks, ensembles)**,
**every plotting function / export format / HTML report path (headless Agg CI proof,
plus a guard that the shipped notebooks stay executed and error-free)**, docking
preparation & parsers (ligand CSV import, receptor prep incl. the mmCIF/gemmi path,
synthetic DLG/Vina/PDBQT fixtures), scoring functions, converter edge cases, and the
SMI/CSV/SDF format-conversion round-trips.

## CI

`.github/workflows/ci.yml` runs the matrix (Python 3.10–3.12): install → ruff → mypy (soft) → pytest.

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
