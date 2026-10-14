# PharmaML Toolkit

A **production-ready, typed, tested** Python toolkit for computational pharmacy, consolidating three
domains into one coherent package:

| Domain | What it does |
|---|---|
| **`pharmaml.data`** | **Bring your own data** — load user CSV/TSV/Excel files, auto-detect the target column, infer classification (2+ classes) vs regression, audit data quality |
| **`pharmaml.ml`** | QSAR machine learning — preprocessing, model registry, cross-validated training (binary **and multiclass**), evaluation, persistence |
| **`pharmaml.pharma`** | Pharmaceutical calculations — unit conversion, pharmacokinetics (PK), pharmacodynamics (PD), dosage design |
| **`pharmaml.docking`** | Molecular docking — ligand/grid preparation, AutoDock4 & Vina batch workflows, result parsing, scoring |
| **`pharmaml.chem`** | Cheminformatics helpers — IC50/Ki ↔ pIC50/pKi conversion, RDKit descriptors, fingerprints, Tanimoto similarity |

This package is the upgraded, re-engineered version of a flat collection of research scripts:
every module now has **type hints, Google-style docstrings, structured logging, custom
exceptions, input validation, and pytest coverage** (227 tests).

---

## Design assumptions

* **Python ≥ 3.10** (union syntax `X | None` used throughout).
* **Core deps only**: `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib` — everything else
  (RDKit, Mordred, XGBoost/LightGBM/CatBoost, Meeko) is **optional and lazily imported**;
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
│   ├── chem/                # converters, descriptors, similarity
│   ├── ml/                  # preprocessing, models, training, evaluation, persistence
│   ├── pharma/              # units, pk, pd, dosage
│   ├── docking/             # preparation, workflow, parsing, scoring
│   ├── viz/                # figures for every domain + HTML reports + multi-format table export
│   └── utils/               # io: read_table, write_table, safe_filename · validation: validate_smiles_syntax, require_columns, coerce_frame
├── tests/                   # 291 pytest cases across all domains
├── examples/                # runnable demos + three executed notebooks (complete ML workflow · complete docking workflow · walkthrough)
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
> §8 multiclass → §9 regression (parity, residuals) → §10 leaderboard →
> §11 feature importance → §12 save/load & scoring new compounds → §13
> one-click HTML report + png/svg/pdf + csv/md/json.

Open it, replace the demo file with your own CSV (and the §2 library with your
own SMILES column), and re-run all cells. §2 needs RDKit —
`pip install 'pharmaml[chem]'`.

### The complete docking process in one notebook

`examples/complete_docking_workflow.ipynb` is the docking **zero-to-one-hundred**
walkthrough — a full virtual-screening campaign, executed end-to-end without a
docking engine (the demo campaign tree uses the real AutoDock4 `.dlg` format):

> §1 ligand preparation (SMILES → 3D PDBQT via Meeko) → §2 grid box + GPF →
> §3 running the engines (AutoDock4 / Vina, guarded failures) → §4 parsing
> (cluster tables, completion audit, campaign scan, idempotent batch re-run) →
> §5 Vina pose tables → §6 rescoring (ligand efficiency + consensus) → §7
> energy-distribution & top-ligand figures → §8 one-click HTML report.

Point `DOCKING_DIR` at your real campaign folder and re-run — parsers, rescoring
and plots work unchanged on real `.dlg` trees. §1 needs
`pip install 'pharmaml[docking]'`.

(`examples/pharmaml_walkthrough.ipynb` remains the every-domain tour including
PK, docking and chem conversions.)

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

No-code alternative via the CLI:

```bash
pharmaml data inspect --data my_compounds.csv --target activity --head 5
pharmaml ml train  --data my_compounds.csv --target activity --task auto
pharmaml ml compare --data my_compounds.csv --target activity
```

`--task auto` (the default) infers classification vs regression from the target column, so
the same command works for a binary, multiclass or pIC50 endpoint file.

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
plus `xgboost` / `lightgbm` / `catboost` when the optional packages are installed.
`list_models()` shows what is available in *your* environment.

#### Multiclass classification (two or more classes)

Every classification component supports **any number of classes**, not just binary:

```python
from pharmaml.data import load_dataset
from pharmaml.ml import train_model, evaluate_model
from pharmaml.ml.preprocessing import make_train_test_split

# A target column with 3+ distinct values is automatically multiclass.
ds = load_dataset("my_compounds.csv", target="activity")   # e.g. low/medium/high
X_train, X_test, y_train, y_test = make_train_test_split(ds.X, ds.y)  # stratified

result = train_model("random_forest", "classification", X_train, y_train)
report = evaluate_model(result.model, X_test, y_test, "classification")
print(report.confusion)   # k x k confusion matrix
print(report.roc_auc)     # one-vs-rest multiclass AUC
```

* Splits are **stratified**, so every class appears in train *and* test folds.
* Metrics are macro/weighted-averaged (`f1_macro`, `precision_macro`, …) plus a k×k
  confusion matrix and one-vs-rest ROC-AUC; MCC and balanced accuracy are multiclass-native.
* `compare_models()` scores with one-vs-one ROC-AUC — and falls back to balanced accuracy
  when a compared model lacks `predict_proba` (e.g. `ridge_classifier`), so leaderboards
  never silently fill with NaN.
* `build_voting(..., voting="hard")` works for any classifier and any class count;
  `voting="soft"` validates that every member exposes `predict_proba` and tells you which
  ones don’t.
* Generate multiclass demo data: `python examples/make_synthetic_qsar_data.py --classes 3`
  or `python examples/run_classification.py --classes 3`.

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
    plot_concentration_profile, plot_dose_accumulation,
    plot_descriptor_distributions, plot_similarity_heatmap,
    plot_energy_distribution, plot_top_ligands,
    HTMLReport, export_table, metrics_to_dataframe, save_visual_bundle, save_figure,
)

fig = plot_confusion_matrix(report)          # ClassificationReport -> heat-map (2+ classes)
save_figure(fig, "confusion.png")            # png / svg / pdf by extension
fig2 = plot_roc_curves(model, X_test, y_test)   # binary or per-class one-vs-rest

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
pharmaml ml train --data my.csv --target activity --task auto
pharmaml ml train --data my.csv --model random_forest --cv 5
pharmaml ml compare --data my.csv --target activity
pharmaml chem pic50 --data data.csv --column "IC50 (nM)" --unit nM
pharmaml chem descriptors --data data.csv --smiles-col SMILES
pharmaml pharma pk --data conc.csv --time time --conc conc
pharmaml pharma dose --cl 3.2 --vd 45 --target-css 6.0 --tau 8 --f 1.0
pharmaml docking scan --directory runs/ --log-name r.dlg
pharmaml docking check --directory runs/
```

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
python -m pytest                 # full suite (291 tests, skips RDKit-only tests if absent)
python -m pytest tests/test_pk.py -v
python -m pytest --cov=pharmaml  # with pytest-cov
```

The suite covers: PK/PD/dosage math (against hand-computed references), unit conversion
round-trips, ML pipeline behaviour (splits, metrics, persistence round-trip), **user-data
loading (CSV/Excel, delimiter sniffing, target auto-detection, multiclass and regression
inference)**, **multiclass end-to-end flows (3–4 classes, scorer fallbacks, ensembles)**,
**every plotting function / export format / HTML report path (headless Agg CI proof,
plus a guard that the shipped notebooks stay executed and error-free)**, docking parsers
(synthetic DLG/Vina/PDBQT fixtures), scoring functions, and converter edge cases.

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
