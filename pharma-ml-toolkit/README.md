# PharmaML Toolkit

A **production-ready, typed, tested** Python toolkit for computational pharmacy, consolidating three
domains into one coherent package:

| Domain | What it does |
|---|---|
| **`pharmaml.ml`** | QSAR machine learning — preprocessing, model registry, cross-validated training, evaluation, persistence |
| **`pharmaml.pharma`** | Pharmaceutical calculations — unit conversion, pharmacokinetics (PK), pharmacodynamics (PD), dosage design |
| **`pharmaml.docking`** | Molecular docking — ligand/grid preparation, AutoDock4 & Vina batch workflows, result parsing, scoring |
| **`pharmaml.chem`** | Cheminformatics helpers — IC50/Ki ↔ pIC50/pKi conversion, RDKit descriptors, fingerprints, Tanimoto similarity |

This package is the upgraded, re-engineered version of a flat collection of research scripts:
every module now has **type hints, Google-style docstrings, structured logging, custom
exceptions, input validation, and pytest coverage** (159 tests).

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

---

## Installation

```bash
# core (ML + pharma + docking parsing/scoring)
pip install -e .

# with cheminformatics extras
pip install -e ".[chem]"

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
│   ├── cli.py               # `pharmaml` command-line interface
│   ├── chem/                # converters, descriptors, similarity
│   ├── ml/                  # preprocessing, models, training, evaluation, persistence
│   ├── pharma/              # units, pk, pd, dosage
│   ├── docking/             # preparation, workflow, parsing, scoring
│   └── utils/               # io (safe read/write), validation
├── tests/                   # 159 pytest cases across all domains
├── examples/                # runnable end-to-end demos (synthetic data)
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### Machine learning (QSAR)

```python
from pharmaml.ml import compare_models, train_model, save_model

report = train_model(
    "random_forest",
    data="dataset.csv",     # CSV/Excel with a "Target" column
    target="Target",
    cv=5,                   # stratified 5-fold cross-validation
    tune=True,              # small grid search
    out="artifacts",        # model + metadata land here
)
print(report.metrics)       # accuracy, ROC-AUC, F1, CV mean ± std

leaderboard = compare_models("dataset.csv", target="Target", cv=5)
print(leaderboard.sort_values("roc_auc_mean", ascending=False))
```

Registry models: `logistic`, `ridge`, `lasso`, `elastic_net`, `svm`, `knn`, `decision_tree`,
`random_forest`, `ada_boost`, `ann`, plus `xgboost` / `lightgbm` / `catboost` when the
optional packages are installed. `list_models()` shows what is available in *your* environment.

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
crcl = creatinine_clearance(age=68, weight_kg=70, scr_mg_dl=1.1, sex="male")
bsa  = bsa_mosteller(weight_kg=70, height_cm=175)          # m²
dose = dose_by_bsa(dose_mg_per_m2=250, weight_kg=70, height_cm=175)

# Regimen design
ld  = loading_dose(target_conc=12.0, vd=45.0)              # mg
md  = maintenance_dose(clearance_=3.2, target_css=6.0, tau=8.0)  # mg per interval
```

### Molecular docking

```python
from pharmaml.docking import (
    compute_grid_box, write_gpf, parse_vina_output,
    parse_dlg_cluster_table, ligand_efficiency, consensus_rescore,
)

box = compute_grid_box("receptor.pdb", padding=4.0)   # AutoDock grid box from binding site
write_gpf("receptor.pdbqt", box, n_points=(60, 60, 60), spacing=0.375, gpf_path="receptor.gpf")

poses = parse_vina_output(open("vina.out").read())    # ranked poses (affinity, RMSD l.b./u.b.)
clusters = parse_dlg_cluster_table(open("r.dlg").read())  # AutoDock4 cluster histogram

le  = ligand_efficiency(affinity=-9.6, heavy_atoms=32)   # kcal/mol per heavy atom
rescored = consensus_rescore({"vina": scores, "ad4": scores2})  # z-score consensus ranking
```

---

## Command-line interface

```
pharmaml --version
pharmaml models                       # list available registry models
pharmaml ml train --data data.csv --target Target --model random_forest --cv 5
pharmaml ml compare --data data.csv --target Target
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
python -m pytest                 # full suite (159 tests, skips RDKit-only tests if absent)
python -m pytest tests/test_pk.py -v
python -m pytest --cov=pharmaml  # with pytest-cov
```

The suite covers: PK/PD/dosage math (against hand-computed references), unit conversion
round-trips, ML pipeline behaviour (splits, metrics, persistence round-trip), docking parsers
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
