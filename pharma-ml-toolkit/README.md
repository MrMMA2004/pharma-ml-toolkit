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
exceptions, input validation, and pytest coverage** (177 tests).

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
│   └── utils/               # io: read_table, write_table, safe_filename · validation: validate_smiles_syntax, require_columns, coerce_frame
├── tests/                   # 177 pytest cases across all domains
├── examples/                # runnable demos + pharmaml_walkthrough.ipynb (executed notebook)
├── docs/                    # INDEX.md — auto-generated module/API index
├── .github/workflows/       # CI matrix (Python 3.10–3.12) + full-deps job
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### Machine learning (QSAR)

```python
import pandas as pd
from pharmaml.ml import (
    clean_numeric_frame, make_train_test_split, train_model,
    evaluate_model, compare_models, save_model, load_model,
)

frame = clean_numeric_frame(pd.read_csv("dataset.csv"), target_col="Target")
X_train, X_test, y_train, y_test = make_train_test_split(
    frame.drop(columns=["Target"]), frame["Target"], test_size=0.2
)

result = train_model("random_forest", "classification", X_train, y_train, cv=5, tune=True)
print(f"CV ROC-AUC {result.cv_mean:.3f} ± {result.cv_std:.3f} in {result.fit_seconds:.1f}s")
print("Best params:", result.best_params)

report = evaluate_model(result.model, X_test, y_test, "classification")
print(f"accuracy={report.accuracy:.3f}  f1={report.f1_macro:.3f}  roc_auc={report.roc_auc:.3f}")

save_model(result.model, "artifacts/rf.joblib", name="random_forest",
           task="classification", metrics={"accuracy": report.accuracy})
model, metadata = load_model("artifacts/rf.joblib")

leaderboard = compare_models(frame.drop(columns=["Target"]), frame["Target"],
                             "classification", cv=5)
print(leaderboard.head())   # one row per model, sorted by CV metric
```

A fully **executed Jupyter notebook** covering all four domains lives at
`examples/pharmaml_walkthrough.ipynb` (every cell has its output committed — run it
top-to-bottom to see training metrics, NCA results and docking tables).

Registry models: `logistic_regression`, `ridge_classifier`, `svm`, `knn`, `decision_tree`,
`random_forest`, `adaboost`, `ann`, `lasso`, `ridge`, `elastic_net`, `linear_regression`,
plus `xgboost` / `lightgbm` / `catboost` when the optional packages are installed.
`list_models()` shows what is available in *your* environment.

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
python -m pytest                 # full suite (177 tests, skips RDKit-only tests if absent)
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
