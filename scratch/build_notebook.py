"""Build + execute examples/pharmaml_walkthrough.ipynb with nbclient."""
import nbformat
from nbclient import NotebookClient

MD = """\
# PharmaML Toolkit — end-to-end walkthrough

This notebook exercises **every domain** of the `pharmaml` package on synthetic data:

| Section | Domain | What you see |
|---|---|---|
| 1 | `pharmaml.data` | **bring your own file**: CSV → target auto-detection → task inference (multiclass / regression) |
| 2 | `pharmaml.ml` | dataset → split → cross-validated training → evaluation → save/load |
| 3 | `pharmaml.ml` | multiclass classification (3 classes) end-to-end |
| 4 | `pharmaml.ml` | model-comparison leaderboard |
| 5 | `pharmaml.pharma` | non-compartmental PK, renal function, dosage design |
| 6 | `pharmaml.docking` | Vina output parsing + ligand-efficiency scoring |
| 7 | `pharmaml.chem` | IC50 → pIC50 endpoint conversion |

**Setup** — install the package first:

```bash
pip install -e ".[dev]"
```

No external datasets are required; every number below is generated with fixed seeds,
so the notebook is fully reproducible.
"""

MD_DATA = """\
## 1 · Bring your own data

`pharmaml.data.load_dataset` is the entry point for **your own files** (CSV/TSV/Excel):
it auto-detects the target column, infers whether the endpoint is classification
(**any number of classes**, string or integer labels) or regression, drops
non-numeric feature columns visibly, and reports class balance before training.

CLI equivalent:

```bash
pharmaml data inspect --data my_compounds.csv --target activity
pharmaml ml train   --data my_compounds.csv --task auto
```
"""

MD_ML = """\
## 2 · Machine learning (QSAR)

Synthetic binary-activity table → stratified split → `random_forest` with 5-fold
cross-validated grid search → held-out evaluation → artifact round-trip.
"""

MD_MULTICLASS = """\
## 3 · Multiclass classification (two or more classes)

A target with 3+ distinct values is handled natively: stratified splits keep every
class in both partitions, metrics are macro-averaged, the confusion matrix is k×k
and ROC-AUC switches to one-vs-rest.
"""

MD_LEADERBOARD = """\
## 4 · Model leaderboard

`compare_models` cross-validates several registry models and returns a
`DataFrame` sorted by the primary metric (ROC-AUC, one-vs-one — falling back to
balanced accuracy when a member lacks `predict_proba`).
"""

MD_PHARMA = """\
## 5 · Pharmaceutical calculations

Non-compartmental analysis of a plasma profile, then renal function
(Cockcroft–Gault), BSA scaling (Mosteller) and regimen design.
"""

MD_DOCKING = """\
## 6 · Molecular docking

Parsing a Vina output table and adding **ligand efficiency**
(`LE = |ΔG| / N_heavy`) — the size-normalised affinity metric.
"""

MD_CHEM = """\
## 7 · Endpoint conversion

pIC50 is the log-transformed IC50 that QSAR models actually regress on.
"""

cells = []

cells.append(nbformat.v4.new_markdown_cell(MD))

cells.append(nbformat.v4.new_code_cell(
"""import numpy as np
import pandas as pd

from pharmaml import __version__

print(f"pharmaml {__version__}")"""
))

cells.append(nbformat.v4.new_markdown_cell(MD_DATA))

cells.append(nbformat.v4.new_code_cell(
"""import tempfile
from pathlib import Path

from pharmaml.data import infer_task, load_dataset

# Simulate a user file: descriptors + string target + SMILES metadata column.
rng = np.random.default_rng(5)
user = pd.DataFrame(rng.normal(size=(180, 6)), columns=[f"descr_{i}" for i in range(6)])
user["activity"] = rng.choice(["inactive", "weak", "active"], size=180, p=[0.45, 0.35, 0.20])
user["SMILES"] = "CCO(C)C"

tmp = Path(tempfile.mkdtemp()) / "my_compounds.csv"
user.to_csv(tmp, index=False)          # <- replace with your own path

ds = load_dataset(tmp)                  # target auto-detected -> 'activity'
print(ds.summary())
print("\\nclasses:", ds.classes)
print("dropped non-numeric columns:", ds.dropped_columns)
print("task inference on a pIC50-like target:",
      infer_task(pd.Series(np.linspace(4, 9, 100))))"""
))

cells.append(nbformat.v4.new_markdown_cell(MD_ML))

cells.append(nbformat.v4.new_code_cell(
"""from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=300, n_features=8, n_informative=5, flip_y=0.04, random_state=42
)
frame = pd.DataFrame(X, columns=[f"descr_{i:02d}" for i in range(X.shape[1])])
frame["Target"] = y
frame.head()"""
))

cells.append(nbformat.v4.new_code_cell(
"""from pharmaml.ml import train_model, evaluate_model, make_train_test_split

X_train, X_test, y_train, y_test = make_train_test_split(
    frame.drop(columns=["Target"]), frame["Target"], test_size=0.2, random_state=42
)

result = train_model(
    "random_forest", "classification", X_train, y_train, cv=5, tune=True
)
print(f"{result.name}: CV ROC-AUC {result.cv_mean:.3f} ± {result.cv_std:.3f} "
      f"in {result.fit_seconds:.1f}s")
print("Best params:", result.best_params)"""
))

cells.append(nbformat.v4.new_code_cell(
"""report = evaluate_model(result.model, X_test, y_test, "classification")
print(f"accuracy={report.accuracy:.3f}  balanced={report.balanced_accuracy:.3f}  "
      f"f1_macro={report.f1_macro:.3f}  roc_auc={report.roc_auc:.3f}")
print("Confusion matrix (rows=true, cols=pred):")
print(report.confusion)"""
))

cells.append(nbformat.v4.new_code_cell(
"""from pathlib import Path

from pharmaml.ml import save_model, load_model

Path("artifacts").mkdir(exist_ok=True)
save_model(
    result.model, "artifacts/rf_demo.joblib", name="random_forest",
    task="classification",
    features=[c for c in frame.columns if c != "Target"],
    best_params=result.best_params,
    metrics={"accuracy": report.accuracy},
)
loaded_model, metadata = load_model("artifacts/rf_demo.joblib")
print("Reloaded:", metadata["name"], "| accuracy", round(metadata["metrics"]["accuracy"], 3))"""
))

cells.append(nbformat.v4.new_markdown_cell(MD_MULTICLASS))

cells.append(nbformat.v4.new_code_cell(
"""from sklearn.datasets import make_classification as _mk

X3, y3 = _mk(
    n_samples=300, n_features=8, n_informative=6, n_redundant=0,
    n_classes=3, n_clusters_per_class=1, random_state=8,
)
multi = pd.DataFrame(X3, columns=[f"descr_{i:02d}" for i in range(X3.shape[1])])
multi["Target"] = y3

X_tr, X_te, y_tr, y_te = make_train_test_split(
    multi.drop(columns=["Target"]), multi["Target"], test_size=0.25
)
mc = train_model("logistic_regression", "classification", X_tr, y_tr, tune=False)
mc_report = evaluate_model(mc.model, X_te, y_te, "classification")

print(f"3-class accuracy={mc_report.accuracy:.3f}  f1_macro={mc_report.f1_macro:.3f}  "
      f"roc_auc(ovr)={mc_report.roc_auc:.3f}")
print("classes present in train and test folds:", sorted(y_tr.unique()), sorted(y_te.unique()))
print("k x k confusion matrix:\\n", mc_report.confusion)"""
))

cells.append(nbformat.v4.new_markdown_cell(MD_LEADERBOARD))

cells.append(nbformat.v4.new_code_cell(
"""from pharmaml.ml import compare_models

leaderboard = compare_models(
    frame.drop(columns=["Target"]), frame["Target"],
    "classification", models=["logistic_regression", "knn", "random_forest"], cv=5,
)
leaderboard"""
))

cells.append(nbformat.v4.new_markdown_cell(MD_PHARMA))

cells.append(nbformat.v4.new_code_cell(
"""from pharmaml.pharma.pk import nca_summary

times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
conc = [0.0, 9.8, 12.4, 10.2, 6.1, 2.3, 0.8, 0.05]

summary = nca_summary(times, conc)
pd.Series(summary.as_dict(), name="value").round(3)"""
))

cells.append(nbformat.v4.new_code_cell(
"""from pharmaml.pharma import (
    bsa_mosteller, creatinine_clearance, dose_by_bsa,
    half_life, loading_dose, maintenance_dose,
)

crcl = creatinine_clearance(age_years=68, weight_kg=70, serum_cr_mg_dl=1.1, sex="male")
bsa = bsa_mosteller(height_cm=175, weight_kg=70)

print(f"CrCl = {crcl:.1f} mL/min (male)   BSA = {bsa:.2f} m²")
print(f"BSA-scaled dose  = {dose_by_bsa(dose_per_m2=250, height_cm=175, weight_kg=70):.0f} mg")
print(f"Loading dose     = {loading_dose(vd=45.0, target_concentration=12.0):.0f} mg")
print(f"Maintenance dose = {maintenance_dose(clearance_=3.2, target_css=6.0, tau=8.0):.1f} mg q8h")
print(f"t½ at ke = 0.115 h⁻¹: {half_life(0.115):.1f} h")"""
))

cells.append(nbformat.v4.new_markdown_cell(MD_DOCKING))

cells.append(nbformat.v4.new_code_cell(
"""from pharmaml.docking import ligand_efficiency, parse_vina_output

VINA_LOG = \"\"\"\\
Estimated Free Energy of Binding   = -10.3 kcal/mol
mode | affinity | dist from best mode
     |          | rmsd l.b. | rmsd u.b.
-----+----------+-----------+----------
   1      -10.3          0          0
   2       -9.6      1.842      2.301
   3       -9.4      2.105      2.777
   4       -8.9      2.733      3.418
-----+----------+-----------+----------
\"\"\"

poses = parse_vina_output(VINA_LOG)
table = pd.DataFrame(
    [(p.mode, p.affinity, p.rmsd_lb, p.rmsd_ub) for p in poses],
    columns=["mode", "affinity", "rmsd_lb", "rmsd_ub"],
)
table["ligand_efficiency"] = [ligand_efficiency(p.affinity, 32) for p in poses]
table.round(3)"""
))

cells.append(nbformat.v4.new_markdown_cell(MD_CHEM))

cells.append(nbformat.v4.new_code_cell(
"""from pharmaml.chem import ic50_to_pic50

for ic50_nM in (10.0, 100.0, 1000.0, 2500.0):
    print(f"IC50 = {ic50_nM:7.1f} nM  ->  pIC50 = {ic50_to_pic50(ic50_nM, 'nM'):.2f}")"""
))

nb = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbformat.validate(nb)

import os

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": "/home/z/my-project/pharma-ml-toolkit/examples"}},
    env={**os.environ, "PYTHONPATH": "/home/z/my-project/pharma-ml-toolkit/src"},
)
client.execute()

out = "/home/z/my-project/pharma-ml-toolkit/examples/pharmaml_walkthrough.ipynb"
nbformat.write(nb, out)
print(f"written: {out} | cells: {len(nb.cells)}")
