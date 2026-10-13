"""Verification script — mirrors examples/pharmaml_walkthrough.ipynb code cells exactly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pharma-ml-toolkit" / "src"))
BASE = Path(__file__).resolve().parent

# --- Cell 1: setup ---
import numpy as np
import pandas as pd
from pharmaml import __version__

print(f"pharmaml {__version__}")

# --- Cell 2: synthetic QSAR dataset ---
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=300, n_features=8, n_informative=5, flip_y=0.04, random_state=42
)
frame = pd.DataFrame(X, columns=[f"descr_{i:02d}" for i in range(X.shape[1])])
frame["Target"] = y
print(frame.head(3).round(2))

# --- Cell 3: split + train with cross-validation ---
from pharmaml.ml import train_model, evaluate_model, make_train_test_split

X_train, X_test, y_train, y_test = make_train_test_split(
    frame.drop(columns=["Target"]), frame["Target"], test_size=0.2, random_state=42
)
result = train_model(
    "random_forest", "classification", X_train, y_train, cv=5, tune=True
)
print(f"{result.name}: CV ROC-AUC {result.cv_mean:.3f} ± {result.cv_std:.3f} "
      f"in {result.fit_seconds:.1f}s")
print("Best params:", result.best_params)

# --- Cell 4: evaluate on the held-out split ---
report = evaluate_model(result.model, X_test, y_test, "classification")
print(f"accuracy={report.accuracy:.3f}  balanced={report.balanced_accuracy:.3f}  "
      f"f1_macro={report.f1_macro:.3f}  roc_auc={report.roc_auc:.3f}")
print("Confusion matrix (rows=true, cols=pred):\n", report.confusion)

# --- Cell 5: model leaderboard ---
from pharmaml.ml import compare_models

leaderboard = compare_models(
    frame.drop(columns=["Target"]), frame["Target"],
    "classification", models=["logistic_regression", "knn", "random_forest"], cv=5,
)
print(leaderboard.round(4).to_string(index=False))

# --- Cell 6: persistence round-trip ---
from pharmaml.ml import save_model, load_model

artifact = BASE / "rf_demo.joblib"
save_model(
    result.model, artifact, name="random_forest", task="classification",
    features=[c for c in frame.columns if c != "Target"],
    best_params=result.best_params, metrics={"accuracy": report.accuracy},
)
loaded_model, metadata = load_model(artifact)
print("Reloaded:", metadata["name"], "| accuracy", round(metadata["metrics"]["accuracy"], 3))

# --- Cell 7: non-compartmental PK ---
from pharmaml.pharma.pk import nca_summary

times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
conc = [0.0, 9.8, 12.4, 10.2, 6.1, 2.3, 0.8, 0.05]
summary = nca_summary(times, conc)
print(pd.Series(summary.as_dict()).round(3).to_string())

# --- Cell 8: renal function & dosing design ---
from pharmaml.pharma import (
    bsa_mosteller, creatinine_clearance, dose_by_bsa,
    loading_dose, maintenance_dose, half_life,
)

crcl = creatinine_clearance(age_years=68, weight_kg=70, serum_cr_mg_dl=1.1, sex="male")
bsa = bsa_mosteller(height_cm=175, weight_kg=70)
print(f"CrCl = {crcl:.1f} mL/min (male)   BSA = {bsa:.2f} m²")
print(f"BSA-scaled dose = {dose_by_bsa(dose_per_m2=250, height_cm=175, weight_kg=70):.0f} mg")
print(f"Loading dose = {loading_dose(vd=45.0, target_concentration=12.0):.0f} mg")
print(f"Maintenance dose = {maintenance_dose(clearance_=3.2, target_css=6.0, tau=8.0):.1f} mg q8h")
print(f"t½ at ke=0.115 h⁻¹: {half_life(0.115):.1f} h")

# --- Cell 9: docking result parsing & scoring ---
from pharmaml.docking import ligand_efficiency, parse_vina_output

VINA_LOG = """\
Estimated Free Energy of Binding   = -10.3 kcal/mol
mode | affinity | dist from best mode
     |          | rmsd l.b. | rmsd u.b.
-----+----------+-----------+----------
   1      -10.3          0          0
   2       -9.6      1.842      2.301
   3       -9.4      2.105      2.777
   4       -8.9      2.733      3.418
-----+----------+-----------+----------
"""
poses = parse_vina_output(VINA_LOG)
table = pd.DataFrame(
    [(p.mode, p.affinity, p.rmsd_lb, p.rmsd_ub) for p in poses],
    columns=["mode", "affinity", "rmsd_lb", "rmsd_ub"],
)
table["ligand_efficiency"] = [ligand_efficiency(p.affinity, 32) for p in poses]
print(table.round(3).to_string(index=False))

# --- Cell 10: endpoint conversion ---
from pharmaml.chem import ic50_to_pic50

for ic50_nM in (10.0, 100.0, 1000.0, 2500.0):
    print(f"IC50 = {ic50_nM:7.1f} nM  →  pIC50 = {ic50_to_pic50(ic50_nM, 'nM'):.2f}")

print("\nALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY")
