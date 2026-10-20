"""Build + execute examples/complete_ml_workflow.ipynb with nbclient.

The notebook is the single "complete ML process — zero to one hundred"
artifact: every stage of a QSAR machine-learning project in structured
order, from importing your own data file to a published HTML report.

Run:  PYTHONPATH=src python scratch/build_ml_notebook.py
"""
import nbformat
from nbclient import NotebookClient

# ---------------------------------------------------------------- markdown
MD_HEADER = """\
# PharmaML — The Complete ML Workflow (Zero to One Hundred)

One notebook, every stage of a QSAR machine-learning project, in the order
you would actually run it:

| § | Stage | Toolkit |
|---|---|---|
| 1 | **Data entry** — import *your own* CSV/Excel | `pharmaml.data` |
| 2 | **Feature calculation** — SMILES → descriptors, fingerprints, similarity | `pharmaml.chem` |
| 3 | Exploratory analysis — audit, class balance, descriptor ranges | `pharmaml.viz` |
| 4 | Endpoint transformation — IC50 → pIC50 | `pharmaml.chem` |
| 5 | Split — stratified train/test | `pharmaml.ml` |
| 6 | Train — grid-searched random forest, CV folds | `pharmaml.ml` |
| 7 | Evaluate — confusion matrix, ROC, metrics table | `pharmaml.viz` |
| 8 | Multiclass — 3-class target end-to-end | `pharmaml.ml` |
| 9 | Regression — parity + residual diagnostics | `pharmaml.viz` |
| 10 | Leaderboard — every model, cross-validated | `pharmaml.ml` |
| 11 | Interpret — feature importance | `pharmaml.viz` |
| 12 | Deploy — save/load, score new compounds | `pharmaml.ml` |
| 13 | Publish — one-click HTML report + png/svg/pdf + csv/md/json | `pharmaml.viz` |

**Setup** — install the package with the plotting extra first:

```bash
pip install -e ".[reports]"
```

Every number below is generated from fixed seeds, so the notebook is fully
reproducible; replace the demo file in §1 with your own and re-run all cells.
"""

MD_1 = """\
## 1 · Data entry — bring your own file

The legacy scripts hardcoded dataset filenames inside ~30 `.py` files.
`load_dataset` is the single entry point for **your** data: point `MY_FILE`
at any CSV/TSV/TXT/Excel file (delimiter sniffed automatically), and the
target column is auto-detected, the task inferred (classification with any
number of classes vs regression), non-numeric columns dropped *visibly*.

CLI equivalents:

```bash
pharmaml data inspect --data my_compounds.csv
pharmaml ml train    --data my_compounds.csv --task auto
```

The demo file carries a `SMILES` column that `load_dataset` drops as
non-numeric — §2 turns those structures into features.
"""

MD_1B = """\
## 2 · Feature calculation — from SMILES to descriptors

The legacy archive scattered feature computation over four scripts
(`Descriptor calculator.py`, `Morgan_calculator.py`, `ECFP_calculation.py`,
`Mordred_calculation.py`) that silently assumed *"we are sure all SMILES are
valid"*. `pharmaml.chem` unifies them with explicit accounting:

* `compute_descriptors(smiles)` — all 2D RDKit descriptors (~217 columns),
  a Morgan count table, or the 1800+ Mordred suite (`descriptor_set=...`);
* `compute_fingerprints(smiles)` — Morgan/ECFP, RDKit path and atom-pair
  fingerprints, binary bits or counts;
* `compute_maccs(smiles)` — the 167-bit MACCS keys;
* `tanimoto_matrix(bits, bits)` — pairwise similarity for heat-maps;
* invalid SMILES are dropped *visibly* (row indices logged), never silently.

Requires RDKit — `pip install 'pharmaml[chem]'`.
"""

CODE_FC_A = """\
from pharmaml.chem import compute_descriptors, compute_fingerprints, compute_maccs

# Silence RDKit's C++ parser spam -- pharmaml logs invalid rows explicitly.
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

# A demo screening library of real drug-like structures (77 valid
# molecules), plus two *deliberately* broken entries so the invalid-SMILES
# accounting is visible.  (In your project: use your own SMILES column --
# frame["SMILES"] works directly.)
LIBRARY = [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O"),
    ("metformin", "CN(C)C(=N)N=C(N)N"),
    ("benzocaine", "CCOC(=O)c1ccc(N)cc1"),
    ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
    ("nicotine", "CN1CCCC1c1cccnc1"),
    ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
    ("metoprolol", "COCCc1ccc(OCC(C)NCC(C)C)cc1"),
    ("lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("procaine", "CCN(CC)CCOC(=O)c1ccc(N)cc1"),
    ("fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
    ("salicylic acid", "OC(=O)c1ccccc1O"),
    ("benzoic acid", "OC(=O)c1ccccc1"),
    ("benzaldehyde", "O=Cc1ccccc1"),
    ("benzamide", "NC(=O)c1ccccc1"),
    ("salicylamide", "NC(=O)c1ccccc1O"),
    ("methyl salicylate", "COC(=O)c1ccccc1O"),
    ("ethyl benzoate", "CCOC(=O)c1ccccc1"),
    ("phenyl acetate", "CC(=O)Oc1ccccc1"),
    ("acetophenone", "CC(=O)c1ccccc1"),
    ("benzophenone", "O=C(c1ccccc1)c1ccccc1"),
    ("benzyl alcohol", "OCc1ccccc1"),
    ("phenetole", "CCOc1ccccc1"),
    ("phenol", "Oc1ccccc1"),
    ("aniline", "Nc1ccccc1"),
    ("anisole", "COc1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("ethylbenzene", "CCc1ccccc1"),
    ("propylbenzene", "CCCc1ccccc1"),
    ("o-xylene", "Cc1ccccc1C"),
    ("mesitylene", "Cc1c(C)c(C)cc(C)c1"),
    ("styrene", "C=Cc1ccccc1"),
    ("allylbenzene", "C=CCc1ccccc1"),
    ("chlorobenzene", "Clc1ccccc1"),
    ("bromobenzene", "Brc1ccccc1"),
    ("fluorobenzene", "Fc1ccccc1"),
    ("nitrobenzene", "O=[N+]([O-])c1ccccc1"),
    ("benzonitrile", "N#Cc1ccccc1"),
    ("biphenyl", "c1ccc(-c2ccccc2)cc1"),
    ("diphenyl ether", "O(c1ccccc1)c1ccccc1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("indole", "c1ccc2[nH]ccc2c1"),
    ("quinoline", "c1ccc2ncccc2c1"),
    ("imidazole", "c1ncc[nH]1"),
    ("pyrrole", "c1cc[nH]c1"),
    ("furan", "c1ccoc1"),
    ("pyridine", "c1ccncc1"),
    ("pyrimidine", "c1cncnc1"),
    ("pyrazine", "c1cnccn1"),
    ("thiazole", "c1cscn1"),
    ("vanillin", "COc1cc(C=O)ccc1O"),
    ("eugenol", "COc1cc(CC=C)ccc1O"),
    ("guaiacol", "COc1ccccc1O"),
    ("catechol", "Oc1ccccc1O"),
    ("resorcinol", "Oc1cccc(O)c1"),
    ("hydroquinone", "Oc1ccc(O)cc1"),
    ("phthalic acid", "OC(=O)c1ccccc1C(=O)O"),
    ("cinnamic acid", "OC(=O)C=Cc1ccccc1"),
    ("cinnamaldehyde", "O=CC=Cc1ccccc1"),
    ("coumarin", "O=C1Oc2ccccc2C=C1"),
    ("sulfanilamide", "Nc1ccc(S(N)(=O)=O)cc1"),
    ("phenacetin", "CCOc1ccc(NC(C)=O)cc1"),
    ("glycine", "NCC(=O)O"),
    ("alanine", "CC(N)C(=O)O"),
    ("valine", "CC(C)CC(N)C(=O)O"),
    ("leucine", "CC(C)CCC(N)C(=O)O"),
    ("phenylalanine", "NC(Cc1ccccc1)C(=O)O"),
    ("tyrosine", "NC(Cc1ccc(O)cc1)C(=O)O"),
    ("serine", "NC(CO)C(=O)O"),
    ("histamine", "CNCCc1c[nH]cn1"),
    ("dopamine", "NCCc1ccc(O)c(O)c1"),
    ("acetylcholine", "CC[N+](C)(C)CCOC(C)=O"),
    ("broken-smiles", "not_a_smiles"),
    ("unclosed-ring", "C1CC"),
]

smiles_series = pd.Series([smi for _, smi in LIBRARY], name="SMILES")

descriptor_frame = compute_descriptors(smiles_series, progress=False)

print(f"{descriptor_frame.shape[0]} valid molecules x "
      f"{descriptor_frame.shape[1] - 1} descriptors (RDKit 2D set)")
print(f"dropped {len(LIBRARY) - len(descriptor_frame)} invalid SMILES ")"""

CODE_FC_B = """\
# A chemist-readable slice of the 217 computed columns.
KEY_DESCRIPTORS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
                   "NumRotatableBonds", "LabuteASA", "RingCount"]

key_frame = descriptor_frame[["SMILES"] + KEY_DESCRIPTORS].round(2)
key_frame.index = [LIBRARY[i][0] for i in descriptor_frame.index]
key_frame.index.name = "compound"
key_frame"""

CODE_FC_C = """\
# Fingerprints: binary bits vs counts, three different families.
bits_morgan, valid_morgan = compute_fingerprints(smiles_series, kind="morgan", n_bits=2048)
counts_morgan, _ = compute_fingerprints(smiles_series, kind="morgan", n_bits=2048,
                                        as_counts=True)
bits_path, _ = compute_fingerprints(smiles_series, kind="rdkit", n_bits=1024)
bits_pair, _ = compute_fingerprints(smiles_series, kind="atom_pair", n_bits=2048)
maccs, valid_maccs = compute_maccs(smiles_series)

print(f"morgan bits   : {bits_morgan.shape}  (mean on-bit rate {bits_morgan.mean():.3f})")
print(f"morgan counts : {counts_morgan.shape}")
print(f"rdkit path    : {bits_path.shape}")
print(f"atom pair     : {bits_pair.shape}")
print(f"MACCS keys    : {maccs.shape}  (valid rows: {len(valid_maccs)})")"""

CODE_FC_D = """\
from pharmaml.viz import plot_descriptor_distributions

fig_desc = plot_descriptor_distributions(
    descriptor_frame[KEY_DESCRIPTORS],
    title="Computed descriptor distributions (library)",
)"""

CODE_FC_E = """\
from pharmaml.chem import tanimoto_matrix
from pharmaml.viz import plot_similarity_heatmap

similarity = tanimoto_matrix(bits_morgan, bits_morgan)
labels = [LIBRARY[i][0] for i in valid_morgan]

fig_similarity = plot_similarity_heatmap(
    similarity, labels=labels,
    title="Library similarity — Tanimoto on Morgan bits",
)"""

CODE_FC_F = """\
# Structures in, model out: the computed descriptors become X.  Here a
# synthetic activity target (LogP + H-bond donors + noise) plays the role
# your measured IC50/pIC50 column plays in a real project.
rng = np.random.default_rng(7)
activity_score = (1.5 * key_frame["MolLogP"] + 1.0 * key_frame["NumHDonors"]
                  + rng.normal(scale=0.25, size=len(key_frame)))

struct_table = key_frame.reset_index(drop=True)
struct_table["activity"] = np.where(
    activity_score > np.median(activity_score), "active", "inactive"
)

struct_file = WORK / "library_with_descriptors.csv"
struct_table.to_csv(struct_file, index=False)

ds_struct = load_dataset(struct_file)   # target auto-detected -> 'activity'
print(ds_struct.summary())"""

CODE_FC_G = """\
from pharmaml.ml import make_train_test_split, train_model, evaluate_model

sX_tr, sX_te, sy_tr, sy_te = make_train_test_split(
    ds_struct.X, ds_struct.y, test_size=0.25, random_state=42
)
struct_result = train_model("logistic_regression", "classification", sX_tr, sy_tr,
                            cv=3, tune=True)
struct_report = evaluate_model(struct_result.model, sX_te, sy_te, "classification")

print(f"{struct_result.name}: CV ROC-AUC = {struct_result.cv_mean:.3f} "
      f"± {struct_result.cv_std:.3f}  ({len(ds_struct.y)}-compound demo library)")
print(f"held-out accuracy: {struct_report.accuracy:.2f} on {len(sy_te)} compounds")"""

CODE_FC_H = """\
from pharmaml.viz import plot_feature_importance

fig_struct_importance = plot_feature_importance(
    struct_result.model, feature_names=ds_struct.feature_names, top_n=8,
    title="Which computed descriptors drive activity?",
)"""

MD_2 = """\
## 3 · Exploratory data analysis

Look at the data *before* modelling: the audit summary, the class balance
(one bar per class, counts annotated) and the descriptor distributions
(histogram grid with means marked) surface problems the legacy scripts
discovered as cryptic sklearn crashes.
"""

MD_3 = """\
## 4 · Endpoint transformation — IC50 → pIC50

Potency data arrives as IC50 (nM); models regress on the log-transformed
pIC50. `add_pic50_column` appends the converted column to any table.
"""

MD_4 = """\
## 5 · Train / test split

`make_train_test_split` stratifies by the target so every class appears in
both partitions with the same proportions.
"""

MD_5 = """\
## 6 · Train one model (with grid search)

`train_model` wraps the estimator in the impute+scale pipeline and runs a
5-fold cross-validated grid search; per-fold scores are returned so the
mean is never mistaken for a guarantee.
"""

MD_6 = """\
## 7 · Evaluate the classifier

The `ClassificationReport` carries every metric plus the confusion matrix;
the plots below are the two figures the legacy scripts produced —
confusion heat-map and ROC curves — now multiclass-aware, with count/share
annotations and per-class one-vs-rest curves.
"""

MD_7 = """\
## 8 · Multiclass — two or more classes

A target with 3+ distinct values is a first-class citizen: stratified
splits, k×k confusion matrix, one-vs-rest ROC per class, macro-averaged
metrics.
"""

MD_8 = """\
## 9 · Regression path

The same pipeline with a continuous pIC50 target: parity scatter against
the identity line, and residual diagnostics (vs predicted + histogram)
that the legacy scripts never drew.
"""

MD_9 = """\
## 10 · Model leaderboard

`compare_models` cross-validates the whole registry at once — the job the
original archive spread over ~30 near-duplicate scripts. The error bars
are fold-to-fold standard deviations.
"""

MD_10 = """\
## 11 · Interpret — feature importance

Which descriptors drive the model? Works for tree ensembles
(`feature_importances_`), linear models (`|coef_|`) and explicit mappings.
"""

MD_11 = """\
## 12 · Deploy — persistence and scoring new compounds

`save_model` writes a joblib bundle *with provenance metadata*; `load_model`
round-trips it. New, unseen rows are scored with one `predict` call.
"""

MD_12 = """\
## 13 · Publish — the one-click visual bundle

`save_visual_bundle` renders every figure and table into a directory **and**
a self-contained `report.html` (figures embedded as base64 — no external
assets, renders identically offline). Tables are exported as csv *and*
markdown; figures as png *and* svg *and* pdf.
"""

MD_13 = """\
## 14 · Recap

| Stage | One-liner |
|---|---|
| Import your data | `ds = load_dataset("my.csv", target="activity")` |
| Compute descriptors | `compute_descriptors(frame["SMILES"])` |
| Compute fingerprints | `compute_fingerprints(smiles, kind="morgan")` |
| Similarity heat-map | `plot_similarity_heatmap(tanimoto_matrix(bits, bits))` |
| Audit | `ds.summary()` |
| Convert endpoints | `add_pic50_column(frame, "IC50 (nM)")` |
| Split | `make_train_test_split(ds.X, ds.y, 0.2)` |
| Train | `train_model("random_forest", ds.task, Xtr, ytr)` |
| Evaluate | `evaluate_model(model, Xte, yte, ds.task)` |
| Compare | `compare_models(ds.X, ds.y, ds.task)` |
| Interpret | `plot_feature_importance(model, ds.feature_names)` |
| Deploy | `save_model(...)` / `load_model(path)` |
| Publish | `save_visual_bundle(figures, tables, metrics, "reports")` |
"""

# ---------------------------------------------------------------- code
CODE_SETUP = """\
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pharmaml import __version__

pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 110)

# Scratch directory for this run's artifacts (use any path you like).
WORK = Path(tempfile.mkdtemp(prefix="ml_workflow_"))

print(f"pharmaml {__version__}")
print(f"artifacts -> {WORK}")"""

# §1 ------------------------------------------------------------------
CODE_1A = """\
# --- DEMO DATA ---------------------------------------------------------
# A realistic QSAR table: 8 descriptors, a 3-level activity target,
# plus SMILES/ID metadata columns that should NOT become features.
# (In your project: skip this cell and set MY_FILE to your own path.)
rng = np.random.default_rng(11)
n = 420
signal = rng.normal(size=(n, 3))
noise = rng.normal(scale=0.4, size=(n, 8))
descriptors = noise.copy()
descriptors[:, :3] += 0.6 * signal      # first 3 descriptors are informative
frame_demo = pd.DataFrame(descriptors, columns=[f"descr_{i:02d}" for i in range(8)])
score = frame_demo[["descr_00", "descr_01", "descr_02"]].sum(axis=1)
frame_demo["activity"] = np.where(
    score > np.quantile(score, 0.72), "active",
    np.where(score > np.quantile(score, 0.35), "weak", "inactive"),
)
frame_demo["SMILES"] = rng.choice(["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"], size=n)
frame_demo["compound_id"] = [f"CMPD-{i:04d}" for i in range(n)]

MY_FILE = WORK / "my_compounds.csv"      # <- replace with your own file
frame_demo.to_csv(MY_FILE, index=False)
print(f"wrote demo table: {MY_FILE}  ({n} rows x {frame_demo.shape[1]} columns)")"""

CODE_1B = """\
# --- THE ACTUAL IMPORT (works unchanged for your file) -----------------
from pharmaml.data import load_dataset

ds = load_dataset(MY_FILE)          # target auto-detected -> 'activity'

print(ds.summary())
print("\\nfeature columns:", ds.feature_names)
print("dropped (non-numeric):", ds.dropped_columns)"""

# §2 ------------------------------------------------------------------
CODE_2 = """\
from pharmaml.viz import plot_class_distribution, plot_descriptor_distributions

fig_balance = plot_class_distribution(ds, title="Class distribution (activity)")"""

CODE_2B = """\
fig_dists = plot_descriptor_distributions(
    ds.X, columns=ds.feature_names[:6], title="Descriptor distributions (first 6)"
)"""

CODE_2C = """\
# Per-class quick statistics for one descriptor.
ds.frame.groupby("activity")["descr_00"].describe().round(3)"""

# §3 ------------------------------------------------------------------
CODE_3 = """\
from pharmaml.chem import add_pic50_column

potency = pd.DataFrame(
    {"compound": ["A", "B", "C", "D"],
     "IC50 (nM)": [12.0, 240.0, 1850.0, 92.0]}
)
converted = add_pic50_column(potency, source_col="IC50 (nM)")
converted.round(3)"""

# §4 ------------------------------------------------------------------
CODE_4 = """\
from pharmaml.ml import make_train_test_split

X_train, X_test, y_train, y_test = make_train_test_split(
    ds.X, ds.y, test_size=0.2, random_state=42
)

split_audit = pd.DataFrame([
    {"partition": "train", "rows": len(y_train),
     **{f"n_{cls}": int((y_train == cls).sum()) for cls in ds.classes}},
    {"partition": "test", "rows": len(y_test),
     **{f"n_{cls}": int((y_test == cls).sum()) for cls in ds.classes}},
])
split_audit"""

# §5 ------------------------------------------------------------------
CODE_5 = """\
from pharmaml.ml import train_model

result = train_model(
    "random_forest", "classification", X_train, y_train, cv=5, tune=True
)
print(f"{result.name}: CV ROC-AUC = {result.cv_mean:.3f} ± {result.cv_std:.3f} "
      f"({result.fit_seconds:.1f}s)")
print("best params:", result.best_params)"""

CODE_5B = """\
from pharmaml.viz import plot_cv_scores

fig_cv = plot_cv_scores(result, metric="ROC-AUC (ovo)")"""

# §6 ------------------------------------------------------------------
CODE_6 = """\
from pharmaml.ml import evaluate_model

report = evaluate_model(result.model, X_test, y_test, "classification")
print(report)

metrics_frame = (
    pd.DataFrame([report.as_dict()]).T.rename(columns={0: "value"}).round(4)
)
metrics_frame"""

CODE_6B = """\
from pharmaml.viz import plot_confusion_matrix

fig_confusion = plot_confusion_matrix(report, title="Confusion matrix (held-out test)")"""

CODE_6C = """\
# Row-normalized view: per-class recall shares along the diagonal.
fig_confusion_norm = plot_confusion_matrix(
    report, normalize=True, title="Confusion matrix — row-normalized"
)"""

CODE_6D = """\
from pharmaml.viz import plot_roc_curves

fig_roc = plot_roc_curves(result.model, X_test, y_test, title="ROC — one vs rest")"""

# §7 ------------------------------------------------------------------
CODE_7A = """\
# A 3-class target with informative descriptors.
from sklearn.datasets import make_classification as _mk

X3, y3 = _mk(n_samples=360, n_features=8, n_informative=6, n_redundant=0,
             n_classes=3, n_clusters_per_class=1, flip_y=0.03, random_state=8)
multi = pd.DataFrame(X3, columns=[f"descr_{i:02d}" for i in range(8)])
multi["Target"] = np.array(["low", "medium", "high"])[y3]

ds_multi = load_dataset(multi.to_csv(WORK / "multiclass.csv", index=False) or WORK / "multiclass.csv")
print(ds_multi.summary())"""

CODE_7B = """\
mX_tr, mX_te, my_tr, my_te = make_train_test_split(ds_multi.X, ds_multi.y, test_size=0.25)

mc = train_model("logistic_regression", "classification", mX_tr, my_tr, tune=False)
mc_report = evaluate_model(mc.model, mX_te, my_te, "classification")

print(f"3-class accuracy={mc_report.accuracy:.3f}  f1_macro={mc_report.f1_macro:.3f}  "
      f"roc_auc(ovr)={mc_report.roc_auc:.3f}")

fig_multi_confusion = plot_confusion_matrix(mc_report, title="Multiclass confusion (3 classes)")"""

CODE_7C = """\
fig_multi_roc = plot_roc_curves(mc.model, mX_te, my_te, title="Multiclass ROC — one vs rest")"""

# §8 ------------------------------------------------------------------
CODE_8A = """\
from make_synthetic_qsar_data import make_regression_table

reg_table = make_regression_table(n_samples=260, seed=21)
reg_table.to_csv(WORK / "regression.csv", index=False)
ds_reg = load_dataset(WORK / "regression.csv")     # numeric target -> regression

print(ds_reg.summary())
print("task:", ds_reg.task, "| target:", ds_reg.target)"""

CODE_8B = """\
rX_tr, rX_te, ry_tr, ry_te = make_train_test_split(ds_reg.X, ds_reg.y, test_size=0.25)

reg_result = train_model("random_forest", "regression", rX_tr, ry_tr, tune=False)
reg_report = evaluate_model(
    reg_result.model, rX_te, ry_te, "regression", keep_predictions=True
)
print(reg_report)"""

CODE_8C = """\
from pharmaml.viz import plot_predicted_vs_actual, plot_residuals

fig_parity = plot_predicted_vs_actual(
    ry_te, reg_report.predictions,
    metrics={"RMSE": reg_report.rmse, "R²": reg_report.r2},
    title="Regression — predicted vs actual (pIC50)",
)"""

CODE_8D = """\
fig_residuals = plot_residuals(ry_te, reg_report.predictions, title="Regression residuals")"""

# §9 ------------------------------------------------------------------
CODE_9 = """\
from pharmaml.ml import compare_models

leaderboard = compare_models(
    ds.X, ds.y, "classification",
    models=["logistic_regression", "knn", "random_forest", "decision_tree"],
    cv=5,
)
leaderboard.round(4)"""

CODE_9B = """\
from pharmaml.viz import plot_model_comparison

fig_leaderboard = plot_model_comparison(leaderboard, title="Model leaderboard (5-fold CV)")"""

# §10 -----------------------------------------------------------------
CODE_10 = """\
from pharmaml.viz import plot_feature_importance

fig_importance = plot_feature_importance(
    result.model, feature_names=ds.feature_names, top_n=8,
    title="Random-forest feature importance",
)"""

# §11 -----------------------------------------------------------------
CODE_11A = """\
from pharmaml.ml import save_model, load_model

MODEL_PATH = WORK / "rf_activity.joblib"
save_model(
    result.model, MODEL_PATH,
    name=result.name, task="classification",
    features=ds.feature_names, best_params=result.best_params,
    metrics=report.as_dict(),
)
loaded, metadata = load_model(MODEL_PATH)
print("reloaded:", metadata["name"], "| trained:", metadata["trained_at"])"""

CODE_11B = """\
# Score five compounds the model has never seen.
new_rows = ds.X.sample(5, random_state=99)
predictions = loaded.predict(new_rows)

scored = pd.DataFrame({
    "compound_id": ds.frame.loc[new_rows.index, "compound_id"].values,
    "predicted_activity": predictions,
})
scored.to_csv(WORK / "scored_new_compounds.csv", index=False)
scored"""

# §12 -----------------------------------------------------------------
CODE_12A = """\
from pharmaml.viz import export_table, metrics_to_dataframe, save_visual_bundle

BUNDLE = WORK / "reports"
artifacts = save_visual_bundle(
    figures={
        "class_distribution": fig_balance,
        "descriptor_distributions": fig_desc,
        "similarity_heatmap": fig_similarity,
        "cv_scores": fig_cv,
        "confusion_matrix": fig_confusion,
        "roc_curves": fig_roc,
        "parity_plot": fig_parity,
        "residuals": fig_residuals,
        "model_leaderboard": fig_leaderboard,
        "feature_importance": fig_importance,
    },
    tables={
        "metrics": metrics_to_dataframe(report),
        "leaderboard": leaderboard,
        "computed_descriptors": key_frame,
        "split_audit": split_audit,
        "scored_compounds": scored,
    },
    metrics=report,
    out_dir=BUNDLE,
    title="QSAR activity model — complete workflow",
    subtitle=f"{result.name} · {ds.n_samples} compounds · {len(ds.classes)} classes",
    formats=("png", "svg"),
    close=True,    # figures already rendered inline above; free the memory
)
pd.DataFrame({"artifact": [p.name for p in artifacts],
              "kind": [p.suffix.lstrip('.') for p in artifacts]})"""

CODE_12B = """\
# The same tables in every format a report could need.
exports = [
    export_table(leaderboard, BUNDLE / "leaderboard.md"),
    export_table(metrics_to_dataframe(report), BUNDLE / "metrics.json"),
    export_table(split_audit, BUNDLE / "split_audit.html"),
    export_table(scored, BUNDLE / "scored_new_compounds.tsv"),
]
[p.name for p in exports]"""

CODE_12C = """\
# The self-contained report: open it in any browser — figures are embedded.
report_html = BUNDLE / "report.html"
size_kb = report_html.stat().st_size / 1024
print(f"{report_html}  ({size_kb:.0f} KB, zero external assets)")
print("First 3 lines:")
print("\\n".join(report_html.read_text().splitlines()[:3]))"""

# ---------------------------------------------------------------- build
cells = [nbformat.v4.new_markdown_cell(MD_HEADER),
         nbformat.v4.new_code_cell(CODE_SETUP)]
for md, code_blocks in (
    (MD_1, [CODE_1A, CODE_1B]),
    (MD_1B, [CODE_FC_A, CODE_FC_B, CODE_FC_C, CODE_FC_D, CODE_FC_E,
             CODE_FC_F, CODE_FC_G, CODE_FC_H]),
    (MD_2, [CODE_2, CODE_2B, CODE_2C]),
    (MD_3, [CODE_3]),
    (MD_4, [CODE_4]),
    (MD_5, [CODE_5, CODE_5B]),
    (MD_6, [CODE_6, CODE_6B, CODE_6C, CODE_6D]),
    (MD_7, [CODE_7A, CODE_7B, CODE_7C]),
    (MD_8, [CODE_8A, CODE_8B, CODE_8C, CODE_8D]),
    (MD_9, [CODE_9, CODE_9B]),
    (MD_10, [CODE_10]),
    (MD_11, [CODE_11A, CODE_11B]),
    (MD_12, [CODE_12A, CODE_12B, CODE_12C]),
    (MD_13, []),
):
    cells.append(nbformat.v4.new_markdown_cell(md))
    cells.extend(nbformat.v4.new_code_cell(code) for code in code_blocks)

nb = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbformat.validate(nb)

import os

client = NotebookClient(
    nb,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": "/home/z/my-project/pharma-ml-toolkit/examples"}},
    env={**os.environ, "PYTHONPATH": "/home/z/my-project/pharma-ml-toolkit/src",
         # cap BLAS/loky workers: the build sandbox is memory-constrained
         "LOKY_MAX_CPU_COUNT": "1", "OMP_NUM_THREADS": "1",
         "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
)
client.execute()

out = "/home/z/my-project/pharma-ml-toolkit/examples/complete_ml_workflow.ipynb"
nbformat.write(nb, out)

n_code = sum(1 for c in nb.cells if c.cell_type == "code")
n_img = sum(
    1 for c in nb.cells if c.cell_type == "code"
    for o in c.get("outputs", [])
    if o.get("data", {}).get("image/png")
)
print(f"written: {out} | cells: {len(nb.cells)} ({n_code} code) | inline figures: {n_img}")
