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
| 2 | Exploratory analysis — audit, class balance, descriptor ranges | `pharmaml.viz` |
| 3 | Endpoint transformation — IC50 → pIC50 | `pharmaml.chem` |
| 4 | Split — stratified train/test | `pharmaml.ml` |
| 5 | Train — grid-searched random forest, CV folds | `pharmaml.ml` |
| 6 | Evaluate — confusion matrix, ROC, metrics table | `pharmaml.viz` |
| 7 | Multiclass — 3-class target end-to-end | `pharmaml.ml` |
| 8 | Regression — parity + residual diagnostics | `pharmaml.viz` |
| 9 | Leaderboard — every model, cross-validated | `pharmaml.ml` |
| 10 | Interpret — feature importance | `pharmaml.viz` |
| 11 | Deploy — save/load, score new compounds | `pharmaml.ml` |
| 12 | Publish — one-click HTML report + png/svg/pdf + csv/md/json | `pharmaml.viz` |

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
"""

MD_2 = """\
## 2 · Exploratory data analysis

Look at the data *before* modelling: the audit summary, the class balance
(one bar per class, counts annotated) and the descriptor distributions
(histogram grid with means marked) surface problems the legacy scripts
discovered as cryptic sklearn crashes.
"""

MD_3 = """\
## 3 · Endpoint transformation — IC50 → pIC50

Potency data arrives as IC50 (nM); models regress on the log-transformed
pIC50. `add_pic50_column` appends the converted column to any table.
"""

MD_4 = """\
## 4 · Train / test split

`make_train_test_split` stratifies by the target so every class appears in
both partitions with the same proportions.
"""

MD_5 = """\
## 5 · Train one model (with grid search)

`train_model` wraps the estimator in the impute+scale pipeline and runs a
5-fold cross-validated grid search; per-fold scores are returned so the
mean is never mistaken for a guarantee.
"""

MD_6 = """\
## 6 · Evaluate the classifier

The `ClassificationReport` carries every metric plus the confusion matrix;
the plots below are the two figures the legacy scripts produced —
confusion heat-map and ROC curves — now multiclass-aware, with count/share
annotations and per-class one-vs-rest curves.
"""

MD_7 = """\
## 7 · Multiclass — two or more classes

A target with 3+ distinct values is a first-class citizen: stratified
splits, k×k confusion matrix, one-vs-rest ROC per class, macro-averaged
metrics.
"""

MD_8 = """\
## 8 · Regression path

The same pipeline with a continuous pIC50 target: parity scatter against
the identity line, and residual diagnostics (vs predicted + histogram)
that the legacy scripts never drew.
"""

MD_9 = """\
## 9 · Model leaderboard

`compare_models` cross-validates the whole registry at once — the job the
original archive spread over ~30 near-duplicate scripts. The error bars
are fold-to-fold standard deviations.
"""

MD_10 = """\
## 10 · Interpret — feature importance

Which descriptors drive the model? Works for tree ensembles
(`feature_importances_`), linear models (`|coef_|`) and explicit mappings.
"""

MD_11 = """\
## 11 · Deploy — persistence and scoring new compounds

`save_model` writes a joblib bundle *with provenance metadata*; `load_model`
round-trips it. New, unseen rows are scored with one `predict` call.
"""

MD_12 = """\
## 12 · Publish — the one-click visual bundle

`save_visual_bundle` renders every figure and table into a directory **and**
a self-contained `report.html` (figures embedded as base64 — no external
assets, renders identically offline). Tables are exported as csv *and*
markdown; figures as png *and* svg *and* pdf.
"""

MD_13 = """\
## 13 · Recap

| Stage | One-liner |
|---|---|
| Import your data | `ds = load_dataset("my.csv", target="activity")` |
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
        "split_audit": split_audit,
        "scored_compounds": scored,
    },
    metrics=report,
    out_dir=BUNDLE,
    title="QSAR activity model — complete workflow",
    subtitle=f"{result.name} · {ds.n_samples} compounds · {len(ds.classes)} classes",
    formats=("png", "svg"),
    close=False,   # keep figures open so they also render above
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
    env={**os.environ, "PYTHONPATH": "/home/z/my-project/pharma-ml-toolkit/src"},
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
