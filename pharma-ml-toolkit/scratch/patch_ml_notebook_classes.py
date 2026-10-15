"""Rewrite the ML notebook's classification section as two recipes.

User request (round 100x-15): "In the classification section, put a code
for 2 classes and another code for multiple classes (with the ability to
select the number of classes)."

The notebook already had the *binary* flow (§6–§7, target with exactly 2
classes) and a §8 multiclass demo — but §8 hardcoded ``n_classes=3`` via
raw sklearn.  This patch turns §8 into **Recipe B** with a selectable
class count:

* §7 header re-framed as **Recipe A — two classes (binary)**;
* §8 header re-framed as **Recipe B — K classes, select K**;
* the §8 data cell now uses the *shipped* generator
  (``examples/make_synthetic_qsar_data.py::make_classification_table``)
  with ``N_CLASSES = 4`` and pharma-style activity labels sliced to K;
* the §8 train/eval cell reports the class count via ``N_CLASSES`` and
  the confusion/ROC titles become f-strings on it;
* the title §-table and the §14 recap table gain the class-count row;
* the whole notebook is re-executed with nbclient (so every output is
  committed) and ``strip_notebook_outputs.py`` re-runs to refresh the
  no-output twin (the drift-guard test enforces source parity).

Run from the toolkit root:

    /home/z/.venv/bin/python scratch/patch_ml_notebook_classes.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "complete_ml_workflow.ipynb"

S7_OLD_HEAD = "## 7 · Evaluate the classifier\n"
S7_NEW_HEAD = "## 7 · Recipe A — evaluate the two-class (binary) classifier\n"

S7_OLD_TAIL = """The `ClassificationReport` carries every metric plus the confusion matrix;
the plots below are the two figures the legacy scripts produced —
confusion heat-map and ROC curves — now multiclass-aware, with count/share
annotations and per-class one-vs-rest curves."""
S7_NEW_TAIL = """The `ClassificationReport` carries every metric plus the confusion matrix;
the plots below are the two figures the legacy scripts produced —
confusion heat-map and ROC curves — with count/share annotations. With
exactly two classes the report reads like a classic active/inactive
screen: a single ROC-AUC and a 2×2 matrix. (Recipe B — any number of
classes — is §8.)"""

S8_OLD_SOURCE = """## 8 · Multiclass — two or more classes

A target with 3+ distinct values is a first-class citizen: stratified
splits, k×k confusion matrix, one-vs-rest ROC per class, macro-averaged
metrics."""
S8_NEW_SOURCE = """## 8 · Recipe B — multiclass with a selectable class count

The same pipeline, any K: set ``N_CLASSES`` below to 2, 3, 4, 5, … and
everything adapts — stratified splits keep every class in both sets, the
confusion matrix grows to K×K, ROC curves become one-vs-rest per class,
and metrics macro-average across classes. (K = 2 reproduces §7's binary
behaviour through the identical code path.)"""

S8_DATA_OLD_ANCHOR = "from sklearn.datasets import make_classification as _mk"
S8_DATA_NEW_SOURCE = """\
# --- Recipe B: K classes — you pick K ------------------------------------
# The shipped generator (examples/make_synthetic_qsar_data.py) builds a
# descriptor table with any class count; set N_CLASSES and re-run.
from make_synthetic_qsar_data import make_classification_table

N_CLASSES = 4                                   # <-- select: 2, 3, 4, 5, ...
ACTIVITY_LABELS = ["inactive", "weak", "moderate", "strong", "potent"]

multi = make_classification_table(
    n_samples=360,
    n_classes=N_CLASSES,
    class_labels=ACTIVITY_LABELS[:N_CLASSES],
    seed=11,
)
multi.to_csv(WORK / "multiclass.csv", index=False)
ds_multi = load_dataset(WORK / "multiclass.csv")
print(ds_multi.summary())"""

S8_TRAIN_OLD_ANCHOR = 'mX_tr, mX_te, my_tr, my_te = make_train_test_split(ds_multi.X, ds_multi.y, test_size=0.25)'
S8_TRAIN_NEW_SOURCE = """\
mX_tr, mX_te, my_tr, my_te = make_train_test_split(ds_multi.X, ds_multi.y, test_size=0.25)

mc = train_model("logistic_regression", "classification", mX_tr, my_tr, tune=False)
mc_report = evaluate_model(mc.model, mX_te, my_te, "classification")

print(f"{N_CLASSES}-class accuracy={mc_report.accuracy:.3f}  f1_macro={mc_report.f1_macro:.3f}  "
      f"roc_auc(ovr)={mc_report.roc_auc:.3f}")

fig_multi_confusion = plot_confusion_matrix(mc_report, title=f"Multiclass confusion ({N_CLASSES} classes)")"""

S8_ROC_OLD_ANCHOR = 'fig_multi_roc = plot_roc_curves(mc.model, mX_te, my_te, title="Multiclass ROC — one vs rest")'
S8_ROC_NEW_SOURCE = """\
fig_multi_roc = plot_roc_curves(mc.model, mX_te, my_te, title=f"Multiclass ROC — {N_CLASSES} classes, one vs rest")"""

TITLE_OLD_ROW = "| 8 | Multiclass — 3-class target end-to-end | `pharmaml.ml` |"
TITLE_NEW_ROW = "| 8 | Multiclass recipe — K-class target, K selectable | `pharmaml.ml` |"

RECAP_ANCHOR = "| Split | `make_train_test_split(ds.X, ds.y, 0.2)` |"
RECAP_NEW = (
    "| Split | `make_train_test_split(ds.X, ds.y, 0.2)` |\n"
    "| Binary or K classes | `make_classification_table(n_classes=N_CLASSES)` |"
)


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)

    if any("N_CLASSES" in c.source for c in nb.cells):
        print("Recipe B already present — nothing to do")
        return 0

    changed = 0

    # --- §7 header: Recipe A framing ---------------------------------------
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.startswith(S7_OLD_HEAD):
            cell.source = S7_NEW_HEAD + cell.source[len(S7_OLD_HEAD):]
            changed += 1
            break
    for cell in nb.cells:
        if cell.cell_type == "markdown" and S7_OLD_TAIL in cell.source:
            cell.source = cell.source.replace(S7_OLD_TAIL, S7_NEW_TAIL)
            changed += 1
            break

    # --- §8 markdown: Recipe B framing --------------------------------------
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source == S8_OLD_SOURCE:
            cell.source = S8_NEW_SOURCE
            changed += 1
            break

    # --- §8 data cell: the generator with selectable K ----------------------
    for cell in nb.cells:
        if cell.cell_type == "code" and S8_DATA_OLD_ANCHOR in cell.source:
            cell.source = S8_DATA_NEW_SOURCE
            changed += 1
            break

    # --- §8 train/eval cell --------------------------------------------------
    for cell in nb.cells:
        if cell.cell_type == "code" and S8_TRAIN_OLD_ANCHOR in cell.source:
            cell.source = S8_TRAIN_NEW_SOURCE
            changed += 1
            break

    # --- §8 ROC cell ----------------------------------------------------------
    for cell in nb.cells:
        if cell.cell_type == "code" and S8_ROC_OLD_ANCHOR in cell.source:
            cell.source = S8_ROC_NEW_SOURCE
            changed += 1
            break

    # --- title §-table row ----------------------------------------------------
    for cell in nb.cells:
        if cell.cell_type == "markdown" and TITLE_OLD_ROW in cell.source:
            cell.source = cell.source.replace(TITLE_OLD_ROW, TITLE_NEW_ROW)
            changed += 1
            break

    # --- §14 recap row ----------------------------------------------------------
    for cell in nb.cells:
        if cell.cell_type == "markdown" and RECAP_ANCHOR in cell.source:
            cell.source = cell.source.replace(RECAP_ANCHOR, RECAP_NEW)
            changed += 1
            break

    print(f"rewrote {changed} cell(s)")
    if changed < 7:
        print("ERROR: expected 7 cell rewrites — anchors drifted")
        return 1

    # --- re-execute --------------------------------------------------------
    os.environ.update(
        {
            "LOKY_MAX_CPU_COUNT": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
    nbformat.write(nb, NOTEBOOK)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    errors = [
        out
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    print(
        f"notebook re-executed: {len(nb.cells)} cells, "
        f"{len(code_cells)} code cells, {len(errors)} error outputs"
    )
    if errors:
        for err in errors[:3]:
            print("ERROR:", err.get("ename"), err.get("evalue"))
        return 1

    # --- refresh the no-output twin (drift-guard parity) ------------------
    result = subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "strip_notebook_outputs.py")],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
