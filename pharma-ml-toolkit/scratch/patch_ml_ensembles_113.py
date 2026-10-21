#!/usr/bin/env python3
"""Insert the meta-ensemble section into complete_ml_workflow.ipynb.

Round 100x-113 — the notebook axis of the per-file coverage contract
reported the Stacking/Voting family (7 rar files) as NOTEBOOK-ABSENT:
the registry tour (§11) never demonstrates build_stacking /
build_voting.  This patch adds a subsection right after the tour cell
that builds, fits and evaluates both ensembles on the same splits the
notebook already trained single models on.

Idempotent: re-running replaces the subsection instead of duplicating.
Run from the toolkit root:  python scratch/patch_ml_ensembles_113.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "examples" / "complete_ml_workflow.ipynb"

MARKER = "### Meta-ensembles — stack them, vote on them"


def _md(source: str) -> Any:
    return nbformat.v4.new_markdown_cell(source)


def _code(source: str) -> Any:
    return nbformat.v4.new_code_cell(source)


CELLS: list[Any] = [
    _md("""\
### Meta-ensembles — stack them, vote on them

The registry tour above trains one family at a time; the two meta-
ensembles instead *combine* families.  ``build_voting`` (the
``Voting.py`` twins) lets the members cast soft votes (probability
averaging, weightable) or hard majority votes; ``build_stacking``
(the ``Stacking.py`` twins) has the members predict out-of-fold and a
meta-model blend those predictions.  Both take registry names, so any
combination the registry knows is one list away."""),
    _code("""\
from pharmaml.ml import build_stacking, build_voting

# Voting (upgrades Classification/Voting.py): three families cast soft
# votes, the random forest's counts double.
vote_clf = build_voting(
    "classification",
    ["random_forest", "knn", "logistic_regression"],
    voting="soft",
    weights=[2, 1, 1],
)
vote_clf.fit(X_train, y_train)
vote_report = evaluate_model(vote_clf, X_test, y_test, "classification")
print(f"soft voting  · accuracy {vote_report.accuracy:.3f}  "
      f"ROC-AUC {vote_report.roc_auc:.3f}")

# Stacking (upgrades Classification/Stacking.py): out-of-fold level-0
# predictions, logistic blender on top.
stack_clf = build_stacking(
    "classification",
    ["random_forest", "knn", "logistic_regression"],
    meta_model="logistic_regression",
)
stack_clf.fit(X_train, y_train)
stack_report = evaluate_model(stack_clf, X_test, y_test, "classification")
print(f"stacking     · accuracy {stack_report.accuracy:.3f}  "
      f"ROC-AUC {stack_report.roc_auc:.3f}")"""),
    _code("""\
# The regression twins (Voting reg.py / Stacking reg.py): members average
# predictions; the default ridge blender sits on the out-of-fold
# predictions.  train_model("voting"/"stacking", ...) reaches the same
# builders with the CV/tuning pipeline wrapped around them.
vote_reg = build_voting("regression", ["random_forest", "knn"])
vote_reg.fit(rX_tr, ry_tr)
print(f"voting regressor  · R² {vote_reg.score(rX_te, ry_te):.3f}")

stack_reg = build_stacking("regression", ["random_forest", "knn"])
stack_reg.fit(rX_tr, ry_tr)
print(f"stacking regressor · R² {stack_reg.score(rX_te, ry_te):.3f}")"""),
]


def main() -> int:
    nb = nbformat.read(NB_PATH, as_version=4)

    # idempotency: drop a previous copy of the subsection
    keep: list[Any] = []
    inside = False
    for cell in nb.cells:
        if cell.cell_type == "markdown" and cell.source.startswith(MARKER):
            inside = True
            continue
        if inside and cell.cell_type == "markdown" and (
            cell.source.startswith("## ") or cell.source.startswith("### ")
        ):
            inside = False
        if not inside:
            keep.append(cell)
    nb.cells = keep

    # insert right after the registry tour cell (## 11 · The full registry)
    insert_at = len(nb.cells)
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and cell.source.startswith(
            "## 11 · The full registry"
        ):
            # skip past the tour's markdown + code cells to the next "## "
            j = idx + 1
            while j < len(nb.cells) and not (
                nb.cells[j].cell_type == "markdown"
                and nb.cells[j].source.startswith("## ")
            ):
                j += 1
            insert_at = j
            break

    nb.cells[insert_at:insert_at] = CELLS
    nbformat.write(nb, NB_PATH)
    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"ml workflow now has {len(nb.cells)} cells ({code_cells} code); "
          f"ensemble section inserted at {insert_at}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
