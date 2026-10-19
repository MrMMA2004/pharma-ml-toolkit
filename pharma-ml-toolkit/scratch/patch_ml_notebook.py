"""Insert the Huber robust-regression cell into ``examples/
complete_ml_workflow.ipynb`` §9 and re-execute.

``huber_regression`` joined the model registry from the rar's
``Regression/Huber.py`` (see ``pharmaml.ml.models``) but no notebook had
demoed it yet.  Like ``patch_docking_notebook2.py`` before it, this
patch finds stable anchor cells instead of rebuilding from scratch:

* after the residuals cell (end of §9): the robustness demo — corrupt
  15 training pIC50 values by +4 log units, train both
  ``linear_regression`` and ``huber_regression`` on the corrupted data,
  and show Huber's clean-test RMSE beating least squares (squared error
  grows quadratically with outliers; Huber's linear-epsilon loss
  saturates),
* the §9 markdown header gains a robustness sentence,
* the §14 recap table gains a "Robust fit" row.

Re-executes the whole notebook with nbclient, then re-runs
``strip_notebook_outputs.py`` so the no-output twin stays in sync with
the drift-guard test.

Run from the toolkit root:

    /home/z/.venv/bin/python scratch/patch_ml_notebook.py
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

HUBER_CELL_SOURCE = """\
from pharmaml.ml import train_model, evaluate_model

# Robust regression — huber_regression (upgrades Regression/Huber.py).
# Gross activity outliers wreck least-squares fits because squared error
# grows quadratically with the residual; Huber's linear-epsilon loss
# saturates instead. Corrupt 15 training pIC50 values by +4 log units,
# then judge both models on the *clean* held-out set:
ry_corrupt = ry_tr.copy()
ry_corrupt.iloc[:15] += 4.0

lin_corrupt = train_model("linear_regression", "regression", rX_tr, ry_corrupt, tune=False)
hub_corrupt = train_model("huber_regression", "regression", rX_tr, ry_corrupt, tune=False)
lin_clean = evaluate_model(lin_corrupt.model, rX_te, ry_te, "regression")
hub_clean = evaluate_model(hub_corrupt.model, rX_te, ry_te, "regression")

print(f"clean-test RMSE — linear_regression: {lin_clean.rmse:.3f}")
print(f"clean-test RMSE — huber_regression : {hub_clean.rmse:.3f}   (same corrupted training data)")
assert hub_clean.rmse < lin_clean.rmse, "Huber should beat plain least squares under outliers\""""

SECTION9_OLD_TAIL = """residual diagnostics (vs predicted + histogram)
that the legacy scripts never drew."""

SECTION9_NEW_TAIL = """residual diagnostics (vs predicted + histogram)
that the legacy scripts never drew.  And when the assay contains gross
outliers, ``huber_regression`` keeps the fit honest where plain squared
error bends."""

RECAP_ROW_ANCHOR = "| Compare | `compare_models(ds.X, ds.y, ds.task)` |"
RECAP_ROW_NEW = (
    "| Compare | `compare_models(ds.X, ds.y, ds.task)` |\n"
    "| Robust fit | `train_model(\"huber_regression\", \"regression\", Xtr, ytr)` |"
)


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)

    if any("huber_regression" in c.source for c in nb.cells):
        print("Huber cell already present — nothing to do")
        return 0

    # --- §9: robustness demo after the residuals cell --------------------
    residual_index = next(
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and "fig_residuals = plot_residuals(" in c.source
    )
    nb.cells.insert(residual_index + 1, nbformat.v4.new_code_cell(HUBER_CELL_SOURCE))
    print(f"inserted Huber robustness cell after cell {residual_index}")

    # --- §9 markdown header ------------------------------------------------
    for cell in nb.cells:
        if cell.cell_type == "markdown" and SECTION9_OLD_TAIL in cell.source:
            cell.source = cell.source.replace(SECTION9_OLD_TAIL, SECTION9_NEW_TAIL)
            break

    # --- §14 recap table gains the robust-fit row --------------------------
    for cell in nb.cells:
        if cell.cell_type == "markdown" and RECAP_ROW_ANCHOR in cell.source:
            cell.source = cell.source.replace(RECAP_ROW_ANCHOR, RECAP_ROW_NEW)
            break

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
