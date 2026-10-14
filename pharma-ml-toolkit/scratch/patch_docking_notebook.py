"""Insert the §2 receptor atom-type figure cell into
``examples/complete_docking_workflow.ipynb`` and re-execute the notebook.

The full builder (``build_docking_notebook.py``) is a one-shot transform of
the pre-§1/§2 notebook, so re-running it would double-insert sections.
This patch instead:

* finds the §2 cell that prints the atom-type census table
  (``extract_pdbqt_atom_types(RECEPTOR)``),
* inserts the ``plot_receptor_atom_types`` figure cell right after it
  (same source as the builder's, so a future full rebuild stays in sync),
* re-executes the whole notebook with nbclient so all outputs — including
  the new inline figure — are committed.

Run from the toolkit root:

    /home/z/.venv/bin/python scratch/patch_docking_notebook.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "complete_docking_workflow.ipynb"

FIGURE_CELL_SOURCE = """\
from pharmaml.viz import plot_receptor_atom_types

# The same census as a figure, colour-coded by chemistry — donor
# hydrogens, heteroatoms, non-polar carbons, metals/ions. A missing HD
# (donor) population is the classic symptom of a mis-prepared receptor:
receptor_fig = plot_receptor_atom_types(
    RECEPTOR, title="Lysozyme receptor — AutoDock atom types"
)"""


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)

    if any("plot_receptor_atom_types" in c.source for c in nb.cells):
        print("figure cell already present — nothing to do")
        return 0

    anchor = next(
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and "extract_pdbqt_atom_types(RECEPTOR)" in c.source
    )
    cell = nbformat.v4.new_code_cell(FIGURE_CELL_SOURCE)
    nb.cells.insert(anchor + 1, cell)
    print(f"inserted figure cell after cell {anchor} (now {len(nb.cells)} cells)")

    os.environ.update(
        {
            "LOKY_MAX_CPU_COUNT": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    NotebookClient(nb, timeout=900, kernel_name="python3").execute()
    nbformat.write(nb, NOTEBOOK)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    errors = [
        out
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    figures = sum(
        1
        for c in code_cells
        for out in c.get("outputs", [])
        if "image/png" in out.get("data", {})
    )
    print(f"error outputs: {len(errors)} | inline figures: {figures}")
    print(f"file size: {NOTEBOOK.stat().st_size / 1024:.0f} KB")
    if errors:
        for err in errors[:3]:
            print("ERROR:", err.get("ename"), str(err.get("evalue"))[:200])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
