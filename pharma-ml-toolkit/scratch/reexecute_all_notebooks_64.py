"""Re-execute the six stale executed notebooks against current code.

Round 100x-64 — the freshness catch-up.  The committed outputs of five
notebooks still print ``pharmaml 1.0.0`` (executed during the round-26
conversion) and the generative one ``1.2.9`` (round 44), while the
shipped tree is 1.2.14 — committed outputs should be reproducible by
the code that ships with them.  This driver re-runs every executed
notebook top-to-bottom with nbclient under the single-threaded env
(MPLBACKEND popped so inline PNGs survive — the 100x-43 lesson),
sources untouched (the drift-guard pins them), then regenerates the
no-output twins via scratch/strip_notebook_outputs.py.

Run from the toolkit root:

    /home/z/.venv/bin/python scratch/reexecute_all_notebooks_64.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

for _var in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.pop("MPLBACKEND", None)

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

NOTEBOOKS = (
    "complete_ml_workflow.ipynb",
    "complete_docking_workflow.ipynb",
    "complete_pk_workflow.ipynb",
    "complete_cellline_workflow.ipynb",
    "complete_generative_workflow.ipynb",
    "complete_deep_learning_workflow.ipynb",
    "pharmaml_walkthrough.ipynb",
)


def main() -> int:
    total = time.perf_counter()
    for name in NOTEBOOKS:
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        started = time.perf_counter()
        client = NotebookClient(
            nb,
            timeout=2400,
            kernel_name="python3",
            resources={"metadata": {"path": str(EXAMPLES)}},
        )
        client.execute()
        nbformat.write(nb, path)

        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        executed = sum(1 for c in code_cells if c.get("execution_count"))
        errors = [o for c in code_cells for o in c.get("outputs", [])
                  if o.output_type == "error"]
        figures = sum(1 for c in code_cells for o in c.get("outputs", [])
                      if o.get("data", {}).get("image/png"))
        if errors or executed != len(code_cells):
            print(f"FAIL {name}: {len(errors)} error(s), "
                  f"{executed}/{len(code_cells)} cells")
            return 1
        print(f"executed {name}: {len(code_cells)} code cells, "
              f"{figures} figures, {time.perf_counter() - started:.0f} s",
              flush=True)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "strip_notebook_outputs.py")],
        check=True, capture_output=True, text=True,
    )
    print(result.stdout.strip())
    print(f"TOTAL {time.perf_counter() - total:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
