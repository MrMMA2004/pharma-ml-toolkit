"""Re-execute examples/complete_cellline_workflow.ipynb in place.

The neural.py target-standardisation fix (round 100x-18) changes the
§9 deep-learning leaderboard numbers, so the committed outputs must be
regenerated.  Source cells are untouched -- only outputs are replayed
with nbclient under the single-threaded env the build script used.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import nbformat
from nbclient import NotebookClient

PATH = "examples/complete_cellline_workflow.ipynb"

nb = nbformat.read(PATH, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

client = NotebookClient(
    nb,
    timeout=1200,
    kernel_name="python3",
    resources={"metadata": {"path": "examples"}},
)
client.execute()

nbformat.write(nb, PATH)
print("re-executed:", PATH)
for cell in nb.cells:
    if cell.cell_type == "code":
        src = "".join(cell.source)
        if "compare_models" in src and "dnn" in src:
            for out in cell.get("outputs", []):
                text = "".join(out.get("text", [])) or "".join(
                    out.get("data", {}).get("text/plain", [])
                )
                if text.strip():
                    print("--- leaderboard output ---")
                    print(text[:600])
