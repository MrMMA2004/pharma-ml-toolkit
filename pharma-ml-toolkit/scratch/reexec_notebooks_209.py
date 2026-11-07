"""Re-execute all 8 notebooks at the shipped version, then regenerate the 7 twins.

The round-209 ship step. Worker-capped, dev-server-stopped (the OOM ritual):
every notebook's setup cell must print ``pharmaml <shipped version>`` or the
freshness guard (tests/test_notebook_variants.py) fails.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLKIT / "src"))

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONPATH"] = str(TOOLKIT / "src")

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

EXAMPLES = TOOLKIT / "examples"
EXECUTED = (
    "complete_ml_workflow",
    "complete_docking_workflow",
    "complete_pk_workflow",
    "complete_cellline_workflow",
    "complete_generative_workflow",
    "complete_deep_learning_workflow",
    "complete_tuning_workflow",
    "pharmaml_walkthrough",
)

VERSION = (TOOLKIT / "VERSION").read_text().strip()
print(f"re-executing {len(EXECUTED)} notebooks at pharmaml {VERSION}", flush=True)

started = time.time()
for name in EXECUTED:
    path = EXAMPLES / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(EXAMPLES)}},
    )
    t0 = time.time()
    client.execute()
    nbformat.write(nb, path)
    # sanity: any error output?
    errors = [
        out.ename
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    size = path.stat().st_size
    print(
        f"  {name}: {time.time() - t0:.0f}s, {size:,} B, "
        f"error-outputs={len(errors) or 0}{errors[:2]}",
        flush=True,
    )
    if errors:
        sys.exit(f"FATAL: {name} produced error outputs")

print(f"all {len(EXECUTED)} notebooks executed in {time.time() - started:.0f}s", flush=True)
