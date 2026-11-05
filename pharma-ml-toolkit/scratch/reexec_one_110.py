"""Re-execute ONE notebook for the v1.2.110 lineage-reconciliation ship.

The 100x-64 driver re-runs all eight notebooks in one process; this
companion takes a single name so the ship ritual can chunk the work
across bounded invocations (the sandbox reaps background processes,
so each notebook runs to completion synchronously). Same contract:
outputs cleared, nbclient under the single-threaded env, the
site-packages prefix scrub, the error/executed-count gate — then the
no-output twin is regenerated for that notebook alone.

Run from the toolkit root:

    python scratch/reexec_one_110.py complete_ml_workflow.ipynb
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
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
_SITE_PACKAGES = sysconfig.get_paths()["purelib"]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: reexec_one_110.py <notebook.ipynb>")
        return 2
    name = sys.argv[1]
    path = EXAMPLES / name
    if not path.exists():
        print(f"no such notebook: {name}")
        return 2

    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    started = time.perf_counter()
    client = NotebookClient(
        nb,
        timeout=520,
        kernel_name="python3",
        resources={"metadata": {"path": str(EXAMPLES)}},
    )
    client.execute()
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                raw = output.get("text", "")
                if isinstance(raw, str):
                    raw = [raw]
                joined = "".join(
                    t.replace(_SITE_PACKAGES + "/", "site-packages/")
                    for t in raw
                )
                output["text"] = joined.splitlines(keepends=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
