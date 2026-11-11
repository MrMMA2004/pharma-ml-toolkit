"""Re-execute all 8 notebooks at the shipped version.

Afterwards run ``python tools/strip_notebook_outputs.py`` to refresh the 7 twins.

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

from typing import Any

import nbformat
from nbclient import NotebookClient

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

# The round-137 ritual, as code (the round-250 find): executed notebooks
# must ship ZERO builder-absolute paths — sklearn's ConvergenceWarning
# traces carry the live venv prefix (``~/.venv/...``), which leaks
# the build machine and is meaningless on a fresh install.  Scrub the
# known prefix to a relative form, then verify nothing survived.
_HOME = Path.home()
_VENV_SITE = str(_HOME / ".venv" / "lib")


def _scrub_text(text: str) -> str:
    return text.replace(_VENV_SITE, "«venv»/lib").replace(str(_HOME), "«home»")


def _scrub_payload(value: Any) -> Any:
    """Scrub one output payload — nbformat reads ``text`` as ``str`` or list."""
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, list):
        return _scrub_text("".join(value)).splitlines(keepends=True)
    return value


def _scrub_notebook(nb: nbformat.NotebookNode) -> int:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if "text" in out:
                out["text"] = _scrub_payload(out["text"])
            data = out.get("data")
            if data:
                for key in list(data.keys()):
                    data[key] = _scrub_payload(data[key])
    # verify: no absolute home path may survive anywhere in the JSON
    dumped = nbformat.writes(nb)
    return dumped.count(str(_HOME))


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
    surviving = _scrub_notebook(nb)
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
        f"error-outputs={len(errors) or 0}{errors[:2]}, "
        f"home-path-survivors={surviving}",
        flush=True,
    )
    if errors:
        sys.exit(f"FATAL: {name} produced error outputs")
    if surviving:
        sys.exit(f"FATAL: {name} still carries {surviving} absolute home path(s)")

print(f"all {len(EXECUTED)} notebooks executed in {time.time() - started:.0f}s", flush=True)
