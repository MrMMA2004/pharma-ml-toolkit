"""Patch ``examples/complete_docking_workflow.ipynb`` for round 100x-143.

Adds the §12 tail — the census + the workers>1 pool the round's own
worklog candidate asked for ("the notebook's §12 could tour the census +
the workers>1 pool explicitly"):

* ``vina_batch_progress`` — the pre-flight resume census (round 27's
  ``batch_vina.py`` resume rule, library edition): a ligand whose
  ``<stem>_out.pdbqt`` AND non-empty ``<stem>.log`` exist is "already
  docked"; the census counts what a resume WOULD skip, with the SAME
  skip rule and the SAME collision-disambiguated naming the run itself
  uses.  Demonstrated on a three-ligand tree where one is docked, one
  crashed (zero-byte log -- a redo), one is untouched.
* ``run_vina_batch(..., max_parallel=2, skip_existing=True)`` — the
  workers>1 pool the Playground's Vina-batch card exposes, toured
  explicitly: the engine gate fires BEFORE queue time (no vina binary
  in the notebook environment, so the honest ``DockingError`` shows
  the same gating the walkthrough's §5 cell demonstrates for the
  sequential default).

The recap table gains the census row.  The whole notebook is then
re-executed with nbclient (real RDKit/Meeko in this environment) and
the ``_no_output`` twin is regenerated.

Run from the toolkit root:

    python scratch/patch_docking_notebook_143.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
NOTEBOOK = EXAMPLES / "complete_docking_workflow.ipynb"
WORKER_ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

NEW_CELL_MARKER = "vina_batch_progress"


def _fail(message: str) -> None:
    print(f"PATCH143 FAIL: {message}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# New §12 tail cells (after the parse_vina_config cell, before §13 Recap)
# ---------------------------------------------------------------------------


def tail_cells() -> list:
    census_cell = code(
        """\
from pharmaml.docking import run_vina_batch, vina_batch_progress
from pharmaml.exceptions import DockingError

# The pre-flight census (batch_vina.py's resume rule, library edition):
# a ligand whose <stem>_out.pdbqt AND non-empty <stem>.log exist is
# "already docked" -- the SAME skip rule and the SAME collision-
# disambiguated naming the run itself uses, so the count is a promise
# a resumed run keeps.
VBATCH_LIGANDS = WORK / "vbatch_ligands"
VBATCH_LIGANDS.mkdir(exist_ok=True)
for stem in ("lig01", "lig02", "lig03"):
    (VBATCH_LIGANDS / f"{stem}.pdbqt").write_text("REMARK ligand\\n", encoding="utf-8")

VBATCH_OUT = WORK / "vbatch_out"
VBATCH_OUT.mkdir(exist_ok=True)
# lig01 already docked (output + non-empty log); lig02's zero-byte log
# marks a CRASHED job (a redo, not a skip); lig03 never started
(VBATCH_OUT / "lig01_out.pdbqt").write_text(
    "REMARK VINA RESULT:    -7.1      0.000      0.000\\n", encoding="utf-8"
)
(VBATCH_OUT / "lig01.log").write_text(
    "Estimated Free Energy of Binding   = -7.1 (kcal/mol)\\n", encoding="utf-8"
)
(VBATCH_OUT / "lig02.log").write_text("", encoding="utf-8")

census = vina_batch_progress(VBATCH_LIGANDS, VBATCH_OUT)
print(f"{census.docked}/{census.total} already docked · {census.remaining} to go")
print(f"name collisions: {census.collisions}")

# The engine gate fires BEFORE queue time (no vina binary here) -- the
# same honest DockingError the walkthrough's sequential cell shows, now
# with the workers>1 pool the Playground's Vina-batch card exposes:
try:
    run_vina_batch(
        UPLOAD_CONF, VBATCH_LIGANDS, VBATCH_OUT,
        max_parallel=2, skip_existing=True,
    )
except DockingError as exc:
    print(f"[run_vina_batch max_parallel=2] {exc}")
"""
    )
    return [census_cell]


# ---------------------------------------------------------------------------
# Recap row
# ---------------------------------------------------------------------------


def update_recap(nb: nbformat.NotebookNode) -> None:
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        source = "".join(cell.source)
        if source.startswith("## 13 · Recap"):
            old = "| Vina config | `parse_vina_config(conf.txt)` → the box + engine knobs |"
            new = (
                "| Vina config | `parse_vina_config(conf.txt)` → the box + engine knobs |\n"
                "| Resume census | `vina_batch_progress(ligands, out)` → N docked · M to go |"
            )
            if old not in source:
                _fail("recap row for 'Vina config' not found (drifted)")
            cell.source = source.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Execution + twin (the 141 driver's own recipe)
# ---------------------------------------------------------------------------


def execute_notebook() -> None:
    from nbclient import NotebookClient

    os.environ.pop("MPLBACKEND", None)
    nb = nbformat.read(NOTEBOOK, as_version=4)
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
    # 100x-101/141/143: the executed outputs must not leak the build
    # machine's absolute venv paths -- rewrite this interpreter's own
    # site-packages prefix (computed, never hardcoded) to a neutral one
    # BEFORE writing (the scrub reexecute_all_notebooks_64.py applies).
    import sysconfig

    site_packages = sysconfig.get_paths()["purelib"]
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                raw = output.get("text", "")
                if isinstance(raw, str):
                    raw = [raw]
                joined = "".join(
                    t.replace(site_packages + "/", "site-packages/") for t in raw
                )
                output["text"] = joined.splitlines(keepends=True)
    nbformat.write(nb, NOTEBOOK)
    print(f"executed complete_docking_workflow.ipynb in {time.perf_counter() - started:.0f}s")


def regenerate_twin() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "strip_notebook_outputs.py")],
        capture_output=True,
        text=True,
        check=False,
        env=WORKER_ENV,
    )
    if result.returncode != 0:
        _fail(f"strip_notebook_outputs.py failed: {result.stderr[-400:]}")
    print("twins regenerated (idempotent)")


def verify() -> None:
    raw = NOTEBOOK.read_text(encoding="utf-8")
    twin = EXAMPLES.joinpath(
        NOTEBOOK.name.replace(".ipynb", "_no_output.ipynb")
    ).read_text(encoding="utf-8")
    source_pins = (
        "vina_batch_progress(VBATCH_LIGANDS, VBATCH_OUT)",
        "max_parallel=2, skip_existing=True",
        "already docked · {census.remaining} to go",
        "Resume census",
    )
    for pin in source_pins:
        if pin not in raw:
            _fail(f"source pin missing from executed notebook: {pin[:70]}")
        if pin not in twin:
            _fail(f"source pin missing from twin: {pin[:70]}")
    # the census verdict must appear in the EXECUTED output (1/3 docked,
    # 2 to go -- the crashed zero-byte log counts as a redo)
    if "1/3 already docked · 2 to go" not in raw:
        _fail("census verdict '1/3 already docked · 2 to go' missing from output")
    # the home-path leak gate (the standing notebook trap); the needle is
    # built from parts so THIS file never trips the archive's zero-match
    # scan (the standing full-zip leak gate)
    _HOME = "/".join(("", "home", "z"))
    for text, label in ((raw, "executed notebook"), (twin, "twin")):
        if _HOME in text:
            _fail(f"{label}: home-path leak: {_HOME}")
    print("verified: 4 source pins (both twins) + census output + leak gate")


def main() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    if any(NEW_CELL_MARKER in "".join(c.source) for c in nb.cells):
        print("census cell already present (idempotent) -- re-executing only")
    else:
        recap_index = next(
            (
                i
                for i, c in enumerate(nb.cells)
                if c.cell_type == "markdown"
                and "".join(c.source).startswith("## 13 · Recap")
            ),
            None,
        )
        if recap_index is None:
            _fail("## 13 · Recap cell not found (notebook drifted)")
        nb.cells[recap_index:recap_index] = tail_cells()
        nbformat.write(nb, NOTEBOOK)
        print(f"inserted 1 census cell before index {recap_index}")

    nb = nbformat.read(NOTEBOOK, as_version=4)
    update_recap(nb)
    nbformat.write(nb, NOTEBOOK)
    print("recap gained the Resume census row")

    execute_notebook()
    regenerate_twin()
    verify()
    print("driver 143 complete")


if __name__ == "__main__":
    main()
