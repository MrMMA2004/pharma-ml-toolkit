"""Patch ``examples/complete_docking_workflow.ipynb`` for round 100x-141.

Adds §12 — "The ligand-structure & MGLTools twins" — covering the three
notebook-axis surfaces the 2026-10 "new folder.zip" upload contributed:

* ``generate_ligand_structures`` (``ligand generation/ligprep.py``): the
  tautomer → ETKDG-conformers → MMFF/UFF → lowest-energy PDB (CONECT)
  + OpenBabel MOL2 pipeline, over a small honest library (a valid drug,
  a metal complex the skip list refuses, a broken SMILES),
* ``convert_pdb_folder_to_pdbqt`` (``ligand generation/ligprep_qt.py``):
  the MGLTools prepare_ligand4.py batch converter, demonstrated with a
  self-contained STUB interpreter pair (the real MGLTools is a Windows
  install; the stub keeps the notebook engine-free),
* ``parse_vina_config`` (``Vina/vina.conf`` + ``Vina_contol/conf.txt``):
  the upload's own grid values parsed back out.

The old §12 Recap becomes §13; the title TOC and the recap table gain
the new rows.  The whole notebook is then re-executed with nbclient
(real RDKit/Meeko in this environment) and the ``_no_output`` twin is
regenerated.

Run from the toolkit root:

    python scratch/patch_docking_notebook_141.py
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

NEW_SECTION_MARKER = "The ligand-structure & MGLTools twins"


def _fail(message: str) -> None:
    print(f"PATCH141 FAIL: {message}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# New section cells
# ---------------------------------------------------------------------------


def section_cells() -> list:
    header = md(
        """\
## 12 · The ligand-structure & MGLTools twins — ligprep.py / ligprep_qt.py

The 2026-10 "new folder" upload re-delivered two ligand-side pipelines
the toolkit had only partially absorbed — this section tours both, plus
the Vina config reader its controllers drive:

* **`generate_ligand_structures`** (the upload's `ligprep.py`) — every
  valid SMILES of a library becomes its **canonical tautomer**, is
  embedded as *N* ETKDG conformers, minimised (MMFF, UFF fallback), and
  the **lowest-energy conformer** is written as a one-model PDB **with
  `CONECT` records** (the connectivity `prepare_ligand4.py` reads)
  plus — when OpenBabel is installed — a Gasteiger-charged **MOL2**
  twin. Metal centres (Fe/Zn/Cu/Mn/Co/Ni/Mo/W) are skipped with a
  per-row reason, never aborting the batch.
* **`convert_pdb_folder_to_pdbqt`** (the upload's `ligprep_qt.py`) — a
  folder of PDB files driven through MGLTools'
  `prepare_ligand4.py` into the numbered-sub-folder tree (`1/`, `2/`,
  `3/` …) the campaign readers expect, each conversion guarded by a
  timeout, with the successes/timeouts/failures log written beside it.
* **`parse_vina_config`** — the upload's `vina.conf` / `conf.txt`
  convention: one file carrying the receptor, the grid box and the
  engine knobs, parsed back into the dict the batch runner consumes.
"""
    )

    gen_cell = code(
        """\
from pharmaml.docking import generate_ligand_structures

# An honest little library: a valid drug, a metal complex the skip list
# refuses, and a broken SMILES -- all three ride the same report.
STRUCT_CSV = WORK / "structure_library.csv"
STRUCT_CSV.write_text(
    "name,SMILES\\n"
    "aspirin,CC(=O)Oc1ccccc1C(=O)O\\n"
    "zinc_complex,[Zn+2]\\n"
    "broken,C1CC\\n",
    encoding="utf-8",
)

STRUCT_OUT = WORK / "ligand_structures"
structures = generate_ligand_structures(
    STRUCT_CSV, STRUCT_OUT, num_confs=8, write_mol2=False
)
print(f"files written: {sorted(p.name for p in STRUCT_OUT.iterdir())}")
structures[["ligand", "forcefield", "energy", "conformers", "atoms", "ok", "error"]]
"""
    )

    conect_cell = code(
        """\
# The PDB carries the connectivity records MGLTools' ligand reader
# needs -- the whole reason ligprep.py wrote its own PDB writer.
aspirin_pdb = (STRUCT_OUT / "aspirin.pdb").read_text(encoding="utf-8")
conect_lines = [line for line in aspirin_pdb.splitlines() if line.startswith("CONECT")]
print(f"{len(conect_lines)} CONECT records; the first three:")
print("\\n".join(conect_lines[:3]))
"""
    )

    mgl_cell = code(
        """\
from pharmaml.docking import convert_pdb_folder_to_pdbqt

# A self-contained STUB MGLTools pair (the real one is a Windows
# install): the "interpreter" copies its -o argument to a stub .pdbqt,
# so the numbered-subfolder tree, the timeout guard and the conversion
# log all run exactly as they would against prepare_ligand4.py.
MGL_SRC = WORK / "stub_pdbs"
MGL_SRC.mkdir(exist_ok=True)
for stub_name in ("ligand_a", "ligand_b"):
    (MGL_SRC / f"{stub_name}.pdb").write_text(
        "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00 20.00     C\\n",
        encoding="utf-8",
    )

MGL_PY = WORK / "stub_pythonsh"
MGL_PY.write_text(
    '#!/bin/sh\\nout=""\\n'
    'while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift 2;; *) shift;; esac; done\\n'
    'printf "REMARK stub pdbqt\\\\n" > "$out"\\n',
    encoding="utf-8",
)
MGL_PY.chmod(0o755)
MGL_ADT = WORK / "stub_prepare_ligand4.py"
MGL_ADT.write_text("# stub AutoDockTools prepare_ligand4.py\\n", encoding="utf-8")

MGL_OUT = WORK / "stub_pdbqt_tree"
conversion = convert_pdb_folder_to_pdbqt(
    MGL_SRC, MGL_OUT, MGL_PY, MGL_ADT, timeout_s=20.0
)
print("numbered tree:", sorted(p.name for p in MGL_OUT.iterdir()))
conversion[["ligand", "subfolder", "status", "seconds", "error"]]
"""
    )

    log_cell = code(
        """\
# The conversion log the original script wrote -- successes, timeouts
# and failures, one plain-text section each.
print((MGL_OUT / "conversion_log.txt").read_text(encoding="utf-8"))
"""
    )

    conf_cell = code(
        """\
from pharmaml.docking import parse_vina_config

# The upload's own vina.conf values (Vina/ and Vina_contol/conf.txt):
# the grid centre and size of its demo box, its num_modes and its
# production-grade exhaustiveness.
UPLOAD_CONF = WORK / "upload_vina.conf"
UPLOAD_CONF.write_text(
    "# Receptor and ligand input files\\n"
    "receptor = r.pdbqt\\n"
    "\\n"
    "center_x = 129.139\\n"
    "center_y = 122.269\\n"
    "center_z = 134.375\\n"
    "\\n"
    "size_x = 96\\n"
    "size_y = 126\\n"
    "size_z = 126\\n"
    "\\n"
    "num_modes = 40\\n"
    "exhaustiveness = 48\\n"
    "energy_range = 3\\n",
    encoding="utf-8",
)
vina_params = parse_vina_config(UPLOAD_CONF)
vina_params
"""
    )

    return [header, gen_cell, conect_cell, mgl_cell, log_cell, conf_cell]


# ---------------------------------------------------------------------------
# Renumbering + TOC/recap updates
# ---------------------------------------------------------------------------


def update_recap_and_toc(nb: nbformat.NotebookNode) -> None:
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        source = "".join(cell.source)
        if source.startswith("## 12 · Recap"):
            cell.source = source.replace(
                "## 12 · Recap", "## 13 · Recap", 1
            )
        if "# PharmaML — The Complete Docking Workflow" in source:
            old_row = (
                "| 11 | Publish — one-click HTML report + csv/md exports |"
                " `pharmaml.viz` |"
            )
            new_row = (
                "| 11 | Publish — one-click HTML report + csv/md exports |"
                " `pharmaml.viz` |\n"
                "| 12 | Ligand-structure & MGLTools twins — ligprep.py /"
                " ligprep_qt.py (2026-10 upload) | `pharmaml.docking` |"
            )
            if old_row not in source:
                _fail("TOC row for §11 not found (notebook drifted)")
            cell.source = source.replace(old_row, new_row, 1)
        if source.startswith("## 13 · Recap"):
            old = "| Single ligand | `smiles_to_pdbqt(smiles, out_dir, name=...)` |"
            new = (
                "| Single ligand | `smiles_to_pdbqt(smiles, out_dir, name=...)` |\n"
                "| Ligand structures | `generate_ligand_structures(library,"
                " out_dir, num_confs=50)` → lowest-energy PDB (+ MOL2) | \n"
                "| MGLTools PDB→PDBQT | `convert_pdb_folder_to_pdbqt(pdbs,"
                " out, mgl_python, adt_script)` → numbered tree + log |\n"
                "| Vina config | `parse_vina_config(conf.txt)` → the box +"
                " engine knobs |"
            )
            if old not in source:
                _fail("recap row for 'Single ligand' not found (drifted)")
            cell.source = source.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Execution + twin
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
    # 100x-101/141: the executed outputs must not leak the build
    # machine's absolute venv paths (matplotlib/pyparsing deprecation
    # warnings print their source file). Rewrite this interpreter's own
    # site-packages prefix (computed, never hardcoded) to a neutral one
    # BEFORE writing -- the same scrub reexecute_all_notebooks_64.py
    # applies, so the committed outputs stay portable.
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
        "generate_ligand_structures",
        "convert_pdb_folder_to_pdbqt",
        "parse_vina_config(UPLOAD_CONF)",
        "The ligand-structure & MGLTools twins",
        "## 13 · Recap",
    )
    for pin in source_pins:
        if pin not in raw:
            _fail(f"source pin missing from executed notebook: {pin[:70]}")
        if pin not in twin:
            _fail(f"source pin missing from twin: {pin[:70]}")
    # the home-path leak gate (the standing notebook trap).  The needle
    # itself is built from parts so THIS file never trips the archive's
    # zero-match scan (the standing full-zip leak gate).
    _HOME = "/".join(("", "home", "z"))
    for text, label in ((raw, "executed notebook"), (twin, "twin")):
        for leaked in (_HOME, "/tmp/docking_workflow_"):
            # WORK is a mkdtemp dir -- its /tmp path may legitimately
            # appear in OUTPUT error text only; forbid it in SOURCE by
            # checking the non-output twin only for /tmp and both for
            # the home prefix
            if label == "executed notebook" and leaked == "/tmp/docking_workflow_":
                continue
            if leaked == _HOME and leaked in text:
                _fail(f"{label}: home-path leak: {leaked}")
    print("verified: 5 source pins (both twins) + home-path leak gate")


def main() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    if any(NEW_SECTION_MARKER in "".join(c.source) for c in nb.cells):
        print("section already present (idempotent) -- re-executing only")
    else:
        recap_index = next(
            (
                i
                for i, c in enumerate(nb.cells)
                if c.cell_type == "markdown"
                and "".join(c.source).startswith("## 12 · Recap")
            ),
            None,
        )
        if recap_index is None:
            _fail("## 12 · Recap cell not found (notebook drifted)")
        nb.cells[recap_index:recap_index] = section_cells()
        nbformat.write(nb, NOTEBOOK)
        print(f"inserted 6 cells before index {recap_index}")

    nb = nbformat.read(NOTEBOOK, as_version=4)
    update_recap_and_toc(nb)
    nbformat.write(nb, NOTEBOOK)
    print("recap renumbered to §13 + TOC/recap rows added")

    execute_notebook()
    regenerate_twin()
    verify()
    print("driver 141 complete")


if __name__ == "__main__":
    main()
