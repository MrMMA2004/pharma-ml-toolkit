"""Insert the DPF + torsion-QA cells (§4) and campaign-housekeeping cells
(§6) into ``examples/complete_docking_workflow.ipynb`` and re-execute.

The full builder (``build_docking_notebook.py``) is a one-shot transform,
so — like ``patch_docking_notebook.py`` before it — this patch finds
stable anchor cells and inserts after them:

* after the ``write_gpf`` cell (end of §4): the ``write_dpf`` cell
  (AutoDock4 parameter file, upgrades ``dpfer.py``) and the torsion-QA
  cell (``check_torsion_consistency`` / ``scan_torsion_consistency``,
  upgrades ``niggamatch.py`` / ``mismatch2.py``),
* after the ``batch_run_directories`` cell (end of §6): the campaign
  housekeeping cell (``collect_files`` + ``reindex_by_number``, upgrades
  ``extaacer.py`` / ``Prettier_extracted_dlgs.py``),
* the §4 markdown header and §11 recap table gain the new verbs.

Re-executes the whole notebook with nbclient, then re-runs
``strip_notebook_outputs.py`` so the no-output twin stays in sync with
the drift-guard test.

Run from the toolkit root:

    python scratch/patch_docking_notebook2.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "complete_docking_workflow.ipynb"

DPF_CELL_SOURCE = """\
from pharmaml.docking import write_dpf

# The DPF is AutoDock4's per-ligand instruction sheet (upgrades dpfer.py):
# ligand atom types, the "about" rotation centre — the ligand centroid —
# TORSDOF, and the full Lamarckian GA + Solis-Wets local-search block.
# Every GA knob is a keyword argument; ga_run defaults to 100.
dpf = write_dpf(reference, receptor_file="r.pdbqt", ga_run=100,
                output_path=WORK / "r.dpf")
for line in dpf.read_text().splitlines():
    if line.startswith(("ligand_types", "move ", "about ", "torsdof",
                        "ga_run", "set_ga", "set_psw1", "analysis")):
        print(line)"""

TORSION_CELL_SOURCE = """\
from pharmaml.docking import check_torsion_consistency, scan_torsion_consistency

# Torsion QA (upgrades niggamatch.py / mismatch2.py): prepare_ligand4.py
# writes one active-torsion REMARK and one BRANCH record per rotatable
# bond, plus a final TORSDOF total — the ligand is only valid when they
# agree. The legacy scripts exported this audit to Excel with red fills;
# pharmaml returns a typed report and auto-detects the writer lineage
# (MGLTools three-way rule vs Meeko's TORSDOF == BRANCH invariant).
qa = check_torsion_consistency(reference)
print(f"reference : TORSDOF={qa.torsdof}  remarks={qa.active_remarks}  "
      f"branches={qa.branches}  ->  {qa.status()}")

# The MGLTools lineage demo — exactly what the rar scripts audited:
mgl_style = WORK / "mgl_style.pdbqt"
mgl_style.write_text(
    "REMARK  1  A    between atoms: T1_2-CA   T3_4-CB\\n"
    "REMARK  2  A    between atoms: T5_6-CG   T7_8-CD\\n"
    "ROOT\\nATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\\n"
    "ENDROOT\\n"
    "BRANCH    1    2\\nBRANCH    2    3\\n"
    "TORSDOF 2\\n",
    encoding="utf-8",
)
print("mgltools :", check_torsion_consistency(mgl_style).status(),
      "(all three counts agree)")
mgl_style.write_text(mgl_style.read_text().replace("TORSDOF 2", "TORSDOF 3"),
                     encoding="utf-8")
print("mgltools :", check_torsion_consistency(mgl_style).status(),
      "(TORSDOF lies — the mismatch the Excel report flagged red)")

torsion_frame = scan_torsion_consistency(LIGAND_DIR)
print(f"\\nligands scanned: {len(torsion_frame)}")
print(torsion_frame.head(6).to_string(index=False))
assert (torsion_frame['status'] == 'OK').all(), 'torsion QA failed'"""

CAMPAIGN_CELL_SOURCE = """\
from pharmaml.docking import collect_files, reindex_by_number

# Campaign housekeeping (upgrades extaacer.py + Prettier_extracted_dlgs.py):
# gather every r.dlg of the tree into one folder, renamed after its
# ligand folder (with _1/_2 dedup suffixes) so results never overwrite…
collected = collect_files(DOCKING_DIR, filename="r.dlg")
print(collected[["folder", "destination", "copied"]].to_string(index=False))

# …then reindex the campaign summary onto the full ligand-number range
# so missing ligands surface as visible NaN rows instead of silently
# absent ones (00 here is the numbering gap by construction).
aligned = reindex_by_number(summary, column="folder", start=1, stop=14)
present = aligned["folder"].notna().sum()
print(f"\\nsummary rows reindexed onto 1..14: {len(aligned)} "
      f"({int(present)} present, {len(aligned) - int(present)} missing)")"""

SECTION4_HEADER_OLD = """## 4 · Grid box & GPF — enclose the binding site"""
SECTION4_HEADER_NEW = """## 4 · Grid box, GPF & DPF — enclose the binding site"""

RECAP_ROW_ANCHOR = "| GPF file | `write_gpf(ligand.pdbqt, box=box)` |"
RECAP_ROW_NEW = (
    "| GPF file | `write_gpf(ligand.pdbqt, box=box)` |\n"
    "| DPF file | `write_dpf(ligand.pdbqt, ga_run=100)` |\n"
    "| Torsion QA | `scan_torsion_consistency(ligand_dir)` → OK / MISMATCH |"
)


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)

    if any("write_dpf(reference" in c.source for c in nb.cells):
        print("DPF cell already present — nothing to do")
        return 0

    # --- §4: DPF + torsion QA after the write_gpf cell ------------------
    gpf_index = next(
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and "write_gpf(reference" in c.source
    )
    nb.cells.insert(gpf_index + 1, nbformat.v4.new_code_cell(DPF_CELL_SOURCE))
    nb.cells.insert(gpf_index + 2, nbformat.v4.new_code_cell(TORSION_CELL_SOURCE))
    print(f"inserted DPF + torsion-QA cells after cell {gpf_index}")

    # --- §4 markdown header ----------------------------------------------
    for cell in nb.cells:
        if cell.cell_type == "markdown" and SECTION4_HEADER_OLD in cell.source:
            cell.source = cell.source.replace(
                SECTION4_HEADER_OLD, SECTION4_HEADER_NEW
            )
            break

    # --- §6: campaign housekeeping after batch_run_directories -----------
    batch_index = next(
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code" and "batch_run_directories(DOCKING_DIR" in c.source
    )
    nb.cells.insert(batch_index + 1, nbformat.v4.new_code_cell(CAMPAIGN_CELL_SOURCE))
    print(f"inserted campaign-housekeeping cell after cell {batch_index}")

    # --- §11 recap table gains the new verbs ------------------------------
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
    NotebookClient(nb, timeout=900, kernel_name="python3").execute()
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
