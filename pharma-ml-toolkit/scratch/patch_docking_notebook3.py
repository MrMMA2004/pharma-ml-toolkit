"""Insert the ``§7 Torsion QA & campaign housekeeping`` section into
``examples/complete_docking_workflow.ipynb`` and re-execute.

The notebook already previews ``check_torsion_consistency`` inline in §4
(single-ligand view during preparation) and reindexes the §6 summary; this
patch adds the *campaign-scale* pass between §6 and the Vina parsing —
the audit you run before queueing the next round:

* a markdown intro (torsion-tree invariants per writer lineage + the
  dry-run-safe housekeeping verbs),
* a torsion-QA cell: ``scan_torsion_consistency`` over the §3 ligand
  folder plus one deliberately corrupted PDBQT (TORSDOF 3 vs two
  BRANCH records vs two active REMARKs) so the report carries a real
  MISMATCH row — the report ``ligprep_match.py`` / ``mismatch2.py``
  exported to Excel with red fills,
* a campaign-housekeeping cell: ``collect_files`` / ``cleanup_tree``
  previews (``dry_run=True``, including the ``removed_dir`` rows of the
  100x-02 preview semantics) and ``distribute_files`` +
  ``delete_named_files`` (adder.py's push, deleter.py's sweep — which
  also keeps the notebook idempotent),
* a markdown outro with the CLI twins.

Later section headers are renumbered (§7..§11 -> §8..§12), the title's
stage table gains the new row, and the recap table gains a
"Campaign sweep" row.

Re-executes the whole notebook with nbclient, then re-runs
``strip_notebook_outputs.py`` so the no-output twin stays in sync with
the drift-guard test.

Run from the toolkit root:

    python scratch/patch_docking_notebook3.py
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

SECTION_MARKER = "## 7 · Torsion QA & campaign housekeeping"

INTRO_SOURCE = """\
## 7 · Torsion QA & campaign housekeeping — audit the tree before the next run

A campaign is rarely *one* run: ligands get hand-edited, folders get renamed,
results get collected and re-collected. Before queueing the next round, two
cheap audits catch the silent failures.

**Torsion QA** (upgrades `others/Docking/ligand generation/ligprep_match.py` and
`mismatch2.py`, which exported the audit to Excel with red fills):
`prepare_ligand4.py` — the MGLTools ligand writer — emits one active-torsion
`REMARK` and one `BRANCH` record per rotatable bond plus a final `TORSDOF`
total, so a consistent ligand satisfies **TORSDOF == active REMARKs ==
BRANCHes**. Meeko-written ligands (§3) carry no active-torsion `REMARK`s at
all; their invariant is simply **TORSDOF == BRANCHes**.
`check_torsion_consistency` audits one file and auto-detects which lineage
wrote it; `scan_torsion_consistency` sweeps a whole tree and returns the
mismatch report as a DataFrame.

**Campaign housekeeping** (upgrades `others/dlg_extracter.py`,
`others/General/adder.py` and `others/General/delete.py`): `collect_files`
gathers named files from the tree, `distribute_files` pushes a file into every
ligand folder, `cleanup_tree` enforces a keep-list, and `delete_named_files`
sweeps one filename away. Every destructive verb takes `dry_run=True` and
*previews* the sweep — including the folders that would disappear — before a
single byte moves.
"""

TORSION_CELL_SOURCE = """\
from pharmaml.docking import check_torsion_consistency, scan_torsion_consistency

# Torsion QA over the §3 ligand folder: the three Meeko ligands carry no
# active-torsion REMARKs, so each row is checked against the Meeko
# invariant TORSDOF == BRANCHes:
ligand_audit = scan_torsion_consistency(LIGAND_DIR)

# The deliberate mismatch (a hand-edited ligand -- the corruption
# ligprep_match.py painted red): TORSDOF claims 3 rotatable bonds but the
# tree holds only two BRANCH records and two active REMARKs. Auto-
# detection sees the REMARKs and applies the MGLTools three-way rule:
bad_ligand = WORK / "bad_torsions.pdbqt"
bad_ligand.write_text(
    "REMARK  1  A    between atoms: T1_2-CA   T3_4-CB\\n"
    "REMARK  2  A    between atoms: T5_6-CG   T7_8-CD\\n"
    "ROOT\\n"
    "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\\n"
    "ENDROOT\\n"
    "BRANCH    1    2\\n"
    "BRANCH    2    3\\n"
    "TORSDOF 3\\n",
    encoding="utf-8",
)
bad = check_torsion_consistency(bad_ligand)
print(f"bad ligand : TORSDOF={bad.torsdof}  remarks={bad.active_remarks}  "
      f"branches={bad.branches}  ->  {bad.status()}")

# One report, both lineages -- the bad row is exactly what the legacy
# Excel audit flagged:
torsion_report = pd.concat(
    [
        ligand_audit,
        pd.DataFrame(
            [
                {
                    "ligand": str(bad_ligand.relative_to(WORK)),
                    "torsdof": bad.torsdof,
                    "active_remarks": bad.active_remarks,
                    "branches": bad.branches,
                    "status": bad.status(),
                }
            ]
        ),
    ],
    ignore_index=True,
)
assert (ligand_audit["status"] == "OK").all(), "clean ligands must audit OK"
assert int((torsion_report["status"] == "MISMATCH").sum()) == 1
torsion_report"""

CAMPAIGN_CELL_SOURCE = """\
from pharmaml.docking import cleanup_tree, collect_files, delete_named_files, distribute_files

# Campaign housekeeping over the run tree -- every destructive verb
# previews first (dry_run=True touches nothing on disk):

# 1. dlg_extracter.py's job, previewed: which r.dlg files a fresh
#    extraction would gather from the §6 run tree (zero bytes written):
recollect = collect_files(DOCKING_DIR, filename="r.dlg",
                          out_dir=WORK / "qa_collected", dry_run=True)
print(f"collect_files   : {len(recollect)} file(s) would be gathered "
      f"({int(recollect['copied'].sum())} copied -- dry run)")

# 2. delete.py's job, previewed: keep only the .dlg results. The grid
#    logs, the §6 collected/ copies and the empty benzocaine folder are
#    all slated for removal -- the "removed_dir" rows preview the
#    folders that would disappear (empty already, or would become empty):
cleanup_preview = cleanup_tree(DOCKING_DIR, keep={"r.dlg"},
                               remove_empty=True, dry_run=True)
print("cleanup_tree    :", cleanup_preview["action"].value_counts().to_dict())

# 3. adder.py's job: push one small file into every ligand folder of
#    §1's campaign tree, then sweep the copies away again --
#    deleter.py's verb, which doubles as this notebook's idempotency
#    guard (nothing of the demo survives the cell):
notes = WORK / "campaign_notes.txt"
notes.write_text("next docking round queued\\n", encoding="utf-8")
pushed = distribute_files([notes], LIGAND_FOLDERS)
swept = delete_named_files(LIGAND_FOLDERS, notes.name)
print(f"distribute/delete: {len(pushed)} copies of {notes.name} pushed, "
      f"{swept} swept away by delete_named_files")

# The cleanup preview itself, paths relative to the run tree:
preview = cleanup_preview.copy()
preview["path"] = [str(Path(p).relative_to(DOCKING_DIR)) for p in preview["path"]]
preview"""

OUTRO_SOURCE = """\
Every verb in this section has a CLI twin, so the same audits run outside the
notebook: `pharmaml docking torsion --dir ligand_folders/` (tree-wide torsion
QA), `pharmaml docking collect --dir docking_run/` (extraction), and the
`pharmaml docking campaign` group — `list`, `distribute`, `cleanup`,
`delete-named` — for the housekeeping sweeps above.
"""

# Full-header renumbers (each old string is unique in the notebook).
HEADER_RENUMBERS = (
    ("## 7 · Parse Vina output — pose tables",
     "## 8 · Parse Vina output — pose tables"),
    ("## 8 · Rescoring — ligand efficiency + consensus",
     "## 9 · Rescoring — ligand efficiency + consensus"),
    ("## 9 · Visualization — energy distribution & top ligands",
     "## 10 · Visualization — energy distribution & top ligands"),
    ("## 10 · Publish — the one-click visual bundle",
     "## 11 · Publish — the one-click visual bundle"),
    ("## 11 · Recap", "## 12 · Recap"),
)

# Title stage-table rows: §7..§10 become §8..§11 with the new §7 on top.
TITLE_ROWS_OLD = """\
| 7 | Parse Vina output — pose tables | `pharmaml.docking` |
| 8 | Rescoring — ligand efficiency + consensus | `pharmaml.docking` |
| 9 | Visualization — energy distribution, top ligands | `pharmaml.viz` |
| 10 | Publish — one-click HTML report + csv/md exports | `pharmaml.viz` |"""

TITLE_ROWS_NEW = """\
| 7 | Torsion QA & campaign housekeeping — audit the tree before the next run | `pharmaml.docking` |
| 8 | Parse Vina output — pose tables | `pharmaml.docking` |
| 9 | Rescoring — ligand efficiency + consensus | `pharmaml.docking` |
| 10 | Visualization — energy distribution, top ligands | `pharmaml.viz` |
| 11 | Publish — one-click HTML report + csv/md exports | `pharmaml.viz` |"""

RECAP_ROW_ANCHOR = "| Scan campaign | `scan_docking_directory(docking_run)` |"
RECAP_ROW_NEW = (
    "| Scan campaign | `scan_docking_directory(docking_run)` |\n"
    "| Campaign sweep | `collect_files(tree, dry_run=True)` / "
    "`cleanup_tree(tree, keep={\"r.dlg\"})` |"
)


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)

    if any(SECTION_MARKER in c.source for c in nb.cells):
        print("Torsion-QA & housekeeping section already present — nothing to do")
        return 0

    # --- insert the new §7 before the "Parse Vina output" header --------
    vina_index = next(
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "markdown" and c.source.startswith(
            "## 7 · Parse Vina output"
        )
    )
    new_cells = [
        nbformat.v4.new_markdown_cell(INTRO_SOURCE),
        nbformat.v4.new_code_cell(TORSION_CELL_SOURCE),
        nbformat.v4.new_code_cell(CAMPAIGN_CELL_SOURCE),
        nbformat.v4.new_markdown_cell(OUTRO_SOURCE),
    ]
    nb.cells[vina_index:vina_index] = new_cells
    print(f"inserted 4 cells before cell {vina_index} (old §7 header)")

    # --- renumber the later section headers ------------------------------
    for old, new in HEADER_RENUMBERS:
        hits = [
            c for c in nb.cells
            if c.cell_type == "markdown" and old in c.source
        ]
        if len(hits) != 1:
            print(f"ERROR: header {old!r} found {len(hits)} time(s)")
            return 1
        hits[0].source = hits[0].source.replace(old, new)
    print("renumbered §7..§11 -> §8..§12")

    # --- title stage table gains the new row ------------------------------
    title = next(c for c in nb.cells if c.cell_type == "markdown"
                 and TITLE_ROWS_OLD in c.source)
    title.source = title.source.replace(TITLE_ROWS_OLD, TITLE_ROWS_NEW)
    print("title stage table renumbered")

    # --- recap table gains the campaign-sweep row --------------------------
    recap = next(
        c for c in nb.cells
        if c.cell_type == "markdown" and RECAP_ROW_ANCHOR in c.source
    )
    recap.source = recap.source.replace(RECAP_ROW_ANCHOR, RECAP_ROW_NEW)
    print("recap table gained the 'Campaign sweep' row")

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
