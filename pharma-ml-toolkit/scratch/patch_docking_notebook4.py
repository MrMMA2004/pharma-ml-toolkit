"""Refresh the §6 batch-run comment in the docking notebook (task 25-F3).

Defect 7 changed ``run_autodock4(skip_existing=True)`` to trust the
AutoDock4 ``Successful Completion`` sentinel instead of bare file
presence, so the shipped interrupted ``10_nicotine`` folder is now
honestly re-run (and fails: missing ``r.dpf`` / no engine) instead of
silently returning ``ok=True``.  The re-executed notebook's committed
output already shows ``10_nicotine -> FAILED: missing r.dpf`` -- this
patch updates the one code-cell *comment* that still described the old
presence-based skip, so the notebook no longer contradicts its own
output.

Anchor-based and assert-once (the ``patch_docking_notebook*.py``
convention): the old comment must appear exactly once, and re-running
the script is a no-op.  Comments do not affect outputs, so no
re-execution is needed; the no-output twin is regenerated afterwards
via ``scratch/strip_notebook_outputs.py`` to keep the drift-guard green.

Run from the toolkit root:

    python scratch/patch_docking_notebook4.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "complete_docking_workflow.ipynb"

OLD_COMMENT = """\
# Idempotent campaign re-run: folders whose outputs (.glg/.dlg) already
# exist are skipped, per-folder failures are collected (benzocaine has no
# files at all), and the interrupted run is *not* re-docked because its
# .dlg exists -- which is exactly why the completion audit above exists
# as a separate check.
"""

NEW_COMMENT = """\
# Idempotent campaign re-run: a folder is skipped only when its r.dlg
# carries AutoDock4's 'Successful Completion' sentinel; per-folder
# failures are collected (benzocaine has no files at all), and the
# interrupted run IS re-docked -- the attempt fails honestly here
# (missing r.dpf, no engine installed) instead of silently reporting
# ok the way a bare file-presence check would.
"""


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    hits = sum(
        1
        for cell in nb.cells
        if cell.cell_type == "code" and OLD_COMMENT in cell.source
    )
    if hits == 0:
        print("unchanged (comment already patched)")
        return 0
    if hits != 1:
        print(f"ABORT: old comment found {hits} times (expected exactly 1)")
        return 1
    for cell in nb.cells:
        if cell.cell_type == "code" and OLD_COMMENT in cell.source:
            cell.source = cell.source.replace(OLD_COMMENT, NEW_COMMENT)
    nbformat.write(nb, NOTEBOOK)
    print("patched: §6 batch-run comment now describes the sentinel check")
    subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "strip_notebook_outputs.py")],
        check=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
