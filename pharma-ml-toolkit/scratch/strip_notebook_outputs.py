"""Generate no-output variants of the executed workflow notebooks.

The four shipped notebooks are committed with every output and inline
figure so readers can follow the analysis offline — the ML notebook
alone is ~1.1 MB. This script derives a bandwidth-friendly twin of each
of the three big workflows:

* every code cell's ``outputs`` are emptied and ``execution_count``
  reset to ``None`` (Jupyter renders them as never-run),
* a short note cell is inserted below the title explaining what was
  stripped and how to get the outputs back,
* nothing else changes — ``strip`` is idempotent, so re-running it
  after a notebook rebuild refreshes the variants in one command.

A pytest drift-guard (tests/test_notebook_variants.py) asserts the
variant's cell sources stay identical to the executed original's
(excluding the inserted note), so the twins can never silently fork.

Run from the toolkit root:

    /home/z/.venv/bin/python scratch/strip_notebook_outputs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

NOTEBOOKS = (
    "complete_ml_workflow.ipynb",
    "complete_docking_workflow.ipynb",
    "complete_pk_workflow.ipynb",
)

NOTE_SOURCE = """\
> **No-output variant.** This notebook is the bandwidth-friendly twin of
> `{name}`: identical cells, every output and inline figure stripped
> ({pct}% of the original size). Open the full twin to read the
> pre-computed results; run this one top-to-bottom (`Shift+Enter`) to
> produce them yourself — the setup cell lists the optional dependencies
> each stage needs.
"""


def variant_name(name: str) -> str:
    stem = name[: -len(".ipynb")]
    return f"{stem}_no_output.ipynb"


def strip(nb: nbformat.NotebookNode, name: str, pct: int) -> None:
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    # replace a previously inserted note (idempotent re-runs)
    nb.cells = [c for c in nb.cells if c.get("id") != "variant-note"]

    note = nbformat.v4.new_markdown_cell(NOTE_SOURCE.format(name=name, pct=pct))
    note.id = "variant-note"
    nb.cells.insert(1, note)


def main() -> int:
    for name in NOTEBOOKS:
        source = EXAMPLES / name
        target = EXAMPLES / variant_name(name)
        if not source.is_file():
            print(f"SKIP {name}: not found")
            continue

        nb = nbformat.read(source, as_version=4)
        src_kb = source.stat().st_size
        strip(nb, name, pct=0)  # placeholder; recomputed after write
        nbformat.write(nb, target)

        # rewrite once with the true percentage of the stripped file
        dst_kb = target.stat().st_size
        pct = round(100 * dst_kb / src_kb)
        strip(nb, name, pct=pct)
        nbformat.write(nb, target)
        dst_kb = target.stat().st_size

        print(
            f"{target.name}: {dst_kb / 1024:7.0f} KB "
            f"(from {src_kb / 1024:.0f} KB, {pct}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
