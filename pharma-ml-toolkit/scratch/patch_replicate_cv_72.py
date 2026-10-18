"""72 driver: the replicate-CV cells (notebook side, closing the 100x-71 arc).

Round 100x-72's notebook work — the library (100x-71:
``replicate_variability`` + ``collapse_replicates(add_cv=True)``) and the
CLI (``--replicate-cv``) now reach the executed cell-line notebook:

1. INSERT three cells after the collapse cell (anchor: ``clean =
   collapse_replicates(screen)`` -- cell 7, §2's tail) and BEFORE
   ``## 3 · Normalization``:
   [markdown intro: the median is silent about spread, the [1.0, 3.0]
   vs [1.9, 2.1] teaching pair] + [code: the demo pair, the real
   screen's variability table, the add_cv companion column] + [markdown
   verdict: same median, ten-fold different noise; the shipped
   duplicate is byte-identical (CV 0.0), singletons carry NaN].
2. PATCH the import cell: ``replicate_variability`` joins the
   ``pharmaml.biology`` import block.
3. PATCH the §10 recap table: a "Replicate variability" row after
   "Collapse replicates".
4. EXECUTE the cell-line notebook with nbclient (cwd=examples/).  The
   new cells only ADD output: no existing name is rebound (AST-audited:
   cv_demo / demo_screen / variability / clean_cv are bound by no other
   cell and appear nowhere else in the notebook), so existing pins
   cannot shift.
5. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
6. VERIFY banned fabrication tokens absent + pinned values present.

Pins (deterministic, probed against data/cellline_screen.csv):

  cellline CV cell:
    demo   : pair A (n=2, median 2.0, std 1.414214, cv 70.710678) vs
             pair B (n=2, median 2.0, std 0.141421, cv 7.071068) --
             the SAME median, ten-fold different noise
    screen : MCF-7 crizotinib n=2, median/mean 3.712163, std 0.0,
             spread 0.0, cv 0.0 (the two rows are byte-identical)
    add_cv : the crizotinib frame -- MCF-7 cv 0.0, T-47D 3.871254 NaN,
             ... 23 rows (one duplicated pair + 22 singleton NaNs)
  earlier demo pins are unaffected (cells are inserted, never reordered).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(ROOT / "tests"))  # BANNED_FABRICATION_TOKENS

from test_notebook_variants import BANNED_FABRICATION_TOKENS

WORKER_ENV = {
    **os.environ,
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

CELLLINE_NOTEBOOK = "complete_cellline_workflow.ipynb"

CV_MARKER = "how noisy was that median"  # idempotency: the intro's title

CV_MD_INTRO = """\
### Replicate variability — how noisy was that median?

`collapse_replicates` resolves a duplicated pair by the median, and the
median is silent about spread: replicates `[1.0, 3.0]` and `[1.9, 2.1]`
both collapse to **2.0**. `replicate_variability` reports what the
collapse is about to discard — replicate count, median, mean, std,
spread and the coefficient of variation (`cv_pct` = std / |mean| × 100;
NaN for single observations or zero means, never a fabricated inf) —
and `collapse_replicates(screen, add_cv=True)` keeps a
`{metric}_replicate_cv_pct` companion column next to every collapsed
median so the noise travels with the answer. The CLI twin is
`pharmaml biology response-audit --replicate-cv`.
"""

CV_CODE = '''\
# The median collapse discards noise by construction: replicates [1.0, 3.0]
# and [1.9, 2.1] both collapse to the SAME median 2.0 -- the CV keeps the
# difference visible BEFORE the collapse (CV% = std / |mean| * 100).
cv_demo = pd.DataFrame({
    "cell_line": ["A", "A", "B", "B"],
    "compound": ["demo", "demo", "demo", "demo"],
    "LN_IC50": [1.0, 3.0, 1.9, 2.1],
})
demo_screen = load_response_table(cv_demo, metrics=["LN_IC50"])
print(replicate_variability(demo_screen).to_string(index=False))
print()

# The shipped screen: one duplicated pair -- MCF-7 x crizotinib.
variability = replicate_variability(screen)
print(variability.to_string(index=False))
print()

# add_cv=True keeps the noise NEXT TO the answer: a companion
# {metric}_replicate_cv_pct column rides along with the collapsed median
# (NaN for singleton pairs -- their spread is undefined, not zero).
clean_cv = collapse_replicates(screen, add_cv=True)
clean_cv.frame[clean_cv.frame["compound"] == "crizotinib"][
    ["cell_line", "compound", "LN_IC50", "LN_IC50_replicate_cv_pct"]
]
'''

CV_MD_VERDICT = """\
Same median, ten-fold different noise: pair A collapses to 2.0 from
replicates that disagree by 70.7 %, pair B from replicates that agree to
7.1 % — the median alone cannot tell them apart, `cv_pct` can. The
shipped screen's single duplicate (`MCF-7` × crizotinib, two
byte-identical rows) collapses with CV 0.0 — the median is exact — while
every singleton pair carries NaN: its spread is undefined, not zero.
Real GDSC exports carry genuine replicate noise; check `cv_pct` before
trusting a collapsed median.
"""

INSERTIONS = {
    CELLLINE_NOTEBOOK: (
        "clean = collapse_replicates(screen)",
        [
            ("markdown", CV_MD_INTRO),
            ("code", CV_CODE),
            ("markdown", CV_MD_VERDICT),
        ],
    ),
}

REPLACEMENTS = {
    CELLLINE_NOTEBOOK: [
        (
            "    load_response_table, audit_response_table, collapse_replicates,",
            "    load_response_table, audit_response_table, replicate_variability,\n"
            "    collapse_replicates,",
        ),
        (
            "| Collapse replicates | `collapse_replicates(screen)` |",
            "| Collapse replicates | `collapse_replicates(screen)` |\n"
            "| Replicate variability | `replicate_variability(screen)` |",
        ),
    ],
}

PINS_SOURCE = {
    CELLLINE_NOTEBOOK: [
        "### Replicate variability — how noisy was that median?",
        "replicate_variability(screen)",
        "add_cv=True",
        "LN_IC50_replicate_cv_pct",
        "| Replicate variability | `replicate_variability(screen)` |",
    ],
}

PINS_OUTPUT = {
    CELLLINE_NOTEBOOK: [
        "70.710678",  # demo pair A's cv_pct
        "7.071068",  # demo pair B's cv_pct (same median, 10x quieter)
        "MCF-7 crizotinib",  # the shipped duplicate pair
        "3.712163",  # its collapsed median (byte-identical replicates)
        "crizotinib  3.871254",  # T-47D's row: the singleton-NaN companion
    ],
}


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def insert_cells() -> None:
    import nbformat

    for name, (marker, cells) in INSERTIONS.items():
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        if any(CV_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the CV cells")
            continue
        anchor = next(
            (i for i, c in enumerate(nb.cells) if marker in "".join(c.source)),
            None,
        )
        if anchor is None:
            _fail(f"{name}: anchor marker {marker!r} not found")
        for offset, (kind, source) in enumerate(cells, start=1):
            if kind == "markdown":
                nb.cells.insert(anchor + offset, nbformat.v4.new_markdown_cell(source))
            else:
                nb.cells.insert(anchor + offset, nbformat.v4.new_code_cell(source))
        nbformat.write(nb, path)
        print(f"inserted {len(cells)} cell(s) into {name} after cell {anchor}")


def replace_cells() -> None:
    import nbformat

    for name, pairs in REPLACEMENTS.items():
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        changed = 0
        for old, new in pairs:
            applied = False
            for cell in nb.cells:
                src = "".join(cell.source)
                if new in src:
                    applied = True  # idempotent re-run: already applied
                    break
                if old in src:
                    cell.source = src.replace(old, new, 1)
                    applied = True
                    changed += 1
                    break
            if not applied:
                _fail(f"{name}: replacement anchor not found:\n{old[:80]}...")
        if changed:
            nbformat.write(nb, path)
        print(f"{name}: {changed} cell replacement(s) applied")


def execute_notebooks() -> None:
    import nbformat
    from nbclient import NotebookClient

    # The kernel must NOT inherit MPLBACKEND=Agg (the repo's pytest
    # default): with it, matplotlib_inline never activates and every
    # figure cell commits a text/plain-only repr (the 100x-43 lesson).
    os.environ.pop("MPLBACKEND", None)
    for name in (CELLLINE_NOTEBOOK,):
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
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
        nbformat.write(nb, path)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        executed = sum(1 for c in code_cells if c.get("execution_count"))
        errors = [
            o
            for c in code_cells
            for o in c.get("outputs", [])
            if o.output_type == "error"
        ]
        if errors or executed != len(code_cells):
            _fail(
                f"{name}: {len(errors)} error output(s), "
                f"{executed}/{len(code_cells)} cells executed"
            )
        figures = sum(
            1
            for c in code_cells
            for o in c.get("outputs", [])
            if o.get("output_type") == "display_data"
            and "image/png" in o.get("data", {})
        )
        print(
            f"executed {name}: {len(code_cells)} code cells, {figures} figures, "
            f"{time.perf_counter() - started:.1f}s"
        )


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


def _notebook_text(path: Path) -> str:
    """Source + output text of a notebook, JSON-decoded (raw file text
    escapes embedded quotes, which would break every pin containing one)."""
    import nbformat

    nb = nbformat.read(path, as_version=4)
    chunks: list[str] = []
    for cell in nb.cells:
        chunks.append("".join(cell.source))
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                chunks.append("".join(output.get("text", [])))
            elif output.get("output_type") in ("execute_result", "display_data"):
                text = output.get("data", {}).get("text/plain", [])
                chunks.append("".join(text) if isinstance(text, list) else str(text))
    return "\n".join(chunks)


def verify() -> None:
    for name, source_pins in PINS_SOURCE.items():
        raw = _notebook_text(EXAMPLES / name)
        # the executed notebook carries source + outputs; the twin carries
        # source only -- check BOTH so the narrative cannot drift from the
        # artifact.
        twin = _notebook_text(
            EXAMPLES / name.replace(".ipynb", "_no_output.ipynb")
        )
        for pin in source_pins:
            if pin not in raw:
                _fail(f"{name}: source pin missing from executed notebook: {pin[:70]}")
            if pin not in twin:
                _fail(f"twin: source pin missing: {pin[:70]}")
        for pin in PINS_OUTPUT[name]:
            if pin not in raw:
                _fail(f"{name}: output pin missing from executed notebook: {pin[:70]}")
        raw_file = (EXAMPLES / name).read_text(encoding="utf-8")
        twin_file = (EXAMPLES / name.replace(".ipynb", "_no_output.ipynb")).read_text(
            encoding="utf-8"
        )
        for token in BANNED_FABRICATION_TOKENS:
            if token in raw_file or token in twin_file:
                _fail(f"{name}: banned fabrication token present: {token!r}")
        print(
            f"verified {name}: {len(source_pins)} source pins + "
            f"{len(PINS_OUTPUT[name])} output pins + fabrication gate"
        )


def main() -> None:
    insert_cells()
    replace_cells()
    execute_notebooks()
    regenerate_twin()
    verify()
    print("driver 72 complete")


if __name__ == "__main__":
    main()
