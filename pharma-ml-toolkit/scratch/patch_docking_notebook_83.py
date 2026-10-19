"""83 driver: the §9 consensus-breakdown cells (notebook side of 100x-82).

Round 100x-82 shipped ``consensus_breakdown`` (library) and the
``docking rescore`` CLI — the introspection twin of ``consensus_rescore``:
one ``<column>_norm`` unit column per evidence column plus the
``consensus`` they produced.  This driver lands the notebook axis (the
open item (a) from 100x-82): the docking workflow's §9 already
demonstrates *who* wins — it now also demonstrates *why*, mirroring how
100x-78 demonstrated ``validate_bundle_input`` in the ML notebook.

1. INSERT four cells after the consensus cell (anchor:
   ``ok["consensus"] = consensus_rescore(``, cell 44) and BEFORE
   ``## 10 · Visualization``:
   [markdown intro: the introspection story — the consensus says WHO
   wins, the breakdown says WHY] + [code: the explainable table —
   ``breakdown = consensus_breakdown(ok, columns=[...])`` joined with the
   folder names, sorted by consensus, all 12 rows] + [code: the driving
   evidence for the top-ranked ligand — the per-row argmax across the
   ``_norm`` units, printed as a verdict] + [markdown: the ``docking
   rescore`` CLI twin with ``--weights`` and the recipe line naming every
   column's direction and weight].
2. PATCH the §9 intro markdown: the consensus sentence gains the
   introspection-twin clause.
3. PATCH the §12 recap: the Rescore row gains ``consensus_breakdown``.
4. EXECUTE the docking notebook with nbclient (cwd=examples/,
   MPLBACKEND popped so inline PNGs survive — the 100x-43 lesson).  The
   new cells only ADD output: no existing name is rebound (AST-audited —
   breakdown / explainable / top_ligand / top_row / driver are bound by
   no other cell), so existing pins cannot shift.
5. TWINS — regenerate the *_no_output.ipynb variants (idempotent).
6. VERIFY banned fabrication tokens absent + pinned values present +
   the AST audit.

Pins (deterministic, probed against examples/data/docking_campaign/
before writing — the demo computes from the notebook's own §9 frame,
never fabricates one):

  explainable table (execute_result text/plain):
                folder  best_lowest_energy_norm  ...
    6      07_metformin   0.488   0.857   1.000   0.782   <- top row
  verdict (stream):
    07_metformin tops the table (consensus 0.782):
      best_lowest_energy_norm    0.488
      best_num_in_cluster_norm   0.857
      ligand_efficiency_norm     1.000
    driving evidence: ligand_efficiency_norm = 1.000
  (the seven-notebook re-execution follows this driver; the other six
  still print 1.2.31 until reexecute_all_notebooks_64.py runs.)
"""

from __future__ import annotations

import ast
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

DOCKING_NOTEBOOK = "complete_docking_workflow.ipynb"

# cell 44 — the consensus_rescore cell (the tail of §9)
BREAKDOWN_MARKER = 'ok["consensus"] = consensus_rescore('

BREAKDOWN_MD_INTRO = """\
### Why the winner wins — the consensus breakdown

`consensus_rescore` answers *who* wins; `consensus_breakdown` answers
*why*. The introspection twin runs the identical validation and
normalisation, but instead of collapsing the evidence into one score it
keeps a 0–1 `<column>_norm` unit for each evidence column (direction
applied, so higher is always better) next to the `consensus` they
produced. Read a row across the units to see which evidence drove a
ligand's rank.
"""

BREAKDOWN_CODE_TABLE = '''\
from pharmaml.docking import consensus_breakdown

# The explainable ranking: one 0-1 unit per evidence column, plus the
# consensus they produced (identical validation and normalisation to
# consensus_rescore -- the shared core, kept side by side).
breakdown = consensus_breakdown(
    ok, columns=["best_lowest_energy", "best_num_in_cluster", "ligand_efficiency"]
)

explainable = ok[["folder"]].join(breakdown).sort_values("consensus", ascending=False)
explainable.round(3)
'''

BREAKDOWN_CODE_DRIVER = '''\
# The WHY for the top of the table: read the winning row ACROSS the
# units -- the per-row argmax names the evidence that carried the rank.
# Metformin does NOT have the best energy in the table (theophylline
# does); ligand efficiency and cluster population carry it.
top_ligand = ok.loc[ok["consensus"].idxmax(), "folder"]
top_row = breakdown.loc[ok["consensus"].idxmax()]
driver = top_row.drop("consensus").idxmax()

print(f"{top_ligand} tops the table (consensus {top_row['consensus']:.3f}):")
for column, unit in top_row.drop("consensus").round(3).items():
    print(f"  {column:26s} {unit:.3f}")
print(f"driving evidence: {driver} = {top_row[driver]:.3f}")
'''

BREAKDOWN_MD_CLI = """\
The same explainable table ships from the terminal — `pharmaml docking
rescore` prints the leaderboard with every `_norm` unit, then the recipe
line naming each column's effective direction and weight, and writes the
full rescored CSV next to the input:

```text
pharmaml docking rescore --data summary.csv \\
    --columns best_lowest_energy,best_num_in_cluster,ligand_efficiency \\
    --weights 1,1,2
```

The recipe line reads `evidence: best_lowest_energy (lower=better,
weight 1), best_num_in_cluster (higher=better, weight 1),
ligand_efficiency (higher=better, weight 2)` — tilt the weights and
watch the units re-explain the new ranking (`--higher` / `--lower` flip
a column's direction).
"""

# --- cell replacements (old snippet -> new snippet) ------------------------

INTRO_OLD = (
    "into one 0–1 score.  Heavy-atom counts are\n"
    "computed from the actual SMILES with RDKit."
)
INTRO_NEW = (
    "into one 0–1 score; `consensus_breakdown`, its introspection twin,\n"
    "keeps one 0–1 `_norm` unit per evidence column next to the score, so\n"
    "the table also says *why* each ligand ranks where it does.  Heavy-atom\n"
    "counts are computed from the actual SMILES with RDKit."
)

RECAP_OLD = (
    "| Rescore | `ligand_efficiency_series(frame)` + "
    "`consensus_rescore(frame, cols)` |"
)
RECAP_NEW = (
    "| Rescore | `ligand_efficiency_series(frame)` + "
    "`consensus_rescore(frame, cols)` + `consensus_breakdown(frame, cols)` |"
)

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    DOCKING_NOTEBOOK: (
        BREAKDOWN_MARKER,
        [
            ("markdown", BREAKDOWN_MD_INTRO),
            ("code", BREAKDOWN_CODE_TABLE),
            ("code", BREAKDOWN_CODE_DRIVER),
            ("markdown", BREAKDOWN_MD_CLI),
        ],
    ),
}

REPLACEMENTS = {
    DOCKING_NOTEBOOK: [
        (INTRO_OLD, INTRO_NEW),
        (RECAP_OLD, RECAP_NEW),
    ],
}

# the names the new cells bind -- audited to appear nowhere else
NEW_NAMES = ("breakdown", "explainable", "top_ligand", "top_row", "driver")
NEW_CELL_MARKERS = (
    "breakdown = consensus_breakdown(",  # the table cell
    "top_ligand = ok.loc[ok[",  # the driving-evidence cell
)

PINS_OUTPUT = {
    DOCKING_NOTEBOOK: [
        # -- the explainable table (this round's cell) ---------------------
        (
            "folder  best_lowest_energy_norm  best_num_in_cluster_norm"
            "  ligand_efficiency_norm  consensus"
        ),
        (
            "6      07_metformin                    0.488"
            "                     0.857                   1.000      0.782"
        ),
        (
            "11     12_lidocaine                    0.161"
            "                     0.143                   0.000      0.101"
        ),
        # -- the driving-evidence verdict (this round's cell) --------------
        "07_metformin tops the table (consensus 0.782):",
        "  best_lowest_energy_norm    0.488",
        "  best_num_in_cluster_norm   0.857",
        "  ligand_efficiency_norm     1.000",
        "driving evidence: ligand_efficiency_norm = 1.000",
        # -- §9 consensus pins (must survive) ------------------------------
        "07_metformin               -8.13                 19.0",
    ],
}

PINS_SOURCE = {
    DOCKING_NOTEBOOK: [
        # -- the breakdown cells ---------------------------------------------
        "from pharmaml.docking import consensus_breakdown",
        'breakdown = consensus_breakdown(',
        (
            'ok, columns=["best_lowest_energy", "best_num_in_cluster",'
            ' "ligand_efficiency"]'
        ),
        'explainable = ok[["folder"]].join(breakdown)',
        'top_ligand = ok.loc[ok["consensus"].idxmax(), "folder"]',
        'driver = top_row.drop("consensus").idxmax()',
        "consensus_rescore` answers *who* wins; `consensus_breakdown` answers",
        "pharmaml docking rescore --data summary.csv",
        "evidence: best_lowest_energy (lower=better,",
        # -- the §9 intro + recap edits ---------------------------------------
        "its introspection twin,",
        "`consensus_breakdown(frame, cols)` |",
        # -- §9 source pins (must survive untouched) -------------------------
        'ok["consensus"] = consensus_rescore(',
        "## 10 · Visualization — energy distribution & top ligands",
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
        if any(
            "from pharmaml.docking import consensus_breakdown"
            in "".join(c.source)
            for c in nb.cells
        ):
            print(f"insert: {name} already carries the breakdown cells")
            continue
        anchor = next(
            (i for i, c in enumerate(nb.cells) if marker in "".join(c.source)),
            None,
        )
        if anchor is None:
            _fail(f"{name}: anchor marker {marker!r} not found")
        for offset, (kind, source) in enumerate(cells, start=1):
            if kind == "markdown":
                nb.cells.insert(
                    anchor + offset, nbformat.v4.new_markdown_cell(source)
                )
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
    # figure cell commits a text/plain-only repr -- the inline PNGs the
    # notebook guards pin would silently vanish.  Unset it for the
    # kernel spawn so ipykernel picks the inline backend.
    os.environ.pop("MPLBACKEND", None)
    for name in (DOCKING_NOTEBOOK,):  # only the docking notebook this round
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
        print(
            f"executed {name}: {len(code_cells)} code cells, "
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


def audit_new_names() -> None:
    """AST audit: the new cells' names are bound NOWHERE else.

    The new code cells only ADD bindings (breakdown / explainable /
    top_ligand / top_row / driver) -- no pre-existing cell binds any of
    them, so no re-bound name can shift a downstream cell's output.
    """

    import nbformat

    nb = nbformat.read(EXAMPLES / DOCKING_NOTEBOOK, as_version=4)
    inserted = {
        i
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code"
        and any(marker in "".join(c.source) for marker in NEW_CELL_MARKERS)
    }
    if len(inserted) != len(NEW_CELL_MARKERS):
        _fail(f"AST audit: expected 2 inserted cells, found {sorted(inserted)}")
    offenders: dict[str, list[int]] = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or i in inserted:
            continue
        tree = ast.parse("".join(cell.source))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names.extend(
                        n.id for n in ast.walk(target) if isinstance(n, ast.Name)
                    )
            elif isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name
            ):
                names.append(node.target.id)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.With)):
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name):
                        names.append(n.id)
                for item in getattr(node, "items", []):
                    if item.optional_vars is not None:
                        names.extend(
                            n.id
                            for n in ast.walk(item.optional_vars)
                            if isinstance(n, ast.Name)
                        )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.append(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.append(node.name)
            for name in names:
                if name in NEW_NAMES:
                    offenders.setdefault(name, []).append(i)
    if offenders:
        _fail(f"AST audit: new names bound elsewhere: {offenders}")
    print(
        f"AST audit: {', '.join(NEW_NAMES)} bound only in the "
        f"{len(inserted)} inserted cells"
    )


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
    audit_new_names()
    verify()
    print("driver 83 complete")


if __name__ == "__main__":
    main()
