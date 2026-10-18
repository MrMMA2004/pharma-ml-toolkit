"""70 driver: the prune cells (notebook side, closing the 100x-67 arc).

Round 100x-70's notebook work — the library (100x-67:
``crowding_distance`` + ``prune``), the CLI (``--pareto-keep``), the
playground (the crowding column) and the encoded figure (100x-69) now
reach the executed generative notebook:

1. INSERT three cells after the Pareto-front verdict cell (anchor:
   ``The front carries 26 of the 613 molecules`` -- cell 23, §4's tail)
   and BEFORE ``## 5 · Seeded generation``:
   [markdown intro: the unwieldy-front problem + the NSGA-II ranking]
   + [code: crowding_distance + prune(8) + the honest audit + the
   crowding-encoded scatter] + [markdown verdict: the corners survive
   and the strongest dominator does not].
2. PATCH the §10 publish cell: ``"pareto_pruned": fig_pruned`` joins
   the ``save_visual_bundle`` figures dict (6 -> 7 figures).
3. PATCH the §11 recap: the "De novo GA" bullet gains the prune
   sentence.
4. EXECUTE the generative notebook with nbclient (cwd=examples/).  The
   new cells only ADD output: no existing name is rebound (AST-audited:
   distances / shortlist / fig_pruned are bound by no other cell and
   appear nowhere else in the notebook), so existing pins cannot shift.
5. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
6. VERIFY banned fabrication tokens absent + pinned values present.

Pins (deterministic, probed under /tmp via the notebook's real config:
population 48, 15 generations, seed 42, the qed+logp+similarity blend):

  generative prune cell:
    audit  : ... 26 of 613 molecules non-dominated ...; pruned to the
             8 most isolated of 26 by crowding distance; strongest
             member 'COC1=Cc2cc(CNC(C)=O)[nH]c2C=CS1' dominates 386
             others  (the PRUNED strongest -- the 557-dominator sits in
             a crowded region and misses the cut: the teachable bit)
    corners: inf corners: 5 of 26
    table  : 8 rows with the crowding column; rank 4
             O=C(c1ccccc1)c1ccccc1 carries the top finite crowding
             1.546628
    figure : <Figure size 720x540 with 1 Axes> -- the crowding-encoded
             scatter (sizes = the PAIR's crowding, amber diamonds = the
             pair's corners; the same projection semantics cell 22
             taught -- the title says 8, the 2-objective projection)
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

GEN_NOTEBOOK = "complete_generative_workflow.ipynb"

PRUNE_MARKER = "crowding-distance shortlist"  # idempotency: the intro's title

PRUNE_MD_INTRO = """\
### Pruning the front — the crowding-distance shortlist

Twenty-six non-dominated molecules is already a lot to eyeball, and real
three-objective runs leave far more (a 300-molecule population over 30
generations measured **46 of 8,108**). Truncating by leaderboard rank
would drop the very corners the front exists to show. The NSGA-II
**crowding distance** ranks every member by its isolation in objective
space — the trade-off corners score `+inf`, crowded middles score low —
and `prune(k)` keeps the corners plus the most isolated interiors,
re-emitted in leaderboard order.
"""

PRUNE_CODE = '''\
# crowding: +inf marks the boundary corners (best/worst on an objective),
# finite values the isolation of the interiors (bigger = more isolated)
distances = front.crowding_distance()
print("inf corners:", sum(1 for d in distances if d == float("inf")),
      "of", len(front))

# keep the corners + the most isolated interiors -- never a rank head,
# which would drop the very corners the front exists to show
shortlist = front.prune(8)
print(shortlist)
print()
print(shortlist.to_dataframe().to_string(index=False))

# the trade-off scatter, crowding-encoded: interior marker size = the
# pair's crowding, the +inf corners are amber diamonds (the same
# projection semantics as above -- the title says 8, the 2-objective
# projection of the 26-member 3-objective front)
fig_pruned = plot_pareto_front(ga_denovo, x="qed", y="similarity",
                               title="Crowding-encoded front")
fig_pruned
'''

PRUNE_MD_VERDICT = """\
All five `+inf` corners survive — as do the three most isolated
interiors (crowding 1.55, 0.93, 0.73). Note what the cut teaches: the
front's strongest dominator (`CCOc1ccc(NC(C)=O)cc1`, 557 dominated
molecules) did **not** make the shortlist — it sits in a crowded region
of objective space. Dominating many molecules and being an isolated
trade-off anchor are different virtues, and the shortlist keeps the
second one. The summary line reports both numbers honestly: *26 of 613
non-dominated; pruned to the 8 most isolated of 26*. The selection is
deterministic and monotonic in `k` (a bigger keep extends, never
reshuffles); the CLI twin is `--pareto-keep N`.
"""

# --- cell replacements (old snippet -> new snippet) ------------------------

PUBLISH_OLD = (
    '    figures={\n'
    '        "corpus_landscape": fig_landscape,\n'
    '        "ga_convergence": fig_ga,\n'
    '        "prior_training": fig_prior,\n'
    '        "rl_finetuning": fig_rl,\n'
    '        "aos_trajectory": fig_aos,\n'
    '        "pareto_front": fig_pareto,\n'
    '    },'
)
PUBLISH_NEW = (
    '    figures={\n'
    '        "corpus_landscape": fig_landscape,\n'
    '        "ga_convergence": fig_ga,\n'
    '        "prior_training": fig_prior,\n'
    '        "rl_finetuning": fig_rl,\n'
    '        "aos_trajectory": fig_aos,\n'
    '        "pareto_front": fig_pareto,\n'
    '        "pareto_pruned": fig_pruned,\n'
    '    },'
)

RECAP_OLD = (
    "* **De novo GA** — 613 unique molecules in seconds, per-generation history,\n"
    "  free breakdown columns and the Pareto front of what the blend hides\n"
    "  — 26 of the 613 non-dominated, the deepest at leaderboard rank 197\n"
    "  — a one-call figure twin included (§4);"
)
RECAP_NEW = (
    "* **De novo GA** — 613 unique molecules in seconds, per-generation history,\n"
    "  free breakdown columns and the Pareto front of what the blend hides\n"
    "  — 26 of the 613 non-dominated, the deepest at leaderboard rank 197,\n"
    "  pruned to the 8 most isolated by crowding distance (the corners\n"
    "  always survive) — two figure twins included (§4);"
)

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "The front carries 26 of the 613 molecules",  # cell 23's opening
        [
            ("markdown", PRUNE_MD_INTRO),
            ("code", PRUNE_CODE),
            ("markdown", PRUNE_MD_VERDICT),
        ],
    ),
}

REPLACEMENTS = {
    GEN_NOTEBOOK: [
        (PUBLISH_OLD, PUBLISH_NEW),
        (RECAP_OLD, RECAP_NEW),
    ],
}

PINS_OUTPUT = {
    GEN_NOTEBOOK: [
        # -- the prune cell's printed audit + table rows (output-only) ----
        "inf corners: 5 of 26",
        "pruned to the 8 most isolated of 26 by crowding distance",
        # the PRUNED strongest member (NOT the 557-dominator -- that one
        # sits in a crowded region and misses the cut: the teachable bit)
        "'COC1=Cc2cc(CNC(C)=O)[nH]c2C=CS1' dominates 386 others",
        " rank                          smiles      qed  logp_band  similarity  dominates  crowding",
        # rank 4 carries the top finite crowding 1.546628
        "    4           O=C(c1ccccc1)c1ccccc1 0.652473   0.656393    0.357143        180  1.546628",
        "<Figure size 720x540 with 1 Axes>",  # the encoded scatter
        # -- 59's Pareto output pins (must survive) ------------------------
        (
            "Pareto front over 3 objectives [qed, logp_band, similarity]: "
            "26 of 613 molecules non-dominated"
        ),
        "'CCOc1ccc(NC(C)=O)cc1' dominates",  # the UNPRUNED audit's strongest
        "557 others",
        "    9                  CC(=O)c1ccccc1 0.517047   0.993880    0.379310        256",
        # -- 43's operator-trajectory output pins (must survive) ----------
        "mutation mix at generation  0: {'topology': 0.5, 'character': 0.5}",
        (
            "mutation mix at generation 15: {'topology': 0.8964223618089852, "
            "'character': 0.10357763819101498}"
        ),
        "<Figure size 840x380 with 2 Axes>",
        "topology    243        211              0.896",
        # -- classic leaderboard / corpus output pins (must survive) ------
        "corpus size : 244 SMILES",
        "12 seed molecules  <-  data/seed_molecules.csv",
        "aspirin scores 0.699",
        "CCOc1ccc(NC(C)=O)cc1 0.600755 0.770679   0.999046    0.219512",
    ],
}

PINS_SOURCE = {
    GEN_NOTEBOOK: [
        # -- the prune cell sources -----------------------------------------
        "distances = front.crowding_distance()",
        "shortlist = front.prune(8)",
        'fig_pruned = plot_pareto_front(ga_denovo, x="qed", y="similarity",',
        "keeps the corners plus the most isolated interiors",
        "did **not** make the shortlist",
        "the CLI twin is `--pareto-keep N`",
        # -- the publish + recap edits --------------------------------------
        '"pareto_pruned": fig_pruned',  # the 7th bundle figure
        "pruned to the 8 most isolated by crowding distance",  # recap bullet
        # -- 59's source pins (must survive untouched) ---------------------
        "from pharmaml.generative import pareto_front",
        "front = pareto_front(ga_denovo)",
        'fig_pareto = plot_pareto_front(ga_denovo, x="qed", y="similarity")',
        "The front carries 26 of the 613 molecules",
        "member sits at leaderboard rank",
        "its title says **8** of 613, not 26",
        "similarity 0.379",
        "`pareto_front` is a pure, deterministic function",
        '"pareto_front": fig_pareto',
        "the deepest at leaderboard rank 197",
        '"aos_trajectory": fig_aos',
        "the mutation mix that learned topology",
        "### Adaptive operators — the GA that learns",
        # -- classic source pins --------------------------------------------
        "0.5 x QED + 0.25 x LogP band [1,3] + 0.25 x similarity to aspirin",
        "ga (structural)",
        "## 11 · Recap",
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
        if any(PRUNE_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the prune cells")
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
    # figure cell commits a text/plain-only repr -- the inline PNGs the
    # TestNotebookArtifacts guard pins would silently vanish.  Unset it
    # for the kernel spawn so ipykernel picks the inline backend.
    os.environ.pop("MPLBACKEND", None)
    for name in (GEN_NOTEBOOK,):  # only the generative notebook this round
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
    print("driver 70 complete")


if __name__ == "__main__":
    main()
