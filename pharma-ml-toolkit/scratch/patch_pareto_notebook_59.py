"""59 driver: the Pareto front cells (notebook side, the axis's round 4).

Round 100x-59's notebook work -- the library (100x-56: ``pareto_front`` +
``plot_pareto_front``), the CLI (100x-57: ``--pareto`` / ``--pareto-plot``)
and the playground card (100x-58) now reach the executed notebook:

1. INSERT three cells after the de novo leaderboard / convergence cell
   (anchor: ``title="De novo GA — 15 generations, 48 molecules"`` -- the
   tail of cell 20, §4) and BEFORE ``## 5 · Seeded generation``:
   [markdown intro: the dominance definition] + [code: the front audit +
   the 26-row table + the qed x similarity scatter] + [markdown verdict:
   the probed trade-off story].
2. PATCH the §10 publish cell: ``"pareto_front": fig_pareto`` joins the
   ``save_visual_bundle`` figures dict (5 -> 6 figures).
3. PATCH the §11 recap: the "De novo GA" bullet gains the front sentence.
4. EXECUTE the generative notebook with nbclient (cwd=examples/).  The
   new cells only ADD output: no existing name is rebound (AST-audited:
   front / pareto_front / plot_pareto_front / fig_pareto are bound by no
   other cell and mentioned nowhere else in the notebook), so existing
   pins cannot shift.
5. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
6. VERIFY banned fabrication tokens absent + pinned values present.

Pins (deterministic, probed twice byte-identical after the notebook's
real prelude cells 0-20 under /tmp/task59/probe_pareto_cells.py):

  generative Pareto-front cell:
    audit  : Pareto front over 3 objectives [qed, logp_band, similarity]:
             26 of 613 molecules non-dominated (0 excluded for non-finite
             values); strongest member 'CCOc1ccc(NC(C)=O)cc1' dominates
             557 others
    table  : 26 rows, columns rank/smiles/qed/logp_band/similarity/
             dominates; rank 9 = CC(=O)c1ccccc1 (acetophenone) carries
             the front's top similarity 0.379310
    figure : <Figure size 720x540 with 1 Axes> -- the qed x similarity
             2-objective PROJECTION front (8 members -- the title's
             "8 of 613", vs the 3-objective front's 26; the documented
             projection semantics), legend ['dominated', 'Pareto front (8)']
    verdict: deepest front member at leaderboard rank 197 of 613
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

PARETO_MARKER = "what the blend hides"  # idempotency: the intro's opening

PARETO_MD_INTRO = """\
### The Pareto front — what the blend hides

The leaderboard ranks by ONE number — the 0.5×QED + 0.25×LogP +
0.25×similarity blend. `pareto_front` re-reads the same run's
per-objective breakdown and extracts the molecules **no other candidate
dominates** — *a* dominates *b* iff *a* ≥ *b* on every objective and
strictly greater on at least one. The front is where the real trade-offs
live: a molecule can sit deep in the blended ranking and still be
unbeatable without paying for it on some objective.
"""

PARETO_CODE = '''\
# the Pareto front of the de novo run -- what the blended ranking hides
from pharmaml.generative import pareto_front
from pharmaml.viz import plot_pareto_front

front = pareto_front(ga_denovo)
print(front)
print()
print(front.to_dataframe().to_string(index=False))

# the trade-off scatter: dominated molecules muted, the front connected.
# x/y pick a 2-objective projection of the 3-objective front above.
fig_pareto = plot_pareto_front(ga_denovo, x="qed", y="similarity")
fig_pareto
'''

PARETO_MD_VERDICT = """\
The front carries 26 of the 613 molecules. Its strongest member,
`CCOc1ccc(NC(C)=O)cc1` (dominating 557 others), is the leaderboard's
rank 1 — as it must be: anything dominating the blend's winner would
have outranked it. The story is the other end: the **deepest** front
member sits at leaderboard rank **197 of 613** — a molecule the blended
ranking buries, yet which cannot be improved on any objective without
sacrificing another. The scatter (qed × similarity) shows the shape of
the trade-off; note its title says **8** of 613, not 26 — plotting two
objectives projects the 3-objective front onto that pair (the same
semantics the CLI's `--pareto-plot` uses). The projection's trade-off
curve bends from the most aspirin-like survivor — rank 9,
`CC(=O)c1ccccc1`, similarity 0.379 — toward high-QED strangers the
corpus never saw. `pareto_front` is a pure, deterministic function of
any `MultiObjective` result; the CLI prints this audit with `--pareto`
and saves the scatter with `--pareto-plot`.
"""

# --- cell replacements (old snippet -> new snippet) ------------------------

PUBLISH_OLD = (
    '    figures={\n'
    '        "corpus_landscape": fig_landscape,\n'
    '        "ga_convergence": fig_ga,\n'
    '        "prior_training": fig_prior,\n'
    '        "rl_finetuning": fig_rl,\n'
    '        "aos_trajectory": fig_aos,\n'
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
    '    },'
)

RECAP_OLD = (
    "* **De novo GA** — 613 unique molecules in seconds, per-generation history\n"
    "  and free breakdown columns (§4);"
)
RECAP_NEW = (
    "* **De novo GA** — 613 unique molecules in seconds, per-generation history,\n"
    "  free breakdown columns and the Pareto front of what the blend hides\n"
    "  — 26 of the 613 non-dominated, the deepest at leaderboard rank 197\n"
    "  — a one-call figure twin included (§4);"
)

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        'title="De novo GA — 15 generations, 48 molecules"',  # cell 20's tail
        [
            ("markdown", PARETO_MD_INTRO),
            ("code", PARETO_CODE),
            ("markdown", PARETO_MD_VERDICT),
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
        # -- printed audit + table rows + figure reprs (output-only) ----
        (
            "Pareto front over 3 objectives [qed, logp_band, similarity]: "
            "26 of 613 molecules non-dominated"
        ),
        "'CCOc1ccc(NC(C)=O)cc1' dominates",  # the audit's strongest member
        "557 others",  # ... and its domination count
        "<Figure size 720x540 with 1 Axes>",  # the rendered scatter
        " rank                          smiles      qed  logp_band  similarity  dominates",
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
        # -- the Pareto-front cell sources ---------------------------------
        "from pharmaml.generative import pareto_front",
        "front = pareto_front(ga_denovo)",
        'fig_pareto = plot_pareto_front(ga_denovo, x="qed", y="similarity")',
        "The front carries 26 of the 613 molecules",
        "member sits at leaderboard rank",
        "its title says **8** of 613, not 26",
        "similarity 0.379",
        "`pareto_front` is a pure, deterministic function",
        # -- the publish + recap edits --------------------------------------
        '"pareto_front": fig_pareto',  # the 6th bundle figure
        "the deepest at leaderboard rank 197",  # the recap bullet extension
        # -- 43's source pins (must survive untouched) ---------------------
        '"aos_trajectory": fig_aos',
        "the mutation mix that learned topology",  # the figure title (source)
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
        if any(PARETO_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the Pareto-front cells")
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
    print("driver 59 complete")


if __name__ == "__main__":
    main()
