"""43 driver: the operator-trajectory figure (notebook side, Phase 2).

Round 100x-43's notebook work -- Phase 1's ``plot_operator_trajectory``
(100x-42-A: ``metadata["operator_trajectory"]`` collection +
``viz/generative_plots.py``) demonstrated in
examples/complete_generative_workflow.ipynb:

1. INSERT two cells after the adaptive-operators interpretation
   markdown (anchor: ``letter over its spirit again.`` -- the tail of
   cell 45, the last cell of the "### Adaptive operators" section) and
   BEFORE ``### Segment swaps``: [code: the learning-curve figure --
   ``fig_aos = plot_operator_trajectory(aos_ab["adaptive"], ...)`` +
   the first/last mutation-mix records printed] + [markdown
   interpretation: the learning-curve story, every number probed].
2. PATCH the §10 publish cell: ``"aos_trajectory": fig_aos`` joins the
   ``save_visual_bundle`` figures dict (4 -> 5 figures).
3. PATCH the §11 recap: the "Adaptive operators" bullet gains one
   sentence naming ``plot_operator_trajectory``.
4. EXECUTE the generative notebook with nbclient (cwd=examples/).  The
   new cells only ADD output: no existing name is rebound (AST-audited:
   fig_aos / traj / plot_operator_trajectory are bound by no other
   cell, no cell after the insertion point references them, and
   aos_ab is bound only by the cell directly above and never rebound),
   so existing pins cannot shift.
5. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
6. VERIFY banned fabrication tokens absent + pinned values present.

Pins (deterministic, probed 4x byte-identical under
/tmp/task43n/probe_out{1,2,3,4}.txt -- the new cell source run
byte-verbatim after the notebook's real prelude cells 0-44; the
figure repr + narrative numbers from probe_repr1.txt /
probe_detail{1,2}.txt):

  generative operator-trajectory cell:
    mutation mix at generation  0: {'topology': 0.5, 'character': 0.5}
    mutation mix at generation 15: {'topology': 0.8964223618089852,
                                    'character': 0.10357763819101498}
    the 0.5 -> 0.896 learning beat: topology p 0.714 by generation 1,
    0.865 by generation 2 (mean score doubling 0.255 -> 0.509 in step),
    a 0.85-0.90 plateau to the 0.8964 close; level1 clone 0.5 -> 0.8932;
    geometry constant 1.0 (the splice mode's single geometry); the
    closing record matches the operator_stats final_probability values
    to <= 2.3e-05 (the documented 4-decimal rounding gap)
    figure: <Figure size 840x380 with 2 Axes> (score + mutation panel)
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

TRAJ_MARKER = "operator trajectory"  # idempotency: the interpretation's opening

TRAJ_CODE = '''\
# the learning curve of the adaptive run above -- score panel + mutation mix
from pharmaml.viz import plot_operator_trajectory

traj = aos_ab["adaptive"].metadata["operator_trajectory"]
print("mutation mix at generation  0:", traj[0]["mutation"])
print("mutation mix at generation 15:", traj[-1]["mutation"])

fig_aos = plot_operator_trajectory(
    aos_ab["adaptive"],
    title="Adaptive operators — the mutation mix that learned topology",
)
fig_aos
'''

TRAJ_MD = """\
The operator trajectory is the learning itself, on one pair of shared
axes. The run starts exactly fair — generation 0's printed record is
`{'topology': 0.5, 'character': 0.5}`, no operator favoured — and a
single generation of windowed credit is enough to flip the balance:
topology's share is already 0.714 at generation 1 and 0.865 at
generation 2, exactly as the population mean doubles from 0.255 to
0.509. From there the mix holds a 0.85–0.90 plateau while the mean
climbs on to its 0.808 close — the two panels move together because
the mix shift *buys* the mean climb — and the best score's 0.966
lands in generation 11, inside the learned regime rather than by luck
at the close. The closing record, `{'topology': 0.8964223618089852,
'character': 0.10357763819101498}`, is the full-precision twin of the
tidy table above: its 0.896 / 0.104 final-probability row differs from
the record only by rounding (the stats carry four decimals, the frame
three). The default figure shows the mutation level — the flagship
story; pass all three `levels` to add level 1's parallel takeover
(clone 0.5 → 0.893, the same lesson) and the degenerate geometry line
at p 1.000.
"""

# --- cell replacements (old snippet -> new snippet) ------------------------

PUBLISH_OLD = (
    '    figures={\n'
    '        "corpus_landscape": fig_landscape,\n'
    '        "ga_convergence": fig_ga,\n'
    '        "prior_training": fig_prior,\n'
    '        "rl_finetuning": fig_rl,\n'
    '    },'
)
PUBLISH_NEW = (
    '    figures={\n'
    '        "corpus_landscape": fig_landscape,\n'
    '        "ga_convergence": fig_ga,\n'
    '        "prior_training": fig_prior,\n'
    '        "rl_finetuning": fig_rl,\n'
    '        "aos_trajectory": fig_aos,\n'
    '    },'
)

RECAP_OLD = (
    "* **Adaptive operators** — the GA re-derives that verdict in-run:\n"
    '  `operator_selection="adaptive"` on the same ring objective pushes the\n'
    "  topology family to final probability 0.896 (211 of 243 draws kept)\n"
    "  and closes at population mean 0.808 vs the fixed-operator control's\n"
    "  0.765 (§6b);"
)
RECAP_NEW = (
    RECAP_OLD[:- len(" (§6b);")]
    + " — and its learning curve is now a one-call figure,\n"
    "  `plot_operator_trajectory` (§6b);"
)

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "letter over its spirit again.",  # tail of cell 45's interpretation
        [
            ("code", TRAJ_CODE),
            ("markdown", TRAJ_MD),
        ],
    ),
}

REPLACEMENTS = {
    GEN_NOTEBOOK: [
        (PUBLISH_OLD, PUBLISH_NEW),
        (RECAP_OLD, RECAP_NEW),
    ],
}

PINS = {
    GEN_NOTEBOOK: [
        # -- the operator-trajectory figure cell (probed 4x byte-identical) --
        "from pharmaml.viz import plot_operator_trajectory",
        "mutation mix at generation  0: {'topology': 0.5, 'character': 0.5}",
        (
            "mutation mix at generation 15: {'topology': 0.8964223618089852, "
            "'character': 0.10357763819101498}"
        ),
        "<Figure size 840x380 with 2 Axes>",  # the rendered learning curve
        "the mutation mix that learned topology",  # the figure title (source)
        # -- the interpretation markdown (narrative pins, probed) -----------
        "The operator trajectory is the learning itself",
        "already 0.714 at generation 1 and 0.865 at",
        "population mean doubles from 0.255 to",
        # -- the publish + recap edits ----------------------------------------
        '"aos_trajectory": fig_aos',  # the 5th bundle figure
        "`plot_operator_trajectory` (§6b);",  # the recap bullet extension
        # -- 39-N's adaptive-operator demo pins (must survive untouched) ----
        "### Adaptive operators — the GA that learns",
        "topology (above)   : best 0.928 (82 unique)",
        "uniform (above)    : best 0.872 (97 unique)",
        "fixed ops (16/16)  : best 0.961 (155 unique)",
        "adaptive (16/16)   : best 0.966 (147 unique)",
        "O.O=C1CCCNC(c2ccccc2)(c2ccccc2)C1",  # the adaptive best: 7-ring + water
        "O=C1CCC(=O)NC(c2ccccc2)(c2ccccc2)C1",  # the fixed-operator best
        "| final mean 0.808",  # adaptive closes ahead of the fixed control
        "| final mean 0.765",
        "crossover     40          6              0.107",  # level1 row
        "clone    246        210              0.893",  # level1 row
        "splice     40          6              1.000",  # geometry row (degenerate)
        "topology    243        211              0.896",  # the discovered family
        "character     43          4              0.104",  # the demoted fallback
        "Adaptive operators** — the GA re-derives",  # the recap bullet
        # -- 37's ring-size A/B demo + census pins (must survive untouched) --
        "30 corpus molecules carry a movable saturated ring",
        "topology : best 0.928 (82 unique) -- NCC1CCNC(c2ccccc2)(c2ccccc2)C(=O)N1",
        "uniform  : best 0.872 (97 unique) -- C=S1C=NCCC2CCCNC(=O)C2=C2CC1CCON2",
        "topology  top-12 ring census: {6: 5, 7: 7}",
        "uniform   top-12 ring census: {5: 3, 6: 5, 7: 12, 8: 3}",
        "CCC1(c2ccccc2)CCNC(=O)NC1=O",  # primidone's 7-ring expansion
        "0.828",  # its total score in the objective probe table
        "7,328 candidates over 295 distinct",
        "1,438 conservative C\u2192C",
        "N\u2194C 368 each way",
        "Ring-size A/B",  # the recap bullet
        # -- 31/32/33/35 regression pins (must survive untouched) -----------
        "NCc1ccc(C(=O)O)cc1",  # insert demo: ibuprofen benzene into glycine
        "CCOc1ccccc1C(C)=O",  # insert demo: aspirin benzene into ethyl acetate
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # bridge demo
        "['C1CCCC1', 'C1CCCNCC1', 'C1CCNC1']",  # topology demo: piperidine
        "morpholine x Random(0) -> C1COCN1",  # topology demo draw
        "seed 0 ->  CC(=O)Nc1ccccc1C(=O)O",  # swap demo: salicylamide
        "seed 5 ->  CC(=O)Oc1ccc(O)cc1",  # swap demo: the O-for-N twin
        "ga (structural)",  # 32's head-to-head fourth arm
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
        if any(TRAJ_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the operator-trajectory cells")
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


def verify() -> None:
    import nbformat

    for name, pins in PINS.items():
        nb = nbformat.read(EXAMPLES / name, as_version=4)
        chunks: list[str] = []
        for cell in nb.cells:
            chunks.append("".join(cell.source))
            for out in cell.get("outputs", []):
                if isinstance(out.get("text"), str):
                    chunks.append(out["text"])
                plain = out.get("data", {}).get("text/plain")
                if isinstance(plain, str):
                    chunks.append(plain)
        text = "\n".join(chunks)
        for token in BANNED_FABRICATION_TOKENS:
            if token in text:
                _fail(f"{name}: banned fabrication token {token!r} present")
        for pin in pins:
            if pin not in text:
                _fail(f"{name}: pin {pin!r} missing from executed output")
        # The inline figures must stay committed: 4 figure cells carried 7
        # png outputs before this round; the learning curve adds 2 more
        # (display_data + execute_result, matching the other figure cells),
        # making 9.
        pngs = sum(
            1
            for cell in nb.cells
            if cell.cell_type == "code"
            for out in cell.get("outputs", [])
            if "image/png" in out.get("data", {})
        )
        if pngs < 9:
            _fail(f"{name}: only {pngs} inline figure png(s) committed (need >= 9)")
        print(
            f"verified {name}: guard clean, {len(pins)} pin(s) present, "
            f"{pngs} inline figure png(s)"
        )


def main() -> None:
    insert_cells()
    replace_cells()
    execute_notebooks()
    regenerate_twin()
    verify()
    print("DONE: operator-trajectory figure documented, executed and pinned")


if __name__ == "__main__":
    main()
