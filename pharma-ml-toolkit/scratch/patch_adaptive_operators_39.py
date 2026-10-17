"""39 driver: the adaptive-operator-selection demo (notebook side).

Round 100x-39's notebook work — the 100x-38 capability demonstrated in
examples/complete_generative_workflow.ipynb:

1. INSERT three cells after the topology A/B interpretation markdown
   (anchor: ``topology walks there.`` -- the last cell of the §6b
   ring-size A/B section) and BEFORE ``### Segment swaps``:
   [markdown framing] + [code: the fresh uniform/adaptive operator
   pair on the ring objective (population 16, generations 16, seed 42,
   mutation="topology", the already-bound ring_objective /
   ring_parents / SEED / topo_ab) + the operator_stats tidy frame] +
   [markdown interpretation].
2. PATCH the section-11 recap: a sibling "Adaptive operators" bullet
   after the "Ring-size A/B" bullet.
3. EXECUTE the generative notebook with nbclient (cwd=examples/).  The
   new cells only ADD output: no existing name is rebound (audited:
   aos_ab / stats / operator_frame / the loop variables are bound by no
   other cell, and no cell after the insertion point references them),
   and every GeneticOptimizer owns its random.Random(seed) -- no shared
   rng stream exists, so existing pins cannot shift.
4. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
5. VERIFY banned fabrication tokens absent + pinned values present.

Pins (deterministic, probed twice byte-identical under
/tmp/task39n/probe_final{1,2}.txt, with the notebook's own pandas
display options replicated):

  generative adaptive-operator demo:
    topology (above)   : best 0.928 (82 unique) | final mean 0.802
    uniform (above)    : best 0.872 (97 unique) | final mean 0.638
    fixed ops (16/16)  : best 0.961 (155 unique) -- O=C1CCC(=O)NC(c2ccccc2)(c2ccccc2)C1
    adaptive (16/16)   : best 0.966 (147 unique) -- O.O=C1CCCNC(c2ccccc2)(c2ccccc2)C1
    adaptive final mean 0.808 vs the fixed control's 0.765, ahead 13/16 gens,
    best in generation 11 vs the control's final generation
    operator_stats: level1 crossover 40 draws / 6 survivors / p 0.107,
    clone 246 / 210 / 0.893; geometry splice 40 / 6 / 1.000 (the splice
    mode's single geometry, bookkeeping only); mutation topology
    243 / 211 / 0.896 vs character 43 / 4 / 0.104 -- the GA re-derives
    the §6b A/B verdict in-run
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

AOS_MD_MARKER = "### Adaptive operators"  # idempotency: section heading

AOS_MD = """\
### Adaptive operators — the GA that learns

The A/B above needed a human to read the tea leaves and pick the
winning operator family; `operator_selection="adaptive"` has the
optimiser make that call itself, in-run — the GA learns which
operators pay off. Three levels of operator choice (crossover vs
clone, the recombination geometry, the mutation family) draw through
credit-assigned softmaxes: each drawn operator earns
`(valid offspring ? score : 0) − mid-parent` on a sliding 20-attempt
window — failed operators are charged, not forgiven — and the next
draw is a softmax over that windowed credit with an exploration floor,
so a failing operator is demoted, never abandoned; one rng draw per
selection keeps an adaptive run exactly as deterministic as a fixed
one. The pair below re-runs the ring objective with the same seed,
objective and corpus parents as the A/B above — topology mutation in
both arms, only the operator selection differs — on a slightly longer
budget (16 × 16 rather than 12 × 12) so the credit window has runway
to learn. And the learning is not a black box: the adaptive result's
`metadata["operator_stats"]` carries, per level and operator, the
draws, the surviving offspring and the final selection probability.
"""

AOS_CODE = '''\
# the adaptive A/B: same objective, seed and corpus parents as above
# (topology mutation in both arms) -- only the operator selection
# differs; the credit needs a slightly longer run to learn
aos_ab = {}
for sel in ("uniform", "adaptive"):
    aos_ab[sel] = GeneticOptimizer(
        ring_objective, population_size=16, generations=16, seed=SEED,
        mutation="topology", operator_selection=sel,
    ).optimize(corpus=ring_parents)

# best / final population mean: the two arms above, then the fresh pair
for label, run in [
    ("topology (above)", topo_ab["topology"]),
    ("uniform (above)", topo_ab["uniform"]),
    ("fixed ops (16/16)", aos_ab["uniform"]),
    ("adaptive (16/16)", aos_ab["adaptive"]),
]:
    print(
        f"{label:19s}: best {run.best_score:.3f} ({len(run)} unique) "
        f"-- {run.best} | final mean {run.history[-1]['mean_score']:.3f}"
    )

# the introspection artifact: what the GA learned about its own operators
stats = aos_ab["adaptive"].metadata["operator_stats"]
operator_frame = pd.DataFrame(
    [
        {
            "level": level,
            "operator": op,
            "draws": entry["draws"],
            "successes": entry["successes"],
            "final_probability": round(entry["final_probability"], 3),
        }
        for level, table in stats.items()
        for op, entry in table.items()
    ]
)
operator_frame
'''

AOS_MD2 = """\
The GA re-derived §6b's verdict inside a single run. The mutation
family's row is that conclusion in miniature: 243 topology draws kept
211 offspring (87 % survival — a topology move is a pre-sanitised
molecule one ring step from its parent) against 43 character draws
that kept 4, and the final selection probability settles at 0.896 for
topology vs 0.104 for character. Level 1 learned the same lesson about
the character splice: cloning survived 210 of 246 draws (p 0.893)
where the splice crossover kept 6 of 40 — so the run spends its
attempts on clone-then-topology-mutate, exactly the §6b-winning
recipe, while the geometry row is bookkeeping only (the splice
crossover has a single geometry, p 1.000 by construction). The
exploitation shows at the scoreboard: the adaptive arm closes at
population mean 0.808 vs 0.765 for the fixed-operator control, ahead
in 13 of the 16 generations, with best 0.966 in generation 11 against
the control's 0.961, which arrives only in the final generation. Both
winners are on-target 7-membered lactams; the adaptive arm's
`O.O=C1CCCNC(c2ccccc2)(c2ccccc2)C1` carries a hitchhiking water
molecule — QED quietly forgives the extra fragment, the objective's
letter over its spirit again.
"""

# --- markdown cell replacement (old snippet -> new snippet) ----------------

RECAP_OLD = (
    "* **Ring-size A/B** — topology vs uniform mutation, same seed and\n"
    "  parents, only the operator differs: best 0.928 in generation 2 vs\n"
    "  0.872 in generation 7, population mean ahead in every generation,\n"
    "  ~4× the offspring survival (§6b);"
)
RECAP_NEW = (
    RECAP_OLD
    + "\n* **Adaptive operators** — the GA re-derives that verdict in-run:\n"
    '  `operator_selection="adaptive"` on the same ring objective pushes the\n'
    "  topology family to final probability 0.896 (211 of 243 draws kept)\n"
    "  and closes at population mean 0.808 vs the fixed-operator control's\n"
    "  0.765 (§6b);"
)

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "topology walks there.",  # end of the topology A/B interpretation
        [
            ("markdown", AOS_MD),
            ("code", AOS_CODE),
            ("markdown", AOS_MD2),
        ],
    ),
}

REPLACEMENTS = {
    GEN_NOTEBOOK: [
        (RECAP_OLD, RECAP_NEW),
    ],
}

PINS = {
    GEN_NOTEBOOK: [
        # -- the adaptive-operator demo (probed twice, byte-identical) ------
        "### Adaptive operators — the GA that learns",  # section heading
        "topology (above)   : best 0.928 (82 unique)",  # the 6b arm re-printed
        "uniform (above)    : best 0.872 (97 unique)",  # the 6b arm re-printed
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
        if any(AOS_MD_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the adaptive-operators cells")
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


def replace_markdown() -> None:
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
        print(f"{name}: {changed} markdown replacement(s) applied")


def execute_notebooks() -> None:
    import nbformat
    from nbclient import NotebookClient

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
        print(f"verified {name}: guard clean, {len(pins)} pin(s) present")


def main() -> None:
    insert_cells()
    replace_markdown()
    execute_notebooks()
    regenerate_twin()
    verify()
    print("DONE: adaptive-operator demo documented, executed and pinned")


if __name__ == "__main__":
    main()
