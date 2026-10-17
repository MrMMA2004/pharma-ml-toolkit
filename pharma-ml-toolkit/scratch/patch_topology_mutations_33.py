"""33 driver: ring-topology mutations (notebook side).

Round 100x-33's flagship: ``ring_topology_moves`` /
``ring_topology_mutate`` -- the single-parent medchem moves (ring
contraction / expansion, the ring-size axis of scaffold morphing) and
the GA's ``mutation="topology"`` mode.  This driver lands the
notebook documentation:

1. INSERT a topology-mutation demo cell pair after the ring-insert
   demo (file-driven: both demo parents looked up from the bundled
   corpus by SMILES value, no inline fabrication).
2. PATCH the generative notebook's recap: a scaffold-hopping bullet
   (the five-geometry crossover + the ring-size mutations -- §6b was
   never recapped before) and the CLI hint gains ``--topology``.
3. PATCH the walkthrough's §8 mode list: the third structural mode
   bullet (``mutation="topology"``).
4. EXECUTE both notebooks with nbclient (cwd=examples/, worker-cap
   env) -- the new cells add output only; existing runs are untouched
   (no operator changes, same seeds).
5. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
6. VERIFY the banned fabrication tokens are absent and the pinned
   spot-check values appear in the executed outputs.

Pins (deterministic, independent of GA trajectories):

  generative 6e topology demo:
    piperidine  moves -> ['C1CCCC1', 'C1CCCNCC1', 'C1CCNC1']
    morpholine  moves -> ['C1CCNC1', 'C1CCOC1', 'C1CNCCOC1', 'C1COCN1']
    morpholine x Random(0) -> 'C1COCN1'
  earlier demo pins are unaffected (no operator changes this round).
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

from test_notebook_variants import BANNED_FABRICATION_TOKENS  # noqa: E402

WORKER_ENV = {
    **os.environ,
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

GEN_NOTEBOOK = "complete_generative_workflow.ipynb"
WALK_NOTEBOOK = "pharmaml_walkthrough.ipynb"

TOPO_MD_MARKER = "### Topology mutations"

TOPO_MD = """\
### Topology mutations — the single-parent ring-size moves

The crossover geometries recombine *two* parents;
`ring_topology_moves` is the single-parent counterpart — the
ring-size axis of scaffold morphing as a **mutation** (SAR 101): one
atom removed from a saturated ring with its neighbours rejoined
(**contraction** — piperidine → pyrrolidine, morpholine → pyrrolidine
/ tetrahydrofuran), or one methylene inserted into a saturated single
ring bond (**expansion** — piperidine → azepane). The same
conservative fences apply: aromatic and unsaturated rings are never
touched, fused/spiro atoms and shared bonds stay put, and the
ring-size window is 5–8. In the GA, `mutation="topology"` draws one
move per offspring (acyclic parents fall back to character mutation);
paired with the scaffold-hop crossover it completes the structural
move set — recombination *plus* ring-size mutation. Both demo parents
are looked up from the bundled corpus again — nothing invented
inline.
"""

TOPO_CODE = '''\
# ring-size mutations -- the single-parent medchem moves, corpus parents
from pharmaml.generative import ring_topology_moves, ring_topology_mutate

piperidine = DEFAULT_CORPUS[DEFAULT_CORPUS.index("C1CCNCC1")]
morpholine = DEFAULT_CORPUS[DEFAULT_CORPUS.index("C1COCCN1")]
print(f"piperidine : {piperidine}")
print(f"  moves    : {ring_topology_moves(piperidine)}")
print(f"morpholine : {morpholine}")
print(f"  moves    : {ring_topology_moves(morpholine)}")

# the GA surface: one move drawn with the caller's rng (None -> fallback)
draw = random.Random(0)
print(f"\\nmorpholine x Random(0) -> {ring_topology_mutate(morpholine, draw)}")

# in the GA: ring-size moves + scaffold-hop recombination together
ga_top = GeneticOptimizer(
    objective, population_size=48, generations=15, seed=SEED,
    mutation="topology", crossover="scaffold_hop",
).optimize(seed_smiles=seed_smiles)
print(f"topology mode: {len(ga_top)} unique, best {ga_top.best_score:.3f}")
ga_top.to_dataframe().head(5)[["smiles", "score", "rank"]]
'''

# --- markdown cell replacements (old snippet -> new snippet) ------------

GEN_RECAP_OLD = """\
* **Lead optimisation** — scaffold-preserving mutation kept 100 % of the
  run on seed frameworks vs 50.5 % for uniform mutation (§6);
"""
GEN_RECAP_NEW = """\
* **Lead optimisation** — scaffold-preserving mutation kept 100 % of the
  run on seed frameworks vs 50.5 % for uniform mutation (§6);
* **Scaffold hopping** — the five-geometry BREED crossover (bridge,
  merge, transplant, fuse, insert) plus the single-parent ring-size
  topology mutations (§6b);
"""

GEN_CLI_OLD = "(+ `--scaffold`, `--scaffold-hop`, `--seeds`);"
GEN_CLI_NEW = "(+ `--scaffold`, `--scaffold-hop`, `--topology`, `--seeds`);"

WALK_8_OLD = """\
  spacer — so the offspring's framework
  differs from *both* parents: the scaffold hop.
"""
WALK_8_NEW = """\
  spacer — so the offspring's framework
  differs from *both* parents: the scaffold hop;
* `mutation="topology"` — ring-size mutation, the single-parent moves:
  one atom removed from a saturated ring (contraction — piperidine →
  pyrrolidine) or one methylene inserted into a saturated single ring
  bond (expansion — piperidine → azepane), the ring-size axis of
  scaffold morphing.
"""

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "the chain re-joined through the aryl spacer",  # end of the insert demo
        [("markdown", TOPO_MD), ("code", TOPO_CODE)],
    ),
}

REPLACEMENTS = {
    GEN_NOTEBOOK: [
        (GEN_RECAP_OLD, GEN_RECAP_NEW),
        (GEN_CLI_OLD, GEN_CLI_NEW),
    ],
    WALK_NOTEBOOK: [(WALK_8_OLD, WALK_8_NEW)],
}

PINS = {
    GEN_NOTEBOOK: [
        "['C1CCCC1', 'C1CCCNCC1', 'C1CCNC1']",  # piperidine move set
        "['C1CCNC1', 'C1CCOC1', 'C1CNCCOC1', 'C1COCN1']",  # morpholine
        "morpholine x Random(0) -> C1COCN1",
        "topology mode:",  # the GA run summary
        "five-geometry BREED crossover",  # the new recap bullet
        "`--topology`",  # the CLI hint
        # 31/32 regression pins (must survive the re-execution untouched)
        "NCc1ccc(C(=O)O)cc1",  # insert demo: ibuprofen benzene into glycine
        "CCOc1ccccc1C(C)=O",  # insert demo: aspirin benzene into ethyl acetate
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # bridge demo
        "ga (structural)",  # 32's head-to-head fourth arm
    ],
    WALK_NOTEBOOK: [
        "ring-size mutation",  # the new mode bullet
        "spliced into a cut acyclic bond",  # 31's bullet stays (line-wrapped)
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
        if any(TOPO_MD_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the topology-mutation cells")
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

    for name in (GEN_NOTEBOOK, WALK_NOTEBOOK):
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
        env=WORKER_ENV,
    )
    if result.returncode != 0:
        _fail(f"strip_notebook_outputs.py failed: {result.stderr[-400:]}")
    print("twins regenerated (5 notebooks, idempotent)")


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
    print("DONE: ring-topology mutations documented, executed and pinned")


if __name__ == "__main__":
    main()
