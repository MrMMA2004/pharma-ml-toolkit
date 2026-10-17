"""28 driver: ring-bond cuts for the scaffold-hop crossover (notebook side).

Round 100x-28's flagship: ``scaffold_hop_crossover`` grows two new cut
geometries -- the **ring merge** (two opened rings zipped into one
combined ring) and the **linker transplant** (a two-ended linker atom
excised from one parent and zipped into a cut ring bond of the other:
ring homologation).  This driver lands the notebook documentation:

1. PATCH the generative notebook's 6b markdown (three geometries) and
   INSERT a ring-topology demo cell pair (file-driven: the demo parents
   are looked up from the bundled corpus by SMILES value, no inline
   fabrication).
2. PATCH the walkthrough's 6b/8 bullet to mention ring recombination.
3. EXECUTE both notebooks with nbclient (cwd=examples/, worker-cap env)
   -- the 6b structural GA run legitimately changes: the corpus carries
   piperidine/morpholine/benzodiazepinone, so the GA can now reach the
   new geometries.
4. TWINS -- regenerate the *_no_output.ipynb variants via
   scratch/strip_notebook_outputs.py (idempotent).
5. VERIFY the banned fabrication tokens are absent and the pinned
   spot-check values appear in the executed outputs.

Pins (deterministic, independent of GA trajectories):

  generative 6b ring demo: morpholine x piperidine -> 'C1CCNCCOCCNCC1'
    (one 12-ring); benzodiazepinone x aspirin -> 'CC1=NCC(=O)N(C)c2ccccc2O1'
    (the 7-ring expanded to an 8-ring by aspirin's ester oxygen)
  generative 6b bridge demo (unchanged, aromatic parents):
    aspirin x naproxen -> 'COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1'
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

RING_MD_MARKER = "### Ring topology"

RING_MD = """\
### Ring topology — the ring-bond cuts

Bridge cuts recombine fragment *arrangement*; ring topology needs its own
moves, and the operator has two (single ring bonds only — an aromatic
ring bond would make the re-zip invent bond orders, so those are never
cut):

* **merge** — both parents are cut at homologous single ring bonds and the
  two opened rings zip into *one* combined ring holding atoms from both
  (morpholine × piperidine below → a single 12-ring carrying both
  nitrogens and the oxygen);
* **transplant** — a two-ended linker atom (acyclic, two single bonds,
  homologous flanks) is excised from one parent and zipped into a cut
  ring bond of the other: ring homologation — the corpus
  benzodiazepinone's 7-ring below becomes an 8-ring oxazepinone with
  aspirin's ester oxygen.

The parents are looked up from the bundled corpus by SMILES value —
nothing is invented inline.
"""

RING_CODE = '''\
# ring topology recombination -- both geometries, corpus molecules only
def ring_sizes(smiles):
    """Ring-size profile (small ring first) for the demo prints."""
    mol = rdkit.Chem.MolFromSmiles(smiles)
    return sorted(len(r) for r in mol.GetRingInfo().AtomRings())

draw = random.Random(0)

# merge: two opened rings zip into ONE combined ring (atoms from both)
morpholine = DEFAULT_CORPUS[DEFAULT_CORPUS.index("C1COCCN1")]
piperidine = DEFAULT_CORPUS[DEFAULT_CORPUS.index("C1CCNCC1")]
merged = scaffold_hop_crossover(morpholine, piperidine, draw)
print(f"morpholine : {morpholine}   rings={ring_sizes(morpholine)}")
print(f"piperidine : {piperidine}   rings={ring_sizes(piperidine)}")
print(f"merged     : {merged}   rings={ring_sizes(merged)}  <- one 12-ring\\n")

# transplant: a linker atom is excised from one parent and zipped into a
# cut ring bond of the other -- ring homologation
benzodiazepinone = DEFAULT_CORPUS[DEFAULT_CORPUS.index("CC1=NCC(=O)N(C)c2ccccc21")]
aspirin = DEFAULT_CORPUS[DEFAULT_CORPUS.index("CC(=O)Oc1ccccc1C(=O)O")]
transplanted = scaffold_hop_crossover(benzodiazepinone, aspirin, draw)
print(f"ring parent  : {benzodiazepinone}   rings={ring_sizes(benzodiazepinone)}")
print(f"linker donor : {aspirin}   (the ester O is a two-ended linker)")
print(
    f"transplanted : {transplanted}   rings={ring_sizes(transplanted)}"
    "  <- the 7-ring expanded to an 8-ring"
)
'''

# --- markdown cell replacements (old snippet -> new snippet) ------------

GEN_6B_OLD = """\
both parents are cut at a pair of *homologous* single acyclic bonds — the
join atoms match on element, degree, aromaticity and formal charge — and
the fragments are recombined cross-wise with `Chem.molzip`. Every ring
system arrives intact, so an offspring is a true hybrid: whenever the
parents' frameworks span different fragments, the child's Murcko scaffold
differs from *both* parents (Nicolaou et al.'s BREED tradition). The
operator also works standalone — `scaffold_hop_crossover(parent_a,
parent_b, rng)` — for surgical use outside the GA.
"""

GEN_6B_NEW = """\
both parents are cut at *homologous* bonds — the join atoms match on
element, degree, aromaticity and formal charge — and recombined with
`Chem.molzip`. Three geometries compete for the draw: **bridge cuts**
(a pair of single acyclic bonds, fragments swapped cross-wise — every
ring system arrives intact, and whenever the parents' frameworks span
different fragments the child's Murcko scaffold differs from *both*
parents, Nicolaou et al.'s BREED tradition), plus the two **ring-bond
cuts** below. The operator also works standalone —
`scaffold_hop_crossover(parent_a, parent_b, rng)` — for surgical use
outside the GA.
"""

WALK_8_OLD = """\
* `crossover="scaffold_hop"` — BREED-style recombination: cut two parents
  at a pair of homologous single bonds, recombine the fragments
  cross-wise, and the offspring's framework differs from *both* parents —
  the scaffold hop.
"""

WALK_8_NEW = """\
* `crossover="scaffold_hop"` — BREED-style recombination: cut two parents
  at a pair of homologous single bonds and recombine them — fragments
  swapped cross-wise, two rings merged into one, or a linker atom
  transplanted into a cut ring bond — so the offspring's framework
  differs from *both* parents: the scaffold hop.
"""

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "in the GA: framework recombination",  # end of the 6b code cell
        [("markdown", RING_MD), ("code", RING_CODE)],
    ),
}

REPLACEMENTS = {
    GEN_NOTEBOOK: [(GEN_6B_OLD, GEN_6B_NEW)],
    WALK_NOTEBOOK: [(WALK_8_OLD, WALK_8_NEW)],
}

PINS = {
    GEN_NOTEBOOK: [
        "C1CCNCCOCCNCC1",       # merge demo child (12-ring)
        "CC1=NCC(=O)N(C)c2ccccc2O1",  # transplant demo child (8-ring)
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # bridge demo
    ],
    WALK_NOTEBOOK: [
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # bridge demo
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
        if any(RING_MD_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the ring-topology cells")
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
            hit = False
            for cell in nb.cells:
                src = "".join(cell.source)
                if old in src:
                    cell.source = src.replace(old, new)
                    hit = True
                    changed += 1
                    break
            if not hit and new.splitlines()[0] not in "".join(
                c.source for c in nb.cells if c.cell_type == "markdown"
            ):
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
    print("DONE: ring-bond cuts documented, executed and pinned")


if __name__ == "__main__":
    main()
