"""35 driver: segment-swap crossover (notebook side).

Round 100x-35's flagship: the SIXTH scaffold-hop recombination
geometry -- two-point segment swap, each parent's bounded acyclic
middle (a single atom or a contiguous path of up to four atoms)
excised from the flank parent and replaced by the piece parent's
homologous middle.  The only size-conserving geometry and the medchem
linker / bioisostere exchange (an ester O for an amide N).  This
driver lands the notebook documentation:

1. INSERT a segment-swap demo cell pair after the topology-mutation
   demo (file-driven: both demo parents looked up from the bundled
   corpus by SMILES value, no inline fabrication).
2. PATCH the generative notebook's §6b geometry intro (five -> six),
   the §9 head-to-head arm description and the §11 recap bullet.
3. PATCH the walkthrough's §8 mode-list bullet (the crossover
   enumeration gains the segment swap).
4. EXECUTE both notebooks with nbclient (cwd=examples/, worker-cap
   env).  Standalone demo pins are unaffected -- the aspirin x
   naproxen bridge demo pair gains ZERO swap candidates (probed), so
   its pinned hybrid and QED output stay byte-identical; GA-run cells
   re-execute with whatever the (now larger) candidate pools draw --
   outputs regenerate, sources untouched.
5. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
6. VERIFY the banned fabrication tokens are absent and the pinned
   spot-check values appear in the executed outputs.

Pins (deterministic, independent of GA trajectories):

  generative 6c swap demo:
    aspirin x paracetamol x Random(0) -> 'CC(=O)Nc1ccccc1C(=O)O'
    aspirin x paracetamol x Random(5) -> 'CC(=O)Oc1ccc(O)cc1'
  earlier demo pins are unaffected (zero swap candidates for every
  pinned standalone pair -- the 202-test green run is the witness).
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
WALK_NOTEBOOK = "pharmaml_walkthrough.ipynb"

SWAP_MD_MARKER = "### Segment swaps"

SWAP_MD = """\
### Segment swaps — the bioisostere exchange

The sixth geometry is two-point crossover, the graph analogue of
classic GA recombination: each parent's bounded acyclic *middle* — a
single atom or a contiguous path of up to four atoms — is excised from
the flank parent and REPLACED by the other parent's homologous middle
(the flanking environments must match, direct or flipped, so each seam
bonds signature-matched atoms).  It is the only size-conserving
geometry — insert grows a recipient, swap exchanges in place — and it
is the medchem **bioisostere** move: an ester oxygen exchanged for an
amide nitrogen, an ether O for a thioether S.  Aspirin × paracetamol
crosses exactly there: the ester and the amide sit in homologous
environments (both flanked by a carbonyl carbon and an aryl carbon),
so the swap yields salicylamide — a real molecule that is neither
parent.  Both demo parents are looked up from the bundled corpus
again — nothing invented inline.
"""

SWAP_CODE = '''\
# two-point segment swaps -- the linker / bioisostere exchange, corpus parents
from pharmaml.generative import scaffold_hop_crossover

aspirin = DEFAULT_CORPUS[DEFAULT_CORPUS.index("CC(=O)Oc1ccccc1C(=O)O")]
paracetamol = DEFAULT_CORPUS[DEFAULT_CORPUS.index("CC(=O)Nc1ccc(O)cc1")]
print(f"parent A:  {aspirin}   (aspirin)")
print(f"parent B:  {paracetamol}   (paracetamol)")

# the ester O and the amide N sit in homologous environments: the swap
# crosses them -- seed 0 lands the salicylamide child, seed 5 the twin
first = scaffold_hop_crossover(aspirin, paracetamol, random.Random(0))
second = scaffold_hop_crossover(aspirin, paracetamol, random.Random(5))
print(f"seed 0 ->  {first}   (salicylamide: neither parent)")
print(f"seed 5 ->  {second}   (the O-for-N twin)")

# in the GA the geometry joins the merged candidate pool invisibly:
# no GeneticOptimizer change, the draw just has one more move class
'''

# --- markdown cell replacements (old snippet -> new snippet) ------------

GEN_6B_OLD = """\
`Chem.molzip`. Five geometries compete for the draw: **bridge cuts**
(a pair of single acyclic bonds, fragments swapped cross-wise — every
ring system arrives intact, and whenever the parents' frameworks span
different fragments the child's Murcko scaffold differs from *both*
parents, Nicolaou et al.'s BREED tradition), the **ring-bond cuts**
below — merges, linker transplants, fused-ring splices — plus the
**ring insert**, a whole ring spliced into a cut acyclic bond as the
linker."""
GEN_6B_NEW = """\
`Chem.molzip`. Six geometries compete for the draw: **bridge cuts**
(a pair of single acyclic bonds, fragments swapped cross-wise — every
ring system arrives intact, and whenever the parents' frameworks span
different fragments the child's Murcko scaffold differs from *both*
parents, Nicolaou et al.'s BREED tradition), the **ring-bond cuts**
below — merges, linker transplants, fused-ring splices — plus the
**ring insert**, a whole ring spliced into a cut acyclic bond as the
linker, and the **segment swap**, each parent's bounded acyclic middle
exchanged for the other's homologous middle (the bioisostere move:
an ester O for an amide N)."""

GEN_9_OLD = "scaffold-preserving mutation plus the five-geometry scaffold-hop"
GEN_9_NEW = "scaffold-preserving mutation plus the six-geometry scaffold-hop"

GEN_RECAP_OLD = """\
* **Scaffold hopping** — the five-geometry BREED crossover (bridge,
  merge, transplant, fuse, insert) plus the single-parent ring-size
  topology mutations (§6b);
"""
GEN_RECAP_NEW = """\
* **Scaffold hopping** — the six-geometry BREED crossover (bridge,
  merge, transplant, fuse, insert, swap) plus the single-parent
  ring-size topology mutations (§6b);
"""

WALK_8_OLD = """\
  bicyclic, or a whole ring spliced into a cut acyclic bond as the
  spacer — so the offspring's framework
  differs from *both* parents: the scaffold hop;
"""
WALK_8_NEW = """\
  bicyclic, or a whole ring spliced into a cut acyclic bond as the
  spacer, or a two-point segment swap exchanging one parent's bounded
  acyclic middle — a single atom or a short path — for the other's
  homologous middle (the bioisostere exchange: an ester O for an
  amide N) — so the offspring's framework
  differs from *both* parents: the scaffold hop;
"""

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "ga_top.to_dataframe()",  # end of the topology-mutation demo
        [("markdown", SWAP_MD), ("code", SWAP_CODE)],
    ),
}

REPLACEMENTS = {
    GEN_NOTEBOOK: [
        (GEN_6B_OLD, GEN_6B_NEW),
        (GEN_9_OLD, GEN_9_NEW),
        (GEN_RECAP_OLD, GEN_RECAP_NEW),
    ],
    WALK_NOTEBOOK: [(WALK_8_OLD, WALK_8_NEW)],
}

PINS = {
    GEN_NOTEBOOK: [
        "seed 0 ->  CC(=O)Nc1ccccc1C(=O)O",  # the swap demo: salicylamide
        "seed 5 ->  CC(=O)Oc1ccc(O)cc1",  # the O-for-N twin
        "six-geometry BREED crossover (bridge,",  # the recap bullet
        "the six-geometry scaffold-hop",  # the §9 arm description
        "the **segment swap**, each parent's bounded acyclic middle",  # §6b
        # 31/32/33 regression pins (must survive the re-execution untouched)
        "NCc1ccc(C(=O)O)cc1",  # insert demo: ibuprofen benzene into glycine
        "CCOc1ccccc1C(C)=O",  # insert demo: aspirin benzene into ethyl acetate
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # bridge demo
        "['C1CCCC1', 'C1CCCNCC1', 'C1CCNC1']",  # topology demo: piperidine
        "morpholine x Random(0) -> C1COCN1",  # topology demo draw
        "ga (structural)",  # 32's head-to-head fourth arm
    ],
    WALK_NOTEBOOK: [
        "two-point segment swap exchanging one parent's bounded",  # new bullet
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
        if any(SWAP_MD_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the segment-swap cells")
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
    print("DONE: segment-swap crossover documented, executed and pinned")


if __name__ == "__main__":
    main()
