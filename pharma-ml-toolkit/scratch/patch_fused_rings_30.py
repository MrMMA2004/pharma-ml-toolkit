"""30 driver: fused-ring splices (notebook side).

Round 100x-30's flagship: ``scaffold_hop_crossover``'s fourth geometry
-- **fuse** -- excises a whole ring from one parent (any ring attached
to its molecule through exactly two single anchor bonds at two distinct
ring atoms, aromatic ones included; the anchor bonds may be ring bonds
of an adjacent ring, which is exactly the fused-system case) and
splices it across the other parent's opened ring bond.  The child is a
fused / bridged bicyclic holding one ring from each parent: the full
Nicolaou move set.  This driver lands the notebook documentation:

1. PATCH the generative notebook's ring-topology markdown (the intro
   now counts THREE ring moves; a new fuse bullet) and INSERT a
   fused-ring demo cell pair after the 29 linker-path demo
   (file-driven: both parents looked up from the bundled corpus by
   SMILES value, no inline fabrication).
2. PATCH the walkthrough's 8 bullet to mention whole-ring splices.
3. EXECUTE both notebooks with nbclient (cwd=examples/, worker-cap env)
   -- the structural GA runs legitimately drift: the corpus carries
   ring-unit donors (56 molecules, incl. aspirin, ibuprofen, the
   diaryl ether, metoprolol) and saturated-ring recipients
   (piperidine, morpholine), so the GA can now reach the fuse
   geometry.
4. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
5. VERIFY the banned fabrication tokens are absent and the pinned
   spot-check values appear in the executed outputs.

Pins (deterministic, independent of GA trajectories):

  generative 6c fuse demo (shared Random(0), call order fixed):
    diaryl ether x morpholine -> 'c1ccc2c(c1)CCNCCO2'   (benzo-fused oxazocane, rings [6, 8])
    metoprolol   x morpholine -> 'c1cc2ccc1CCNCCO2'   (bridged tricyclic, rings [6, 10, 10])
  generative 6b demos (28/29 pins, pool-invariant or draw-stable):
    morpholine x piperidine -> 'C1CCNCCOCCNCC1'         (merge; neither parent has a ring unit)
    benzodiazepinone x aspirin -> 'CC1=NCC(=O)N(C)c2ccccc2O1'  (transplant; aspirin's unit
      candidates append AFTER the old pool, and the second shared-0 draw lands inside it)
    nicotine x cimetidine -> 'CN1CCSCCC1c1cccnc1'       (path; nicotine's unit hosts carry a
      degree-1 carbon, which no ring-bond endpoint can ever match)
    nicotine x lysine     -> 'CN1CCCCCC1c1cccnc1'       (path; same)
    aspirin x naproxen    -> 'COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1'  (bridge;
      naproxen's rings are all-aromatic, so no single ring bond exists to fuse into)
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

FUSE_MD_MARKER = "### Fused-ring splices"

FUSE_MD = """\
### Fused-ring splices — whole rings across the cut

The fourth geometry excises a whole ring from one parent — any ring
attached to its molecule through exactly two single anchor bonds,
aromatic ones included — and splices it across the other parent's
opened ring bond. The offspring is a fused or bridged ring system
holding one ring from each parent: the Nicolaou fused-ring move,
completing the BREED set. Both demo donors are looked up from the
bundled corpus again — nothing invented inline.
"""

FUSE_CODE = '''\
# fused-ring splices -- whole rings across the cut, corpus parents
draw = random.Random(0)

# the corpus diaryl ether's phenethylamine ring hangs off the rest of
# the molecule through two single anchor bonds (the ether O and the
# ethylamine CH2); spliced across morpholine's opened C-O bond it gives
# a benzo-fused oxazocane -- one ring from each parent
diaryl_ether = DEFAULT_CORPUS[DEFAULT_CORPUS.index("Cc1ccccc1Oc1ccccc1CCN(C)C")]
morpholine = DEFAULT_CORPUS[DEFAULT_CORPUS.index("C1COCCN1")]
fused = scaffold_hop_crossover(diaryl_ether, morpholine, draw)
print(f"ring donor    : {diaryl_ether}   rings={ring_sizes(diaryl_ether)}")
print(f"ring parent   : {morpholine}   rings={ring_sizes(morpholine)}")
print(
    f"fused         : {fused}   rings={ring_sizes(fused)}"
    "  <- the aryl ring spliced across the morpholine cut\\n"
)

# metoprolol's para-phenylene: the two anchors sit opposite each other,
# so BOTH benzene paths around the ring close independently -> a
# bridged tricyclic, not a simple bicycle
metoprolol = DEFAULT_CORPUS[DEFAULT_CORPUS.index("COCCc1ccc(OCC(O)CNC(C)C)cc1")]
bridged = scaffold_hop_crossover(metoprolol, morpholine, draw)
print(f"ring donor    : {metoprolol}   rings={ring_sizes(metoprolol)}")
print(f"ring parent   : {morpholine}   rings={ring_sizes(morpholine)}")
print(
    f"bridged       : {bridged}   rings={ring_sizes(bridged)}"
    "  <- para anchors: both benzene paths close independently"
)
'''

# --- markdown cell replacements (old snippet -> new snippet) ------------

GEN_RINGMD_OLD = """\
The parents are looked up from the bundled corpus by SMILES value —
nothing is invented inline.
"""

GEN_RINGMD_NEW = """\
* **fuse** — a whole ring is excised from one parent — any ring
  attached to its molecule through exactly two single anchor bonds at
  two distinct ring atoms, aromatic ones included (the anchor bonds may
  themselves be ring bonds of an adjacent ring: that is what a fused
  system is) — and spliced across the other parent's opened ring bond:
  a fused or bridged bicyclic holding one ring from each parent, the
  full Nicolaou move set — the fused-ring demo after the linker paths
  splices the corpus diaryl ether's aryl ring into morpholine (a
  benzo-fused oxazocane) and metoprolol's para-phenylene across it (a
  bridged tricyclic).

The parents are looked up from the bundled corpus by SMILES value —
nothing is invented inline.
"""

GEN_INTRO_OLD = """\
moves, and the operator has two (single ring bonds only — an aromatic
ring bond would make the re-zip invent bond orders, so those are never
cut):
"""

GEN_INTRO_NEW = """\
moves, and the operator has three (single ring bonds only — an aromatic
ring bond would make the re-zip invent bond orders, so those are never
cut):
"""

WALK_8_OLD = """\
  swapped cross-wise, two rings merged into one, or a two-ended linker
  — a single atom or a short acyclic path — transplanted into a cut
  ring bond — so the offspring's framework
"""

WALK_8_NEW = """\
  swapped cross-wise, two rings merged into one, a two-ended linker
  — a single atom or a short acyclic path — transplanted into a cut
  ring bond, or a whole ring spliced across the cut into a fused
  bicyclic — so the offspring's framework
"""

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "two methylenes longer",  # end of the 29 linker-path demo cell
        [("markdown", FUSE_MD), ("code", FUSE_CODE)],
    ),
}

REPLACEMENTS = {
    GEN_NOTEBOOK: [
        (GEN_RINGMD_OLD, GEN_RINGMD_NEW),
        (GEN_INTRO_OLD, GEN_INTRO_NEW),
    ],
    WALK_NOTEBOOK: [(WALK_8_OLD, WALK_8_NEW)],
}

PINS = {
    GEN_NOTEBOOK: [
        "c1ccc2c(c1)CCNCCO2",   # fuse demo: diaryl ether aryl ring splice
        "c1cc2ccc1CCNCCO2",   # fuse demo: metoprolol bridged tricyclic
        "CN1CCSCCC1c1cccnc1",   # path demo: cimetidine -CH2-S- transplant
        "CN1CCCCCC1c1cccnc1",   # path demo: lysine -CH2CH2- transplant
        "C1CCNCCOCCNCC1",       # merge demo child (12-ring, from 28)
        "CC1=NCC(=O)N(C)c2ccccc2O1",  # transplant demo child (from 28)
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
        if any(FUSE_MD_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the fused-ring cells")
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
    print("DONE: fused-ring splices documented, executed and pinned")


if __name__ == "__main__":
    main()
