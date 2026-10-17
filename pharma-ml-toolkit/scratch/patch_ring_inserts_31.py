"""31 driver: ring-as-linker inserts (notebook side).

Round 100x-31's flagship: ``scaffold_hop_crossover``'s fifth geometry
-- **insert** -- completes the {excised piece} x {cut type} matrix: a
whole ring is excised from one parent (the same two-anchor donor
machinery as the fuse geometry) and spliced INTO a cut *acyclic* bond
of the other parent.  Both bridge halves are re-joined *through* the
ring: the ring becomes the spacer -- ring-as-linker.  This driver
lands the notebook documentation:

1. PATCH the generative notebook's ring-topology markdown (the intro
   now counts three ring-bond cuts plus one ring-piece insertion; a
   new insert bullet) and INSERT a ring-insert demo cell pair after
   the 30 fused-ring demo (file-driven: both parents looked up from
   the bundled corpus by SMILES value, no inline fabrication).
2. PATCH the walkthrough's 8 bullet to mention ring-as-linker inserts.
3. EXECUTE both notebooks with nbclient (cwd=examples/, worker-cap env)
   -- the structural GA runs legitimately drift again: 1119 corpus
   pairs carry viable insert candidates, so the GA can now reach the
   insert geometry.
4. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
5. VERIFY the banned fabrication tokens are absent and the pinned
   spot-check values appear in the executed outputs.

Pins (deterministic, independent of GA trajectories):

  generative 6d insert demo (shared Random(20), call order fixed):
    ibuprofen x glycine     -> 'NCc1ccc(C(=O)O)cc1'    (aminomethylbenzoic acid -- the benzene
                              spliced into glycine's cut CH2-C(=O) bond, between its amine and acid)
    aspirin x ethyl acetate -> 'CCOc1ccccc1C(C)=O'    (the ester chain re-joined through the aryl
                              spacer, aspirin's benzene into ethyl acetate's cut O-C(=O) bond)
  earlier demo pins are pool-invariant under the insert geometry (verified by signature analysis
  AND empirically): morpholine x piperidine / benzodiazepinone x aspirin / nicotine x cimetidine
  / nicotine x lysine / aspirin x naproxen (no viable insert candidate: no host-signature match
  against any acyclic bond endpoint), the 30 fuse demos (morpholine recipients have no acyclic
  bond to cut), and every decalin/indane test pair.
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

INSERT_MD_MARKER = "### Ring inserts"

INSERT_MD = """\
### Ring inserts — the ring becomes the linker

The fifth geometry completes the {excised piece} × {cut type} matrix:
the same two-anchor ring piece as the fused splice, but spliced INTO
a cut *acyclic* bond of the other parent — both bridge halves re-join
*through* the ring. The ring becomes the spacer: ring-as-linker. Both
demo parents are looked up from the bundled corpus again — nothing
invented inline.
"""

INSERT_CODE = '''\
# ring-as-linker inserts -- the ring becomes the spacer, corpus parents
draw = random.Random(20)

# ibuprofen's benzene (its two anchor hosts: the isobutyl CH2 and the
# acetic CH) is excised and spliced INTO glycine's cut CH2-C(=O) bond:
# the ring lands between glycine's amine and its acid -> an
# aminomethylbenzoic acid
ibuprofen = DEFAULT_CORPUS[DEFAULT_CORPUS.index("CC(C)Cc1ccc(C(C)C(=O)O)cc1")]
glycine = DEFAULT_CORPUS[DEFAULT_CORPUS.index("NCC(=O)O")]
spliced = scaffold_hop_crossover(ibuprofen, glycine, draw)
print(f"ring donor  : {ibuprofen}   rings={ring_sizes(ibuprofen)}")
print(f"chain cut   : {glycine}  (the CH2-C(=O) bond)")
print(
    f"spliced     : {spliced}   rings={ring_sizes(spliced)}"
    "  <- the benzene between glycine's amine and acid\\n"
)

# aspirin's benzene into ethyl acetate's cut O-C(=O) bond: the ester
# chain re-joined through the aryl spacer
aspirin = DEFAULT_CORPUS[DEFAULT_CORPUS.index("CC(=O)Oc1ccccc1C(=O)O")]
ethyl_acetate = DEFAULT_CORPUS[DEFAULT_CORPUS.index("CCOC(C)=O")]
esterified = scaffold_hop_crossover(aspirin, ethyl_acetate, draw)
print(f"ring donor  : {aspirin}   rings={ring_sizes(aspirin)}")
print(f"chain cut   : {ethyl_acetate}  (the O-C(=O) bond)")
print(
    f"esterified  : {esterified}   rings={ring_sizes(esterified)}"
    "  <- the chain re-joined through the aryl spacer"
)
'''

# --- markdown cell replacements (old snippet -> new snippet) ------------

GEN_RINGMD_OLD = """\
  bridged tricyclic).

The parents are looked up from the bundled corpus by SMILES value —
nothing is invented inline.
"""

GEN_RINGMD_NEW = """\
  bridged tricyclic);
* **insert** — the same two-anchor ring piece, but spliced INTO a cut
  *acyclic* bond of the other parent: both bridge halves re-join
  *through* the ring, which becomes the spacer — ring-as-linker, the
  {excised piece} × {cut type} matrix's last cell (the ring-insert
  demo after the fused splices puts ibuprofen's benzene into
  glycine's cut CH2-C(=O) bond, giving an aminomethylbenzoic acid).

The parents are looked up from the bundled corpus by SMILES value —
nothing is invented inline.
"""

GEN_INTRO_OLD = """\
moves, and the operator has three (single ring bonds only — an aromatic
ring bond would make the re-zip invent bond orders, so those are never
cut):
"""

GEN_INTRO_NEW = """\
moves, and the operator has three ring-bond cuts plus one ring-piece
insertion into a cut acyclic bond (single ring bonds only — an aromatic
ring bond would make the re-zip invent bond orders, so those are never
cut):
"""

WALK_8_OLD = """\
  swapped cross-wise, two rings merged into one, a two-ended linker
  — a single atom or a short acyclic path — transplanted into a cut
  ring bond, or a whole ring spliced across the cut into a fused
  bicyclic — so the offspring's framework
"""

WALK_8_NEW = """\
  swapped cross-wise, two rings merged into one, a two-ended linker
  — a single atom or a short acyclic path — transplanted into a cut
  ring bond, a whole ring spliced across the cut into a fused
  bicyclic, or a whole ring spliced into a cut acyclic bond as the
  spacer — so the offspring's framework
"""

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "both benzene paths close independently",  # end of the 30 fuse demo
        [("markdown", INSERT_MD), ("code", INSERT_CODE)],
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
        "NCc1ccc(C(=O)O)cc1",   # insert demo: ibuprofen benzene into glycine
        "CCOc1ccccc1C(C)=O",   # insert demo: aspirin benzene into ethyl acetate
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
        if any(INSERT_MD_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the ring-insert cells")
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
    print("DONE: ring-as-linker inserts documented, executed and pinned")


if __name__ == "__main__":
    main()
