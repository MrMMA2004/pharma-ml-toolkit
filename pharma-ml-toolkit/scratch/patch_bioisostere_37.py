"""37 driver: bioisostere mutations + swap census + stderr purge (notebook side).

Round 100x-37's notebook landing, three items in one re-execution:

1. CENSUS (carried open item f from 100x-36): the segment-swap markdown
   in the generative notebook gains the pinned quantitative census line
   (7,328 candidates / 295 linker chemistries / 1,738 bioisostere
   exchanges) that README + scaffold_hop.py already carry.
2. BIOISOSTERE DEMO (the round's flagship): a markdown + code cell pair
   after the segment-swap demo -- bioisostere_moves / bioisostere_mutate
   on corpus parents (aspirin: 21 moves, salicylamide among them; the
   mutate draw pins seed 0 -> pyridine variant, seed 1 -> salicylamide
   itself; piperidine's classic ring bioisosteres). Plus the recap
   bullet / CLI line and the walkthrough's mode list.
3. STDERR PURGE (the round's hygiene fix): re-executing both notebooks
   under the fixed scaffold_hop (rdApp.warning now disabled during
   zipping) removes the 361 "Incomplete atom labelling" leak lines
   baked into the shipped outputs (28+7 stderr stream blocks). The
   driver pins the purge NEGATIVELY: the token must be absent after
   execution.

Conventions (established in 26/28/29/30/31/32/33/35): insert-marker +
replace_markdown idempotence checks (new-in-src BEFORE old-in-src);
execution via nbclient with cwd=examples/ and the worker-cap env; twins
regenerated through strip_notebook_outputs.py; banned fabrication
tokens absent; demo pins present in executed outputs.

Pins (deterministic, corpus-driven):
  bioisostere demo (generative, after the swap demo):
    bioisostere_moves(aspirin) -> 21 exchanges, salicylamide among them
    bioisostere_mutate(aspirin, Random(0)) -> 'CC(=O)Oc1ccncc1C(=O)O'
    bioisostere_mutate(aspirin, Random(1)) -> 'CC(=O)Nc1ccccc1C(=O)O'
  negative pin (both notebooks): 'Incomplete atom labelling' absent.
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

BIOISO_MD_MARKER = "### Bioisostere mutations"

BIOISO_MD = """\
### Bioisostere mutations — the single-parent twin

The swap needs *both* chemistries present in the population — an ester
in one parent, an amide in the other. The bioisostere mutation family
is the single-parent twin that invents the exchange on the spot: one
atom exchanged for another from the C/N/O/S replacement alphabet
(halogens available through ``allowed``), fenced by valence and
aromaticity rules (aryl carbon swaps only with nitrogen — the
pyridines; a quaternary carbon is never handed nitrogen's valence).
Aspirin alone reaches salicylamide — the celebrated swap child, now
single-parent — and piperidine enumerates the classic ring
bioisosteres: piperazine, morpholine, thiomorpholine. The GA wiring is
``mutation="bioisostere"`` (CLI: ``--bioisostere``), the draw falling
back to character mutation when a molecule admits no exchange.
"""

BIOISO_CODE = '''\
# single-atom bioisostere mutations -- the swap's single-parent twin
from pharmaml.generative import bioisostere_moves, bioisostere_mutate

moves = bioisostere_moves(aspirin)
print(f"aspirin admits {len(moves)} single-atom exchanges")
print(f"salicylamide among them: {'CC(=O)Nc1ccccc1C(=O)O' in moves}")

# the GA's mutation draw: seed 0 lands the pyridine variant (an aryl C
# -> N), seed 1 salicylamide itself (the ester O -> the amide N)
first = bioisostere_mutate(aspirin, random.Random(0))
second = bioisostere_mutate(aspirin, random.Random(1))
print(f"seed 0 ->  {first}   (aryl C -> N)")
print(f"seed 1 ->  {second}   (salicylamide, single-parent)")

# piperidine's classic ring bioisosteres (all corpus-driven)
ring_moves = bioisostere_moves("C1CCNCC1")
print(f"piperidine -> {len(ring_moves)} exchanges:")
for mutant in ring_moves:
    print(f"  {mutant}")
'''

# --- markdown cell replacements (old snippet -> new snippet) ------------

# 1. the census paragraph appended to the segment-swap markdown
SWAP_CENSUS_OLD = """\
so the swap yields salicylamide — a real molecule that is neither
parent.  Both demo parents are looked up from the bundled corpus
again — nothing invented inline.
"""
SWAP_CENSUS_NEW = """\
so the swap yields salicylamide — a real molecule that is neither
parent.  Both demo parents are looked up from the bundled corpus
again — nothing invented inline.

Across the 244-molecule bundled corpus the swap scan admits 7,328
candidates over 295 distinct linker chemistries — 1,738 of them
heteroatom bioisostere exchanges (O↔N the classic ester/amide class).
"""

# 2. the recap bullet (families, plural)
GEN_RECAP_OLD = """\
* **Scaffold hopping** — the six-geometry BREED crossover (bridge,
  merge, transplant, fuse, insert, swap) plus the single-parent
  ring-size topology mutations (§6b);
"""
GEN_RECAP_NEW = """\
* **Scaffold hopping** — the six-geometry BREED crossover (bridge,
  merge, transplant, fuse, insert, swap) plus the single-parent
  mutation families — ring-size topology moves and single-atom
  bioisostere exchanges (§6b);
"""

# 3. the recap CLI line gains --bioisostere
GEN_CLI_OLD = "(+ `--scaffold`, `--scaffold-hop`, `--topology`, `--seeds`)"
GEN_CLI_NEW = (
    "(+ `--scaffold`, `--scaffold-hop`, `--topology`, `--bioisostere`,"
    " `--seeds`)"
)

# 4. walkthrough §8: the mode count + a bioisostere bullet
WALK_MODES_OLD = "genetic algorithm with two\nstructural modes beyond the classic character mutation:"
WALK_MODES_NEW = "genetic algorithm with four\nstructural modes beyond the classic character mutation:"

WALK_BIOISO_OLD = """\
* `mutation="topology"` — ring-size mutation, the single-parent moves:
  one atom removed from a saturated ring (contraction — piperidine →
  pyrrolidine) or one methylene inserted into a saturated single ring
  bond (expansion — piperidine → azepane), the ring-size axis of
  scaffold morphing.
"""
WALK_BIOISO_NEW = """\
* `mutation="topology"` — ring-size mutation, the single-parent moves:
  one atom removed from a saturated ring (contraction — piperidine →
  pyrrolidine) or one methylene inserted into a saturated single ring
  bond (expansion — piperidine → azepane), the ring-size axis of
  scaffold morphing;
* `mutation="bioisostere"` — single-atom heteroatom exchange, the
  single-parent twin of the segment swap: one atom swapped for another
  from the C/N/O/S replacement alphabet (aspirin alone → salicylamide;
  piperidine → piperazine / morpholine / thiomorpholine), the
  heteroatom axis of scaffold morphing.
"""

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "no GeneticOptimizer change, the draw just has one more move class",
        [("markdown", BIOISO_MD), ("code", BIOISO_CODE)],
    ),
}

REPLACEMENTS = {
    GEN_NOTEBOOK: [
        (SWAP_CENSUS_OLD, SWAP_CENSUS_NEW),
        (GEN_RECAP_OLD, GEN_RECAP_NEW),
        (GEN_CLI_OLD, GEN_CLI_NEW),
    ],
    WALK_NOTEBOOK: [
        (WALK_MODES_OLD, WALK_MODES_NEW),
        (WALK_BIOISO_OLD, WALK_BIOISO_NEW),
    ],
}

PINS = {
    GEN_NOTEBOOK: [
        # the bioisostere demo (this round)
        "aspirin admits 21 single-atom exchanges",
        "seed 0 ->  CC(=O)Oc1ccncc1C(=O)O",  # pyridine variant
        "seed 1 ->  CC(=O)Nc1ccccc1C(=O)O",  # salicylamide, single-parent
        "piperidine -> 12 exchanges",
        "C1CNCCN1",  # piperazine among the printed ring bioisosteres
        "C1COCCN1",  # morpholine
        "C1CSCCN1",  # thiomorpholine
        # the census paragraph (carried item f)
        "7,328",
        "295 distinct linker chemistries",
        "1,738 of them",
        # the recap bullet + CLI line
        "bioisostere exchanges (§6b);",
        "--bioisostere",
        # 31/32/33/35 regression pins (must survive the re-execution)
        "seed 0 ->  CC(=O)Nc1ccccc1C(=O)O",  # the swap demo: salicylamide
        "seed 5 ->  CC(=O)Oc1ccc(O)cc1",  # the O-for-N twin
        "NCc1ccc(C(=O)O)cc1",  # insert demo: ibuprofen benzene into glycine
        "CCOc1ccccc1C(C)=O",  # insert demo: aspirin benzene into ethyl acetate
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # bridge demo
        "['C1CCCC1', 'C1CCCNCC1', 'C1CCNC1']",  # topology demo: piperidine
        "morpholine x Random(0) -> C1COCN1",  # topology demo draw
        "ga (structural)",  # 32's head-to-head fourth arm
    ],
    WALK_NOTEBOOK: [
        "`mutation=\"bioisostere\"`",  # the new mode bullet
        "four",  # the mode count fix
        "aspirin alone → salicylamide",
        "two-point segment swap exchanging one parent's bounded",  # 35's bullet
        "spliced into a cut acyclic bond",  # 31's bullet stays
    ],
}

# The stderr-purge witness: these tokens must be ABSENT from outputs.
NEGATIVE_PINS = {
    GEN_NOTEBOOK: ["Incomplete atom labelling, cannot make bond"],
    WALK_NOTEBOOK: ["Incomplete atom labelling, cannot make bond"],
}


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def insert_cells() -> None:
    import nbformat

    for name, (marker, cells) in INSERTIONS.items():
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        if any(BIOISO_MD_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the bioisostere cells")
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
        for token in NEGATIVE_PINS[name]:
            if token in text:
                _fail(f"{name}: stderr-leak token {token!r} still present")
        print(
            f"verified {name}: guard clean, {len(pins)} pin(s) present, "
            f"{len(NEGATIVE_PINS[name])} negative pin(s) absent"
        )


def main() -> None:
    insert_cells()
    replace_markdown()
    execute_notebooks()
    regenerate_twin()
    verify()
    print("DONE: bioisostere mutations documented, census landed, stderr purged")


if __name__ == "__main__":
    main()
