"""27 driver: generative sections for the walkthrough + generative notebooks.

Round 100x-27's open item (e): the walkthrough notebook gets its molecule-
generation section (§8) and the generative workflow notebook gets §6b for
the round's flagship operator (``crossover="scaffold_hop"``), so the two
notebooks document the full operator matrix.  Both sections are FILE-
DRIVEN (the seeds load from examples/data/seed_molecules.csv; the
standalone demo crosses two of those shipped molecules) -- no inline data
fabrication, the 54-assertion guard stays green.

Steps (idempotent -- re-running only re-executes):

1. INSERT the new cells when their section markers are absent.
2. EXECUTE both notebooks with nbclient (cwd=examples/, worker-cap env).
3. TWINS -- regenerate the generative ``*_no_output.ipynb`` variant via
   scratch/strip_notebook_outputs.py (the walkthrough has no twin).
4. VERIFY the banned fabrication tokens are absent and the pinned
   spot-check values appear in the executed outputs.

Pins recorded from the 2026-10 rdkit 2026.09.1 / pharmaml 1.0.0 run:

  walkthrough §8: de novo (pop 32, gen 5, seed 42) -> 137 unique, best
    0.855 'COc1ccc2[nH]c(CCC(C)=O)cc2c1' (the 100x-22-a back-compat pin);
    seeded+hop (gen 8) -> 203 unique, best 0.894; aspirin x naproxen
    hybrid -> 'COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1'
  generative §6b: structural mode (pop 48, gen 15, seed 42) -> 522
    unique, best 0.886 'COc1ccc2[nH]c(CCNC(C)=O)cc2c1'
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

WALK_MARKER = "## 8 · Molecule generation"
GEN_MARKER = "## 6b · Scaffold hopping"

WALK_MD = """\
## 8 · Molecule generation — invent, don't just analyse

`pharmaml.generative` closes the loop: instead of only analysing molecules
you already have, it *optimises* molecules towards an objective.
`GeneticOptimizer` is a deterministic SMILES genetic algorithm with two
structural modes beyond the classic character mutation:

* `mutation="scaffold"` — keep the parent's Murcko framework verbatim and
  resample its decorations (lead optimisation);
* `crossover="scaffold_hop"` — BREED-style recombination: cut two parents
  at a pair of homologous single bonds, recombine the fragments
  cross-wise, and the offspring's framework differs from *both* parents —
  the scaffold hop.

The seed molecules load from the shipped `data/seed_molecules.csv` (a
dozen named drugs) — swap in your own file for a seeded run over your
chemistry.

One honest observation to set expectations: the character-level GA
invents *freely* — its best de novo molecule below even carries an
aromatic phosphorus (`[pH]`): chemically valid, decidedly exotic, and
exactly why the structural modes exist — they only recombine chemistry
that already works.
"""

WALK_CODE_1 = """\
from pharmaml.generative import (
    GeneticOptimizer, LogPBandObjective, MultiObjective, QedObjective,
)

objective = MultiObjective(
    [QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[0.7, 0.3])

de_novo = GeneticOptimizer(
    objective, population_size=32, generations=5, seed=42).optimize()
print(f"de novo: {len(de_novo)} unique, best score {de_novo.best_score:.3f}")
print(f"best:    {de_novo.best}")

seed_table = pd.read_csv(DATA_DIR / "seed_molecules.csv")
seeds = seed_table["smiles"].tolist()
seeded = GeneticOptimizer(
    objective, population_size=32, generations=8, seed=42,
    crossover="scaffold_hop").optimize(seed_smiles=seeds)
print(f"seeded + scaffold-hop: {len(seeded)} unique, best {seeded.best_score:.3f}")
seeded.to_dataframe().head(5)[["smiles", "score", "rank"]]
"""

WALK_CODE_2 = """\
import random

from pharmaml.generative import scaffold_hop_crossover

# the standalone operator: one homologous cut per parent, one hybrid child
draw = random.Random(0)
hybrid = scaffold_hop_crossover(seeds[0], seeds[9], draw)  # aspirin x naproxen
print(f"parent A:  {seeds[0]}   (aspirin)")
print(f"parent B:  {seeds[9]}   (naproxen)")
print(f"hybrid:    {hybrid}")
"""

GEN_MD = """\
## 6b · Scaffold hopping — cut two parents, recombine the fragments

`crossover="scaffold_hop"` is recombination with chemistry bookkeeping:
both parents are cut at a pair of *homologous* single acyclic bonds — the
join atoms match on element, degree, aromaticity and formal charge — and
the fragments are recombined cross-wise with `Chem.molzip`. Every ring
system arrives intact, so an offspring is a true hybrid: whenever the
parents' frameworks span different fragments, the child's Murcko scaffold
differs from *both* parents (Nicolaou et al.'s BREED tradition). The
operator also works standalone — `scaffold_hop_crossover(parent_a,
parent_b, rng)` — for surgical use outside the GA.
"""

GEN_CODE = """\
import random

from pharmaml.generative import GeneticOptimizer, scaffold_hop_crossover

# standalone: aspirin x naproxen, one homologous cut in each parent
draw = random.Random(0)
hybrid = scaffold_hop_crossover(seed_smiles[0], seed_smiles[9], draw)
print(f"parent A:  {seed_smiles[0]}   (aspirin)")
print(f"parent B:  {seed_smiles[9]}   (naproxen)")
print(f"hybrid:    {hybrid}\\n")

# in the GA: framework recombination + decoration resampling together
ga_hop = GeneticOptimizer(
    objective, population_size=48, generations=15, seed=SEED,
    mutation="scaffold", crossover="scaffold_hop",
).optimize(seed_smiles=seed_smiles)
print(f"structural mode: {len(ga_hop)} unique, best {ga_hop.best_score:.3f}")
ga_hop.to_dataframe().head(5)[["smiles", "score", "rank"]]
"""

# (notebook, marker, [(kind, source), ...]) -- inserted AFTER the cell that
# precedes the marker's logical section end.
INSERTIONS = {
    "pharmaml_walkthrough.ipynb": (
        WALK_MARKER,
        [
            ("markdown", WALK_MD),
            ("code", WALK_CODE_1),
            ("code", WALK_CODE_2),
        ],
        "ic50_to_pic50",  # §7's code cell (the notebook's last cell)
    ),
    "complete_generative_workflow.ipynb": (
        GEN_MARKER,
        [
            ("markdown", GEN_MD),
            ("code", GEN_CODE),
        ],
        'ga_scaffold = runs["scaffold"]',  # §6's final code cell
    ),
}

PINS = {
    "pharmaml_walkthrough.ipynb": (
        "de novo: 137 unique",  # MultiObjective run, pop 32 / gen 5 / seed 42
        "seeded + scaffold-hop: 203 unique",
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # hybrid
        "COc1cnc[pH]c(C)cc(Cc2cnc(N)nc2N)c1",  # the honest [pH] best
    ),
    "complete_generative_workflow.ipynb": (
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # hybrid
        "522 unique",  # structural mode
        "COc1ccc2[nH]c(CCNC(C)=O)cc2c1",  # structural best
    ),
}


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def _cell(kind: str, source: str):
    import nbformat.v4

    if kind == "markdown":
        return nbformat.v4.new_markdown_cell(source)
    return nbformat.v4.new_code_cell(source)


def insert_cells() -> None:
    import nbformat

    for name, (marker, cells, anchor) in INSERTIONS.items():
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        if any(marker in "".join(c.source) for c in nb.cells):
            print(f"insert {name}: marker already present -- skipped")
            continue
        positions = [
            i for i, c in enumerate(nb.cells) if anchor in "".join(c.source)
        ]
        if not positions:
            _fail(f"{name}: anchor {anchor!r} not found")
        # insert after the LAST cell matching the anchor (the section's
        # final cell), keeping later sections in place
        at = positions[-1] + 1
        for offset, (kind, source) in enumerate(cells):
            nb.cells.insert(at + offset, _cell(kind, source))
        nbformat.write(nb, path)
        print(f"insert {name}: {len(cells)} cell(s) after index {at - 1}")


def execute_notebooks() -> None:
    import nbformat
    from nbclient import NotebookClient

    for name in INSERTIONS:
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
    execute_notebooks()
    regenerate_twin()
    verify()
    print("DONE: generative sections landed file-driven and executed")


if __name__ == "__main__":
    main()
