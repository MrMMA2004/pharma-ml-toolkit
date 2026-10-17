"""37 driver SKETCH: topology-mutation A/B + swap-census line (notebook side).

Round 100x-37's notebook work (design by review agent 100x-37-r3; this
is the implementation sketch, verbatim-ready):

1. INSERT four cells after the topology-mutation demo (anchor: the
   ``ga_top.to_dataframe()`` cell) and BEFORE "### Segment swaps":
   [markdown framing] + [code: the ring-rich objective] + [code: the
   A/B runs + comparison] + [markdown interpretation].  Same seed, same
   corpus parents, same hyperparameters as the section-6 A/B
   (population_size=12, generations=12, seed=SEED) -- ONLY the mutation
   mode differs.
2. PATCH the segment-swap markdown ("### Segment swaps") with the
   pinned corpus census line (100x-36 open item f).
3. PATCH the section-11 recap with a "Ring-size A/B" bullet.
4. EXECUTE the generative notebook with nbclient (cwd=examples/).  The
   new cells only ADD output: no existing name is rebound (audited:
   RingSizeBandObjective / sat_ring_sizes / ring_objective /
   ring_parents / topo_ab / history_ab / primidone /
   expanded_primidone / probe_rows / objective_frame are all new), and
   every GeneticOptimizer owns its random.Random(SEED) -- no shared rng
   stream exists, so existing pins cannot shift.
5. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
6. VERIFY banned fabrication tokens absent + pinned values present.

Pins (deterministic, probed twice -- /tmp/task37r3/probe10_out*.txt):

  generative ring-size A/B:
    30 corpus molecules carry a movable saturated ring
    topology : best 0.928 (82 unique) -- NCC1CCNC(c2ccccc2)(c2ccccc2)C(=O)N1
    uniform  : best 0.872 (97 unique) -- C=S1C=NCCC2CCCNC(=O)C2=C2CC1CCON2
    topology  top-12 ring census: {6: 5, 7: 7}
    uniform   top-12 ring census: {5: 3, 6: 5, 7: 12, 8: 3}
    CCC1(c2ccccc2)CCNC(=O)NC1=O   (primidone's 7-ring expansion, total 0.828)
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

from test_notebook_variants import BANNED_FABRICATION_TOKENS  # noqa: E402

WORKER_ENV = {
    **os.environ,
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

GEN_NOTEBOOK = "complete_generative_workflow.ipynb"

AB_MD_MARKER = "### Topology vs uniform"  # idempotency: section heading

AB_MD = """\
### Topology vs uniform — the ring-size A/B

§6 ran the classic A/B (scaffold-preserving vs uniform mutation,
framework retention); here is the ring-size twin, on an objective where
ring sizes are the whole game.  `RingSizeBandObjective` — a custom
`ScoringObjective`, §2's documented extension surface — bands the *mean
size of the saturated rings* (exactly the rings the topology moves act
on) around 7 members, and rides in a `MultiObjective` with the shipped
`QedObjective` and a lead-size `MolecularWeightBandObjective`: the
target is a drug-like molecule with a 7-membered saturated ring — the
azepane/oxazepane axis of scaffold morphing.  The two arms share the
seed, the corpus parents (the movable subset of the bundled corpus)
and every hyperparameter; only the mutation mode differs.  Watch the
population mean and the valid-offspring fraction: a topology move is
always a pre-sanitised molecule one ring-size step from its parent —
a character edit on a lead-sized SMILES is usually fatal.  Every
parent is looked up from the bundled corpus again — nothing invented
inline.
"""

OBJECTIVE_CODE = '''\
# a ring-rich objective: 7-membered saturated rings at lead size
import math

from rdkit import Chem

from pharmaml.generative import (
    MolecularWeightBandObjective,
    MultiObjective,
    QedObjective,
    ScoringObjective,
)


def sat_ring_sizes(mol):
    """Sizes of the fully-saturated single-bond rings -- the movable ones."""
    sizes = []
    for ring in mol.GetRingInfo().AtomRings():
        bonds = [
            mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])
            for i in range(len(ring))
        ]
        if all(
            b.GetBondType() == Chem.BondType.SINGLE and not b.GetIsAromatic()
            for b in bonds
        ):
            sizes.append(len(ring))
    return sizes


class RingSizeBandObjective(ScoringObjective):
    """Gaussian band on the mean size of the saturated rings."""

    name = "ring_size_band"

    def __init__(self, lo, hi):
        super().__init__()
        self.centre = (lo + hi) / 2.0
        self.sigma = max((hi - lo) / 2.0, 1e-6)

    def score(self, smiles):
        mol = self._parse(smiles)
        if mol is None:
            return 0.0
        sizes = sat_ring_sizes(mol)
        if not sizes:
            return 0.0  # nothing movable: a ring objective cannot be satisfied
        z = (sum(sizes) / len(sizes) - self.centre) / self.sigma
        return math.exp(-0.5 * z * z)


ring_objective = MultiObjective(
    [
        RingSizeBandObjective(6.0, 8.0),  # the 7-ring band
        QedObjective(),
        MolecularWeightBandObjective(220.0, 380.0),  # lead size
    ]
)

# the objective on the demo parent above, its one-step move set, and a
# lead-sized corpus parent (primidone) plus its 7-ring expansion
primidone = DEFAULT_CORPUS[DEFAULT_CORPUS.index("CCC1(c2ccccc2)CNC(=O)NC1=O")]
expanded_primidone = next(
    m
    for m in ring_topology_moves(primidone)
    if sat_ring_sizes(Chem.MolFromSmiles(m)) == [7]
)
probe_rows = [("piperidine", piperidine)]
probe_rows += [
    (f"move -> {sat_ring_sizes(Chem.MolFromSmiles(m))}", m)
    for m in ring_topology_moves(piperidine)
]
probe_rows += [("primidone (218 Da)", primidone), ("move -> [7]", expanded_primidone)]
pd.DataFrame(
    [
        {
            "molecule": label,
            "smiles": smi,
            **{
                k: round(v, 3)
                for k, v in ring_objective.score_with_breakdown(smi)[1].items()
            },
            "total": round(ring_objective.score(smi), 3),
        }
        for label, smi in probe_rows
    ]
)
'''

AB_CODE = '''\
# the A/B: same objective, same seed, same corpus -- only the mutation differs
from collections import Counter

ring_parents = [s for s in DEFAULT_CORPUS if ring_topology_moves(s)]
print(f"{len(ring_parents)} corpus molecules carry a movable saturated ring")

topo_ab = {}
for mode in ("topology", "uniform"):
    topo_ab[mode] = GeneticOptimizer(
        ring_objective, population_size=12, generations=12, seed=SEED,
        mutation=mode,
    ).optimize(corpus=ring_parents)
    run = topo_ab[mode]
    print(f"{mode:9s}: best {run.best_score:.3f} ({len(run)} unique) -- {run.best}")

# ring-size census of each arm's top-12 elite: surgical vs broad
for mode, run in topo_ab.items():
    census = Counter()
    for smi in run.smiles[:12]:
        census.update(sat_ring_sizes(Chem.MolFromSmiles(smi)))
    print(f"{mode:9s} top-12 ring census: {dict(sorted(census.items()))}")

# trajectories side by side: best / population mean / offspring survival
history_ab = pd.DataFrame(
    {
        (mode, metric): [round(h[metric], 3) for h in run.history]
        for mode, run in topo_ab.items()
        for metric in ("best_score", "mean_score", "valid_offspring_fraction")
    }
)
history_ab.columns.names = ["mutation", "metric"]
history_ab
'''

AB_MD2 = """\
The topology arm wins on every axis that matters: its best —
`NCC1CCNC(c2ccccc2)(c2ccccc2)C(=O)N1`, a drug-like lead built on a
7-membered ring — scores 0.928 and appears in generation 2; uniform
needs seven generations to reach 0.872, and its winner is an exotic
fused chimera that satisfies the objective's letter rather than its
spirit.  The population mean leads in all eleven generations after the
shared start (0.802 vs 0.638 at the end) and the valid-offspring
fraction runs about 4× higher (0.42 vs 0.10 averaged) — roughly nine in
ten character-edited offspring die before scoring.  The census line is
the surgical-vs-broad story in miniature: the topology elite are twelve
single-ring molecules spanning two ring sizes — one expansion step from
the corpus parents — while the uniform elite pack 23 rings across four
sizes.  Uniform *can* find 7-rings by luck; it just scatters, where
topology walks there.
"""

# --- markdown cell replacements (old snippet -> new snippet) ------------

# 100x-36 open item (f): the pinned swap census, notebook voice.
CENSUS_OLD = """\
parent.  Both demo parents are looked up from the bundled corpus
again — nothing invented inline.
"""
CENSUS_NEW = """\
parent.  At corpus scale the exchange space is wide: across the bundled
244 molecules the swap scan admits 7,328 candidates over 295 distinct
linker chemistries — 1,738 heteroatom bioisostere exchanges (O↔N 153
each way, the ester/amide class above), 1,438 conservative C→C, and
N↔C 368 each way.  Both demo parents are looked up from the bundled
corpus again — nothing invented inline.
"""

RECAP_OLD = """\
* **Scaffold hopping** — the six-geometry BREED crossover (bridge,
  merge, transplant, fuse, insert, swap) plus the single-parent
  ring-size topology mutations (§6b);
"""
RECAP_NEW = """\
* **Scaffold hopping** — the six-geometry BREED crossover (bridge,
  merge, transplant, fuse, insert, swap) plus the single-parent
  ring-size topology mutations (§6b);
* **Ring-size A/B** — topology vs uniform mutation, same seed and
  parents, only the operator differs: best 0.928 in generation 2 vs
  0.872 in generation 7, population mean ahead in every generation,
  ~4× the offspring survival (§6b);
"""

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "ga_top.to_dataframe()",  # end of the topology-mutation demo cell
        [
            ("markdown", AB_MD),
            ("code", OBJECTIVE_CODE),
            ("code", AB_CODE),
            ("markdown", AB_MD2),
        ],
    ),
}

REPLACEMENTS = {
    GEN_NOTEBOOK: [
        (CENSUS_OLD, CENSUS_NEW),
        (RECAP_OLD, RECAP_NEW),
    ],
}

PINS = {
    GEN_NOTEBOOK: [
        # -- the new A/B demo (probed twice, byte-identical) ----------------
        "30 corpus molecules carry a movable saturated ring",
        "topology : best 0.928 (82 unique) -- NCC1CCNC(c2ccccc2)(c2ccccc2)C(=O)N1",
        "uniform  : best 0.872 (97 unique) -- C=S1C=NCCC2CCCNC(=O)C2=C2CC1CCON2",
        "topology  top-12 ring census: {6: 5, 7: 7}",
        "uniform   top-12 ring census: {5: 3, 6: 5, 7: 12, 8: 3}",
        "CCC1(c2ccccc2)CCNC(=O)NC1=O",  # primidone's 7-ring expansion
        "0.828",  # its total score in the objective probe table
        # -- the census line + recap (100x-36 item f / new bullet) ----------
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
        if any(AB_MD_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the topology A/B cells")
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
    print("DONE: topology A/B + swap census documented, executed and pinned")


if __name__ == "__main__":
    main()
