"""100x-31 fail-first probe: the ring-as-linker insert geometry is absent.

Goal: the fifth crossover geometry -- "insert" -- completing the
{excised piece} x {cut type} matrix:

  acyclic piece -> ring cut     = transplant / transplant_path  (28/29)
  ring piece    -> ring cut     = fuse                          (30)
  acyclic piece -> acyclic cut  = bridge (fragments swap)       (classic)
  ring piece    -> acyclic cut  = INSERT                        (missing!)

The insert move excises a whole ring from one parent (the same
_ring_units donor machinery) and splices it INTO a cut acyclic bond of
the other parent: the two bridge fragments are re-joined THROUGH the
ring -- the ring becomes the spacer (a ring-as-linker scaffold hop).

Probe: decalin x hexane.  Decalin's left 6-ring is an admissible donor
(its hosts are degree-2 sp3 CH2s); hexane's middle C-C bonds have
degree-2 sp3 CH2 endpoints -- homologous.  Every other geometry is
unavailable for this pair (decalin has no acyclic bond to bridge-cut,
hexane has no ring bond to open or fuse into, neither donates a
linker path to the other's ring cut), so the CURRENT operator must
return None for every seed; with insert, children must be
dialkylcyclohexanes (one 6-ring, 12 atoms, single fragment).
"""

from __future__ import annotations

import random
import sys

from rdkit import Chem

sys.path.insert(0, "src")

from pharmaml.generative import scaffold_hop_crossover  # noqa: E402
from pharmaml.generative import scaffold_hop as sh  # noqa: E402


def probe(tag: str, smi_a: str, smi_b: str) -> None:
    print(f"--- {tag}: {smi_a} x {smi_b}")
    units_a = sh._ring_units(Chem.MolFromSmiles(smi_a))
    units_b = sh._ring_units(Chem.MolFromSmiles(smi_b))
    print(f"    ring units: A={[(u[1], u[2], u[3]) for u in units_a]} B={units_b}")
    shapes = {}
    for seed in range(200):
        child = scaffold_hop_crossover(smi_a, smi_b, random.Random(seed))
        if child is None:
            continue
        mol = Chem.MolFromSmiles(child)
        n_rings = mol.GetRingInfo().NumRings()
        shapes.setdefault(n_rings, []).append((seed, child))
    for n_rings, hits in sorted(shapes.items()):
        print(f"    {n_rings}-ring children: {len(hits)} seeds, e.g. "
              f"{hits[0][1]} (seed {hits[0][0]})")
    single_ring = shapes.get(1, [])
    donor_ring = len(units_a[0][0]) if units_a else len(units_b[0][0])
    expected_atoms = donor_ring + Chem.MolFromSmiles(smi_b).GetNumAtoms() if units_a else None
    if single_ring:
        print(f"    single-ring (spacer) children exist: {single_ring[0][1]}")
    else:
        print("    single-ring spacer children: NONE in 200 seeds")
    print("    FAIL-FIRST: insert geometry absent -> expected a "
          f"{expected_atoms}-atom single-ring spacer child")


probe("A", "C1CCC2CCCCC2C1", "CCCCCC")  # decalin x hexane
probe("B", "C1Cc2ccccc2C1", "CCCCCCC")  # indane x heptane (aromatic donor)
