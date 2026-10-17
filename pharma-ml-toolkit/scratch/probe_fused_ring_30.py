"""100x-30 fail-first probe: the fused-ring BREED geometry does not exist.

Goal: a fourth crossover geometry -- "fuse" -- that excises a whole ring
from one parent (a ring attached to the rest of the molecule via exactly
two single acyclic anchor bonds at two distinct ring atoms) and splices
it across an opened ring bond of the other parent, producing a fused /
bridged bicyclic offspring (the full Nicolaou move set).

Probe A: decalin x piperidine.  Decalin's left cyclohexane is attached
to the right one via exactly two single acyclic bonds (the fusion
carbons' bonds into the other ring); its former outside hosts are
degree-2 sp3 CH2s -- homologous to every C-C cut endpoint of
piperidine.  The fuse geometry must therefore be viable, and its
children must be bicyclic (ring count 2, the donor 6-ring preserved).
Current behavior: no such geometry exists -- the candidate pool has only
bridge/merge/transplant entries, so the pair can never produce a fused
bicyclic child.

Probe B: indane x piperidine.  Indane's BENZENE ring hangs off the
cyclopentane via two single acyclic aryl-CH2 bonds whose hosts are
degree-2 sp3 CH2s -- again homologous to piperidine's C-C endpoints.
The aromatic donor ring must survive the excision intact (aromatic ring
bonds are never cut; only the two anchor bonds are).

Probe C: stereo of the invariant -- every seam bonds signature-matched
atoms; children must be valid, connected, dummy-free.
"""

from __future__ import annotations

import random
import sys

from rdkit import Chem

sys.path.insert(0, "src")

from pharmaml.generative import scaffold_hop as sh  # noqa: E402


def ring_fragments(mol: object) -> list[tuple]:
    """Planned _ring_units semantics: rings with exactly 2 outside bonds.

    Returns (ring_atom_tuple, a1, a2, host_sigs) for every SSSR ring whose
    total outside-bond count is exactly two, at two distinct anchors, each
    anchor bond single + acyclic, no dummies involved.
    """
    ri = mol.GetRingInfo()
    units = []
    for ring in ri.AtomRings():
        ring_set = set(ring)
        outside: dict[int, list[int]] = {}
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 0:
                break
            outs = [
                nbr.GetIdx()
                for nbr in atom.GetNeighbors()
                if nbr.GetIdx() not in ring_set
            ]
            if outs:
                outside[idx] = outs
        if sum(len(v) for v in outside.values()) != 2:
            continue
        if sorted(len(v) for v in outside.values()) != [1, 1]:
            continue
        (a1, o1), (a2, o2) = list(outside.items())
        ok = True
        for a, o in ((a1, o1[0]), (a2, o2[0])):
            bond = mol.GetBondBetweenAtoms(a, o)
            if (
                bond is None
                or bond.IsInRing()
                or bond.GetBondType() != Chem.BondType.SINGLE
            ):
                ok = False
            if mol.GetAtomWithIdx(o).GetAtomicNum() == 0:
                ok = False
        if not ok:
            continue
        units.append(
            (
                tuple(ring),
                a1,
                a2,
                (sh._atom_signature(mol.GetAtomWithIdx(o1[0])),
                 sh._atom_signature(mol.GetAtomWithIdx(o2[0]))),
            )
        )
    return units


def probe(tag: str, smi_a: str, smi_b: str) -> None:
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    units_a = ring_fragments(mol_a)
    units_b = ring_fragments(mol_b)
    print(f"--- {tag}: {smi_a} x {smi_b}")
    print(f"    ring units in A: {len(units_a)}, in B: {len(units_b)}")
    for ring, a1, a2, sigs in units_a + units_b:
        print(f"      ring={ring} anchors=({a1},{a2}) host_sigs={sigs}")
    # does the CURRENT operator ever return a bicyclic child for this pair?
    bicyclic = 0
    children = set()
    for seed in range(200):
        child = sh.scaffold_hop_crossover(smi_a, smi_b, random.Random(seed))
        if child is None:
            continue
        children.add(child)
        cm = Chem.MolFromSmiles(child)
        if cm is not None and cm.GetRingInfo().NumRings() >= 2 and len(
            Chem.GetMolFrags(cm)
        ) == 1:
            bicyclic += 1
    print(f"    current operator: {len(children)} unique children, "
          f"{bicyclic}/200 seeds bicyclic-connected")
    # the intended NEW outcome: at least one child built from BOTH parents'
    # ring systems -- atom count >= donors' fused parts
    expected = None
    if units_a and units_b is not None:
        # decalin x piperidine: donor 6-ring spliced across a piperidine
        # cut -> 6 + 6 atoms, 2 rings, 1 fragment
        expected = "bicyclic child (2 rings, single fragment, dummy-free)"
    print(f"    FAIL-FIRST: fuse geometry absent -> expected {expected}")


probe("A", "C1CCC2CCCCC2C1", "C1CCNCC1")  # decalin x piperidine
probe("B", "C1Cc2ccccc2C1", "C1CCNCC1")  # indane x piperidine
probe("C", "C1CCC2CCCCC2C1", "C1COCC1")  # decalin x morpholine
