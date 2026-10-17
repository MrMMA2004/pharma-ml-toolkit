"""100x-28 probe: ring-bond cuts for the scaffold-hop crossover.

Two new BREED-style geometries, prototyped standalone before the module
edit:

G2 (ring-merge): cut one SINGLE ring bond in each parent -> each parent
   becomes ONE fragment carrying [*:1] and [*:2] at the cut ends; molzip
   joins map1<->map1 and map2<->map2 -> a single combined ring holding
   atoms from both parents.

G3 (linker transplant): cut one SINGLE ring bond in A (ring-opened, two
   dummies); excise a two-ended linker from B -- a degree-2 atom X whose
   two (acyclic, single-bonded) neighbours n1/n2 are cut off, X keeping
   its own two dummies -- then molzip -> A's ring closed through X:
   ring expansion / homologation with one atom from B.

Checks: molzip multi-pair semantics, sanitisation, canonical SMILES,
ring sizes, determinism.
"""

import random

from rdkit import Chem
from rdkit.rdBase import DisableLog

DisableLog("rdApp.error")


def atom_sig(atom):
    return (
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetIsAromatic(),
        atom.GetFormalCharge(),
    )


def cut_ring_bond(mol, bond_idx):
    """Cut ONE ring bond; mol stays connected; ends get [*:1]/[*:2]."""
    rw = Chem.RWMol(mol)
    bond = rw.GetBondWithIdx(bond_idx)
    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    rw.RemoveBond(a1, a2)
    d1 = rw.AddAtom(Chem.Atom(0))
    d2 = rw.AddAtom(Chem.Atom(0))
    rw.AddBond(a1, d1, Chem.BondType.SINGLE)
    rw.AddBond(a2, d2, Chem.BondType.SINGLE)
    out = rw.GetMol()
    frags = Chem.GetMolFrags(out, asMols=True)
    if len(frags) != 1:
        return None
    frag = frags[0]
    Chem.SanitizeMol(frag)
    dummies = [a for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return None
    # label the dummy sitting on the a1 side map 1, a2 side map 2
    for d in dummies:
        neighbours = d.GetNeighbors()
        assert len(neighbours) == 1
        host = neighbours[0]
        # host idx in frag may differ from mol; match by map-free identity
        d.SetAtomMapNum(0)
    # relabel by host atomic signature order: first dummy in atom order = 1
    order = sorted(
        dummies, key=lambda d: d.GetNeighbors()[0].GetIdx()
    )
    order[0].SetAtomMapNum(1)
    order[1].SetAtomMapNum(2)
    return frag


def excise_atom(mol, atom_idx):
    """Excise degree-2 atom X; return fragment {X + [*:1] + [*:2]}."""
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(atom_idx)
    if atom.GetDegree() != 2:
        return None
    neighbours = [n.GetIdx() for n in atom.GetNeighbors()]
    bonds = [rw.GetBondBetweenAtoms(atom_idx, n) for n in neighbours]
    if any(b is None or b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing()
           for b in bonds):
        return None
    host_sigs = [atom_sig(mol.GetAtomWithIdx(n)) for n in neighbours]
    # cut X loose: dummies cap X only; the host sides are discarded
    for n in neighbours:
        rw.RemoveBond(atom_idx, n)
    d1 = rw.AddAtom(Chem.Atom(0))
    d2 = rw.AddAtom(Chem.Atom(0))
    rw.AddBond(atom_idx, d1, Chem.BondType.SINGLE)
    rw.AddBond(atom_idx, d2, Chem.BondType.SINGLE)
    out = rw.GetMol()
    frags = Chem.GetMolFrags(out, asMols=True)
    donor = None
    for f in frags:
        if any(a.GetAtomicNum() == 0 for a in f.GetAtoms()):
            if sum(1 for a in f.GetAtoms() if a.GetAtomicNum() == 0) == 2:
                donor = f
    if donor is None:
        return None
    Chem.SanitizeMol(donor)
    dummies = [a for a in donor.GetAtoms() if a.GetAtomicNum() == 0]
    for d in dummies:
        d.SetAtomMapNum(0)
    order = sorted(dummies, key=lambda d: d.GetNeighbors()[0].GetIdx())
    order[0].SetAtomMapNum(1)
    order[1].SetAtomMapNum(2)
    return donor, host_sigs


def ring_single_bonds(mol):
    out = []
    for bond in mol.GetBonds():
        if bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
            out.append(bond)
    return out


def g2_merge(parent_a, parent_b):
    mol_a = Chem.MolFromSmiles(parent_a)
    mol_b = Chem.MolFromSmiles(parent_b)
    if mol_a is None or mol_b is None:
        return None
    cands = []
    sig_a = {a.GetIdx(): atom_sig(a) for a in mol_a.GetAtoms()}
    sig_b = {a.GetIdx(): atom_sig(a) for a in mol_b.GetAtoms()}
    for ba in ring_single_bonds(mol_a):
        a1, a2 = ba.GetBeginAtomIdx(), ba.GetEndAtomIdx()
        for bb in ring_single_bonds(mol_b):
            b1, b2 = bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()
            if sig_a[a1] == sig_b[b1] and sig_a[a2] == sig_b[b2]:
                cands.append((ba.GetIdx(), bb.GetIdx()))
    if not cands:
        return None
    ba_idx, bb_idx = cands[0]
    fa = cut_ring_bond(mol_a, ba_idx)
    fb = cut_ring_bond(mol_b, bb_idx)
    if fa is None or fb is None:
        return None
    child = Chem.molzip(fa, fb)
    Chem.SanitizeMol(child)
    return Chem.MolToSmiles(child)


def g3_transplant(parent_a, parent_b, donor_atom_idx=None):
    mol_a = Chem.MolFromSmiles(parent_a)
    mol_b = Chem.MolFromSmiles(parent_b)
    if mol_a is None or mol_b is None:
        return None
    sig_a = {a.GetIdx(): atom_sig(a) for a in mol_a.GetAtoms()}
    # collect donor candidates: degree-2 atoms with 2 acyclic single bonds
    donors = []
    for atom in mol_b.GetAtoms():
        if atom.GetDegree() != 2 or atom.IsInRing():
            continue
        nbrs = atom.GetNeighbors()
        bonds = [mol_b.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()) for n in nbrs]
        if any(b is None or b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing()
               for b in bonds):
            continue
        donors.append(atom.GetIdx())
    # ring cut candidates in A with homologous endpoints to donor flanks
    results = []
    for donor_idx in (donors if donor_atom_idx is None else [donor_atom_idx]):
        exc = excise_atom(mol_b, donor_idx)
        if exc is None:
            continue
        donor_frag, host_sigs = exc
        for ba in ring_single_bonds(mol_a):
            a1, a2 = ba.GetBeginAtomIdx(), ba.GetEndAtomIdx()
            direct = host_sigs[0] == sig_a[a1] and host_sigs[1] == sig_a[a2]
            flipped = host_sigs[0] == sig_a[a2] and host_sigs[1] == sig_a[a1]
            if not (direct or flipped):
                continue
            fa = cut_ring_bond(mol_a, ba.GetIdx())
            if fa is None:
                continue
            try:
                child = Chem.molzip(fa, donor_frag)
                Chem.SanitizeMol(child)
                smi = Chem.MolToSmiles(child)
            except Exception:
                continue
            if smi:
                results.append((donor_idx, ba.GetIdx(), smi))
    return results


print("== G2 ring-merge ==")
for a, b in [
    ("C1CCCCC1", "C1CCNCC1"),      # cyclohexane x piperidine
    ("C1CCNCC1", "C1COCCO1"),      # piperidine x dioxane? (morpholine w/o N)
    ("C1CCOC1", "C1CCNCC1"),       # oxolane-ish x piperidine
    ("c1ccccc1", "C1CCNCC1"),      # aromatic -> excluded by single-bond rule
]:
    try:
        child = g2_merge(a, b)
    except Exception as exc:
        child = f"EXC {type(exc).__name__}: {exc}"
    print(f"  {a} x {b} -> {child}")
    if isinstance(child, str) and not child.startswith("EXC"):
        m = Chem.MolFromSmiles(child)
        if m:
            print(f"    rings: {m.GetRingInfo().AtomRings()} "
                  f"n={m.GetNumAtoms()}")

print()
print("== G3 linker transplant ==")
for a, b in [
    ("C1CCNCC1", "CCOCC"),        # piperidine x diethyl ether -> oxazepane?
    ("C1CCCCC1", "CCOCC"),        # cyclohexane x diethyl ether
    ("C1CCNCC1", "CCSCC"),        # piperidine x thioether
    ("C1CCCCC1", "CCNCC"),        # cyclohexane x diethylamine? N deg2
    ("C1CCOC1", "CCOCC"),         # oxolane x ether
]:
    try:
        res = g3_transplant(a, b)
    except Exception as exc:
        res = f"EXC {type(exc).__name__}: {exc}"
    if isinstance(res, str):
        print(f"  {a} x {b} -> {res}")
        continue
    print(f"  {a} x {b} -> {len(res)} candidates")
    seen = set()
    for donor_idx, bond_idx, smi in res:
        if smi in seen:
            continue
        seen.add(smi)
        m = Chem.MolFromSmiles(smi)
        rings = m.GetRingInfo().AtomRings() if m else None
        print(f"    donor#{donor_idx} bond#{bond_idx}: {smi}  n={m.GetNumAtoms() if m else '?'} rings={[len(r) for r in rings] if rings else '?'}")

print()
print("== molzip multi-pair sanity: benzene opened both sides ==")
fa = cut_ring_bond(Chem.MolFromSmiles("C1CCCCC1"), 0)
fb = cut_ring_bond(Chem.MolFromSmiles("C1CCNCC1"), 0)
if fa and fb:
    print("  fa:", Chem.MolToSmiles(fa), "| fb:", Chem.MolToSmiles(fb))
    child = Chem.molzip(fa, fb)
    Chem.SanitizeMol(child)
    print("  zipped:", Chem.MolToSmiles(child),
          "rings:", [len(r) for r in child.GetRingInfo().AtomRings()])

print()
print("== determinism check (structure-level) ==")
# g3 with fixed donor+bond choice should be a pure function
r1 = g3_transplant("C1CCNCC1", "CCOCC", donor_atom_idx=2)
r2 = g3_transplant("C1CCNCC1", "CCOCC", donor_atom_idx=2)
print("  repeatable:", r1 == r2)
