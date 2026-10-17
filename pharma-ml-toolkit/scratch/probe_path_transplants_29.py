"""100x-29 probe: multi-atom linker path transplants (empirical design).

Validates BEFORE landing:
  1. molzip mechanics for a multi-atom donor [*:1]-p1-..-pn-[*:2] vs an
     opened ring (two new bonds, correct ring expansion).
  2. The planned enumeration (maximal chains -> sublists length 2..4) and
     its determinism, incl. hexane/diethylene glycol pinned unit lists.
  3. The homology matching + FLIPPED orientation reversal (piperazine x
     an amino-ether donor): correct child has no O-N seam bond.
  4. Back-compat: existing pinned pairs (piperidine x CCOCC etc.) keep
     byte-identical pools/children.
  5. Ring-growth caps: piperidine x triethylene glycol never exceeds a
     10-ring (4-atom path cap).
  6. GA-level drift for the notebook configs (generative 6b structural,
     walkthrough 8 seeded+hop).
"""
from __future__ import annotations

import random
import sys

sys.path.insert(0, "src")

from rdkit import Chem  # noqa: E402
from rdkit.rdBase import DisableLog  # noqa: E402

from pharmaml.generative import scaffold_hop as sh  # noqa: E402
from pharmaml.generative.scaffold_hop import (  # noqa: E402
    _atom_signature,
    _cut_into_two,
    _open_ring,
    _zip_child,
    scaffold_hop_crossover,
)

DisableLog("rdApp.error")

# ---------------------------------------------------------------- helpers


def _sig(atom):
    return _atom_signature(atom)


def ring_sizes(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return sorted(len(r) for r in mol.GetRingInfo().AtomRings())


def has_bond_between(smiles, sym1, sym2):
    mol = Chem.MolFromSmiles(smiles)
    for b in mol.GetBonds():
        pair = {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()}
        if pair == {sym1, sym2}:
            return True
    return False


# ------------------------------------------------- planned v29 machinery

_MAX_PATH = 4


def is_chain_atom(mol, atom):
    return (
        atom.GetDegree() == 2
        and not atom.IsInRing()
        and atom.GetAtomicNum() != 0
    )


def walk_maximal_chain(mol, start_idx):
    """Maximal run of chain atoms through start_idx, both directions."""
    start = mol.GetAtomWithIdx(start_idx)
    nbs = list(start.GetNeighbors())
    chain = [start_idx]
    for seed in (nbs[1], nbs[0]):  # second side first, so list reads L->R
        prev, cur = start_idx, seed.GetIdx()
        while is_chain_atom(mol, mol.GetAtomWithIdx(cur)):
            chain.append(cur)
            nxt = [
                n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
                if n.GetIdx() != prev
            ]
            prev, cur = cur, nxt[0]
        if seed is nbs[1]:
            chain.reverse()  # the nbs[1] side becomes the front
    return chain


def excise_linker_path(mol, path):
    """Planned ``_excise_linker_path``: donor + host sigs, or None."""
    try:
        if not path or len(path) > _MAX_PATH:
            return None
        seen = set()
        for idx in path:
            atom = mol.GetAtomWithIdx(idx)
            if not is_chain_atom(mol, atom) or idx in seen:
                return None
            seen.add(idx)
        for i, idx in enumerate(path):
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                if i > 0 and nbr.GetIdx() == path[i - 1]:
                    continue
                if i < len(path) - 1 and nbr.GetIdx() == path[i + 1]:
                    continue
                if len(path) == 1:
                    pass  # n == 1: both neighbours are hosts (checked below)
            for nbr in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond is None or bond.IsInRing():
                    return None
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    return None
        # hosts: n == 1 -> GetNeighbors order; n >= 2 -> outside neighbour
        if len(path) == 1:
            nbrs = list(mol.GetAtomWithIdx(path[0]).GetNeighbors())
            host_first, host_last = nbrs[0], nbrs[1]
        else:
            n0 = [
                n for n in mol.GetAtomWithIdx(path[0]).GetNeighbors()
                if n.GetIdx() != path[1]
            ]
            n1 = [
                n for n in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
                if n.GetIdx() != path[-2]
            ]
            if len(n0) != 1 or len(n1) != 1:
                return None
            host_first, host_last = n0[0], n1[0]
        rw = Chem.RWMol()
        for idx in path:
            anchor = Chem.Atom(mol.GetAtomWithIdx(idx))
            anchor.SetAtomMapNum(0)
            anchor.SetIsotope(0)
            anchor.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            rw.AddAtom(anchor)
        d_first = rw.AddAtom(Chem.Atom(0))
        d_second = rw.AddAtom(Chem.Atom(0))
        for i in range(len(path) - 1):
            rw.AddBond(i, i + 1, Chem.BondType.SINGLE)
        rw.AddBond(0, d_first, Chem.BondType.SINGLE)
        rw.AddBond(len(path) - 1, d_second, Chem.BondType.SINGLE)
        donor = rw.GetMol()
        Chem.SanitizeMol(donor)
        donor.GetAtomWithIdx(d_first).SetAtomMapNum(1)
        donor.GetAtomWithIdx(d_second).SetAtomMapNum(2)
        return donor, (_sig(host_first), _sig(host_last))
    except Exception:
        return None


def linker_path_units(mol, max_atoms=_MAX_PATH):
    """Planned ``_linker_path_units``: [(path_tuple, host_sigs)] (n >= 2)."""
    units = []
    assigned = set()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in assigned or not is_chain_atom(mol, atom):
            continue
        chain = walk_maximal_chain(mol, idx)
        assigned.update(chain)
        if chain[0] > chain[-1]:
            chain.reverse()
        for start_pos in range(len(chain)):
            for length in range(2, min(max_atoms, len(chain) - start_pos) + 1):
                path = tuple(chain[start_pos:start_pos + length])
                excision = excise_linker_path(mol, path)
                if excision is None:
                    continue
                units.append((path, excision[1]))
    return units


def crossover_v29(parent_a, parent_b, rng, *, max_smiles_length=120):
    """Prototype: current pool + transplant_path geometry (planned impl)."""
    if max_smiles_length < sh._MIN_LENGTH_CAP:
        raise ValueError("cap")
    mol_a = Chem.MolFromSmiles(parent_a)
    mol_b = Chem.MolFromSmiles(parent_b)
    if mol_a is None or mol_b is None:
        return None
    if len(Chem.GetMolFrags(mol_a)) != 1 or len(Chem.GetMolFrags(mol_b)) != 1:
        return None
    sig_a = {a.GetIdx(): _sig(a) for a in mol_a.GetAtoms()}
    sig_b = {a.GetIdx(): _sig(a) for a in mol_b.GetAtoms()}

    candidates = []
    for bond_a in mol_a.GetBonds():
        if bond_a.IsInRing() or bond_a.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = bond_a.GetBeginAtomIdx(), bond_a.GetEndAtomIdx()
        for bond_b in mol_b.GetBonds():
            if bond_b.IsInRing() or bond_b.GetBondType() != Chem.BondType.SINGLE:
                continue
            b1, b2 = bond_b.GetBeginAtomIdx(), bond_b.GetEndAtomIdx()
            direct = sig_a[a1] == sig_b[b1] and sig_a[a2] == sig_b[b2]
            flipped = sig_a[a1] == sig_b[b2] and sig_a[a2] == sig_b[b1]
            if direct or flipped:
                candidates.append(("bridge", bond_a.GetIdx(), bond_b.GetIdx(), direct, flipped))

    ring_bonds_a = [
        b for b in mol_a.GetBonds()
        if b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE
    ]
    ring_bonds_b = [
        b for b in mol_b.GetBonds()
        if b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE
    ]
    for bond_a in ring_bonds_a:
        a1, a2 = bond_a.GetBeginAtomIdx(), bond_a.GetEndAtomIdx()
        for bond_b in ring_bonds_b:
            b1, b2 = bond_b.GetBeginAtomIdx(), bond_b.GetEndAtomIdx()
            direct = sig_a[a1] == sig_b[b1] and sig_a[a2] == sig_b[b2]
            flipped = sig_a[a1] == sig_b[b2] and sig_a[a2] == sig_b[b1]
            if direct or flipped:
                candidates.append(("merge", bond_a.GetIdx(), bond_b.GetIdx(), direct))

    for ring_side, other, ring_bonds, sig_ring in (
        ("a", mol_b, ring_bonds_a, sig_a),
        ("b", mol_a, ring_bonds_b, sig_b),
    ):
        for bond in ring_bonds:
            r1, r2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for atom in other.GetAtoms():
                if atom.GetDegree() != 2 or atom.IsInRing():
                    continue
                excision = excise_linker_path(other, (atom.GetIdx(),))
                if excision is None:
                    continue
                _, host_sigs = excision
                direct = host_sigs[0] == sig_ring[r1] and host_sigs[1] == sig_ring[r2]
                flipped = host_sigs[0] == sig_ring[r2] and host_sigs[1] == sig_ring[r1]
                if direct or flipped:
                    candidates.append(("transplant", ring_side, bond.GetIdx(), atom.GetIdx()))
        # NEW: multi-atom path units
        for path, host_sigs in linker_path_units(other):
            for bond in ring_bonds:
                r1, r2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                direct = host_sigs[0] == sig_ring[r1] and host_sigs[1] == sig_ring[r2]
                flipped = host_sigs[0] == sig_ring[r2] and host_sigs[1] == sig_ring[r1]
                if direct or flipped:
                    candidates.append(
                        ("transplant_path", ring_side, bond.GetIdx(), path, direct)
                    )
    if not candidates:
        return None
    choice = candidates[rng.randrange(len(candidates))]

    if choice[0] == "bridge":
        _, ba, bb, direct, flipped = choice
        cut_a = _cut_into_two(mol_a, ba)
        cut_b = _cut_into_two(mol_b, bb)
        if cut_a is None or cut_b is None:
            return None
        fa1, fa2 = cut_a
        fb1, fb2 = cut_b
        pairings = []
        if direct:
            pairings += [(fa1, fb1), (fa2, fb2)]
        if flipped:
            pairings += [(fa1, fb2), (fa2, fb1)]
        rng.shuffle(pairings)
        for fx, fy in pairings:
            child = _zip_child(fx, fy, max_smiles_length)
            if child is not None:
                return child
        return None
    if choice[0] == "merge":
        _, ba, bb, direct = choice
        open_a = _open_ring(mol_a, ba)
        open_b = _open_ring(mol_b, bb)
        if open_a is None or open_b is None:
            return None
        if not direct:
            for dummy in open_b.GetAtoms():
                if dummy.GetAtomicNum() == 0:
                    dummy.SetAtomMapNum(3 - dummy.GetAtomMapNum())
        return _zip_child(open_a, open_b, max_smiles_length)
    if choice[0] == "transplant":
        _, ring_side, bond_idx, atom_idx = choice
        ring_mol, donor_mol = (mol_a, mol_b) if ring_side == "a" else (mol_b, mol_a)
        opened = _open_ring(ring_mol, bond_idx)
        excision = excise_linker_path(donor_mol, (atom_idx,))
        if opened is None or excision is None:
            return None
        return _zip_child(opened, excision[0], max_smiles_length)
    # transplant_path
    _, ring_side, bond_idx, path, direct = choice
    ring_mol, donor_mol = (mol_a, mol_b) if ring_side == "a" else (mol_b, mol_a)
    opened = _open_ring(ring_mol, bond_idx)
    ordered = path if direct else tuple(reversed(path))
    excision = excise_linker_path(donor_mol, ordered)
    if opened is None or excision is None:
        return None
    return _zip_child(opened, excision[0], max_smiles_length)


# ---------------------------------------------------------------- checks
print("=" * 72)
print("1) molzip mechanics: donor [*:1]CO[*:2] into opened piperidine")
opened = _open_ring(Chem.MolFromSmiles("C1CCNCC1"), 0)
donor = excise_linker_path(
    Chem.MolFromSmiles("OCCOCCO"), (2, 3)
)  # C2-O(middle) path in diethylene glycol
print("   donor host sigs:", donor[1])
child = _zip_child(opened, donor[0], 120)
print("   child:", child, "rings:", ring_sizes(child))

print("=" * 72)
print("2) enumeration pins")
hexane = Chem.MolFromSmiles("CCCCCC")
print("   hexane units:")
for path, sigs in linker_path_units(hexane):
    print("    ", path, sigs)
deg = Chem.MolFromSmiles("OCCOCCO")
print("   diethylene glycol units:")
for path, sigs in linker_path_units(deg):
    print("    ", path, sigs)
tri = Chem.MolFromSmiles("OCCOCCOCCO")
print("   triethylene glycol max unit len:", max(
    (len(p) for p, _ in linker_path_units(tri)), default=0))

print("=" * 72)
print("3) ring growth: piperidine x hexane (expect 7- and 8-rings)")
sizes = {}
for seed in range(18):
    c = crossover_v29("C1CCNCC1", "CCCCCC", random.Random(seed))
    sizes.setdefault(tuple(ring_sizes(c)), []).append((seed, c))
for k in sorted(sizes):
    print("   rings", k, "seeds", [s for s, _ in sizes[k]][:4], "e.g.", sizes[k][0][1])
print("   OLD code sizes:", sorted({
    tuple(ring_sizes(scaffold_hop_crossover("C1CCNCC1", "CCCCCC", random.Random(s))))
    for s in range(18)
}))

print("=" * 72)
print("4) piperidine x diethylene glycol (expect 7/8/9)")
sizes = {}
for seed in range(30):
    c = crossover_v29("C1CCNCC1", "OCCOCCO", random.Random(seed))
    sizes.setdefault(tuple(ring_sizes(c)), []).append((seed, c))
for k in sorted(sizes):
    print("   rings", k, "seeds", [s for s, _ in sizes[k]][:3], "e.g.", sizes[k][0][1])

print("=" * 72)
print("5) cap: piperidine x triethylene glycol (no ring > 10)")
big = [
    ring_sizes(crossover_v29("C1CCNCC1", "OCCOCCOCCO", random.Random(s)))
    for s in range(40)
]
print("   observed:", sorted({tuple(x) for x in big}))

print("=" * 72)
print("6) FLIPPED path: piperazine x CNCOCCC")
# find seeds landing on flipped transplant_path candidates
for seed in range(20):
    c = crossover_v29("N1CCNCC1", "CNCOCCC", random.Random(seed))
    if c is None:
        print(f"   seed {seed}: None")
        continue
    print(f"   seed {seed}: {c}  rings={ring_sizes(c)} O-N bond={has_bond_between(c, 'O', 'N')}")

print("=" * 72)
print("7) back-compat: pairs whose pools must stay byte-identical")
for pair in [
    ("C1CCNCC1", "CCOCC"),
    ("CCOCC", "C1CCNCC1"),
    ("C1CCNCC1", "CCSCC"),
    ("C1CCCCC1", "CCNCC"),
    ("C1CCCCC1", "C1CCNCC1"),
    ("C1CCNCC1", "C1COCCN1"),
    ("CC(=O)Oc1ccccc1C(=O)O", "COc1ccc2cc(C(C)C(=O)O)ccc2c1"),
]:
    same = True
    for seed in range(12):
        old = scaffold_hop_crossover(*pair, random.Random(seed))
        new = crossover_v29(*pair, random.Random(seed))
        if old != new:
            same = False
            print("   DRIFT", pair, seed, old, new)
    print("   ", pair, "identical:", same)

print("=" * 72)
print("8) GA drift for the notebook configs")
from pharmaml.generative import (  # noqa: E402
    DEFAULT_CORPUS, GeneticOptimizer, LogPBandObjective,
    MultiObjective, QedObjective,
)

objective = MultiObjective(
    [QedObjective(), LogPBandObjective(1.0, 3.0)], weights=[0.7, 0.3])
seeds_csv = [
    "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CN(C)C(=N)N=C(N)N", "CN1CCCC1c1cccnc1",
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "CN=C(NC#N)NCCSCc1[nH]cnc1C",
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C", "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
]
ga_old = GeneticOptimizer(
    objective, population_size=32, generations=8, seed=42,
    crossover="scaffold_hop").optimize(seed_smiles=seeds_csv)
print("   walkthrough seeded+hop OLD:", len(ga_old), "unique, best", round(ga_old.best_score, 3))

# generative 6b: structural mode, 48/15, SEED from the notebook
SEED = 20250101  # placeholder; real value read from notebook in patcher
ga_old6b = GeneticOptimizer(
    objective, population_size=48, generations=15, seed=SEED,
    mutation="scaffold", crossover="scaffold_hop",
).optimize(seed_smiles=seeds_csv)
print("   generative 6b OLD (seed guess):", len(ga_old6b), "unique, best", round(ga_old6b.best_score, 3))
print("   (exact numbers re-derived after landing via the notebook patcher)")
