"""100x-33 fail-first probe: ring-topology mutations do not exist yet.

Part A shows the intended API is missing (ImportError today).  Part B
hand-rolls the two moves with raw RDKit to validate the chemistry and
pin the expected behaviour the real implementation must reproduce.
"""

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")

# --- Part A: the API does not exist (fail-first) ------------------------
try:
    from pharmaml.generative import ring_topology_moves  # noqa: F401

    print("A: ring_topology_moves EXISTS (unexpected -- probe stale)")
except ImportError as exc:
    print(f"A: ImportError as expected -> {exc}")

try:
    from pharmaml.generative import ring_topology_mutate  # noqa: F401

    print("A: ring_topology_mutate EXISTS (unexpected -- probe stale)")
except ImportError as exc:
    print(f"A: ImportError as expected -> {exc}")

try:
    from pharmaml.generative.genetic import GeneticOptimizer

    ga = GeneticOptimizer(lambda s: 0.0, population_size=4, generations=1,
                          mutation="topology")
    print("A: mutation='topology' ACCEPTED (unexpected -- probe stale)")
except Exception as exc:  # ValueError today: not in ('uniform', 'scaffold')
    print(f"A: mutation='topology' rejected as expected -> {type(exc).__name__}: {exc}")


# --- Part B: hand-rolled chemistry validation ---------------------------
def contract(mol: Chem.Mol, ring: tuple[int, ...], pos: int) -> Chem.Mol | None:
    """Remove ring atom ring[pos], joining its two ring neighbours."""
    n = len(ring)
    atom = ring[pos]
    prev_a, next_a = ring[(pos - 1) % n], ring[(pos + 1) % n]
    if mol.GetBondBetweenAtoms(prev_a, next_a) is not None:
        return None  # chord exists -- removal would shortcut the ring
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(atom)
    lo, hi = (prev_a, next_a) if prev_a < next_a else (next_a, prev_a)
    # indices above the removed atom shift down by one
    lo = lo - 1 if lo > atom else lo
    hi = hi - 1 if hi > atom else hi
    rw.AddBond(lo, hi, Chem.BondType.SINGLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return None


def expand(mol: Chem.Mol, ring: tuple[int, ...], pos: int) -> Chem.Mol | None:
    """Insert one CH2 into the ring bond ring[pos]-ring[pos+1]."""
    n = len(ring)
    a, b = ring[pos], ring[(pos + 1) % n]
    bond = mol.GetBondBetweenAtoms(a, b)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    rw = Chem.RWMol(mol)
    rw.RemoveBond(a, b)
    carbon = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(a, carbon, Chem.BondType.SINGLE)
    rw.AddBond(carbon, b, Chem.BondType.SINGLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return None


def ring_info(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol, mol.GetRingInfo().AtomRings()


def moves_of(smiles: str, min_source: int = 6, max_result: int = 8) -> list[str]:
    """Reference enumeration (contracts + expansions, saturated rings)."""
    mol, rings = ring_info(smiles)
    ri = mol.GetRingInfo()
    out: set[str] = set()
    for ring in rings:
        n = len(ring)
        ring_bonds = [
            mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % n]) for i in range(n)
        ]
        saturated = all(
            b is not None
            and b.GetBondType() == Chem.BondType.SINGLE
            and not b.GetIsAromatic()
            for b in ring_bonds
        )
        if not saturated:
            continue
        if n >= min_source:
            for pos in range(n):
                atom = ring[pos]
                if ri.NumAtomRings(atom) != 1:
                    continue  # fused/spiro atom -- conservative skip
                child = contract(mol, ring, pos)
                if child is not None:
                    out.add(Chem.MolToSmiles(child))
        if n < max_result:
            for pos in range(n):
                bond = mol.GetBondBetweenAtoms(ring[pos], ring[(pos + 1) % n])
                if ri.NumBondRings(bond.GetIdx()) != 1:
                    continue  # shared fused bond -- conservative skip
                child = expand(mol, ring, pos)
                if child is not None:
                    out.add(Chem.MolToSmiles(child))
    return sorted(out)


print("\nB: reference move enumeration")
for smiles in (
    "C1CCNCC1",      # piperidine
    "C1COCCN1",      # morpholine
    "c1ccccc1",      # benzene (aromatic -- no moves)
    "C1=CCCCC1",     # cyclohexene (unsaturated -- no moves)
    "C1CCCCC1",      # cyclohexane
    "N1CCCC1",       # pyrrolidine (5-ring: expand only)
    "N1CCCCCCC1",    # azocane (8-ring: contract only)
    "C1CCC2CCCCC2C1",  # decalin (fused: only non-fused atoms/bonds) (fused: only non-fused atoms/bonds)
):
    print(f"  {smiles:16s} -> {moves_of(smiles)}")

# determinism sanity: the reference enumeration is a pure function
assert moves_of("C1CCNCC1") == moves_of("C1CCNCC1")

# corpus reachability: how many corpus molecules have moves at all?
from pharmaml.generative import DEFAULT_CORPUS

reachable = sum(1 for s in DEFAULT_CORPUS if moves_of(s))
total_moves = sum(len(moves_of(s)) for s in DEFAULT_CORPUS)
print(f"\nB: corpus reachability: {reachable}/{len(DEFAULT_CORPUS)} molecules, "
      f"{total_moves} distinct one-step mutants in total")
