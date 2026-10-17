"""Ring-topology mutations -- the single-parent medchem moves.

The crossover geometries of :func:`~pharmaml.generative.scaffold_hop_\
crossover` recombine TWO parents; this module is the single-parent
counterpart, the ring-size axis of scaffold morphing as a *mutation*:
one molecule in, one ring-size step out.  Two moves, both SAR 101:

* **contract** -- remove one atom from a saturated ring, joining its
  two ring neighbours directly (piperidine -> pyrrolidine, morpholine
  -> pyrrolidine / tetrahydrofuran / an oxazolidine, decalin ->
  hydrindane);
* **expand** -- insert one methylene into a saturated single ring bond
  (piperidine -> azepane, pyrrolidine -> piperidine, cyclohexane ->
  cycloheptane).

The same conservative fences the crossover obeys apply here: only
*fully saturated* single-bond rings are touched (an aromatic ring
would force the re-zip to invent bond orders; a double bond in the
ring changes unsaturation bookkeeping), only atoms that belong to
exactly one ring are contractible (fused / spiro atoms stay put --
removing one would merge ring systems), only bonds that belong to
exactly one ring are expandable (never a shared fused bond), and the
ring-size window ``[min_ring_size, max_ring_size]`` (default 5..8)
polices both directions -- a 5-ring never contracts below 5, an 8-ring
never expands past 8.  Stereochemistry is not manually repaired: if a
move invalidates a stereocentre the sanitiser simply drops it, and
unsanitisable results are discarded wholesale.

``ring_topology_moves`` is the introspection surface (every distinct
one-step mutant, sorted, deduplicated -- a pure function of the
molecule); ``ring_topology_mutate`` is the GA surface (one move drawn
with the caller's ``random.Random``).  Unparseable or blank input has
no moves; ``ring_topology_mutate`` returns ``None`` whenever no viable
move exists or the chosen mutant would exceed ``max_smiles_length`` --
the caller falls back, never crashes (the package's robustness-first
contract).

Example
-------
>>> from pharmaml.generative import ring_topology_moves
>>> ring_topology_moves("C1CCNCC1")  # piperidine's full one-step move set
['C1CCCC1', 'C1CCCNCC1', 'C1CCNC1']
>>> ring_topology_moves("c1ccccc1")  # aromatic rings are never touched
[]
>>> from pharmaml.generative import ring_topology_mutate
>>> import random
>>> ring_topology_mutate("C1COCCN1", random.Random(0))  # one morpholine move
'C1COCN1'
>>> ring_topology_mutate("CCO", random.Random(0)) is None  # acyclic: no move
True
"""

from __future__ import annotations

import random

from pharmaml.exceptions import DataValidationError

from ._utils import require_rdkit

__all__ = ["ring_topology_moves", "ring_topology_mutate"]


def _validate_window(min_ring_size: int, max_ring_size: int) -> None:
    """Reject absurd ring-size windows with an actionable message."""
    if not 3 <= min_ring_size < max_ring_size <= 12:
        raise DataValidationError(
            f"ring-size window must satisfy 3 <= min_ring_size < max_ring_size"
            f" <= 12, got min={min_ring_size}, max={max_ring_size}"
        )


def _contract(mol: object, ring: tuple[int, ...], pos: int) -> str | None:
    """Remove ``ring[pos]`` and join its two ring neighbours directly.

    Returns the canonical mutant SMILES, or ``None`` when the two
    neighbours are already bonded (a chord -- removal would shortcut
    the ring into a triangle-plus-tail) or the result does not
    sanitise (valence overflows).  Atom-index arithmetic accounts for
    the ``RWMol.RemoveAtom`` shift (indices above the removed atom
    drop by one).
    """
    Chem = require_rdkit()
    size = len(ring)
    atom = ring[pos]
    prev_a = ring[(pos - 1) % size]
    next_a = ring[(pos + 1) % size]
    if mol.GetBondBetweenAtoms(prev_a, next_a) is not None:  # type: ignore[attr-defined]
        return None
    rw = Chem.RWMol(mol)  # type: ignore[arg-type]
    rw.RemoveAtom(atom)
    lo, hi = sorted((prev_a, next_a))
    lo = lo - 1 if lo > atom else lo
    hi = hi - 1 if hi > atom else hi
    rw.AddBond(lo, hi, Chem.BondType.SINGLE)
    mutant = rw.GetMol()
    try:
        Chem.SanitizeMol(mutant)
    except Exception:  # noqa: BLE001 - RDKit raises a family of errors
        return None
    return Chem.MolToSmiles(mutant)


def _expand(mol: object, ring: tuple[int, ...], pos: int) -> str | None:
    """Insert one methylene into the ring bond ``ring[pos]-ring[pos+1]``.

    Returns the canonical mutant SMILES, or ``None`` when the bond is
    not a plain single bond or the result does not sanitise.
    """
    Chem = require_rdkit()
    size = len(ring)
    a = ring[pos]
    b = ring[(pos + 1) % size]
    bond = mol.GetBondBetweenAtoms(a, b)  # type: ignore[attr-defined]
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    rw = Chem.RWMol(mol)  # type: ignore[arg-type]
    rw.RemoveBond(a, b)
    carbon = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(a, carbon, Chem.BondType.SINGLE)
    rw.AddBond(carbon, b, Chem.BondType.SINGLE)
    mutant = rw.GetMol()
    try:
        Chem.SanitizeMol(mutant)
    except Exception:  # noqa: BLE001 - RDKit raises a family of errors
        return None
    return Chem.MolToSmiles(mutant)


def ring_topology_moves(
    smiles: str,
    *,
    min_ring_size: int = 5,
    max_ring_size: int = 8,
) -> list[str]:
    """Every distinct one-step ring-size mutant of ``smiles`` (sorted).

    Enumerates the full contract/expand move set over every qualifying
    saturated ring: contraction candidates are ring atoms that belong
    to exactly one ring (never fused/spiro), expansion candidates are
    ring bonds that belong to exactly one ring (never a shared fused
    bond); unsanitisable results and chord-blocked contractions are
    skipped silently.  The result is deduplicated and sorted -- a pure
    function of the molecule, so callers can pin it.

    Unparseable or blank ``smiles`` yields an empty list (introspection,
    not validation -- this is an operator-family function, robustness
    first).

    :param smiles: any SMILES string (already-canonical input is the
        norm, but any parseable form works).
    :param min_ring_size: smallest ring a move may *produce*
        (contraction fires only on rings of at least this size + 1).
    :param max_ring_size: largest ring a move may *produce*
        (expansion fires only on rings of at most this size - 1).
    :returns: the sorted list of distinct canonical mutant SMILES.
    """
    _validate_window(min_ring_size, max_ring_size)
    Chem = require_rdkit()
    from rdkit.rdBase import DisableLog, EnableLog

    # sanitisation failures are EXPECTED for some candidate moves (that
    # is why they are skipped, not raised) -- keep RDKit's stderr quiet
    # (the canonical_smiles guard-family precedent).
    DisableLog("rdApp.error")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        ri = mol.GetRingInfo()
        found: set[str] = set()
        for ring in ri.AtomRings():
            size = len(ring)
            ring_bonds = [
                mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % size])  # type: ignore[attr-defined]
                for i in range(size)
            ]
            if not all(
                bond is not None
                and bond.GetBondType() == Chem.BondType.SINGLE
                and not bond.GetIsAromatic()
                for bond in ring_bonds
            ):
                continue  # aromatic or unsaturated ring -- never touched
            if size - 1 >= min_ring_size:
                for pos in range(size):
                    if ri.NumAtomRings(ring[pos]) != 1:
                        continue  # fused/spiro atom -- conservative skip
                    mutant = _contract(mol, ring, pos)
                    if mutant is not None:
                        found.add(mutant)
            if size + 1 <= max_ring_size:
                for pos in range(size):
                    bond = mol.GetBondBetweenAtoms(  # type: ignore[attr-defined]
                        ring[pos], ring[(pos + 1) % size]
                    )
                    if bond is None or ri.NumBondRings(bond.GetIdx()) != 1:
                        continue  # shared fused bond -- conservative skip
                    mutant = _expand(mol, ring, pos)
                    if mutant is not None:
                        found.add(mutant)
        return sorted(found)
    finally:
        EnableLog("rdApp.error")


def ring_topology_mutate(
    smiles: str,
    rng: random.Random,
    *,
    max_smiles_length: int = 120,
    min_ring_size: int = 5,
    max_ring_size: int = 8,
) -> str | None:
    """Draw one ring-size mutant of ``smiles`` with ``rng`` (or ``None``).

    The GA-facing wrapper around :func:`ring_topology_moves`: enumerates
    the move set, keeps only mutants within ``max_smiles_length``, and
    draws one uniformly with the caller's ``random.Random``.
    Deterministic given the rng state; ``None`` when the molecule has
    no viable move (acyclic, aromatic-only rings, over-length mutants)
    or does not parse -- the caller (the genetic algorithm's
    ``mutation="topology"`` mode) falls back to character mutation so
    evolution never stalls.

    :param smiles: the parent SMILES (any parseable form).
    :param rng: the caller's ``random.Random`` -- the single source of
        randomness (the package's determinism contract).
    :param max_smiles_length: mutants longer than this are not drawn
        (``None`` when no mutant fits).
    :param min_ring_size: smallest ring a move may produce.
    :param max_ring_size: largest ring a move may produce.
    :returns: the canonical mutant SMILES, or ``None``.
    """
    if max_smiles_length <= 0:
        raise DataValidationError(
            f"max_smiles_length must be positive, got {max_smiles_length}"
        )
    moves = ring_topology_moves(
        smiles, min_ring_size=min_ring_size, max_ring_size=max_ring_size
    )
    viable = [move for move in moves if len(move) <= max_smiles_length]
    if not viable:
        return None
    return viable[rng.randrange(len(viable))]
