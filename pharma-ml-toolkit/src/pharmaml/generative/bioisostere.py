"""Single-atom bioisostere mutations -- the heteroatom-exchange moves.

The crossover geometries of :func:`~pharmaml.generative.scaffold_hop_\
crossover` recombine TWO parents and the ring-size mutations of
:mod:`~pharmaml.generative.topology` reshape rings; this module is the
third single-parent medchem axis, the *heteroatom* axis of scaffold
morphing as a *mutation*: one molecule in, one atom exchanged out.
The classic bioisostere pairs all live here -- an ester O for an amide
N (aspirin -> salicylamide, the swap crossover's celebrated child,
reachable single-parent), an ether O for a thioether S, a ring CH2 for
an N/O/S (piperidine -> piperazine / morpholine / thiomorpholine), an
aryl CH for an N (benzene -> pyridine).

The same conservative fences the crossover obeys apply here: dummies
(``[*:n]`` attachment points and ``*`` wildcards) are never mutated;
aromatic sites only exchange C <-> N (an aryl C -> O/S does not
Kekulize -- probed -- and the sanitisation gate rejects it anyway);
every candidate passes the RDKit sanitisation gate, so
valence-overflowing exchanges are discarded wholesale (a quaternary
C -> N, a carbonyl O -> C saddled with three bonds); and each mutant
is re-canonicalised through a full parse -- ``ReplaceAtom`` can leave
aromatic atoms in an odd ``:N:``-style spelling that only a round-trip
kills (probed).  Stereochemistry is not manually repaired: if an
exchange invalidates a stereocentre the sanitiser simply drops it.

``bioisostere_moves`` is the introspection surface (every distinct
single-atom mutant, sorted, deduplicated -- a pure function of the
molecule); ``bioisostere_mutate`` is the GA surface (one move drawn
with the caller's ``random.Random``).  Unparseable or blank input has
no moves; ``bioisostere_mutate`` returns ``None`` whenever no viable
move exists or the chosen mutant would exceed ``max_smiles_length`` --
the caller falls back, never crashes (the package's robustness-first
contract).

Example
-------
>>> from pharmaml.generative import bioisostere_moves
>>> bioisostere_moves("c1ccccc1")  # benzene's six equivalent aryl C -> one move
['c1ccncc1']
>>> # the 100x-35 swap demo child, now reachable single-parent:
>>> # aspirin's ester O exchanged for an amide N -> salicylamide
>>> "CC(=O)Nc1ccccc1C(=O)O" in bioisostere_moves("CC(=O)Oc1ccccc1C(=O)O")
True
>>> # the methyl ester -> amide exchange of the linker-chemistry axis
>>> "CNC(C)=O" in bioisostere_moves("CC(=O)OC")
True
>>> from pharmaml.generative import bioisostere_mutate
>>> import random
>>> bioisostere_mutate("c1ccccc1", random.Random(0))  # one move: pyridine
'c1ccncc1'
>>> bioisostere_mutate("not-a-smiles", random.Random(0)) is None
True
"""

from __future__ import annotations

import random

from pharmaml.exceptions import DataValidationError

from ._utils import require_rdkit

__all__ = ["bioisostere_moves", "bioisostere_mutate"]

#: Default replacement alphabet -- the four organic-period elements that
#: dominate medchem bioisosteres (C/N/O/S).  Halogens (F/Cl) are accepted
#: through ``allowed`` for callers who want the wider classic set.
_DEFAULT_ALLOWED = ("C", "N", "O", "S")

#: Atomic number of every element the ``allowed`` tuple may name.
_SYMBOL_TO_Z = {"C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17}

#: Aromatic-site fence: only C and N keep a Kekule-consistent aromatic
#: ring (an aryl C -> O/S does not Kekulize -- probed; the sanitisation
#: gate rejects those exchanges anyway, this fence keeps the candidate
#: pool honest).
_AROMATIC_ALLOWED = ("C", "N")


def _validate_allowed(allowed: tuple[str, ...]) -> None:
    """Reject empty or unknown replacement alphabets, actionable message."""
    if not allowed:
        raise DataValidationError(
            "allowed must name at least one element (got an empty tuple)"
        )
    unknown = sorted({symbol for symbol in allowed if symbol not in _SYMBOL_TO_Z})
    if unknown:
        raise DataValidationError(
            f"allowed elements must come from {sorted(_SYMBOL_TO_Z)}, got "
            f"unknown symbol(s) {unknown}"
        )


def bioisostere_moves(
    smiles: str,
    *,
    allowed: tuple[str, ...] = _DEFAULT_ALLOWED,
) -> list[str]:
    """Every distinct single-atom bioisostere mutant of ``smiles`` (sorted).

    Enumerates the full exchange move set over every eligible atom: each
    atom of the molecule is replaced, one at a time, by every element of
    ``allowed`` other than itself.  Dummies (atomic number 0) are never
    mutated; aromatic atoms only exchange C <-> N; every candidate must
    sanitise (valence fences: a quaternary C -> N is discarded) and is
    re-canonicalised through a full parse (kills ``ReplaceAtom``'s odd
    ``:N:`` aromatic spellings).  The result is deduplicated and sorted
    -- a pure function of the molecule, so callers can pin it.

    Unparseable or blank ``smiles`` yields an empty list (introspection,
    not validation -- this is an operator-family function, robustness
    first).

    :param smiles: any SMILES string (already-canonical input is the
        norm, but any parseable form works).
    :param allowed: the replacement alphabet (default ``("C", "N", "O",
        "S")``); each entry must be one of C/N/O/S/F/Cl.
    :returns: the sorted list of distinct canonical mutant SMILES.
    """
    _validate_allowed(allowed)
    Chem = require_rdkit()
    from rdkit.rdBase import DisableLog, EnableLog

    # sanitisation failures are EXPECTED for some candidate exchanges
    # (that is why they are skipped, not raised) -- keep RDKit's stderr
    # quiet (the canonical_smiles guard-family precedent, warning
    # included: the 100x-37 molzip leak taught that lesson).
    DisableLog("rdApp.error")
    DisableLog("rdApp.warning")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        parent = Chem.MolToSmiles(mol)
        found: set[str] = set()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                continue  # attachment dummies / wildcards never mutate
            current = atom.GetSymbol()
            targets = [t for t in allowed if t != current]
            if atom.GetIsAromatic():
                targets = [t for t in targets if t in _AROMATIC_ALLOWED]
            for target in targets:
                rw = Chem.RWMol(mol)  # type: ignore[arg-type]
                rw.ReplaceAtom(atom.GetIdx(), Chem.Atom(_SYMBOL_TO_Z[target]))
                try:
                    mutant = rw.GetMol()
                    Chem.SanitizeMol(mutant)
                    spelling = Chem.MolToSmiles(mutant)
                except Exception:  # noqa: BLE001, S112 — candidate exchanges are expected to fail valence
                    continue
                # round-trip: ReplaceAtom can leave aromatic atoms in a
                # ":N:"-notation spelling that only a re-parse kills
                roundtrip = Chem.MolFromSmiles(spelling)
                if roundtrip is None:
                    continue
                canonical = Chem.MolToSmiles(roundtrip)
                if canonical and canonical != parent:
                    found.add(canonical)
        return sorted(found)
    finally:
        EnableLog("rdApp.error")
        EnableLog("rdApp.warning")


def bioisostere_mutate(
    smiles: str,
    rng: random.Random,
    *,
    max_smiles_length: int = 120,
    allowed: tuple[str, ...] = _DEFAULT_ALLOWED,
) -> str | None:
    """Draw one single-atom bioisostere mutant of ``smiles`` (or ``None``).

    The GA-facing wrapper around :func:`bioisostere_moves`: enumerates
    the move set, keeps only mutants within ``max_smiles_length``, and
    draws one uniformly with the caller's ``random.Random``.
    Deterministic given the rng state; ``None`` when the molecule has
    no viable move (unparseable input, every exchange fenced or
    rejected by sanitisation, over-length mutants) -- the caller (the
    genetic algorithm's ``mutation="bioisostere"`` mode) falls back to
    character mutation so evolution never stalls.

    :param smiles: the parent SMILES (any parseable form).
    :param rng: the caller's ``random.Random`` -- the single source of
        randomness (the package's determinism contract).
    :param max_smiles_length: mutants longer than this are not drawn
        (``None`` when no mutant fits).
    :param allowed: the replacement alphabet (see
        :func:`bioisostere_moves`).
    :returns: the canonical mutant SMILES, or ``None``.
    """
    if max_smiles_length <= 0:
        raise DataValidationError(
            f"max_smiles_length must be positive, got {max_smiles_length}"
        )
    moves = bioisostere_moves(smiles, allowed=allowed)
    viable = [move for move in moves if len(move) <= max_smiles_length]
    if not viable:
        return None
    return viable[rng.randrange(len(viable))]
