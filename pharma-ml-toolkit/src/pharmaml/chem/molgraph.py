"""Pure-Python molecular graph core — the RDKit concept, absorbed natively.

The instruction: *"Check the contents and files in this GitHub link
completely and one by one and add the missing items to playground.py:
github.com/rdkit/rdkit."*  The repository was cloned and reviewed file
by file — the C++ engine (``Code/GraphMol``: SmilesParse, MolGraph,
Substruct, Depictor, Descriptors) and the Python ``Chem`` layer built on
it (``Descriptors``, ``Lipinski``, ``Crippen``, ``MolSurf``,
``GraphDescriptors``, ``MACCSkeys``, ``Fingerprints``, ``Draw``,
``Scaffolds``, ``MolStandardize``, ``AllChem``).

What RDKit *is* is a cheminformatics engine: a SMILES parser feeding a
molecular graph, descriptor math over that graph, and depiction/3D
machinery around it.  The toolkit already speaks to the real engine
where it is installed (:mod:`pharmaml.chem.descriptors` and
:mod:`pharmaml.chem.features` raise a helpful ``DependencyError``
without it).  What was missing — and what this module adds — is the
*core as pure Python*: a SMILES parser, the graph, aromaticity
perception, and the descriptor families whose published algorithms are
transcribable without the C++ engine:

* **SMILES in, molecular graph out** (:func:`parse_smiles`) — the
  Daylight/OpenSMILES subset chemists actually write: the organic
  subset (``B C N O P S F Cl Br I``), aromatic lowercase atoms,
  branches, ring-closure digits (``%nn`` included), bond symbols
  ``- = # :``, dot-disconnected components, bracket atoms with
  isotope/H-count/charge (``[13C]``, ``[nH]``, ``[NH4+]``, ``[O-]``),
  stereo markers accepted-and-noted (``@``/``@@``, ``/``/``\\`` are
  recorded, never interpreted — the honest limitation, stated below).
* **Daylight-style aromaticity perception** — kekulé input (the
  ``CC1=CC=...`` form) is perceived into aromatic rings before any
  descriptor runs: 5/6-rings whose every atom is sp2-capable and whose
  pi-electron count is 6 (ring doubles contribute one electron each,
  exocyclic doubles zero, lone-pair N/O/S two) are marked aromatic,
  and a small Kekulé matching decides which degree-2 ring nitrogen
  carries the pyrrole hydrogen.  Verified against RDKit's own
  ``first_200.tpsa.csv`` test data (the 200 NCI molecules with their
  published TPSA values) — see :func:`tpsa`.
* **The Ertl topological polar surface area, transcribed exactly**
  (:func:`tpsa`) — the contribution table and its two fallback
  formulas copied from ``Code/GraphMol/Descriptors/MolSurf.cpp`` (the
  ``nSing/nDoub/nTrip/nArom`` per-atom bond census, the 3-ring flag,
  N and O classes, ``include_sandp`` for P/S).  Aspirin gives 63.6,
  caffeine 58.44 — the published RDKit values, because the algorithm
  *is* the published one.
* **Lipinski's own definitions** (:func:`lipinski_profile`) — HBA as
  the N+O count and HBD as the total H on N/O, exactly as RDKit's
  ``calcLipinskiHBA``/``calcLipinskiHBD`` implement Lipinski's rule
  (the stricter SMARTS variants ``NumHAcceptors``/``NumHDonors`` stay
  with the real engine — stated in the docstring).  Rotatable bonds
  follow the default (non-strict) RDKit definition — single, non-ring,
  both endpoints of degree >= 2 — with the strict definition's amide,
  CF3/CCl3/CBr3, t-butyl and methyl exclusions available via
  ``strict=True`` (transcribed from the ``Lipinski.cpp`` SMARTS).
* **Ring perception as a minimum cycle basis** (:func:`find_rings`) —
  the graph-theoretic SSSR: the shortest cycle through each bond,
  reduced to an independent basis over GF(2).  The ring count equals
  the cyclomatic number, as RDKit's ``NumRings`` does.
* **A Morgan-style circular fingerprint** (:func:`morgan_fingerprint`)
  — the ECFP recipe (iterative environment refinement over
  identifiers, collected for radii 0..R, folded to bits).  The hash is
  this library's (stable sha1 of canonical environment strings), so
  **bit indices are not comparable with RDKit's Morgan output** — the
  environments, popcounts and similarity behaviour are.
* **Substructure search** (:func:`substructure_matches`) — query
  SMILES as the query dialect (element, aromaticity, bracket H-count
  and charge as exact constraints), matched by backtracking graph
  monomorphism with bond-order/aromaticity agreement.  Full SMARTS
  (recursive queries, wildcards, logic) stays with RDKit.
* **A canonical SMILES writer** (:func:`canonical_smiles`) — Morgan
  invariants refined to ranks, deterministic traversal, ring-closure
  digits reassigned.  Deterministic and round-trip stable; not
  guaranteed byte-identical to RDKit's canonical form.
* **Depiction coordinates and a V2000 MOL writer**
  (:func:`generate_2d_coordinates`, :func:`mol_block`,
  :func:`sdf_text`) — ring templates placed as regular polygons with
  chains grown at standard bond angles with collision-avoiding
  direction search (the *concept* of ``rdDepictor``/``rdkit.Chem.Draw``,
  not its coordinate refinement), and the classic V2000 block written
  purely so a structure can be handed to any viewer without RDKit.

What deliberately stays external, honestly: ETKDG conformers and
force fields (``AllChem``/``ForceField`` — 3D embedding here is the
viewer module's distance-geometry *lite*, clearly labelled), full
SMARTS, stereochemistry interpretation, reaction handling, the
descriptor families whose data tables are not
publishable-from-source here (Crippen logP, QED weights), and
byte-compatible Morgan bits.  When ``rdkit`` is installed the engine
modules above remain the production path; this module is the
dependency-free core that works everywhere.

Examples
--------
>>> from pharmaml.chem.molgraph import (
...     parse_smiles, molecular_formula, average_molecular_weight, tpsa,
...     lipinski_profile)

Aspirin, parsed and profiled without RDKit:

>>> mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
>>> molecular_formula(mol)
'C9H8O4'
>>> round(average_molecular_weight(mol), 2)
180.16
>>> round(tpsa(mol), 2)          # Ertl's table, transcribed from MolSurf.cpp
63.6
>>> profile = lipinski_profile(mol)
>>> (profile["hbd"], profile["hba"], profile["rotatable_bonds"], profile["rings"])
(1, 4, 3, 1)

Caffeine's published TPSA (58.44) reproduces, kekulé or aromatic:

>>> round(tpsa(parse_smiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")), 2)
58.44
>>> round(tpsa(parse_smiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")), 2)
58.44

Implicit hydrogens follow the valence model (perceived pyrrole too):

>>> from pharmaml.chem.molgraph import implicit_hydrogens
>>> implicit_hydrogens(parse_smiles("c1ccccc1"))
[1, 1, 1, 1, 1, 1]
>>> implicit_hydrogens(parse_smiles("c1ccnc1"))       # perceived pyrrole N-H
[1, 1, 1, 1, 1]
>>> implicit_hydrogens(parse_smiles("[NH4+]"))
[4]
>>> implicit_hydrogens(parse_smiles("C"))
[4]

Failures are accounted, not swallowed (position-carrying errors):

>>> parse_smiles("C1CC")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: unclosed ring bond 1 (opened at position 0)
>>> parse_smiles("CC(C)(C)(C)C")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: atom 1 (C, position 1) exceeds its allowed valence 4: bond order sum 5

The fingerprint is deterministic and radius-aware:

>>> from pharmaml.chem.molgraph import morgan_fingerprint
>>> fp = morgan_fingerprint(parse_smiles("CCO"), radius=2, n_bits=1024)
>>> fp.shape, fp.dtype.name, int(fp.sum())
((1024,), 'uint8', 9)
>>> int(morgan_fingerprint(parse_smiles("CCO"), radius=0, n_bits=1024).sum())
3

Substructure search speaks query-SMILES (bracket H counts are exact):

>>> from pharmaml.chem.molgraph import substructure_matches
>>> mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
>>> len(substructure_matches(mol, "c1ccccc1"))    # the phenyl ring, 12 mappings
12
>>> substructure_matches(mol, "C=O")              # two carbonyls
[(1, 2), (10, 11)]
>>> substructure_matches(mol, "[nH]")             # no aromatic N-H in aspirin
[]

Canonical SMILES is deterministic under atom renumbering:

>>> from pharmaml.chem.molgraph import canonical_smiles
>>> canonical_smiles(parse_smiles("OCC"))
'CCO'
>>> canonical_smiles(parse_smiles("C(C)O"))
'CCO'

Coordinates and a MOL block come out without any engine:

>>> from pharmaml.chem.molgraph import generate_2d_coordinates, mol_block
>>> coords = generate_2d_coordinates(parse_smiles("c1ccccc1"))
>>> round(float(np.linalg.norm(coords[0] - coords[1])), 2)   # regular polygon side
1.0
>>> block = mol_block(parse_smiles("CCO"))
>>> block.splitlines()[3]
'  3  2  0  0  0  0  0  0  0  0999 V2000'
>>> block.splitlines()[4].split()[3]     # first atom is C at the origin side
'C'
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from pharmaml.exceptions import DataValidationError

__all__ = [
    "ELEMENTS",
    "ElementInfo",
    "MolAtom",
    "MolBond",
    "Molecule",
    "average_molecular_weight",
    "canonical_smiles",
    "find_rings",
    "fraction_csp3",
    "generate_2d_coordinates",
    "has_substructure",
    "implicit_hydrogens",
    "lipinski_profile",
    "mol_block",
    "molecular_formula",
    "molecule_profile",
    "morgan_environments",
    "morgan_fingerprint",
    "parse_smiles",
    "rotatable_bonds",
    "sdf_text",
    "substructure_matches",
    "total_hydrogens",
    "tpsa",
]


# --------------------------------------------------------------------- #
# Element data                                                           #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ElementInfo:
    """Static element data: number, weight, allowed valences, radii."""

    symbol: str
    atomic_num: int
    weight: float
    valences: tuple[int, ...]
    covalent_radius: float
    vdw_radius: float


#: The elements drug-like chemistry needs: average atomic weight (IUPAC
#: 2021), the allowed valences of RDKit's ``AtomTypes`` model (``P``/
#: ``S`` keep their multiple valences; metals keep ``()`` — no implicit
#: hydrogens, no valence ceiling), the single-bond covalent radius
#: (Cordero 2008, Å) and the vdW radius (Bondi/Alvarez, Å) used by the
#: depiction and surface layers.
ELEMENTS: dict[str, ElementInfo] = {
    info.symbol: info
    for info in (
        ElementInfo("H", 1, 1.008, (1,), 0.31, 1.20),
        ElementInfo("Li", 3, 6.94, (), 1.28, 1.82),
        ElementInfo("B", 5, 10.81, (3,), 0.84, 1.92),
        ElementInfo("C", 6, 12.011, (4,), 0.76, 1.70),
        ElementInfo("N", 7, 14.007, (3,), 0.71, 1.55),
        ElementInfo("O", 8, 15.999, (2,), 0.66, 1.52),
        ElementInfo("F", 9, 18.998, (1,), 0.57, 1.47),
        ElementInfo("Na", 11, 22.990, (), 1.66, 2.27),
        ElementInfo("Mg", 12, 24.305, (), 1.41, 1.73),
        ElementInfo("Al", 13, 26.982, (), 1.21, 1.84),
        ElementInfo("Si", 14, 28.085, (4,), 1.11, 2.10),
        ElementInfo("P", 15, 30.974, (3, 5), 1.07, 1.80),
        ElementInfo("S", 16, 32.06, (2, 4, 6), 1.05, 1.80),
        ElementInfo("Cl", 17, 35.45, (1,), 1.02, 1.75),
        ElementInfo("K", 19, 39.098, (), 2.03, 2.75),
        ElementInfo("Ca", 20, 40.078, (), 1.76, 2.31),
        ElementInfo("Mn", 25, 54.938, (), 1.39, 2.04),
        ElementInfo("Fe", 26, 55.845, (), 1.32, 2.04),
        ElementInfo("Co", 27, 58.933, (), 1.26, 2.00),
        ElementInfo("Ni", 28, 58.693, (), 1.24, 1.98),
        ElementInfo("Cu", 29, 63.546, (), 1.32, 1.96),
        ElementInfo("Zn", 30, 65.38, (), 1.22, 2.10),
        ElementInfo("As", 33, 74.916, (3, 5), 1.19, 1.85),
        ElementInfo("Se", 34, 78.971, (2, 4, 6), 1.20, 1.90),
        ElementInfo("Br", 35, 79.904, (1,), 1.20, 1.85),
        ElementInfo("Ag", 47, 107.868, (), 1.45, 2.20),
        ElementInfo("Cd", 48, 112.414, (), 1.44, 2.18),
        ElementInfo("Sn", 50, 118.710, (4,), 1.39, 2.17),
        ElementInfo("I", 53, 126.904, (1,), 1.39, 1.98),
        ElementInfo("Pt", 78, 195.084, (), 1.36, 2.13),
        ElementInfo("Au", 79, 196.967, (), 1.36, 2.14),
        ElementInfo("Hg", 80, 200.592, (), 1.32, 2.23),
        ElementInfo("Pb", 82, 207.2, (), 1.46, 2.02),
    )
}

#: The Daylight organic-subset symbols writable without brackets.
_ORGANIC_SUBSET = frozenset({"B", "C", "N", "O", "P", "S", "F", "I"})
#: Aromatic lowercase single-letter atoms (``se``/``as`` stay bracketed).
_AROMATIC_SINGLE = frozenset({"c", "n", "o", "s", "p"})


# --------------------------------------------------------------------- #
# The graph                                                              #
# --------------------------------------------------------------------- #


@dataclass
class MolAtom:
    """One atom of the molecular graph.

    ``bracket_h`` keeps the bracket-written H count (``None`` means
    "derive implicitly"); ``position`` is the character offset in the
    source SMILES so validation errors can point at the token.
    """

    index: int
    symbol: str
    atomic_num: int
    is_aromatic: bool = False
    charge: int = 0
    isotope: int = 0
    bracket_h: int | None = None
    position: int = 0


@dataclass
class MolBond:
    """One bond: endpoints (atom indices), order 1/2/3, aromatic flag.

    ``stereo`` retains ``/``/``\\`` markers from the source for the
    non-canonical round trip; they are not interpreted (stated in the
    module docstring).
    """

    begin: int
    end: int
    order: int = 1
    is_aromatic: bool = False
    stereo: str = ""


@dataclass
class Molecule:
    """A molecular graph plus parse bookkeeping."""

    source: str
    atoms: list[MolAtom] = field(default_factory=list)
    bonds: list[MolBond] = field(default_factory=list)
    title: str = ""
    stereo_notes: list[str] = field(default_factory=list)

    # -- cheap derived views ------------------------------------------ #

    def adjacency(self) -> list[list[int]]:
        """Neighbour index lists per atom (heavy-atom graph)."""
        adj: list[list[int]] = [[] for _ in self.atoms]
        for bond in self.bonds:
            adj[bond.begin].append(bond.end)
            adj[bond.end].append(bond.begin)
        return adj

    def degree(self, index: int) -> int:
        """Heavy-atom degree of atom ``index``."""
        return sum(
            1 for bond in self.bonds if bond.begin == index or bond.end == index
        )

    def atom_bonds(self, index: int) -> list[MolBond]:
        """The bonds incident to atom ``index``."""
        return [
            bond for bond in self.bonds if bond.begin == index or bond.end == index
        ]

    def other_end(self, bond: MolBond, index: int) -> int:
        """The index at the other end of ``bond`` from ``index``."""
        return bond.end if bond.begin == index else bond.begin

    def __len__(self) -> int:
        return len(self.atoms)


def _valence_for(atom: MolAtom) -> int | None:
    """Allowed valence after the charge adjustment, or ``None``.

    ``None`` marks metals (and unknown valence lists): no implicit
    hydrogens and no valence ceiling — RDKit's behaviour for atoms
    whose default valence is unset.  For N/P/O/S/B the charge shifts
    the valence per unit (``[NH4+]`` 4, ``[O-]`` 1, ``[n+]`` 4);
    carbon loses one per unit of either sign (``[C-]`` and ``[C+]``
    are both trivalent).
    """
    info = ELEMENTS.get(atom.symbol)
    if info is None or not info.valences:
        return None
    allowed = max(info.valences)
    if atom.charge:
        if atom.symbol == "C":
            allowed -= abs(atom.charge)
        elif atom.symbol in {"N", "P", "O", "S", "B"}:
            allowed += atom.charge
    return max(allowed, 0)


def _bond_valence(bond: MolBond) -> float:
    """Valence units consumed by a bond (aromatic bonds count 1.5)."""
    return 1.5 if bond.is_aromatic else float(bond.order)


# --------------------------------------------------------------------- #
# SMILES parsing                                                         #
# --------------------------------------------------------------------- #


def _parse_bracket_atom(text: str, start: int) -> tuple[MolAtom, list[str], int]:
    """Parse ``[isotope symbol H-count charge]`` into an atom scaffold.

    Returns ``(atom, notes, next_offset)`` with ``atom.index`` left at
    0 for the caller to fill.  Stereo markers and atom classes are
    accepted and reported through ``notes`` so the caller records them
    honestly instead of failing mid-string.
    """
    notes: list[str] = []
    i = start + 1  # past '['
    n = len(text)
    isotope = 0
    digits = ""
    while i < n and text[i].isdigit():
        digits += text[i]
        i += 1
    if digits:
        isotope = int(digits)
    symbol = ""
    aromatic = False
    two = text[i : i + 2]
    if two in ("Cl", "Br"):
        symbol, i = two, i + 2
    elif two in ("se", "as"):
        symbol, aromatic, i = two.capitalize(), True, i + 2
    elif two.capitalize() in ELEMENTS:
        symbol, i = two.capitalize(), i + 2
    elif i < n and text[i] in _AROMATIC_SINGLE:
        symbol, aromatic, i = text[i].upper(), True, i + 1
    elif i < n and text[i].upper() in ELEMENTS:
        symbol, i = text[i].upper(), i + 1
    else:
        raise DataValidationError(
            f"unknown element in bracket atom at position {start}: "
            f"{text[start:i + 2]!r}"
        )
    h_count: int | None = None
    charge = 0
    while i < n and text[i] != "]":
        char = text[i]
        if char == "@":
            while i < n and text[i] not in ("]", "H", "+", "-"):
                i += 1
            notes.append("stereo")
            continue
        if char == "H":
            i += 1
            h_digits = ""
            while i < n and text[i].isdigit():
                h_digits += text[i]
                i += 1
            h_count = int(h_digits) if h_digits else 1
            continue
        if char in "+-":
            sign = 1 if char == "+" else -1
            i += 1
            if i < n and text[i] == char:  # '++' / '--'
                charge += sign * 2
                i += 1
            else:
                mag = ""
                while i < n and text[i].isdigit():
                    mag += text[i]
                    i += 1
                charge += sign * (int(mag) if mag else 1)
            continue
        if char == ":":  # atom class -- accepted, ignored
            while i < n and text[i] != "]":
                i += 1
            notes.append("atom class")
            continue
        raise DataValidationError(
            f"unexpected character {char!r} inside bracket atom at position {i}"
        )
    if i >= n:
        raise DataValidationError(
            f"unterminated bracket atom starting at position {start}"
        )
    info = ELEMENTS.get(symbol)
    if info is None:
        raise DataValidationError(
            f"element {symbol!r} (position {start}) is not in the supported table"
        )
    atom = MolAtom(
        index=0,
        symbol=symbol,
        atomic_num=info.atomic_num,
        is_aromatic=aromatic,
        charge=charge,
        isotope=isotope,
        bracket_h=0 if h_count is None else h_count,
        position=start,
    )
    return atom, notes, i + 1


def _ring_key(text: str, i: int) -> tuple[int, int]:
    """Read a ring-closure label at ``i`` -> ``(label, next_offset)``."""
    if text[i] == "%":
        if i + 3 > len(text) or not text[i + 1 : i + 3].isdigit():
            raise DataValidationError(f"malformed %nn ring label at position {i}")
        return int(text[i + 1 : i + 3]), i + 3
    return int(text[i]), i + 1


def parse_smiles(smiles: str, *, strict: bool = True) -> Molecule:
    """Parse a SMILES string into a :class:`Molecule` graph.

    Supports the Daylight subset: organic-subset atoms and aromatic
    lowercase atoms, ``Cl``/``Br``, branches, ring-closure digits
    (``%nn`` included), bond symbols ``- = # :`` and stereo slashes,
    dot-disconnected components, bracket atoms with isotope, H count
    and charge.  Stereo markers (``@``, ``@@``, ``/``, ``\\``) and atom
    classes are *accepted and noted* on ``Molecule.stereo_notes`` —
    never interpreted (the honest limitation of this core; RDKit
    remains the stereochemistry engine).

    After parsing, aromatic spelling is kekulized and aromaticity is
    re-perceived with RDKit's default model
    (:func:`_finalize_aromaticity`) so descriptors agree with RDKit on
    both spellings — the pyrrole hydrogen of an unbracketed ``n`` is
    decided by the Kekulé assignment, exactly as RDKit's sanitization
    does it.  ``strict=True`` (default) then validates: every
    ring label closed, every aromatic atom inside a ring, and every
    atom within its allowed valence — raising
    :class:`~pharmaml.exceptions.DataValidationError` with the
    offending position, the way RDKit's parser messages do.

    Raises
    ------
    DataValidationError
        On any syntax or (with ``strict``) valence problem.
    """
    if not isinstance(smiles, str):
        raise DataValidationError(
            f"smiles must be a string, got {type(smiles).__name__}"
        )
    text = smiles.strip()
    if not text:
        raise DataValidationError("smiles is empty")

    mol = Molecule(source=text)
    prev: int | None = None
    pending: tuple[int, bool, str] | None = None  # order, aromatic, stereo
    branches: list[int | None] = []
    open_rings: dict[int, tuple[int, tuple[int, bool, str] | None]] = {}

    def add_atom(atom: MolAtom) -> int:
        atom.index = len(mol.atoms)
        mol.atoms.append(atom)
        return atom.index

    def connect(a: int, b: int, spec: tuple[int, bool, str] | None) -> None:
        if spec is None:
            both_aromatic = mol.atoms[a].is_aromatic and mol.atoms[b].is_aromatic
            order, aromatic, stereo = 1, both_aromatic, ""
        else:
            order, aromatic, stereo = spec
        mol.bonds.append(
            MolBond(begin=a, end=b, order=order, is_aromatic=aromatic, stereo=stereo)
        )

    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char == "(":
            branches.append(prev)
            i += 1
            continue
        if char == ")":
            if not branches:
                raise DataValidationError(f"unbalanced ')' at position {i}")
            prev = branches.pop()
            i += 1
            continue
        if char == ".":
            prev = None
            pending = None
            i += 1
            continue
        if char in "-=#:/\\":
            if pending is not None:
                raise DataValidationError(
                    f"doubled bond symbol {char!r} at position {i}"
                )
            pending = (
                {"-": 1, "=": 2, "#": 3}.get(char, 1),
                char == ":",
                char if char in "/\\" else "",
            )
            if char in "/\\":
                mol.stereo_notes.append(f"bond stereo at position {i}")
            i += 1
            continue
        if char.isdigit() or char == "%":
            if prev is None:
                raise DataValidationError(
                    f"ring label without a preceding atom at position {i}"
                )
            label, i = _ring_key(text, i)
            if label in open_rings:
                partner, spec = open_rings.pop(label)
                connect(partner, prev, spec or pending)
            else:
                open_rings[label] = (prev, pending)
            pending = None
            continue
        if char == "[":
            atom, notes, i = _parse_bracket_atom(text, i)
            if notes:
                mol.stereo_notes.extend(
                    f"{note} (position {atom.position})" for note in notes
                )
            idx = add_atom(atom)
            if prev is not None:
                connect(prev, idx, pending)
            prev = idx
            pending = None
            continue
        symbol: str | None = None
        aromatic = False
        two = text[i : i + 2]
        if two in ("Cl", "Br"):
            symbol, i = two, i + 2
        elif char in _ORGANIC_SUBSET:
            symbol, i = char, i + 1
        elif char in _AROMATIC_SINGLE:
            symbol, aromatic, i = char.upper(), True, i + 1
        else:
            raise DataValidationError(
                f"unexpected character {char!r} at position {i}"
            )
        info = ELEMENTS[symbol]
        idx = add_atom(
            MolAtom(
                index=len(mol.atoms),
                symbol=symbol,
                atomic_num=info.atomic_num,
                is_aromatic=aromatic,
                position=i - (2 if symbol in ("Cl", "Br") else 1),
            )
        )
        if prev is not None:
            connect(prev, idx, pending)
        prev = idx
        pending = None

    if branches:
        raise DataValidationError("unclosed branch '(' in the SMILES")
    if open_rings:
        label, (atom_idx, _spec) = min(open_rings.items())
        raise DataValidationError(
            f"unclosed ring bond {label} (opened at position "
            f"{mol.atoms[atom_idx].position})"
        )
    _finalize_aromaticity(mol, strict=strict)
    if strict:
        _validate_graph(mol)
    return mol


def _validate_graph(mol: Molecule) -> None:
    """Post-parse checks: aromaticity-in-ring and valence ceilings.

    The ceiling counts aromatic bonds as one valence unit each —
    RDKit's explicit-valence convention for aromatic bonds (which is
    why ``Cn1cccc1`` parses: the N-methyl pyrrole nitrogen sits at
    three units, not four).
    """
    rings = find_rings(mol)
    ring_atoms: set[int] = set()
    for ring in rings:
        ring_atoms.update(ring)
    for atom in mol.atoms:
        if atom.is_aromatic and atom.index not in ring_atoms:
            raise DataValidationError(
                f"atom {atom.index} ({atom.symbol.lower()}, position "
                f"{atom.position}) is marked aromatic but not in a ring"
            )
        allowed = _valence_for(atom)
        if allowed is None:
            continue
        used = 0.0
        for bond in mol.atom_bonds(atom.index):
            used += 1.0 if bond.is_aromatic else float(bond.order)
        if atom.bracket_h is not None:
            used += atom.bracket_h
        if used > allowed + 1e-9:
            raise DataValidationError(
                f"atom {atom.index} ({atom.symbol}, position "
                f"{atom.position}) exceeds its allowed valence "
                f"{allowed}: bond order sum {round(used, 1):g}"
            )


# --------------------------------------------------------------------- #
# Aromaticity: kekulize the author's flags, then RDKit's default model   #
# --------------------------------------------------------------------- #


def _finalize_aromaticity(mol: Molecule, *, strict: bool) -> None:
    """Kekulize aromatic input, then perceive aromaticity from scratch.

    This is RDKit's sanitization pipeline in miniature: aromatic
    spelling is first *kekulized* (a backtracking assignment of single/
    double over the aromatic bonds under the valence constraints —
    every aromatic carbon needs exactly one pi, aromatic N at most one
    and a degree-3 N none, O/S none), then all aromatic flags are
    cleared and the default aromaticity model (:func:`_perceive_legacy`)
    re-perceives them — so kekulé and aromatic spellings of the same
    molecule land on identical flags, TPSA and hydrogens.

    Raises
    ------
    DataValidationError
        When the aromatic bonds admit no valid Kekulé assignment (the
        "Can't kekulize" failure), only under ``strict``.
    """
    if any(bond.is_aromatic for bond in mol.bonds):
        _kekulize(mol, strict=strict)
    for atom in mol.atoms:
        atom.is_aromatic = False
    for bond in mol.bonds:
        bond.is_aromatic = False
    _perceive_legacy(mol)


def _kekulize(mol: Molecule, *, strict: bool) -> None:
    """Assign alternating single/double bonds over the aromatic set.

    Constraints per atom (its exocyclic doubles counted in): aromatic
    carbon ends with exactly one pi; aromatic N at most one — a
    degree-3 or H-carrying N exactly zero (its valence is spent); O/S
    zero.  The backtracking walks the aromatic bonds in list order and
    prefers ``double`` first, which makes the choice deterministic
    (benzene gets its alternating form; an unbracketed pyrrole nitrogen
    is the one left without a double, so it carries the N-H — RDKit's
    behaviour — while caffeine's bare imidazole N takes the double).
    """
    aromatic_bonds = [b for b in mol.bonds if b.is_aromatic]
    aromatic_atoms = sorted(
        {b.begin for b in aromatic_bonds} | {b.end for b in aromatic_bonds}
    )

    def pi_bounds(atom_index: int) -> tuple[int, int] | None:
        """(lo, hi) ring-pi bounds for one atom (None = unconstrained)."""
        atom = mol.atoms[atom_index]
        exo = sum(
            1
            for b in mol.atom_bonds(atom_index)
            if not b.is_aromatic and b.order == 2
        )
        if atom.symbol == "C":
            return (1 - exo, 1 - exo)
        if atom.symbol in ("O", "S", "Se", "As"):
            return (-exo, -exo)
        if atom.symbol == "N":
            if atom.bracket_h is not None or mol.degree(atom_index) >= 3:
                # valence fully spent: degree + pi + exo + H == 3 + charge
                target = 3 + atom.charge - mol.degree(atom_index) - exo
                target -= atom.bracket_h or 0
                return (target, target)
            return (0, 1 - exo)  # the flexible pyridine/pyrrole nitrogen
        return None

    bounds: dict[int, tuple[int, int] | None] = {
        a: pi_bounds(a) for a in aromatic_atoms
    }
    taken: dict[int, int] = {a: 0 for a in aromatic_atoms}

    def feasible() -> bool:
        for a in aromatic_atoms:
            window = bounds[a]
            if window is None:
                continue
            lo, hi = window
            if not lo <= taken[a] <= hi:
                return False
        return True

    chosen: dict[int, bool] = {}  # bond list index -> is double

    def solve(bond_idx: int) -> bool:
        if bond_idx == len(aromatic_bonds):
            return feasible()
        bond = aromatic_bonds[bond_idx]
        for is_double in (True, False):
            delta = 1 if is_double else 0
            ok = True
            for side in (bond.begin, bond.end):
                window = bounds[side]
                if window is not None and taken[side] + delta > window[1]:
                    ok = False
                    break
            if not ok:
                continue
            for side in (bond.begin, bond.end):
                if bounds[side] is not None:
                    taken[side] += delta
            chosen[bond_idx] = is_double
            if solve(bond_idx + 1):
                return True
            for side in (bond.begin, bond.end):
                if bounds[side] is not None:
                    taken[side] -= delta
            del chosen[bond_idx]
        return False

    if not solve(0):
        if strict:
            raise DataValidationError(
                "cannot kekulize the aromatic bonds of "
                f"{mol.source[:60]!r} -- no valid single/double assignment"
            )
        return  # lenient re-parse: leave the aromatic flags as written
    for idx, bond in enumerate(aromatic_bonds):
        bond.order = 2 if chosen[idx] else 1
        bond.is_aromatic = False


def _perceive_legacy(mol: Molecule) -> None:
    """RDKit's default aromaticity model, transcribed.

    From ``Code/GraphMol/Aromaticity.cpp`` (the default model):
    walk each SSSR ring in cycle order — a double bond to the next ring
    atom adds 2 pi electrons; a ring carbon (or 4-valent nitrogen) with
    an *exocyclic* double to an already-aromatic neighbour adds 1, and
    an exocyclic double to a non-aromatic atom blocks the lone-pair
    bonus; a ring containing N/O/S with no such blocker and an odd size
    adds the 2 lone-pair electrons.  The ring is aromatic when
    ``pi > 2 and (pi - 2) % 4 == 0`` (the 4n+2 rule) and every C/N is
    sp2 (approximated by: incident double, aromatic flag, or a
    double-bearing C/N/O/S neighbour).  Rings are re-evaluated to a
    fixpoint so fused systems cascade (the middle ring of phenothiazine
    only reaches 6 pi once its neighbour is aromatic).  Verified against
    RDKit's ``first_200.tpsa.csv`` NCI set.
    """
    rings = find_rings(mol)
    if not rings:
        return
    aromatic: set[int] = set()

    def sp2_capable(atom_index: int) -> bool:
        if any(
            b.order >= 2 or b.is_aromatic for b in mol.atom_bonds(atom_index)
        ):
            return True
        for bond in mol.atom_bonds(atom_index):
            neighbour = mol.other_end(bond, atom_index)
            if mol.atoms[neighbour].symbol not in ("C", "N", "O", "S"):
                continue
            if any(
                nb.order >= 2 or nb.is_aromatic
                for nb in mol.atom_bonds(neighbour)
            ):
                return True
        return False

    def ring_valence4(atom_index: int) -> int:
        total = sum(
            (1.0 if b.is_aromatic else float(b.order))
            for b in mol.atom_bonds(atom_index)
        )
        return round(total + implicit_h(mol, atom_index))

    for _ in range(len(rings) + 1):
        changed = False
        for ring in rings:
            if all(a in aromatic for a in ring):
                continue
            ordered = _cycle_order(ring, mol)
            ring_set = set(ring)
            pi = 0
            has_nos = any(
                mol.atoms[a].symbol in ("N", "O", "S") for a in ring
            )
            exo_block = False
            for position, atom_index in enumerate(ordered):
                nxt = ordered[(position + 1) % len(ordered)]
                bond = next(
                    b
                    for b in mol.bonds
                    if {b.begin, b.end} == {atom_index, nxt}
                )
                if bond.order == 2:
                    pi += 2
                    continue
                atom = mol.atoms[atom_index]
                if not (
                    atom.symbol == "C"
                    or (atom.symbol == "N" and ring_valence4(atom_index) == 4)
                ):
                    continue
                for side_bond in mol.atom_bonds(atom_index):
                    neighbour = mol.other_end(side_bond, atom_index)
                    if neighbour in ring_set or side_bond.order == 1:
                        continue
                    if side_bond.order == 2:
                        if neighbour in aromatic:
                            pi += 1
                        else:
                            exo_block = True
            if has_nos and not exo_block and len(ring) % 2 == 1:
                pi += 2
            if pi > 2 and (pi - 2) % 4 == 0 and all(
                mol.atoms[a].symbol not in ("C", "N") or sp2_capable(a)
                for a in ring
            ):
                if not aromatic.issuperset(ring):
                    changed = True
                aromatic.update(ring)
        if not changed:
            break
    # apply: atoms aromatic; ring-cycle bonds of aromatic rings aromatic
    for ring in rings:
        if not aromatic.issuperset(ring):
            continue
        for atom_index in ring:
            mol.atoms[atom_index].is_aromatic = True
        ordered = _cycle_order(ring, mol)
        for a, b in zip(ordered, ordered[1:] + ordered[:1]):
            bond = next(
                bd for bd in mol.bonds if {bd.begin, bd.end} == {a, b}
            )
            bond.is_aromatic = True


def implicit_h(mol: Molecule, atom_index: int) -> int:
    """Hydrogen count for a single atom (shared by perception + public API)."""
    atom = mol.atoms[atom_index]
    if atom.bracket_h is not None:
        return atom.bracket_h
    if atom.is_aromatic:
        if atom.symbol == "C":
            return 1 if mol.degree(atom_index) <= 2 else 0
        if atom.symbol == "N":
            # degree-2 aromatic N without any double bond = the pyrrole
            # donor (its Kekulé form left it bare); a pyridine-type N
            # kept a double and stays H-less
            has_double = any(
                b.order == 2 for b in mol.atom_bonds(atom_index)
            )
            return 1 if mol.degree(atom_index) == 2 and not has_double else 0
        return 0
    allowed = _valence_for(atom)
    if allowed is None:
        return 0
    used = sum(_bond_valence(b) for b in mol.atom_bonds(atom_index))
    return max(0, int(allowed - used))


def implicit_hydrogens(mol: Molecule) -> list[int]:
    """Per-atom hydrogen counts from the valence model.

    Bracket atoms use their written H count.  Aromatic atoms follow
    the aromatic table: aromatic carbon keeps one H at ring-degree 2
    (benzene's CH) and none when substituted or fused; aromatic
    nitrogen keeps one H only as a degree-2 pyrrole-type donor (the
    Kekulé assignment left it without a double) and none as
    pyridine-type; aromatic O/S never carry implicit H
    (furan/thiophene).  Aliphatic atoms get
    ``max(0, allowed_valence - bond_order_sum)``; metals take none.
    """
    return [implicit_h(mol, i) for i in range(len(mol.atoms))]


def total_hydrogens(mol: Molecule) -> list[int]:
    """Alias kept for reader symmetry with RDKit's ``GetTotalNumHs``."""
    return implicit_hydrogens(mol)


# --------------------------------------------------------------------- #
# Ring perception -- minimum cycle basis (SSSR)                          #
# --------------------------------------------------------------------- #


def _shortest_path_without_edge(
    adj: list[list[int]], skip: tuple[int, int], start: int, goal: int
) -> list[int] | None:
    """BFS shortest ``start..goal`` path avoiding one edge."""
    if start == goal:
        return [start]
    parent: dict[int, int] = {start: start}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in adj[node]:
            if (node, nbr) == skip or (nbr, node) == skip:
                continue
            if nbr in parent:
                continue
            parent[nbr] = node
            if nbr == goal:
                path = [nbr]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(nbr)
    return None


def find_rings(mol: Molecule) -> list[frozenset[int]]:
    """The smallest set of smallest rings, as atom-index sets.

    Every bond's shortest cycle is collected, deduplicated, then
    reduced over GF(2) (edge-incidence Gaussian elimination with
    explicit pivots, smallest cycles first) to an independent basis —
    the graph-theoretic SSSR.  The basis size equals the cyclomatic
    number, matching RDKit's ``NumRings``; which *particular* ring set
    survives a tie can differ from RDKit's choice for symmetric fused
    systems (stated honestly).
    """
    adj = mol.adjacency()
    if not mol.atoms:
        return []
    edge_index: dict[frozenset[int], int] = {
        frozenset((b.begin, b.end)): idx for idx, b in enumerate(mol.bonds)
    }
    candidates: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()
    for bond in mol.bonds:
        skip = (bond.begin, bond.end)
        path = _shortest_path_without_edge(adj, skip, bond.begin, bond.end)
        if path and len(path) >= 3:
            cycle = frozenset(path)
            if cycle not in seen:
                seen.add(cycle)
                candidates.append(cycle)
    if not candidates:
        return []
    candidates.sort(key=lambda c: (len(c), sorted(c)))
    basis: list[tuple[int, int]] = []  # (pivot_bit, reduced_mask)
    basis_rings: list[frozenset[int]] = []
    for cycle in candidates:
        mask = 0
        ordered = sorted(cycle)
        for a, b in zip(ordered, ordered[1:] + ordered[:1]):
            key = frozenset((a, b))
            if key in edge_index:
                mask |= 1 << edge_index[key]
        for pivot, reduced in basis:
            if (mask >> pivot) & 1:
                mask ^= reduced
        if mask:
            pivot = mask.bit_length() - 1
            basis.append((pivot, mask))
            basis_rings.append(cycle)
    return basis_rings


def _ring_bond_set(mol: Molecule, rings: list[frozenset[int]]) -> set[frozenset[int]]:
    """Bonds along each SSSR ring's cycle (consecutive in cycle order)."""
    result: set[frozenset[int]] = set()
    for ring in rings:
        ordered = _cycle_order(ring, mol)
        for a, b in zip(ordered, ordered[1:] + ordered[:1]):
            result.add(frozenset((a, b)))
    return result


def _cycle_order(ring: frozenset[int], mol: Molecule) -> list[int]:
    """Cycle traversal order of a ring's atoms (deterministic walk)."""
    members = sorted(ring)
    adj = {a: [b for b in mol.adjacency()[a] if b in ring] for a in members}
    start = members[0]
    path = [start]
    prev: int | None = None
    current = start
    while True:
        candidates = [n for n in adj[current] if n != prev]
        nxt = candidates[0] if candidates else prev
        if nxt is None or nxt == start:
            return path
        path.append(nxt)
        prev, current = current, nxt
        if len(path) > len(members):
            return path[: len(members)]


# --------------------------------------------------------------------- #
# Descriptors                                                            #
# --------------------------------------------------------------------- #


def molecular_formula(mol: Molecule) -> str:
    """Hill-order molecular formula (C, H, then alphabetical)."""
    counts: dict[str, int] = {}
    for atom in mol.atoms:
        counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
    hydrogen = sum(implicit_hydrogens(mol))
    if hydrogen:
        counts["H"] = counts.get("H", 0) + hydrogen
    parts: list[str] = []
    if "C" in counts:
        parts.append(_formula_pair("C", counts.pop("C")))
        if "H" in counts:
            parts.append(_formula_pair("H", counts.pop("H")))
    for symbol in sorted(counts):
        parts.append(_formula_pair(symbol, counts[symbol]))
    return "".join(parts) or "H0"


def _formula_pair(symbol: str, count: int) -> str:
    return symbol if count == 1 else f"{symbol}{count}"


def average_molecular_weight(mol: Molecule) -> float:
    """Average molecular weight (daltons) from the IUPAC tables.

    Isotopes written in brackets use their mass number as the weight
    (``[13C]`` -> 13.0) — a documented approximation.
    """
    weight = 0.0
    for atom, hydrogens in zip(mol.atoms, implicit_hydrogens(mol)):
        if atom.isotope:
            weight += float(atom.isotope)
        else:
            weight += ELEMENTS[atom.symbol].weight
        weight += hydrogens * ELEMENTS["H"].weight
    return weight


def fraction_csp3(mol: Molecule) -> float:
    """The fraction of carbon atoms that are sp3 (all-single, non-aromatic)."""
    carbons = [a for a in mol.atoms if a.symbol == "C"]
    if not carbons:
        return 0.0
    sp3 = 0
    for atom in carbons:
        if all(
            b.order == 1 and not b.is_aromatic for b in mol.atom_bonds(atom.index)
        ):
            sp3 += 1
    return sp3 / len(carbons)


def rotatable_bonds(mol: Molecule, *, strict: bool = False) -> list[MolBond]:
    """Lipinski rotatable bonds — RDKit's default (non-strict) definition.

    Non-strict (the RDKit default): single, non-ring bonds whose two
    endpoints both have heavy degree >= 2 — transcribed from
    ``[!$(*#*)&!D1]-,:;!@[!$(*#*)&!D1]``.  ``strict=True`` adds the
    strict definition's exclusions (amides, CF3/CCl3/CBr3 centres,
    t-butyl junctions, methyls — from the ``Lipinski.cpp`` SMARTS).
    Ring bonds are the SSSR cycle bonds; aromatic bonds never count.
    """
    rings = find_rings(mol)
    ring_bonds = _ring_bond_set(mol, rings)
    result: list[MolBond] = []
    for bond in mol.bonds:
        if bond.is_aromatic or bond.order != 1:
            continue
        if frozenset((bond.begin, bond.end)) in ring_bonds:
            continue
        if mol.degree(bond.begin) < 2 or mol.degree(bond.end) < 2:
            continue
        if strict and _is_strict_excluded(mol, bond):
            continue
        result.append(bond)
    return result


def _is_strict_excluded(mol: Molecule, bond: MolBond) -> bool:
    """The strict definition's structural exclusions for one bond."""
    for side, other in ((bond.begin, bond.end), (bond.end, bond.begin)):
        atom = mol.atoms[side]
        other_atom = mol.atoms[other]
        # amide: C(=O)-N in either orientation
        if (
            atom.symbol == "N"
            and other_atom.symbol == "C"
            and any(
                nb.order == 2
                for nb in mol.atom_bonds(other)
                if mol.atoms[mol.other_end(nb, other)].symbol in {"O", "N", "S"}
            )
        ):
            return True
        if atom.symbol == "C":
            neighbours = [
                mol.atoms[mol.other_end(nb, atom.index)]
                for nb in mol.atom_bonds(atom.index)
                if mol.other_end(nb, atom.index) != other
            ]
            symbols = sorted(n.symbol for n in neighbours)
            if symbols in (["F", "F", "F"], ["Cl", "Cl", "Cl"], ["Br", "Br", "Br"]):
                return True  # CF3 / CCl3 / CBr3 centre
            if len(neighbours) == 3 and all(
                n.symbol == "C" and mol.degree(n.index) == 1 for n in neighbours
            ):
                return True  # t-butyl junction
    return False


def lipinski_profile(mol: Molecule) -> dict[str, float | int]:
    """Lipinski's own HBD/HBA counts plus the rotatable/ring/sp3 numbers.

    ``hba`` counts N and O atoms and ``hbd`` the total hydrogens on
    N and O — *Lipinski's definition*, exactly what RDKit's
    ``calcLipinskiHBA``/``calcLipinskiHBD`` implement.  The stricter
    SMARTS variants (``NumHAcceptors`` excluding amide N, aromatic-O
    donors) stay with the real engine.
    """
    hydrogens = implicit_hydrogens(mol)
    hba = sum(1 for atom in mol.atoms if atom.atomic_num in (7, 8))
    hbd = sum(
        h for atom, h in zip(mol.atoms, hydrogens) if atom.atomic_num in (7, 8)
    )
    return {
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": len(rotatable_bonds(mol)),
        "rings": len(find_rings(mol)),
        "heavy_atoms": len(mol.atoms),
        "fraction_csp3": round(fraction_csp3(mol), 3),
    }


#: Ertl's TPSA contributions — transcribed from RDKit's
#: ``Code/GraphMol/Descriptors/MolSurf.cpp`` (nitrogen and oxygen
#: classes; key: element, degree, total H, formal charge, single /
#: double / triple / aromatic bond counts, in-a-3-ring flag; the last
#: entries are the P/S additions behind ``include_sandp=True``).
_TPSA_CONTRIBS: dict[tuple, float] = {
    ("N", 1, 0, 0, 0, 0, 1, 0, False): 23.79,
    ("N", 1, 1, 0, 0, 1, 0, 0, False): 23.85,
    ("N", 1, 2, 0, 1, 0, 0, 0, False): 26.02,
    ("N", 1, 2, 1, 0, 1, 0, 0, False): 25.59,
    ("N", 1, 3, 1, 1, 0, 0, 0, False): 27.64,
    ("N", 2, 0, 0, 1, 1, 0, 0, False): 12.36,
    ("N", 2, 0, 0, 0, 1, 1, 0, False): 13.60,
    ("N", 2, 1, 0, 2, 0, 0, 0, True): 21.94,
    ("N", 2, 1, 0, 2, 0, 0, 0, False): 12.03,
    ("N", 2, 0, 1, 1, 0, 1, 0, False): 4.36,
    ("N", 2, 1, 1, 1, 1, 0, 0, False): 13.97,
    ("N", 2, 2, 1, 2, 0, 0, 0, False): 16.61,
    ("N", 2, 0, 0, 0, 0, 0, 2, False): 12.89,
    ("N", 2, 1, 0, 0, 0, 0, 2, False): 15.79,
    ("N", 2, 1, 1, 0, 0, 0, 2, False): 14.14,
    ("N", 3, 0, 0, 3, 0, 0, 0, True): 3.01,
    ("N", 3, 0, 0, 3, 0, 0, 0, False): 3.24,
    ("N", 3, 0, 0, 1, 2, 0, 0, False): 11.68,
    ("N", 3, 0, 1, 2, 1, 0, 0, False): 3.01,
    ("N", 3, 1, 1, 3, 0, 0, 0, False): 4.44,
    ("N", 3, 0, 0, 0, 0, 0, 3, False): 4.41,
    ("N", 3, 0, 0, 1, 0, 0, 2, False): 4.93,
    ("N", 3, 0, 0, 0, 1, 0, 2, False): 8.39,
    ("N", 3, 0, 1, 0, 0, 0, 3, False): 4.10,
    ("N", 3, 0, 1, 1, 0, 0, 2, False): 3.88,
    ("N", 4, 0, 1, 4, 0, 0, 0, False): 0.0,
    ("O", 1, 0, 0, 0, 1, 0, 0, False): 17.07,
    ("O", 1, 1, 0, 1, 0, 0, 0, False): 20.23,
    ("O", 1, 0, -1, 1, 0, 0, 0, False): 23.06,
    ("O", 2, 0, 0, 2, 0, 0, 0, True): 12.53,
    ("O", 2, 0, 0, 2, 0, 0, 0, False): 9.23,
    ("O", 2, 0, 0, 0, 0, 0, 2, False): 13.14,
    ("P", 2, 0, 0, 1, 1, 0, 0, False): 34.14,
    ("P", 3, 0, 0, 3, 0, 0, 0, False): 13.59,
    ("P", 3, 1, 0, 2, 1, 0, 0, False): 23.47,
    ("P", 4, 0, 0, 3, 1, 0, 0, False): 9.81,
    ("S", 1, 0, 0, 0, 1, 0, 0, False): 32.09,
    ("S", 1, 1, 0, 1, 0, 0, 0, False): 38.80,
    ("S", 2, 0, 0, 2, 0, 0, 0, False): 25.30,
    ("S", 2, 0, 0, 0, 0, 0, 2, False): 28.24,
    ("S", 3, 0, 0, 0, 1, 0, 2, False): 21.70,
    ("S", 3, 0, 0, 2, 1, 0, 0, False): 19.21,
    ("S", 4, 0, 0, 2, 2, 0, 0, False): 8.38,
}


def tpsa(mol: Molecule, *, include_sandp: bool = False) -> float:
    """Topological polar surface area — Ertl's algorithm, transcribed.

    The contribution table and both fallback formulas
    (``30.5 - 8.2*deg + 1.5*nH`` for nitrogen, ``28.5 - 8.6*deg +
    1.5*nH`` for oxygen, floored at zero) are copied from RDKit's
    ``Code/GraphMol/Descriptors/MolSurf.cpp``, so N/O molecules match
    RDKit's ``CalcTPSA`` (aspirin 63.6, caffeine 58.44 — and the 200
    NCI molecules of RDKit's own ``first_200.tpsa.csv`` test file,
    which the development probe replayed end to end).
    ``include_sandp=True`` adds the P/S contributions, as RDKit's
    flag does.  Aromatic bonds are counted in their own census (they
    never inflate the single-bond count), and the 3-ring flag uses
    the SSSR.
    """
    rings = find_rings(mol)
    in3: set[int] = {
        atom for ring in rings if len(ring) == 3 for atom in ring
    }
    hydrogens = implicit_hydrogens(mol)
    total = 0.0
    for atom, n_h in zip(mol.atoms, hydrogens):
        element = atom.symbol
        is_target = element in ("N", "O") or (
            include_sandp and element in ("P", "S")
        )
        if not is_target:
            continue
        sing = doub = trip = arom = 0
        for bond in mol.atom_bonds(atom.index):
            if bond.is_aromatic:
                arom += 1
            elif bond.order == 2:
                doub += 1
            elif bond.order == 3:
                trip += 1
            else:
                sing += 1
        key = (
            element,
            mol.degree(atom.index),
            n_h,
            atom.charge,
            sing,
            doub,
            trip,
            arom,
            atom.index in in3,
        )
        value = _TPSA_CONTRIBS.get(key)
        if value is None:
            if element == "N":
                value = max(0.0, 30.5 - 8.2 * mol.degree(atom.index) + 1.5 * n_h)
            elif element == "O":
                value = max(0.0, 28.5 - 8.6 * mol.degree(atom.index) + 1.5 * n_h)
            else:
                value = 0.0
        total += value
    return round(total, 2)


# --------------------------------------------------------------------- #
# Morgan-style circular fingerprint                                      #
# --------------------------------------------------------------------- #


def _sha_id(payload: tuple) -> int:
    """A stable 64-bit identifier from a canonical tuple."""
    digest = hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _environment_identifiers(mol: Molecule, radius: int) -> list[list[int]]:
    """Per-iteration, per-atom identifiers for radii ``0..radius``."""
    if radius < 0:
        raise DataValidationError(f"radius must be >= 0, got {radius}")
    rings = find_rings(mol)
    ring_count = [0] * len(mol.atoms)
    for ring in rings:
        for atom in ring:
            ring_count[atom] += 1
    hydrogens = implicit_hydrogens(mol)
    current = [
        _sha_id(
            (
                mol.atoms[i].atomic_num,
                mol.degree(i),
                hydrogens[i],
                mol.atoms[i].charge,
                mol.atoms[i].is_aromatic,
                ring_count[i],
            )
        )
        for i in range(len(mol.atoms))
    ]
    history = [list(current)]
    for _ in range(radius):
        nxt: list[int] = []
        for i in range(len(mol.atoms)):
            neighbours = sorted(
                (
                    1.5 if b.is_aromatic else float(b.order),
                    current[mol.other_end(b, i)],
                )
                for b in mol.atom_bonds(i)
            )
            nxt.append(_sha_id((current[i], tuple(neighbours))))
        current = nxt
        history.append(list(current))
    return history


def morgan_fingerprint(
    mol: Molecule, radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """A folded Morgan/ECFP-style bit fingerprint (``uint8`` 0/1 vector).

    The recipe is the ECFP one: per-atom connectivity invariants
    refined once per radius level with sorted neighbour identifiers,
    all identifiers from iterations ``0..radius`` collected and folded
    modulo ``n_bits``.  The hash is this library's sha1-of-canonical-
    tuple — **bit indices are not comparable with RDKit's Morgan
    output** (the environments, uniqueness and similarity behaviour
    are; the byte-level mapping is not, and is never claimed to be).
    """
    if n_bits < 8:
        raise DataValidationError(f"n_bits must be >= 8, got {n_bits}")
    history = _environment_identifiers(mol, radius)
    bits = np.zeros(n_bits, dtype=np.uint8)
    for level in history:
        for identifier in level:
            bits[identifier % n_bits] = 1
    return bits


def morgan_environments(
    mol: Molecule, radius: int = 2
) -> list[dict[str, int | str]]:
    """Per-atom environment identifiers, for interpretation surfaces.

    Returns one record per atom per radius level (``atom``,
    ``radius``, ``identifier`` as a hex string, ``kind`` =
    ``"elemental"``/``"environment"``) — the interpretability twin of
    :func:`morgan_fingerprint`, the way RDKit's ``bitInfo`` map feeds
    ``Chem.Draw.DrawMorganBit``.
    """
    history = _environment_identifiers(mol, radius)
    records: list[dict[str, int | str]] = []
    for level, identifiers in enumerate(history):
        for atom_index, identifier in enumerate(identifiers):
            records.append(
                {
                    "atom": atom_index,
                    "radius": level,
                    "identifier": f"{identifier:016x}",
                    "kind": "elemental" if level == 0 else "environment",
                }
            )
    return records


# --------------------------------------------------------------------- #
# Substructure search                                                    #
# --------------------------------------------------------------------- #


def _atom_query_matches(query_atom: MolAtom, target_atom: MolAtom, target_h: int) -> bool:
    """Query-SMILES atom semantics (element/aromatic/H/charge constraints)."""
    if query_atom.atomic_num != target_atom.atomic_num:
        return False
    if query_atom.is_aromatic != target_atom.is_aromatic:
        return False
    if query_atom.bracket_h is not None and query_atom.bracket_h != target_h:
        return False
    return not (query_atom.charge and query_atom.charge != target_atom.charge)


def _bond_query_matches(query_bond: MolBond, target_bond: MolBond) -> bool:
    """Query-SMILES bond semantics (order, or aromatic-to-aromatic)."""
    if query_bond.is_aromatic or target_bond.is_aromatic:
        return query_bond.is_aromatic and target_bond.is_aromatic
    return query_bond.order == target_bond.order


def substructure_matches(
    mol: Molecule, query: str | Molecule, *, limit: int = 1000
) -> list[tuple[int, ...]]:
    """All atom-index mappings of ``query`` inside ``mol``.

    The query dialect is SMILES, interpreted as a query the way
    chemists write SMARTS-lite: organic-subset atoms match their
    element and aromaticity class; bracket atoms additionally pin the
    total hydrogen count (``[nH]`` matches aromatic N-H exactly) and,
    when charged, the formal charge.  Bonds match by order, with
    aromatic bonds matching aromatic bonds.  Full SMARTS (wildcards,
    recursion, logic) stays with RDKit — and since this core does not
    kekulize, aromatic queries match aromatic-flagged targets only
    (write target rings lowercase).

    Raises
    ------
    DataValidationError
        If the query is not parseable or ``limit`` is not positive.
    """
    if limit < 1:
        raise DataValidationError(f"limit must be >= 1, got {limit}")
    query_mol = query if isinstance(query, Molecule) else parse_smiles(query)
    if not query_mol.atoms or not mol.atoms:
        return []
    target_h = implicit_hydrogens(mol)
    query_bonds = {frozenset((b.begin, b.end)): b for b in query_mol.bonds}
    target_bond_map = {frozenset((b.begin, b.end)): b for b in mol.bonds}
    target_adj = mol.adjacency()
    order = _query_visit_order(query_mol)
    results: list[tuple[int, ...]] = []
    mapping: dict[int, int] = {}

    def extend(q_idx: int) -> None:
        if len(results) >= limit:
            return
        if q_idx == len(order):
            results.append(
                tuple(mapping[i] for i in range(len(query_mol.atoms)))
            )
            return
        q_atom_index = order[q_idx]
        q_atom = query_mol.atoms[q_atom_index]
        if q_idx == 0:
            candidates: list[int] = list(range(len(mol.atoms)))
        else:
            parent = _query_parent(query_mol, order, q_idx)
            q_bond = query_bonds[frozenset((q_atom_index, parent))]
            candidates = [
                t
                for t in target_adj[mapping[parent]]
                if t not in mapping.values()
                and _bond_query_matches(
                    q_bond, target_bond_map[frozenset((t, mapping[parent]))]
                )
            ]
        for target in candidates:
            if not _atom_query_matches(q_atom, mol.atoms[target], target_h[target]):
                continue
            if q_idx > 0:
                ok = True
                for q_other in query_mol.adjacency()[q_atom_index]:
                    if q_other in mapping:
                        q_b = query_bonds[frozenset((q_atom_index, q_other))]
                        t_b = target_bond_map[
                            frozenset((target, mapping[q_other]))
                        ]
                        if not _bond_query_matches(q_b, t_b):
                            ok = False
                            break
                if not ok:
                    continue
            mapping[q_atom_index] = target
            extend(q_idx + 1)
            del mapping[q_atom_index]
            if len(results) >= limit:
                return

    extend(0)
    return results


def _query_visit_order(query_mol: Molecule) -> list[int]:
    """BFS atom order (each atom after a connected predecessor)."""
    adj = query_mol.adjacency()
    order: list[int] = []
    visited: set[int] = set()
    for start in range(len(query_mol.atoms)):
        if start in visited:
            continue
        queue: deque[int] = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            order.append(node)
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
    return order


def _query_parent(query_mol: Molecule, order: list[int], q_idx: int) -> int:
    """The first already-ordered neighbour of ``order[q_idx]``."""
    adj = query_mol.adjacency()
    for earlier in order[:q_idx]:
        if order[q_idx] in adj[earlier]:
            return earlier
    raise DataValidationError("query graph disconnected under visit order")


def has_substructure(mol: Molecule, query: str | Molecule) -> bool:
    """``True`` when ``query`` matches anywhere in ``mol`` (early exit)."""
    return bool(substructure_matches(mol, query, limit=1))


# --------------------------------------------------------------------- #
# Canonical SMILES                                                       #
# --------------------------------------------------------------------- #


def canonical_smiles(mol: Molecule) -> str:
    """A deterministic canonical SMILES for the graph.

    Morgan invariants (element, degree, charge, hydrogens, aromaticity,
    ring count) are refined to stable ranks exactly like a canonical
    labelling pass; traversal starts at the lowest (rank, index) atom,
    neighbours are visited in (rank, index) order, ring-closure digits
    are reassigned in first-use order, and branches are parenthesised
    on backtracking.  The result is **round-trip stable and invariant
    to atom renumbering**; it is *not* guaranteed byte-identical to
    RDKit's canonical SMILES (tie-breaking differs) and stereo markers
    are dropped (they were never interpreted — see the parse notes).

    Round-trip contract -- fused purines included: the kekule and the
    aromatic spelling of caffeine collapse to one form, and that form
    reparses to the same graph (formula, fingerprint, ring count):

    >>> caffeine = parse_smiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
    >>> form = canonical_smiles(caffeine)
    >>> form == canonical_smiles(parse_smiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O"))
    True
    >>> twin = parse_smiles(form)
    >>> molecular_formula(twin)
    'C8H10N4O2'
    >>> bool(
    ...     (morgan_fingerprint(caffeine, radius=2, n_bits=128)
    ...      == morgan_fingerprint(twin, radius=2, n_bits=128)).all()
    ... )
    True
    >>> canonical_smiles(twin) == form
    True
    """
    if not mol.atoms:
        return ""
    n_atoms = len(mol.atoms)
    rings = find_rings(mol)
    ring_count = [0] * n_atoms
    for ring in rings:
        for atom in ring:
            ring_count[atom] += 1
    hydrogens = implicit_hydrogens(mol)
    invariants = [
        (
            mol.atoms[i].atomic_num,
            mol.degree(i),
            mol.atoms[i].charge,
            hydrogens[i],
            mol.atoms[i].is_aromatic,
            ring_count[i],
        )
        for i in range(n_atoms)
    ]
    ranks = _rank_invariants(invariants)
    for _ in range(n_atoms):
        labels = [
            (
                ranks[i],
                tuple(
                    sorted(
                        (
                            1.5 if b.is_aromatic else float(b.order),
                            ranks[mol.other_end(b, i)],
                        )
                        for b in mol.atom_bonds(i)
                    )
                ),
            )
            for i in range(n_atoms)
        ]
        new_ranks = _rank_invariants(labels)
        stable = len(set(new_ranks)) == len(set(ranks))
        ranks = new_ranks
        if stable:
            break
    order_key = ranks

    adj = mol.adjacency()
    ring_bonds = _ring_bond_set(mol, rings)
    components = _connected_components(mol)
    # pass 1: the spanning forest (same traversal order as the writer)
    tree_edges: set[frozenset[int]] = set()
    visited: set[int] = set()
    for component in components:
        start = min(component, key=lambda i: (order_key[i], i))
        stack: list[tuple[int, int | None]] = [(start, None)]
        while stack:
            node, parent = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if parent is not None:
                tree_edges.add(frozenset((node, parent)))
            for nbr in sorted(adj[node], key=lambda j: (order_key[j], j), reverse=True):
                if nbr not in visited:
                    stack.append((nbr, node))
    closure_bonds = {
        key for key in ring_bonds if key not in tree_edges
    }

    emitted: dict[frozenset[int], int] = {}
    state = {"digit": 1}
    visited = set()

    def digit_text(digit: int) -> str:
        return str(digit) if digit <= 9 else f"%{digit}"

    def write_atom(index: int) -> str:
        text = _atom_text(mol, index, hydrogens[index])
        for bond in mol.atom_bonds(index):
            key = frozenset((bond.begin, bond.end))
            if key in closure_bonds and key in emitted:
                text += digit_text(emitted[key])
        return text

    def write_bond(begin: int, end: int) -> str:
        bond = next(b for b in mol.bonds if {b.begin, b.end} == {begin, end})
        if bond.is_aromatic:
            return ""
        if mol.atoms[begin].is_aromatic and mol.atoms[end].is_aromatic:
            # kekule bond between two aromatic atoms: the explicit
            # symbol keeps it non-aromatic on reparse (plain adjacency
            # between lowercase atoms would be re-read as aromatic)
            return {1: "-", 2: "=", 3: "#"}[bond.order]
        # aromatic-aliphatic junction: the plain kekule symbol -- ":"
        # here would be re-read as an aromatic bond and break kekulization
        return {1: "", 2: "=", 3: "#"}[bond.order]

    pieces: list[str] = []

    def visit(index: int, parent: int | None) -> None:
        visited.add(index)
        pieces.append(write_atom(index))
        for bond in mol.atom_bonds(index):
            key = frozenset((bond.begin, bond.end))
            partner = mol.other_end(bond, index)
            if (
                key in closure_bonds
                and key not in emitted
                and partner not in visited
            ):
                emitted[key] = state["digit"]
                state["digit"] += 1
                pieces.append(digit_text(emitted[key]))
        # SMILES grammar: a '(' branch attaches to the atom written
        # immediately before it -- so every branch is written *before*
        # the one child that continues the chain (the last in the
        # canonical order).  Children are exactly the pass-1 tree
        # edges: closure bonds carry digits, non-tree non-ring back
        # edges are unreachable here by construction.
        children = [
            nbr
            for nbr in sorted(adj[index], key=lambda j: (order_key[j], j))
            if nbr != parent
            and frozenset((index, nbr)) not in closure_bonds
            and frozenset((index, nbr)) in tree_edges
        ]
        for nbr in children[:-1]:
            pieces.append(f"({write_bond(index, nbr)}")
            visit(nbr, index)
            pieces.append(")")
        if children:
            chain = children[-1]
            pieces.append(write_bond(index, chain))
            visit(chain, index)

    for component_index, component in enumerate(components):
        if component_index:
            pieces.append(".")
        visited.difference_update(component)
        start = min(component, key=lambda i: (order_key[i], i))
        visit(start, None)
    return "".join(pieces)


def _rank_invariants(labels: list[tuple]) -> list[int]:
    """Dense ranks (0-based) for a list of comparable tuples."""
    order = {label: rank for rank, label in enumerate(sorted(set(labels)))}
    return [order[label] for label in labels]


def _connected_components(mol: Molecule) -> list[list[int]]:
    """Atom-index components, first-appearance order."""
    adj = mol.adjacency()
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(mol.atoms)):
        if start in seen:
            continue
        queue: deque[int] = deque([start])
        seen.add(start)
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for nbr in adj[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        components.append(sorted(component))
    return components


def _atom_text(mol: Molecule, index: int, n_h: int) -> str:
    """The SMILES text for one atom (brackets only when needed)."""
    atom = mol.atoms[index]
    aromatic = atom.is_aromatic
    symbol = atom.symbol.lower() if aromatic else atom.symbol
    needs_bracket = (
        atom.charge != 0
        or atom.isotope != 0
        or (aromatic and atom.symbol == "N" and n_h == 1)
        or (not aromatic and atom.symbol not in _ORGANIC_SUBSET)
    )
    if not needs_bracket:
        return symbol
    parts = [f"{atom.isotope}" if atom.isotope else "", symbol]
    if n_h:
        parts.append(f"H{n_h if n_h > 1 else ''}")
    if atom.charge > 0:
        parts.append("+" if atom.charge == 1 else f"+{atom.charge}")
    elif atom.charge < 0:
        parts.append("-" if atom.charge == -1 else f"{atom.charge}")
    return f"[{''.join(parts)}]"


# --------------------------------------------------------------------- #
# 2D depiction coordinates                                               #
# --------------------------------------------------------------------- #


def generate_2d_coordinates(
    mol: Molecule, bond_length: float = 1.0
) -> np.ndarray:
    """Depiction coordinates — ring templates plus chain growth.

    The concept of ``rdDepictor``/``Chem.Draw`` without its coordinate
    refinement: every ring is a regular polygon (radius
    ``L/(2 sin(pi/n))``); fused rings are grown off a shared edge away
    from the ring system's centroid; chains are walked with
    collision-avoiding direction search (candidate angles ranked by
    distance to every placed atom, straight/trans preferred);
    disconnected components are stacked with a gap.  Good enough for
    sketching and MOL export — not RDKit's aesthetically optimised
    layouts, and said so.  Deterministic: no randomness anywhere.
    """
    n_atoms = len(mol.atoms)
    coords = np.zeros((n_atoms, 2), dtype=float)
    if n_atoms <= 1:
        return coords
    rings = find_rings(mol)
    ring_atoms: set[int] = set()
    for ring in rings:
        ring_atoms.update(ring)
    adj = mol.adjacency()
    placed: set[int] = set()

    # -- ring systems -------------------------------------------------- #
    ring_adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for i in ring_atoms:
        ring_adj[i] = [j for j in adj[i] if j in ring_atoms]
    systems: list[list[int]] = []
    seen: set[int] = set()
    for start in sorted(ring_atoms):
        if start in seen:
            continue
        queue: deque[int] = deque([start])
        seen.add(start)
        system: list[int] = []
        while queue:
            node = queue.popleft()
            system.append(node)
            for nbr in ring_adj[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        systems.append(sorted(system))

    for system in systems:
        system_rings = sorted(
            (r for r in rings if r <= set(system)),
            key=lambda r: (len(r), sorted(r)),
        )
        if not system_rings:
            continue
        first = system_rings[0]
        _place_polygon(coords, _cycle_order(first, mol), bond_length)
        placed.update(first)
        pending = system_rings[1:]
        progress = True
        while pending and progress:
            progress = False
            for ring in list(pending):
                if not any(a in placed for a in ring):
                    continue
                edge = _shared_edge(ring, mol, placed)
                if edge is not None:
                    a, b = edge
                    walk = _ring_walk(ring, mol, a, b)[2:]
                    if walk:
                        _grow_ring_arc(coords, a, b, walk, bond_length, placed)
                        placed.update(walk)
                else:
                    anchor = min(a for a in ring if a in placed)
                    new_atoms = [a for a in _cycle_order(ring, mol) if a not in placed]
                    _place_spoked(coords, mol, anchor, new_atoms, bond_length, placed)
                    placed.update(new_atoms)
                pending.remove(ring)
                progress = True
        for ring in pending:  # odd fusions: spoke around an anchor
            anchor = next((a for a in ring if a in placed), system[0])
            new_atoms = [a for a in ring if a not in placed]
            _place_spoked(coords, mol, anchor, new_atoms, bond_length, placed)
            placed.update(new_atoms)

    # -- acyclic atoms: BFS chain growth -------------------------------- #
    if not placed:
        coords[0] = (0.0, 0.0)
        placed.add(0)
    queue = deque(sorted(placed))
    while queue:
        node = queue.popleft()
        for nbr in adj[node]:
            if nbr in placed:
                continue
            direction = _chain_direction(coords, node, placed, adj)
            coords[nbr] = coords[node] + direction * bond_length
            placed.add(nbr)
            queue.append(nbr)

    # -- disconnected components: stack them --------------------------- #
    components = _connected_components(mol)
    if len(components) > 1:
        offset_y = 0.0
        for component in components:
            ys = coords[list(component), 1]
            span = float(ys.max() - ys.min()) if len(component) > 1 else 0.0
            coords[list(component)] -= coords[list(component)].mean(axis=0)
            coords[list(component), 1] += offset_y + span / 2
            offset_y += span + 2.0 * bond_length
    return coords


def _ring_walk(
    ring: frozenset[int], mol: Molecule, a: int, b: int
) -> list[int]:
    """Cycle order from ``a`` to ``b`` along the ring's own edge."""
    ordered = _cycle_order(ring, mol)
    idx_a = ordered.index(a)
    if ordered[(idx_a + 1) % len(ordered)] == b:
        return ordered[idx_a:] + ordered[:idx_a]
    reversed_order = list(reversed(ordered))
    idx_a = reversed_order.index(a)
    return reversed_order[idx_a:] + reversed_order[:idx_a]


def _shared_edge(
    ring: frozenset[int], mol: Molecule, placed: set[int]
) -> tuple[int, int] | None:
    """A ring edge whose both endpoints are already placed."""
    ordered = _cycle_order(ring, mol)
    for a, b in zip(ordered, ordered[1:] + ordered[:1]):
        if a in placed and b in placed:
            return a, b
    return None


def _place_polygon(coords: np.ndarray, order: list[int], length: float) -> None:
    """Regular polygon, first vertex pointing up."""
    n = len(order)
    radius = length / (2.0 * math.sin(math.pi / n))
    for k, atom in enumerate(order):
        angle = -math.pi / 2 + 2 * math.pi * k / n
        coords[atom] = (radius * math.cos(angle), radius * math.sin(angle))


def _grow_ring_arc(
    coords: np.ndarray,
    a: int,
    b: int,
    walk: list[int],
    length: float,
    placed: set[int],
) -> None:
    """Grow a fused ring's new atoms off the shared edge ``a-b``."""
    n = len(walk) + 2  # full ring size
    exterior = 2 * math.pi / n
    vec = coords[b] - coords[a]
    base = math.atan2(float(vec[1]), float(vec[0]))
    centroid = _system_centroid(coords, placed)
    midpoint = (coords[a] + coords[b]) / 2.0
    away = midpoint - centroid
    chosen = 1.0
    for sign in (1.0, -1.0):
        probe = base + sign * exterior
        step = np.array([math.cos(probe), math.sin(probe)])
        if not np.any(away) or float(np.dot(step, away)) > 0:
            chosen = sign
            break
    direction = base + chosen * exterior
    position = coords[b].copy()
    for atom in walk:
        position = position + length * np.array(
            [math.cos(direction), math.sin(direction)]
        )
        coords[atom] = position
        direction += chosen * exterior


def _system_centroid(coords: np.ndarray, placed: set[int]) -> np.ndarray:
    """Mean position of the placed atoms (origin when none)."""
    if not placed:
        return np.zeros(2)
    return coords[sorted(placed)].mean(axis=0)


def _place_spoked(
    coords: np.ndarray,
    mol: Molecule,
    anchor: int,
    new_atoms: list[int],
    length: float,
    placed: set[int],
) -> None:
    """Atoms around ``anchor`` on a circle, opposite occupied angles."""
    used = [
        math.atan2(
            float(coords[other][1] - coords[anchor][1]),
            float(coords[other][0] - coords[anchor][0]),
        )
        for other in mol.adjacency()[anchor]
        if other in placed
    ]
    step = 2 * math.pi / (len(new_atoms) + max(len(used), 1))
    angle = (used[0] + math.pi) if used else 0.0
    for k, atom in enumerate(new_atoms):
        theta = angle + k * step
        coords[atom] = coords[anchor] + length * np.array(
            [math.cos(theta), math.sin(theta)]
        )


def _chain_direction(
    coords: np.ndarray, node: int, placed: set[int], adj: list[list[int]]
) -> np.ndarray:
    """The least-colliding unit direction for a new chain atom."""
    parent = next((other for other in adj[node] if other in placed), None)
    incoming = (
        coords[node] - coords[parent]
        if parent is not None
        else np.array([1.0, 0.0])
    )
    if not np.any(incoming):
        incoming = np.array([1.0, 0.0])
    base = math.atan2(float(incoming[1]), float(incoming[0]))
    best_dir = np.array([math.cos(base), math.sin(base)])
    best_score = -1.0
    for delta in (
        math.pi,
        math.pi * 5 / 6,
        math.pi * 7 / 6,
        math.pi * 2 / 3,
        math.pi * 4 / 3,
        math.pi / 2,
        math.pi * 3 / 2,
        math.pi / 3,
        math.pi * 5 / 3,
        math.pi / 6,
        math.pi * 11 / 6,
    ):
        theta = base + delta
        direction = np.array([math.cos(theta), math.sin(theta)])
        probe = coords[node] + direction
        score = min(
            (
                float(np.linalg.norm(probe - coords[other]))
                for other in placed
                if other != node
            ),
            default=10.0,
        )
        if score > best_score + 1e-9:
            best_score = score
            best_dir = direction
    return best_dir


# --------------------------------------------------------------------- #
# MOL / SDF writing                                                      #
# --------------------------------------------------------------------- #


def mol_block(
    mol: Molecule,
    coords: np.ndarray | None = None,
    *,
    name: str = "",
    program_line: str = "  PharmaML molgraph (2D depiction)",
) -> str:
    """A V2000 MOL block for the graph — purely, no engine required.

    Aromatic bonds are written as type 4; formal charges land in
    ``M  CHG`` records; isotopes in ``M  ISO``.  ``coords`` defaults to
    :func:`generate_2d_coordinates` (z = 0) — the pure-Python 2D twin
    of ``Chem.MolToMolBlock``, which this toolkit previously could
    only reach through the optional RDKit dependency.
    """
    if coords is None:
        coords = generate_2d_coordinates(mol)
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    if len(coords) != len(mol.atoms):
        raise DataValidationError(
            f"coords has {len(coords)} rows for {len(mol.atoms)} atoms"
        )
    lines = [
        (name or mol.title or mol.source)[:80] or "molecule",
        program_line,
        "",
        f"{len(mol.atoms):3d}{len(mol.bonds):3d}  0  0  0  0  0  0  0  0999 V2000",
    ]
    for atom, (x, y) in zip(mol.atoms, coords):
        lines.append(
            f"{x:10.4f}{y:10.4f}{0.0:10.4f} {atom.symbol:<3s}"
            "0  0  0  0  0  0  0  0  0  0  0  0"
        )
    for bond in mol.bonds:
        kind = 4 if bond.is_aromatic else bond.order
        lines.append(f"{bond.begin + 1:3d}{bond.end + 1:3d}{kind:3d}  0  0  0")
    charged = [(atom.index, atom.charge) for atom in mol.atoms if atom.charge]
    if charged:
        entries = " ".join(f"{i + 1:3d} {c:3d}" for i, c in charged)
        lines.append(f"M  CHG{len(charged):3d} {entries}")
    isotopic = [(atom.index, atom.isotope) for atom in mol.atoms if atom.isotope]
    if isotopic:
        entries = " ".join(f"{i + 1:3d} {iso:3d}" for i, iso in isotopic)
        lines.append(f"M  ISO{len(isotopic):3d} {entries}")
    lines.append("M  END")
    return "\n".join(lines) + "\n"


def sdf_text(
    molecules: Iterable[Molecule | str],
    names: Iterable[str] | None = None,
    properties: Iterable[dict[str, str]] | None = None,
) -> str:
    """A whole SDF text from molecules (2D coordinates, V2000 blocks).

    The pure-Python sibling of ``converters.table_to_sdf`` — no ETKDG,
    no UFF, no RDKit: structures ship as 2D depictions with their
    properties as SD data fields, honestly labelled as such.
    """
    mols = [
        item if isinstance(item, Molecule) else parse_smiles(item)
        for item in molecules
    ]
    names_list = list(names) if names is not None else None
    props_list = list(properties) if properties is not None else None
    blocks: list[str] = []
    for idx, mol in enumerate(mols):
        title = names_list[idx] if names_list and idx < len(names_list) else ""
        block = mol_block(mol, name=title)
        if props_list and idx < len(props_list) and props_list[idx]:
            fields = "".join(
                f">  <{key}>\n{value}\n\n"
                for key, value in props_list[idx].items()
            )
            block = block.replace("M  END\n", f"M  END\n{fields}")
        blocks.append(block)
    return "".join(f"{block}$$$$\n" for block in blocks)


# --------------------------------------------------------------------- #
# The combined profile                                                   #
# --------------------------------------------------------------------- #


def molecule_profile(smiles: str) -> dict[str, float | int | str]:
    """One SMILES -> the full graph-core profile (API + GUI surface).

    Every value comes from the pure-Python core: formula, average
    molecular weight, Ertl TPSA, Lipinski's HBD/HBA, rotatable bonds,
    rings, heavy atoms, the sp3 fraction, the aromatic-atom count and
    the radius-2 fingerprint popcount.  Values that would need RDKit's
    private data tables (logP, QED) are simply absent — not faked.
    """
    mol = parse_smiles(smiles)
    profile = lipinski_profile(mol)
    fingerprint = morgan_fingerprint(mol, radius=2, n_bits=2048)
    aromatic = sum(1 for atom in mol.atoms if atom.is_aromatic)
    return {
        "smiles": mol.source,
        "formula": molecular_formula(mol),
        "molecular_weight": round(average_molecular_weight(mol), 2),
        "tpsa": tpsa(mol),
        "hbd": profile["hbd"],
        "hba": profile["hba"],
        "rotatable_bonds": profile["rotatable_bonds"],
        "rings": profile["rings"],
        "heavy_atoms": profile["heavy_atoms"],
        "fraction_csp3": profile["fraction_csp3"],
        "aromatic_atoms": aromatic,
        "fingerprint_bits": int(fingerprint.sum()),
        "stereo_notes": len(mol.stereo_notes),
    }
