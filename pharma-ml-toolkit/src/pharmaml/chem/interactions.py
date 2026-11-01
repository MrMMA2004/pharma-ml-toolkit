r"""ProLIF's protein-ligand interaction fingerprints, absorbed natively.

``cvignoux/ProLIF`` (the "Protein-Ligand Interaction Fingerprints"
library reviewed in this round's original pass and re-derived here
from its published detector geometry) computes interactions between
a ligand and a protein from *coordinates + chemistry flags*, then
collapses them into a bit per (ligand residue, protein residue,
interaction family) — the IFP every docking post-analysis uses.
The RDKit molecular-perception layer it stands on is what this
toolkit replaces with explicit typing; the detector geometry and the
fingerprint algebra transfer natively:

* :class:`InteractionAtom` / :class:`Residue` -- the typed atom model
  (element, coordinates, formal charge, donor/acceptor/hydrophobic
  flags) and its residue grouping.  :func:`assign_types` types the
  standard amino acids by atom name (backbone N donor / O acceptor,
  LYS-NZ and ARG guanidinium cationic, ASP/GLU carboxylates anionic,
  SER/THR/TYR/ASN/GLN/HIS/CYS donors+acceptors, the hydrophobic
  residue carbons), and metals by element.
* :func:`perceive_bonds` / :func:`perceive_rings` -- distance-based
  connectivity (covalent radii + 30% tolerance) and ring perception
  (spanning-tree fundamental cycles, size 3-7, aromatic when all
  ring atoms are C/N) for ligands whose bonding is not known.
* :func:`detect_interactions` -- **all eleven families with ProLIF's
  geometric conventions**: Hydrophobic (non-polar atom pairs ≤ 4.5 Å),
  HBDonor/HBAcceptor (donor-acceptor heavy atoms ≤ 3.5 Å, named from
  the *ligand's* side of the bond), Cationic/Anionic (opposite formal
  charges ≤ 4.5 Å, again ligand-side naming), PiCation (cation to
  aromatic centroid ≤ 4.5 Å with a ≤ 2.0 Å plane offset),
  PiStackingFF / PiStackingEF (ring centroids ≤ 5.0 Å with normal
  angle ≤ 30° for face-to-face; ≤ 5.5 Å with angle ≥ 50° and offset
  ≤ 2.5 Å for edge-to-face), VdWContact (3.0-4.0 Å), XBond (Cl/Br/I
  to acceptor ≤ 4.0 Å with C-X···A angle ≥ 120°) and MetalCoordination
  (metals to donor/acceptor ≤ 3.0 Å).
* :class:`InteractionReport` / :class:`InteractionFingerprint` -- the
  hit table, the residue-pair × family bit matrix
  (``to_dataframe()``), the flat bit vector, Tanimoto similarity
  between poses, and the per-family / per-residue counts.
* :func:`protein_from_pdb` / :func:`ligand_from_pdb` -- the bridge to
  :mod:`pharmaml.chem.structure`'s PDB parser, typing on the way in.

The honest scope, stated once: RDKit's SMARTS-based perception is
replaced by name/element heuristics (explicit flags override them
whenever you have better chemistry); the D-H···A angle criterion of
the H-bond detectors is not applied because the model carries no
explicit hydrogens (the heavy-atom distance carries the decision,
exactly like distance-only mode); ProLIF's bit names are kept, the
cutoffs are exposed as :class:`InteractionCutoffs` parameters whose
defaults re-derive the library's documented geometry.

Examples
--------
>>> from pharmaml.chem.interactions import (
...     InteractionFingerprint, detect_interactions, demo_complex,
... )
>>> ligand, protein = demo_complex()
>>> report = detect_interactions(ligand, protein)
>>> sorted(report.count_by_type())
['Cationic', 'HBAcceptor', 'HBDonor', 'Hydrophobic', 'MetalCoordination', 'PiCation', 'PiStackingFF', 'VdWContact', 'XBond']
>>> fp = report.fingerprint()
>>> fp.residue_pairs()
[('LIG1', 'ALA1'), ('LIG1', 'ASP4'), ('LIG1', 'GLY5'), ('LIG1', 'LYS2'), ('LIG1', 'PHE3'), ('LIG1', 'SER6'), ('LIG1', 'ZN7')]
>>> float(fp.tanimoto(fp))
1.0
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.chem.structure import COVALENT_RADII
from pharmaml.exceptions import DataValidationError

__all__ = [
    "INTERACTION_FAMILIES",
    "InteractionAtom",
    "InteractionCutoffs",
    "InteractionFingerprint",
    "InteractionReport",
    "Residue",
    "assign_types",
    "demo_complex",
    "detect_interactions",
    "fingerprint_diversity",
    "ligand_from_pdb",
    "perceive_bonds",
    "perceive_rings",
    "protein_from_pdb",
]


#: The eleven interaction families, in canonical (sorted) bit order.
INTERACTION_FAMILIES: tuple[str, ...] = (
    "Anionic",
    "Cationic",
    "HBDonor",
    "HBAcceptor",
    "Hydrophobic",
    "MetalCoordination",
    "PiCation",
    "PiStackingEF",
    "PiStackingFF",
    "VdWContact",
    "XBond",
)

#: Elements treated as metals for the coordination detector.
METAL_ELEMENTS = frozenset({"ZN", "MG", "MN", "CA", "FE", "NI", "CO", "CU", "NA", "K"})

#: Elements treated as halogen-bond donors (F is too electronegative).
HALOGEN_ELEMENTS = frozenset({"CL", "BR", "I"})


@dataclass(frozen=True)
class InteractionCutoffs:
    """Every distance/angle threshold, exposed and documented.

    The defaults re-derive ProLIF's documented detector geometry
    (hydrophobic 4.5 Å, H-bond 3.5 Å, ionic 4.5 Å, cation-π 4.5 Å
    with a 2.0 Å plane offset, face-to-face stacking 5.0 Å / 30°,
    edge-to-face 5.5 Å / 50° / 2.5 Å, van der Waals 3.0-4.0 Å,
    halogen bond 4.0 Å / 120°, metal coordination 3.0 Å).
    """

    hydrophobic: float = 4.5
    hbond: float = 3.5
    ionic: float = 4.5
    cation_pi: float = 4.5
    cation_pi_offset: float = 2.0
    pi_stack_ff: float = 5.0
    pi_stack_ff_angle: float = 30.0
    pi_stack_ef: float = 5.5
    pi_stack_ef_angle: float = 50.0
    pi_stack_ef_offset: float = 2.5
    vdw_min: float = 3.0
    vdw_max: float = 4.0
    xbond: float = 4.0
    xbond_angle: float = 120.0
    metal: float = 3.0


@dataclass
class InteractionAtom:
    """One typed atom of the interaction model.

    Attributes
    ----------
    symbol : str
        Element symbol (``"C"``, ``"NZ"``-style names live in ``name``).
    xyz : np.ndarray
        ``(3,)`` coordinates in Å.
    charge : int
        Formal charge (``+1`` for a protonated amine, ``-1`` for a
        carboxylate oxygen, ``+2`` for a zinc...).
    is_donor : bool
        Bears (or could bear) a hydrogen — H-bond donor.
    is_acceptor : bool
        Has lone pairs — H-bond acceptor.
    is_hydrophobic : bool
        Non-polar carbon/sulfur, by typing or by hand.
    name : str
        Atom name (PDB-style, e.g. ``"OD1"``).
    """

    symbol: str
    xyz: np.ndarray
    charge: int = 0
    is_donor: bool = False
    is_acceptor: bool = False
    is_hydrophobic: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(3)
        self.symbol = self.symbol.upper()


@dataclass
class Residue:
    """A named group of interaction atoms.

    Attributes
    ----------
    name, number, chain : str, int, str
        The residue identity (``("ALA", 1, "A")`` by default).
    atoms : list[InteractionAtom]
    rings : list[list[int]]
        Aromatic rings as lists of atom indices (perceived or given).
    """

    name: str
    number: int
    chain: str = "A"
    atoms: list[InteractionAtom] = field(default_factory=list)
    rings: list[list[int]] = field(default_factory=list)

    @property
    def res_id(self) -> str:
        """The fingerprint row/column id, e.g. ``"ALA1"``."""

        return f"{self.name}{self.number}"

    def coords(self) -> np.ndarray:
        """``(n, 3)`` coordinate stack."""

        if not self.atoms:
            raise DataValidationError(f"residue {self.res_id} has no atoms")
        return np.array([a.xyz for a in self.atoms], dtype=float)

    def ring_centroid(self, ring_index: int) -> np.ndarray:
        """Centroid of one aromatic ring."""

        ring = self.rings[ring_index]
        return self.coords()[ring].mean(axis=0)

    def ring_normal(self, ring_index: int) -> np.ndarray:
        """Best-fit plane normal of one aromatic ring (unit vector)."""

        ring = self.rings[ring_index]
        points = self.coords()[ring]
        centred = points - points.mean(axis=0)
        # the plane normal is the *minimum*-variance direction (the last
        # singular vector); vt[0] would be the largest, i.e. in-plane
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        return vt[-1] / np.linalg.norm(vt[-1])


# --------------------------------------------------------------------------
# typing: the RDKit layer, replaced honestly
# --------------------------------------------------------------------------

#: Atom names that donate H-bonds in the standard residues.
_DONOR_NAMES = frozenset(
    {
        "N",  # backbone amide
        "NZ",  # LYS
        "NE", "NH1", "NH2",  # ARG
        "ND1", "NE2",  # HIS
        "ND2",  # ASN
        "NE1",  # TRP
        "OG", "OG1",  # SER / THR
        "OH",  # TYR
        "SG",  # CYS
    }
)

#: Atom names that accept H-bonds in the standard residues.
_ACCEPTOR_NAMES = frozenset(
    {
        "O",  # backbone carbonyl
        "OD1", "OD2",  # ASP / ASN
        "OE1", "OE2",  # GLU / GLN
        "ND1", "NE2",  # HIS
        "OG", "OG1",  # SER / THR
        "OH",  # TYR
        "SG",  # CYS
        "SD",  # MET
    }
)

#: Residues whose carbons/sulfurs count as hydrophobic.
_HYDROPHOBIC_RESIDUES = frozenset(
    {"ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO", "GLY"}
)


def assign_types(residue: Residue, *, standard: bool = True) -> Residue:
    """Type a residue's atoms by element and PDB atom name.

    The rules, applied in order:

    * metals (by element) → ``+2`` charge;
    * anionic: ASP ``OD1/OD2`` and GLU ``OE1/OE2`` → charge ``-1``;
    * cationic: LYS ``NZ`` and ARG ``NE/NH1/NH2`` → charge ``+1``;
    * donors: backbone ``N`` (residue 1 excluded), LYS ``NZ``,
      ARG ``NE/NH*``, HIS/ASN ``ND*``, GLN/TRP ``NE*``,
      SER/THR/TYR ``OG*/OH``, CYS ``SG``;
    * acceptors: backbone ``O``, ASP/GLU/ASN/GLN ``OD*/OE*``,
      HIS ``ND1/NE2``, SER/THR/TYR ``O*``, CYS ``SG``, MET ``SD``;
    * hydrophobic: C/S atoms of the hydrophobic residues
      (ALA VAL LEU ILE PHE TRP MET PRO GLY), except backbone
      carbonyl carbons.

    Explicit flags already set on an atom are preserved (your
    chemistry beats the heuristic).

    Parameters
    ----------
    residue : Residue
        The residue to type in place.
    standard : bool
        ``False`` skips the amino-acid tables (element-only typing:
        N/O → donor+acceptor candidates, C/S → hydrophobic).

    Returns
    -------
    Residue
        The same residue, atoms typed.

    Examples
    --------
    >>> lys = Residue("LYS", 2, atoms=[
    ...     InteractionAtom("N", [0, 0, 0], name="N"),
    ...     InteractionAtom("N", [3, 0, 0], name="NZ"),
    ... ])
    >>> _ = assign_types(lys)
    >>> (lys.atoms[0].is_donor, lys.atoms[1].charge)
    (True, 1)
    """

    for index, atom in enumerate(residue.atoms):
        if atom.symbol in METAL_ELEMENTS:
            atom.charge = atom.charge or 2
            continue
        if not standard:
            if atom.symbol in ("N", "O") and not (atom.is_donor or atom.is_acceptor):
                atom.is_donor = True
                atom.is_acceptor = True
            if atom.symbol in ("C", "S") and not atom.is_hydrophobic:
                atom.is_hydrophobic = True
            continue
        name = atom.name.strip().upper()
        if not name:
            name = atom.symbol
        if residue.name == "ASP" and name in ("OD1", "OD2"):
            atom.charge = min(atom.charge, -1) if atom.charge < 0 else -1
        elif residue.name == "GLU" and name in ("OE1", "OE2"):
            atom.charge = -1 if atom.charge >= 0 else atom.charge
        elif residue.name == "LYS" and name == "NZ" or residue.name == "ARG" and name in ("NE", "NH1", "NH2"):
            atom.charge = 1 if atom.charge <= 0 else atom.charge
        is_backbone_n = name == "N" and index == 0
        if atom.symbol == "N":
            if name in _DONOR_NAMES or is_backbone_n:
                atom.is_donor = True
            if name in _ACCEPTOR_NAMES and name != "N":
                atom.is_acceptor = True
        elif atom.symbol == "O":
            if name in _DONOR_NAMES:
                atom.is_donor = True
            if name in _ACCEPTOR_NAMES or name == "O":
                atom.is_acceptor = True
        elif atom.symbol == "S":
            if name in _DONOR_NAMES:
                atom.is_donor = True
            if name in _ACCEPTOR_NAMES:
                atom.is_acceptor = True
        if (
            atom.symbol in ("C", "S")
            and residue.name in _HYDROPHOBIC_RESIDUES
            and name != "C"  # the backbone carbonyl carbon is polar
        ):
            atom.is_hydrophobic = True
    return residue


# --------------------------------------------------------------------------
# connectivity perception for ligands
# --------------------------------------------------------------------------


def perceive_bonds(atoms: Sequence[InteractionAtom], *, tolerance: float = 1.3) -> list[tuple[int, int]]:
    """Distance-based bonds: radii sum × tolerance, no bonded H's.

    Parameters
    ----------
    atoms : sequence of InteractionAtom
    tolerance : float
        The radii-sum multiplier (1.3 = generous, forgiving of idealised
        coordinates; 1.15 for real refined geometries).

    Examples
    --------
    >>> atoms = [InteractionAtom("C", [0, 0, 0]), InteractionAtom("N", [1.4, 0, 0])]
    >>> perceive_bonds(atoms)
    [(0, 1)]
    """

    if not atoms:
        return []
    coords = np.array([a.xyz for a in atoms], dtype=float)
    bonds: list[tuple[int, int]] = []
    n = len(atoms)
    for i in range(n):
        radius_i = COVALENT_RADII.get(atoms[i].symbol, 0.8)
        for j in range(i + 1, n):
            radius_j = COVALENT_RADII.get(atoms[j].symbol, 0.8)
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance <= (radius_i + radius_j) * tolerance and distance > 0.4:
                bonds.append((i, j))
    return bonds


def perceive_rings(
    atoms: Sequence[InteractionAtom], bonds: Sequence[tuple[int, int]] | None = None
) -> list[list[int]]:
    """Aromatic rings from connectivity (spanning-tree fundamental cycles).

    Rings of size 3-7 whose atoms are all C or N count as aromatic —
    the benzene/indole/azole world, without the aromaticity theory.

    Examples
    --------
    >>> ring = [InteractionAtom("C", [1.39 * c, 1.204 * s, 0])
    ...         for c, s in zip((1, .5, -.5, -1, -.5, .5), (0, 1, 1, 0, -1, -1))]
    >>> perceive_rings(ring)
    [[4, 5, 0, 1, 2, 3]]
    """

    if bonds is None:
        bonds = perceive_bonds(atoms)
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(atoms))}
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)
    # spanning tree by BFS; each non-tree edge closes a fundamental cycle
    parent: dict[int, int] = {}
    seen: set[int] = set()
    cycles: list[list[int]] = []
    for start in adjacency:
        if start in seen:
            continue
        seen.add(start)
        parent[start] = start
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    parent[neighbour] = node
                    queue.append(neighbour)
                elif parent[node] != neighbour and neighbour != node:
                    cycle = _fundamental_cycle(parent, node, neighbour)
                    if 3 <= len(cycle) <= 7:
                        cycles.append(cycle)
    aromatic: list[list[int]] = []
    seen_keys: set[frozenset[int]] = set()
    for cycle in cycles:
        key = frozenset(cycle)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if all(atoms[i].symbol in ("C", "N") for i in cycle):
            aromatic.append(cycle)
    return aromatic


def _fundamental_cycle(parent: dict[int, int], u: int, v: int) -> list[int]:
    """The cycle closed by the non-tree edge u-v, as an ordered list."""

    # ancestor chain of u, root last
    chain_u: list[int] = []
    node = u
    guard = 0
    while guard <= len(parent):
        chain_u.append(node)
        if parent[node] == node:
            break
        node = parent[node]
        guard += 1
    depth = {n: i for i, n in enumerate(chain_u)}
    # walk v up until it meets u's chain
    path_v: list[int] = []
    node = v
    guard = 0
    while guard <= len(parent):
        if node in depth:
            branch = chain_u[: depth[node] + 1]
            return branch + list(reversed(path_v))
        path_v.append(node)
        node = parent[node]
        guard += 1
    return [u, v]


# --------------------------------------------------------------------------
# the detectors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionHit:
    """One detected atom-level contact.

    Attributes
    ----------
    family : str
        One of :data:`INTERACTION_FAMILIES`.
    ligand_residue, protein_residue : str
        The residue ids (``"LIG1"``, ``"ASP4"``).
    ligand_atoms, protein_atoms : tuple[int, ...]
        Atom indices driving the contact.
    distance : float
        The driving distance in Å (centroid distance for the π
        families, angle in degrees appended for XBond via ``detail``).
    detail : str
        Extra geometry (e.g. the C-X···A angle).
    """

    family: str
    ligand_residue: str
    protein_residue: str
    ligand_atoms: tuple[int, ...]
    protein_atoms: tuple[int, ...]
    distance: float
    detail: str = ""


class InteractionReport:
    """The full hit list of one :func:`detect_interactions` run.

    Examples
    --------
    >>> ligand, protein = demo_complex()
    >>> report = detect_interactions(ligand, protein)
    >>> report.count_by_type()['PiStackingFF']
    1
    """

    def __init__(self, hits: Sequence[InteractionHit]) -> None:
        self.hits = list(hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self) -> Iterator[InteractionHit]:
        return iter(self.hits)

    def __repr__(self) -> str:
        return f"InteractionReport({len(self.hits)} hits)"

    def count_by_type(self) -> dict[str, int]:
        """Hits per interaction family (families with zero hits omitted)."""

        counts: dict[str, int] = {}
        for hit in self.hits:
            counts[hit.family] = counts.get(hit.family, 0) + 1
        return dict(sorted(counts.items()))

    def count_by_residue(self) -> dict[str, dict[str, int]]:
        """``{protein residue: {family: hits}}`` — the per-pocket view."""

        counts: dict[str, dict[str, int]] = {}
        for hit in self.hits:
            per_residue = counts.setdefault(hit.protein_residue, {})
            per_residue[hit.family] = per_residue.get(hit.family, 0) + 1
        return dict(sorted(counts.items()))

    def to_dataframe(self) -> pd.DataFrame:
        """The hit table (one row per atom-level contact)."""

        return pd.DataFrame(
            [
                {
                    "family": hit.family,
                    "ligand": hit.ligand_residue,
                    "protein": hit.protein_residue,
                    "ligand_atoms": ",".join(str(i) for i in hit.ligand_atoms),
                    "protein_atoms": ",".join(str(i) for i in hit.protein_atoms),
                    "distance": round(hit.distance, 2),
                    "detail": hit.detail,
                }
                for hit in self.hits
            ]
        )

    def fingerprint(self) -> InteractionFingerprint:
        """Collapse to the residue-pair × family bit matrix."""

        bits: set[tuple[str, str, str]] = set()
        for hit in self.hits:
            bits.add((hit.ligand_residue, hit.protein_residue, hit.family))
        return InteractionFingerprint(bits)

    def summary(self) -> str:
        """A human one-screen read of the interaction profile."""

        lines = [f"{len(self.hits)} atom-level contacts, {len(self.fingerprint().bits)} IFP bits"]
        for residue, families in self.count_by_residue().items():
            profile = ", ".join(f"{name}×{n}" for name, n in sorted(families.items()))
            lines.append(f"  {residue:<6} {profile}")
        return "\n".join(lines)


class InteractionFingerprint:
    """The IFP: one bit per (ligand residue, protein residue, family).

    The ProLIF data model — residue-pair rows, interaction-family
    columns — with Tanimoto similarity over the flat bit vector.

    Examples
    --------
    >>> ligand, protein = demo_complex()
    >>> fp = detect_interactions(ligand, protein).fingerprint()
    >>> int(fp.to_bitvector().sum())
    15
    """

    def __init__(self, bits: Iterable[tuple[str, str, str]]):
        self.bits: set[tuple[str, str, str]] = set(bits)

    # -- structure -------------------------------------------------------- #

    def residue_pairs(self) -> list[tuple[str, str]]:
        """The sorted ``(ligand, protein)`` residue pairs with any bit."""

        return sorted({(lig, prot) for lig, prot, _family in self.bits})

    def families_for(self, ligand: str, protein: str) -> list[str]:
        """Which families are set for one residue pair."""

        return sorted(
            family for lig, prot, family in self.bits if lig == ligand and prot == protein
        )

    def to_bitvector(self) -> np.ndarray:
        """The flat boolean vector (sorted bit keys, deterministic)."""

        keys = self._universe(self.bits)
        return np.array([key in self.bits for key in keys], dtype=bool)

    @staticmethod
    def _universe(*bitsets: Iterable[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        merged: set[tuple[str, str, str]] = set()
        for bitset in bitsets:
            merged.update(bitset)
        return sorted(merged)

    # -- algebra ----------------------------------------------------------- #

    def tanimoto(self, other: InteractionFingerprint) -> float:
        """Tanimoto coefficient of two fingerprints (1.0 = identical)."""

        keys = self._universe(self.bits, other.bits)
        if not keys:
            return 1.0
        a = {key for key in keys if key in self.bits}
        b = {key for key in keys if key in other.bits}
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    def difference(self, other: InteractionFingerprint) -> list[tuple[str, str, str]]:
        """Bits set here but not there (what a pose lost)."""

        return sorted(self.bits - other.bits)

    # -- views -------------------------------------------------------------- #

    def to_dataframe(self) -> pd.DataFrame:
        """The ProLIF ``to_df`` view: residue-pair rows × family columns."""

        pairs = self.residue_pairs()
        index = pd.MultiIndex.from_tuples(pairs, names=["ligand", "protein"])
        frame = pd.DataFrame(
            np.zeros((len(pairs), len(INTERACTION_FAMILIES)), dtype=bool),
            index=index,
            columns=list(INTERACTION_FAMILIES),
        )
        for lig, prot, family in self.bits:
            frame.loc[(lig, prot), family] = True
        return frame

    def __repr__(self) -> str:
        return f"InteractionFingerprint({len(self.bits)} bits)"


def fingerprint_diversity(fingerprints: Sequence[InteractionFingerprint]) -> float:
    """Mean pairwise Tanimoto *distance* across a pose series.

    0.0 = every pose interacts identically; approaching 1.0 = every
    pose touches something different — the number pose-selection
    workflows optimise.
    """

    if len(fingerprints) < 2:
        raise DataValidationError("fingerprint_diversity: need at least two fingerprints")
    total = 0.0
    count = 0
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            total += 1.0 - fingerprints[i].tanimoto(fingerprints[j])
            count += 1
    return total / count


def _residues_like(
    entries: Sequence[Residue] | Residue,
) -> list[Residue]:
    if isinstance(entries, Residue):
        return [entries]
    return list(entries)


def detect_interactions(
    ligand: Residue | Sequence[Residue],
    protein: Residue | Sequence[Residue],
    *,
    cutoffs: InteractionCutoffs | None = None,
) -> InteractionReport:
    """Run every detector family over a ligand/protein pair.

    Parameters
    ----------
    ligand : Residue or sequence of Residue
        The ligand (one residue for small molecules, several for
        peptides/nucleotides).
    protein : Residue or sequence of Residue
        The binding site (or the whole protein — the detectors are
        distance-bounded and cheap).
    cutoffs : InteractionCutoffs, optional
        Override any threshold; defaults re-derive ProLIF's geometry.

    Returns
    -------
    InteractionReport
        Every atom-level hit, collapsible to the IFP.

    Examples
    --------
    >>> ligand, protein = demo_complex()
    >>> report = detect_interactions(ligand, protein)
    >>> report.count_by_type()['XBond']
    1
    """

    rules = cutoffs or InteractionCutoffs()
    lig_residues = _residues_like(ligand)
    prot_residues = _residues_like(protein)
    hits: list[InteractionHit] = []

    for lig_res in lig_residues:
        lig_coords = lig_res.coords()
        for prot_res in prot_residues:
            prot_coords = prot_res.coords()
            distances = np.linalg.norm(
                lig_coords[:, None, :] - prot_coords[None, :, :], axis=-1
            )
            hits.extend(
                _detect_pair(lig_res, prot_res, distances, rules)
            )
    hits.sort(key=lambda h: (h.protein_residue, h.family, round(h.distance, 3)))
    return InteractionReport(hits)


def _detect_pair(
    lig_res: Residue,
    prot_res: Residue,
    distances: np.ndarray,
    rules: InteractionCutoffs,
) -> list[InteractionHit]:
    """All families between one ligand residue and one protein residue."""

    hits: list[InteractionHit] = []
    lig_id, prot_id = lig_res.res_id, prot_res.res_id

    def add(
        family: str,
        i: int,
        j: int,
        distance: float,
        detail: str = "",
    ) -> None:
        hits.append(
            InteractionHit(family, lig_id, prot_id, (i,), (j,), float(distance), detail)
        )

    n_lig, n_prot = distances.shape

    # -- hydrophobic: non-polar pairs within the cutoff ------------------ #
    for i, atom in enumerate(lig_res.atoms):
        if not atom.is_hydrophobic:
            continue
        for j, partner in enumerate(prot_res.atoms):
            if partner.is_hydrophobic and distances[i, j] <= rules.hydrophobic:
                add("Hydrophobic", i, j, distances[i, j])

    # -- hydrogen bonds, named from the ligand's side --------------------- #
    for i, atom in enumerate(lig_res.atoms):
        for j, partner in enumerate(prot_res.atoms):
            if distances[i, j] > rules.hbond:
                continue
            if atom.is_donor and partner.is_acceptor:
                add("HBDonor", i, j, distances[i, j])
            if atom.is_acceptor and partner.is_donor:
                add("HBAcceptor", i, j, distances[i, j])

    # -- ionic: opposite formal charges ----------------------------------- #
    for i, atom in enumerate(lig_res.atoms):
        for j, partner in enumerate(prot_res.atoms):
            if distances[i, j] > rules.ionic:
                continue
            if atom.charge > 0 and partner.charge < 0:
                add("Cationic", i, j, distances[i, j])
            if atom.charge < 0 and partner.charge > 0:
                add("Anionic", i, j, distances[i, j])

    # -- van der Waals contacts: the 3.0-4.0 Å shell ----------------------- #
    shell = (distances > rules.vdw_min) & (distances <= rules.vdw_max)
    for i, j in zip(*np.nonzero(shell)):
        add("VdWContact", int(i), int(j), distances[i, j])

    # -- cation-pi: rings on one side, cations on the other ---------------- #
    for ring_owner, cation_owner, swapped in (
        (lig_res, prot_res, False),
        (prot_res, lig_res, True),
    ):
        for ring_index in range(len(ring_owner.rings)):
            centroid = ring_owner.ring_centroid(ring_index)
            normal = ring_owner.ring_normal(ring_index)
            for k, atom in enumerate(cation_owner.atoms):
                if atom.charge <= 0:
                    continue
                offset = abs(float(np.dot(atom.xyz - centroid, normal)))
                span = float(np.linalg.norm(atom.xyz - centroid))
                if span <= rules.cation_pi and offset <= rules.cation_pi_offset:
                    detail = f"plane offset {offset:.2f} A"
                    if swapped:
                        hits.append(
                            InteractionHit(
                                "PiCation", lig_id, prot_id, (ring_index,), (k,), span, detail
                            )
                        )
                    else:
                        hits.append(
                            InteractionHit(
                                "PiCation", lig_id, prot_id, (k,), (ring_index,), span, detail
                            )
                        )

    # -- pi stacking: ring pairs, face-to-face then edge-to-face ----------- #
    for lig_ring in range(len(lig_res.rings)):
        lig_centroid = lig_res.ring_centroid(lig_ring)
        lig_normal = lig_res.ring_normal(lig_ring)
        for prot_ring in range(len(prot_res.rings)):
            prot_centroid = prot_res.ring_centroid(prot_ring)
            prot_normal = prot_res.ring_normal(prot_ring)
            span = float(np.linalg.norm(lig_centroid - prot_centroid))
            if span > rules.pi_stack_ef:
                continue
            cosine = abs(float(np.dot(lig_normal, prot_normal)))
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            offset = abs(float(np.dot(prot_centroid - lig_centroid, lig_normal)))
            family = ""
            if span <= rules.pi_stack_ff and angle <= rules.pi_stack_ff_angle:
                family = "PiStackingFF"
            elif (
                span <= rules.pi_stack_ef
                and angle >= rules.pi_stack_ef_angle
                and offset <= rules.pi_stack_ef_offset
            ):
                family = "PiStackingEF"
            if family:
                hits.append(
                    InteractionHit(
                        family,
                        lig_id,
                        prot_id,
                        (lig_ring,),
                        (prot_ring,),
                        span,
                        f"normal angle {angle:.1f} deg, offset {offset:.2f} A",
                    )
                )

    # -- halogen bonds: ligand Cl/Br/I to protein acceptors ---------------- #
    lig_bonds = perceive_bonds(lig_res.atoms)
    for i, atom in enumerate(lig_res.atoms):
        if atom.symbol not in HALOGEN_ELEMENTS:
            continue
        carbons = [b for b in lig_bonds if i in b]
        for j, partner in enumerate(prot_res.atoms):
            if not partner.is_acceptor or distances[i, j] > rules.xbond:
                continue
            detail = ""
            if carbons:
                carbon_index = carbons[0][0] if carbons[0][0] != i else carbons[0][1]
                c_x = lig_res.atoms[carbon_index].xyz - atom.xyz
                x_a = partner.xyz - atom.xyz
                cos_angle = float(
                    np.dot(c_x, x_a)
                    / (np.linalg.norm(c_x) * np.linalg.norm(x_a))
                )
                angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
                if angle < rules.xbond_angle:
                    continue
                detail = f"C-X...A angle {angle:.1f} deg"
            add("XBond", i, j, distances[i, j], detail)

    # -- metal coordination: metals to donor/acceptor atoms ---------------- #
    for metal_side, other_side, swapped in (
        (prot_res, lig_res, True),
        (lig_res, prot_res, False),
    ):
        for m, metal in enumerate(metal_side.atoms):
            if metal.symbol not in METAL_ELEMENTS:
                continue
            for k, partner in enumerate(other_side.atoms):
                if not (partner.is_donor or partner.is_acceptor):
                    continue
                metal_xyz = metal.xyz
                span = float(np.linalg.norm(partner.xyz - metal_xyz))
                if span > rules.metal:
                    continue
                if swapped:
                    hits.append(
                        InteractionHit("MetalCoordination", lig_id, prot_id, (k,), (m,), span)
                    )
                else:
                    hits.append(
                        InteractionHit("MetalCoordination", lig_id, prot_id, (m,), (k,), span)
                    )
    _ = (n_lig, n_prot)
    return hits


# --------------------------------------------------------------------------
# PDB bridge
# --------------------------------------------------------------------------


def _residues_from_atoms(
    records: Sequence[Any], *, standard: bool, ligand: bool
) -> list[Residue]:
    groups: dict[tuple, list[int]] = {}
    order: list[tuple] = []
    for index, record in enumerate(records):
        key = (record.chain, record.resseq, record.resname)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(index)
    residues: list[Residue] = []
    for chain, resseq, resname in order:
        indices = groups[(chain, resseq, resname)]
        atoms = [
            InteractionAtom(
                symbol=record.element,
                xyz=record.xyz,
                name=record.name,
                charge=2 if record.element in METAL_ELEMENTS else 0,
            )
            for record in (records[i] for i in indices)
        ]
        residue = Residue(resname, resseq, chain, atoms)
        assign_types(residue, standard=standard)
        if ligand:
            residue.rings = perceive_rings(residue.atoms)
        else:
            residue.rings = _standard_rings(resname, residue)
        residues.append(residue)
    return residues


def _standard_rings(resname: str, residue: Residue) -> list[list[int]]:
    """Sidechain rings of the standard residues, by atom name."""

    names = [atom.name.strip().upper() for atom in residue.atoms]
    index_of = {name: i for i, name in enumerate(names)}
    if resname in ("PHE", "TYR"):
        ring = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
    elif resname == "TRP":
        ring = ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
    elif resname == "HIS":
        ring = ["CG", "ND1", "CD2", "CE1", "NE2"]
    elif resname in ("ARG", "PRO"):
        ring = ["CG", "CD", "NE"] if resname == "ARG" else ["N", "CA", "C", "CD", "CG"]
    else:
        return []
    return [[index_of[n] for n in ring if n in index_of]]


def protein_from_pdb(text: str, *, selection: str = "") -> list[Residue]:
    """Typed protein residues from PDB text.

    Uses :func:`pharmaml.chem.structure.parse_pdb_block`, drops
    HETATMs (waters, ligands, metals — call :func:`ligand_from_pdb`
    for those), applies :func:`assign_types` and the standard
    sidechain rings.
    """

    from pharmaml.chem.structure import parse_pdb_block, select_atoms

    records = parse_pdb_block(text)
    if selection:
        records = select_atoms(records, selection)
    polymer = [r for r in records if not r.hetatm]
    if not polymer:
        raise DataValidationError("protein_from_pdb: no ATOM records in the block")
    return _residues_from_atoms(polymer, standard=True, ligand=False)


def ligand_from_pdb(text: str) -> list[Residue]:
    """Typed ligand residues from PDB text (HETATM records only).

    Element-only typing plus distance-perceived bonds and rings —
    override any flag on the returned atoms when you know better.
    """

    from pharmaml.chem.structure import parse_pdb_block

    records = [r for r in parse_pdb_block(text) if r.hetatm]
    if not records:
        raise DataValidationError("ligand_from_pdb: no HETATM records in the block")
    return _residues_from_atoms(records, standard=False, ligand=True)


# --------------------------------------------------------------------------
# the demo complex: every family, by design
# --------------------------------------------------------------------------


def demo_complex() -> tuple[Residue, list[Residue]]:
    """A designed ligand/protein pair that triggers nine families.

    The ligand is a chlorinated benzene ring with a protonated amine,
    a neutral donor NH and a carbonyl acceptor.  The "protein" is
    seven residues placed around it: ALA (hydrophobic), LYS
    (cation-π), PHE (π-stacking), ASP (ionic), GLY (backbone
    H-bonds both ways), SER (halogen-bond acceptor) and a zinc
    (metal coordination).

    Returns
    -------
    (Residue, list[Residue])
        The ligand and the protein residue list.
    """

    cos, sin = np.cos(np.pi / 3), np.sin(np.pi / 3)
    ring = [
        (1.39, 0.0, 0.0),
        (1.39 * cos, 1.39 * sin, 0.0),
        (-1.39 * cos, 1.39 * sin, 0.0),
        (-1.39, 0.0, 0.0),
        (-1.39 * cos, -1.39 * sin, 0.0),
        (1.39 * cos, -1.39 * sin, 0.0),
    ]
    lig_atoms = [
        InteractionAtom("C", np.asarray(xyz, dtype=float), is_hydrophobic=True, name=f"C{n}")
        for n, xyz in enumerate(ring, start=1)
    ]
    lig_atoms += [
        InteractionAtom("N", np.asarray([0.0, 2.8, 0.0], dtype=float),
                        charge=1, is_donor=True, name="N1"),
        InteractionAtom("O", np.asarray([2.5, -2.5, 0.0], dtype=float),
                        is_acceptor=True, name="O1"),
        InteractionAtom("N", np.asarray([-2.5, -2.5, 0.0], dtype=float),
                        is_donor=True, name="N2"),
        InteractionAtom("CL", np.asarray([-2.9, 0.0, 0.0], dtype=float), name="CL1"),
    ]
    ligand = Residue("LIG", 1, "A", lig_atoms, rings=[[0, 1, 2, 3, 4, 5]])

    def atom(
        symbol: str, xyz: Sequence[float], name: str = "", **flags: Any
    ) -> InteractionAtom:
        return InteractionAtom(symbol, np.asarray(xyz, dtype=float), name=name, **flags)

    protein = [
        Residue(
            "ALA", 1, "A",
            [atom("C", [4.0, 2.5, 0.0], "CB", is_hydrophobic=True)],
        ),
        Residue(
            "LYS", 2, "A",
            [atom("N", [4.3, 0.0, 0.5], "NZ", charge=1)],
        ),
        Residue(
            "PHE", 3, "A",
            [atom("C", [x, y, 4.0], f"C{n}", is_hydrophobic=True)
             for n, (x, y, _) in enumerate(ring, start=1)],
            rings=[[0, 1, 2, 3, 4, 5]],
        ),
        Residue(
            "ASP", 4, "A",
            [
                atom("O", [-0.2, 6.2, 0.0], "OD1", charge=-1, is_acceptor=True),
                atom("O", [1.1, 6.6, 0.2], "OD2", charge=-1, is_acceptor=True),
            ],
        ),
        Residue(
            "GLY", 5, "A",
            [
                atom("N", [2.5, -5.0, 0.0], "N", is_donor=True),
                atom("O", [-2.5, -5.3, 0.0], "O", is_acceptor=True),
            ],
        ),
        Residue(
            "SER", 6, "A",
            [atom("O", [-6.2, 0.5, 0.0], "OG", is_donor=True, is_acceptor=True)],
        ),
        Residue(
            "ZN", 7, "A",
            [atom("ZN", [5.0, -2.5, 0.0], "ZN", charge=2)],
        ),
    ]
    return ligand, protein
