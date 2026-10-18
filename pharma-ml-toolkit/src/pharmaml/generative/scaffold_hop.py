"""BREED-style scaffold-hop crossover for the genetic algorithm.

The structural recombination operator for
:class:`~pharmaml.generative.GeneticOptimizer`: two parent molecules are
cut and recombined so an offspring is a hybrid that keeps parts of each
parent (Nicolaou et al.'s BREED algorithm tradition, in the pragmatic
spirit of this package: RDKit graph surgery + ``Chem.molzip``, no
learned model).  Six cut geometries cover the four things a scaffold
hop actually has to recombine -- fragment *arrangement*, ring
*topology*, ring-*system* topology and *linker chemistry* -- completing
the {excised piece} x {cut type} matrix:

* **bridge** (the classic): both parents are cut at a pair of
  *homologous* single acyclic bonds and the fragments recombined
  cross-wise; every fragment arrives intact, so ring systems survive by
  construction, and whenever the parents' frameworks span different
  fragments (biphenyls, diaryl ethers, sulfonyl bridges) the child's
  framework differs from BOTH parents.
* **merge**: both parents are cut at homologous single *ring* bonds and
  the two opened rings are zipped into one combined ring holding atoms
  from both parents (ring expansion / macrocycle formation).
* **transplant**: one parent is cut at a single ring bond and a
  two-ended linker from the other parent -- a single atom (an ether O,
  a methylene, a secondary amine N, ...) or a short contiguous acyclic
  *path* of up to four atoms (an ethylene bridge ``-CH2CH2-``, an
  oxyethylene ``-OCH2-`` segment, a diethylene ``-OCH2CH2O-``) -- is
  excised and zipped into the cut: ring homologation by one atom or a
  whole chain (piperidine + an ether oxygen -> an oxazepane, + an
  ethylene bridge -> an azocane).
* **fuse**: a whole *ring* is excised from one parent -- any ring
  (aromatic ones included) attached to the rest of its molecule through
  exactly two single anchor bonds at two distinct ring atoms, the
  Nicolaou fused-ring move -- and spliced across an opened ring bond of
  the other parent: the child is a fused / bridged bicyclic holding one
  ring from each parent (decalin x piperidine -> an azadecalin core,
  indane x piperidine -> a benzazocine core).
* **insert**: the matrix's last cell -- a whole ring is excised from
  one parent (the same donor machinery) and spliced INTO a cut
  *acyclic* bond of the other: both bridge fragments are re-joined
  *through* the ring, so the ring becomes the spacer -- the
  ring-as-linker move (decalin x hexane -> a dipropylcyclohexane).
* **swap**: two-point segment-swap crossover, the graph analogue of
  classic two-point GA recombination -- each parent's bounded acyclic
  *middle* (a single atom or a contiguous path of up to four atoms)
  can be excised from the flank parent and REPLACED by the other
  parent's homologous middle: the only size-conserving geometry (insert
  grows, swap exchanges) and the medchem linker / bioisostere exchange
  move -- an ester O swapped for an amide N (aspirin x paracetamol ->
  salicylamide), an ether O for a thioether S.  Across the shipped
  244-molecule corpus the swap scan admits 7,328 candidates over 295
  distinct linker chemistries, 1,738 of them heteroatom bioisostere
  exchanges.

All cuts are *single* bonds (aromatic ring bonds are never cut -- the
re-zip would have to invent bond orders; the fuse geometry's two anchor
bonds are single bonds too, but they may themselves be ring bonds of an
*adjacent* ring -- that is exactly what a fused system is), and the join
points must be **homologous**: the atoms joined by each new bond match
on (atomic number, degree, aromaticity, formal charge), so
recombination happens at chemically analogous positions (an aryl-CH2
bond crosses with another aryl-CH2 bond, not with a sulfonamide S-N).

The geometry of the drawn cut can be biased: pass ``geometry_weights``
(a mapping ``{geometry name: non-negative weight}``; :data:`GEOMETRY_ORDER`
lists the seven names in scan order) and the geometry is picked with
probability proportional to its weight, renormalised over the
geometries **present** in the pair's candidate pool -- keys naming
geometries absent from the pool simply place no mass, and a geometry
left at weight zero is never drawn.

Deterministic given the caller's ``random.Random``; ``None`` whenever no
viable cut exists (aromatic-only parents, no homologous pair, children
unsanitisable or over the length cap) -- the caller falls back, never
crashes.

Example
-------
>>> from pharmaml.generative import scaffold_hop_crossover
>>> import random
>>> child = scaffold_hop_crossover("c1ccccc1CC(=O)O", "c1ccncc1CO",
...                                random.Random(11))
>>> child is not None   # a valid hybrid of both parents
True
>>> # ring expansion by a whole linker path: piperidine x diethylene
>>> # glycol, the -CH2-O- path transplanted into a cut C-C bond -> 8-ring
>>> scaffold_hop_crossover("C1CCNCC1", "OCCOCCO", random.Random(0))
'C1CNCCCOC1'
>>> # fused-ring splice (the Nicolaou move): decalin's left ring
>>> # excised and spliced across a cut piperidine bond -> a bicyclic
>>> # (a 6-ring + an 8-ring: an azadecalin core)
>>> scaffold_hop_crossover("C1CCC2CCCCC2C1", "C1CCNCC1", random.Random(0))
'C1CCC2CCNCCCC2C1'
>>> # ring-as-linker insert: decalin's ring spliced INTO a cut hexane
>>> # bond -> a single dipropylcyclohexane spacer ring
>>> scaffold_hop_crossover("C1CCC2CCCCC2C1", "CCCCCC", random.Random(5))
'CCCC1CCCCC1CCC'
>>> # two-point segment swap: aspirin x paracetamol -- the ester O
>>> # exchanged for the amide N -> salicylamide (a bioisostere child)
>>> scaffold_hop_crossover("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1",
...                        random.Random(0))
'CC(=O)Nc1ccccc1C(=O)O'
>>> # forcing a geometry: transplant-only weights never draw a path
>>> # candidate -- piperidine x diethylene glycol keeps homologating the
>>> # ring one atom at a time (the single-atom linker transplants)
>>> scaffold_hop_crossover("C1CCNCC1", "OCCOCCO", random.Random(1),
...                        geometry_weights={"transplant": 1.0})
'C1CNCCOC1'
>>> # benzene has no acyclic single bond to cut: no crossover possible
>>> scaffold_hop_crossover("c1ccccc1", "c1ccncc1", random.Random(0)) is None
True
>>> # the sanity gate (100x-53): aspirin x naproxen at this draw state
>>> # breeds 'OO' (hydrogen peroxide) -- contract-conformant but
>>> # chemically comic; the opt-in min_heavy_atoms gate rejects it and
>>> # the bridge pairing loop falls through to a REAL hybrid child
>>> scaffold_hop_crossover("CC(=O)Oc1ccccc1C(=O)O",
...                        "COc1ccc2cc(ccc2c1)C(C)C(=O)O", random.Random(25))
'OO'
>>> scaffold_hop_crossover("CC(=O)Oc1ccccc1C(=O)O",
...                        "COc1ccc2cc(ccc2c1)C(C)C(=O)O", random.Random(25),
...                        min_heavy_atoms=3)
'COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1'
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping

import numpy as np

from ._utils import require_rdkit

#: Minimum ``max_smiles_length`` the operator accepts (matches the
#: :class:`~pharmaml.generative.GeneticOptimizer` floor, so the standalone
#: use cannot be configured weaker than the GA use).
_MIN_LENGTH_CAP = 20

#: Maximum atoms in an excised multi-atom linker *path*.  Drug-like
#: linkers are short (ethylene, oxyethylene, diethyleneoxy); longer
#: chains would grow the candidate pool quadratically and the offspring
#: past drug-like size -- the ``max_smiles_length`` cap polices the
#: output, this cap keeps the search honest.
_MAX_LINKER_PATH_ATOMS = 4

#: The seven recombination-geometry operator names, in the fixed scan /
#: selection order (the names accepted by ``scaffold_hop_crossover``'s
#: ``geometry_weights`` -- the transplant family's single-atom and
#: multi-atom buckets count as two operators).  The scan builds the
#: candidate pool in this order, so a weighted draw that iterates it is
#: deterministic and pool-order-consistent.
GEOMETRY_ORDER: tuple[str, ...] = (
    "bridge",
    "merge",
    "transplant",
    "transplant_path",
    "fuse",
    "insert",
    "swap",
)


def _checked_min_heavy_atoms(value: object) -> int:
    """Validate a ``min_heavy_atoms`` gate value (the house int rule).

    Plain and numpy integers are accepted (a numpy integer is normalised
    to ``int``); bools are rejected explicitly -- they are ``int``
    subclasses but ``True`` is not a meaningful atom count (the 100x-41-G
    descriptors convention).  Anything below 1 would gate nothing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(  # noqa: TRY004 — consistent module error currency
            f"min_heavy_atoms must be an integer >= 1, got {value!r}"
        )
    if int(value) < 1:
        raise ValueError(f"min_heavy_atoms must be an integer >= 1, got {value!r}")
    return int(value)


def _atom_signature(atom: object) -> tuple[int, int, bool, int]:
    """Homology key: two atoms match when these four fields all agree.

    Atomic number and formal charge fix the element/charge class, degree
    fixes the connectivity class (a methyl carbon never crosses with a
    quaternary one), aromaticity separates aryl from alkyl carbons --
    the pragmatic BREED-style environment descriptor, cheap and strict
    enough that recombination happens at analogous positions.
    """
    return (
        atom.GetAtomicNum(),  # type: ignore[attr-defined]
        atom.GetDegree(),  # type: ignore[attr-defined]
        atom.GetIsAromatic(),  # type: ignore[attr-defined]
        atom.GetFormalCharge(),  # type: ignore[attr-defined]
    )


def _cut_into_two(
    mol: object,
    bond_idx: int,
) -> tuple[object, object] | None:
    """Cut ``mol`` at single bond ``bond_idx``; return ``(begin, end)`` fragments.

    Both returned fragments carry exactly one ``[*:1]`` dummy at the cut
    (map number 1), so ``Chem.molzip`` on any two of them forms exactly
    one new bond.  ``None`` on any RDKit failure -- robustness first.
    """
    Chem = require_rdkit()
    try:
        rw = Chem.RWMol(mol)
        bond = rw.GetBondWithIdx(bond_idx)
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rw.RemoveBond(begin, end)
        # Valence bookkeeping is exact for single bonds: each endpoint
        # loses one single-bonded neighbour and gains one single-bonded
        # dummy, so no implicit hydrogens appear or vanish -- the cut
        # fragments sanitise exactly like the parent.
        d_begin = rw.AddAtom(Chem.Atom(0))
        d_end = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(begin, d_begin, Chem.BondType.SINGLE)
        rw.AddBond(end, d_end, Chem.BondType.SINGLE)
        cut = rw.GetMol()
        idx_frags = Chem.GetMolFrags(cut)
        mol_frags = Chem.GetMolFrags(cut, asMols=True)
        if len(idx_frags) != 2 or len(mol_frags) != 2:  # pragma: no cover
            return None  # a bridge bond always splits a connected mol in two
        begin_frag = end_frag = None
        for indices, fragment in zip(idx_frags, mol_frags):
            if d_begin in indices:
                begin_frag = fragment
            if d_end in indices:
                end_frag = fragment
        if begin_frag is None or end_frag is None:  # pragma: no cover
            return None
        for fragment in (begin_frag, end_frag):
            Chem.SanitizeMol(fragment)
            dummies = [
                atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0
            ]
            if len(dummies) != 1:  # pragma: no cover
                return None
            dummy = dummies[0]
            dummy.SetAtomMapNum(1)
            dummy.SetIsotope(0)
        return begin_frag, end_frag
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None


def _open_ring(mol: object, bond_idx: int) -> object | None:
    """Cut one single *ring* bond; return the ring-opened molecule.

    A ring cut never disconnects a connected molecule: the result is ONE
    fragment carrying two dummies -- ``[*:1]`` on the former begin atom
    of the cut bond, ``[*:2]`` on the former end atom -- so
    ``Chem.molzip`` against another two-ended piece forms two new bonds
    (a ring merge or a linker transplant).  ``None`` on any RDKit
    failure.
    """
    Chem = require_rdkit()
    try:
        rw = Chem.RWMol(mol)
        bond = rw.GetBondWithIdx(bond_idx)
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rw.RemoveBond(begin, end)
        # Same exact valence bookkeeping as _cut_into_two: each endpoint
        # trades one single-bonded neighbour for one single-bonded dummy.
        d_begin = rw.AddAtom(Chem.Atom(0))
        d_end = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(begin, d_begin, Chem.BondType.SINGLE)
        rw.AddBond(end, d_end, Chem.BondType.SINGLE)
        opened = rw.GetMol()
        if len(Chem.GetMolFrags(opened)) != 1:  # pragma: no cover
            return None  # a ring cut keeps a connected mol connected
        Chem.SanitizeMol(opened)
        # RWMol.GetMol() preserves atom indices, so the begin/end atoms
        # are still addressable -- label each dummy by its host's side.
        dummies = [
            atom for atom in opened.GetAtoms() if atom.GetAtomicNum() == 0
        ]
        if len(dummies) != 2:  # pragma: no cover
            return None
        for dummy in dummies:
            host = dummy.GetNeighbors()[0].GetIdx()
            if host == begin:
                dummy.SetAtomMapNum(1)
            elif host == end:
                dummy.SetAtomMapNum(2)
            else:  # pragma: no cover
                return None
            dummy.SetIsotope(0)
        return opened
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None


def _excise_linker_path(
    mol: object,
    path: tuple[int, ...],
) -> tuple[object, tuple] | None:
    """Cut a contiguous acyclic linker path loose; return ``(donor, host_sigs)``.

    A linker path is either a single atom (the classic two-ended linker:
    an ether O, a methylene, a secondary amine N, a thioether S ...) or
    a contiguous chain of up to ``_MAX_LINKER_PATH_ATOMS`` such atoms
    (an ethylene bridge, an oxyethylene segment).  Every path atom must
    have exactly two neighbours, both reached via single acyclic bonds,
    and be outside every ring; attachment dummies (atomic number 0) are
    never excisable -- a parent's dummy must not be transplanted into a
    child.  The donor is a fresh molecule holding the path atoms and two
    dummies -- ``[*:1]`` replacing the bond to the first path atom's
    outside host, ``[*:2]`` the last's -- ready to be zipped into a ring
    cut of the other parent.  ``host_sigs`` are the two former outside
    neighbours' signatures, in the path's own orientation, for the
    homology check.  ``None`` when any validation fails (a chain that
    runs through a double-bonded atom -- e.g. an aldehyde carbon --
    never excises).

    Valence bookkeeping is exact for every seam: each path end atom
    trades one single-bonded host for one single-bonded dummy, interior
    atoms keep both neighbours, so the donor sanitises exactly like the
    corresponding parent fragment.
    """
    Chem = require_rdkit()
    try:
        if not path or len(path) > _MAX_LINKER_PATH_ATOMS:
            return None
        if len(set(path)) != len(path):
            return None  # a repeating index is not a path
        for idx in path:
            atom = mol.GetAtomWithIdx(idx)  # type: ignore[attr-defined]
            if atom.GetDegree() != 2 or atom.IsInRing():
                return None
            if atom.GetAtomicNum() == 0:  # attachment dummy: never excised
                return None
            # degree 2 -> exactly these two bonds, so one loop covers both
            # the interior path bonds and the two host bonds
            for nbr in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(  # type: ignore[attr-defined]
                    idx, nbr.GetIdx()
                )
                if (
                    bond is None
                    or bond.IsInRing()
                    or bond.GetBondType() != Chem.BondType.SINGLE
                ):
                    return None
        # consecutive path atoms must actually be bonded (the enumeration
        # emits true chains; a hand-made tuple may not be one)
        for left, right in itertools.pairwise(path):
            if mol.GetBondBetweenAtoms(left, right) is None:  # type: ignore[attr-defined]
                return None
        # hosts: n == 1 -> the atom's two neighbours in GetNeighbors
        # order (the classic linker semantics); n >= 2 -> each end's
        # unique outside neighbour (the other neighbour is the next path
        # atom, so the lookups are unambiguous)
        if len(path) == 1:
            nbrs = list(mol.GetAtomWithIdx(path[0]).GetNeighbors())  # type: ignore[attr-defined]
            host_first, host_second = nbrs[0], nbrs[1]
        else:
            first_outside = [
                nbr
                for nbr in mol.GetAtomWithIdx(path[0]).GetNeighbors()  # type: ignore[attr-defined]
                if nbr.GetIdx() != path[1]
            ]
            last_outside = [
                nbr
                for nbr in mol.GetAtomWithIdx(path[-1]).GetNeighbors()  # type: ignore[attr-defined]
                if nbr.GetIdx() != path[-2]
            ]
            if len(first_outside) != 1 or len(last_outside) != 1:
                return None
            host_first, host_second = first_outside[0], last_outside[0]
        rw = Chem.RWMol()
        for idx in path:
            anchor = Chem.Atom(mol.GetAtomWithIdx(idx))  # type: ignore[attr-defined]
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
        # The donor was hand-built (path atoms in order, then the two
        # dummies in exactly that atom order), so the dummy indices are
        # fixed: label the first-host-side dummy [*:1], the second [*:2].
        donor.GetAtomWithIdx(d_first).SetAtomMapNum(1)
        donor.GetAtomWithIdx(d_second).SetAtomMapNum(2)
        return donor, (_atom_signature(host_first), _atom_signature(host_second))
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None


def _linker_path_units(mol: object) -> list[tuple[tuple[int, ...], tuple]]:
    """Enumerate multi-atom linker units: ``[(path, host_sigs)]``.

    Every maximal run ("chain") of linker-eligible atoms -- degree two,
    acyclic, non-dummy -- contributes its contiguous sublists of two up
    to ``_MAX_LINKER_PATH_ATOMS`` atoms as candidate excisions; the
    single atoms keep their dedicated scan in
    :func:`scaffold_hop_crossover` so that geometry's candidate pool is
    unchanged.  A sublist is admitted only when
    :func:`_excise_linker_path` validates it -- a chain may run through
    a double-bonded atom (an aldehyde carbon, a carbonyl), and any unit
    containing it never excises, so it is silently dropped here.

    Determinism: chains are discovered in order of their smallest atom
    index (the first unvisited eligible atom in index order is its
    chain's minimum), each chain oriented smallest-index end first, and
    sublists emitted left-to-right, shorter before longer within the
    same start -- a pure function of the molecule, so the crossover's
    candidate pool stays a pure function of the parents.
    """
    assigned: set[int] = set()
    units: list[tuple[tuple[int, ...], tuple]] = []

    def eligible(atom: object) -> bool:
        return (
            atom.GetDegree() == 2  # type: ignore[attr-defined]
            and not atom.IsInRing()  # type: ignore[attr-defined]
            and atom.GetAtomicNum() != 0  # type: ignore[attr-defined]
        )

    for atom in mol.GetAtoms():  # type: ignore[attr-defined]
        start = atom.GetIdx()  # type: ignore[attr-defined]
        if start in assigned or not eligible(atom):
            continue
        # Walk the maximal chain through `start` in both directions:
        # both of its neighbours are eligible-or-host, each eligible
        # neighbour extends the run until a host (degree != 2, ring
        # atom, dummy) terminates it.
        chain = [start]
        neighbours = list(atom.GetNeighbors())  # type: ignore[attr-defined]
        for is_first_side, seed in (
            (True, neighbours[1]),
            (False, neighbours[0]),
        ):
            prev, cur = start, seed.GetIdx()
            while eligible(mol.GetAtomWithIdx(cur)):  # type: ignore[attr-defined]
                chain.append(cur)
                nxt = [
                    nbr.GetIdx()
                    for nbr in mol.GetAtomWithIdx(cur).GetNeighbors()  # type: ignore[attr-defined]
                    if nbr.GetIdx() != prev
                ]
                prev, cur = cur, nxt[0]
            if is_first_side:
                chain.reverse()  # the far side becomes the front
        assigned.update(chain)
        if chain[0] > chain[-1]:
            chain.reverse()  # canonical orientation: smaller end first
        for begin in range(len(chain)):
            for length in range(
                2, min(_MAX_LINKER_PATH_ATOMS, len(chain) - begin) + 1
            ):
                path = tuple(chain[begin:begin + length])
                excision = _excise_linker_path(mol, path)
                if excision is None:
                    continue
                units.append((path, excision[1]))
    return units


def _middle_units(mol: object) -> list[tuple[tuple[int, ...], tuple]]:
    """Enumerate swap middles: ``[(path, host_sigs)]``.

    The swap geometry's inventory of bounded acyclic middles -- the exact
    transplant eligibility, reused: single eligible atoms first in
    atom-index order (the transplant scan's convention), then the
    multi-atom linker paths in :func:`_linker_path_units` order (two up
    to ``_MAX_LINKER_PATH_ATOMS`` atoms).  Every unit is admitted only
    when :func:`_excise_linker_path` validates it, so a middle is
    simultaneously excisable from its flank (as the replaced segment)
    and donatable as the piece (the segment moved in).  A pure function
    of the molecule, so the candidate pool stays a pure function of the
    parents.
    """
    units: list[tuple[tuple[int, ...], tuple]] = []
    for atom in mol.GetAtoms():  # type: ignore[attr-defined]
        if atom.GetDegree() != 2 or atom.IsInRing():  # type: ignore[attr-defined]
            continue
        excision = _excise_linker_path(mol, (atom.GetIdx(),))
        if excision is None:
            continue
        units.append(((atom.GetIdx(),), excision[1]))
    units.extend(_linker_path_units(mol))
    return units


def _excise_middle(
    mol: object,
    path: tuple[int, ...],
) -> tuple[object, object] | None:
    """Cut a bounded acyclic middle out of ``mol``; ``(flank_first, flank_second)``.

    The swap geometry's flank step: ``path`` is excised and *discarded*
    (the piece parent's middle replaces it at assembly time).  Both host
    bonds -- single and acyclic by :func:`_excise_linker_path`
    admission -- are cut, one dummy is attached to each former host
    (``[*:1]`` on the path[0]-side flank, ``[*:2]`` on the other), and
    the molecule splits into exactly three fragments: the two flanks
    survive, the middle is dropped.  Admission guarantees the split --
    any host-to-host detour around the middle would close a ring through
    it, and every path atom is ring-free.  ``None`` on any RDKit
    failure, the robustness-first contract of every helper here.
    """
    Chem = require_rdkit()
    try:
        if _excise_linker_path(mol, path) is None:
            return None  # not a bounded acyclic middle
        if len(path) == 1:
            nbrs = list(mol.GetAtomWithIdx(path[0]).GetNeighbors())  # type: ignore[attr-defined]
            host_first, host_second = nbrs[0].GetIdx(), nbrs[1].GetIdx()
        else:
            first_outside = [
                nbr.GetIdx()
                for nbr in mol.GetAtomWithIdx(path[0]).GetNeighbors()  # type: ignore[attr-defined]
                if nbr.GetIdx() != path[1]
            ]
            last_outside = [
                nbr.GetIdx()
                for nbr in mol.GetAtomWithIdx(path[-1]).GetNeighbors()  # type: ignore[attr-defined]
                if nbr.GetIdx() != path[-2]
            ]
            if len(first_outside) != 1 or len(last_outside) != 1:
                return None
            host_first, host_second = first_outside[0], last_outside[0]
        rw = Chem.RWMol(mol)
        rw.RemoveBond(path[0], host_first)
        rw.RemoveBond(path[-1], host_second)
        d_first = rw.AddAtom(Chem.Atom(0))
        d_second = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(host_first, d_first, Chem.BondType.SINGLE)
        rw.AddBond(host_second, d_second, Chem.BondType.SINGLE)
        cut = rw.GetMol()
        idx_frags = Chem.GetMolFrags(cut)
        mol_frags = Chem.GetMolFrags(cut, asMols=True)
        if len(idx_frags) != 3 or len(mol_frags) != 3:
            return None  # a detour around the middle closed a ring: not excisable
        flank_first = flank_second = None
        for indices, fragment in zip(idx_frags, mol_frags):
            if d_first in indices:
                flank_first = fragment
            if d_second in indices:
                flank_second = fragment
        if flank_first is None or flank_second is None:  # pragma: no cover
            return None
        for fragment, map_num in ((flank_first, 1), (flank_second, 2)):
            Chem.SanitizeMol(fragment)
            dummies = [
                atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0
            ]
            if len(dummies) != 1:  # pragma: no cover
                return None
            dummies[0].SetAtomMapNum(map_num)
            dummies[0].SetIsotope(0)
        return flank_first, flank_second
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None


def _excise_ring(
    mol: object,
    ring: tuple[int, ...],
    a1: int,
    a2: int,
) -> tuple[object, tuple] | None:
    """Cut a whole ring loose from its two anchor bonds; ``(piece, host_sigs)``.

    The Nicolaou fused-ring move's donor half: ``ring`` is a ring of
    ``mol`` attached to the rest of the molecule through exactly two
    single anchor bonds -- ``a1``-host and ``a2``-host, at two DISTINCT
    ring atoms with exactly one outside neighbour each (a spiro-style
    single anchor carrying two outside bonds, or a tri-substituted ring
    with three, has no two-ended semantics and never excises).  The
    anchor bonds may themselves be ring bonds of an ADJACENT ring --
    that is exactly the fused-system case (decalin's left ring hangs off
    the right one through two bonds of the right ring) -- but they must
    be single (an aromatic fusion bond, e.g. indole's benzene-pyrrole
    seam, is never cut).  The returned piece is the intact ring carrying
    ``[*:1]`` at ``a1`` (replacing ``a1``'s former host) and ``[*:2]`` at
    ``a2``, ready to be spliced across an opened ring bond of the other
    parent; ``host_sigs`` are the two former hosts' signatures in
    ``(a1, a2)`` order for the homology check.  ``None`` on any
    validation failure (hand-made ``ring`` tuples, dummies involved,
    unsanitisable piece) -- robustness first.

    Valence bookkeeping is exact at both seams: each anchor trades one
    single-bonded outside host for one single-bonded dummy, every ring
    bond is left untouched, so the piece sanitises exactly like the ring
    sat in the parent.  Aromatic rings survive intact -- only the two
    anchor bonds are cut, and they are single by construction.
    """
    Chem = require_rdkit()
    try:
        ring_set = set(ring)
        if len(ring_set) != len(ring) or len(ring) < 3:
            return None  # not a ring: repeating index or too few atoms
        if a1 == a2 or a1 not in ring_set or a2 not in ring_set:
            return None  # anchors must be two distinct ring atoms
        outside: dict[int, list[int]] = {}
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)  # type: ignore[attr-defined]
            if atom.GetAtomicNum() == 0:
                return None  # a parent's dummy is never transplantable
            outs = [
                nbr.GetIdx()
                for nbr in atom.GetNeighbors()  # type: ignore[attr-defined]
                if nbr.GetIdx() not in ring_set
            ]
            if outs:
                outside[idx] = outs
        if sum(len(v) for v in outside.values()) != 2:
            return None  # a two-ended piece needs exactly two outside bonds
        if sorted(len(v) for v in outside.values()) != [1, 1]:
            return None  # ...at two distinct anchors, one bond each
        if set(outside) != {a1, a2}:
            return None  # the caller's anchors are not the ring's anchors
        hosts: list[int] = []
        for anchor in (a1, a2):
            host = outside[anchor][0]
            if mol.GetAtomWithIdx(host).GetAtomicNum() == 0:  # type: ignore[attr-defined]
                return None  # a dummy host would seam to another dummy
            bond = mol.GetBondBetweenAtoms(anchor, host)  # type: ignore[attr-defined]
            if (
                bond is None
                or bond.GetBondType() != Chem.BondType.SINGLE
            ):
                return None  # aromatic / double anchors are never cut
            hosts.append(host)
        rw = Chem.RWMol(mol)
        rw.RemoveBond(a1, hosts[0])
        rw.RemoveBond(a2, hosts[1])
        d1 = rw.AddAtom(Chem.Atom(0))
        d2 = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(a1, d1, Chem.BondType.SINGLE)
        rw.AddBond(a2, d2, Chem.BondType.SINGLE)
        # map numbers are atom properties: they survive GetMolFrags, so
        # the two dummies stay identifiable after the renumbering
        rw.GetAtomWithIdx(d1).SetAtomMapNum(1)
        rw.GetAtomWithIdx(d2).SetAtomMapNum(2)
        cut = rw.GetMol()
        idx_frags = Chem.GetMolFrags(cut)
        mol_frags = Chem.GetMolFrags(cut, asMols=True)
        piece = None
        for indices, fragment in zip(idx_frags, mol_frags):
            if d1 in indices and d2 in indices:
                piece = fragment
        if piece is None:  # pragma: no cover
            return None  # both dummies sit on ring atoms: same fragment
        if piece.GetNumAtoms() != len(ring) + 2:  # pragma: no cover
            return None  # belt-and-braces: the piece is the ring + 2 dummies
        Chem.SanitizeMol(piece)
        if piece.GetRingInfo().NumRings() < 1:  # pragma: no cover
            return None  # belt-and-braces: the piece keeps the ring cycle
        return piece, (
            _atom_signature(mol.GetAtomWithIdx(hosts[0])),  # type: ignore[attr-defined]
            _atom_signature(mol.GetAtomWithIdx(hosts[1])),  # type: ignore[attr-defined]
        )
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None


def _ring_units(mol: object) -> list[tuple[tuple[int, ...], int, int, tuple]]:
    """Enumerate excisable rings: ``[(ring, a1, a2, host_sigs)]``.

    Every SSSR ring whose total attachment to the rest of the molecule
    is exactly two single anchor bonds at two distinct ring atoms (one
    each) is a candidate donor for the fuse geometry -- a saturated
    fused ring (decalin, hydrindane), an aromatic ring hanging off an
    aliphatic ring (indane's benzene, through its two single aryl-CH2
    bonds), a bridged loop (norbornane's 5-rings), or a plain
    disubstituted monocycle.  Rings with zero, one, three or more
    outside bonds, spiro anchors, aromatic or double anchor bonds, or
    dummy involvement never appear.

    Determinism: rings are visited in ``RingInfo().AtomRings()`` order
    (a pure function of the molecule), and each admitted ring
    contributes exactly one unit with its two anchors in atom-index
    order -- a pure function of the molecule, so the crossover's
    candidate pool stays a pure function of the parents.
    """
    units: list[tuple[tuple[int, ...], int, int, tuple]] = []
    ri = mol.GetRingInfo()  # type: ignore[attr-defined]
    for ring in ri.AtomRings():
        ring_set = set(ring)
        outside: dict[int, list[int]] = {}
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)  # type: ignore[attr-defined]
            if atom.GetAtomicNum() == 0:
                outside = {}
                break
            outs = [
                nbr.GetIdx()
                for nbr in atom.GetNeighbors()  # type: ignore[attr-defined]
                if nbr.GetIdx() not in ring_set
            ]
            if outs:
                outside[idx] = outs
        if sum(len(v) for v in outside.values()) != 2:
            continue
        if sorted(len(v) for v in outside.values()) != [1, 1]:
            continue
        a1, a2 = sorted(outside)
        if _excise_ring(mol, tuple(ring), a1, a2) is None:
            continue  # anchor bonds not single, or a dummy host
        hosts = (outside[a1][0], outside[a2][0])
        units.append(
            (
                tuple(ring),
                a1,
                a2,
                (
                    _atom_signature(mol.GetAtomWithIdx(hosts[0])),  # type: ignore[attr-defined]
                    _atom_signature(mol.GetAtomWithIdx(hosts[1])),  # type: ignore[attr-defined]
                ),
            )
        )
    return units


def _zip_child(
    frag_x: object,
    frag_y: object,
    max_smiles_length: int,
    min_heavy_atoms: int = 1,
) -> str | None:
    """Zip two prepared fragments into one child SMILES, or ``None``.

    Belt-and-braces: a child that still carries an attachment dummy
    (atomic number 0) is rejected -- no operator of this module may
    ever leak a ``*`` into an offspring (round 100x-28 review finding
    1) -- and a child below ``min_heavy_atoms`` heavy atoms (atomic
    number > 1) is rejected by the sanity gate (round 100x-53; the
    default of 1 keeps the gate off and the historical behaviour
    byte-identical).
    """
    Chem = require_rdkit()
    try:
        child = Chem.molzip(frag_x, frag_y)
        Chem.SanitizeMol(child)
        if any(atom.GetAtomicNum() == 0 for atom in child.GetAtoms()):
            return None
        heavy = sum(1 for atom in child.GetAtoms() if atom.GetAtomicNum() > 1)
        if heavy < min_heavy_atoms:
            return None
        smiles = Chem.MolToSmiles(child)
    except Exception:  # noqa: BLE001 - no viable child
        return None
    if smiles and len(smiles) <= max_smiles_length:
        return smiles
    return None


def _zip_spliced(
    frag_begin: object,
    piece: object,
    frag_end: object,
    max_smiles_length: int,
    min_heavy_atoms: int = 1,
) -> str | None:
    """Splice ``piece`` between two bridge fragments; child SMILES/``None``.

    The insert geometry's assembler: ``frag_begin`` and ``frag_end``
    are the two halves of a bridge cut (the begin fragment's dummy is
    ``[*:1]``, the end fragment's was relabelled to ``[*:2]``) and
    ``piece`` is a two-ended excised ring (``[*:1]`` at one anchor,
    ``[*:2]`` at the other).  Two ``Chem.molzip`` calls chain them --
    begin to the piece's map-1 side, then the composite to the end
    fragment -- so both seams bond signature-matched atoms and no dummy
    survives.  Same belt-and-braces contract as :func:`_zip_child`,
    including the ``min_heavy_atoms`` sanity gate.
    """
    Chem = require_rdkit()
    try:
        half = Chem.molzip(frag_begin, piece)
        child = Chem.molzip(half, frag_end)
        Chem.SanitizeMol(child)
        if any(atom.GetAtomicNum() == 0 for atom in child.GetAtoms()):
            return None
        heavy = sum(1 for atom in child.GetAtoms() if atom.GetAtomicNum() > 1)
        if heavy < min_heavy_atoms:
            return None
        smiles = Chem.MolToSmiles(child)
    except Exception:  # noqa: BLE001 - no viable child
        return None
    if smiles and len(smiles) <= max_smiles_length:
        return smiles
    return None


def scaffold_hop_crossover(
    parent_a: str,
    parent_b: str,
    rng: random.Random,
    *,
    max_smiles_length: int = 120,
    geometry_weights: Mapping[str, float] | None = None,
    min_heavy_atoms: int = 1,
) -> str | None:
    """BREED-style crossover: homologous cuts, hybrid child.

    Six cut geometries compete for the rng draw -- a **bridge** cut
    per parent (fragment swap, the classic), a **ring merge** (one
    single ring bond cut per parent, the opened rings zipped into one
    combined ring), a **linker transplant** (a single ring bond cut
    in one parent, a two-ended linker -- a single atom or a contiguous
    acyclic path of up to four atoms -- excised from the other and
    zipped into the cut: ring homologation by one atom or a whole
    chain), a **fused-ring splice** (a whole ring excised from one
    parent between its two single anchor bonds and spliced across an
    opened ring bond of the other: a fused / bridged bicyclic child,
    the full Nicolaou move set), a **ring insert** (a whole ring
    excised from one parent and spliced INTO a cut acyclic bond of the
    other: both bridge halves re-joined through the ring -- the
    ring-as-linker move) and a **segment swap** (each parent's bounded
    acyclic middle -- a single atom or a contiguous path of up to four
    atoms -- excised from the flank parent and replaced by the other
    parent's homologous middle: the only size-conserving geometry, the
    linker / bioisostere exchange -- an ester O for an amide N, an
    ether O for a thioether S).  Aromatic ring bonds are never cut
    (the re-zip would have to invent bond orders).

    Parameters
    ----------
    parent_a, parent_b:
        Canonical (or any parseable) SMILES of the two parents.  Both must
        be single connected molecules -- a dot-disconnected parent has no
        meaningful fragment recombination and yields ``None``.
    rng:
        The caller's dedicated ``random.Random`` instance (the GA passes
        its own; standalone callers pass ``random.Random(seed)``).  The
        returned child is a pure function of the parents, the cap and the
        rng state -- same state, same child.
    max_smiles_length:
        Hard cap on the child's canonical SMILES length (default 120, the
        :class:`~pharmaml.generative.GeneticOptimizer` default).  All
        viable pairings of a drawn bridge candidate are tried before
        giving up.
    min_heavy_atoms:
        Minimum heavy-atom (atomic number > 1) sanity gate on the child,
        opt-in (default 1 = gate off, the historical behaviour
        byte-identical).  Motivating comic: aspirin x naproxen breeds
        ``'OO'`` (hydrogen peroxide) or ``'CC'`` (ethane) as bridge
        children on roughly a quarter of draw states -- single-fragment,
        sanitisable, under the cap, i.e. contract-conformant but with no
        medchem meaning.  ``min_heavy_atoms=3`` rejects exactly that
        <=2-atom class while keeping every remotely drug-like child;
        a gated-out child is not an error -- the bridge loop tries the
        next pairing, and the operator returns ``None`` when nothing
        survives.  Plain and numpy integers are accepted; bools and
        values below 1 raise ``ValueError``.
    geometry_weights:
        Optional per-geometry draw bias: a mapping
        ``{geometry name: non-negative weight}`` over the seven names in
        :data:`GEOMETRY_ORDER`.  The geometry is picked with probability
        proportional to its weight renormalised over the geometries
        present in this pair's candidate pool, then the candidate
        uniformly within the geometry (one ``rng.random()`` plus one
        ``rng.randrange()``); keys naming absent geometries simply place
        no mass, and a geometry left at weight zero is never drawn.
        ``None`` (the default) keeps the historical uniform draw over
        the flat candidate pool -- note the distinction: an empty
        mapping ``{}`` is NOT "no weights" but a weights mapping that
        places no mass on any geometry, so it raises the no-mass
        ``ValueError`` (pass ``None``, or omit the keyword, for the
        uniform draw).  Raises ``ValueError`` on negative /
        non-finite weights or when the weights place no mass on any
        geometry present in the pool.

    Returns
    -------
    str | None
        Canonical SMILES of the hybrid child, or ``None`` when no viable
        crossover exists (unparseable parent, no homologous cut pair,
        children unsanitisable or over the cap).  ``None`` is a
        "try the other operator" signal, never an error.

    Example
    -------
    >>> from pharmaml.generative import scaffold_hop_crossover
    >>> import random
    >>> # aspirin x benzamide: homologous bridge cuts, hybrids come out
    >>> child = scaffold_hop_crossover("CC(=O)Oc1ccccc1C(=O)O",
    ...                                 "NC(=O)c1ccccc1", random.Random(0))
    >>> child is not None
    True
    >>> # ring recombination: piperidine x diethyl ether -- the ether O
    >>> # is transplanted into a cut C-C ring bond -> a 7-ring oxazepane
    >>> child = scaffold_hop_crossover("C1CCNCC1", "CCOCC", random.Random(0))
    >>> child is not None
    True
    >>> # benzene has no acyclic single bond and no cuttable ring bond
    >>> scaffold_hop_crossover("c1ccccc1", "c1ccncc1", random.Random(0)) is None
    True
    """
    child, _geometry = _scaffold_hop_core(
        parent_a,
        parent_b,
        rng,
        max_smiles_length=max_smiles_length,
        geometry_weights=geometry_weights,
        min_heavy_atoms=min_heavy_atoms,
    )
    return child


def _scaffold_hop_core(
    parent_a: str,
    parent_b: str,
    rng: random.Random,
    *,
    max_smiles_length: int = 120,
    geometry_weights: Mapping[str, float] | None = None,
    min_heavy_atoms: int = 1,
) -> tuple[str | None, str | None]:
    """BREED-style crossover core, traced for adaptive operator selection.

    Same cuts, robustness contract and rng discipline as
    :func:`scaffold_hop_crossover` (whose docstring carries the full
    geometry narrative), with two additions the GA's adaptive mode
    needs:

    * the return carries the **drawn geometry** next to the child, so a
      credit-assignment loop can reward the operator that actually
      fired -- ``(None, geometry)`` when the drawn candidate produced no
      viable child (the product was discarded, not the operator) and
      ``(None, None)`` when no candidate existed at all;
    * ``geometry_weights`` selects the geometry to draw: one
      ``rng.random()`` over the renormalised weights in
      :data:`GEOMETRY_ORDER` order, then one ``rng.randrange()`` for the
      uniform within-geometry candidate pick.  ``None`` keeps the
      historical single uniform draw over the flat candidate pool
      (byte-identical line -- the default GA path must never move).

    The weights are validated eagerly (negative / non-finite values
    raise ``ValueError`` before any RDKit work); the no-mass-on-present
    check raises from the draw, between the scan and the execution, so
    neither robustness ``except`` can swallow a caller's configuration
    error.
    """
    Chem = require_rdkit()
    from rdkit.rdBase import DisableLog, EnableLog

    if max_smiles_length < _MIN_LENGTH_CAP:
        raise ValueError(
            f"max_smiles_length must be >= {_MIN_LENGTH_CAP}, "
            f"got {max_smiles_length}"
        )
    min_heavy_atoms = _checked_min_heavy_atoms(min_heavy_atoms)
    if geometry_weights is not None:
        for geometry, weight in geometry_weights.items():
            try:
                value = float(weight)  # huge ints overflow isfinite itself
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    "geometry_weights values must be finite non-negative "
                    f"numbers, got {weight!r} for geometry {geometry!r}"
                ) from None
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "geometry_weights values must be finite non-negative "
                    f"numbers, got {weight!r} for geometry {geometry!r}"
                )

    # Unsanitisable recombinations are EXPECTED (the caller falls back);
    # RDKit logs every sanitisation failure to stderr -- keep the GA loop
    # quiet (the canonical_smiles / _scaffold_mutate precedent).
    DisableLog("rdApp.error")
    try:
        mol_a = Chem.MolFromSmiles(parent_a)
        mol_b = Chem.MolFromSmiles(parent_b)
        if mol_a is None or mol_b is None:
            return None, None
        if len(Chem.GetMolFrags(mol_a)) != 1 or len(Chem.GetMolFrags(mol_b)) != 1:
            return None, None  # dot-disconnected parent: no recombination

        sig_a = {atom.GetIdx(): _atom_signature(atom) for atom in mol_a.GetAtoms()}
        sig_b = {atom.GetIdx(): _atom_signature(atom) for atom in mol_b.GetAtoms()}

        # Homologous cut candidates, four geometries in a fixed scan
        # order (determinism: the pool is a pure function of the parents;
        # the rng draw then picks one uniformly).
        #
        # bridge: single acyclic bonds whose endpoints match in some
        # orientation (the classic fragment swap -- ring bonds excluded
        # here because a bridge-style cut must split the molecule).
        candidates: list[tuple] = []
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

        # merge: single RING bonds whose endpoints match -- the opened
        # rings zip into one combined ring (atom count = both rings').
        # The orientation matters: a direct match zips begin-begin and
        # end-end; a flipped-only match must zip begin-end so each seam
        # bond joins signature-matched atoms (the module's invariant).
        ring_bonds_a = [
            bond for bond in mol_a.GetBonds()
            if bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE
        ]
        ring_bonds_b = [
            bond for bond in mol_b.GetBonds()
            if bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE
        ]
        for bond_a in ring_bonds_a:
            a1, a2 = bond_a.GetBeginAtomIdx(), bond_a.GetEndAtomIdx()
            for bond_b in ring_bonds_b:
                b1, b2 = bond_b.GetBeginAtomIdx(), bond_b.GetEndAtomIdx()
                direct = sig_a[a1] == sig_b[b1] and sig_a[a2] == sig_b[b2]
                flipped = sig_a[a1] == sig_b[b2] and sig_a[a2] == sig_b[b1]
                if direct or flipped:
                    candidates.append(("merge", bond_a.GetIdx(), bond_b.GetIdx(), direct))

        # transplant: a single ring bond in one parent + a two-ended
        # linker in the other -- a single atom or a multi-atom path
        # (homologous flanks: the linker's former outside hosts must
        # match the cut bond's endpoints, so the linker lands in a
        # chemically analogous environment).  Both argument orders are
        # scanned.  For single atoms the orientation is admission-only
        # (the anchor bonds to BOTH ring endpoints, so the child's edge
        # set is orientation-independent); a *path* is asymmetric, so
        # flipped-only matches must excise the reversed path at
        # execution time (handled below).
        for ring_side, other, ring_bonds, sig_ring in (
            ("a", mol_b, ring_bonds_a, sig_a),
            ("b", mol_a, ring_bonds_b, sig_b),
        ):
            for bond in ring_bonds:
                r1, r2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                for atom in other.GetAtoms():
                    if atom.GetDegree() != 2 or atom.IsInRing():
                        continue
                    excision = _excise_linker_path(other, (atom.GetIdx(),))
                    if excision is None:
                        continue
                    _, host_sigs = excision
                    direct = host_sigs[0] == sig_ring[r1] and host_sigs[1] == sig_ring[r2]
                    flipped = host_sigs[0] == sig_ring[r2] and host_sigs[1] == sig_ring[r1]
                    if direct or flipped:
                        candidates.append(
                            ("transplant", ring_side, bond.GetIdx(), atom.GetIdx())
                        )
            # the same ring bonds against multi-atom linker paths of the
            # donor (round 100x-29): ring expansion by 2-4 atoms.  The
            # guard skips the enumeration for acyclic donors (no ring
            # bond to transplant into -- the units would be dead weight)
            if ring_bonds:
                for path, host_sigs in _linker_path_units(other):
                    for bond in ring_bonds:
                        r1, r2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        direct = host_sigs[0] == sig_ring[r1] and host_sigs[1] == sig_ring[r2]
                        flipped = host_sigs[0] == sig_ring[r2] and host_sigs[1] == sig_ring[r1]
                        if direct or flipped:
                            candidates.append(
                                ("transplant_path", ring_side, bond.GetIdx(), path, direct)
                            )
                # the same ring bonds against whole excisable RINGS of
                # the donor (round 100x-30): the fused-ring splice, the
                # Nicolaou move -- the donor ring is cut loose from its
                # two single anchor bonds and spliced across the opened
                # bond, giving a fused / bridged bicyclic child.  The
                # anchor bonds' former hosts must match the cut bond's
                # endpoints (homologous flanks), direct or flipped; a
                # flipped-only match swaps the piece's dummy labels at
                # execution time so each seam bonds signature-matched
                # atoms (the merge geometry's label-swap trick).
                for ring, a1, a2, host_sigs in _ring_units(other):
                    for bond in ring_bonds:
                        r1, r2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        direct = host_sigs[0] == sig_ring[r1] and host_sigs[1] == sig_ring[r2]
                        flipped = host_sigs[0] == sig_ring[r2] and host_sigs[1] == sig_ring[r1]
                        if direct or flipped:
                            candidates.append(
                                ("fuse", ring_side, bond.GetIdx(), ring, a1, a2, direct)
                            )

        # insert (round 100x-31): the matrix's last cell -- a whole
        # excisable ring of one parent spliced INTO a cut acyclic bond
        # of the other.  The bridge-style cut splits the recipient into
        # two halves, and both halves are re-joined THROUGH the excised
        # ring: the ring becomes the spacer (ring-as-linker).  The
        # donor ring's former outside hosts must match the cut bond's
        # endpoints (homologous flanks), direct or flipped; a
        # flipped-only match swaps the piece's dummy labels at
        # execution time (the fuse geometry's label-swap trick).
        for donor_side, donor, recipient, sig_recipient in (
            ("a", mol_a, mol_b, sig_b),
            ("b", mol_b, mol_a, sig_a),
        ):
            acyclic_bonds = [
                bond for bond in recipient.GetBonds()
                if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE
            ]
            if not acyclic_bonds:
                continue  # no bond to splice into -- units are dead weight
            for ring, a1, a2, host_sigs in _ring_units(donor):
                for bond in acyclic_bonds:
                    b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    direct = (
                        host_sigs[0] == sig_recipient[b1]
                        and host_sigs[1] == sig_recipient[b2]
                    )
                    flipped = (
                        host_sigs[0] == sig_recipient[b2]
                        and host_sigs[1] == sig_recipient[b1]
                    )
                    if direct or flipped:
                        candidates.append(
                            ("insert", donor_side, bond.GetIdx(), ring, a1, a2, direct)
                        )

        # swap (round 100x-35): the sixth geometry -- two-point
        # segment-swap crossover, the graph analogue of classic
        # two-point GA recombination.  Each parent's bounded acyclic
        # MIDDLE (a single atom or a contiguous path of up to four, the
        # transplant eligibility) is excised from the flank parent and
        # REPLACED by the piece parent's homologous middle: the only
        # size-conserving geometry (insert grows a recipient, swap
        # exchanges in place) and the medchem linker / bioisostere
        # exchange move (an ester O swapped for an amide N: aspirin x
        # paracetamol -> a salicylamide child; an ether O for a
        # thioether S).  Homology: the two middles' host signatures must
        # match direct or flipped, so each assembly seam bonds a
        # signature-matched host to the incoming middle's end atom --
        # the same invariant every other geometry enforces.  Both
        # argument orders are scanned; a flipped-only match excises the
        # REVERSED piece path at execution time (the transplant_path
        # trick; a single-atom middle is its own reversal).
        middle_units_a = _middle_units(mol_a)
        middle_units_b = _middle_units(mol_b)
        for flank_side, flank_units, piece_units in (
            ("a", middle_units_a, middle_units_b),
            ("b", middle_units_b, middle_units_a),
        ):
            for flank_path, flank_sigs in flank_units:
                for piece_path, piece_sigs in piece_units:
                    direct = flank_sigs == piece_sigs
                    flipped = (
                        flank_sigs[0] == piece_sigs[1]
                        and flank_sigs[1] == piece_sigs[0]
                    )
                    if direct or flipped:
                        candidates.append(
                            ("swap", flank_side, flank_path, piece_path, direct)
                        )
        if not candidates:
            return None, None
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None, None
    finally:
        EnableLog("rdApp.error")

    if geometry_weights is None:
        # uniform mode: the historical flat-pool draw, byte-identical --
        # the default GA path must never move.
        choice = candidates[rng.randrange(len(candidates))]
    else:
        # weighted mode (the adaptive GA passes its live selection
        # probabilities): one rng.random() picks the geometry -- weights
        # renormalised over the geometries PRESENT in this pair's pool,
        # GEOMETRY_ORDER iteration -- and one rng.randrange() picks the
        # candidate uniformly within the drawn geometry's bucket.
        buckets: dict[str, list[int]] = {}
        for index, candidate in enumerate(candidates):
            buckets.setdefault(candidate[0], []).append(index)
        present = [name for name in GEOMETRY_ORDER if name in buckets]
        weights = [float(geometry_weights.get(name, 0.0)) for name in present]
        total = sum(weights)
        if total <= 0.0:
            raise ValueError(
                "geometry_weights place no mass on the geometries present "
                f"in the candidate pool (present: {present}, weights: "
                f"{dict(geometry_weights)})"
            )
        pick = rng.random() * total
        cumulative = 0.0
        drawn = present[0]
        for name, weight in zip(present, weights):
            if weight > 0.0:
                drawn = name  # float-tail guard: last positive-weight geometry
            cumulative += weight
            if pick < cumulative:
                break
        bucket = buckets[drawn]
        choice = candidates[bucket[rng.randrange(len(bucket))]]
    drawn_geometry = choice[0]

    # Unsanitisable recombinations are EXPECTED (the caller falls back);
    # RDKit logs every sanitisation failure to stderr -- keep the GA loop
    # quiet (the canonical_smiles / _scaffold_mutate precedent).
    DisableLog("rdApp.error")
    try:
        if choice[0] == "bridge":
            _, bond_a_idx, bond_b_idx, direct, flipped = choice
            cut_a = _cut_into_two(mol_a, bond_a_idx)
            cut_b = _cut_into_two(mol_b, bond_b_idx)
            if cut_a is None or cut_b is None:
                return None, drawn_geometry
            frag_a_begin, frag_a_end = cut_a
            frag_b_begin, frag_b_end = cut_b
            # Eligible pairings join signature-matched atoms: begin-begin
            # and end-end when the direct orientation matched, begin-end
            # and end-begin when only the flipped one did (all four when
            # both).
            pairings: list[tuple[object, object]] = []
            if direct:
                pairings.append((frag_a_begin, frag_b_begin))
                pairings.append((frag_a_end, frag_b_end))
            if flipped:
                pairings.append((frag_a_begin, frag_b_end))
                pairings.append((frag_a_end, frag_b_begin))
            rng.shuffle(pairings)
            for frag_x, frag_y in pairings:
                child = _zip_child(frag_x, frag_y, max_smiles_length, min_heavy_atoms)
                if child is not None:
                    return child, drawn_geometry
            return None, drawn_geometry

        if choice[0] == "merge":
            _, bond_a_idx, bond_b_idx, direct = choice
            open_a = _open_ring(mol_a, bond_a_idx)
            open_b = _open_ring(mol_b, bond_b_idx)
            if open_a is None or open_b is None:
                return None, drawn_geometry
            if not direct:
                # flipped-only match: swap open_b's dummy labels so the
                # seams join begin_a-end_b / end_a-begin_b -- the
                # signature-matched orientation
                for dummy in open_b.GetAtoms():  # type: ignore[attr-defined]
                    if dummy.GetAtomicNum() == 0:
                        dummy.SetAtomMapNum(3 - dummy.GetAtomMapNum())
            return _zip_child(open_a, open_b, max_smiles_length, min_heavy_atoms), drawn_geometry

        # transplant (single-atom linker: orientation-independent)
        if choice[0] == "transplant":
            _, ring_side, ring_bond_idx, donor_atom_idx = choice
            if ring_side == "a":
                ring_mol, donor_mol = mol_a, mol_b
            else:
                ring_mol, donor_mol = mol_b, mol_a
            opened = _open_ring(ring_mol, ring_bond_idx)
            excision = _excise_linker_path(donor_mol, (donor_atom_idx,))
            if opened is None or excision is None:
                return None, drawn_geometry
            donor, _ = excision
            return _zip_child(opened, donor, max_smiles_length, min_heavy_atoms), drawn_geometry

        # transplant_path (multi-atom linker: orientation-aware)
        if choice[0] == "transplant_path":
            _, ring_side, ring_bond_idx, path, direct = choice
            if ring_side == "a":
                ring_mol, donor_mol = mol_a, mol_b
            else:
                ring_mol, donor_mol = mol_b, mol_a
            opened = _open_ring(ring_mol, ring_bond_idx)
            # a flipped-only match must excise the REVERSED path so each
            # seam bonds a path atom to a signature-matched ring endpoint
            # (the single-atom transplant is symmetric -- reversing a
            # one-atom path is the identity -- which is why orientation
            # could be admission-only before paths existed)
            excision = _excise_linker_path(
                donor_mol, path if direct else tuple(reversed(path))
            )
            if opened is None or excision is None:
                return None, drawn_geometry
            donor, _ = excision
            return _zip_child(opened, donor, max_smiles_length, min_heavy_atoms), drawn_geometry

        # fuse (whole-ring splice: orientation-aware like the merge -- a
        # flipped-only match swaps the piece's dummy labels so each seam
        # bonds an anchor to a signature-matched ring endpoint; excising
        # with the anchors swapped instead would build the same piece,
        # so the label swap is the cheaper bookkeeping)
        if choice[0] == "fuse":
            _, ring_side, ring_bond_idx, ring, a1, a2, direct = choice
            if ring_side == "a":
                ring_mol, donor_mol = mol_a, mol_b
            else:
                ring_mol, donor_mol = mol_b, mol_a
            opened = _open_ring(ring_mol, ring_bond_idx)
            excision = _excise_ring(donor_mol, ring, a1, a2)
            if opened is None or excision is None:
                return None, drawn_geometry
            donor, _ = excision
            if not direct:
                # flipped-only match: swap the piece's dummy labels so the
                # seams join a1-end_b / a2-begin_b -- the signature-matched
                # orientation (the merge geometry's label-swap trick)
                for dummy in donor.GetAtoms():  # type: ignore[attr-defined]
                    if dummy.GetAtomicNum() == 0:
                        dummy.SetAtomMapNum(3 - dummy.GetAtomMapNum())
            return _zip_child(opened, donor, max_smiles_length), drawn_geometry

        # swap (two-point segment-swap: the flank's middle is excised and
        # REPLACED by the piece's homologous middle -- the only
        # size-conserving geometry, the linker / bioisostere exchange).
        # _excise_middle already labels the flanks [*:1]/[*:2], so the
        # reused _zip_spliced assembler chains them through the piece's
        # two dummies with no relabelling (the insert geometry's end-frag
        # relabel is a no-op on pre-labelled map-2 dummies).  A
        # flipped-only match excises the REVERSED piece path so each seam
        # bonds a signature-matched host to the middle's end atom -- the
        # transplant_path trick; a single-atom middle is its own
        # reversal.
        if choice[0] == "swap":
            _, flank_side, flank_path, piece_path, direct = choice
            if flank_side == "a":
                flank_mol, piece_mol = mol_a, mol_b
            else:
                flank_mol, piece_mol = mol_b, mol_a
            flanks = _excise_middle(flank_mol, flank_path)
            excision = _excise_linker_path(
                piece_mol, piece_path if direct else tuple(reversed(piece_path))
            )
            if flanks is None or excision is None:
                return None, drawn_geometry
            flank_first, flank_second = flanks
            piece, _ = excision
            return _zip_spliced(
                flank_first, piece, flank_second, max_smiles_length, min_heavy_atoms
            ), drawn_geometry

        # insert (ring-as-linker: the recipient's bridge cut leaves two
        # [*:1]-dummied halves; the end half's dummy is relabelled to
        # [*:2] so the excised ring splices BETWEEN them -- begin to
        # the ring's map-1 side, then the composite to the end half.
        # A flipped-only match swaps the ring's own labels so each seam
        # bonds an anchor to a signature-matched endpoint)
        _, donor_side, bond_idx, ring, a1, a2, direct = choice
        if donor_side == "a":
            donor_mol, recipient_mol = mol_a, mol_b
        else:
            donor_mol, recipient_mol = mol_b, mol_a
        cut = _cut_into_two(recipient_mol, bond_idx)
        excision = _excise_ring(donor_mol, ring, a1, a2)
        if cut is None or excision is None:
            return None, drawn_geometry
        begin_frag, end_frag = cut
        piece, _ = excision
        for atom in end_frag.GetAtoms():  # type: ignore[attr-defined]
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(2)
        if not direct:
            for dummy in piece.GetAtoms():  # type: ignore[attr-defined]
                if dummy.GetAtomicNum() == 0:
                    dummy.SetAtomMapNum(3 - dummy.GetAtomMapNum())
        return _zip_spliced(
            begin_frag, piece, end_frag, max_smiles_length, min_heavy_atoms
        ), drawn_geometry
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None, drawn_geometry
    finally:
        EnableLog("rdApp.error")
