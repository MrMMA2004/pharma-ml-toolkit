"""BREED-style scaffold-hop crossover for the genetic algorithm.

The structural recombination operator for
:class:`~pharmaml.generative.GeneticOptimizer`: two parent molecules are
cut and recombined so an offspring is a hybrid that keeps parts of each
parent (Nicolaou et al.'s BREED algorithm tradition, in the pragmatic
spirit of this package: RDKit graph surgery + ``Chem.molzip``, no
learned model).  Three cut geometries cover the two things a scaffold
hop actually has to recombine -- fragment *arrangement* and ring
*topology*:

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
  two-ended linker atom from the other parent (acyclic, two single
  bonds, homologous flanks -- an ether O, a methylene, a secondary
  amine N, ...) is excised and zipped into the cut: ring homologation
  (piperidine + an ether oxygen -> an oxazepane).

All cuts are *single* bonds (aromatic ring bonds are never cut -- the
re-zip would have to invent bond orders), and the join points must be
**homologous**: the atoms joined by each new bond match on (atomic
number, degree, aromaticity, formal charge), so recombination happens
at chemically analogous positions (an aryl-CH2 bond crosses with
another aryl-CH2 bond, not with a sulfonamide S-N).

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
>>> # benzene has no acyclic single bond to cut: no crossover possible
>>> scaffold_hop_crossover("c1ccccc1", "c1ccncc1", random.Random(0)) is None
True
"""

from __future__ import annotations

import random

from ._utils import require_rdkit

#: Minimum ``max_smiles_length`` the operator accepts (matches the
#: :class:`~pharmaml.generative.GeneticOptimizer` floor, so the standalone
#: use cannot be configured weaker than the GA use).
_MIN_LENGTH_CAP = 20


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


def _excise_linker_atom(mol: object, atom_idx: int) -> tuple[object, tuple] | None:
    """Cut a two-ended linker atom loose; return ``(donor, host_sigs)``.

    A linker is an atom with exactly two neighbours, both reached via
    single acyclic bonds (an ether O, a methylene, a secondary amine
    N, a thioether S ...).  Attachment dummies (atomic number 0) are
    never linkers -- a parent's dummy must not be transplanted into a
    child.  The donor is a fresh molecule holding the
    atom and two dummies -- ``[*:1]`` replacing the bond to the first
    host, ``[*:2]`` the second -- ready to be zipped into a ring cut of
    the other parent.  ``host_sigs`` are the two former neighbours'
    signatures (in the atom's own neighbour order) for the homology
    check.  ``None`` when the atom is not a two-ended linker.
    """
    Chem = require_rdkit()
    try:
        atom = mol.GetAtomWithIdx(atom_idx)  # type: ignore[attr-defined]
        if atom.GetDegree() != 2 or atom.IsInRing():
            return None
        if atom.GetAtomicNum() == 0:  # attachment dummy: never a linker
            return None
        neighbours = list(atom.GetNeighbors())
        host_sigs = tuple(_atom_signature(nbr) for nbr in neighbours)
        for nbr in neighbours:
            bond = mol.GetBondBetweenAtoms(  # type: ignore[attr-defined]
                atom_idx, nbr.GetIdx()
            )
            if (
                bond is None
                or bond.IsInRing()
                or bond.GetBondType() != Chem.BondType.SINGLE
            ):
                return None
        rw = Chem.RWMol()
        anchor = Chem.Atom(atom)
        anchor.SetAtomMapNum(0)
        anchor.SetIsotope(0)
        anchor.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        new_idx = rw.AddAtom(anchor)
        d_first = rw.AddAtom(Chem.Atom(0))
        d_second = rw.AddAtom(Chem.Atom(0))
        rw.AddBond(new_idx, d_first, Chem.BondType.SINGLE)
        rw.AddBond(new_idx, d_second, Chem.BondType.SINGLE)
        donor = rw.GetMol()
        Chem.SanitizeMol(donor)
        # The donor was hand-built (anchor, first dummy, second dummy in
        # exactly that atom order), so the dummy indices are fixed: label
        # the first-host-side dummy [*:1], the second [*:2].
        donor.GetAtomWithIdx(1).SetAtomMapNum(1)
        donor.GetAtomWithIdx(2).SetAtomMapNum(2)
        return donor, host_sigs
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None


def _zip_child(
    frag_x: object,
    frag_y: object,
    max_smiles_length: int,
) -> str | None:
    """Zip two prepared fragments into one child SMILES, or ``None``.

    Belt-and-braces: a child that still carries an attachment dummy
    (atomic number 0) is rejected -- no operator of this module may
    ever leak a ``*`` into an offspring (round 100x-28 review finding 1).
    """
    Chem = require_rdkit()
    try:
        child = Chem.molzip(frag_x, frag_y)
        Chem.SanitizeMol(child)
        if any(atom.GetAtomicNum() == 0 for atom in child.GetAtoms()):
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
) -> str | None:
    """BREED-style crossover: homologous cuts, hybrid child.

    Three cut geometries compete for the rng draw -- a **bridge** cut
    per parent (fragment swap, the classic), a **ring merge** (one
    single ring bond cut per parent, the opened rings zipped into one
    combined ring) and a **linker transplant** (a single ring bond cut
    in one parent, a two-ended linker atom excised from the other and
    zipped into the cut: ring homologation).  Aromatic ring bonds are
    never cut (the re-zip would have to invent bond orders).

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
    Chem = require_rdkit()
    from rdkit.rdBase import DisableLog, EnableLog

    if max_smiles_length < _MIN_LENGTH_CAP:
        raise ValueError(
            f"max_smiles_length must be >= {_MIN_LENGTH_CAP}, "
            f"got {max_smiles_length}"
        )

    # Unsanitisable recombinations are EXPECTED (the caller falls back);
    # RDKit logs every sanitisation failure to stderr -- keep the GA loop
    # quiet (the canonical_smiles / _scaffold_mutate precedent).
    DisableLog("rdApp.error")
    try:
        mol_a = Chem.MolFromSmiles(parent_a)
        mol_b = Chem.MolFromSmiles(parent_b)
        if mol_a is None or mol_b is None:
            return None
        if len(Chem.GetMolFrags(mol_a)) != 1 or len(Chem.GetMolFrags(mol_b)) != 1:
            return None  # dot-disconnected parent: no recombination semantics

        sig_a = {atom.GetIdx(): _atom_signature(atom) for atom in mol_a.GetAtoms()}
        sig_b = {atom.GetIdx(): _atom_signature(atom) for atom in mol_b.GetAtoms()}

        # Homologous cut candidates, three geometries in a fixed scan
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
        # linker atom in the other (homologous flanks: the linker's
        # former neighbours must match the cut bond's endpoints, so the
        # linker lands in a chemically analogous environment).  Both
        # argument orders are scanned.  (Orientation is admission-only:
        # the anchor bonds to BOTH ring endpoints, so the child's edge
        # set is orientation-independent.)
        for ring_side, other, ring_bonds, sig_ring in (
            ("a", mol_b, ring_bonds_a, sig_a),
            ("b", mol_a, ring_bonds_b, sig_b),
        ):
            for bond in ring_bonds:
                r1, r2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                for atom in other.GetAtoms():
                    if atom.GetDegree() != 2 or atom.IsInRing():
                        continue
                    excision = _excise_linker_atom(other, atom.GetIdx())
                    if excision is None:
                        continue
                    _, host_sigs = excision
                    direct = host_sigs[0] == sig_ring[r1] and host_sigs[1] == sig_ring[r2]
                    flipped = host_sigs[0] == sig_ring[r2] and host_sigs[1] == sig_ring[r1]
                    if direct or flipped:
                        candidates.append(
                            ("transplant", ring_side, bond.GetIdx(), atom.GetIdx())
                        )
        if not candidates:
            return None

        choice = candidates[rng.randrange(len(candidates))]

        if choice[0] == "bridge":
            _, bond_a_idx, bond_b_idx, direct, flipped = choice
            cut_a = _cut_into_two(mol_a, bond_a_idx)
            cut_b = _cut_into_two(mol_b, bond_b_idx)
            if cut_a is None or cut_b is None:
                return None
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
                child = _zip_child(frag_x, frag_y, max_smiles_length)
                if child is not None:
                    return child
            return None

        if choice[0] == "merge":
            _, bond_a_idx, bond_b_idx, direct = choice
            open_a = _open_ring(mol_a, bond_a_idx)
            open_b = _open_ring(mol_b, bond_b_idx)
            if open_a is None or open_b is None:
                return None
            if not direct:
                # flipped-only match: swap open_b's dummy labels so the
                # seams join begin_a-end_b / end_a-begin_b -- the
                # signature-matched orientation
                for dummy in open_b.GetAtoms():  # type: ignore[attr-defined]
                    if dummy.GetAtomicNum() == 0:
                        dummy.SetAtomMapNum(3 - dummy.GetAtomMapNum())
            return _zip_child(open_a, open_b, max_smiles_length)

        # transplant
        _, ring_side, ring_bond_idx, donor_atom_idx = choice
        if ring_side == "a":
            ring_mol, donor_mol = mol_a, mol_b
        else:
            ring_mol, donor_mol = mol_b, mol_a
        opened = _open_ring(ring_mol, ring_bond_idx)
        excision = _excise_linker_atom(donor_mol, donor_atom_idx)
        if opened is None or excision is None:
            return None
        donor, _ = excision
        return _zip_child(opened, donor, max_smiles_length)
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None
    finally:
        EnableLog("rdApp.error")
