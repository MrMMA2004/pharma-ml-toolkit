"""BREED-style scaffold-hop crossover for the genetic algorithm.

The missing structural recombination operator for
:class:`~pharmaml.generative.GeneticOptimizer`: two parent molecules are
cut at a pair of *homologous* single acyclic bonds and the fragments are
recombined cross-wise, so an offspring is a hybrid that keeps one part of
each parent (Nicolaou et al.'s BREED algorithm tradition, in the pragmatic
spirit of this package: RDKit graph surgery + ``Chem.molzip``, no learned
model).

What makes it a *scaffold hop*:

* a character-level splice (the default crossover) routinely destroys
  ring systems, so the population's Murcko frameworks drift only by
  luck; here every cut is a **single acyclic (bridge) bond**, so each
  fragment arrives intact -- ring systems survive by construction;
* the cut points must be **homologous**: the atoms joined by the new bond
  match on (atomic number, degree, aromaticity, formal charge), so
  recombination happens at chemically analogous positions (an aryl-CH2
  bond crosses with another aryl-CH2 bond, not with a sulfonamide S-N);
* whenever the parents' frameworks span different fragments (biphenyls,
  diaryl ethers, sulfonyl bridges -- extremely common in drugs), the
  child's framework differs from BOTH parents: the offspring has hopped
  scaffolds.

Deterministic given the caller's ``random.Random``; ``None`` whenever no
viable cut exists (acyclic parents, no homologous bond pair, both
recombinations unsanitisable or over the length cap) -- the caller falls
back, never crashes.

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


def scaffold_hop_crossover(
    parent_a: str,
    parent_b: str,
    rng: random.Random,
    *,
    max_smiles_length: int = 120,
) -> str | None:
    """BREED-style crossover: one homologous cut per parent, hybrid child.

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
        :class:`~pharmaml.generative.GeneticOptimizer` default).  Both
        recombination pairings are tried before giving up.

    Returns
    -------
    str | None
        Canonical SMILES of the hybrid child, or ``None`` when no viable
        crossover exists (unparseable parent, acyclic parent with no
        ring-bearing fragment to swap, no homologous single-bond pair,
        both children unsanitisable or over the cap).  ``None`` is a
        "try the other operator" signal, never an error.

    Example
    -------
    >>> from pharmaml.generative import scaffold_hop_crossover
    >>> import random
    >>> # aspirin x benzamide: homologous cuts exist, hybrids come out
    >>> child = scaffold_hop_crossover("CC(=O)Oc1ccccc1C(=O)O",
    ...                                 "NC(=O)c1ccccc1", random.Random(0))
    >>> child is not None
    True
    >>> # benzene has no acyclic single bond: nothing to cut
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

        # Homologous cut pairs: single acyclic (bridge) bonds whose
        # endpoints match in some orientation.  Ring bonds are excluded
        # (cutting one does not split the molecule, so the fragment swap
        # has no meaning); non-single bonds are excluded (the cut would
        # have to invent bond orders).
        candidates: list[tuple[int, int, bool, bool]] = []
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
                    candidates.append((bond_a.GetIdx(), bond_b.GetIdx(), direct, flipped))
        if not candidates:
            return None

        bond_a_idx, bond_b_idx, direct, flipped = candidates[
            rng.randrange(len(candidates))
        ]
        cut_a = _cut_into_two(mol_a, bond_a_idx)
        cut_b = _cut_into_two(mol_b, bond_b_idx)
        if cut_a is None or cut_b is None:
            return None
        frag_a_begin, frag_a_end = cut_a
        frag_b_begin, frag_b_end = cut_b

        # Eligible pairings join signature-matched atoms: begin-begin and
        # end-end when the direct orientation matched, begin-end and
        # end-begin when only the flipped one did (all four when both).
        pairings: list[tuple[object, object]] = []
        if direct:
            pairings.append((frag_a_begin, frag_b_begin))
            pairings.append((frag_a_end, frag_b_end))
        if flipped:
            pairings.append((frag_a_begin, frag_b_end))
            pairings.append((frag_a_end, frag_b_begin))
        rng.shuffle(pairings)

        for frag_x, frag_y in pairings:
            try:
                child = Chem.molzip(frag_x, frag_y)
                Chem.SanitizeMol(child)
                smiles = Chem.MolToSmiles(child)
            except Exception:  # noqa: BLE001, S112 - try the sibling pairing
                continue
            if smiles and len(smiles) <= max_smiles_length:
                return smiles
        return None
    except Exception:  # noqa: BLE001 - robustness-first fallback contract
        return None
    finally:
        EnableLog("rdApp.error")
