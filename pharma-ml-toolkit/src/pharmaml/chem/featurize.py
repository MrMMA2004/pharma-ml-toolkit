"""Molecule → graph featurization — the ``rdkit/rdkit`` bridge role.

``rdkit`` (reviewed file by file — ``scratch/gh187_review.md``) sits inside
deep-learning frameworks in exactly one role: turning molecules into the
atom/bond feature tensors a GNN consumes (``rdkit/Chem`` for the graph,
``rdkit/ML`` for the descriptor plumbing).  This toolkit's ``chem/molgraph``
already IS the offline molecular graph (SMILES parse, rings, kekulization),
so the featurizer is the one genuinely missing piece: pure tables over what
:meth:`~pharmaml.chem.molgraph.Molecule` already knows — no RDKit import,
no conda, no compiled boost.

The feature vocabulary follows the canonical MoleculeNet/OGB featurisations,
restricted to what ``molgraph`` can state honestly:

* atoms — the element one-hot over ``ELEMENTS`` (the drug-like 29), the
  heavy-atom degree bucket, the total-H bucket (implicit + bracket),
  the aromatic flag, the formal-charge one-hot (−1/0/+1) and the
  in-ring flag;
* bonds — the order one-hot (1/2/3), the aromatic flag and the
  both-endpoints-in-a-ring flag.

The honest scope, stated once: no 3-D conformers, no stereochemistry
encoding (``molgraph`` retains but does not interpret ``/``/``\\``), no
partial charges — the tables above are the whole story.

Examples
--------
>>> from pharmaml.chem.featurize import smiles_to_graph, ATOM_WIDTH
>>> g = smiles_to_graph("CCO")
>>> (g.num_nodes, g.num_edges)
(3, 4)
>>> g.x.shape == (3, ATOM_WIDTH)
True
>>> float(g.edge_attr.sum() > 0)
1.0
"""

from __future__ import annotations

import numpy as np

from pharmaml.ml.gnn import Data

from .molgraph import (
    ELEMENTS,
    MolAtom,
    MolBond,
    Molecule,
    find_rings,
    implicit_hydrogens,
    parse_smiles,
)

__all__ = [
    "ATOM_FEATURE_NAMES",
    "ATOM_WIDTH",
    "BOND_FEATURE_NAMES",
    "BOND_WIDTH",
    "ELEMENT_ORDER",
    "atom_features",
    "bond_features",
    "molecule_to_data",
    "smiles_to_graph",
]

#: The element one-hot order — the drug-like palette ``molgraph`` ships.
ELEMENT_ORDER: tuple[str, ...] = tuple(ELEMENTS)

DEGREE_BUCKETS: int = 6
H_BUCKETS: int = 5
#: charge one-hot order: (-1, 0, +1)
CHARGE_LEVELS: tuple[int, ...] = (-1, 0, 1)

ATOM_FEATURE_NAMES: tuple[str, ...] = (
    *(f"element_{symbol}" for symbol in ELEMENT_ORDER),
    *(f"degree_{i}" for i in range(DEGREE_BUCKETS)),
    *(f"total_h_{i}" for i in range(H_BUCKETS)),
    "aromatic",
    *(f"charge_{level:+d}" for level in CHARGE_LEVELS),
    "in_ring",
)
BOND_WIDTH: int = 5
BOND_FEATURE_NAMES: tuple[str, ...] = (
    "order_single",
    "order_double",
    "order_triple",
    "aromatic",
    "in_ring",
)

#: Width of one atom's feature row (the GNN's input width for molecules).
ATOM_WIDTH: int = len(ATOM_FEATURE_NAMES)


def _one_hot(position: int, width: int) -> np.ndarray:
    row = np.zeros(width)
    if 0 <= position < width:
        row[position] = 1.0
    return row


def _ring_atoms(mol: Molecule) -> set[int]:
    rings = find_rings(mol)
    atoms: set[int] = set()
    for ring in rings:
        atoms.update(ring)
    return atoms


def atom_features(
    mol: Molecule, atom: MolAtom, ring_atoms: set[int] | None = None
) -> np.ndarray:
    """One atom's feature row (the docstring's vocabulary, assembled)."""

    if ring_atoms is None:
        ring_atoms = _ring_atoms(mol)
    hydrogens = implicit_hydrogens(mol)[atom.index]
    element_position = (
        ELEMENT_ORDER.index(atom.symbol) if atom.symbol in ELEMENT_ORDER else -1
    )
    degree = len(mol.adjacency()[atom.index])
    charge = int(np.clip(atom.charge, -1, 1))
    return np.concatenate(
        [
            _one_hot(element_position, len(ELEMENT_ORDER)),
            _one_hot(min(degree, DEGREE_BUCKETS - 1), DEGREE_BUCKETS),
            _one_hot(min(hydrogens, H_BUCKETS - 1), H_BUCKETS),
            np.array([1.0 if atom.is_aromatic else 0.0]),
            _one_hot(CHARGE_LEVELS.index(charge), len(CHARGE_LEVELS)),
            np.array([1.0 if atom.index in ring_atoms else 0.0]),
        ]
    )


def bond_features(mol: Molecule, bond: MolBond, ring_atoms: set[int] | None = None) -> np.ndarray:
    """One bond's feature row (order one-hot, aromatic, in-ring)."""

    if ring_atoms is None:
        ring_atoms = _ring_atoms(mol)
    order_position = {1: 0, 2: 1, 3: 2}.get(int(bond.order), 0)
    in_ring = float(bond.begin in ring_atoms and bond.end in ring_atoms)
    return np.concatenate(
        [
            _one_hot(order_position, 3),
            np.array([1.0 if bond.is_aromatic else 0.0, in_ring]),
        ]
    )


def molecule_to_data(mol: Molecule, *, y: float | None = None) -> Data:
    """Featurize a :class:`~pharmaml.chem.molgraph.Molecule` into a
    :class:`~pharmaml.ml.gnn.Data` graph (undirected, bond features on
    both directions)."""

    ring_atoms = _ring_atoms(mol)
    x = np.stack(
        [atom_features(mol, atom, ring_atoms) for atom in mol.atoms]
    )
    rows: list[np.ndarray] = []
    src: list[int] = []
    dst: list[int] = []
    for bond in mol.bonds:
        features = bond_features(mol, bond, ring_atoms)
        for a, b in ((bond.begin, bond.end), (bond.end, bond.begin)):
            src.append(a)
            dst.append(b)
            rows.append(features)
    if rows:
        edge_attr = np.stack(rows)
        edge_index = np.stack([np.array(src), np.array(dst)])
    else:  # a single atom: the GNN still needs a (2, 0) edge axis
        edge_attr = np.zeros((0, BOND_WIDTH))
        edge_index = np.zeros((2, 0), dtype=int)
    label = None if y is None else np.array(float(y))
    return Data(x, edge_index, edge_attr=edge_attr, y=label)


def smiles_to_graph(smiles: str, *, y: float | None = None) -> Data:
    """SMILES → :class:`~pharmaml.ml.gnn.Data`, the one-call bridge the
    Deep Learning tab's molecule path uses."""

    return molecule_to_data(parse_smiles(smiles), y=y)
