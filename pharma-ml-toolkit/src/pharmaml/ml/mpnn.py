"""The D-MPNN message-passing network, pure NumPy (100x-175).

``chemprop/chemprop`` (v2.3.1, reviewed file by file in round 175) is
the reference: ``nn/message_passing/base.py`` + ``mixins.py`` (the
directed-edge update), ``featurizers/atom.py`` / ``bond.py`` (the
multi-hot blocks), ``nn/agg.py`` (the aggregation registry),
``nn/ffn.py`` / ``predictors.py`` (the FFN head) and ``cli/train.py``
(the recipe).  ``swansonk14/admet_ai`` v2 ships exactly this
architecture as its production ADMET model, and ``deepchem`` carries
its D-MPNN cousin in ``models/torch_models/dmpnn.py`` -- so one
faithful port covers three of the four round-175 repositories.

The equations, exactly as chemprop implements them (hidden states live
on *directed* edges ``e = v -> w``; ``rev(e)`` is the opposite edge):

* ``H_pre0_e = W_i [x_v ; e_vw]``  -- source atom features joined with
  the (direction-independent) bond features; **no** bias (chemprop's
  default), and *no* concatenation of both endpoints' features.
* ``H^(0) = relu(H_pre0)``
* for ``t = 1 .. depth-1``:
  ``m_e = sum_{e'=(u->v)} H^(t-1)_{e'}  -  H^(t-1)_{rev(e)}``
  (the incoming-edge sum gathered at the source atom, reverse edge
  excluded -- ``scatter_sum - H[rev]``),
  ``H^(t)_e = relu(H_pre0_e + W_h m_e)``  -- the residual uses the
  **pre-activation** ``H_pre0``, the quirk every chemprop version
  ships (their README calls it out; v2 kept it for parity).
* readout into atoms: ``M_v = sum_{e arriving at v} H^(T)_e`` then
  ``h_v = relu(W_o [x_v ; M_v] + b_o)`` (``W_o`` *has* a bias).
* aggregation to the molecule fingerprint: ``sum`` / ``mean`` /
  ``norm`` (= sum / 100, chemprop's CLI default) / ``norm`` scaling is
  the constant ``aggregation_norm``.
* FFN head: ``z = relu(W_1 fp + b_1)``, ``y = W_2 z + b_2`` -- one
  hidden layer, chemprop's ``ffn_num_layers=2`` equivalent.

Gradients are hand-derived for this exact composition (scatter-add
back-propagates as a gather; the reverse-edge subtraction as a negated
gather) and pinned by a finite-difference check in the doctests -- the
same evidence standard the toolkit's own tape autograd
(:mod:`pharmaml.ml.autograd`) is held to.  The optimiser is Adam, the
fit recipe standardises targets on the training set and unscales
predictions (chemprop's ``StandardScaler`` + ``UnscaleTransform``
flow).

Honest boundaries, stated once.  The atom/bond blocks are
*chemprop-shaped*, not chemprop-identical: the multi-hot layout keeps
chemprop's order (atomic number, degree, formal charge, num-H,
aromatic, mass) but the element list is the toolkit parser's 37
elements, hybridisation and chirality blocks are absent because
:mod:`pharmaml.chem.molgraph` deliberately does not perceive them, and
the bond block has no conjugation bit for the same reason.  Torch-only
extensions -- the attentive aggregation, MVE/evidential/quantile
heads, multicomponent encoders, MolAtomBond heads, the Myerson-value
atom attributions and checkpoint ensembling -- are out of scope; the
toolkit's answer to those lives in :mod:`pharmaml.ml.autograd` (the
engine), :mod:`pharmaml.ml.ensemble`-style averaging loops and
:mod:`pharmaml.ml.benchmetrics` (the scorers).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np

from pharmaml.chem.molgraph import ELEMENTS, Molecule, parse_smiles
from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "ATOM_FDIM",
    "BOND_FDIM",
    "DMPNN",
    "atom_features",
    "bond_features",
    "build_batch",
    "directed_graph",
    "gradient_check",
]

_LOG = get_logger(__name__)


# ------------------------------------------------------------------ #
# Featurization (chemprop-shaped blocks on the pure parser)           #
# ------------------------------------------------------------------ #

#: The parser's element list, ordered by atomic number (the one-hot block).
_ATOMIC_NUMS: tuple[int, ...] = tuple(
    sorted(info.atomic_num for info in ELEMENTS.values())
)
_DEGREES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
#: chemprop v1's formal-charge block order (0 last, then the unknown pad).
_CHARGES: tuple[int, ...] = (-1, -2, 1, 2, 0)
_NUM_HS: tuple[int, ...] = (0, 1, 2, 3, 4)

ATOM_FDIM = len(_ATOMIC_NUMS) + 1 + len(_DEGREES) + 1 + len(_CHARGES) + 1 + len(_NUM_HS) + 1 + 1 + 1
BOND_FDIM = 3 + 1 + 1 + 1  # order one-hot, aromatic, in-ring, stereo marker


def _one_hot(value: int, choices: tuple[int, ...]) -> np.ndarray:
    """chemprop's block: one-hot with the trailing unknown pad slot."""
    vec = np.zeros(len(choices) + 1, dtype=float)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        vec[len(choices)] = 1.0  # unknown -> pad bit
    return vec


def _num_implicit_h(atom, bond_valence: float) -> int:
    """Heavy-atom implicit H count (bracket H + valence remainder)."""
    from pharmaml.chem.molgraph import _valence_for

    bracket = atom.bracket_h or 0
    allowed = _valence_for(atom)
    if allowed is None:
        return bracket
    return max(int(allowed - bond_valence), 0) + bracket


def atom_features(mol: Molecule) -> np.ndarray:
    """The ``(n_atoms, ATOM_FDIM)`` multi-hot atom block.

    Layout: atomic-number one-hot (parser elements + pad), degree
    0-5 + pad, formal charge (chemprop v1 order) + pad, num-H 0-4 +
    pad, aromatic flag, mass/100.

    Examples
    --------
    >>> mol = parse_smiles("CCO")
    >>> feats = atom_features(mol)
    >>> feats.shape
    (3, 55)
    >>> float(feats[0, _ATOMIC_NUMS.index(6)])      # carbon one-hot
    1.0
    >>> float(feats[2, _ATOMIC_NUMS.index(8)])      # oxygen one-hot
    1.0
    """
    adjacency: list[list[int]] = mol.adjacency()
    rows = []
    for atom in mol.atoms:
        blocks = [
            _one_hot(atom.atomic_num, _ATOMIC_NUMS),
            _one_hot(len(adjacency[atom.index]), _DEGREES),
            _one_hot(atom.charge, _CHARGES),
            _one_hot(
                _num_implicit_h(atom, sum(1.0 if not b.is_aromatic else 1.5 for b in mol.atom_bonds(atom.index))),
                _NUM_HS,
            ),
            np.array([1.0 if atom.is_aromatic else 0.0]),
            np.array([0.01 * ELEMENTS[atom.symbol].weight]),
        ]
        rows.append(np.concatenate(blocks))
    if not rows:
        return np.zeros((0, ATOM_FDIM))
    return np.vstack(rows)


def _ring_bonds(mol: Molecule) -> set[int]:
    """Indices of bonds on some cycle (spanning-tree back edges)."""
    adjacency: list[list[int]] = mol.adjacency()
    parent: dict[int, int] = {}
    depth: dict[int, int] = {}
    ring_bonds: set[int] = set()
    bond_lookup = {}
    for i, bond in enumerate(mol.bonds):
        bond_lookup[(bond.begin, bond.end)] = i
        bond_lookup[(bond.end, bond.begin)] = i

    def mark_cycle(a: int, b: int) -> None:
        """Mark bonds along the tree path a..b plus edge (a, b)."""
        path = [a]
        while path[-1] != b and parent.get(path[-1]) is not None:
            path.append(parent[path[-1]])
        for u, v in itertools.pairwise(path):
            idx = bond_lookup.get((u, v))
            if idx is not None:
                ring_bonds.add(idx)
        idx = bond_lookup.get((a, b))
        if idx is not None:
            ring_bonds.add(idx)

    visited: set[int] = set()
    for root in range(len(mol.atoms)):
        if root in visited:
            continue
        stack = [root]
        visited.add(root)
        parent[root] = -1
        depth[root] = 0
        while stack:
            node = stack.pop()
            for nb in adjacency[node]:
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = node
                    depth[nb] = depth[node] + 1
                    stack.append(nb)
                elif nb != parent[node] and depth.get(node, 0) >= depth.get(nb, 0):
                    # back edge -> the cycle it closes
                    mark_cycle(node, nb)
    return ring_bonds


def bond_features(mol: Molecule) -> np.ndarray:
    """The ``(n_bonds, BOND_FDIM)`` bond block.

    Order one-hot (single/double/triple), aromatic flag, in-ring flag,
    stereo-marker flag (the parser's ``/``/``\\`` notes -- present, not
    interpreted).  No conjugation bit: the pure parser does not
    perceive conjugation.

    Examples
    --------
    >>> mol = parse_smiles("C=C")
    >>> bond_features(mol).round(1).tolist()
    [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    """
    rings = _ring_bonds(mol)
    rows = []
    for i, bond in enumerate(mol.bonds):
        if bond.is_aromatic:
            order_block = np.zeros(3)  # aromatic: no Kekulé order one-hot
        else:
            order_block = _one_hot(bond.order, (1, 2, 3))[:3]
        rows.append(
            np.concatenate(
                [
                    order_block,
                    np.array(
                        [
                            1.0 if bond.is_aromatic else 0.0,
                            1.0 if i in rings else 0.0,
                            1.0 if bond.stereo else 0.0,
                        ]
                    ),
                ]
            )
        )
    if not rows:
        return np.zeros((0, BOND_FDIM))
    return np.vstack(rows)


def directed_graph(mol: Molecule) -> dict[str, np.ndarray]:
    """The directed-edge view chemprop's message passing consumes.

    Every bond becomes two directed edges; ``src``/``dst`` are the
    edge endpoint arrays, ``rev`` the index of each edge's opposite,
    and ``E`` the (direction-independent) bond features stacked twice
    -- exactly chemprop's ``edge_index`` / ``rev_edge_index`` layout,
    where consecutive pairs ``2k, 2k+1`` are mutual reverses.

    Examples
    --------
    >>> mol = parse_smiles("CCO")
    >>> g = directed_graph(mol)
    >>> g["src"].tolist(), g["dst"].tolist()
    ([0, 1, 1, 2], [1, 0, 2, 1])
    >>> g["rev"].tolist()
    [1, 0, 3, 2]
    >>> g["E"].shape
    (4, 6)
    """
    bonds = bond_features(mol)
    src: list[int] = []
    dst: list[int] = []
    edge_rows: list[np.ndarray] = []
    for bond in mol.bonds:
        src.extend((bond.begin, bond.end))
        dst.extend((bond.end, bond.begin))
        edge_rows.extend((bonds[len(src) // 2 - 1], bonds[len(src) // 2 - 1]))
    rev = [i ^ 1 for i in range(len(src))]  # pairs are mutual reverses
    E = np.vstack(edge_rows) if edge_rows else np.zeros((0, BOND_FDIM))
    return {
        "V": atom_features(mol),
        "E": E,
        "src": np.asarray(src, dtype=int),
        "dst": np.asarray(dst, dtype=int),
        "rev": np.asarray(rev, dtype=int),
    }


class BatchGraph:
    """A disjoint-graph batch (chemprop's ``BatchMolGraph`` in NumPy).

    Molecules are concatenated with node/edge index offsets and a
    ``graph_id`` per atom, so one forward pass scores the whole batch.
    """

    def __init__(self, molecules: Sequence[Molecule]) -> None:
        if not molecules:
            raise DataValidationError("a batch needs at least one molecule")
        V_parts: list[np.ndarray] = []
        E_parts: list[np.ndarray] = []
        src_parts: list[np.ndarray] = []
        dst_parts: list[np.ndarray] = []
        rev_parts: list[np.ndarray] = []
        graph_ids: list[np.ndarray] = []
        atom_offset = 0
        edge_offset = 0
        for graph, mol in enumerate(molecules):
            g = directed_graph(mol)
            n_atoms = g["V"].shape[0]
            n_edges = g["src"].shape[0]
            V_parts.append(g["V"])
            E_parts.append(g["E"])
            src_parts.append(g["src"] + atom_offset)
            dst_parts.append(g["dst"] + atom_offset)
            rev_parts.append(g["rev"] + edge_offset)
            graph_ids.append(np.full(n_atoms, graph, dtype=int))
            atom_offset += n_atoms
            edge_offset += n_edges
        self.V = np.vstack(V_parts)
        self.E = np.vstack(E_parts) if E_parts else np.zeros((0, BOND_FDIM))
        self.src = np.concatenate(src_parts) if src_parts else np.zeros(0, dtype=int)
        self.dst = np.concatenate(dst_parts) if dst_parts else np.zeros(0, dtype=int)
        self.rev = np.concatenate(rev_parts) if rev_parts else np.zeros(0, dtype=int)
        self.graph_id = np.concatenate(graph_ids)
        self.n_graphs = len(molecules)
        self.n_atoms = self.V.shape[0]
        self.n_edges = self.src.shape[0]


def build_batch(molecules: Sequence[Molecule | str]) -> BatchGraph:
    """Build a :class:`BatchGraph` from Molecules or SMILES strings."""
    parsed = [
        m if isinstance(m, Molecule) else parse_smiles(str(m))
        for m in molecules
    ]
    if not parsed:
        raise DataValidationError("build_batch needs at least one molecule")
    return BatchGraph(parsed)


# ------------------------------------------------------------------ #
# The network                                                          #
# ------------------------------------------------------------------ #


def _relu_grad(pre: np.ndarray) -> np.ndarray:
    return (pre > 0.0).astype(float)


class DMPNN:
    """Directed message-passing neural network (chemprop's MPNN).

    Parameters
    ----------
    hidden_size:
        Message/FFN hidden width (chemprop default 300; the demos use
        small widths for speed).
    depth:
        Message-passing depth -- ``depth`` means ``depth - 1`` update
        iterations after the initial projection, chemprop's convention
        (default 3).
    aggregation:
        ``'norm'`` (sum / ``aggregation_norm``, chemprop's default),
        ``'sum'`` or ``'mean'``.
    dropout:
        Training-time dropout on the messages and the atom readout
        (chemprop applies the same rate there; default 0.0).
    n_outputs:
        Output width (one property; chemprop's ``n_tasks`` head).

    Examples
    --------
    >>> model = DMPNN(hidden_size=8, depth=2, seed=0)
    >>> y = model.predict(["CCO", "c1ccccc1"])
    >>> y.shape
    (2, 1)
    >>> bool(np.isfinite(y).all())
    True
    """

    def __init__(
        self,
        hidden_size: int = 300,
        depth: int = 3,
        ffn_hidden: int | None = None,
        aggregation: str = "norm",
        aggregation_norm: float = 100.0,
        dropout: float = 0.0,
        n_outputs: int = 1,
        seed: int | None = 0,
    ) -> None:
        if hidden_size < 1 or depth < 1:
            raise DataValidationError("hidden_size and depth must be >= 1")
        if aggregation not in ("norm", "sum", "mean"):
            raise DataValidationError(
                "aggregation must be 'norm', 'sum' or 'mean'"
            )
        if not 0.0 <= dropout < 1.0:
            raise DataValidationError("dropout must be in [0, 1)")
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.ffn_hidden = int(ffn_hidden if ffn_hidden is not None else hidden_size)
        self.aggregation = aggregation
        self.aggregation_norm = float(aggregation_norm)
        self.dropout = float(dropout)
        self.n_outputs = int(n_outputs)
        self._rng = np.random.default_rng(seed)
        seed_int = None if seed is None else int(seed) + 12345
        init_rng = np.random.default_rng(seed_int)
        h = self.hidden_size
        # chemprop's shapes: W_i (A+B -> h, no bias), W_h (h -> h, no
        # bias), W_o (A+h -> h, bias), then the 2-layer FFN head.
        self.weights: dict[str, np.ndarray] = {
            "Wi": self._xavier(init_rng, ATOM_FDIM + BOND_FDIM, h),
            "Wh": self._xavier(init_rng, h, h),
            "Wo": self._xavier(init_rng, ATOM_FDIM + h, h),
            "bo": np.zeros(h),
            "W1": self._xavier(init_rng, h, self.ffn_hidden),
            "b1": np.zeros(self.ffn_hidden),
            "W2": self._xavier(init_rng, self.ffn_hidden, self.n_outputs),
            "b2": np.zeros(self.n_outputs),
        }
        self._y_mean = 0.0
        self._y_std = 1.0
        self._adam: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._adam_t = 0

    @staticmethod
    def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-bound, bound, size=(fan_out, fan_in))

    # -- forward ------------------------------------------------------ #

    def _forward(self, batch: BatchGraph, training: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        """Predictions for the batch plus the backprop cache."""
        W = self.weights
        n_atoms, n_edges = batch.n_atoms, batch.n_edges
        h = self.hidden_size
        V, E = batch.V, batch.E
        src, dst, rev = batch.src, batch.dst, batch.rev
        # Directed-edge inputs: [source atom features ; bond features].
        X_e = np.hstack((V[src], E)) if n_edges else np.zeros((0, ATOM_FDIM + BOND_FDIM))
        H_pre0 = X_e @ W["Wi"].T if n_edges else np.zeros((0, h))
        H = np.maximum(H_pre0, 0.0)
        if training and self.dropout > 0.0 and n_edges:
            mask0 = (self._rng.random(H.shape) >= self.dropout) / (1.0 - self.dropout)
            H = H * mask0
        else:
            mask0 = None
        caches: list[dict[str, Any]] = []
        for _ in range(1, self.depth):
            # m_e = scatter(H -> dst)[src] - H[rev]
            M_in = np.zeros((n_atoms, h))
            if n_edges:
                np.add.at(M_in, dst, H)
                m = M_in[src] - H[rev]
            else:
                m = np.zeros((0, h))
            a = H_pre0 + m @ W["Wh"].T if n_edges else np.zeros((0, h))
            H_new = np.maximum(a, 0.0)
            if training and self.dropout > 0.0 and n_edges:
                mask_t = (self._rng.random(H_new.shape) >= self.dropout) / (1.0 - self.dropout)
                H_new = H_new * mask_t
            else:
                mask_t = None
            caches.append({"m": m, "a": a, "mask": mask_t})
            H = H_new
        # Readout: incoming-edge sum per atom, then W_o [x_v ; M_v].
        M_v = np.zeros((n_atoms, h))
        if n_edges:
            np.add.at(M_v, dst, H)
        X_v = np.hstack((V, M_v))
        pre_o = X_v @ W["Wo"].T + W["bo"]
        h_v = np.maximum(pre_o, 0.0)
        if training and self.dropout > 0.0:
            mask_o = (self._rng.random(h_v.shape) >= self.dropout) / (1.0 - self.dropout)
            h_v = h_v * mask_o
        else:
            mask_o = None
        # Aggregation to per-molecule fingerprints.
        graph_sizes = np.bincount(batch.graph_id, minlength=batch.n_graphs).astype(float)
        fp = np.zeros((batch.n_graphs, h))
        np.add.at(fp, batch.graph_id, h_v)
        if self.aggregation == "mean":
            safe = np.where(graph_sizes > 0, graph_sizes, 1.0)
            fp = fp / safe[:, None]
        elif self.aggregation == "norm":
            fp = fp / self.aggregation_norm
        pre_1 = fp @ W["W1"].T + W["b1"]
        z = np.maximum(pre_1, 0.0)
        y = z @ W["W2"].T + W["b2"]
        cache = {
            "batch": batch,
            "X_e": X_e,
            "H_pre0": H_pre0,
            "H_final": H,
            "mask0": mask0,
            "layers": caches,
            "M_v": M_v,
            "X_v": X_v,
            "pre_o": pre_o,
            "mask_o": mask_o,
            "h_v": h_v,
            "graph_sizes": graph_sizes,
            "fp": fp,
            "pre_1": pre_1,
            "z": z,
        }
        return y, cache

    def _loss_and_grads(
        self, batch: BatchGraph, targets: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """MSE loss (0.5-scaled) and hand-derived gradients."""
        W = self.weights
        y, cache = self._forward(batch, training=True)
        targets = np.asarray(targets, dtype=float).reshape(y.shape)
        diff = y - targets
        loss = 0.5 * float(np.mean(diff**2))
        n_out = y.shape[0]
        dy = diff / n_out
        grads: dict[str, np.ndarray] = {name: np.zeros_like(w) for name, w in W.items()}
        # FFN head.
        grads["W2"] = dy.T @ cache["z"]
        grads["b2"] = dy.sum(axis=0)
        dz = dy @ W["W2"]
        dpre1 = dz * _relu_grad(cache["pre_1"])
        grads["W1"] = dpre1.T @ cache["fp"]
        grads["b1"] = dpre1.sum(axis=0)
        dfp = dpre1 @ W["W1"]
        # Aggregation backward.
        dh_v = np.zeros_like(cache["h_v"])
        for graph in range(batch.n_graphs):
            rows = batch.graph_id == graph
            if self.aggregation == "mean":
                dh_v[rows] = dfp[graph] / max(cache["graph_sizes"][graph], 1.0)
            elif self.aggregation == "norm":
                dh_v[rows] = dfp[graph] / self.aggregation_norm
            else:
                dh_v[rows] = dfp[graph]
        dpost_o = dh_v * _relu_grad(cache["pre_o"])
        if cache["mask_o"] is not None:
            dpost_o = dpost_o * cache["mask_o"]
        grads["Wo"] = dpost_o.T @ cache["X_v"]
        grads["bo"] = dpost_o.sum(axis=0)
        dX_v = dpost_o @ W["Wo"]
        dM_v = dX_v[:, ATOM_FDIM:]
        # Readout scatter backward is a pure gather: each directed edge
        # contributes to exactly one atom.
        src, dst, rev = batch.src, batch.dst, batch.rev
        n_edges = batch.n_edges
        h = self.hidden_size
        dH = dM_v[dst] if n_edges else np.zeros((0, h))
        # Unwind the message-passing layers.
        H_pre0 = cache["H_pre0"]
        dH_pre0 = np.zeros_like(H_pre0)
        for layer in reversed(cache["layers"]):
            dpost = dH * _relu_grad(layer["a"])
            if layer["mask"] is not None:
                dpost = dpost * layer["mask"]
            dH_pre0 += dpost  # the pre-activation residual
            dm = dpost @ W["Wh"]
            grads["Wh"] += dpost.T @ layer["m"]
            # m_e = M_in[src] - H_prev[rev]: backward is two gathers
            # (rev is a permutation, so the subtraction scatters cleanly).
            dM_in = np.zeros((batch.n_atoms, h))
            if n_edges:
                np.add.at(dM_in, src, dm)
                dH = dM_in[dst] - dm[rev]
        # H^(0) = relu(H_pre0) (plus mask0).
        d0 = dH * _relu_grad(H_pre0)
        if cache["mask0"] is not None:
            d0 = d0 * cache["mask0"]
        dH_pre0 += d0
        if n_edges:
            grads["Wi"] = dH_pre0.T @ cache["X_e"]
        return loss, grads

    # -- optimisation -------------------------------------------------- #

    def _adam_step(self, grads: dict[str, np.ndarray], lr: float) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._adam_t += 1
        t = self._adam_t
        for name, grad in grads.items():
            if name not in self._adam:
                self._adam[name] = (np.zeros_like(grad), np.zeros_like(grad))
            m, v = self._adam[name]
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad * grad
            self._adam[name] = (m, v)
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            self.weights[name] = self.weights[name] - lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        molecules: Sequence[Molecule | str],
        targets: Sequence[float] | np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int | None = None,
        standardize: bool = True,
        seed: int | None = 0,
    ) -> list[float]:
        """Train with Adam on standardised targets (chemprop's recipe).

        Targets are z-scored with the *training* mean/std (chemprop
        fits its ``StandardScaler`` on train only); predictions are
        unscaled automatically.  Mini-batches iterate the molecules in
        seeded shuffled order.

        Examples
        --------
        >>> smiles = ["C", "CC", "CCC", "CCCC", "CCO", "CCCO",
        ...           "c1ccccc1", "Cc1ccccc1", "CC(=O)O", "CCN"]
        >>> targets = [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 6.0, 7.0, 3.0, 2.0]
        >>> model = DMPNN(hidden_size=12, depth=2, aggregation="sum", seed=3)
        >>> history = model.fit(smiles, targets, epochs=80, lr=0.05)
        >>> history[-1] < 0.5 * history[0]
        True
        >>> import numpy as np
        >>> pred = model.predict(smiles[:4]).ravel()
        >>> float(np.corrcoef(pred, targets[:4])[0, 1]) > 0.9
        True
        """
        parsed = [
            m if isinstance(m, Molecule) else parse_smiles(str(m))
            for m in molecules
        ]
        if not parsed:
            raise DataValidationError("fit needs at least one molecule")
        y = np.asarray(targets, dtype=float).ravel()
        if y.shape[0] != len(parsed):
            raise DataValidationError("one target per molecule required")
        if standardize:
            self._y_mean = float(y.mean())
            std = float(y.std())
            self._y_std = std if std > 0 else 1.0
        else:
            self._y_mean, self._y_std = 0.0, 1.0
        y_scaled = (y - self._y_mean) / self._y_std
        rng = np.random.default_rng(seed)
        n = len(parsed)
        if batch_size is None or batch_size >= n:
            batches = [list(range(n))]
        else:
            batches = [
                list(order[start : start + batch_size])
                for start in range(0, n, batch_size)
                for order in [rng.permutation(n)]
            ]
        history: list[float] = []
        for _ in range(max(1, int(epochs))):
            epoch_loss = 0.0
            for idx in batches:
                batch = BatchGraph([parsed[i] for i in idx])
                loss, grads = self._loss_and_grads(batch, y_scaled[list(idx)])
                self._adam_step(grads, lr)
                epoch_loss += loss * len(idx)
            history.append(epoch_loss / n)
        return history

    def predict(
        self, molecules: Sequence[Molecule | str]
    ) -> np.ndarray:
        """Predictions in original target units ``(n, n_outputs)``."""
        batch = build_batch(molecules)
        y, _ = self._forward(batch, training=False)
        return y * self._y_std + self._y_mean

    def fingerprint(self, molecules: Sequence[Molecule | str]) -> np.ndarray:
        """The learned molecular fingerprints (chemprop's CLI command)."""
        batch = build_batch(molecules)
        _, cache = self._forward(batch, training=False)
        return cache["fp"]

    def n_parameters(self) -> int:
        """Total trainable parameter count."""
        return int(sum(w.size for w in self.weights.values()))

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"DMPNN(hidden_size={self.hidden_size}, depth={self.depth}, "
            f"aggregation={self.aggregation!r}, "
            f"n_parameters={self.n_parameters()})"
        )


# ------------------------------------------------------------------ #
# Evidence: finite-difference gradient check                          #
# ------------------------------------------------------------------ #


def gradient_check(
    molecules: Sequence[Molecule | str],
    targets: Sequence[float] | np.ndarray,
    model: DMPNN | None = None,
    epsilon: float = 1e-5,
) -> float:
    """Max relative error between analytic and numeric gradients.

    Perturbs every parameter of a small model by ``±epsilon`` and
    compares the central difference of the loss with
    :meth:`DMPNN._loss_and_grads` -- the same evidence standard as
    :func:`pharmaml.ml.autograd.grad_check`.  Returns the worst
    relative deviation.

    Examples
    --------
    >>> err = gradient_check(["CCO", "C=N"], [1.0, -1.0],
    ...                      DMPNN(hidden_size=4, depth=3, seed=5))
    >>> err < 1e-4
    True
    """
    if model is None:
        model = DMPNN(hidden_size=4, depth=3, seed=0)
    batch = build_batch(molecules)
    y_target = np.asarray(targets, dtype=float).reshape(batch.n_graphs, model.n_outputs)

    def loss_only() -> float:
        y, _ = model._forward(batch, training=False)
        diff = y - y_target
        return 0.5 * float(np.mean(diff**2))

    _, grads = model._loss_and_grads(batch, y_target)
    worst = 0.0
    for name, grad in grads.items():
        flat = grad.ravel()
        weights_flat = model.weights[name].ravel()
        for i in range(flat.size):
            original = weights_flat[i]
            weights_flat[i] = original + epsilon
            plus = loss_only()
            weights_flat[i] = original - epsilon
            minus = loss_only()
            weights_flat[i] = original
            numeric = (plus - minus) / (2.0 * epsilon)
            analytic = float(flat[i])
            scale = max(1e-8, abs(numeric), abs(analytic))
            worst = max(worst, abs(numeric - analytic) / scale)
    return worst



