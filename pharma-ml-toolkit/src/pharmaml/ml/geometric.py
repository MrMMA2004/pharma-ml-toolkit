"""Geometric graph learning -- pytorch_geometric + DIG-xgraph, absorbed.

Round 187 checked pytorch_geometric 2.9.0 and DIG's GraphXAI
(``GNNExplainer``) and re-implemented them offline from the upstream
specs: the ``edge_index`` [2, E] data model (row 0 the SOURCE ``j``,
row 1 the TARGET ``i``, messages aggregated by target), ``gcn_norm``
with PyG's exact normalisation arithmetic, the GCN / SAGE / GAT
convolutions as numpy layers with hand-derived backward passes, PyG's
``Batch`` collation and ``global_*_pool`` reductions, and DIG's
per-class sigmoid-gated edge-mask explainer with the size/entropy
regularisers and the 4-forward fidelity protocol.  DIG is GPL-3.0
licensed, so nothing is copied from it -- every algorithm here is a
clean-room re-implementation from the published equations and
docstrings.  GINConv is documented N/A: GCN/SAGE/GAT already cover
PyG's aggregation families (add / mean / attention).

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.geometric import (
...     add_self_loops, collate_graphs, gcn_norm, global_mean_pool,
...     motif_dataset, remove_self_loops, to_edge_index, GraphRecord,
... )
>>> adjacency = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
>>> edge_index, weight = gcn_norm(to_edge_index(adjacency), 3)
>>> edge_index.tolist()
[[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]
>>> [round(w, 4) for w in weight]
[0.4082, 0.4082, 0.4082, 0.4082, 0.5, 0.3333, 0.5]
>>> add_self_loops(np.array([[0, 1], [1, 0]]), 3).tolist()
[[0, 1, 0, 1, 2], [1, 0, 0, 1, 2]]
>>> remove_self_loops(add_self_loops(np.array([[0, 1], [1, 0]]), 3)).tolist()
[[0, 1], [1, 0]]
>>> features = np.array([[1.], [3.], [5.], [7.]])
>>> global_mean_pool(features, np.array([0, 0, 1, 1])).tolist()
[[2.0], [6.0]]
>>> first = GraphRecord(np.zeros((2, 1)), np.array([[0], [1]]), 0)
>>> second = GraphRecord(np.ones((3, 1)), np.array([[0, 1], [1, 2]]), 1)
>>> x, edges, batch, y = collate_graphs([first, second])
>>> edges.tolist(), batch.tolist(), y.tolist()
([[0, 2, 3], [1, 3, 4]], [0, 0, 1, 1, 1], [0, 1])
>>> graphs = motif_dataset(n_graphs=6, seed=0)
>>> [g.y for g in graphs]
[1, 0, 1, 0, 1, 0]
>>> graphs[0].x.shape[1], len(graphs[0].motif_edges)
(2, 6)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pharmaml.exceptions import PharmaMLError
from pharmaml.logger import get_logger

_LOGGER = get_logger(__name__)

#: The upstream release this module's semantics were checked against.
ABSORBED_VERSION = "2.9.0"

#: The house motif's undirected edges (local node ids 0-4): square
#: 0-1-2-3-0 plus the roof node 4 wired to the two top corners 1 and 2
#: -- the "house" of the GNNExplainer paper's synthetic benchmark.
HOUSE_MOTIF_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 3),
    (1, 4),
    (2, 4),
)

#: Leaky-ReLU slope PyG's ``GATConv`` defaults to (``negative_slope``).
DEFAULT_GAT_SLOPE = 0.2

#: Softmax denominator epsilon (PyG's ``torch_geometric.utils.softmax``
#: adds ``1e-16`` to every group sum).
ATTENTION_EPS = 1e-16


class GeometricError(PharmaMLError):
    """Raised for malformed edge lists, invalid layer or trainer
    configurations, and explainer misuse."""


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _as_edge_index(edge_index: Any, num_nodes: int | None = None) -> np.ndarray:
    """Coerce to an int64 (2, E) array with row 0 = sources, row 1 = targets."""
    array = np.asarray(edge_index)
    if array.ndim != 2 or array.shape[0] != 2:
        raise GeometricError(
            f"edge_index must be a (2, E) array (row 0 sources, row 1 targets), "
            f"got shape {array.shape}"
        )
    edges = array.astype(np.int64)
    if edges.size and not np.array_equal(edges, array):
        raise GeometricError("edge_index entries must be integers")
    if num_nodes is not None and edges.size:
        lo, hi = int(edges.min()), int(edges.max())
        if lo < 0 or hi >= num_nodes:
            raise GeometricError(
                f"edge_index node ids must lie in [0, {num_nodes}), got "
                f"[{lo}, {hi}]"
            )
    return edges


def _as_node_features(x: Any, in_channels: int | None = None) -> np.ndarray:
    """Coerce node features to a float64 (N, F) array with N >= 1."""
    features = np.asarray(x, dtype=float)
    if features.ndim != 2:
        raise GeometricError(f"node features must be 2-D (N, F), got {features.shape}")
    if features.shape[0] == 0:
        raise GeometricError("node features need at least one node")
    if in_channels is not None and features.shape[1] != in_channels:
        raise GeometricError(
            f"node features need {in_channels} channels, got {features.shape[1]}"
        )
    return features


def _as_edge_mask(edge_mask: Any, expected: int) -> np.ndarray:
    """Coerce an edge mask to a float vector of ``expected`` entries."""
    if edge_mask is None:
        return np.ones(expected)
    mask = np.asarray(edge_mask, dtype=float).reshape(-1)
    if mask.size != expected:
        raise GeometricError(
            f"edge_mask needs {expected} entries (one per edge), got {mask.size}"
        )
    return mask


def _index_add(index: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """Scatter-add ``values`` rows into ``size`` buckets by ``index``.

    The numpy stand-in for ``torch_geometric.utils.scatter`` with
    ``reduce="sum"``: ``values`` is (E, ...) and the result is
    ``(size, ...)`` with every row summed into its bucket.  Implemented
    with per-channel ``bincount`` (uninitialised buckets read as zero),
    which beats ``np.add.at`` by an order of magnitude on the batched
    graphs the trainer feeds through here.
    """
    idx = np.asarray(index, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.zeros((size,) + np.shape(values)[1:], dtype=float)
    flat = np.asarray(values, dtype=float).reshape(idx.size, -1)
    out = np.zeros((size, flat.shape[1]), dtype=float)
    for channel in range(flat.shape[1]):
        out[:, channel] = np.bincount(idx, weights=flat[:, channel], minlength=size)
    return out.reshape((size,) + np.shape(values)[1:])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# Edge utilities -- the torch_geometric.utils data model
# --------------------------------------------------------------------------- #
def to_edge_index(adjacency: Any) -> np.ndarray:
    """Extract a (2, E) ``edge_index`` from a dense adjacency matrix.

    A row-major scan of the nonzero entries: ``adjacency[j, i] != 0``
    yields the directed edge ``j -> i`` (row = source).  A symmetric
    input therefore yields every undirected edge in both directions,
    which is what the convolutions here expect.  Diagonal entries pass
    through as self-loops; weights are ignored (structure only).

    >>> to_edge_index(np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])).tolist()
    [[0, 1, 1, 2], [1, 0, 2, 1]]
    """
    dense = np.asarray(adjacency)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise GeometricError(
            f"adjacency must be a square 2-D matrix, got shape {dense.shape}"
        )
    rows, cols = np.nonzero(dense)
    return np.stack([rows.astype(np.int64), cols.astype(np.int64)])


def from_edge_index(edge_index: Any, num_nodes: int) -> np.ndarray:
    """Build the dense adjacency matrix ``A[source, target]``.

    The exact inverse of :func:`to_edge_index`'s row-major convention
    (``to_dense_adj`` semantics).  Duplicate edges accumulate into the
    matrix entry (the honest statement of coalescing by summation).

    >>> from_edge_index(np.array([[0, 1], [1, 0]]), 2).astype(int).tolist()
    [[0, 1], [1, 0]]
    """
    edges = _as_edge_index(edge_index, num_nodes)
    dense = np.zeros((int(num_nodes), int(num_nodes)), dtype=float)
    np.add.at(dense, (edges[0], edges[1]), 1.0)
    return dense


def degree(edge_index: Any, num_nodes: int) -> np.ndarray:
    """In-degree counts: how many edges TARGET each node (PyG's
    ``degree(edge_index[1])``, the message-passing direction).

    >>> degree(np.array([[0, 1, 1, 2], [1, 0, 2, 0]]), 3).tolist()
    [2, 1, 1]
    """
    edges = _as_edge_index(edge_index, num_nodes)
    return np.bincount(edges[1], minlength=int(num_nodes)).astype(np.int64)


def add_self_loops(edge_index: Any, num_nodes: int) -> np.ndarray:
    """Append one self-loop per node, no dedup -- PyG's
    ``add_self_loops`` semantics (existing loops are kept as-is).

    >>> add_self_loops(np.array([[0, 1], [1, 0]]), 3).tolist()
    [[0, 1, 0, 1, 2], [1, 0, 0, 1, 2]]
    """
    edges = _as_edge_index(edge_index, num_nodes)
    loops = np.arange(int(num_nodes), dtype=np.int64)
    return np.concatenate([edges, np.stack([loops, loops])], axis=1)


def remove_self_loops(edge_index: Any) -> np.ndarray:
    """Drop every edge with ``source == target`` (order preserved).

    >>> remove_self_loops(np.array([[0, 1, 2], [1, 0, 2]])).tolist()
    [[0, 1], [1, 0]]
    """
    edges = _as_edge_index(edge_index)
    keep = edges[0] != edges[1]
    return edges[:, keep]


def to_undirected(edge_index: Any) -> np.ndarray:
    """Symmetrise: every undirected pair in BOTH directions, coalesced.

    The two directions are concatenated, then exact duplicates are
    coalesced with ``np.unique`` over the sorted (source, target)
    columns -- the output is the lexicographically sorted canonical
    undirected edge list, one column per direction per pair.

    >>> to_undirected(np.array([[0, 2], [1, 0]])).tolist()
    [[0, 0, 1, 2], [1, 2, 0, 0]]
    """
    edges = _as_edge_index(edge_index)
    both = np.concatenate([edges, edges[::-1]], axis=1)
    return np.unique(both, axis=1)


def gcn_norm(
    edge_index: Any,
    num_nodes: int,
    edge_weight: Any = None,
    improved: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """The GCN normalisation, PyG's ``gcn_norm`` semantics exactly.

    Existing self-loops are stripped and re-added for every node with
    weight ``fill_value`` (2.0 when ``improved``, else 1.0); the degree
    ``deg[i]`` counts edge WEIGHTS targeting ``i`` (self-loop included);
    each weight is rescaled to ``w'_{j,i} = w / sqrt(deg[j] * deg[i])``.
    Degrees of zero map to weight 0 (PyG's inf-to-zero guard -- only
    reachable for isolated nodes WITHOUT a self-loop, impossible here
    because loops are always added, but the guard stays).

    Returns the loop-augmented edge list (loops appended LAST, in node
    order) and the matching normalised weights -- the canonical edge
    ordering every masked forward path in this module relies on.

    >>> edges, weight = gcn_norm(np.array([[0, 1, 1, 2], [1, 0, 2, 1]]), 3)
    >>> edges.shape[1], round(float(weight[0]), 4)
    (7, 0.4082)
    """
    edges = _as_edge_index(edge_index, num_nodes)
    num_nodes = int(num_nodes)
    if num_nodes < 1:
        raise GeometricError(f"num_nodes must be >= 1, got {num_nodes}")
    fill = 2.0 if improved else 1.0
    stripped = remove_self_loops(edges)
    if edge_weight is None:
        weights = np.ones(stripped.shape[1], dtype=float)
    else:
        given = np.asarray(edge_weight, dtype=float).reshape(-1)
        if given.size != edges.shape[1]:
            raise GeometricError(
                f"edge_weight needs {edges.shape[1]} entries, got {given.size}"
            )
        weights = given[edges[0] != edges[1]]
    loops = np.arange(num_nodes, dtype=np.int64)
    out_edges = np.concatenate(
        [stripped, np.stack([loops, loops])], axis=1
    )
    out_weight = np.concatenate([weights, np.full(num_nodes, fill, dtype=float)])
    deg = np.bincount(
        out_edges[1], weights=out_weight, minlength=num_nodes
    ).astype(float)
    inv_sqrt = 1.0 / np.sqrt(deg)
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    out_weight = inv_sqrt[out_edges[0]] * out_weight * inv_sqrt[out_edges[1]]
    return out_edges, out_weight


# --------------------------------------------------------------------------- #
# Layers -- numpy forward AND backward, torch-free
# --------------------------------------------------------------------------- #
class GraphLayer:
    """Base contract for the numpy message-passing layers.

    A layer owns its parameter arrays (``weights``-style attributes,
    created float64 by the constructor), a ``grads`` dict of same-shaped
    gradients ACCUMULATED by :meth:`backward` (call :meth:`zero_grad`
    between optimiser steps), and ``params()`` listing ``(name, array)``
    pairs so an external optimiser can mutate the arrays in place.
    ``backward`` must be called on the cache of the most recent
    :meth:`forward` -- the standard manual-autograd contract.
    """

    grads: dict[str, np.ndarray]

    def __init__(self) -> None:
        self.grads = {}

    def forward(
        self,
        x: Any,
        edge_index: Any,
        edge_weight: Any = None,
        edge_mask: Any = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: Any) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> list[tuple[str, np.ndarray]]:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for grad in self.grads.values():
            grad[...] = 0.0


class GCNLayer(GraphLayer):
    """The graph convolution, PyG's ``GCNConv`` arithmetic in numpy.

    ``out_i = relu(sum_{(j,i)} w'_{j,i} * m_{j,i} * (x_j @ W.T)_i + b)``
    -- the transformed features are aggregated (PyG applies ``lin``
    BEFORE ``propagate``), with ``w'`` from :func:`gcn_norm` (recomputed
    each forward unless ``edge_weight`` hands in pre-normalised values)
    and ``m`` an optional per-edge multiplicative gate: the
    GNNExplainer injection point (the caller gates with a sigmoid; this
    layer just multiplies).  ``W`` is [out, in] glorot-uniform, ``b``
    zeros.

    ``backward`` fills ``grads["weights"]``, ``grads["bias"]`` and
    ``grads["edge_mask"]`` and returns the gradient w.r.t. ``x``.  The
    edge-mask entry is the gradient w.r.t. the MASK argument as passed
    (the quantity a mask-training explainer needs): the per-edge dot
    of the routed upstream gradient with the transformed SOURCE
    features, scaled by the normalised edge weight -- the chain-rule
    factor the plain "effective weight" gradient would omit.
    """

    def __init__(self, in_channels: int, out_channels: int, seed: int = 0) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise GeometricError(
                f"GCNLayer channels must be >= 1, got in={in_channels}, "
                f"out={out_channels}"
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (self.in_channels + self.out_channels))
        self.weights = rng.uniform(-limit, limit, (self.out_channels, self.in_channels))
        self.bias = np.zeros(self.out_channels, dtype=float)
        self.grads = {
            "weights": np.zeros_like(self.weights),
            "bias": np.zeros_like(self.bias),
            "edge_mask": np.zeros(0, dtype=float),
        }
        self._cache: dict[str, Any] = {}

    def params(self) -> list[tuple[str, np.ndarray]]:
        return [("weights", self.weights), ("bias", self.bias)]

    def forward(
        self,
        x: Any,
        edge_index: Any,
        edge_weight: Any = None,
        edge_mask: Any = None,
    ) -> np.ndarray:
        features = _as_node_features(x, self.in_channels)
        num_nodes = features.shape[0]
        edges_in = _as_edge_index(edge_index, num_nodes)
        edges, weight = gcn_norm(edges_in, num_nodes, edge_weight=edge_weight)
        mask = _as_edge_mask(edge_mask, edges.shape[1])
        effective = weight * mask
        sources, targets = edges[0], edges[1]
        transformed = features @ self.weights.T
        messages = transformed[sources] * effective[:, None]
        aggregated = _index_add(targets, messages, num_nodes)
        out = np.maximum(aggregated + self.bias, 0.0)
        self._cache = {
            "features": features,
            "sources": sources,
            "targets": targets,
            "transformed": transformed,
            "weight": weight,
            "effective": effective,
            "out": out,
        }
        self.grads["edge_mask"] = np.zeros(edges.shape[1], dtype=float)
        return out

    def backward(self, grad_out: Any) -> np.ndarray:
        if not self._cache:
            raise GeometricError("backward needs a forward cache (call forward first)")
        upstream = np.asarray(grad_out, dtype=float)
        if upstream.shape != self._cache["out"].shape:
            raise GeometricError(
                f"grad_out shape {upstream.shape} != forward output "
                f"{self._cache['out'].shape}"
            )
        routed = upstream * (self._cache["out"] > 0.0)
        self.grads["bias"] += routed.sum(axis=0)
        targets = self._cache["targets"]
        sources = self._cache["sources"]
        transformed = self._cache["transformed"]
        self.grads["edge_mask"] += np.einsum(
            "ec,ec->e", routed[targets], transformed[sources]
        ) * self._cache["weight"]
        d_transformed = _index_add(
            sources, routed[targets] * self._cache["effective"][:, None], upstream.shape[0]
        )
        self.grads["weights"] += d_transformed.T @ self._cache["features"]
        return d_transformed @ self.weights


class SAGELayer(GraphLayer):
    """The graphsage convolution, PyG's ``SAGEConv`` in numpy.

    ``out_i = (mean over in-neighbours j of m_{j,i} * x_j) @ W_l.T +
    x_i @ W_r.T + b_l`` -- the bias sits on ``lin_l`` ONLY (PyG's
    root-vs-neighbour split), the neighbour count is clamped to a
    minimum of 1 so an isolated node aggregates ZEROES (PyG's mean
    aggregation over an empty set), and both weights are
    U(+/- 1/sqrt(fan_in)).  The edge list is consumed as-is (no
    self-loop surgery -- PyG's SAGEConv adds none).  ``edge_weight`` is
    refused: the neighbour mean is unweighted.

    ``backward`` fills ``grads["weights_l"]``, ``grads["weights_r"]``,
    ``grads["bias"]``, ``grads["edge_mask"]`` and returns the gradient
    w.r.t. ``x`` (root term plus the scatter-mean of the neighbours).
    """

    def __init__(self, in_channels: int, out_channels: int, seed: int = 0) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise GeometricError(
                f"SAGELayer channels must be >= 1, got in={in_channels}, "
                f"out={out_channels}"
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        rng = np.random.default_rng(seed)
        limit = 1.0 / np.sqrt(self.in_channels)
        self.weights_l = rng.uniform(-limit, limit, (self.out_channels, self.in_channels))
        self.weights_r = rng.uniform(-limit, limit, (self.out_channels, self.in_channels))
        self.bias = np.zeros(self.out_channels, dtype=float)
        self.grads = {
            "weights_l": np.zeros_like(self.weights_l),
            "weights_r": np.zeros_like(self.weights_r),
            "bias": np.zeros_like(self.bias),
            "edge_mask": np.zeros(0, dtype=float),
        }
        self._cache: dict[str, Any] = {}

    def params(self) -> list[tuple[str, np.ndarray]]:
        return [
            ("weights_l", self.weights_l),
            ("weights_r", self.weights_r),
            ("bias", self.bias),
        ]

    def forward(
        self,
        x: Any,
        edge_index: Any,
        edge_weight: Any = None,
        edge_mask: Any = None,
    ) -> np.ndarray:
        if edge_weight is not None:
            raise GeometricError(
                "SAGELayer aggregates an unweighted neighbour mean; "
                "edge_weight is not supported"
            )
        features = _as_node_features(x, self.in_channels)
        num_nodes = features.shape[0]
        edges = _as_edge_index(edge_index, num_nodes)
        sources, targets = edges[0], edges[1]
        mask = _as_edge_mask(edge_mask, edges.shape[1])
        counts = np.maximum(
            np.bincount(targets, minlength=num_nodes).astype(float), 1.0
        )
        sums = _index_add(targets, features[sources] * mask[:, None], num_nodes)
        aggregated = sums / counts[:, None]
        out = aggregated @ self.weights_l.T + features @ self.weights_r.T + self.bias
        self._cache = {
            "features": features,
            "sources": sources,
            "targets": targets,
            "mask": mask,
            "counts": counts,
            "aggregated": aggregated,
        }
        self.grads["edge_mask"] = np.zeros(edges.shape[1], dtype=float)
        return out

    def backward(self, grad_out: Any) -> np.ndarray:
        if not self._cache:
            raise GeometricError("backward needs a forward cache (call forward first)")
        upstream = np.asarray(grad_out, dtype=float)
        if upstream.shape[0] != self._cache["features"].shape[0] or upstream.shape[1] != (
            self.out_channels
        ):
            raise GeometricError(
                f"grad_out shape {upstream.shape} != forward output "
                f"({self._cache['features'].shape[0]}, {self.out_channels})"
            )
        self.grads["bias"] += upstream.sum(axis=0)
        self.grads["weights_l"] += upstream.T @ self._cache["aggregated"]
        self.grads["weights_r"] += upstream.T @ self._cache["features"]
        d_aggregated = upstream @ self.weights_l
        d_root = upstream @ self.weights_r
        d_sums = d_aggregated / self._cache["counts"][:, None]
        targets = self._cache["targets"]
        sources = self._cache["sources"]
        self.grads["edge_mask"] += np.einsum(
            "ec,ec->e", d_sums[targets], self._cache["features"][sources]
        )
        d_neighbours = _index_add(
            sources, d_sums[targets] * self._cache["mask"][:, None], upstream.shape[0]
        )
        return d_root + d_neighbours


class GATLayer(GraphLayer):
    """The graph attention convolution, PyG's ``GATConv`` in numpy.

    ``z = x @ W.T`` with ``W`` [heads*out, in]; per-edge per-head logits
    ``alpha = z_j . a_src[h] + z_i . a_dst[h]`` passed through
    leaky-ReLU (``negative_slope``) and a per-TARGET-node softmax with
    the ``+1e-16`` denominator; messages ``softmax * z_j`` aggregate by
    target into [N, heads, out] and the heads concatenate into
    [N, heads*out] plus a zero bias.  Self-loops are removed then
    re-added (every node attends to itself).  ``edge_weight`` is
    refused -- attention IS the weighting.  ``edge_mask`` gates each
    message after the softmax (the masked-path contract shared with
    :class:`GCNLayer`).  ``last_attention`` exposes the [E, heads]
    softmax alphas for GUI attention grids.

    ``backward`` fills ``grads`` for ``weights`` / ``bias`` /
    ``att_src`` / ``att_dst`` / ``edge_mask`` -- including the exact
    softmax backward ``dlogit_e = alpha_e * (g_e - sum_{e' in group}
    alpha_{e'} g_{e'})`` -- and returns the gradient w.r.t. ``x``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        negative_slope: float = DEFAULT_GAT_SLOPE,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1 or heads < 1:
            raise GeometricError(
                f"GATLayer channels/heads must be >= 1, got in={in_channels}, "
                f"out={out_channels}, heads={heads}"
            )
        if not 0.0 < negative_slope < 1.0:
            raise GeometricError(
                f"negative_slope must be in (0, 1), got {negative_slope}"
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.heads = int(heads)
        self.negative_slope = float(negative_slope)
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (self.in_channels + self.heads * self.out_channels))
        self.weights = rng.uniform(
            -limit, limit, (self.heads * self.out_channels, self.in_channels)
        )
        att_limit = np.sqrt(6.0 / (self.out_channels + self.heads))
        self.att_src = rng.uniform(-att_limit, att_limit, (self.heads, self.out_channels))
        self.att_dst = rng.uniform(-att_limit, att_limit, (self.heads, self.out_channels))
        self.bias = np.zeros(self.heads * self.out_channels, dtype=float)
        self.grads = {
            "weights": np.zeros_like(self.weights),
            "bias": np.zeros_like(self.bias),
            "att_src": np.zeros_like(self.att_src),
            "att_dst": np.zeros_like(self.att_dst),
            "edge_mask": np.zeros(0, dtype=float),
        }
        self.last_attention: np.ndarray = np.zeros((0, self.heads), dtype=float)
        self._cache: dict[str, Any] = {}

    def params(self) -> list[tuple[str, np.ndarray]]:
        return [
            ("weights", self.weights),
            ("bias", self.bias),
            ("att_src", self.att_src),
            ("att_dst", self.att_dst),
        ]

    def forward(
        self,
        x: Any,
        edge_index: Any,
        edge_weight: Any = None,
        edge_mask: Any = None,
    ) -> np.ndarray:
        if edge_weight is not None:
            raise GeometricError(
                "GATLayer computes its own attention weights; edge_weight is not supported"
            )
        features = _as_node_features(x, self.in_channels)
        num_nodes = features.shape[0]
        edges = add_self_loops(
            remove_self_loops(_as_edge_index(edge_index, num_nodes)), num_nodes
        )
        sources, targets = edges[0], edges[1]
        mask = _as_edge_mask(edge_mask, edges.shape[1])
        z = features @ self.weights.T
        z_heads = z.reshape(num_nodes, self.heads, self.out_channels)
        z_src, z_tgt = z_heads[sources], z_heads[targets]
        logits = np.einsum("eho,ho->eh", z_src, self.att_src) + np.einsum(
            "eho,ho->eh", z_tgt, self.att_dst
        )
        leaky = np.where(logits > 0.0, logits, self.negative_slope * logits)
        shifted = np.exp(leaky - leaky.max())
        denominator = np.stack(
            [
                np.bincount(targets, weights=shifted[:, head], minlength=num_nodes)
                for head in range(self.heads)
            ],
            axis=1,
        ) + ATTENTION_EPS
        alpha = shifted / denominator[targets]
        messages = alpha[:, :, None] * mask[:, None, None] * z_src
        aggregated = _index_add(targets, messages, num_nodes)
        out = aggregated.reshape(num_nodes, self.heads * self.out_channels) + self.bias
        self.last_attention = alpha
        self._cache = {
            "features": features,
            "sources": sources,
            "targets": targets,
            "mask": mask,
            "z_heads": z_heads,
            "z_src": z_src,
            "z_tgt": z_tgt,
            "alpha": alpha,
            "leaky_pos": logits > 0.0,
        }
        self.grads["edge_mask"] = np.zeros(edges.shape[1], dtype=float)
        return out

    def backward(self, grad_out: Any) -> np.ndarray:
        if not self._cache:
            raise GeometricError("backward needs a forward cache (call forward first)")
        upstream = np.asarray(grad_out, dtype=float)
        num_nodes = self._cache["features"].shape[0]
        expected = (num_nodes, self.heads * self.out_channels)
        if upstream.shape != expected:
            raise GeometricError(
                f"grad_out shape {upstream.shape} != forward output {expected}"
            )
        targets = self._cache["targets"]
        sources = self._cache["sources"]
        alpha = self._cache["alpha"]
        mask = self._cache["mask"]
        z_src = self._cache["z_src"]
        z_tgt = self._cache["z_tgt"]
        d_heads = upstream.reshape(num_nodes, self.heads, self.out_channels)
        self.grads["bias"] += d_heads.sum(axis=0).reshape(-1)
        self.grads["edge_mask"] += np.einsum(
            "eho,eho->e", d_heads[targets], alpha[:, :, None] * z_src
        )
        d_alpha = np.einsum(
            "eho,eho->eh", d_heads[targets] * mask[:, None, None], z_src
        )
        d_z = _index_add(
            sources, alpha[:, :, None] * mask[:, None, None] * d_heads[targets], num_nodes
        )
        group_sums = _index_add(targets, alpha * d_alpha, num_nodes)
        d_logit = alpha * (d_alpha - group_sums[targets])
        d_logit = d_logit * np.where(
            self._cache["leaky_pos"], 1.0, self.negative_slope
        )
        d_z = d_z + _index_add(
            sources, d_logit[:, :, None] * self.att_src[None, :, :], num_nodes
        )
        d_z = d_z + _index_add(
            targets, d_logit[:, :, None] * self.att_dst[None, :, :], num_nodes
        )
        self.grads["att_src"] += np.einsum("eh,eho->ho", d_logit, z_src)
        self.grads["att_dst"] += np.einsum("eh,eho->ho", d_logit, z_tgt)
        d_flat = d_z.reshape(num_nodes, self.heads * self.out_channels)
        self.grads["weights"] += d_flat.T @ self._cache["features"]
        return d_flat @ self.weights


# --------------------------------------------------------------------------- #
# Pooling + batching -- PyG's global pool / Batch contract
# --------------------------------------------------------------------------- #
def _as_batch(batch: Any, num_nodes: int, size: int | None) -> tuple[np.ndarray, int]:
    if batch is None:
        return np.zeros(num_nodes, dtype=np.int64), 1
    raw = np.asarray(batch)
    if raw.ndim != 1 or raw.size != num_nodes:
        raise GeometricError(
            f"batch must be a length-{num_nodes} node->graph id vector, "
            f"got shape {raw.shape}"
        )
    ids = raw.astype(np.int64)
    if ids.size and not np.array_equal(ids, raw):
        raise GeometricError("batch entries must be integers")
    if int(ids.min()) < 0:
        raise GeometricError(f"batch graph ids must be >= 0, got min {int(ids.min())}")
    count = int(size) if size is not None else int(ids.max()) + 1
    if count < 1:
        raise GeometricError(f"batch size must be >= 1, got {count}")
    if int(ids.max()) >= count:
        raise GeometricError(
            f"batch references graph {int(ids.max())} outside [0, {count})"
        )
    return ids, count


def global_add_pool(x: Any, batch: Any = None, size: int | None = None) -> np.ndarray:
    """Sum the node features of every graph in the batch -> [B, F].

    Empty groups read as ZERO rows; ``batch=None`` reduces the whole
    matrix as one graph.
    """
    features = _as_node_features(x)
    ids, count = _as_batch(batch, features.shape[0], size)
    return _index_add(ids, features, count)


def global_mean_pool(x: Any, batch: Any = None, size: int | None = None) -> np.ndarray:
    """Mean of each graph's node features -> [B, F], counts clamped to
    a minimum of 1 so empty groups give clean zero rows.

    >>> global_mean_pool(np.array([[1.], [3.], [5.], [7.]]),
    ...                  np.array([0, 0, 1, 1])).tolist()
    [[2.0], [6.0]]
    """
    features = _as_node_features(x)
    ids, count = _as_batch(batch, features.shape[0], size)
    pooled, _ = _mean_pool_parts(features, ids, count)
    return pooled


def _mean_pool_parts(
    features: np.ndarray, ids: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    sums = _index_add(ids, features, count)
    counts = np.maximum(
        np.bincount(ids, minlength=count).astype(float), 1.0
    )
    return sums / counts[:, None], counts


def global_max_pool(x: Any, batch: Any = None, size: int | None = None) -> np.ndarray:
    """Per-graph max of the node features -> [B, F]; empty groups (and
    single-graph empties) read as ZERO rows."""
    features = _as_node_features(x)
    ids, count = _as_batch(batch, features.shape[0], size)
    pooled = np.full((count, features.shape[1]), -np.inf)
    np.maximum.at(pooled, ids, features)
    pooled[~np.isfinite(pooled)] = 0.0
    return pooled


@dataclass
class GraphRecord:
    """One labelled graph: node features, edge list, class label.

    ``motif_edges`` carries the ground-truth motif pairs (canonical
    sorted endpoints, this graph's node indexing) for the synthetic
    :func:`motif_dataset` -- the evaluation hook the explainability
    tests score against.  It is empty when no motif was planted.
    """

    x: np.ndarray
    edge_index: np.ndarray
    y: int
    name: str | None = None
    motif_edges: tuple[tuple[int, int], ...] = ()


def collate_graphs(
    graphs: Sequence[GraphRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PyG's ``Batch`` contract: many graphs into one big disconnected
    graph.  Edge lists are offset by the exclusive cumsum of node
    counts, features stack in order, ``batch`` maps every node to its
    graph id, and ``y`` collects the labels -> ``(x, edge_index,
    batch, y)``.

    >>> a = GraphRecord(np.zeros((2, 1)), np.array([[0], [1]]), 0)
    >>> b = GraphRecord(np.ones((3, 1)), np.array([[0, 1], [1, 2]]), 1)
    >>> x, edges, batch, y = collate_graphs([a, b])
    >>> edges.tolist(), batch.tolist(), y.tolist()
    ([[0, 2, 3], [1, 3, 4]], [0, 0, 1, 1, 1], [0, 1])
    """
    records = list(graphs)
    if not records:
        raise GeometricError("collate_graphs needs at least one GraphRecord")
    widths = {int(np.asarray(record.x).shape[1]) for record in records}
    if len(widths) > 1:
        raise GeometricError(
            f"all graphs must share the feature width, got {sorted(widths)}"
        )
    node_counts = [int(np.asarray(record.x).shape[0]) for record in records]
    if min(node_counts) < 1:
        raise GeometricError("every graph needs at least one node")
    offsets = np.concatenate([[0], np.cumsum(node_counts)]).astype(np.int64)
    features = np.vstack([np.asarray(record.x, dtype=float) for record in records])
    edge_blocks = [
        _as_edge_index(record.edge_index, node_counts[position]) + offsets[position]
        for position, record in enumerate(records)
    ]
    edges = np.concatenate(edge_blocks, axis=1)
    batch = np.repeat(
        np.arange(len(records), dtype=np.int64), np.asarray(node_counts)
    )
    y = np.asarray([int(record.y) for record in records], dtype=np.int64)
    return features, edges, batch, y


# --------------------------------------------------------------------------- #
# The optimiser -- Adam over plain numpy arrays
# --------------------------------------------------------------------------- #
class _Adam:
    """Adam with bias correction over float64 arrays mutated IN PLACE.

    ``ml/neural.py``'s Adam is torch's and ``ml/autograd.py``'s drives
    graph-built ``Tensor`` objects; neither fits layers that accumulate
    plain-array grads by hand, so this is the local minimal Adam (same
    arithmetic as the house autograd one: bias-corrected m/v, no
    weight decay).  ``step`` receives the gradient list aligned with
    the constructor's array list.

    >>> param = np.array([8.0])
    >>> opt = _Adam([param], lr=0.2)
    >>> for _ in range(300):
    ...     opt.step([2.0 * (param ** 2 - 4.0) * param])
    >>> round(float(param[0]), 2)
    2.0
    """

    def __init__(
        self,
        arrays: Sequence[np.ndarray],
        lr: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise GeometricError(f"Adam lr must be > 0, got {lr}")
        self._arrays = [np.asarray(a, dtype=float) for a in arrays]
        for original, converted in zip(arrays, self._arrays):
            if original is not converted:
                raise GeometricError(
                    "Adam params must be float64 arrays held by reference "
                    "(in-place updates)"
                )
        self._lr = float(lr)
        self._betas = (float(betas[0]), float(betas[1]))
        self._eps = float(eps)
        self._m = [np.zeros_like(a) for a in self._arrays]
        self._v = [np.zeros_like(a) for a in self._arrays]
        self._t = 0

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """One Adam update; ``grads`` aligns with the constructor arrays."""
        if len(grads) != len(self._arrays):
            raise GeometricError(
                f"Adam step needs {len(self._arrays)} grads, got {len(grads)}"
            )
        self._t += 1
        beta1, beta2 = self._betas
        for index, (param, grad) in enumerate(zip(self._arrays, grads)):
            g = np.asarray(grad, dtype=float)
            if g.shape != param.shape:
                raise GeometricError(
                    f"Adam grad shape {g.shape} != param shape {param.shape}"
                )
            self._m[index] = beta1 * self._m[index] + (1.0 - beta1) * g
            self._v[index] = beta2 * self._v[index] + (1.0 - beta2) * (g * g)
            m_hat = self._m[index] / (1.0 - beta1**self._t)
            v_hat = self._v[index] / (1.0 - beta2**self._t)
            param -= self._lr * m_hat / (np.sqrt(v_hat) + self._eps)


# --------------------------------------------------------------------------- #
# The GNN trainer
# --------------------------------------------------------------------------- #
class GNNClassifier:
    """A two-conv graph classifier: conv -> relu -> conv -> relu ->
    ``global_mean_pool`` (graph level) -> linear head -> softmax.

    ``layer_kind`` selects the convolution (``"gcn"`` / ``"sage"`` /
    ``"gat"``); every parameter -- both conv layers and the head -- is
    driven by one :class:`_Adam` over the collated batch (PyG's
    ``Batch`` as one big disconnected graph).  Deterministic per
    ``seed`` end to end.

    ``forward_logits`` is the masked path the explainer consumes:
    ``edge_mask`` is a per-edge gate over the CANONICAL edge list
    (gcn_norm's loop-augmented ordering, loops appended last) that
    multiplies each conv's normalised weights; the caller pins the
    self-loop entries to 1.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        n_classes: int,
        layer_kind: str = "gcn",
        seed: int = 0,
    ) -> None:
        if in_channels < 1 or hidden < 1:
            raise GeometricError(
                f"in_channels/hidden must be >= 1, got {in_channels}/{hidden}"
            )
        if n_classes < 2:
            raise GeometricError(f"n_classes must be >= 2, got {n_classes}")
        if layer_kind not in ("gcn", "sage", "gat"):
            raise GeometricError(
                f"layer_kind must be one of 'gcn', 'sage', 'gat', got {layer_kind!r}"
            )
        self.in_channels = int(in_channels)
        self.hidden = int(hidden)
        self.n_classes = int(n_classes)
        self.layer_kind = str(layer_kind)
        rng = np.random.default_rng(seed)
        self.conv1 = self._make_layer(
            self.in_channels, self.hidden, int(rng.integers(0, 2**31 - 1))
        )
        self.conv2 = self._make_layer(
            self.hidden, self.hidden, int(rng.integers(0, 2**31 - 1))
        )
        limit = np.sqrt(6.0 / (self.hidden + self.n_classes))
        self.head_w = rng.uniform(-limit, limit, (self.n_classes, self.hidden))
        self.head_b = np.zeros(self.n_classes, dtype=float)
        self._head_grads = {
            "head_w": np.zeros_like(self.head_w),
            "head_b": np.zeros_like(self.head_b),
        }
        self._fitted = False

    def _make_layer(self, in_channels: int, out_channels: int, seed: int) -> GraphLayer:
        if self.layer_kind == "gcn":
            return GCNLayer(in_channels, out_channels, seed=seed)
        if self.layer_kind == "sage":
            return SAGELayer(in_channels, out_channels, seed=seed)
        return GATLayer(in_channels, out_channels, seed=seed)

    def _param_pairs(self) -> list[tuple[np.ndarray, np.ndarray]]:
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in (self.conv1, self.conv2):
            for name, array in layer.params():
                pairs.append((array, layer.grads[name]))
        pairs.append((self.head_w, self._head_grads["head_w"]))
        pairs.append((self.head_b, self._head_grads["head_b"]))
        return pairs

    def _zero_grads(self) -> None:
        self.conv1.zero_grad()
        self.conv2.zero_grad()
        self._head_grads["head_w"][...] = 0.0
        self._head_grads["head_b"][...] = 0.0

    def _split_mask(
        self, edge_mask: Any, edge_index: np.ndarray, num_nodes: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if edge_mask is None:
            return None, None
        mask = np.asarray(edge_mask, dtype=float).reshape(-1)
        non_loop = int((edge_index[0] != edge_index[1]).sum())
        canonical = non_loop + num_nodes
        if self.layer_kind == "sage":
            if non_loop != edge_index.shape[1]:
                raise GeometricError(
                    "the sage masked path needs a self-loop-free edge list "
                    "(SAGE consumes the raw list)"
                )
            if mask.size == non_loop:
                return mask, mask
            if mask.size == canonical:
                return mask[:non_loop], mask[:non_loop]
            raise GeometricError(
                f"edge_mask needs {non_loop} (or {canonical}) entries, got {mask.size}"
            )
        if mask.size != canonical:
            raise GeometricError(
                f"edge_mask needs {canonical} entries (canonical loop-augmented "
                f"list), got {mask.size}"
            )
        return mask, mask

    def _forward_cached(
        self,
        x: Any,
        edge_index: Any,
        batch: Any = None,
        edge_mask: Any = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        features = _as_node_features(x, self.in_channels)
        num_nodes = features.shape[0]
        edges = _as_edge_index(edge_index, num_nodes)
        mask1, mask2 = self._split_mask(edge_mask, edges, num_nodes)
        h1 = np.maximum(self.conv1.forward(features, edges, edge_mask=mask1), 0.0)
        h2 = np.maximum(self.conv2.forward(h1, edges, edge_mask=mask2), 0.0)
        ids, count = _as_batch(batch, num_nodes, None)
        pooled, counts = _mean_pool_parts(h2, ids, count)
        logits = pooled @ self.head_w.T + self.head_b
        cache: dict[str, Any] = {
            "h1": h1,
            "h2": h2,
            "ids": ids,
            "count": count,
            "counts": counts,
            "pooled": pooled,
            "batched": batch is not None,
        }
        return logits, cache

    def _backward(self, cache: dict[str, Any], grad_logits: np.ndarray) -> np.ndarray:
        ids = cache["ids"]
        counts = cache["counts"]
        d_pooled = grad_logits @ self.head_w
        self._head_grads["head_w"] += grad_logits.T @ cache["pooled"]
        self._head_grads["head_b"] += grad_logits.sum(axis=0)
        if cache["batched"]:
            d_h2 = d_pooled[ids] / counts[ids][:, None]
        else:
            d_h2 = np.repeat(d_pooled / counts[0], cache["h2"].shape[0], axis=0)
        d_h2 = d_h2 * (cache["h2"] > 0.0)
        d_h1 = self.conv2.backward(d_h2)
        d_h1 = d_h1 * (cache["h1"] > 0.0)
        return self.conv1.backward(d_h1)

    # -- public surface -------------------------------------------------- #
    def forward_logits(
        self,
        x: Any,
        edge_index: Any,
        batch: Any = None,
        edge_mask: Any = None,
    ) -> np.ndarray:
        """Class logits [B, n_classes]; the masked path (see class docstring)."""
        logits, _ = self._forward_cached(x, edge_index, batch, edge_mask)
        return logits

    def predict_proba(
        self,
        x: Any,
        edge_index: Any,
        batch: Any = None,
        edge_mask: Any = None,
    ) -> np.ndarray:
        """Softmax probabilities [B, n_classes] (stable max-shift)."""
        return _softmax(self.forward_logits(x, edge_index, batch, edge_mask))

    def predict(
        self,
        x: Any,
        edge_index: Any,
        batch: Any = None,
        edge_mask: Any = None,
    ) -> np.ndarray:
        """Argmax class ids, one per graph in the batch."""
        return np.argmax(
            self.forward_logits(x, edge_index, batch, edge_mask), axis=1
        ).astype(np.int64)

    def node_embeddings(self, x: Any, edge_index: Any, batch: Any = None) -> np.ndarray:
        """Post-conv2 node embeddings [N, hidden] (the GUI's grid source)."""
        _, cache = self._forward_cached(x, edge_index, batch)
        return cache["h2"]

    def fit(
        self,
        graphs: Sequence[GraphRecord],
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> GNNClassifier:
        """Full-batch Adam training over the collated graphs.

        Loss is mean cross-entropy computed on logits via logsumexp
        (numerically stable); the batch is collated ONCE, so every epoch
        is a deterministic pass over the same big disconnected graph.
        """
        records = list(graphs)
        if not records:
            raise GeometricError("fit needs at least one GraphRecord")
        if epochs < 1:
            raise GeometricError(f"epochs must be >= 1, got {epochs}")
        features, edges, batch, y = collate_graphs(records)
        if features.shape[1] != self.in_channels:
            raise GeometricError(
                f"graphs carry {features.shape[1]} feature channels, "
                f"model expects {self.in_channels}"
            )
        if y.min() < 0 or y.max() >= self.n_classes:
            raise GeometricError(
                f"labels must lie in [0, {self.n_classes}), got "
                f"[{int(y.min())}, {int(y.max())}]"
            )
        pairs = self._param_pairs()
        opt = _Adam([param for param, _ in pairs], lr=lr)
        rows = np.arange(y.size)
        onehot = np.eye(self.n_classes, dtype=float)[y]
        report = max(1, int(epochs) // 10)
        for epoch in range(int(epochs)):
            logits, cache = self._forward_cached(features, edges, batch)
            shifted = logits - logits.max(axis=1, keepdims=True)
            sum_exp = np.exp(shifted).sum(axis=1)
            loss = float(np.mean(np.log(sum_exp) - logits[rows, y]))
            probs = np.exp(shifted) / sum_exp[:, None]
            self._zero_grads()
            self._backward(cache, (probs - onehot) / y.size)
            opt.step([grad for _, grad in pairs])
            if verbose and (epoch % report == 0 or epoch == int(epochs) - 1):
                _LOGGER.info(
                    "geometric fit epoch %d/%d ce-loss %.6f", epoch, int(epochs), loss
                )
        self._fitted = True
        return self

    def score(self, graphs: Sequence[GraphRecord]) -> float:
        """Classification accuracy over a list of :class:`GraphRecord`."""
        records = list(graphs)
        if not records:
            raise GeometricError("score needs at least one GraphRecord")
        features, edges, batch, y = collate_graphs(records)
        predicted = self.predict(features, edges, batch)
        return float((predicted == y).mean())


# --------------------------------------------------------------------------- #
# The motif dataset -- the GNNExplainer paper's synthetic, graph level
# --------------------------------------------------------------------------- #
def motif_dataset(
    n_graphs: int = 120, seed: int = 0, motif: str = "house"
) -> list[GraphRecord]:
    """The GNNExplainer synthetic: random base graphs, half carrying a
    house motif.

    Label 1: an Erdos-Renyi base graph (10-16 nodes, p=0.25) with the
    5-node house (:data:`HOUSE_MOTIF_EDGES` -- square plus roof)
    attached by one edge between a random base node and motif node 0.
    Label 0: the plain base graph.  Node features are ``[1.0,
    degree/10]`` ONLY -- the label lives in the structure, never in
    the features (no leak).  Every undirected edge ships in both
    directions.  ``motif='house'`` is the only motif (a decoy variant
    is honestly not implemented); ``motif_edges`` on each positive
    record carries the six planted pairs for evaluation.

    Labels alternate (even record index -> 1), so ``n_graphs`` even
    gives an exactly balanced set.
    """
    if motif != "house":
        raise GeometricError(
            f"motif must be 'house' (the only house motif), got {motif!r}"
        )
    if n_graphs < 1:
        raise GeometricError(f"n_graphs must be >= 1, got {n_graphs}")
    rng = np.random.default_rng(seed)
    records: list[GraphRecord] = []
    for index in range(int(n_graphs)):
        label = 1 if index % 2 == 0 else 0
        n_base = int(rng.integers(10, 17))
        pairs: set[tuple[int, int]] = set()
        for low in range(n_base):
            for high in range(low + 1, n_base):
                if rng.random() < 0.25:
                    pairs.add((low, high))
        motif_pairs: tuple[tuple[int, int], ...] = ()
        if label == 1:
            offset = n_base
            motif_pairs = tuple(
                (low + offset, high + offset) for low, high in HOUSE_MOTIF_EDGES
            )
            pairs.update(motif_pairs)
            attach = int(rng.integers(0, n_base))
            pairs.add((attach, offset))
        directed: list[list[int]] = [[], []]
        for low, high in sorted(pairs):
            directed[0].extend([low, high])
            directed[1].extend([high, low])
        edge_index = np.asarray(directed, dtype=np.int64)
        node_degrees = np.zeros(n_base + (5 if label == 1 else 0), dtype=float)
        for low, high in pairs:
            node_degrees[low] += 1.0
            node_degrees[high] += 1.0
        features = np.column_stack(
            [np.ones(node_degrees.size), node_degrees / 10.0]
        )
        records.append(
            GraphRecord(
                x=features,
                edge_index=edge_index,
                y=label,
                name=f"motif-{index}",
                motif_edges=motif_pairs,
            )
        )
    return records


# --------------------------------------------------------------------------- #
# GNNExplainer -- DIG's per-class edge masks, clean-room
# --------------------------------------------------------------------------- #
def control_sparsity(scores: Any, sparsity: float) -> np.ndarray:
    """Binarise scores keeping the top ``(1 - sparsity)`` fraction.

    ``split_point = int((1.0 - sparsity) * len(scores))`` entries stay
    1 (descending score, ties broken by position -- DIG's
    ``control_sparsity`` arithmetic); the rest are 0.

    >>> control_sparsity([0.1, 0.9, 0.5, 0.7], 0.5).astype(int).tolist()
    [0, 1, 0, 1]
    """
    values = np.asarray(scores, dtype=float).reshape(-1)
    if not 0.0 <= sparsity <= 1.0:
        raise GeometricError(f"sparsity must be in [0, 1], got {sparsity}")
    split_point = int((1.0 - sparsity) * values.size)
    order = np.argsort(-values, kind="stable")
    mask = np.zeros(values.size, dtype=float)
    mask[order[:split_point]] = 1.0
    return mask


@dataclass
class GnnExplanation:
    """One graph's GNNExplainer verdict.

    ``edge_index`` is the canonical loop-augmented list (loops LAST);
    ``edge_scores`` / ``hard_mask`` map every class to its sigmoid
    scores / 0-1 mask over that list (self-loop entries pinned to 1);
    ``probs`` carries DIG's 4-forward protocol at ``label`` (origin /
    masked / maskout / zero), with ``fidelity = p_origin - p_maskout``
    and ``fidelity_inv = p_origin - p_masked``; ``sparsity`` is the
    fraction of maskable (non-loop) edges dropped by the hard mask.
    """

    label: int
    edge_index: np.ndarray
    edge_scores: dict[int, np.ndarray]
    hard_mask: dict[int, np.ndarray]
    probs: dict[str, float]
    fidelity: float
    fidelity_inv: float
    sparsity: float

    def top_edges(self, k: int = 10, label: int | None = None) -> list[tuple[float, int, int]]:
        """The ``k`` highest-scoring edges as ``(score, source, target)``.

        Self-loops are pinned by construction and carry no ranking
        signal, so they are excluded; ties break by edge position.
        """
        if k < 0:
            raise GeometricError(f"k must be >= 0, got {k}")
        chosen = self.label if label is None else int(label)
        if chosen not in self.edge_scores:
            raise GeometricError(f"no edge scores for class {chosen}")
        scores = np.asarray(self.edge_scores[chosen], dtype=float)
        sources, targets = self.edge_index[0], self.edge_index[1]
        positions = np.flatnonzero(sources != targets)
        order = positions[np.argsort(-scores[positions], kind="stable")]
        return [
            (float(scores[position]), int(sources[position]), int(targets[position]))
            for position in order[: int(k)]
        ]


def gnn_explain(
    model: GNNClassifier,
    x: Any,
    edge_index: Any,
    y: int | None = None,
    epochs: int = 100,
    lr: float = 0.01,
    coff_edge_size: float = 0.001,
    coff_edge_ent: float = 0.001,
    sparsity: float = 0.7,
    seed: int = 0,
) -> GnnExplanation:
    """DIG's graph-level GNNExplainer, clean-room in numpy.

    One mask per class is trained from scratch (logits ~ N(0,
    sqrt(2/N)) over the canonical loop-augmented edge list; self-loop
    entries are pinned to sigmoid 1 and never updated) by Adam on
    ``CE(logits at class, masked) + coff_edge_size * sum(sigmoid) +
    coff_edge_ent * mean binary entropy``.  Injection is
    sigmoid-gated EVERYWHERE: the effective edge weight is ``w' *
    sigmoid(logit)``.  Honest delta: DIG multiplies by the RAW mask
    during optimisation (their quirk) and sigmoids only inside the
    loss; gating with the sigmoid in the forward too is numerically
    saner and lands in the same optima neighbourhood.

    The 4-forward protocol then scores the label class: ``fidelity =
    p_origin - p_maskout`` (explanation removed), ``fidelity_inv =
    p_origin - p_masked`` (explanation kept); the maskout complement
    and the zero forward keep the pinned self-loops on, and
    ``sparsity`` is measured over the maskable edges only.  A feature
    mask is NOT trained (DIG's ``coff_node_feat`` is N/A here: the
    motif features are two tabular columns -- the honest scope note).
    """
    if not isinstance(model, GNNClassifier):
        raise GeometricError(
            "gnn_explain needs a GNNClassifier (the numpy geometric model)"
        )
    features = _as_node_features(x, model.in_channels)
    num_nodes = features.shape[0]
    edges_in = _as_edge_index(edge_index, num_nodes)
    canonical, _ = gcn_norm(edges_in, num_nodes)
    n_loops = num_nodes
    n_non_loop = canonical.shape[1] - n_loops
    if y is None:
        y = int(model.predict(features, edges_in)[0])
    y = int(y)
    if not 0 <= y < model.n_classes:
        raise GeometricError(f"y must be in [0, {model.n_classes}), got {y}")
    if epochs < 1:
        raise GeometricError(f"epochs must be >= 1, got {epochs}")
    if not 0.0 <= sparsity <= 1.0:
        raise GeometricError(f"sparsity must be in [0, 1], got {sparsity}")
    rng = np.random.default_rng(seed)
    std = np.sqrt(2.0 / num_nodes)
    scores: dict[int, np.ndarray] = {}
    hard: dict[int, np.ndarray] = {}
    for klass in range(model.n_classes):
        free = rng.normal(0.0, std, size=n_non_loop)
        opt = _Adam([free], lr=lr)
        for _ in range(int(epochs)):
            gate = _sigmoid(free)
            full_mask = np.concatenate([gate, np.ones(n_loops)])
            logits, cache = model._forward_cached(
                features, edges_in, None, full_mask
            )
            model._zero_grads()
            row = logits[0]
            d_logits = _softmax(row).reshape(1, -1)
            d_logits[0, klass] -= 1.0
            model._backward(cache, d_logits)
            grad_mask = (
                model.conv1.grads["edge_mask"] + model.conv2.grads["edge_mask"]
            )
            grad_free = grad_mask[:n_non_loop] * gate * (1.0 - gate)
            grad_free += coff_edge_size * gate * (1.0 - gate)
            if n_non_loop:
                grad_free += (
                    coff_edge_ent
                    / n_non_loop
                    * np.log((1.0 - gate + 1e-15) / (gate + 1e-15))
                    * gate
                    * (1.0 - gate)
                )
            opt.step([grad_free])
        gate = _sigmoid(free)
        scores[klass] = np.concatenate([gate, np.ones(n_loops)])
        hard[klass] = np.concatenate(
            [control_sparsity(gate, sparsity), np.ones(n_loops)]
        )
    p_origin = float(model.predict_proba(features, edges_in)[0, y])
    hard_y = hard[y]
    p_masked = float(
        model.predict_proba(features, edges_in, edge_mask=hard_y)[0, y]
    )
    maskout = np.concatenate([1.0 - hard_y[:n_non_loop], np.ones(n_loops)])
    p_maskout = float(
        model.predict_proba(features, edges_in, edge_mask=maskout)[0, y]
    )
    zero_mask = np.concatenate([np.zeros(n_non_loop), np.ones(n_loops)])
    p_zero = float(
        model.predict_proba(features, edges_in, edge_mask=zero_mask)[0, y]
    )
    kept = float(hard_y[:n_non_loop].sum())
    return GnnExplanation(
        label=y,
        edge_index=canonical,
        edge_scores=scores,
        hard_mask=hard,
        probs={
            "origin": p_origin,
            "masked": p_masked,
            "maskout": p_maskout,
            "zero": p_zero,
        },
        fidelity=p_origin - p_maskout,
        fidelity_inv=p_origin - p_masked,
        sparsity=1.0 - kept / n_non_loop if n_non_loop else 0.0,
    )
