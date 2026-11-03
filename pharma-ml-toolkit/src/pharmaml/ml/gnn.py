"""Graph neural networks — ``pyg-team/pytorch_geometric``'s message passing,
absorbed natively on the toolkit's own reverse-mode tape.

``torch_geometric`` (reviewed file by file — ``scratch/gh187_review.md``) is,
above all, a *message-passing framework*: graphs ride as ``Data`` objects
(``x`` node features, ``edge_index`` the 2×E endpoint list), layers update
every node from its neighbours through a scatter-reduce, and pooling folds
nodes into graph vectors.  The CUDA kernels, the C++ ``torch_scatter``
extension, the dataloaders and the hetero/3D towers are exactly what this
toolkit does not vendor — the *mathematics* transfers natively on
:mod:`pharmaml.ml.autograd`'s tape (whose ``scatter_add`` / ``scatter_max`` /
``scatter_softmax`` nodes are this round's ``utils/scatter`` absorption):

* :class:`Data` / :class:`Batch` — the graph container and the collate
  (node-offset edge remap, the ``batch`` membership vector, ``ptr``),
  ``to_undirected``, ``add_self_loops``, :func:`degree`.
* :class:`GCNConv` (Kipf & Welling) — ``D^-1/2 (A+I) D^-1/2 X W`` with the
  symmetric normalisation and optional per-edge weights.
* :class:`SAGEConv` (Hamilton et al.) — ``W_l . mean(neighbours) + W_r . x``.
* :class:`GINConv` (Xu et al.) — ``(1 + eps) x + sum(neighbours)`` through a
  2-layer MLP, the WL-test-maximal aggregator.
* :class:`GATConv` (Veličković et al.) — multi-head scaled attention over
  edge scores (LeakyReLU(1/2), per-destination softmax, concat heads).
* :class:`EdgeConv` (Wang et al.) — ``max_j ReLU(Θ x_i + Φ (x_j − x_i))``.
* Pooling — :func:`global_add_pool` / ``mean`` / ``max`` and the
  differentiable-select :class:`TopKPooling` / :class:`SAGPool` (score,
  select top-k per graph, multiply the kept rows by their scores — pyg's
  exact gradient path, selection treated as constant).
* :class:`GraphClassifier` — the sklearn-style estimator the Deep Learning
  tab trains: conv stack → pool → linear head, Adam, per-epoch loss history
  (the Run board reads it).

The honest scope, stated once: CPU float64 eager execution at teaching
scale, node-level (dim=0) scatter only, no dataloaders/samplers/hetero
graphs, no C++ acceleration — synthetic graphs come from
:func:`synthetic_motif_graphs` and molecules from
:mod:`pharmaml.chem.featurize`.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.gnn import Data, degree, add_self_loops
>>> g = Data(x=np.ones((3, 1)), edge_index=np.array([[0, 1], [1, 2]]))
>>> degree(g.edge_index, 3).astype(int).tolist()
[0, 1, 1]
>>> looped = add_self_loops(g.edge_index, 3)
>>> looped.shape
(2, 5)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from pharmaml.exceptions import DataValidationError, PharmaMLError

from .autograd import (
    Adam,
    Linear,
    Module,
    Parameter,
    ReLU,
    Sequential,
    Tensor,
    cross_entropy,
    no_grad,
    scatter_add,
    scatter_max,
    scatter_softmax,
)

__all__ = [
    "GnnError",
    "Data",
    "Batch",
    "degree",
    "add_self_loops",
    "to_undirected",
    "GCNConv",
    "SAGEConv",
    "GINConv",
    "GATConv",
    "EdgeConv",
    "global_add_pool",
    "global_mean_pool",
    "global_max_pool",
    "TopKPooling",
    "SAGPooling",
    "GraphClassifier",
    "synthetic_motif_graphs",
    "CONV_NAMES",
    "POOL_NAMES",
]


class GnnError(PharmaMLError):
    """A graph-shape or message-passing contract was violated."""


# --------------------------------------------------------------------------
# the pyg data layer
# --------------------------------------------------------------------------


@dataclass
class Data:
    """One graph: node features, the 2×E endpoint list, optional extras.

    ``edge_index[0]`` is the SOURCE endpoint and ``edge_index[1]`` the
    DESTINATION of every directed edge — messages flow source →
    destination (pyg's ``flow="source_to_target"`` default).
    """

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray | None = None
    edge_weight: np.ndarray | None = None
    y: np.ndarray | None = None
    #: ground-truth edge importance (explainability fixtures); the dataclass
    #: carries it alongside edge_attr so the explainers can score against it
    edge_label: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        self.edge_index = np.asarray(self.edge_index, dtype=int).reshape(2, -1)
        if self.edge_index.size:
            n = self.x.shape[0]
            if self.edge_index.min() < 0 or self.edge_index.max() >= n:
                raise DataValidationError("Data: edge_index names a node outside 0..n-1")
        if self.edge_attr is not None:
            self.edge_attr = np.asarray(self.edge_attr, dtype=float)
        if self.edge_weight is not None:
            self.edge_weight = np.asarray(self.edge_weight, dtype=float).reshape(-1)

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_undirected(self) -> "Data":
        """Both directions of every edge, duplicates collapsed.

        >>> import numpy as np
        >>> from pharmaml.ml.gnn import Data
        >>> g = Data(np.ones((2, 1)), np.array([[0], [1]]),
        ...          edge_label=np.array([1.0]))
        >>> g.to_undirected().edge_index.tolist()
        [[0, 1], [1, 0]]
        >>> g.to_undirected().edge_label.tolist()
        [1.0, 1.0]
        """
        both = to_undirected(self.edge_index)
        lookup: dict[tuple[int, int], int] = {}
        for i, (a, b) in enumerate(self.edge_index.T):
            lookup.setdefault((int(a), int(b)), i)
            lookup.setdefault((int(b), int(a)), i)

        def pick(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            rows = [arr[lookup[(int(a), int(b))]] for a, b in both.T]
            stacked = np.stack(rows) if np.asarray(rows[0]).ndim else np.asarray(rows)
            return stacked

        return Data(
            self.x,
            both,
            edge_attr=pick(self.edge_attr),
            edge_weight=pick(self.edge_weight) if self.edge_weight is not None else None,
            y=self.y,
            edge_label=pick(self.edge_label),
        )

    def with_self_loops(self) -> "Data":
        """A copy whose edges include every self-loop (message passing's
        non-empty-neighbourhood guarantee)."""
        looped = add_self_loops(self.edge_index, self.num_nodes)
        weight = None
        if self.edge_weight is not None:
            weight = np.concatenate([self.edge_weight, np.ones(self.num_nodes)])
        return Data(
            self.x, looped, edge_attr=self.edge_attr, edge_weight=weight, y=self.y
        )


def to_undirected(edge_index: np.ndarray) -> np.ndarray:
    """The edge list with both directions appended (deduplicated lazily by
    the caller's aggregation — GCN's symmetric normalisation is insensitive
    to exact duplicates only when weights split, so dedupe here)."""

    index = np.asarray(edge_index, dtype=int).reshape(2, -1)
    both = np.concatenate([index, index[::-1]], axis=1)
    seen: set[tuple[int, int]] = set()
    cols: list[int] = []
    for col in range(both.shape[1]):
        key = (int(both[0, col]), int(both[1, col]))
        if key not in seen:
            seen.add(key)
            cols.append(col)
    return both[:, cols]


def add_self_loops(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """``(i, i)`` for every node, appended to the edge list."""

    index = np.asarray(edge_index, dtype=int).reshape(2, -1)
    loops = np.arange(num_nodes, dtype=int)
    return np.concatenate([index, np.stack([loops, loops])], axis=1)


def degree(
    edge_index: np.ndarray, num_nodes: int, *, weights: np.ndarray | None = None
) -> np.ndarray:
    """Destination in-degree (optionally edge-weighted)."""

    index = np.asarray(edge_index, dtype=int).reshape(2, -1)
    ones = (
        np.ones(index.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    )
    out = np.zeros(num_nodes)
    np.add.at(out, index[1], ones)
    return out


@dataclass
class Batch:
    """Collated graphs: one big node matrix + offset edges + membership.

    ``batch[i]`` names the graph node ``i`` belongs to (pooling reads it);
    ``ptr`` holds the per-graph row offsets.
    """

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray | None
    edge_weight: np.ndarray | None
    batch: np.ndarray
    ptr: np.ndarray
    y: np.ndarray | None
    edge_labels: list[np.ndarray | None]

    @classmethod
    def from_data_list(cls, graphs: Sequence[Data]) -> "Batch":
        if not graphs:
            raise GnnError("Batch: no graphs to collate")
        xs: list[np.ndarray] = []
        ei: list[np.ndarray] = []
        ea: list[np.ndarray] | None = [] if all(g.edge_attr is not None for g in graphs) else None
        ew: list[np.ndarray] | None = [] if all(g.edge_weight is not None for g in graphs) else None
        batch = np.zeros(sum(g.num_nodes for g in graphs), dtype=int)
        ys: list[np.ndarray] = []
        labels: list[np.ndarray | None] = []
        offset = 0
        for i, g in enumerate(graphs):
            xs.append(g.x)
            ei.append(g.edge_index + offset)
            if ea is not None and g.edge_attr is not None:
                ea.append(g.edge_attr)
            if ew is not None and g.edge_weight is not None:
                ew.append(g.edge_weight)
            batch[offset : offset + g.num_nodes] = i
            if g.y is not None:
                ys.append(np.asarray(g.y, dtype=float).reshape(-1))
            labels.append(g.edge_label)
            offset += g.num_nodes
        ptr = np.zeros(len(graphs) + 1, dtype=int)
        ptr[1:] = np.cumsum([g.num_nodes for g in graphs])
        return cls(
            x=np.concatenate(xs),
            edge_index=np.concatenate(ei, axis=1),
            edge_attr=np.concatenate(ea) if ea else None,
            edge_weight=np.concatenate(ew) if ew else None,
            batch=batch,
            ptr=ptr,
            y=np.concatenate(ys) if ys else None,
            edge_labels=labels,
        )

    @property
    def num_graphs(self) -> int:
        return int(self.ptr.size - 1)


# --------------------------------------------------------------------------
# the convolutions (every one honours an optional per-edge weight, the
# explainer hook: a mask tensor multiplied into the messages)
# --------------------------------------------------------------------------


def _validate_index(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    index = np.asarray(edge_index, dtype=int).reshape(2, -1)
    if index.size and (index.min() < 0 or index.max() >= num_nodes):
        raise GnnError("conv: edge_index names a node outside the graph")
    return index


class GCNConv(Module):
    """``X' = D^-1/2 (A + I) D^-1/2 X W`` (self-loops included).

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.gnn import GCNConv
    >>> conv = GCNConv(2, 3, seed=7)
    >>> x = np.arange(6, dtype=float).reshape(3, 2)
    >>> ei = np.array([[0, 1, 1, 2], [1, 2, 0, 0]])
    >>> conv(Tensor(x), ei).shape
    (3, 3)
    """

    def __init__(self, in_channels: int, out_channels: int, *, seed: int | None = None) -> None:
        self.lin = Linear(in_channels, out_channels, seed=seed)

    def forward(
        self, x: Tensor, edge_index: object, edge_weight: Tensor | None = None
    ) -> Tensor:
        index = _validate_index(np.asarray(edge_index), x.data.shape[0])
        looped = add_self_loops(index, x.data.shape[0])
        if edge_weight is None:
            weights = np.ones(looped.shape[1])
        else:
            base = np.asarray(edge_weight.numpy() if hasattr(edge_weight, "numpy") else edge_weight).reshape(-1)
            weights = np.concatenate([base, np.ones(x.data.shape[0])])
        deg = degree(looped, x.data.shape[0], weights=weights)
        norm_src = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
        scale = norm_src[looped[0]] * norm_src[looped[1]] * weights
        messages = self.lin.forward(x)[looped[0]] * Tensor(scale.reshape(-1, 1))
        return scatter_add(messages, looped[1], x.data.shape[0])


class SAGEConv(Module):
    """``x' = W_l . mean(neighbours) + W_r . x`` (GraphSAGE's mean)."""

    def __init__(self, in_channels: int, out_channels: int, *, seed: int | None = None) -> None:
        self.lin_l = Linear(in_channels, out_channels, seed=seed)
        self.lin_r = Linear(in_channels, out_channels, seed=None if seed is None else seed + 1)

    def forward(
        self, x: Tensor, edge_index: object, edge_weight: Tensor | None = None
    ) -> Tensor:
        index = _validate_index(np.asarray(edge_index), x.data.shape[0])
        ones = Tensor(np.ones((index.shape[1], 1)))
        src_weight = edge_weight if edge_weight is not None else ones
        neighbour_sum = scatter_add(x[index[0]] * src_weight, index[1], x.data.shape[0])
        counts = scatter_add(src_weight, index[1], x.data.shape[0])
        safe = counts + Tensor((counts.numpy() == 0).astype(float))
        mean = neighbour_sum / safe
        return self.lin_l.forward(mean) + self.lin_r.forward(x)


class GINConv(Module):
    """``x' = MLP((1 + eps) x + sum(neighbours))`` (GIN's sum aggregator)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        eps: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.eps = float(eps)
        self.mlp = Sequential(
            Linear(in_channels, out_channels, seed=seed),
            ReLU(),
            Linear(out_channels, out_channels, seed=None if seed is None else seed + 1),
        )

    def forward(
        self, x: Tensor, edge_index: object, edge_weight: Tensor | None = None
    ) -> Tensor:
        index = _validate_index(np.asarray(edge_index), x.data.shape[0])
        ones = Tensor(np.ones((index.shape[1], 1)))
        src_weight = edge_weight if edge_weight is not None else ones
        aggregated = scatter_add(x[index[0]] * src_weight, index[1], x.data.shape[0])
        return self.mlp.forward((1.0 + self.eps) * x + aggregated)


class GATConv(Module):
    """Multi-head graph attention (per-destination softmax over edges).

    ``heads`` outputs are concatenated (pyg's ``concat=True`` default), so
    the output width is ``heads * out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        heads: int = 1,
        negative_slope: float = 0.2,
        seed: int | None = None,
    ) -> None:
        if heads < 1:
            raise DataValidationError("GATConv: heads must be >= 1")
        self.heads = int(heads)
        self.slope = float(negative_slope)
        self.lin = Linear(in_channels, heads * out_channels, seed=seed)
        #: the last forward's per-edge attention (return_attention_weights)
        self.last_alpha: np.ndarray | None = None
        bound = 1.0 / np.sqrt(in_channels)
        rng = np.random.default_rng(seed)
        # a_l, a_r: one score vector per head over the projected features
        width = heads * out_channels
        self.a_l = Parameter(rng.uniform(-bound, bound, size=(width,)))
        self.a_r = Parameter(rng.uniform(-bound, bound, size=(width,)))

    def forward(
        self, x: Tensor, edge_index: object, edge_weight: Tensor | None = None
    ) -> Tensor:
        index = _validate_index(np.asarray(edge_index), x.data.shape[0])
        src, dst = index[0], index[1]
        n = x.data.shape[0]
        projected = self.lin.forward(x)  # (N, heads*out)
        score = (projected[dst] * self.a_l).sum(axis=1) + (projected[src] * self.a_r).sum(axis=1)
        alpha = scatter_softmax(score.leaky_relu(self.slope).reshape(-1, 1), dst, n).reshape(-1)
        if edge_weight is not None:
            alpha = alpha * edge_weight.reshape(-1)
        # pyg's return_attention_weights hook: the detached per-edge
        # coefficients, aligned with the edge columns (the AttentionExplainer
        # and the GUI's attention overlay read this)
        self.last_alpha = alpha.numpy().copy()
        # attention gathers from SOURCES into DESTINATIONS: the pyg flow is
        # source_to_target, so aggregate over src rows keyed by dst
        return scatter_add(projected[src] * alpha.reshape(-1, 1), dst, n)


class EdgeConv(Module):
    """``x'_i = max_j ReLU(Θ x_i + Φ (x_j − x_i))`` over directed edges."""

    def __init__(self, in_channels: int, out_channels: int, *, seed: int | None = None) -> None:
        self.lin_a = Linear(in_channels, out_channels, seed=seed)
        self.lin_b = Linear(in_channels, out_channels, seed=None if seed is None else seed + 1)

    def forward(
        self, x: Tensor, edge_index: object, edge_weight: Tensor | None = None
    ) -> Tensor:
        index = _validate_index(np.asarray(edge_index), x.data.shape[0])
        src, dst = index[0], index[1]
        n = x.data.shape[0]
        message = self.lin_a.forward(x)[dst] + self.lin_b.forward(x[dst] - x[src])
        message = message.relu()
        if edge_weight is not None:
            message = message * edge_weight.reshape(-1, 1)
        return scatter_max(message, dst, n)


CONV_NAMES: tuple[str, ...] = ("gcn", "sage", "gin", "gat", "edge")


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------


def _pool_scatter(x: Tensor, batch: np.ndarray, size: int, reduce: str) -> Tensor:
    if reduce == "sum":
        return scatter_add(x, batch, size)
    if reduce == "mean":
        total = scatter_add(x, batch, size)
        counts = np.bincount(batch, minlength=size).astype(float)
        safe = counts + (counts == 0).astype(float)
        return total / Tensor(safe.reshape(-1, 1))
    if reduce == "max":
        return scatter_max(x, batch, size)
    raise DataValidationError(f"pool: unknown reduce {reduce!r}")


def global_add_pool(x: Tensor, batch: np.ndarray, size: int) -> Tensor:
    """Sum every graph's node rows into one graph vector."""

    return _pool_scatter(x, np.asarray(batch, dtype=int), size, "sum")


def global_mean_pool(x: Tensor, batch: np.ndarray, size: int) -> Tensor:
    return _pool_scatter(x, np.asarray(batch, dtype=int), size, "mean")


def global_max_pool(x: Tensor, batch: np.ndarray, size: int) -> Tensor:
    return _pool_scatter(x, np.asarray(batch, dtype=int), size, "max")


POOL_NAMES: tuple[str, ...] = ("mean", "sum", "max", "topk", "sag")


def _topk_per_graph(score: np.ndarray, ptr: np.ndarray, ratio: float) -> np.ndarray:
    """Row indices of the top ``ceil(ratio * n_g)`` scores per graph."""

    keep: list[np.ndarray] = []
    for g in range(ptr.size - 1):
        rows = np.arange(ptr[g], ptr[g + 1])
        k = max(1, int(np.ceil(ratio * rows.size)))
        order = rows[np.argsort(-score[rows], kind="stable")]
        keep.append(np.sort(order[:k]))
    return np.concatenate(keep)


class TopKPooling(Module):
    """g-Ratio select pooling (pyg ``TopKPooling``): keep the top-k scored
    rows per graph and rescale them by ``sigmoid(score)`` — the multiplier
    carries the gradient, the selection is a constant."""

    def __init__(self, in_channels: int, *, ratio: float = 0.5, seed: int | None = None) -> None:
        if not 0.0 < ratio <= 1.0:
            raise DataValidationError("TopKPooling: ratio must be in (0, 1]")
        self.ratio = float(ratio)
        self.score = Linear(in_channels, 1, seed=seed)

    def forward(
        self, x: Tensor, batch: np.ndarray, ptr: np.ndarray
    ) -> tuple[Tensor, np.ndarray, np.ndarray]:
        s = self.score.forward(x).reshape(-1)
        keep = _topk_per_graph(s, ptr, self.ratio)
        gate = s[keep].sigmoid()
        pooled_x = x[keep] * gate.reshape(-1, 1)
        new_batch = batch[keep]
        remap = np.full(x.data.shape[0], -1, dtype=int)
        remap[keep] = np.arange(keep.size)
        return pooled_x, new_batch, remap


class SAGPooling(Module):
    """SAGPooling: TopK select scored by a learned linear attention, the
    kept rows rescaled by their raw scores (pyg's ``GAT``-free variant)."""

    def __init__(self, in_channels: int, *, ratio: float = 0.5, seed: int | None = None) -> None:
        if not 0.0 < ratio <= 1.0:
            raise DataValidationError("SAGPooling: ratio must be in (0, 1]")
        self.ratio = float(ratio)
        self.score = Linear(in_channels, 1, seed=seed)

    def forward(
        self, x: Tensor, batch: np.ndarray, ptr: np.ndarray
    ) -> tuple[Tensor, np.ndarray, np.ndarray]:
        s = self.score.forward(x).reshape(-1)
        keep = _topk_per_graph(s, ptr, self.ratio)
        pooled_x = x[keep] * s[keep].reshape(-1, 1)
        new_batch = batch[keep]
        remap = np.full(x.data.shape[0], -1, dtype=int)
        remap[keep] = np.arange(keep.size)
        return pooled_x, new_batch, remap


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------


class GraphClassifier(ClassifierMixin, BaseEstimator):
    """Conv stack → pool → linear head, trained on the tape.

    Parameters mirror the Deep Learning tab's other estimators; ``conv``
    picks the message-passing family, ``pool`` the graph readout.  The
    per-epoch loss lands in ``loss_history_`` (the Run board reads it)
    and ``history_`` keeps the last forward's internals the explainers
    reuse (the final conv's node embeddings, per-graph logits).

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.ml.gnn import GraphClassifier, synthetic_motif_graphs
    >>> graphs, y = synthetic_motif_graphs(18, seed=11)
    >>> model = GraphClassifier(hidden=8, layers=2, conv="gcn", pool="mean",
    ...                         epochs=120, lr=0.08, seed=11)
    >>> model.fit(graphs, y)
    GraphClassifier(...)
    >>> float(model.score(graphs, y)) >= 0.9
    True
    """

    def __init__(
        self,
        hidden: int = 16,
        layers: int = 2,
        conv: str = "gcn",
        pool: str = "mean",
        lr: float = 0.05,
        epochs: int = 200,
        weight_decay: float = 0.0,
        heads: int = 1,
        ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.hidden = hidden
        self.layers = layers
        self.conv = conv
        self.pool = pool
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.heads = heads
        self.ratio = ratio
        self.seed = seed

    # -- sklearn protocol ---------------------------------------------------

    def fit(self, graphs: Sequence[Data], y: Iterable[int]) -> "GraphClassifier":
        labels = np.asarray(list(y), dtype=int).reshape(-1)
        if len(graphs) != labels.size:
            raise GnnError("fit: one label per graph required")
        if not graphs:
            raise GnnError("fit: no graphs")
        self.classes_ = np.unique(labels)
        if self.classes_.size < 2:
            raise GnnError("fit: needs >= 2 classes")
        self.n_features_in_ = int(graphs[0].x.shape[1])
        self.conv_ = self.conv if self.conv in CONV_NAMES else None
        if self.conv_ is None:
            raise GnnError(f"fit: conv {self.conv!r} not in {CONV_NAMES}")
        if self.pool not in POOL_NAMES:
            raise GnnError(f"fit: pool {self.pool!r} not in {POOL_NAMES}")

        self._build_layers(self.n_features_in_)
        batch = Batch.from_data_list(list(graphs))
        self._train(batch, labels)
        return self

    def _build_layers(self, in_dim: int | None = None) -> None:
        seed = self.seed
        self.convs: list[Module] = []
        dims = in_dim if in_dim is not None else self.n_features_in_
        out_width = 0
        for i in range(max(1, int(self.layers))):
            out = int(self.hidden)
            if self.conv_ == "gcn":
                layer: Module = GCNConv(dims, out, seed=None if seed is None else seed + 10 * i)
                out_width = out
            elif self.conv_ == "sage":
                layer = SAGEConv(dims, out, seed=None if seed is None else seed + 10 * i)
                out_width = out
            elif self.conv_ == "gin":
                layer = GINConv(dims, out, seed=None if seed is None else seed + 10 * i)
                out_width = out
            elif self.conv_ == "gat":
                layer = GATConv(dims, out, heads=max(1, int(self.heads)), seed=None if seed is None else seed + 10 * i)
                out_width = out * max(1, int(self.heads))
            else:
                layer = EdgeConv(dims, out, seed=None if seed is None else seed + 10 * i)
                out_width = out
            self.convs.append(layer)
            dims = out_width
        self.pool_module: Module | None = None
        pooled_dim = out_width
        if self.pool in ("topk", "sag"):
            cls = TopKPooling if self.pool == "topk" else SAGPooling
            self.pool_module = cls(out_width, ratio=self.ratio, seed=None if seed is None else seed + 99)
        self.head = Linear(pooled_dim, self.classes_.size, seed=None if seed is None else seed + 7)

    def _forward(self, batch: Batch, edge_weight: Tensor | None = None) -> tuple[Tensor, list[Tensor]]:
        x = Tensor(batch.x)
        embeddings = [x]
        for conv in self.convs:
            x = conv.forward(x, batch.edge_index, edge_weight)
            x = x.relu()
            embeddings.append(x)
        if self.pool_module is not None:
            x, new_batch, _remap = self.pool_module.forward(x, batch.batch, batch.ptr)
            readout = _pool_scatter(x, new_batch, batch.num_graphs, "mean")
        else:
            readout = _pool_scatter(x, batch.batch, batch.num_graphs, self.pool)
        return self.head.forward(readout), embeddings

    def _train(self, batch: Batch, labels: np.ndarray) -> None:
        params: list[Parameter] = []
        modules: list[Module] = list(self.convs)
        if self.pool_module is not None:
            modules.append(self.pool_module)
        modules.append(self.head)
        for module in modules:
            params.extend(module.parameters())
        self._optimizer = Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        encoded = np.searchsorted(self.classes_, labels)
        self.loss_history_: list[float] = []
        for _epoch in range(max(1, int(self.epochs))):
            self._optimizer.zero_grad()
            logits, _embeddings = self._forward(batch)
            loss = cross_entropy(logits, encoded)
            loss.backward()
            self._optimizer.step()
            self.loss_history_.append(float(loss.item()))
        self.history_: dict[str, object] = {}
        with no_grad():
            logits, embeddings = self._forward(batch)
        self.history_["node_embeddings"] = embeddings[-1].numpy()
        self.history_["logits"] = logits.numpy()
        self.history_["batch"] = batch.batch.copy()
        self.history_["edge_index"] = batch.edge_index.copy()

    def decision_function(self, graphs: Sequence[Data]) -> np.ndarray:
        logits, _ = self._decide(Batch.from_data_list(list(graphs)))
        return logits

    def _decide(self, batch: Batch) -> tuple[np.ndarray, list[Tensor]]:
        with no_grad():
            logits, embeddings = self._forward(batch)
        return logits.numpy(), embeddings

    def predict_proba(self, graphs: Sequence[Data]) -> np.ndarray:
        logits, _ = self._decide(Batch.from_data_list(list(graphs)))
        shifted = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, graphs: Sequence[Data]) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(graphs), axis=1)]


# --------------------------------------------------------------------------
# synthetic graphs (the tests' and the GUI's fixture: one motif per class,
# with the motif edges carried as edge_label for the explainers)
# --------------------------------------------------------------------------


def synthetic_motif_graphs(
    n_graphs: int,
    *,
    seed: int | None = None,
    n_classes: int = 3,
    n_features: int = 2,
) -> tuple[list[Data], np.ndarray]:
    """One motif per class (ring / star / path) over a random scatter of
    noise nodes; every graph carries ``edge_label`` (1 motif / 0 noise)
    so the explainers can score against ground truth.

    Examples
    --------
    >>> from pharmaml.ml.gnn import synthetic_motif_graphs
    >>> graphs, y = synthetic_motif_graphs(6, seed=3)
    >>> len(graphs), sorted(set(y.tolist()))
    (6, [0, 1, 2])
    >>> all(g.edge_label is not None for g in graphs)
    True
    """

    if n_graphs < 1 or n_classes < 2:
        raise DataValidationError("synthetic_motif_graphs: need >= 1 graph, >= 2 classes")
    rng = np.random.default_rng(seed)
    graphs: list[Data] = []
    ys = np.zeros(n_graphs, dtype=int)
    for i in range(n_graphs):
        cls = int(i % n_classes)
        ys[i] = cls
        noise_n = int(rng.integers(2, 5))
        motif: list[tuple[int, int]] = []
        if cls == 0:  # ring
            k = int(rng.integers(4, 6))
            motif = [(j, (j + 1) % k) for j in range(k)]
        elif cls == 1:  # star
            k = int(rng.integers(4, 6))
            motif = [(0, j) for j in range(1, k)]
        else:  # path
            k = int(rng.integers(4, 6))
            motif = [(j, j + 1) for j in range(k - 1)]
        total = k + noise_n
        x = rng.normal(size=(total, n_features))
        x[:k] += 1.5  # motif nodes are slightly hotter, like DIG's syn datasets
        motif_index = np.array([list(e) for e in motif], dtype=int).T
        noise_pairs: list[tuple[int, int]] = []
        for _ in range(int(rng.integers(1, 3))):
            a, b = int(rng.integers(k, total)), int(rng.integers(k, total))
            if a != b:
                noise_pairs.append((a, b))
        noise_index = np.array(noise_pairs, dtype=int).T if noise_pairs else np.zeros((2, 0), dtype=int)
        edge_index = np.concatenate([motif_index, noise_index], axis=1)
        edge_label = np.concatenate(
            [np.ones(motif_index.shape[1]), np.zeros(noise_index.shape[1])]
        )
        undirected = Data(x, edge_index, y=np.array(cls), edge_label=edge_label)
        graphs.append(undirected.to_undirected())
    return graphs, ys
