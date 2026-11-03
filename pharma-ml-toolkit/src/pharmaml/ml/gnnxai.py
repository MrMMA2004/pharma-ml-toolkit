"""Graph explainability — DIG's ``xgraph`` and PyG's ``explain`` lineages,
absorbed offline onto the toolkit's GNN.

Both repos (reviewed file by file — ``scratch/gh187_review.md``) answer the
same question — *which edges made the model say that?* — with overlapping
inventories: DIG's ``dig/xgraph/method`` ships GNNExplainer / PGExplainer /
SubgraphX / GradCam / GNN-GI / GNN-LRP, pyg's ``torch_geometric/explain``
ships GNNExplainer / PGExplainer / AttentionExplainer / fidelity metrics.
The two GNNExplainer twins are merged here into one engine credited to both
lineages; the GradCam/GI/LRP variants need conv-grid hooks the toolkit's
dense graphs do not have (their edge-relevance spirit is what the mask
scorers below implement).

What is here, one by one:

* :class:`GNNExplainer` (Ying et al. 2019; DIG ``gnnexplainer.py`` +
  pyg ``gnn_explainer.py``) — a per-edge mask learned by gradient descent
  against the model's masked prediction, regularised by the size and
  entropy terms both repos ship.
* :class:`PGExplainer` (Luo et al. 2020; DIG ``pgexplainer.py`` + pyg
  ``pg_explainer.py``) — a parametric edge scorer over the frozen model's
  node embeddings, trained once across the dataset.
* :class:`AttentionExplainer` (pyg ``attention_explainer.py``) — the GAT
  coefficients themselves, the zero-training baseline.
* :class:`SubgraphX` (Yuan et al. 2021; DIG ``subgraphx.py``) —
  Shapley-guided connected-subgraph search at teaching scale.
* the evaluation pair (DIG ``xgraph/evaluation/metrics.py`` + pyg
  ``explain/metric``) — :func:`fidelity_plus` / :func:`fidelity_minus` /
  :func:`sparsity` / :func:`accuracy_at_k`.

The honest scope, stated once: every explainer assumes the
:class:`~pharmaml.ml.gnn.GraphClassifier` contract (a ``_forward`` that
honours per-edge weights); FlowX and the full Shapley explainer are
recorded as STRETCH in the review tables.

Examples
--------
>>> from pharmaml.ml.gnn import GraphClassifier, synthetic_motif_graphs
>>> from pharmaml.ml.gnnxai import GNNExplainer, accuracy_at_k
>>> graphs, y = synthetic_motif_graphs(24, seed=5)
>>> model = GraphClassifier(hidden=12, layers=2, conv="gcn", pool="mean",
...                         epochs=150, lr=0.08, seed=5).fit(graphs, y)
>>> explainer = GNNExplainer(iters=120, seed=5)
>>> report = explainer.explain(model, graphs[0])
>>> report.method
'gnnexplainer'
>>> float(report.scores.mean()) >= 0.0
True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .autograd import Adam, Linear, Module, Parameter, ReLU, Sequential, Tensor, no_grad
from .gnn import Batch, Data, GATConv, GraphClassifier, GnnError

__all__ = [
    "EdgeExplanation",
    "GNNExplainer",
    "PGExplainer",
    "AttentionExplainer",
    "SubgraphX",
    "fidelity_plus",
    "fidelity_minus",
    "sparsity",
    "accuracy_at_k",
    "EXPLAINER_NAMES",
]


EXPLAINER_NAMES: tuple[str, ...] = ("gnnexplainer", "pgexplainer", "attention", "subgraphx")


@dataclass
class EdgeExplanation:
    """One graph's edge importance: scores in [0, 1] aligned with the
    graph's ``edge_index`` columns, the top-k selection and the explainer's
    own metadata (the GUI's caption and the saved CSV read these)."""

    scores: np.ndarray
    top_edges: np.ndarray
    method: str
    target_class: int
    extra: dict[str, object] = field(default_factory=dict)


def _prepare(model: GraphClassifier, graph: Data) -> tuple[Batch, int]:
    """Freeze the model's parameters and collate the single graph."""

    batch = Batch.from_data_list([graph])
    if graph.y is not None:
        target = int(np.asarray(graph.y).reshape(-1)[0])
    else:
        target = int(np.argmax(model.predict_proba([graph])[0]))
    return batch, target


def _masked_probability(
    model: GraphClassifier, batch: Batch, target: int, weights: np.ndarray | Tensor
) -> Tensor:
    """log-probability of ``target`` under the edge-weighted forward —
    the stable log-softmax path, differentiable through the weights."""

    logits, _ = model._forward(batch, edge_weight=weights)
    shifted = logits - Tensor(np.max(logits.numpy(), axis=1, keepdims=True))
    log_p = shifted - shifted.exp().sum(axis=1, keepdims=True).log()
    return log_p[:, target]


def _freeze(model: GraphClassifier, state: bool) -> list[tuple[Parameter, bool]]:
    touched: list[tuple[Parameter, bool]] = []
    modules: list[Module] = list(model.convs)
    if model.pool_module is not None:
        modules.append(model.pool_module)
    modules.append(model.head)
    for module in modules:
        for param in module.parameters():
            touched.append((param, param.requires_grad))
            param.requires_grad = state
    return touched


def _restore(touched: list[tuple[Parameter, bool]]) -> None:
    for param, was in touched:
        param.requires_grad = was
        param.grad = None


class GNNExplainer:
    """The per-edge mask, learned against the frozen model.

    The loss is the masked cross-entropy plus pyg's two regularisers —
    the size term (mean mask mass) and the binary entropy term — with
    the mask parameterised as logits through a sigmoid (DIG's
    ``symmetric_edge_mask`` utility plays the same game).
    """

    def __init__(
        self,
        *,
        iters: int = 200,
        lr: float = 0.1,
        size_reg: float = 1e-3,
        entropy_reg: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        self.iters = int(iters)
        self.lr = float(lr)
        self.size_reg = float(size_reg)
        self.entropy_reg = float(entropy_reg)
        self.seed = seed

    def explain(self, model: GraphClassifier, graph: Data, *, top_k: int = 8) -> EdgeExplanation:
        batch, target = _prepare(model, graph)
        n_edges = batch.edge_index.shape[1]
        rng = np.random.default_rng(self.seed)
        mask = Parameter(rng.normal(scale=0.01, size=(n_edges, 1)))
        frozen = _freeze(model, False)
        last_loss = float("nan")
        try:
            optimizer = Adam([mask], lr=self.lr)
            eps = 1e-6
            for _ in range(max(1, self.iters)):
                optimizer.zero_grad()
                p = mask.sigmoid()
                prob = _masked_probability(model, batch, target, p)
                size_term = p.mean()
                entropy = -(p * (p + eps).log() + (1.0 - p) * (1.0 - p + eps).log()).mean()
                loss = -prob + self.size_reg * size_term + self.entropy_reg * entropy
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
        finally:
            _restore(frozen)
        with no_grad():
            scores = mask.sigmoid().reshape(-1).numpy().copy()
            scores = np.clip(scores, 0.0, 1.0)
        top = np.argsort(-scores, kind="stable")[: max(1, min(top_k, n_edges))]
        return EdgeExplanation(
            scores=scores,
            top_edges=np.sort(top),
            method="gnnexplainer",
            target_class=target,
            extra={"iters": self.iters, "final_loss": last_loss},
        )


class PGExplainer:
    """The parametric edge scorer over the frozen model's embeddings.

    Trained once across the dataset (each graph against its own label),
    then explains any graph in one forward — DIG trains per class and pyg
    per task; the merged engine trains one scorer over the label set and
    records the per-graph target at explanation time.
    """

    def __init__(
        self,
        *,
        iters: int = 150,
        lr: float = 0.02,
        size_reg: float = 1e-4,
        hidden: int = 32,
        seed: int | None = None,
    ) -> None:
        self.iters = int(iters)
        self.lr = float(lr)
        self.size_reg = float(size_reg)
        self.hidden = int(hidden)
        self.seed = seed
        self.scorer: Sequential | None = None
        self.loss_history_: list[float] = []

    def fit(
        self, model: GraphClassifier, graphs: list[Data], y: np.ndarray
    ) -> "PGExplainer":
        if not graphs:
            raise GnnError("PGExplainer.fit: no graphs")
        width = int(model.history_["node_embeddings"].shape[1])
        seed = self.seed
        self.scorer = Sequential(
            Linear(2 * width, self.hidden, seed=seed),
            ReLU(),
            Linear(self.hidden, 1, seed=None if seed is None else seed + 1),
        )
        optimizer = Adam(self.scorer.parameters(), lr=self.lr)
        frozen_batches = []
        for graph in graphs:
            batch, target = _prepare(model, graph)
            with no_grad():
                _logits, embeddings = model._decide(batch)
            node_np = embeddings[-1].numpy()
            ei = batch.edge_index
            edge_h = Tensor(np.concatenate([node_np[ei[1]], node_np[ei[0]]], axis=1))
            frozen_batches.append((batch, target, edge_h))
        labels = np.asarray(y, dtype=int).reshape(-1)
        if labels.size != len(graphs):
            raise GnnError("PGExplainer.fit: one label per graph required")
        for _ in range(max(1, self.iters)):
            optimizer.zero_grad()
            total = Tensor(0.0)
            for _i, (batch, target, edge_h) in enumerate(frozen_batches):
                p = self.scorer.forward(edge_h).sigmoid()
                prob = _masked_probability(model, batch, target, p)
                size_term = p.mean()
                loss = -prob + self.size_reg * size_term
                total = total + loss
            total.backward()
            optimizer.step()
            self.loss_history_.append(float(total.item()))
        return self

    def explain(self, model: GraphClassifier, graph: Data, *, top_k: int = 8) -> EdgeExplanation:
        if self.scorer is None:
            raise GnnError("PGExplainer.explain: fit() first")
        batch, target = _prepare(model, graph)
        with no_grad():
            _logits, embeddings = model._decide(batch)
            node_h = embeddings[-1]
            ei = batch.edge_index
            edge_h = np.concatenate([node_h[ei[1]], node_h[ei[0]]], axis=1)
            scores = self.scorer.forward(Tensor(edge_h)).sigmoid().reshape(-1).numpy().copy()
        scores = np.clip(scores, 0.0, 1.0)
        n_edges = scores.size
        top = np.argsort(-scores, kind="stable")[: max(1, min(top_k, n_edges))]
        return EdgeExplanation(
            scores=scores,
            top_edges=np.sort(top),
            method="pgexplainer",
            target_class=target,
            extra={"trained_iters": self.iters},
        )


class AttentionExplainer:
    """The GAT coefficients themselves (pyg's zero-training baseline).

    The score of an undirected edge is the mean of its two directed
    coefficients — the aggregation pyg's attention explainer leaves to
    the caller.
    """

    def explain(
        self, model: GraphClassifier, graph: Data, *, top_k: int = 8, layer: int = -1
    ) -> EdgeExplanation:
        batch, target = _prepare(model, graph)
        gat_layers = [c for c in model.convs if isinstance(c, GATConv)]
        if not gat_layers:
            raise GnnError("AttentionExplainer: the model has no GAT layer")
        chosen = gat_layers[layer]
        with no_grad():
            model._decide(batch)
        if chosen.last_alpha is None:
            raise GnnError("AttentionExplainer: no attention recorded")
        alpha = np.asarray(chosen.last_alpha, dtype=float).reshape(-1)
        ei = batch.edge_index
        if alpha.size != ei.shape[1]:
            raise GnnError("AttentionExplainer: attention does not align with the edges")
        # pair every edge with its opposite direction (when present) and mean
        lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(ei.T)}
        scores = np.zeros(alpha.size)
        for i, (a, b) in enumerate(ei.T):
            opposite = lookup.get((int(b), int(a)))
            scores[i] = (
                (alpha[i] + alpha[opposite]) / 2.0 if opposite is not None else alpha[i]
            )
        scores = np.clip(scores, 0.0, 1.0)
        top = np.argsort(-scores, kind="stable")[: max(1, min(top_k, scores.size))]
        return EdgeExplanation(
            scores=scores,
            top_edges=np.sort(top),
            method="attention",
            target_class=target,
            extra={"layer": int(layer)},
        )


class SubgraphX:
    """Shapley-guided connected-subgraph search (DIG ``subgraphx.py``).

    Teaching scale: sample connected candidate subgraphs (random frontier
    growth), score each by the restricted prediction of the target class,
    keep the best, then attribute its nodes by sampled Shapley values.
    """

    def __init__(
        self,
        *,
        max_nodes: int = 4,
        n_subgraphs: int = 64,
        n_shapley: int = 16,
        seed: int | None = None,
    ) -> None:
        self.max_nodes = int(max_nodes)
        self.n_subgraphs = int(n_subgraphs)
        self.n_shapley = int(n_shapley)
        self.seed = seed

    def _restricted_prob(self, model: GraphClassifier, batch: Batch, target: int, nodes: set[int], n_edges: int) -> float:
        ei = batch.edge_index
        weights = np.zeros(n_edges)
        for col in range(n_edges):
            if int(ei[0, col]) in nodes and int(ei[1, col]) in nodes:
                weights[col] = 1.0
        with no_grad():
            prob = _masked_probability(model, batch, target, Tensor(weights.reshape(-1, 1)))
        return float(prob.numpy().reshape(-1)[0])

    def explain(self, model: GraphClassifier, graph: Data) -> EdgeExplanation:
        batch, target = _prepare(model, graph)
        ei = batch.edge_index
        n_nodes = batch.x.shape[0]
        n_edges = ei.shape[1]
        adjacency: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
        for col in range(n_edges):
            adjacency[int(ei[0, col])].append(int(ei[1, col]))
            adjacency[int(ei[1, col])].append(int(ei[0, col]))
        rng = np.random.default_rng(self.seed)

        def grow(seed_node: int, size: int) -> set[int]:
            chosen = {seed_node}
            frontier = list(dict.fromkeys(adjacency[seed_node]))
            while len(chosen) < size and frontier:
                pick = int(rng.integers(0, len(frontier)))
                node = frontier.pop(pick)
                if node in chosen:
                    continue
                chosen.add(node)
                frontier.extend(n for n in adjacency[node] if n not in chosen)
            return chosen

        best_nodes: set[int] = set()
        best_value = -np.inf
        for _ in range(max(1, self.n_subgraphs)):
            seed_node = int(rng.integers(0, n_nodes))
            size = int(rng.integers(1, min(self.max_nodes, n_nodes) + 1))
            nodes = grow(seed_node, size)
            value = self._restricted_prob(model, batch, target, nodes, n_edges) - 0.01 * len(nodes)
            if value > best_value:
                best_value, best_nodes = value, nodes

        # sampled Shapley over the winning subgraph's nodes
        members = sorted(best_nodes)
        shapley = np.zeros(n_nodes)
        full = self._restricted_prob(model, batch, target, set(members), n_edges)
        for _ in range(max(1, self.n_shapley)):
            order = rng.permutation(members)
            coalition: set[int] = set()
            current = self._restricted_prob(model, batch, target, coalition, n_edges)
            for node in order:
                with_node = self._restricted_prob(model, batch, target, coalition | {node}, n_edges)
                shapley[node] += with_node - current
                coalition.add(node)
                current = with_node
        shapley /= max(1, self.n_shapley)

        scores = np.zeros(n_edges)
        for col in range(n_edges):
            if int(ei[0, col]) in best_nodes and int(ei[1, col]) in best_nodes:
                scores[col] = 1.0
        top = np.argsort(-scores, kind="stable")[: max(1, int(scores.sum()))]
        return EdgeExplanation(
            scores=scores,
            top_edges=np.sort(top),
            method="subgraphx",
            target_class=target,
            extra={
                "subgraph_nodes": members,
                "node_shapley": {int(n): float(shapley[n]) for n in members},
                "restricted_prob": full,
            },
        )


# --------------------------------------------------------------------------
# the DIG xgraph / pyg metric evaluation pair
# --------------------------------------------------------------------------


def _prediction_prob(model: GraphClassifier, batch: Batch, target: int, weights: np.ndarray) -> float:
    with no_grad():
        logits, _ = model._forward(batch, edge_weight=Tensor(weights.reshape(-1, 1)))
    shifted = logits.numpy() - logits.numpy().max(axis=1, keepdims=True)
    weights_exp = np.exp(shifted)
    probs = weights_exp / weights_exp.sum(axis=1, keepdims=True)
    return float(probs[0, target])


def fidelity_plus(
    model: GraphClassifier, graph: Data, explanation: EdgeExplanation
) -> float:
    """DIG's fidelity+: drop in the target's probability when the TOP
    edges are removed — the bigger, the more the mask mattered."""

    batch, target = _prepare(model, graph)
    n = batch.edge_index.shape[1]
    base = _prediction_prob(model, batch, target, np.ones(n))
    weights = np.ones(n)
    weights[explanation.top_edges] = 0.0
    return base - _prediction_prob(model, batch, target, weights)


def fidelity_minus(
    model: GraphClassifier, graph: Data, explanation: EdgeExplanation
) -> float:
    """DIG's fidelity−: drop when ONLY the top edges remain (everything
    else removed) — a faithful mask keeps this near zero."""

    batch, target = _prepare(model, graph)
    n = batch.edge_index.shape[1]
    base = _prediction_prob(model, batch, target, np.ones(n))
    weights = np.zeros(n)
    weights[explanation.top_edges] = 1.0
    return base - _prediction_prob(model, batch, target, weights)


def sparsity(explanation: EdgeExplanation) -> float:
    """The fraction of edges the top-k selection removes."""

    total = explanation.scores.size
    if total == 0:
        return 0.0
    return 1.0 - explanation.top_edges.size / total


def accuracy_at_k(explanation: EdgeExplanation, graph: Data, k: int | None = None) -> float:
    """Of the top-k scored edges, the fraction that are ground-truth
    motif edges (``edge_label``) — the synthetic graphs' sanity check."""

    if graph.edge_label is None:
        raise GnnError("accuracy_at_k: the graph has no edge_label ground truth")
    labels = np.asarray(graph.edge_label, dtype=float).reshape(-1)
    if labels.size != explanation.scores.size:
        raise GnnError("accuracy_at_k: labels do not align with the scores")
    k = explanation.top_edges.size if k is None else max(1, min(int(k), labels.size))
    top = np.argsort(-explanation.scores, kind="stable")[:k]
    return float(labels[top].mean())
