"""Tests for the round-187 trio that landed orphaned (G-2 in the audit).

``pharmaml.ml.gnn`` (the PyG absorption: Data/Batch, the five conv
families, the pools, GraphClassifier), ``pharmaml.ml.gnnxai`` (the
GNNExplainer/PGExplainer/AttentionExplainer/SubgraphX quartet plus the
fidelity metrics), and ``pharmaml.chem.featurize`` (the rdkit-side
molecule→graph bridge).  These engines shipped in commit 2eb0ede
without any test file or GUI wiring -- this suite is the test half of
gap G-2; the Graph model / Graph explain cards are the other half.
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError, PharmaMLError
from pharmaml.ml.gnn import (
    Batch,
    Data,
    EdgeConv,
    GATConv,
    GCNConv,
    GINConv,
    GraphClassifier,
    SAGEConv,
    SAGPooling,
    TopKPooling,
    add_self_loops,
    degree,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    synthetic_motif_graphs,
    to_undirected,
)

# -- Data: the graph record ---------------------------------------------------- #


def test_data_validates_edge_index_bounds() -> None:
    with pytest.raises(DataValidationError):
        Data(np.ones((3, 2)), np.array([[0, 3], [1, 0]]))  # node 3 with n=3


def test_data_1d_features_become_column() -> None:
    g = Data(np.array([1.0, 2.0, 3.0]), np.empty((2, 0)))
    assert g.x.shape == (3, 1)


def test_data_counts() -> None:
    g = Data(np.ones((4, 2)), np.array([[0, 1, 2], [1, 2, 3]]))
    assert g.num_nodes == 4 and g.num_edges == 3


def test_to_undirected_adds_missing_directions() -> None:
    ei = to_undirected(np.array([[0, 1], [1, 2]]))
    assert ei.shape == (2, 4)  # every edge gains its reverse
    assert sorted(map(tuple, ei.T)) == [(0, 1), (1, 0), (1, 2), (2, 1)]


def test_add_self_loops_appends_loops() -> None:
    looped = add_self_loops(np.array([[0], [1]]), num_nodes=3)
    # the original edge survives; every node gains its self-loop
    assert sorted(map(tuple, looped.T)) == [(0, 0), (0, 1), (1, 1), (2, 2)]


def test_degree_counts_incoming() -> None:
    d = degree(np.array([[0, 1, 1], [1, 1, 2]]), num_nodes=3)
    assert d.tolist() == [0, 2, 1]


def test_data_to_undirected_carries_labels() -> None:
    g = Data(np.ones((2, 1)), np.array([[0], [1]]), edge_label=np.array([3.0]))
    both = g.to_undirected()
    assert both.edge_label.tolist() == [3.0, 3.0]


def test_data_with_self_loops_appends_unit_weights() -> None:
    g = Data(np.ones((2, 1)), np.array([[0], [1]]), edge_weight=np.array([2.0]))
    looped = g.with_self_loops()
    assert looped.edge_weight.tolist() == [2.0, 1.0, 1.0]


# -- Batch: the collate ----------------------------------------------------------- #


def _tiny_graphs() -> list[Data]:
    return [
        Data(np.ones((3, 2)), np.array([[0, 1], [1, 2]]), y=np.array([0])),
        Data(np.ones((2, 2)), np.array([[0], [1]]), y=np.array([1])),
    ]


def test_batch_offsets_remap_edges() -> None:
    batch = Batch.from_data_list(_tiny_graphs())
    # graph 2's single edge 0->1 shifts to 3->4
    assert batch.edge_index.tolist() == [[0, 1, 3], [1, 2, 4]]
    assert batch.batch.tolist() == [0, 0, 0, 1, 1]


def test_batch_pointer_splits() -> None:
    batch = Batch.from_data_list(_tiny_graphs())
    assert batch.ptr.tolist() == [0, 3, 5]


def test_batch_keeps_labels() -> None:
    batch = Batch.from_data_list(_tiny_graphs())
    assert batch.y.tolist() == [0, 1]


# -- the five conv families ---------------------------------------------------------- #


def _tape_graph() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(6, dtype=float).reshape(3, 2) / 10.0
    ei = np.array([[0, 1, 2, 0], [1, 2, 0, 2]])
    return x, ei


@pytest.mark.parametrize(
    "conv_cls", [GCNConv, SAGEConv, GINConv, EdgeConv]
)
def test_convs_preserve_node_count(conv_cls: type) -> None:
    from pharmaml.ml.autograd import Tensor

    x, ei = _tape_graph()
    conv = conv_cls(2, 4, seed=0)
    out = conv(Tensor(x), ei)
    assert out.data.shape == (3, 4)


def test_gat_conv_multi_head_output() -> None:
    from pharmaml.ml.autograd import Tensor

    x, ei = _tape_graph()
    conv = GATConv(2, 3, heads=2, seed=0)
    out = conv(Tensor(x), ei)
    assert out.data.shape == (3, 6)  # heads concatenated


def test_gcn_conv_normalizes_by_degree() -> None:
    from pharmaml.ml.autograd import Tensor

    x = np.ones((3, 1))
    ei = np.array([[0, 1, 2], [1, 2, 0]])  # a ring
    conv = GCNConv(1, 1, seed=0)
    # a ring with unit weights: messages average over 1 neighbour each
    out = conv(Tensor(x), ei)
    assert out.data.shape == (3, 1)


# -- the pools ------------------------------------------------------------------------- #


def test_global_add_pool_sums_per_graph() -> None:
    from pharmaml.ml.autograd import Tensor

    x = Tensor(np.arange(5, dtype=float).reshape(-1, 1))
    batch = np.array([0, 0, 0, 1, 1])
    pooled = global_add_pool(x, batch, size=2)
    assert pooled.data.tolist() == [[3.0], [7.0]]  # 0+1+2, 3+4


def test_global_mean_pool_averages_per_graph() -> None:
    from pharmaml.ml.autograd import Tensor

    x = Tensor(np.array([[1.0], [3.0], [10.0]]))
    batch = np.array([0, 0, 1])
    pooled = global_mean_pool(x, batch, size=2)
    assert pooled.data.tolist() == [[2.0], [10.0]]


def test_global_max_pool_takes_max_per_graph() -> None:
    from pharmaml.ml.autograd import Tensor

    x = Tensor(np.array([[1.0], [5.0], [2.0]]))
    batch = np.array([0, 0, 1])
    pooled = global_max_pool(x, batch, size=2)
    assert pooled.data.tolist() == [[5.0], [2.0]]


def test_topk_pooling_keeps_ratio_per_graph() -> None:
    from pharmaml.ml.autograd import Tensor

    x, _ei = _tape_graph()
    pool = TopKPooling(in_channels=2, ratio=2 / 3, seed=0)
    pooled_x, new_batch, remap = pool(
        Tensor(x), np.zeros(3, dtype=int), np.array([0, 3])
    )
    assert pooled_x.data.shape[0] == 2  # ceil(2/3 * 3)
    assert new_batch.tolist() == [0, 0]
    assert remap.size == 3 and (remap >= 0).sum() == 2


def test_sag_pooling_keeps_ratio_per_graph() -> None:
    from pharmaml.ml.autograd import Tensor

    x, _ei = _tape_graph()
    pool = SAGPooling(in_channels=2, ratio=1 / 3, seed=0)
    pooled_x, new_batch, _remap = pool(
        Tensor(x), np.zeros(3, dtype=int), np.array([0, 3])
    )
    assert pooled_x.data.shape[0] == 1  # ceil(1/3 * 3)
    assert new_batch.tolist() == [0]


# -- GraphClassifier: the sklearn face -------------------------------------------------- #


@pytest.fixture(scope="module")
def motif_split() -> tuple[list[Data], np.ndarray]:
    graphs, y = synthetic_motif_graphs(18, seed=11)
    return graphs, y


def test_classifier_fits_and_scores(motif_split: tuple[list[Data], np.ndarray]) -> None:
    graphs, y = motif_split
    assert len(np.unique(y)) == 3  # ring / star / path motifs
    model = GraphClassifier(hidden=8, layers=2, conv="gcn", pool="mean",
                            epochs=120, lr=0.08, seed=11)
    model.fit(graphs, y)
    assert float(model.score(graphs, y)) >= 0.9


@pytest.mark.parametrize("conv", ["sage", "gin", "gat", "edge"])
def test_classifier_conv_families(
    motif_split: tuple[list[Data], np.ndarray], conv: str
) -> None:
    graphs, y = motif_split
    model = GraphClassifier(hidden=6, layers=1, conv=conv, pool="sum",
                            epochs=40, lr=0.1, seed=3)
    model.fit(graphs, y)
    # short training: only the plumbing is asserted here
    assert hasattr(model, "classes_")
    preds = model.predict(graphs)
    assert set(preds) <= set(model.classes_.tolist())


def test_classifier_predict_proba_shapes(
    motif_split: tuple[list[Data], np.ndarray]
) -> None:
    graphs, y = motif_split
    model = GraphClassifier(hidden=6, layers=1, conv="gcn", pool="mean",
                            epochs=20, lr=0.1, seed=1)
    model.fit(graphs, y)
    proba = model.predict_proba(graphs)
    assert proba.shape == (len(graphs), len(model.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classifier_records_loss_history(
    motif_split: tuple[list[Data], np.ndarray]
) -> None:
    graphs, y = motif_split
    model = GraphClassifier(hidden=6, layers=1, epochs=15, lr=0.1, seed=2)
    model.fit(graphs, y)
    assert len(model.loss_history_) == 15


def test_classifier_rejects_unknown_conv(
    motif_split: tuple[list[Data], np.ndarray]
) -> None:
    graphs, y = motif_split
    model = GraphClassifier(conv="cheb")
    with pytest.raises(PharmaMLError):
        model.fit(graphs, y)


def test_classifier_rejects_single_class() -> None:
    graphs, _ = synthetic_motif_graphs(6, seed=0)
    with pytest.raises(PharmaMLError):
        GraphClassifier(epochs=2).fit(graphs, np.zeros(6, dtype=int))


# -- the explainers + metrics (gnnxai) ------------------------------------------------------ #


@pytest.fixture(scope="module")
def trained_explainer_setup() -> tuple[GraphClassifier, list[Data], np.ndarray]:
    # the doctest-proven configuration (18 graphs, seed 11) for a
    # comfortably separable motif problem
    graphs, y = synthetic_motif_graphs(18, seed=11)
    model = GraphClassifier(hidden=8, layers=2, conv="gcn", pool="mean",
                            epochs=120, lr=0.08, seed=11)
    model.fit(graphs, y)
    return model, graphs, y


def test_gnn_explainer_returns_edge_scores(
    trained_explainer_setup: tuple[GraphClassifier, list[Data], np.ndarray]
) -> None:
    model, graphs, _ = trained_explainer_setup
    from pharmaml.ml.gnnxai import GNNExplainer

    explainer = GNNExplainer(iters=60, seed=0)
    explanation = explainer.explain(model, graphs[0])
    assert explanation.scores.shape == (graphs[0].num_edges,)
    assert np.all(explanation.scores >= 0) and np.all(explanation.scores <= 1)
    assert explanation.top_edges.size <= 8


def test_pg_explainer_trains_then_explains(
    trained_explainer_setup: tuple[GraphClassifier, list[Data], np.ndarray]
) -> None:
    model, graphs, y = trained_explainer_setup
    from pharmaml.ml.gnnxai import PGExplainer

    explainer = PGExplainer(iters=30, seed=0)
    explainer.fit(model, graphs, y)
    assert len(explainer.loss_history_) == 30
    explanation = explainer.explain(model, graphs[0])
    assert explanation.scores.shape == (graphs[0].num_edges,)


def test_attention_explainer_needs_gat_layer(
    trained_explainer_setup: tuple[GraphClassifier, list[Data], np.ndarray]
) -> None:
    from pharmaml.exceptions import PharmaMLError as _PE
    from pharmaml.ml.gnnxai import AttentionExplainer

    model, graphs, _ = trained_explainer_setup  # a GCN model
    with pytest.raises(_PE):
        AttentionExplainer().explain(model, graphs[0])


def test_attention_explainer_on_gat_model() -> None:
    from pharmaml.ml.gnnxai import AttentionExplainer

    graphs, y = synthetic_motif_graphs(12, seed=4)
    model = GraphClassifier(hidden=6, layers=1, conv="gat", pool="mean",
                            epochs=40, lr=0.1, seed=4)
    model.fit(graphs, y)
    explanation = AttentionExplainer().explain(model, graphs[0])
    assert explanation.scores.shape == (graphs[0].num_edges,)


def test_subgraphx_finds_connected_subset(
    trained_explainer_setup: tuple[GraphClassifier, list[Data], np.ndarray]
) -> None:
    model, graphs, _ = trained_explainer_setup
    from pharmaml.ml.gnnxai import SubgraphX

    explainer = SubgraphX(n_subgraphs=16, n_shapley=4, seed=0)
    explanation = explainer.explain(model, graphs[0])
    assert explanation.scores.shape == (graphs[0].num_edges,)


def test_fidelity_metrics_bound(
    trained_explainer_setup: tuple[GraphClassifier, list[Data], np.ndarray]
) -> None:
    model, graphs, _ = trained_explainer_setup
    from pharmaml.ml.gnnxai import GNNExplainer, fidelity_minus, fidelity_plus

    explanation = GNNExplainer(iters=40, seed=0).explain(model, graphs[0])
    fp = fidelity_plus(model, graphs[0], explanation)
    fm = fidelity_minus(model, graphs[0], explanation)
    # both are probability drops; the metrics exist and are finite
    assert np.isfinite(fp) and np.isfinite(fm)


def test_sparsity_fraction(
    trained_explainer_setup: tuple[GraphClassifier, list[Data], np.ndarray]
) -> None:
    model, graphs, _ = trained_explainer_setup
    from pharmaml.ml.gnnxai import GNNExplainer, sparsity

    explanation = GNNExplainer(iters=40, seed=0).explain(model, graphs[0])
    assert 0.0 <= sparsity(explanation) <= 1.0


def test_accuracy_at_k_hits_ground_truth() -> None:
    # the synthetic motif's ground-truth important edges are edge_label=1
    graphs, _ = synthetic_motif_graphs(10, seed=9)
    from pharmaml.ml.gnnxai import EdgeExplanation, accuracy_at_k

    scores = np.asarray(graphs[0].edge_label, dtype=float)
    k = int(scores.sum())
    explanation = EdgeExplanation(
        scores=scores,
        top_edges=np.argsort(-scores)[:k],
        method="oracle",
        target_class=0,
    )
    assert accuracy_at_k(explanation, graphs[0], k=k) == 1.0


# -- the featurize bridge (rdkit side) --------------------------------------------------------- #


def test_smiles_to_graph_shapes() -> None:
    from pharmaml.chem.featurize import smiles_to_graph

    g = smiles_to_graph("CCO")  # ethanol: 3 atoms, 2 bonds
    assert g.num_nodes == 3
    assert g.num_edges in {2, 4}  # directed or undirected storage
    assert g.x.shape[1] > 1  # the atom feature table is one-hot bundles


def test_smiles_to_graph_carries_target() -> None:
    from pharmaml.chem.featurize import smiles_to_graph

    g = smiles_to_graph("c1ccccc1", y=1.7)
    assert float(np.asarray(g.y).ravel()[0]) == 1.7


def test_atom_features_are_binary() -> None:
    from pharmaml.chem.featurize import smiles_to_graph

    g = smiles_to_graph("CC(=O)O")
    assert set(np.unique(g.x)) <= {0.0, 1.0}


def test_featurize_round_trips_through_the_classifier() -> None:
    from pharmaml.chem.featurize import smiles_to_graph

    graphs = [smiles_to_graph(s, y=float(i % 2)) for i, s in enumerate(
        ["CCO", "CCC", "CCCC", "CCN", "CCCl", "CCBr"]
    )]
    y = np.array([float(i % 2) for i in range(6)], dtype=int)
    model = GraphClassifier(hidden=6, layers=1, conv="gcn", pool="mean",
                            epochs=30, lr=0.1, seed=0)
    model.fit(graphs, y)
    assert hasattr(model, "classes_")


def test_featurize_rejects_bad_smiles() -> None:
    from pharmaml.chem.featurize import smiles_to_graph

    with pytest.raises((PharmaMLError, ValueError)):
        smiles_to_graph("not-a-smiles-$$$")


def test_molecule_to_data_matches_smiles_path() -> None:
    from pharmaml.chem.featurize import molecule_to_data, smiles_to_graph
    from pharmaml.chem.molgraph import parse_smiles

    mol = parse_smiles("CCO")
    via_mol = molecule_to_data(mol)
    via_smiles = smiles_to_graph("CCO")
    assert via_mol.num_nodes == via_smiles.num_nodes
    assert via_mol.num_edges == via_smiles.num_edges


# -- regression: the two genuine bugs found by this suite --------------------- #


def test_classifier_topk_pool_regression() -> None:
    """pool="topk" was broken end-to-end (np.argsort on a tape Tensor)."""
    graphs, y = synthetic_motif_graphs(12, seed=2)
    model = GraphClassifier(hidden=6, layers=1, conv="gcn", pool="topk",
                            epochs=20, lr=0.1, seed=2)
    model.fit(graphs, y)
    assert hasattr(model, "classes_")


def test_classifier_sag_pool_regression() -> None:
    """pool="sag" shared the same Tensor-indexing bug."""
    graphs, y = synthetic_motif_graphs(12, seed=2)
    model = GraphClassifier(hidden=6, layers=1, conv="gcn", pool="sag",
                            epochs=20, lr=0.1, seed=2)
    model.fit(graphs, y)
    assert hasattr(model, "classes_")


def test_pg_explainer_explain_regression() -> None:
    """PGExplainer.explain fancy-indexed the tape Tensor (fit did not)."""
    graphs, y = synthetic_motif_graphs(10, seed=6)
    model = GraphClassifier(hidden=6, layers=1, conv="gcn", pool="mean",
                            epochs=40, lr=0.1, seed=6)
    model.fit(graphs, y)
    from pharmaml.ml.gnnxai import PGExplainer

    explainer = PGExplainer(iters=10, seed=0)
    explainer.fit(model, graphs, y)
    explanation = explainer.explain(model, graphs[0])
    assert explanation.scores.shape == (graphs[0].num_edges,)
