"""Tests for ``pharmaml.ml.pruning`` -- magnitude/channel pruning engines.

The lost round-188/189 engine rebuilt per ``scratch/round189_state_after_rollback.md``
§3: magnitude pruning (global + local scopes, deterministic tie-breaks),
structured channel pruning with the paired bias cut, the Zhu-Gupta cubic
schedule, the iterative prune-and-retrain loop, and the report bookkeeping.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pharmaml.ml.autograd import Linear, Module, ReLU, Sequential
from pharmaml.ml.pruning import (
    LayerPruning,
    PruningReport,
    apply_magnitude_pruning,
    count_sparsity,
    iterative_prune_and_retrain,
    magnitude_mask,
    prune_linear_layer,
    prune_linear_model,
    sparsity_schedule,
    structured_channel_mask,
)

# -- magnitude_mask ----------------------------------------------------------- #


def test_mask_keeps_exact_count_global() -> None:
    w = np.arange(1.0, 11.0)  # magnitudes 1..10
    mask = magnitude_mask(w, 0.3, scope="global")
    assert int(mask.sum()) == 7  # ceil(0.7 * 10)
    assert mask.astype(int).tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_mask_global_threshold_is_global() -> None:
    w = np.array([[10.0, 0.1, 0.2], [0.3, 9.0, 0.4]])
    mask = magnitude_mask(w, 0.5, scope="global")
    # the two big entries survive from different rows
    assert mask[0, 0] and mask[1, 1]
    assert int(mask.sum()) == 3  # ceil(0.5 * 6)


def test_mask_local_thresholds_per_row() -> None:
    w = np.array([[10.0, 9.0, 0.1], [0.05, 0.04, 0.03]])
    mask = magnitude_mask(w, 1 / 3, scope="local")
    assert mask[0].tolist() == [True, True, False]
    assert mask[1].tolist() == [True, False, False] or mask[1].tolist() == [
        True, True, False
    ]


def test_mask_ties_break_to_earlier_index() -> None:
    w = np.array([1.0, 1.0, 1.0, 1.0])
    mask = magnitude_mask(w, 0.5, scope="global")
    assert mask.astype(int).tolist() == [1, 1, 0, 0]


def test_mask_zero_sparsity_keeps_all() -> None:
    w = np.random.default_rng(0).normal(size=(3, 4))
    assert magnitude_mask(w, 0.0).all()


def test_mask_sparsity_near_one_keeps_one_per_unit() -> None:
    w = np.ones((2, 5))
    mask = magnitude_mask(w, 0.99, scope="local")
    assert mask.astype(int).sum(axis=1).tolist() == [1, 1]


def test_mask_shape_preserved() -> None:
    w = np.random.default_rng(1).normal(size=(5, 7, 2))
    mask = magnitude_mask(w, 0.4, scope="global")
    assert mask.shape == w.shape


def test_mask_1d_input() -> None:
    mask = magnitude_mask(np.array([0.1, -0.9, 0.3]), 1 / 3)
    assert mask.astype(int).tolist() == [0, 1, 1]


def test_mask_empty_block() -> None:
    mask = magnitude_mask(np.array([]), 0.5)
    assert mask.size == 0


@pytest.mark.parametrize("sparsity", [-0.1, 1.0, 1.5])
def test_mask_rejects_out_of_range_sparsity(sparsity: float) -> None:
    with pytest.raises(ValueError):
        magnitude_mask(np.ones((2, 2)), sparsity)


def test_mask_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError):
        magnitude_mask(np.ones((2, 2)), 0.5, scope="layerwise")


def test_mask_survivors_are_the_largest_magnitudes() -> None:
    w = np.array([[0.05, 0.9, -0.8, 0.1]])
    mask = magnitude_mask(w, 0.5, scope="local")
    kept = np.abs(w[mask])
    dropped = np.abs(w[~mask])
    assert kept.min() > dropped.max()


# -- count_sparsity ------------------------------------------------------------ #


def test_count_sparsity_ratio() -> None:
    assert count_sparsity(np.zeros((3, 3))) == 1.0
    assert count_sparsity(np.ones(4)) == 0.0


def test_count_sparsity_empty_is_zero() -> None:
    assert count_sparsity(np.array([])) == 0.0


# -- apply_magnitude_pruning (the tape walk) ------------------------------------ #


def _two_layer_net() -> Sequential:
    return Sequential(Linear(4, 3, seed=1), ReLU(), Linear(3, 1, seed=2))


def test_apply_prunes_sequential_children_in_list() -> None:
    net = _two_layer_net()
    report = apply_magnitude_pruning(net, 0.5, scope="global")
    assert len(report.layers) == 2
    assert report.layers[0].shape == (3, 4)
    assert report.layers[1].shape == (1, 3)


def test_apply_in_place_zeroes_the_weights() -> None:
    net = _two_layer_net()
    before = np.asarray(net.layers[0].weight.data).copy()
    apply_magnitude_pruning(net, 0.5, scope="global")
    after = np.asarray(net.layers[0].weight.data)
    assert (after == 0).sum() >= 1
    # non-zero survivors keep their original value
    survivors = after != 0
    assert np.allclose(after[survivors], before[survivors])


def test_apply_report_counts_match_weights() -> None:
    net = _two_layer_net()
    report = apply_magnitude_pruning(net, 0.5, scope="global")
    assert report.n_weights == 12 + 3
    assert report.n_zeroed == int((np.asarray(net.layers[0].weight.data) == 0).sum()
                                  + (np.asarray(net.layers[2].weight.data) == 0).sum())
    assert report.n_kept == report.n_weights - report.n_zeroed


def test_apply_achieved_sparsity_close_to_target() -> None:
    net = Sequential(
        Linear(20, 20, seed=5), ReLU(), Linear(20, 20, seed=6), ReLU(),
        Linear(20, 10, seed=7),
    )
    report = apply_magnitude_pruning(net, 0.5, scope="global")
    assert abs(report.achieved_sparsity - 0.5) < 0.05


def test_apply_not_in_place_leaves_module_untouched() -> None:
    net = _two_layer_net()
    before = np.asarray(net.layers[0].weight.data).copy()
    report = apply_magnitude_pruning(net, 0.5, scope="global", in_place=False)
    after = np.asarray(net.layers[0].weight.data)
    assert np.array_equal(before, after)
    assert report.achieved_sparsity > 0  # the report still counts the cut


def test_apply_local_scope_per_layer_sparsity() -> None:
    net = _two_layer_net()
    report = apply_magnitude_pruning(net, 0.5, scope="local")
    for layer in report.layers:
        assert abs(layer.sparsity - 0.5) < 0.2


def test_apply_nested_module_tree() -> None:
    class Inner(Module):
        def __init__(self) -> None:
            self.dense = Linear(3, 2, seed=1)

    class Outer(Module):
        def __init__(self) -> None:
            self.inner = Inner()
            self.head = Linear(2, 1, seed=2)

    report = apply_magnitude_pruning(Outer(), 0.4)
    assert len(report.layers) == 2
    assert report.layers[0].name.startswith("inner.dense")
    assert report.layers[1].name == "head"


def test_report_as_dict_round_trip() -> None:
    net = _two_layer_net()
    payload = apply_magnitude_pruning(net, 0.5).as_dict()
    assert payload["scope"] == "local"
    assert payload["target_sparsity"] == 0.5
    assert len(payload["layers"]) == 2
    assert set(payload["layers"][0]) == {
        "name", "shape", "n_weights", "n_zeroed", "sparsity",
    }


def test_report_zero_sparsity_keeps_everything() -> None:
    net = _two_layer_net()
    report = apply_magnitude_pruning(net, 0.0)
    assert report.n_zeroed == 0
    assert report.achieved_sparsity == 0.0


# -- structured_channel_mask ----------------------------------------------------- #


def test_channel_mask_drops_weakest_rows() -> None:
    w = np.array([[1.0, 1.0], [0.1, 0.1], [0.5, 0.5]])
    mask = structured_channel_mask(w, 1 / 3)
    assert mask.astype(int).tolist() == [[1, 1], [0, 0], [1, 1]]


def test_channel_mask_keeps_whole_rows_only() -> None:
    w = np.random.default_rng(2).normal(size=(6, 4)) * np.array([1, 5, 1, 1])
    mask = structured_channel_mask(w, 0.5)
    rows_kept = mask[:, 0]
    assert np.all(mask.sum(axis=1)[rows_kept] == w.shape[1])  # all-or-nothing
    assert np.all(mask.sum(axis=1)[~rows_kept] == 0)
    assert int(rows_kept.sum()) == 3  # ceil(0.5 * 6)


def test_channel_mask_never_empties_the_layer() -> None:
    w = np.ones((3, 2))
    mask = structured_channel_mask(w, 0.99)
    assert int(mask[:, 0].sum()) == 1


def test_channel_mask_ties_break_earlier() -> None:
    w = np.ones((4, 2))
    mask = structured_channel_mask(w, 0.5)
    assert mask[:, 0].astype(int).tolist() == [1, 1, 0, 0]


def test_channel_mask_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        structured_channel_mask(np.ones(6), 0.5)


@pytest.mark.parametrize("sparsity", [-0.5, 1.0])
def test_channel_mask_rejects_bad_sparsity(sparsity: float) -> None:
    with pytest.raises(ValueError):
        structured_channel_mask(np.ones((3, 3)), sparsity)


# -- prune_linear_layer ------------------------------------------------------------ #


def test_prune_linear_layer_magnitude_cuts_bias_not() -> None:
    layer = Linear(6, 4, seed=3)
    bias_before = np.asarray(layer.bias.data).copy()
    outcome = prune_linear_layer(layer, 0.5, structured=False)
    assert outcome.shape == (4, 6)
    assert (np.asarray(layer.weight.data) == 0).sum() >= 4
    assert np.array_equal(np.asarray(layer.bias.data), bias_before)


def test_prune_linear_layer_structured_cuts_paired_bias() -> None:
    layer = Linear(6, 4, seed=4)
    outcome = prune_linear_layer(layer, 0.5, structured=True)
    w = np.asarray(layer.weight.data)
    bias = np.asarray(layer.bias.data)
    dead_rows = (w == 0).all(axis=1)
    assert dead_rows.sum() == 2  # ceil(0.5 * 4) -> 2 channels survive
    assert (bias[dead_rows] == 0).all()  # the paired bias entries go too
    assert outcome.n_zeroed == int((w == 0).sum())


def test_prune_linear_layer_report_fields() -> None:
    layer = Linear(5, 3, seed=5)
    outcome = prune_linear_layer(layer, 0.4, structured=False)
    assert isinstance(outcome, LayerPruning)
    assert outcome.n_kept + outcome.n_zeroed == outcome.n_weights


# -- prune_linear_model (sklearn coef_) ----------------------------------------------- #


def test_prune_linear_model_flat_block() -> None:
    coef = np.array([0.1, -0.7, 0.4])
    pruned, sparsity = prune_linear_model(coef, 1 / 3)
    assert pruned.tolist() == [0.0, -0.7, 0.4]
    assert sparsity == pytest.approx(1 / 3)


def test_prune_linear_model_original_not_mutated() -> None:
    coef = np.array([0.5, -0.5, 0.2])
    prune_linear_model(coef, 0.5)
    assert coef.tolist() == [0.5, -0.5, 0.2]


# -- sparsity_schedule ------------------------------------------------------------------ #


def test_schedule_lands_on_final() -> None:
    assert sparsity_schedule(0.1, 0.9, 5)[-1] == 0.9


def test_schedule_starts_at_initial() -> None:
    assert sparsity_schedule(0.2, 0.8, 5)[0] == pytest.approx(0.2)


def test_schedule_is_monotone() -> None:
    levels = sparsity_schedule(0.0, 0.9, 7)
    assert all(b >= a for a, b in itertools.pairwise(levels))


def test_schedule_single_step_short_circuits() -> None:
    assert sparsity_schedule(0.0, 0.75, 1) == [0.75]


def test_schedule_two_steps() -> None:
    assert sparsity_schedule(0.0, 0.8, 2) == [0.0, 0.8]


def test_schedule_cubic_midpoint() -> None:
    levels = sparsity_schedule(0.0, 0.8, 3)
    assert levels[1] == pytest.approx(0.7)  # 0.8 - 0.8 * 0.5^3


# -- iterative_prune_and_retrain ----------------------------------------------------------- #


def test_iterative_loop_calls_retrain_per_level() -> None:
    net = Sequential(Linear(6, 4, seed=3), ReLU(), Linear(4, 2, seed=4))
    calls: list[float] = []

    def retrain(module: Module) -> float:
        calls.append(1.0)
        return 0.5

    _, report, values = iterative_prune_and_retrain(
        net, retrain, target_sparsity=0.5, steps=2
    )
    assert values == [0.5, 0.5]
    assert len(calls) == 2
    assert 0.3 <= report.achieved_sparsity <= 0.6


def test_iterative_loop_masks_only_tighten() -> None:
    net = Sequential(Linear(8, 8, seed=6), ReLU())
    seen: list[np.ndarray] = []

    def retrain(module: Module) -> float:
        seen.append(np.asarray(module.layers[0].weight.data).copy())
        return 0.0

    iterative_prune_and_retrain(
        net, retrain, target_sparsity=0.75, steps=4, initial_sparsity=0.0
    )
    # every entry zeroed at step i is still zero at step i+1
    for early, late in itertools.pairwise(seen):
        assert ((early == 0) & (late != 0)).sum() == 0


def test_iterative_loop_reaches_target() -> None:
    net = Sequential(Linear(20, 20, seed=8), ReLU(), Linear(20, 20, seed=9))
    _, report, _ = iterative_prune_and_retrain(
        net, lambda m: 0.0, target_sparsity=0.6, steps=3
    )
    assert report.achieved_sparsity >= 0.55


# -- the report dataclasses ----------------------------------------------------------------- #


def test_layer_pruning_kept_property() -> None:
    layer = LayerPruning(
        name="l0", shape=(3, 4), n_weights=12, n_zeroed=4, sparsity=1 / 3
    )
    assert layer.n_kept == 8


def test_empty_report_zero_division_guard() -> None:
    report = PruningReport()
    assert report.achieved_sparsity == 0.0
    assert report.n_weights == 0
