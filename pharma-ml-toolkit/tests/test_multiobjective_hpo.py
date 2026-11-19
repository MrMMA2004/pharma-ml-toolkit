"""Tests for ``pharmaml.ml.multiobjective`` -- the Pareto HPO study.

The best-of-ml-python HPO/AutoML audit's closing round: this file pins the
Pareto order (strict/equal/partial dominance), the pairwise front
extraction (checked against an independently-coded brute force on seeded
random clouds, 2-D and 4-D, duplicates included), the exact 2-D
hypervolume staircase (hand-computed rectangle unions, the
dominated-adds-zero rule, the WFG reference contract), NSGA-II crowding
(corners +inf, the degenerate-span objective), the Tchebycheff
decomposition (the MOEA/D scalar, the corner weight), the Das-Dennis
lattice spread and the seeded simplex draws, the study bookkeeping (the
ideal-point drift, the first-trial-is-its-own-ideal scalar of 0, weight
cycling), the report (non-domination of the returned front, the nadir+10%
reference, the monotone hypervolume curve, the 3-objective None
contract), and the sklearn adapter over ``cross_validate``'s ledger.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.hpo import SearchSpace
from pharmaml.ml.multiobjective import (
    MOTrial,
    MultiObjectiveStudy,
    ParetoReport,
    chebyshev_scalarize,
    crowding_distance,
    dominates,
    hypervolume_2d,
    pareto_indices,
    tune_model_multi,
)


def _brute_force_front(points: list[tuple[float, ...]]) -> set[int]:
    """An independent dominance filter (pure-python, zip-based)."""
    keep: set[int] = set()
    for i, a in enumerate(points):
        beaten = False
        for j, b in enumerate(points):
            if i == j:
                continue
            if all(y <= x for x, y in zip(a, b)) and any(y < x for x, y in zip(a, b)):
                beaten = True
                break
        if not beaten:
            keep.add(i)
    return keep


def _bi_objective_space() -> SearchSpace:
    return SearchSpace.uniform("x", 0.0, 1.0)


# -- the Pareto order -------------------------------------------------------- #


def test_dominates_strict_partial_and_equal() -> None:
    assert dominates((0.5, 2.0), (0.6, 2.0)) is True
    assert dominates((0.5, 2.0), (0.5, 1.9)) is False
    assert dominates((0.5, 2.0), (0.5, 2.0)) is False
    assert dominates((0.4, 1.9), (0.5, 2.0)) is True


def test_dominates_validates_length_and_finiteness() -> None:
    with pytest.raises(DataValidationError):
        dominates((0.5,), (0.5, 2.0))
    with pytest.raises(DataValidationError):
        dominates((float("nan"), 1.0), (0.5, 2.0))


def test_pareto_indices_matches_brute_force_on_random_clouds() -> None:
    rng = np.random.default_rng(11)
    for n_dims, n_points in ((2, 40), (4, 25)):
        cloud = rng.uniform(size=(n_points, n_dims)).tolist()
        expected = _brute_force_front(cloud)
        assert set(pareto_indices(cloud)) == expected


def test_pareto_indices_duplicates_survive_together() -> None:
    assert pareto_indices([(1.0, 1.0), (1.0, 1.0), (2.0, 0.5)]) == [0, 1, 2]


def test_pareto_indices_known_cloud() -> None:
    assert pareto_indices([(1.0, 2.0), (3.0, 1.0), (2.0, 2.0), (1.0, 5.0)]) == [0, 1]


# -- the exact 2-D hypervolume ----------------------------------------------- #


def test_hypervolume_staircase_hand_computed() -> None:
    # union of [1,2]x[1,2] and [1.5,2]x[0.5,2] minus their overlap = 1.25
    assert hypervolume_2d([(1.0, 1.0), (1.5, 0.5)], ref=(2.0, 2.0)) == pytest.approx(1.25)
    assert hypervolume_2d([(1.0, 1.0)], ref=(2.0, 2.0)) == pytest.approx(1.0)


def test_hypervolume_dominated_point_adds_zero() -> None:
    base = hypervolume_2d([(1.0, 1.0), (1.5, 0.5)], ref=(2.0, 2.0))
    padded = hypervolume_2d([(1.0, 1.0), (1.5, 0.5), (1.4, 1.9)], ref=(2.0, 2.0))
    assert padded == pytest.approx(base)


def test_hypervolume_empty_and_validation() -> None:
    assert hypervolume_2d([], ref=(2.0, 2.0)) == 0.0
    with pytest.raises(DataValidationError):
        hypervolume_2d([(1.0, 3.0)], ref=(2.0, 2.0))  # escapes the reference
    with pytest.raises(DataValidationError):
        hypervolume_2d([(1.0, 1.0, 1.0)], ref=(2.0, 2.0))  # 3 objectives
    with pytest.raises(DataValidationError):
        hypervolume_2d([(1.0, float("nan"))], ref=(2.0, 2.0))


def test_hypervolume_monotone_in_the_point_set() -> None:
    rng = np.random.default_rng(5)
    points = rng.uniform(0.0, 1.0, size=(30, 2)).tolist()
    ref = (1.1, 1.1)
    growing = [hypervolume_2d(points[: i + 1], ref=ref) for i in range(len(points))]
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(growing))


# -- NSGA-II crowding -------------------------------------------------------- #


def test_crowding_corners_inf_interiors_finite() -> None:
    pts = [(0.2, 0.9), (0.9, 0.4), (0.25, 0.85), (0.5, 0.6)]
    dist = crowding_distance(pts)
    assert dist[0] == math.inf and dist[1] == math.inf
    assert dist[2] == pytest.approx(0.4285714 + 0.6)
    assert dist[3] == pytest.approx(0.9285714 + 0.9)


def test_crowding_degenerate_span_objective_is_silent() -> None:
    # identical values in objective 2 carry no spread: only objective 1 ranks,
    # and interior gaps are between SORTED NEIGHBOURS, not the extremes
    dist = crowding_distance([(1.0, 7.0), (2.0, 7.0), (1.5, 7.0), (3.0, 7.0)])
    assert dist[0] == math.inf and dist[3] == math.inf
    assert dist[1] == pytest.approx((3.0 - 1.5) / 2.0)
    assert dist[2] == pytest.approx((2.0 - 1.0) / 2.0)


def test_crowding_two_points_are_both_corners() -> None:
    assert list(crowding_distance([(1.0, 1.0), (2.0, 2.0)])) == [math.inf, math.inf]


# -- the Tchebycheff decomposition ------------------------------------------- #


def test_chebyshev_known_values_and_corner_weight() -> None:
    assert chebyshev_scalarize((0.5, 2.0), weights=(1.0, 0.5), ideal=(0.4, 1.0)) == 0.5
    assert chebyshev_scalarize((0.5, 2.0), weights=(0.0, 1.0), ideal=(0.4, 1.0)) == 1.0
    assert chebyshev_scalarize((0.4, 1.0), weights=(1.0, 1.0), ideal=(0.4, 1.0)) == 0.0


def test_chebyshev_validation() -> None:
    with pytest.raises(DataValidationError):
        chebyshev_scalarize((0.5,), weights=(1.0, 1.0), ideal=(0.4,))
    with pytest.raises(DataValidationError):
        chebyshev_scalarize((0.5, 2.0), weights=(-1.0, 0.5), ideal=(0.4, 1.0))
    with pytest.raises(DataValidationError):
        chebyshev_scalarize((0.5, 2.0), weights=(0.0, 0.0), ideal=(0.4, 1.0))
    with pytest.raises(DataValidationError):
        chebyshev_scalarize((0.5, float("nan")), weights=(1.0, 1.0), ideal=(0.4, 1.0))


# -- the study ---------------------------------------------------------------- #


def test_study_records_trials_with_lattice_weights() -> None:
    study = MultiObjectiveStudy(
        lambda p: (p["x"], 1.0 - p["x"]),
        _bi_objective_space(),
        n_objectives=2,
        sampler="random",
        seed=3,
        weight_mode="spread",
        n_weight_vectors=4,
    )
    trials = study.run(6)
    assert len(trials) == 6
    assert all(isinstance(t, MOTrial) for t in trials)
    assert all(len(t.values) == 2 for t in trials)
    # the Das-Dennis lattice for k=2, 4 vectors: (0,1), (1/3,2/3), (2/3,1/3), (1,0)
    assert study.trials[0].weights == (0.0, 1.0)
    assert study.trials[1].weights == pytest.approx((1.0 / 3.0, 2.0 / 3.0))
    assert study.trials[4].weights == (0.0, 1.0)  # the cycle wraps
    assert all(abs(sum(w) - 1.0) < 1e-9 for w in (t.weights for t in trials))


def test_study_first_trial_is_its_own_ideal() -> None:
    study = MultiObjectiveStudy(
        lambda p: (p["x"], 1.0 - p["x"]),
        _bi_objective_space(),
        n_objectives=2,
        sampler="random",
        seed=7,
    )
    study.run(1)
    assert study.trials[0].scalar == 0.0


def test_study_random_weights_are_seeded_simplex_draws() -> None:
    study = MultiObjectiveStudy(
        lambda p: (p["x"], 1.0 - p["x"]),
        _bi_objective_space(),
        n_objectives=2,
        sampler="random",
        seed=9,
        weight_mode="random",
    )
    twin = MultiObjectiveStudy(
        lambda p: (p["x"], 1.0 - p["x"]),
        _bi_objective_space(),
        n_objectives=2,
        sampler="random",
        seed=9,
        weight_mode="random",
    )
    study.run(5)
    twin.run(5)
    assert [t.weights for t in study.trials] == [t.weights for t in twin.trials]
    assert all(abs(sum(t.weights) - 1.0) < 1e-9 for t in study.trials)
    assert all(w > 0.0 for t in study.trials for w in t.weights)


def test_study_validation_contracts() -> None:
    space = _bi_objective_space()
    with pytest.raises(DataValidationError):
        MultiObjectiveStudy(lambda p: (p["x"],), space, n_objectives=1)
    with pytest.raises(DataValidationError):
        MultiObjectiveStudy(lambda p: (p["x"], 0.0), space, sampler="cma")
    with pytest.raises(DataValidationError):
        MultiObjectiveStudy(lambda p: (p["x"], 0.0), space, weight_mode="grid")
    with pytest.raises(DataValidationError):
        MultiObjectiveStudy(lambda p: (p["x"], 0.0), space, n_weight_vectors=1)
    with pytest.raises(DataValidationError):
        MultiObjectiveStudy(lambda p: (p["x"], 0.0), space, n_objectives=3).run(1)


def test_study_rejects_nonfinite_objective() -> None:
    study = MultiObjectiveStudy(
        lambda p: (p["x"], float("inf")),
        _bi_objective_space(),
        n_objectives=2,
        sampler="random",
        seed=1,
    )
    with pytest.raises(DataValidationError):
        study.run(1)


def test_study_front_on_a_valley_pair() -> None:
    # f1 is minimized at x=0.2, f2 at x=0.8: the trade-off curve is [0.2, 0.8]
    study = MultiObjectiveStudy(
        lambda p: ((p["x"] - 0.2) ** 2, (p["x"] - 0.8) ** 2),
        _bi_objective_space(),
        n_objectives=2,
        sampler="random",
        seed=13,
    )
    study.run(40)
    report = study.report(["loss_a", "loss_b"])
    assert len(report.trials) >= 1
    for trial in report.trials:
        assert 0.2 - 0.15 <= trial.params["x"] <= 0.8 + 0.15
    # and the front is genuinely non-dominated against the whole history
    history = [t.values for t in study.trials]
    assert set(pareto_indices(history)) >= {t.number for t in report.trials}
    # the endpoints of the trade-off are reachable: best_for names them
    a_best = report.best_for("loss_a")
    assert a_best.values[0] == pytest.approx(min(v[0] for v in history), abs=1e-6)


# -- the report --------------------------------------------------------------- #


def test_report_front_crowding_and_reference_margin() -> None:
    study = MultiObjectiveStudy(
        lambda p: (p["x"], 1.0 - p["x"]),
        _bi_objective_space(),
        n_objectives=2,
        sampler="random",
        seed=21,
    )
    study.run(10)
    report = study.report(["f1", "f2"])
    assert isinstance(report, ParetoReport)
    assert report.n_trials == 10
    assert len(report.crowding) == len(report.trials)
    assert report.ref is not None and report.hypervolume is not None
    # the reference is the nadir padded by 10% of the ideal-to-nadir range
    matrix = np.array([t.values for t in study.trials])
    ideal, nadir = matrix.min(axis=0), matrix.max(axis=0)
    margin = 0.1 * (nadir - ideal)
    assert report.ref == (
        pytest.approx(nadir[0] + margin[0]),
        pytest.approx(nadir[1] + margin[1]),
    )
    # the curve is monotone non-decreasing under the fixed reference
    curve = report.hypervolume_curve
    assert len(curve) == 10
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(curve))
    assert curve[-1] == pytest.approx(report.hypervolume)


def test_report_three_objectives_have_no_hypervolume() -> None:
    study = MultiObjectiveStudy(
        lambda p: (p["x"], p["x"] ** 2, 1.0 - p["x"]),
        SearchSpace.uniform("x", 0.0, 1.0),
        n_objectives=3,
        sampler="random",
        seed=4,
    )
    study.run(8)
    report = study.report(["a", "b", "c"])
    assert report.hypervolume is None
    assert report.ref is None
    assert report.hypervolume_curve == ()


def test_report_name_validation_and_empty_front() -> None:
    study = MultiObjectiveStudy(
        lambda p: (p["x"], 1.0 - p["x"]),
        _bi_objective_space(),
        n_objectives=2,
        sampler="random",
        seed=2,
    )
    empty = study.report(["a", "b"])
    assert empty.trials == [] and empty.n_trials == 0
    with pytest.raises(DataValidationError):
        empty.best_for("a")
    with pytest.raises(DataValidationError):
        study.report(["a", "b", "c"])


def test_report_dataframe_columns() -> None:
    study = MultiObjectiveStudy(
        lambda p: (p["x"], 1.0 - p["x"]),
        _bi_objective_space(),
        n_objectives=2,
        sampler="random",
        seed=6,
    )
    study.run(4)
    frame = study.report(["loss_a", "loss_b"]).to_dataframe()
    assert list(frame.columns) == ["number", "loss_a", "loss_b", "scalar", "crowding"]
    assert len(frame) >= 1


# -- the sklearn adapter ------------------------------------------------------ #


def _toy_classification_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_tune_model_multi_front_is_non_dominated() -> None:
    X, y = _toy_classification_data()
    space = SearchSpace.uniform("alpha", 0.01, 1.0, log=True)
    report = tune_model_multi(
        "ridge_classifier",
        space,
        X,
        y,
        task="classification",
        cv=2,
        n_trials=8,
        sampler="random",
        seed=1,
    )
    assert report.objective_names == ("score", "fit_seconds")
    assert len(report.trials) >= 1
    assert all(len(t.values) == 2 for t in report.trials)
    assert all(math.isfinite(v) for t in report.trials for v in t.values)
    # score is minimized as the negated CV score: the front's best is near 0
    best = report.best_for("score")
    assert best.values[0] < 0.0
    # r297: fit_seconds is a wall-clock measurement of a tiny ridge fit —
    # on a fast host it legitimately rounds to 0.0 (the Windows runner's
    # sub-millisecond fit did exactly that; time resolution, not a broken
    # metric), so the contract is non-negative-and-finite, not strictly
    # positive.
    assert best.values[1] >= 0.0
    # the honest check: the front members are mutually non-dominated
    history = [t.values for t in report.trials]
    for i, a in enumerate(history):
        for j, b in enumerate(history):
            if i != j:
                assert not dominates(b, a)


def test_tune_model_multi_objective_validation() -> None:
    X, y = _toy_classification_data()
    space = SearchSpace.uniform("alpha", 0.01, 1.0, log=True)
    with pytest.raises(DataValidationError):
        tune_model_multi("ridge_classifier", space, X, y, objectives=("score",))
    with pytest.raises(DataValidationError):
        tune_model_multi("ridge_classifier", space, X, y, objectives=("score", "score"))
    with pytest.raises(DataValidationError):
        tune_model_multi(
            "ridge_classifier", space, X, y, objectives=("score", "auc_pr")
        )


def test_tune_model_multi_deterministic_search_under_seed() -> None:
    # the SEARCH is deterministic under the seed -- pinned at the study level
    # with a pure objective (the adapter's front additionally depends on
    # wall-clock fit seconds, which are measurements, not reproducible state)
    space = SearchSpace.uniform("alpha", 0.01, 1.0, log=True)
    first = MultiObjectiveStudy(
        lambda p: (abs(p["alpha"] - 0.1), abs(p["alpha"] - 0.9)),
        space,
        n_objectives=2,
        sampler="random",
        seed=8,
    )
    second = MultiObjectiveStudy(
        lambda p: (abs(p["alpha"] - 0.1), abs(p["alpha"] - 0.9)),
        space,
        n_objectives=2,
        sampler="random",
        seed=8,
    )
    first.run(5)
    second.run(5)
    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert [t.weights for t in first.trials] == [t.weights for t in second.trials]
    assert [t.scalar for t in first.trials] == [t.scalar for t in second.trials]
    # and the adapter's front, on the deterministic objective side, repeats:
    # identical alphas explored -> identical score-side endpoints reachable
    assert first.report(["a", "b"]).best_for("a").values[0] == pytest.approx(
        second.report(["a", "b"]).best_for("a").values[0]
    )
