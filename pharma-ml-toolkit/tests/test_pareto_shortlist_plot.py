"""Tests for full-front shortlisting and multi-objective study plots."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml import MultiObjectiveStudy, ParetoShortlist
from pharmaml.ml.hpo import SearchSpace, TrialState
from pharmaml.ml.multiobjective import MOTrial
from pharmaml.viz import close_figure, plot_study_pareto, save_figure


def _study(points):
    dimensions = len(points[0]) if points else 2
    study = MultiObjectiveStudy(lambda _: (0.0,) * dimensions,
                                SearchSpace.uniform("x", 0, 1), n_objectives=dimensions)
    for index, values in enumerate(points):
        study.trials.append(MOTrial(index, {"x": index}, tuple(values),
                                    (1.0,) * dimensions, 0.0))
    return study


def test_prune_keeps_corners_and_largest_crowding_in_study_order():
    study = _study([(0.2, 0.9), (0.9, 0.4), (0.25, 0.85), (0.5, 0.6), (1.0, 1.0)])
    report = study.report(["loss", "cost"])
    original = list(report.trials)
    old_hv = report.hypervolume
    shortlist = report.prune(3)
    assert isinstance(shortlist, ParetoShortlist)
    assert [trial.number for trial in shortlist.trials] == [0, 1, 3]
    assert shortlist.n_front == 4 and shortlist.n_trials == 5
    assert shortlist.crowding == (report.crowding[0], report.crowding[1], report.crowding[3])
    assert report.trials == original and report.hypervolume == old_hv
    assert not hasattr(shortlist, "hypervolume")
    assert list(shortlist.to_dataframe().columns) == ["number", "loss", "cost", "scalar", "crowding"]
    assert shortlist.to_dataframe()["number"].tolist() == [0, 1, 3]


def test_prune_ties_empty_and_excess_keep():
    report = _study([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]).report(["a", "b"])
    assert [t.number for t in report.prune(1).trials] == [0]
    assert len(report.prune(10).trials) == 3
    empty = _study([]).report(["a", "b"]).prune(3)
    assert empty.trials == () and empty.n_front == 0
    assert list(empty.to_dataframe().columns) == ["number", "a", "b", "scalar", "crowding"]
    with pytest.raises(DataValidationError):
        report.crowding = (math.nan, 0.0, 1.0)
        report.prune(1)


@pytest.mark.parametrize("keep", [0, -1, 1.5, True, "2"])
def test_prune_rejects_invalid_keep(keep):
    with pytest.raises(DataValidationError, match="positive integer"):
        _study([(1, 2)]).report().prune(keep)


def test_projection_keeps_full_dimensional_front_and_excludes_failed(tmp_path):
    # Trial 1 appears dominated by trial 0 in x/y, but is better in z.
    study = _study([(0, 0, 2), (1, 1, 0), (2, 2, 3), (4, 4, 4)])
    study.trials[3].state = TrialState.FAIL
    fig = plot_study_pareto(study, "x", "y", objective_names=["x", "y", "z"])
    try:
        ax = fig.axes[0]
        by_label = {item.get_label(): item for item in ax.collections}
        np.testing.assert_allclose(by_label["Full-objective Pareto front (2)"].get_offsets(),
                                   [[0, 0], [1, 1]])
        np.testing.assert_allclose(by_label["Dominated"].get_offsets(), [[2, 2]])
        assert not ax.lines
        assert ax.get_xlabel() == "x (minimize)"
        assert "2 of 3 completed trials" in ax.get_title(loc="left")
        for ext in ("png", "svg", "pdf"):
            target = tmp_path / f"study.{ext}"
            save_figure(fig, target)
            assert target.stat().st_size > 0
    finally:
        close_figure(fig)


def test_plot_two_objectives_defaults_and_empty():
    study = _study([(0, 2), (1, 1), (2, 0)])
    fig = plot_study_pareto(study, "objective_1", "objective_2")
    try:
        assert len(fig.axes[0].collections) == 1
        assert fig.axes[0].get_ylabel() == "objective_2 (minimize)"
    finally:
        close_figure(fig)
    with pytest.raises(DataValidationError, match="no completed"):
        plot_study_pareto(_study([]), "objective_1", "objective_2")


@pytest.mark.parametrize("x,y,names", [("x", "x", ["x", "y"]),
                                        ("x", "z", ["x", "y"]),
                                        ("x", "y", ["x", "x"]),
                                        ("x", "y", ["x"])])
def test_plot_invalid_names(x, y, names):
    with pytest.raises(DataValidationError):
        plot_study_pareto(_study([(1, 2)]), x, y, objective_names=names)
