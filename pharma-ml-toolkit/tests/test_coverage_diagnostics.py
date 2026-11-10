"""Held-out coverage by group, Wilson scores and conformal integration."""

import math

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml import (
    calibrate_split_conformal,
    calibrate_split_conformal_classifier,
    coverage_by_group,
)


def test_wilson_known_values_overall_and_by_group():
    covered = [True] * 5 + [False] * 5
    groups = ["first"] * 5 + ["second"] * 5
    report = coverage_by_group(covered, groups, sizes=list(range(10)))
    overall = report.overall
    assert (overall.n_test, overall.n_covered, overall.coverage) == (10, 5, 0.5)
    assert overall.lower == pytest.approx(0.2365930905)
    assert overall.upper == pytest.approx(0.7634069095)
    assert overall.mean_size == 4.5
    assert [r.group for r in report.groups] == ["first", "second"]
    assert report.groups[0].coverage == 1
    assert report.groups[0].lower == pytest.approx(0.5655175352)
    assert report.groups[0].upper == pytest.approx(1.0)
    assert report.groups[0].mean_size == 2
    assert report.groups[1].coverage == 0
    assert report.groups[1].lower == pytest.approx(0)
    assert report.groups[1].upper == pytest.approx(0.4344824648)
    frame = report.to_dataframe()
    assert list(frame.columns) == ["group", "n_test", "n_covered", "coverage", "lower", "upper", "mean_size"]
    assert frame.group.tolist() == groups[::5]


def test_first_seen_groups_and_positional_pandas_alignment():
    hits = pd.Series([True, False, True, False], index=[4, 3, 2, 1])
    group = pd.Series(["z", "x", "z", "x"], index=[1, 2, 3, 4])
    report = coverage_by_group(hits, group)
    assert [r.group for r in report.groups] == ["z", "x"]
    assert [r.coverage for r in report.groups] == [1, 0]
    assert report.groups[0].mean_size is None
    assert report.confidence == 0.95


def test_confidence_controls_interval_and_single_observation():
    first = coverage_by_group([True, False], [0, 0], confidence=0.8).overall
    second = coverage_by_group([True, False], [0, 0], confidence=0.99).overall
    assert second.lower < first.lower < 0.5 < first.upper < second.upper
    singleton = coverage_by_group([False], ["one"]).overall
    assert singleton.coverage == 0 and 0 <= singleton.lower <= singleton.upper < 1
    highest = np.nextafter(1.0, 0.0)
    assert coverage_by_group([True], ["a"], confidence=highest).overall.upper <= 1


def test_large_sizes_do_not_overflow_mean():
    report = coverage_by_group([True, False, True], ["a", "a", "b"],
                               sizes=[1e308, 1e308, 0])
    assert math.isfinite(report.overall.mean_size)
    assert report.groups[0].mean_size == pytest.approx(1e308)
    assert report.groups[1].mean_size == 0


@pytest.mark.parametrize("hits", [[], [0, 1], [True, None], ["True"], [[True], [False]]])
def test_invalid_coverage_vector(hits):
    with pytest.raises(DataValidationError, match="covered"):
        coverage_by_group(hits, ["a"] * len(hits))


@pytest.mark.parametrize("groups", [[], ["a"], ["a", None], ["a", float("nan")],
                                   ["a", float("inf")], [["a"], ["b"]]])
def test_invalid_groups(groups):
    with pytest.raises(DataValidationError, match="groups"):
        coverage_by_group([True, False], groups)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, True, "0.9", float("inf"), float("nan"), 10 ** 400])
def test_invalid_confidence(confidence):
    with pytest.raises(DataValidationError, match="confidence"):
        coverage_by_group([True, False], [0, 1], confidence=confidence)


@pytest.mark.parametrize("sizes", [[], [1], [1, -1], [1, float("nan")], [1, float("inf")],
                                  [True, 2], ["1", 2], [[1], [2]], [1, 10 ** 400]])
def test_invalid_sizes(sizes):
    with pytest.raises(DataValidationError, match="sizes"):
        coverage_by_group([True, False], [0, 1], sizes=sizes)


def test_regression_results_expose_row_coverage_without_dataframe_break():
    class Zero:
        def predict(self, X):
            return np.zeros(len(X))

    cal = calibrate_split_conformal(Zero(), [[0]] * 9, [1] * 9, alpha=0.1)
    result = cal.predict_interval([[0]] * 4, [1, -1, 2, 0])
    np.testing.assert_array_equal(result.covered, [True, True, False, True])
    assert result.coverage == 0.75 and result.n_covered == 3
    report = coverage_by_group(result.covered, ["x", "x", "y", "y"],
                               sizes=result.upper - result.lower)
    assert [r.coverage for r in report.groups] == [1, 0.5]
    assert report.overall.coverage == result.coverage and report.overall.mean_size == 2
    assert list(result.to_dataframe().columns) == ["prediction", "lower", "upper"]
    assert cal.predict_interval([[0]]).covered is None


def test_classification_sets_integrate_with_report():
    class Model:
        classes_ = ["a", "b"]

        def predict_proba(self, X):
            return np.asarray(X)

    cal = calibrate_split_conformal_classifier(Model(), [[0.9, 0.1]] * 9, ["a"] * 9)
    result = cal.predict_set([[0.9, 0.1], [0.9, 0.1]], ["a", "b"])
    report = coverage_by_group(result.covered, ["batch-1", "batch-2"], sizes=result.set_sizes)
    assert [r.coverage for r in report.groups] == [1, 0]
    assert report.overall.coverage == result.coverage
    assert report.overall.mean_size == result.mean_set_size
