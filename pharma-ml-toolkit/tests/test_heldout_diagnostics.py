"""Checks on calculations and train-only distance references."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.heldout_diagnostics import (
    classification_probability_diagnostics,
    distance_domain_diagnostics,
)
from pharmaml.ml.preprocessing import build_preprocessor


def test_brier_and_fixed_bins_including_exact_edges():
    p = [[0, 1], [0.5, 0.5], [1, 0]]
    result = classification_probability_diagnostics(p, ["B", "A", "B"], ["A", "B"], bins=2)
    assert result["multiclass_brier_sum"] == pytest.approx((0 + 0.5 + 2) / 3)
    rows = result["per_class"][0]["bins"]
    assert rows[0] == {"lower_inclusive": 0, "upper": 0.5, "upper_inclusive": False,
                       "n": 1, "mean_probability": 0, "observed_fraction": 0}
    assert rows[1]["n"] == 2 and rows[1]["upper_inclusive"]
    assert rows[1]["mean_probability"] == 0.75 and rows[1]["observed_fraction"] == 0.5


@pytest.mark.parametrize("prob,labels", [([[0.2, 0.2]], ["A"]), ([[float("nan"), 0]], ["A"]),
                                         ([[0.2, 0.8]], ["unknown"]), ([[0.2, 0.8]], [])])
def test_invalid_probability_inputs_fail(prob, labels):
    with pytest.raises(DataValidationError):
        classification_probability_diagnostics(prob, labels, ["A", "B"])


def test_domain_threshold_uses_train_only_and_imputer_is_not_refit():
    train = pd.DataFrame({"x": [0., 1, 2, 3, np.nan, 5, 6, 7, 8, 9]})
    test = pd.DataFrame({"x": [3., 500.]})
    model = Pipeline([("prep", build_preprocessor()), ("model", Ridge())]).fit(train, range(len(train)))
    before = model.named_steps["prep"].named_steps["impute"].statistics_.copy()
    report = distance_domain_diagnostics(model, train, test, ["x"], covered=[True, False])
    other = distance_domain_diagnostics(model, train, pd.DataFrame({"x": [999., 2000.]}), ["x"],
                                        covered=[False, False])
    assert report["distance_threshold"] == other["distance_threshold"]
    assert report["n_inside"] == 1
    assert report["coverage_by_distance"][0]["coverage"] == 1
    assert report["coverage_by_distance"][1]["coverage"] == 0
    assert np.array_equal(before, model.named_steps["prep"].named_steps["impute"].statistics_)


def test_domain_insufficient_train_is_not_silently_reconfigured():
    train = pd.DataFrame({"x": [0., 1.]})
    model = Pipeline([("prep", build_preprocessor()), ("model", Ridge())]).fit(train, [0, 1])
    assert distance_domain_diagnostics(model, train, train, ["x"], covered=[True, True])["status"] == "unavailable"
