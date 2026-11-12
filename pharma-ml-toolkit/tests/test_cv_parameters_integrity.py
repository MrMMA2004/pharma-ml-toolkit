"""Cross-validation must evaluate the parameters it reports."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from pharmaml.exceptions import ModelError
from pharmaml.ml import cv_results


@pytest.fixture
def dataset():
    X = pd.DataFrame({"x": np.linspace(-2, 2, 30), "z": np.sin(np.arange(30))})
    y = 4 * X["x"].to_numpy() + 0.5 * X["z"].to_numpy()
    return X, y


def test_reported_parameters_are_applied_in_each_fold(dataset, monkeypatch):
    X, y = dataset
    observed = []
    original = Ridge.fit
    def spy(self, X, y, **kwargs):
        observed.append(self.alpha)
        return original(self, X, y, **kwargs)
    monkeypatch.setattr(Ridge, "fit", spy)
    result = cv_results.cross_validate_model(
        X, y, "regression", "ridge", n_folds=3, best_params={"alpha": 1000.0}
    )
    assert observed == [1000.0] * 3
    assert result.best_params == {"alpha": 1000.0}
    assert any("nested CV" in note for note in result.notes)


def test_parameterized_cv_scores_match_independent_fold_calculation(dataset):
    X, y = dataset
    result = cv_results.cross_validate_model(
        X, y, "regression", "ridge", n_folds=3, seed=42, best_params={"alpha": 1000.0}
    )
    pipeline = cv_results._build_pipeline(X, "ridge", "regression")
    pipeline.set_params(est__alpha=1000.0)
    expected = []
    for train, val in KFold(n_splits=3, shuffle=True, random_state=42).split(X):
        fitted = clone(pipeline).fit(X.iloc[train], y[train])
        expected.append(fitted.score(X.iloc[val], y[val]))
    np.testing.assert_allclose(result.metrics["r2"].folds, expected)
    default = cv_results.cross_validate_model(X, y, "regression", "ridge", n_folds=3)
    assert not np.allclose(result.metrics["r2"].folds, default.metrics["r2"].folds)


def test_unknown_parameters_are_rejected_before_fitting(dataset):
    X, y = dataset
    with pytest.raises(ModelError, match="parameter"):
        cv_results.cross_validate_model(X, y, "regression", "ridge", n_folds=3,
                                        best_params={"does_not_exist": 123})


def test_duplicate_parameter_spellings_are_rejected(dataset):
    X, y = dataset
    with pytest.raises(ModelError, match="duplicate"):
        cv_results.cross_validate_model(X, y, "regression", "ridge", n_folds=3,
                                        best_params={"alpha": 1, "est__alpha": 100})


def test_estimator_prefixed_parameter_is_applied(dataset):
    X, y = dataset
    bare = cv_results.cross_validate_model(X, y, "regression", "ridge", n_folds=3,
                                          best_params={"alpha": 1000.0})
    prefixed = cv_results.cross_validate_model(X, y, "regression", "ridge", n_folds=3,
                                              best_params={"est__alpha": 1000.0})
    np.testing.assert_allclose(bare.metrics["r2"].folds, prefixed.metrics["r2"].folds)
