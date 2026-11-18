"""Descriptor alignment must not silently change applicability results."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.applicability import (
    applicability_domain,
    bootstrap_uncertainty,
    knn_applicability_domain,
    quantile_prediction_intervals,
)


@pytest.fixture
def data():
    X = pd.DataFrame({"mw": np.arange(20.0), "tpsa": np.arange(20.0) ** 2})
    y = X["mw"] * 2 + X["tpsa"]
    return X, y, Ridge().fit(X, y)


def run_analysis(kind, X, y, model, query):
    if kind == "knn":
        return knn_applicability_domain(X, query, k=2)
    if kind == "bootstrap":
        return bootstrap_uncertainty(model, X, y, query, n_models=2)
    if kind == "quantile":
        return quantile_prediction_intervals(X, y, query, n_estimators=2)
    return applicability_domain(model, X, y, query, k=2, n_models=2)


@pytest.mark.parametrize("kind", ["knn", "bootstrap", "quantile", "full"])
@pytest.mark.parametrize("columns", [["tpsa", "mw"], ["mw", "other"]])
def test_reordered_or_renamed_columns_rejected(data, kind, columns):
    X, y, model = data
    query = X.iloc[:3].copy()
    if columns == ["tpsa", "mw"]:
        query = query[columns]
    else:
        query.columns = columns
    with pytest.raises(ModelError, match="feature names and order"):
        run_analysis(kind, X, y, model, query)


@pytest.mark.parametrize("kind", ["knn", "bootstrap", "quantile", "full"])
@pytest.mark.parametrize("duplicate_side", ["train", "test"])
def test_duplicate_dataframe_columns_rejected(data, kind, duplicate_side):
    X, y, model = data
    query = X.iloc[:3].copy()
    if duplicate_side == "train":
        X = X.copy()
        X.columns = ["mw", "mw"]
    else:
        query.columns = ["mw", "mw"]
    with pytest.raises(DataValidationError, match=f"X_{duplicate_side} has duplicate"):
        run_analysis(kind, X, y, model, query)


@pytest.mark.parametrize("kind", ["knn", "bootstrap", "quantile", "full"])
def test_valid_dataframes_and_numpy_inputs(data, kind):
    X, y, model = data
    query = X.iloc[:3].copy()
    run_analysis(kind, X, y, model, query)
    array_model = Ridge().fit(X.to_numpy(), y.to_numpy())
    run_analysis(kind, X.to_numpy(), y.to_numpy(), array_model, query.to_numpy())


def test_mixed_inputs_use_positional_alignment(data):
    X, _, _ = data
    query = X.iloc[:3]
    expected = knn_applicability_domain(X, query, k=2)
    mixed = knn_applicability_domain(X, query.to_numpy(), k=2)
    np.testing.assert_allclose(expected.test_knn_mean, mixed.test_knn_mean)
    mixed_reverse = knn_applicability_domain(X.to_numpy(), query, k=2)
    np.testing.assert_allclose(expected.test_knn_mean, mixed_reverse.test_knn_mean)


def test_column_labels_not_coerced_to_strings(data):
    X, _, _ = data
    X.columns = [1, "1"]
    query = X.iloc[:3].copy()
    knn_applicability_domain(X, query, k=2)
    with pytest.raises(ModelError, match="feature names and order"):
        knn_applicability_domain(X, query.iloc[:, ::-1], k=2)
