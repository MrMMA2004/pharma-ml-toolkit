"""Grouped CV must not accept unknown leakage-unit labels."""
import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.cv_results import build_cv_splitter, cross_validate_model


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, "", "   "])
@pytest.mark.parametrize("strategy", ["group", "stratified_group"])
def test_grouped_splitter_rejects_missing_or_blank_labels(strategy, missing):
    groups = ["a", "a", "b", "b", "c", missing]
    with pytest.raises(DataValidationError, match="missing|blank"):
        build_cv_splitter("classification", strategy=strategy, n_folds=2,
                          y=[0, 1, 0, 1, 0, 1], groups=groups)


def test_group_length_checked_before_splitter_returned():
    with pytest.raises(DataValidationError, match="one label per row"):
        build_cv_splitter("regression", strategy="group", n_folds=2,
                          y=[0, 1, 2, 3], groups=["a", "b", "c"])


def test_grouped_cv_rejects_blank_labels_before_model_fit(monkeypatch):
    from sklearn.linear_model import Ridge
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before fitting")
    monkeypatch.setattr(Ridge, "fit", forbidden)
    with pytest.raises(DataValidationError, match="missing|blank"):
        cross_validate_model(pd.DataFrame({"x": range(6)}), range(6),
                             "regression", "ridge", strategy="group", n_folds=2,
                             groups=["a", "a", "b", "b", "", ""])
