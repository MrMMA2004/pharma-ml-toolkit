"""Regression tests for the actual grouping used by scaffold cross-validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.cv_results import build_cv_splitter, compare_models_cv, cross_validate_model
from pharmaml.ml.splits import scaffold_groups

RING_SMILES = ["c1ccccc1O", "c1ccncc1O", "c1ccc2ccccc2c1O", "c1ccoc1C"] * 12


def test_scaffold_cv_uses_scaffolds_even_with_unrelated_groups(monkeypatch):
    """A misleading `scaffold` report must not be built from user group IDs."""
    X = pd.DataFrame({"x": np.arange(len(RING_SMILES), dtype=float)})
    y = np.arange(len(RING_SMILES), dtype=float)
    unrelated = [f"row-{i}" for i in range(len(y))]
    expected = np.asarray(scaffold_groups(RING_SMILES))
    seen = []
    original_split = GroupKFold.split

    def capture(self, X, y=None, groups=None):
        seen.append(np.asarray(groups).copy())
        for train, val in original_split(self, X, y, groups):
            assert set(expected[train]).isdisjoint(set(expected[val]))
            yield train, val

    monkeypatch.setattr(GroupKFold, "split", capture)
    result = cross_validate_model(X, y, "regression", "ridge", strategy="scaffold",
                                  n_folds=4, smiles=RING_SMILES, groups=unrelated)
    assert result.strategy == "scaffold"
    assert any("ignores groups" in note for note in result.notes)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], expected)


def test_scaffold_cv_requires_smiles_aligned_with_targets():
    with pytest.raises(DataValidationError, match="one SMILES per target row"):
        build_cv_splitter("regression", strategy="scaffold", n_folds=3,
                          y=range(24), smiles=RING_SMILES)
    X = pd.DataFrame({"x": np.arange(24, dtype=float)})
    with pytest.raises(DataValidationError, match="one SMILES per target row"):
        cross_validate_model(X, np.arange(24), "regression", "ridge",
                             strategy="scaffold", n_folds=3, smiles=RING_SMILES)



def test_scaffold_comparison_and_plain_group_cv_use_correct_keys(monkeypatch):
    X = pd.DataFrame({"x": np.arange(len(RING_SMILES), dtype=float)})
    y = np.arange(len(RING_SMILES), dtype=float)
    groups = [f"g{i % 6}" for i in range(len(y))]
    expected_scaffolds = np.asarray(scaffold_groups(RING_SMILES))
    calls = []
    original_split = GroupKFold.split

    def capture(self, X, y=None, groups=None):
        calls.append(np.asarray(groups).copy())
        yield from original_split(self, X, y, groups)

    monkeypatch.setattr(GroupKFold, "split", capture)
    results, _ = compare_models_cv(X, y, "regression", ["ridge", "lasso"],
                                   strategy="scaffold", n_folds=4,
                                   smiles=RING_SMILES, groups=groups)
    assert len(results) == 2 and all(r.strategy == "scaffold" for r in results)
    assert all(any("ignores groups" in note for note in r.notes) for r in results)
    assert len(calls) == 2
    for observed in calls:
        np.testing.assert_array_equal(observed, expected_scaffolds)

    calls.clear()
    cross_validate_model(X, y, "regression", "ridge", strategy="group", n_folds=3,
                         groups=groups, smiles=RING_SMILES)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], np.asarray(groups))
