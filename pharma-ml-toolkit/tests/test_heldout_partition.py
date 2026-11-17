"""Disjoint calibration/evaluation and subgroup-alignment regression tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.heldout_partition import aligned_group_values, split_calibration_evaluation


def test_disjoint_and_deterministic_with_retained_row_alignment():
    X = pd.DataFrame({"feature": range(23)}, index=range(100, 123))
    y = np.arange(23) * 2
    x_cal, x_eval, y_cal, y_eval = split_calibration_evaluation(X, y)
    assert len(x_cal) + len(x_eval) == len(X)
    assert not set(x_cal.index) & set(x_eval.index)
    assert set(x_cal.index) | set(x_eval.index) == set(X.index)
    np.testing.assert_array_equal(y_cal, x_cal["feature"].to_numpy() * 2)
    np.testing.assert_array_equal(y_eval, x_eval["feature"].to_numpy() * 2)
    again = split_calibration_evaluation(X, y)
    assert x_eval.index.equals(again[1].index)


@pytest.mark.parametrize("X,y", [(np.ones((2, 1)), [1]), ([[1]], [1])])
def test_invalid_split_refused(X, y):
    with pytest.raises(DataValidationError, match="cannot partition"):
        split_calibration_evaluation(X, y)


def test_group_labels_follow_scored_index_order_and_original_values():
    frame = pd.DataFrame({"f": [10., 20., 30., 40.], "label": [0, 1, 1, 0],
                          "group": ["A", "B", "C", "D"]}, index=[11, 12, 13, 14])
    x = frame.loc[[14, 11, 13], ["f"]]
    assert aligned_group_values(frame, x, [0, 0, 1], "label", "group").tolist() == ["D", "A", "C"]


def test_changed_or_missing_data_and_unverifiable_indices_are_refused():
    frame = pd.DataFrame({"f": [10, 20, 30], "label": [0, 1, 0], "group": ["A", "B", "C"]})
    x = frame.loc[[2, 0], ["f"]].copy()
    y = [0, 0]
    changed = frame.copy()
    changed.loc[2, "f"] = 31
    with pytest.raises(DataValidationError, match="features or target"):
        aligned_group_values(changed, x, y, "label", "group")
    changed = frame.copy()
    changed.loc[0, "label"] = 1
    with pytest.raises(DataValidationError, match="features or target"):
        aligned_group_values(changed, x, y, "label", "group")
    for bad_frame, bad_x in ((frame, x.to_numpy()), (frame, pd.concat([x, x])),
                             (pd.concat([frame, frame]), x), (frame.iloc[:1], x)):
        with pytest.raises(DataValidationError, match="alignment"):
            aligned_group_values(bad_frame, bad_x, y, "label", "group")
