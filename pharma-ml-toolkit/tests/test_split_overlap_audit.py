"""Audit exact leakage keys across train/calibration/test partitions."""

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml import audit_split_overlap


def test_three_partitions_occurrences_overlaps_and_duplicates():
    report = audit_split_overlap({
        "train": ["a", "a", "b", "c", "d"],
        "cal": ["b", "e", "e"],
        "test": ["a", "b", "e", "z"],
    })
    assert report.partition_names == ("train", "cal", "test")
    assert report.partition_rows == (5, 3, 4)
    assert report.duplicate_rows == (1, 1, 0)
    assert report.overlap_rows == (3, 3, 3)
    assert report.n_shared_keys == 3 and report.has_overlap
    assert [entry.key for entry in report.shared_keys] == ["a", "b", "e"]
    assert [entry.counts for entry in report.shared_keys] == [(2, 0, 1), (1, 1, 1), (0, 2, 1)]
    np.testing.assert_array_equal(report.pairwise, [[4, 1, 2], [1, 2, 2], [2, 2, 4]])
    assert report.to_dataframe().to_dict("records") == [
        {"key": "a", "n_train": 2, "n_cal": 0, "n_test": 1},
        {"key": "b", "n_train": 1, "n_cal": 1, "n_test": 1},
        {"key": "e", "n_train": 0, "n_cal": 2, "n_test": 1},
    ]
    assert report.pairwise_dataframe().loc["train", "test"] == 2


def test_no_cross_overlap_can_still_have_within_split_duplicates():
    report = audit_split_overlap({"left": ["a", "a", "b"], "right": ["c", "d", "d"]})
    assert not report.has_overlap and report.n_shared_keys == 0
    assert report.overlap_rows == (0, 0) and report.duplicate_rows == (1, 1)
    assert list(report.to_dataframe().columns) == ["key", "n_left", "n_right"]
    assert report.to_dataframe().empty
    np.testing.assert_array_equal(report.pairwise, [[2, 0], [0, 2]])


def test_composite_keys_pandas_numpy_generators_and_key_equality():
    report = audit_split_overlap({
        "train": pd.Series([("p1", 1), ("p2", 2)], index=[99, 42]),
        "test": (x for x in [("p1", 1.0), ("p1", 2)]),
        "external": np.array(["other", "p1"], dtype=object),
    })
    assert report.n_shared_keys == 1
    assert report.shared_keys[0].key == ("p1", 1)
    assert report.shared_keys[0].counts == (1, 1, 0)
    assert report.to_dataframe().iloc[0]["key"] == ("p1", 1)
    assert report.partition_rows == (2, 2, 2)


def test_no_string_normalization_or_canonicalization():
    report = audit_split_overlap({"train": ["CCO", " x", "A"], "test": ["OCC", "x", "a"]})
    assert not report.has_overlap


@pytest.mark.parametrize("parts", [None, {}, {"only": ["x"]}, {"": ["x"], "test": ["x"]},
                                   {3: ["x"], "test": ["x"]}])
def test_invalid_partition_names_and_mapping(parts):
    with pytest.raises(DataValidationError, match="partition"):
        audit_split_overlap(parts)


@pytest.mark.parametrize("bad", [[], iter(()), "CCO", b"CCO", None,
                                  pd.DataFrame({"x": ["a"]}), np.array([["a"]]),
                                  {"x": 1}, 42])
def test_invalid_or_empty_partition_sequence(bad):
    with pytest.raises(DataValidationError, match="partition 'test'"):
        audit_split_overlap({"train": ["valid"], "test": bad})


@pytest.mark.parametrize("key", [None, np.nan, np.inf, -np.inf, pd.NA, "", "  ", b"", [],
                                  {"k": 1}, (), ("a", None), ("a", np.nan)])
def test_invalid_keys_are_never_silently_ignored(key):
    with pytest.raises(DataValidationError, match="partition 'test' row 0"):
        audit_split_overlap({"train": ["valid"], "test": [key]})


def test_consistent_results_under_row_reordering():
    first = audit_split_overlap({"a": ["x", "y", "y", "z"], "b": ["z", "y", "q"]})
    second = audit_split_overlap({"a": ["z", "y", "x", "y"], "b": ["q", "y", "z"]})
    assert first.partition_rows == second.partition_rows
    assert first.duplicate_rows == second.duplicate_rows
    assert first.overlap_rows == second.overlap_rows
    np.testing.assert_array_equal(first.pairwise, second.pairwise)
    assert {x.key: x.counts for x in first.shared_keys} == {x.key: x.counts for x in second.shared_keys}


def test_large_repeated_key_does_not_create_quadratic_row_pairs():
    report = audit_split_overlap({"train": ["same"] * 10_000, "test": ["same"] * 7_000})
    assert report.n_shared_keys == 1
    assert report.overlap_rows == (10_000, 7_000)
    assert report.duplicate_rows == (9_999, 6_999)
    assert report.pairwise[0, 1] == 1


def test_unhashable_numeric_subclass_is_rejected_with_typed_error():
    class UnhashableInt(int):
        __hash__ = None

    with pytest.raises(DataValidationError, match="partition 'test' row 0.*unhashable"):
        audit_split_overlap({"train": [1], "test": [UnhashableInt(1)]})
