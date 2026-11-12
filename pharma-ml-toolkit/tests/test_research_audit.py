"""Dataset audit describes exact findings without claiming independence."""
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.research_audit import audit_dataset


def test_audit_exact_overlap_missing_duplicate_and_balance():
    frame = pd.DataFrame({"id": ["private", "private", "other", "third"],
                          "split": ["train", "test", "test", "train"],
                          "label": ["yes", "no", None, "yes"],
                          "value": [1, 1, None, 2]}, index=[9, 9, 4, 1])
    report = audit_dataset(frame, key="id", partition="split", target="label")
    assert report.rows == 4 and report.columns == 4
    assert report.missing_by_column["value"] == 1 and report.duplicate_keys == 1
    assert report.shared_keys == 1 and report.overlap_rows == {"train": 1, "test": 1}
    assert dict(report.class_counts) == {"'yes'": 2, "'no'": 1, "<missing>": 1}
    assert "private" not in repr(report)
    assert "cannot rule out" in report.limitations


def test_no_identifier_audit_is_explicitly_not_run():
    result = audit_dataset(pd.DataFrame({"x": [1, 1]}))
    assert result.duplicate_rows == 1 and result.shared_keys is None
    assert result.duplicate_keys is None and result.class_counts is None


@pytest.mark.parametrize("key", [None, "", "  "])
def test_missing_key_rejected_for_split_audit(key):
    frame = pd.DataFrame({"id": ["a", key], "split": ["train", "test"]})
    with pytest.raises(DataValidationError, match="known key"):
        audit_dataset(frame, key="id", partition="split")


def test_missing_keys_counted_without_split():
    frame = pd.DataFrame({"id": ["a", None, " "]})
    assert audit_dataset(frame, key="id").missing_keys == 2


@pytest.mark.parametrize("labels", [["train", None], ["train", ""], ["train", "train"]])
def test_invalid_partition_labels_or_single_partition_rejected(labels):
    with pytest.raises(DataValidationError):
        audit_dataset(pd.DataFrame({"id": ["a", "b"], "split": labels}), key="id", partition="split")


def test_duplicate_column_names_and_unknown_columns_fail():
    with pytest.raises(DataValidationError, match="column names"):
        audit_dataset(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(DataValidationError, match="key column"):
        audit_dataset(pd.DataFrame({"x": [1]}), key="unknown")
