"""Separate-file audit uses literal identifiers and the real overlap engine."""
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.file_split_audit import audit_partition_files


def test_literal_ids_composite_keys_and_counts(tmp_path):
    train, test = tmp_path / "train.csv", tmp_path / "test.tsv"
    train.write_text("id,site\n001,A\nNA,B\nSECRET,C\nSECRET,C\n", encoding="utf-8")
    test.write_text("id\tsite\n1\tA\nNA\tB\nSECRET\tC\n", encoding="utf-8")
    report = audit_partition_files({"train": train, "test": test}, ["id", "site"])
    assert report.n_shared_keys == 2
    assert report.duplicate_rows == (1, 0)
    assert report.overlap_rows == (3, 2)
    assert report.pairwise.tolist() == [[3, 2], [2, 3]]


def test_excel_and_delimiter(tmp_path):
    a, b = tmp_path / "a.xlsx", tmp_path / "b.csv"
    pd.DataFrame({"id": ["001", "NA"]}).to_excel(a, index=False)
    b.write_text("id;value\n1;x\n", encoding="utf-8")
    report = audit_partition_files({"train": a, "test": b}, ["id"], "semicolon")
    assert not report.has_overlap


def test_fail_closed(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("id\n001\n", encoding="utf-8")
    b.write_text("id\n\n", encoding="utf-8")
    for partitions, keys in [({"train": a}, ["id"]),
                             ({"bad name": a, "test": b}, ["id"]),
                             ({"train": a, "test": b}, []),
                             ({"train": a, "test": b}, ["id", "id"]),
                             ({"train": a, "test": b}, ["missing"]),
                             ({"train": a, "test": b}, ["id"])]:
        with pytest.raises(DataValidationError):
            audit_partition_files(partitions, keys)
    assert a.read_text(encoding="utf-8") == "id\n001\n"
