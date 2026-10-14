"""Tests for regulatory-approval filtering (the ``approved.py`` upgrade).

Covers the DGIdb TSV workflow of the Persian-commented original:
case/whitespace-tolerant column matching, status filtering, sorted
unique drug-name collection across DataFrames, single paths and lists
of paths, plus the single-column CSV export.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.pharma.approvals import filter_approved_drugs

logging.disable(logging.CRITICAL)


def _dgidb_frame() -> pd.DataFrame:
    """DGIdb-style export with mixed-case headers and mixed statuses."""
    return pd.DataFrame(
        {
            "Drug": ["Aspirin", "Ibuprofen", "Warfarin", "  Aspirin  ", "Paracetamol"],
            "Regulatory Approval": [
                "Approved",
                " approved ",
                "Withdrawn",
                "APPROVED",
                "Investigational",
            ],
        }
    )


class TestFilterApprovedDrugs:
    def test_dataframe_input_sorted_unique_names(self) -> None:
        frame = _dgidb_frame()
        result = filter_approved_drugs(frame)
        # "  Aspirin  " deduplicates with "Aspirin" after stripping;
        # status matching is case/whitespace-insensitive
        assert result == ["Aspirin", "Ibuprofen"]

    def test_caller_frame_not_mutated(self) -> None:
        frame = _dgidb_frame()
        filter_approved_drugs(frame)
        assert list(frame.columns) == ["Drug", "Regulatory Approval"]
        assert frame.shape == (5, 2)

    def test_custom_status(self) -> None:
        result = filter_approved_drugs(_dgidb_frame(), approval_status="Withdrawn")
        assert result == ["Warfarin"]

    def test_tsv_path_input(self, tmp_path: Path) -> None:
        export = tmp_path / "1.tsv"
        frame = pd.DataFrame(
            {
                "  Drug ": ["Diazepam", "Naproxen"],
                "Regulatory Approval ": ["Approved", "Withdrawn"],
            }
        )
        frame.to_csv(export, sep="\t", index=False)
        result = filter_approved_drugs(export)
        assert result == ["Diazepam"]

    def test_csv_path_input(self, tmp_path: Path) -> None:
        export = tmp_path / "interaction.csv"
        _dgidb_frame().to_csv(export, index=False)
        assert filter_approved_drugs(export) == ["Aspirin", "Ibuprofen"]

    def test_list_of_paths(self, tmp_path: Path) -> None:
        first = tmp_path / "1.tsv"
        second = tmp_path / "2.tsv"
        pd.DataFrame(
            {"drug": ["Aspirin", "Statins"], "regulatory approval": ["Approved", "Approved"]}
        ).to_csv(first, sep="\t", index=False)
        pd.DataFrame(
            {"drug": ["Statins", "Glitazones"], "regulatory approval": ["Approved", "Rejected"]}
        ).to_csv(second, sep="\t", index=False)
        result = filter_approved_drugs([first, second])
        assert result == ["Aspirin", "Statins"]  # Statins deduplicated across files

    def test_out_path_writes_single_column_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "approved_drugs.csv"
        result = filter_approved_drugs(_dgidb_frame(), out_path=out)
        assert out.is_file()
        written = pd.read_csv(out)
        assert list(written.columns) == ["drug"]
        assert list(written["drug"]) == result == ["Aspirin", "Ibuprofen"]

    def test_missing_column_names_offending_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.tsv"
        pd.DataFrame({"compound": ["x"], "regulatory approval": ["Approved"]}).to_csv(
            bad, sep="\t", index=False
        )
        with pytest.raises(DataValidationError, match="bad.tsv"):
            filter_approved_drugs(bad)

    def test_missing_column_names_dataframe(self) -> None:
        with pytest.raises(DataValidationError, match="input DataFrame"):
            filter_approved_drugs(pd.DataFrame({"drug": ["x"], "status": ["Approved"]}))

    def test_unsupported_input_type_raises(self) -> None:
        with pytest.raises(DataValidationError, match="DataFrame, a path, or a list"):
            filter_approved_drugs({"drug": ["x"]})

    def test_list_with_non_path_element_raises(self, tmp_path: Path) -> None:
        good = tmp_path / "ok.tsv"
        pd.DataFrame({"drug": ["x"], "regulatory approval": ["Approved"]}).to_csv(
            good, sep="\t", index=False
        )
        with pytest.raises(DataValidationError, match="must contain paths"):
            filter_approved_drugs([good, 42])

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            filter_approved_drugs(tmp_path / "ghost.tsv")


class TestMissingSentinels:
    """pd.NA drug cells must be skipped, never collected as '<NA>'."""

    def test_pd_na_drug_cell_skipped(self) -> None:
        frame = pd.DataFrame(
            {
                "Drug": ["Aspirin", pd.NA, "Ibuprofen"],
                "Regulatory Approval": ["Approved", "Approved", "Approved"],
            }
        )
        names = filter_approved_drugs(frame)
        assert names == ["Aspirin", "Ibuprofen"]
        assert "<NA>" not in names

    def test_nat_drug_cell_skipped(self) -> None:
        frame = pd.DataFrame(
            {
                "Drug": ["Warfarin", pd.NaT],
                "Regulatory Approval": ["Approved", "Approved"],
            }
        )
        names = filter_approved_drugs(frame)
        assert names == ["Warfarin"]
        assert "NaT" not in names
