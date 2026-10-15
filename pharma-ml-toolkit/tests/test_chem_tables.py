"""Tests for ligand-table reconciliation, SMILES annotation and merging.

Mirrors the upgrade of the five rar table utilities: ``notcommon.py``
(set differences with ``.pdbqt`` extension stripping), ``common_between_
two_excel.py`` (the intersection), ``smilesgetter.py`` (SMILES lookup
annotation of difference columns), ``CSV_merger.py`` (row-wise concat
with named-columns-first reordering) and ``filemerger.py`` (folder
merge that skips ``~$`` Excel lock files and always writes the result).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from pharmaml.chem.tables import (
    TableComparison,
    annotate_with_smiles,
    compare_tables,
    merge_table_folder,
    merge_tables,
)
from pharmaml.exceptions import DataValidationError

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# compare_tables -- notcommon.py + common_between_two_excel.py
# ---------------------------------------------------------------------------


class TestCompareTables:
    def test_path_inputs_full_reconciliation(self, tmp_path: Path) -> None:
        docking = tmp_path / "d_summary.csv"
        ligands = tmp_path / "ligands.csv"
        pd.DataFrame({"Ligand": ["lig1.pdbqt", "LIG2", " lig3 "]}).to_csv(
            docking, index=False
        )
        pd.DataFrame({"Ligand": ["lig1", "lig2", "lig4"]}).to_csv(ligands, index=False)

        result = compare_tables(docking, ligands, "Ligand")

        assert isinstance(result, TableComparison)
        assert result.column == "Ligand"
        # .pdbqt stripped, case folded, whitespace trimmed
        assert result.common == ["lig1", "lig2"]
        assert result.only_in_a == ["lig3"]
        assert result.only_in_b == ["lig4"]

    def test_summary_counts(self) -> None:
        frame_a = pd.DataFrame({"name": ["a", "b", "c"]})
        frame_b = pd.DataFrame({"name": ["b", "c", "d", "e"]})
        result = compare_tables(frame_a, frame_b, "name")
        assert result.summary() == {
            "common": 2,
            "only_in_a": 1,
            "only_in_b": 2,
        }

    def test_without_normalization_names_differ(self) -> None:
        frame_a = pd.DataFrame({"name": ["LIG1.pdbqt"]})
        frame_b = pd.DataFrame({"name": ["lig1"]})
        raw = compare_tables(frame_a, frame_b, "name", normalize=False)
        assert raw.common == []
        assert raw.only_in_a == ["LIG1.pdbqt"]
        assert raw.only_in_b == ["lig1"]

    def test_normalization_flags(self) -> None:
        frame_a = pd.DataFrame({"name": ["Lig1.pdbqt", "LIG2"]})
        frame_b = pd.DataFrame({"name": ["LIG1", "LIG2"]})
        # strip extension but keep case: only the exactly-equal pair matches
        kept = compare_tables(frame_a, frame_b, "name", lowercase=False)
        assert kept.common == ["LIG2"]
        assert kept.only_in_a == ["Lig1"]
        assert kept.only_in_b == ["LIG1"]
        # fold case but keep extension: the .pdbqt suffix now blocks the match
        folded = compare_tables(frame_a, frame_b, "name", strip_extension=False)
        assert folded.common == ["lig2"]
        assert folded.only_in_a == ["lig1.pdbqt"]
        assert folded.only_in_b == ["lig1"]

    def test_xlsx_path_input(self, tmp_path: Path) -> None:
        first = tmp_path / "file1.xlsx"
        second = tmp_path / "file2.xlsx"
        pd.DataFrame({"Ligand": ["shared", "only-a"]}).to_excel(first, index=False)
        pd.DataFrame({"Ligand": ["shared", "only-b"]}).to_excel(second, index=False)
        result = compare_tables(first, second, "Ligand")
        assert result.common == ["shared"]
        assert result.only_in_a == ["only-a"]
        assert result.only_in_b == ["only-b"]

    def test_missing_column_names_offending_dataframe(self) -> None:
        frame_a = pd.DataFrame({"wrong": ["x"]})
        frame_b = pd.DataFrame({"Ligand": ["y"]})
        with pytest.raises(DataValidationError, match="table_a"):
            compare_tables(frame_a, frame_b, "Ligand")

    def test_missing_column_names_offending_file(self, tmp_path: Path) -> None:
        frame_a = pd.DataFrame({"Ligand": ["y"]})
        bad_file = tmp_path / "ligands.csv"
        bad_file.write_text("Compound\nx\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="ligands.csv"):
            compare_tables(frame_a, bad_file, "Ligand")

    def test_unsupported_input_type_raises(self) -> None:
        with pytest.raises(DataValidationError, match="DataFrame or a path"):
            compare_tables(12345, pd.DataFrame({"Ligand": []}), "Ligand")


# ---------------------------------------------------------------------------
# annotate_with_smiles -- smilesgetter.py
# ---------------------------------------------------------------------------


class TestAnnotateWithSmiles:
    def _reference(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Ligand": ["lig1.pdbqt", "LIG2", "lig3"],
                "SMILES": ["CCO", "c1ccccc1", "CC(=O)O"],
            }
        )

    def test_two_name_columns_annotated(self) -> None:
        differences = pd.DataFrame(
            {
                "Ligand1": ["lig1", "lig2", "unknown"],
                "Ligand2": ["LIG3", "ghost", "lig1.PDBQT"],
            }
        )
        result = annotate_with_smiles(
            differences, self._reference(), ["Ligand1", "Ligand2"]
        )
        assert list(result["Ligand1_SMILES"]) == ["CCO", "c1ccccc1", ""]
        assert list(result["Ligand2_SMILES"]) == ["CC(=O)O", "", "CCO"]
        # original columns ride along untouched
        assert list(result["Ligand1"]) == ["lig1", "lig2", "unknown"]

    def test_reference_path_input(self, tmp_path: Path) -> None:
        ref_path = tmp_path / "ligands.csv"
        self._reference().to_csv(ref_path, index=False)
        frame = pd.DataFrame({"Ligand1": ["LIG2"]})
        result = annotate_with_smiles(frame, ref_path, ["Ligand1"])
        assert result.loc[0, "Ligand1_SMILES"] == "c1ccccc1"

    def test_input_frame_not_mutated(self) -> None:
        frame = pd.DataFrame({"Ligand1": ["lig1"]})
        annotate_with_smiles(frame, self._reference(), ["Ligand1"])
        assert list(frame.columns) == ["Ligand1"]

    def test_missing_reference_column_raises(self) -> None:
        frame = pd.DataFrame({"Ligand1": ["lig1"]})
        bad_reference = pd.DataFrame({"Ligand": ["lig1"], "smi": ["CCO"]})
        with pytest.raises(DataValidationError, match="SMILES"):
            annotate_with_smiles(frame, bad_reference, ["Ligand1"])

    def test_missing_name_column_in_frame_raises(self) -> None:
        frame = pd.DataFrame({"other": ["lig1"]})
        with pytest.raises(DataValidationError, match="name column"):
            annotate_with_smiles(frame, self._reference(), ["Ligand1"])


# ---------------------------------------------------------------------------
# merge_tables -- CSV_merger.py
# ---------------------------------------------------------------------------


class TestMergeTables:
    def _frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame_a = pd.DataFrame(
            {"pIC50": [1.0], "SMILES": ["CCO"], "State": ["solid"], "zz_col": [1]}
        )
        frame_b = pd.DataFrame(
            {"pEC50": [2.0], "SMILES": ["CCC"], "State": ["liquid"], "aa_col": [2]}
        )
        return frame_a, frame_b

    def test_first_columns_first_rest_sorted(self) -> None:
        merged = merge_tables(
            list(self._frames()), first_columns=["SMILES", "State", "not_there"]
        )
        assert list(merged.columns) == ["SMILES", "State", "aa_col", "pEC50", "pIC50", "zz_col"]
        assert len(merged) == 2

    def test_sort_remaining_false_keeps_concat_order(self) -> None:
        merged = merge_tables(
            list(self._frames()),
            first_columns=["SMILES"],
            sort_remaining=False,
        )
        # remainder keeps first-seen concat order: pIC50, State, zz_col, pEC50, aa_col
        assert list(merged.columns) == [
            "SMILES",
            "pIC50",
            "State",
            "zz_col",
            "pEC50",
            "aa_col",
        ]

    def test_no_first_columns_plain_concat(self) -> None:
        merged = merge_tables(list(self._frames()))
        assert len(merged) == 2
        assert merged.index.tolist() == [0, 1]  # ignore_index parity

    def test_path_inputs(self, tmp_path: Path) -> None:
        first = tmp_path / "pic50.csv"
        second = tmp_path / "pec50.csv"
        pd.DataFrame({"SMILES": ["CCO"], "pIC50": [1.0]}).to_csv(first, index=False)
        pd.DataFrame({"SMILES": ["CCC"], "pEC50": [2.0]}).to_csv(second, index=False)
        merged = merge_tables([first, second], first_columns=["SMILES"])
        assert list(merged.columns) == ["SMILES", "pEC50", "pIC50"]
        assert list(merged["SMILES"]) == ["CCO", "CCC"]

    def test_single_table_returns_copy(self) -> None:
        original = pd.DataFrame({"a": [1], "b": [2]})
        merged = merge_tables([original])
        merged["a"] = 999
        assert original.loc[0, "a"] == 1

    def test_empty_list_raises(self) -> None:
        with pytest.raises(DataValidationError, match="at least one"):
            merge_tables([])


# ---------------------------------------------------------------------------
# merge_table_folder -- filemerger.py
# ---------------------------------------------------------------------------


class TestMergeTableFolder:
    def _make_files(self, folder: Path) -> None:
        pd.DataFrame({"SMILES": ["CCO"], "value": [1]}).to_csv(
            folder / "a.csv", index=False
        )
        pd.DataFrame({"SMILES": ["CCC"], "value": [2]}).to_csv(
            folder / "b.csv", index=False
        )
        pd.DataFrame({"SMILES": ["CCCO"], "value": [3]}).to_csv(
            folder / "c.csv", index=False
        )

    def test_merges_all_csvs_and_writes_output(self, tmp_path: Path) -> None:
        self._make_files(tmp_path)
        merged = merge_table_folder(tmp_path)
        assert list(merged["value"]) == [1, 2, 3]
        output = tmp_path / "merged_output.csv"
        assert output.is_file()
        assert list(pd.read_csv(output)["SMILES"]) == ["CCO", "CCC", "CCCO"]

    def test_skips_temp_and_hidden_files(self, tmp_path: Path) -> None:
        self._make_files(tmp_path)
        (tmp_path / "~$excel_lock.csv").write_text("SMILES,value\nbad,9\n", encoding="utf-8")
        (tmp_path / ".partial.csv").write_text("SMILES,value\nbad,8\n", encoding="utf-8")
        merged = merge_table_folder(tmp_path, first_columns=["SMILES"])
        assert list(merged["value"]) == [1, 2, 3]

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(DataValidationError, match="No .csv files"):
            merge_table_folder(empty)

    def test_custom_out_path(self, tmp_path: Path) -> None:
        self._make_files(tmp_path)
        out = tmp_path / "out" / "combined.csv"
        merged = merge_table_folder(tmp_path, out_path=out)
        assert out.is_file()
        assert list(merged["value"]) == [1, 2, 3]

    def test_rerun_does_not_ingest_own_output(self, tmp_path: Path) -> None:
        self._make_files(tmp_path)
        first = merge_table_folder(tmp_path)  # writes merged_output.csv inside
        second = merge_table_folder(tmp_path)  # must skip the merged output
        assert list(first["value"]) == list(second["value"]) == [1, 2, 3]

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        ghost = tmp_path / "ghost_dir"
        with pytest.raises(DataValidationError, match="not a directory"):
            merge_table_folder(ghost)


class TestMissingSentinels:
    """pd.NA / pd.NaT must collapse to '', never leak '<NA>'/'NaT'."""

    def test_compare_tables_ignores_pd_na_and_nat(self) -> None:
        a = pd.DataFrame({"Ligand": ["keep.pdbqt", pd.NA, pd.NaT]})
        b = pd.DataFrame({"Ligand": ["KEEP", pd.NA, "extra"]})
        comparison = compare_tables(a, b, column="Ligand")
        assert comparison.common == ["keep"]
        assert comparison.only_in_a == []
        assert comparison.only_in_b == ["extra"]
        assert "<NA>" not in comparison.only_in_b
        assert "NaT" not in comparison.only_in_a

    def test_annotate_with_smiles_ignores_pd_na_reference_cells(self) -> None:
        frame = pd.DataFrame({"Ligand": ["lig1", "lig2"]})
        reference = pd.DataFrame(
            {"Ligand": ["lig1", pd.NA, "lig2"], "SMILES": ["CCO", "CCN", pd.NA]}
        )
        annotated = annotate_with_smiles(frame, reference, name_columns=["Ligand"])
        assert annotated["Ligand_SMILES"].tolist() == ["CCO", ""]

    def test_compare_tables_drops_blank_names_not_phantom_common(self) -> None:
        """Whitespace-only cells must not become a shared '' ligand."""
        a = pd.DataFrame({"Ligand": ["ligand1", "   ", "\t", "ligand2"]})
        b = pd.DataFrame({"Ligand": ["ligand1", " ", "ligand3"]})
        comparison = compare_tables(a, b, column="Ligand")
        assert comparison.common == ["ligand1"]
        assert comparison.only_in_a == ["ligand2"]
        assert comparison.only_in_b == ["ligand3"]
        assert "" not in comparison.common
        assert "" not in comparison.only_in_a
        assert "" not in comparison.only_in_b

    def test_compare_tables_bare_extension_keeps_dotfile_semantics(self) -> None:
        """splitext treats a leading-dot cell as a hidden filename, not an extension."""
        a = pd.DataFrame({"Ligand": [".pdbqt"]})
        b = pd.DataFrame({"Ligand": ["ligand3"]})
        comparison = compare_tables(a, b, column="Ligand")
        assert comparison.only_in_a == [".pdbqt"]
        assert comparison.common == []

    def test_compare_tables_all_blank_names_yield_empty_buckets(self) -> None:
        a = pd.DataFrame({"Ligand": ["  ", "", None]})
        b = pd.DataFrame({"Ligand": ["\t"]})
        comparison = compare_tables(a, b, column="Ligand")
        assert comparison.common == []
        assert comparison.only_in_a == []
        assert comparison.only_in_b == []
        assert comparison.summary() == {"common": 0, "only_in_a": 0, "only_in_b": 0}
