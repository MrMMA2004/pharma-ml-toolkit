"""Tests for the molecular file-format converters (.smi / table / .sdf).

Mirrors the upgrade of the last three rar conversion utilities:
``Convert SMI to CSV.py`` (memory-flat streaming + BOM CSV output),
``Convert CSV to SDF.py`` (3D-embedded ligand libraries with per-row
failure accounting) and ``Sdf reading.ipynb`` (SDF -> table round-trip),
plus the ``pharmaml data convert`` CLI subcommand that surfaces them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")

from rdkit import Chem

from pharmaml.chem.converters import (
    SdfWriteReport,
    sdf_to_table,
    smi_to_csv,
    table_to_sdf,
    table_to_smi,
)
from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError, DependencyError

logging.disable(logging.CRITICAL)

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
INVALID = "not_a_smiles"


@pytest.fixture()
def ligand_frame() -> pd.DataFrame:
    """Three real small molecules with SD-tag columns."""
    return pd.DataFrame(
        {
            "name": ["ethanol", "benzene", "aspirin"],
            "SMILES": [ETHANOL, BENZENE, ASPIRIN],
            "pIC50": [1.51, 2.53, 3.5],
            "origin": ["a", "b", "c"],
        }
    )


@pytest.fixture()
def smi_file(tmp_path: Path) -> Path:
    """A .smi list with multi-word IDs, a bare SMILES and blank lines."""
    path = tmp_path / "compounds.smi"
    path.write_text(
        f"{ETHANOL} ethanol small molecule\n"
        "\n"
        f"{BENZENE} benzene\n"
        f"{ASPIRIN} aspirin\n"
        f"{ETHANOL}\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# smi_to_csv -- the "Convert SMI to CSV.py" upgrade
# ---------------------------------------------------------------------------


class TestSmiToCsv:
    def test_streams_rows_with_joined_ids(self, smi_file: Path) -> None:
        out = smi_to_csv(smi_file)
        assert out == smi_file.with_suffix(".csv")
        frame = pd.read_csv(out, encoding="utf-8-sig", keep_default_na=False)
        assert list(frame.columns) == ["SMILES", "ID"]
        assert list(frame["SMILES"]) == [ETHANOL, BENZENE, ASPIRIN, ETHANOL]
        assert list(frame["ID"]) == [
            "ethanol small molecule",
            "benzene",
            "aspirin",
            "",
        ]

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        src = tmp_path / "gaps.smi"
        src.write_text("\n\nCCO ethane\n\n   \nCC propane\n\n", encoding="utf-8")
        frame = pd.read_csv(smi_to_csv(src), encoding="utf-8-sig", keep_default_na=False)
        assert len(frame) == 2

    def test_output_carries_bom(self, smi_file: Path) -> None:
        out = smi_to_csv(smi_file)
        assert out.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            smi_to_csv(tmp_path / "ghost.smi")

    def test_custom_out_path(self, smi_file: Path, tmp_path: Path) -> None:
        custom = tmp_path / "nested" / "custom.csv"
        out = smi_to_csv(smi_file, custom)
        assert out == custom
        assert custom.is_file()


# ---------------------------------------------------------------------------
# table_to_sdf -- the "Convert CSV to SDF.py" upgrade
# ---------------------------------------------------------------------------


class TestTableToSdf:
    def test_from_dataframe(self, ligand_frame: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "lig.sdf"
        report = table_to_sdf(ligand_frame, out, name_col="name")
        assert isinstance(report, SdfWriteReport)
        assert out.is_file()
        assert report.molecules == 3
        assert report.failures == 0
        assert report.failed_smiles == []
        assert report.embedded_3d == 3
        assert report.source == Path("<dataframe>")
        assert report.output_sdf == out
        assert report.smiles_column == "SMILES"

    def test_from_csv_path(self, ligand_frame: pd.DataFrame, tmp_path: Path) -> None:
        csv = tmp_path / "ligands.csv"
        ligand_frame.to_csv(csv, index=False)
        report = table_to_sdf(csv, tmp_path / "lig.sdf", embed3d=False)
        assert report.source == csv
        assert report.molecules == 3

    def test_audit_str_alignment(self, ligand_frame: pd.DataFrame, tmp_path: Path) -> None:
        report = table_to_sdf(ligand_frame, tmp_path / "l.sdf", embed3d=False)
        text = str(report)
        assert "Table -> SDF conversion" in text
        assert "molecules     : 3" in text
        assert "3D embedded   : 0" in text
        assert "smiles column : SMILES" in text
        assert "failures      : 0" in text

    def test_tags_and_names_round_trip(
        self, ligand_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        sdf = tmp_path / "lig.sdf"
        table_to_sdf(ligand_frame, sdf, name_col="name", embed3d=False)
        back = sdf_to_table(sdf)
        assert list(back.columns) == ["SMILES", "name", "pIC50", "origin"]
        for original, got in zip(ligand_frame["SMILES"], back["SMILES"]):
            assert Chem.MolToSmiles(Chem.MolFromSmiles(original)) == got
        assert list(back["name"]) == ["ethanol", "benzene", "aspirin"]
        assert list(back["origin"]) == ["a", "b", "c"]
        assert list(back["pIC50"]) == pytest.approx([1.51, 2.53, 3.5])

    def test_default_ligand_n_naming(self, tmp_path: Path) -> None:
        sdf = tmp_path / "l.sdf"
        table_to_sdf(pd.DataFrame({"SMILES": [ETHANOL, BENZENE]}), sdf, embed3d=False)
        back = sdf_to_table(sdf)
        assert list(back["name"]) == ["ligand_1", "ligand_2"]

    def test_name_col_empty_cell_falls_back_to_row_number(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"name": ["ethanol", None], "SMILES": [ETHANOL, BENZENE]})
        sdf = tmp_path / "l.sdf"
        table_to_sdf(frame, sdf, name_col="name", embed3d=False)
        back = sdf_to_table(sdf)
        assert list(back["name"]) == ["ethanol", "2"]

    def test_invalid_smiles_accounted_not_written(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {"SMILES": [ETHANOL, INVALID, BENZENE], "activity": [1.0, 2.0, 3.0]}
        )
        sdf = tmp_path / "l.sdf"
        report = table_to_sdf(frame, sdf, embed3d=False)
        assert report.molecules == 2
        assert report.failures == 1
        assert report.failed_smiles == [INVALID]
        back = sdf_to_table(sdf)
        assert len(back) == 2
        assert INVALID not in set(back["SMILES"])

    def test_blank_smiles_cell_counted_as_failure(self, tmp_path: Path) -> None:
        # 100x-19-a family fix: RDKit parses "" as a *valid* 0-atom molecule,
        # which either crashed 3D embedding with "molecule has no atoms" or
        # silently wrote an empty SDF record.  Blank cells are parse failures.
        frame = pd.DataFrame({"SMILES": [ETHANOL, "", BENZENE]})
        sdf = tmp_path / "l.sdf"
        report = table_to_sdf(frame, sdf, embed3d=False)
        assert report.molecules == 2
        assert report.failures == 1
        assert report.failed_smiles == [""]
        assert sdf.read_text(encoding="utf-8").count("$$$$") == 2

    def test_blank_smiles_cell_with_3d_embedding(self, tmp_path: Path) -> None:
        # The embed3d=True path crashed outright (ValueError: molecule has
        # no atoms) before the blank-cell guard.
        frame = pd.DataFrame({"SMILES": [ETHANOL, ""]})
        sdf = tmp_path / "l.sdf"
        report = table_to_sdf(frame, sdf)
        assert report.molecules == 1
        assert report.failures == 1
        assert report.embedded_3d == 1
        assert sdf.read_text(encoding="utf-8").count("$$$$") == 1

    def test_embed3d_false_fast_path(self, ligand_frame: pd.DataFrame, tmp_path: Path) -> None:
        sdf = tmp_path / "l.sdf"
        report = table_to_sdf(ligand_frame, sdf, embed3d=False)
        assert report.embedded_3d == 0
        assert report.notes == []
        mols = list(Chem.SDMolSupplier(str(sdf)))
        assert len(mols) == 3
        assert all(mol is not None for mol in mols)

    def test_embed_failure_falls_back_with_note(
        self, ligand_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rdkit.Chem import AllChem

        # Simulate an ETKDG failure (modern ETKDG embeds even pathological
        # rings, so we force the -1 return code the real code handles).
        monkeypatch.setattr(AllChem, "EmbedMolecule", lambda mol, *a, **kw: -1)
        sdf = tmp_path / "l.sdf"
        report = table_to_sdf(ligand_frame, sdf, name_col="name")
        assert report.molecules == 3  # non-fatal: every molecule still written
        assert report.embedded_3d == 0
        assert len(report.notes) == 3
        assert all("embedding failed" in note for note in report.notes)
        assert all("kept without coordinates" in note for note in report.notes)
        back = sdf_to_table(sdf)
        assert list(back["name"]) == ["ethanol", "benzene", "aspirin"]

    def test_uff_non_convergence_is_noted(
        self, ligand_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rdkit.Chem import AllChem

        monkeypatch.setattr(AllChem, "UFFOptimizeMolecule", lambda mol: 1)
        report = table_to_sdf(ligand_frame, tmp_path / "l.sdf")
        assert report.embedded_3d == 3
        assert any("UFF" in note for note in report.notes)

    def test_uff_disabled_skips_optimisation(
        self, ligand_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rdkit.Chem import AllChem

        calls: list[int] = []
        monkeypatch.setattr(
            AllChem, "UFFOptimizeMolecule", lambda mol: calls.append(1) or 1
        )
        report = table_to_sdf(ligand_frame, tmp_path / "l.sdf", uff=False)
        assert report.embedded_3d == 3
        assert calls == []
        assert report.notes == []

    def test_all_invalid_raises(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"SMILES": [INVALID, "definitely not smiles either"]})
        with pytest.raises(DataValidationError, match="no valid molecules"):
            table_to_sdf(frame, tmp_path / "l.sdf")

    def test_empty_frame_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="no rows"):
            table_to_sdf(pd.DataFrame({"SMILES": []}), tmp_path / "l.sdf")

    def test_missing_smiles_column_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="Missing required"):
            table_to_sdf(pd.DataFrame({"smi": [ETHANOL]}), tmp_path / "l.sdf")

    def test_missing_name_column_raises(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"SMILES": [ETHANOL]})
        with pytest.raises(DataValidationError, match="name column"):
            table_to_sdf(frame, tmp_path / "l.sdf", name_col="Name")

    def test_rdkit_missing_raises_dependency_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "rdkit", None)
        monkeypatch.setitem(sys.modules, "rdkit.Chem", None)
        with pytest.raises(DependencyError, match="pip install rdkit"):
            table_to_sdf(pd.DataFrame({"SMILES": [ETHANOL]}), tmp_path / "l.sdf")


# ---------------------------------------------------------------------------
# sdf_to_table -- the "Sdf reading.ipynb" upgrade
# ---------------------------------------------------------------------------


class TestSdfToTable:
    def test_writes_out_table_when_given(
        self, ligand_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        sdf = tmp_path / "lig.sdf"
        table_to_sdf(ligand_frame, sdf, name_col="name", embed3d=False)
        out = tmp_path / "back.csv"
        back = sdf_to_table(sdf, out=out)
        assert out.is_file()
        reread = pd.read_csv(out)
        assert list(reread["SMILES"]) == list(back["SMILES"])
        assert list(reread["name"]) == ["ethanol", "benzene", "aspirin"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            sdf_to_table(tmp_path / "ghost.sdf")

    def test_empty_sdf_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.sdf"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(DataValidationError, match="empty or unreadable"):
            sdf_to_table(empty)

    def test_unparseable_sdf_raises(self, tmp_path: Path) -> None:
        junk = tmp_path / "junk.sdf"
        junk.write_text("definitely not molfile content\n$$$$\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="no parseable molecules"):
            sdf_to_table(junk)

    def test_rdkit_missing_raises_dependency_error(
        self, ligand_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        sdf = tmp_path / "lig.sdf"
        table_to_sdf(ligand_frame, sdf, embed3d=False)
        monkeypatch.setitem(sys.modules, "rdkit", None)
        monkeypatch.setitem(sys.modules, "rdkit.Chem", None)
        with pytest.raises(DependencyError, match="pip install rdkit"):
            sdf_to_table(sdf)


# ---------------------------------------------------------------------------
# table_to_smi -- the inverse of the .smi reader
# ---------------------------------------------------------------------------


class TestTableToSmi:
    def test_lines_format(self, ligand_frame: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "lig.smi"
        path = table_to_smi(ligand_frame, out, name_col="name")
        assert path == out
        assert out.read_text(encoding="utf-8").splitlines() == [
            f"{ETHANOL} ethanol",
            f"{BENZENE} benzene",
            f"{ASPIRIN} aspirin",
        ]

    def test_name_omitted_without_name_col(
        self, ligand_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = tmp_path / "lig.smi"
        table_to_smi(ligand_frame, out)
        assert out.read_text(encoding="utf-8").splitlines() == [
            ETHANOL,
            BENZENE,
            ASPIRIN,
        ]

    def test_empty_name_cell_omitted(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"name": ["ethanol", None], "SMILES": [ETHANOL, BENZENE]})
        out = tmp_path / "l.smi"
        table_to_smi(frame, out, name_col="name")
        assert out.read_text(encoding="utf-8").splitlines() == [
            f"{ETHANOL} ethanol",
            BENZENE,
        ]

    def test_blank_smiles_cells_skipped(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"SMILES": [ETHANOL, None, BENZENE]})
        out = tmp_path / "l.smi"
        table_to_smi(frame, out)
        assert out.read_text(encoding="utf-8").splitlines() == [ETHANOL, BENZENE]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            table_to_smi(tmp_path / "ghost.csv", tmp_path / "g.smi")

    def test_missing_smiles_column_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="Missing required"):
            table_to_smi(pd.DataFrame({"smi": [ETHANOL]}), tmp_path / "g.smi")


# ---------------------------------------------------------------------------
# CLI: pharmaml data convert
# ---------------------------------------------------------------------------


class TestCliDataConvert:
    def test_csv_to_sdf_happy_path(self, tmp_path: Path, capsys) -> None:
        csv = tmp_path / "ligands.csv"
        pd.DataFrame(
            {"name": ["ethanol", "benzene"], "SMILES": [ETHANOL, BENZENE]}
        ).to_csv(csv, index=False)
        out = tmp_path / "lig.sdf"
        code = main(
            [
                "data", "convert", "--data", str(csv), "--to", "sdf",
                "--no-3d", "--out", str(out),
            ]
        )
        assert code == 0
        assert out.is_file()
        stdout = capsys.readouterr().out
        assert "Table -> SDF conversion" in stdout
        assert "molecules     : 2" in stdout
        assert f"output -> {out}" in stdout

    def test_default_out_derived_from_input_stem(self, tmp_path: Path) -> None:
        csv = tmp_path / "ligands.csv"
        pd.DataFrame({"SMILES": [ETHANOL]}).to_csv(csv, index=False)
        code = main(["data", "convert", "--data", str(csv), "--to", "sdf", "--no-3d"])
        assert code == 0
        assert (tmp_path / "ligands.sdf").is_file()

    def test_smi_to_csv(self, tmp_path: Path, capsys) -> None:
        smi = tmp_path / "compounds.smi"
        smi.write_text(f"{ETHANOL} ethanol\n{BENZENE} benzene\n", encoding="utf-8")
        code = main(["data", "convert", "--data", str(smi), "--to", "csv"])
        assert code == 0
        out = tmp_path / "compounds.csv"
        assert out.is_file()
        frame = pd.read_csv(out, encoding="utf-8-sig", keep_default_na=False)
        assert list(frame["SMILES"]) == [ETHANOL, BENZENE]
        assert list(frame["ID"]) == ["ethanol", "benzene"]
        assert "converted" in capsys.readouterr().out

    def test_smi_to_sdf_bridge_uses_id_column(self, tmp_path: Path) -> None:
        smi = tmp_path / "compounds.smi"
        smi.write_text(f"{ETHANOL} ethanol\n{BENZENE} benzene\n", encoding="utf-8")
        out = tmp_path / "c.sdf"
        code = main(
            [
                "data", "convert", "--data", str(smi), "--to", "sdf",
                "--no-3d", "--name-col", "ID", "--out", str(out),
            ]
        )
        assert code == 0
        back = sdf_to_table(out)
        assert list(back["name"]) == ["ethanol", "benzene"]

    def test_sdf_to_csv(self, tmp_path: Path) -> None:
        sdf = tmp_path / "lig.sdf"
        table_to_sdf(
            pd.DataFrame({"name": ["ethanol"], "SMILES": [ETHANOL]}),
            sdf,
            name_col="name",
            embed3d=False,
        )
        out = tmp_path / "back.csv"
        code = main(["data", "convert", "--data", str(sdf), "--to", "csv", "--out", str(out)])
        assert code == 0
        frame = pd.read_csv(out)
        assert frame["SMILES"].iloc[0] == ETHANOL
        assert frame["name"].iloc[0] == "ethanol"

    def test_csv_to_smi(self, tmp_path: Path) -> None:
        csv = tmp_path / "ligands.csv"
        pd.DataFrame(
            {"name": ["ethanol", "benzene"], "SMILES": [ETHANOL, BENZENE]}
        ).to_csv(csv, index=False)
        out = tmp_path / "lig.smi"
        code = main(
            [
                "data", "convert", "--data", str(csv), "--to", "smi",
                "--name-col", "name", "--out", str(out),
            ]
        )
        assert code == 0
        assert out.read_text(encoding="utf-8").splitlines() == [
            f"{ETHANOL} ethanol",
            f"{BENZENE} benzene",
        ]

    def test_invalid_smiles_exit_code(self, tmp_path: Path, capsys) -> None:
        csv = tmp_path / "mixed.csv"
        pd.DataFrame({"SMILES": [ETHANOL, INVALID]}).to_csv(csv, index=False)
        out = tmp_path / "mixed.sdf"
        code = main(
            ["data", "convert", "--data", str(csv), "--to", "sdf", "--no-3d", "--out", str(out)]
        )
        assert code == 1
        captured = capsys.readouterr()
        assert "failures      : 1" in captured.out
        assert "invalid SMILES skipped" in captured.err
        # the valid molecule was still written
        back = sdf_to_table(out)
        assert list(back["SMILES"]) == [ETHANOL]

    def test_sdf_to_sdf_rejected(self, tmp_path: Path, capsys) -> None:
        sdf = tmp_path / "x.sdf"
        table_to_sdf(pd.DataFrame({"SMILES": [ETHANOL]}), sdf, embed3d=False)
        code = main(["data", "convert", "--data", str(sdf), "--to", "sdf"])
        assert code == 1
        captured = capsys.readouterr()
        assert "error:" in captured.err
        assert "already an SDF" in captured.err

    def test_missing_input_file_reported_cleanly(self, tmp_path: Path, capsys) -> None:
        code = main(["data", "convert", "--data", str(tmp_path / "ghost.csv"), "--to", "sdf"])
        assert code == 1
        assert "error:" in capsys.readouterr().err


class TestSdfToTableDroppedRecords:
    """26-R5 defect 4: unparseable SDF records were dropped silently.

    A 2-record SDF with 1 corrupt record yielded 1 row, NO warning and a
    CLI exit 0 -- the mirror image of ``table_to_sdf``, which counts and
    reports every failed row.
    """

    @staticmethod
    def _mixed_sdf(tmp_path: Path) -> Path:
        """One valid ethanol record + one corrupt record."""
        good = tmp_path / "good.sdf"
        table_to_sdf(
            pd.DataFrame({"name": ["ethanol"], "SMILES": [ETHANOL]}),
            good,
            name_col="name",
            embed3d=False,
        )
        mixed = tmp_path / "mixed.sdf"
        mixed.write_text(
            good.read_text(encoding="utf-8") + "corrupt not-a-molfile\n$$$$\n",
            encoding="utf-8",
        )
        return mixed

    def test_dropped_records_warned_and_counted(self, tmp_path: Path, caplog) -> None:
        # two silencers must be lifted for caplog (suite convention, cf.
        # tests/test_descriptors.py::_visible_logs): this module disables
        # logging at import time and CLI tests set propagate=False on the
        # pharmaml root logger.
        prev_disable = logging.root.manager.disable
        pharma_root = logging.getLogger("pharmaml")
        prev_propagate = pharma_root.propagate
        logging.disable(logging.NOTSET)
        pharma_root.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="pharmaml.chem.converters"):
                frame = sdf_to_table(self._mixed_sdf(tmp_path))
        finally:
            pharma_root.propagate = prev_propagate
            logging.disable(prev_disable)

        assert len(frame) == 1
        assert frame["SMILES"].iloc[0] == ETHANOL  # the valid row survives
        assert "1 of 2 SDF records unparseable and dropped" in caplog.text
        assert frame.attrs["sdf_dropped_records"] == 1
        assert frame.attrs["sdf_total_records"] == 2

    def test_clean_sdf_has_zero_drop_counts(self, ligand_frame: pd.DataFrame, tmp_path: Path) -> None:
        sdf = tmp_path / "clean.sdf"
        table_to_sdf(ligand_frame, sdf, embed3d=False)
        frame = sdf_to_table(sdf)
        assert len(frame) == 3
        assert frame.attrs["sdf_dropped_records"] == 0
        assert frame.attrs["sdf_total_records"] == 3


class TestCliDataConvertSdfDrops:
    """CLI side of 26-R5 defect 4: `data convert` reports drops, exit 1."""

    def test_sdf_to_csv_reports_drop_and_exits_1(self, tmp_path: Path, capsys) -> None:
        good = tmp_path / "good.sdf"
        table_to_sdf(
            pd.DataFrame({"name": ["ethanol"], "SMILES": [ETHANOL]}),
            good,
            name_col="name",
            embed3d=False,
        )
        mixed = tmp_path / "mixed.sdf"
        mixed.write_text(
            good.read_text(encoding="utf-8") + "corrupt not-a-molfile\n$$$$\n",
            encoding="utf-8",
        )
        out = tmp_path / "back.csv"
        code = main(
            ["data", "convert", "--data", str(mixed), "--to", "csv", "--out", str(out)]
        )
        assert code == 1
        captured = capsys.readouterr()
        assert "1 of 2 SDF records unparseable and dropped" in captured.err
        # the surviving molecule was still converted
        assert out.is_file()
        assert pd.read_csv(out)["SMILES"].iloc[0] == ETHANOL

    def test_sdf_to_smi_reports_drop_and_exits_1(self, tmp_path: Path, capsys) -> None:
        good = tmp_path / "good.sdf"
        table_to_sdf(
            pd.DataFrame({"name": ["ethanol"], "SMILES": [ETHANOL]}),
            good,
            name_col="name",
            embed3d=False,
        )
        mixed = tmp_path / "mixed.sdf"
        mixed.write_text(
            good.read_text(encoding="utf-8") + "corrupt not-a-molfile\n$$$$\n",
            encoding="utf-8",
        )
        out = tmp_path / "back.smi"
        code = main(
            ["data", "convert", "--data", str(mixed), "--to", "smi", "--out", str(out)]
        )
        assert code == 1
        captured = capsys.readouterr()
        assert "1 of 2 SDF records unparseable and dropped" in captured.err
        assert out.read_text(encoding="utf-8").splitlines() == [f"{ETHANOL} ethanol"]

    def test_clean_sdf_to_csv_still_exits_0(self, tmp_path: Path, capsys) -> None:
        # no drop -> the historical exit 0 must be preserved
        sdf = tmp_path / "clean.sdf"
        table_to_sdf(pd.DataFrame({"SMILES": [ETHANOL]}), sdf, embed3d=False)
        out = tmp_path / "back.csv"
        code = main(
            ["data", "convert", "--data", str(sdf), "--to", "csv", "--out", str(out)]
        )
        assert code == 0
        assert "unparseable and dropped" not in capsys.readouterr().err


class TestCliCsvToCsvGuard:
    """26-R5 defect 5: csv -> csv rewrote the INPUT in place.

    The same-format guard existed only for smi/sdf; a semicolon CSV was
    re-serialised with commas straight over the input file (exit 0).
    """

    def test_csv_to_csv_rejected_input_never_mutated(self, tmp_path: Path, capsys) -> None:
        euro = tmp_path / "euro.csv"
        euro.write_text("SMILES;name\nCCO;ethanol\nCCC;propane\n", encoding="utf-8")
        before = euro.read_bytes()
        code = main(["data", "convert", "--data", str(euro), "--to", "csv"])
        assert code == 1
        err = capsys.readouterr().err
        assert "error:" in err
        assert "already a CSV file" in err
        assert euro.read_bytes() == before

    def test_tsv_to_csv_still_converts(self, tmp_path: Path, capsys) -> None:
        # the guard is suffix-based: a real delimiter change keeps working
        tsv = tmp_path / "table.tsv"
        tsv.write_text("SMILES\tname\nCCO\tethanol\n", encoding="utf-8")
        code = main(["data", "convert", "--data", str(tsv), "--to", "csv"])
        assert code == 0
        out = tmp_path / "table.csv"
        assert out.is_file()
        frame = pd.read_csv(out)
        assert list(frame["SMILES"]) == [ETHANOL]
