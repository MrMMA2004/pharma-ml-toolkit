"""Tests for the ligand-library import and protein-receptor preparation.

The ligand tests mirror the ``smilestopdbqt.py`` / ``ligand_preparation_
autodock4.py`` upgrade: CSV import, SMILES validation accounting,
one-folder-per-ligand campaign trees and ``failed_smiles.csv`` parity.

The receptor tests exercise every cleaning rule (waters, non-protein
HETATMs, alternate locations, chain selection, hydrogen stripping),
the Meeko polymer pipeline on a real PDB (crambin 1CRN, public-domain
structure) and the PDBQT passthrough mode that needs no Meeko at all.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pharmaml.docking.ligands import (
    LigandPreparationReport,
    load_ligand_csv,
    prepare_ligand_library,
)
from pharmaml.docking.preparation import (
    compute_grid_box,
    smiles_to_pdbqt,
)
from pharmaml.docking.receptor import (
    ProteinReport,
    distribute_receptor,
    prepare_protein,
)
from pharmaml.exceptions import (
    DataValidationError,
    DependencyError,
    DockingError,
)

rdkit = pytest.importorskip("rdkit")

CRAMBIN = Path(__file__).parent / "data" / "1crn.pdb"


def atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
    record: str = "ATOM  ",
    altloc: str = " ",
) -> str:
    """Column-exact PDB ATOM/HETATM record (altloc at col 17)."""
    return (
        f"{record}{serial:5d} {name:>4s}{altloc}{resname:>3s} {chain}"
        f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
    )


def pdbqt_line(serial: int, name: str, resname: str, chain: str, resseq: int,
               charge: float, atom_type: str) -> str:
    """Column-exact PDBQT record (charge at cols 71-76, type last)."""
    return (
        f"ATOM  {serial:5d} {name:>4s} {resname:>3s} {chain}{resseq:4d}    "
        f"{1.0:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{0.0:6.2f}"
        f"    {charge:+6.3f}{atom_type:>3s}"
    )


# ---------------------------------------------------------------------------
# load_ligand_csv
# ---------------------------------------------------------------------------


class TestLoadLigandCsv:
    def test_happy_path_with_names(self, tmp_path: Path) -> None:
        csv = tmp_path / "ligands.csv"
        csv.write_text(
            "name,SMILES\naspirin,CC(=O)Oc1ccccc1C(=O)O\n,"
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O\n",
            encoding="utf-8",
        )
        lib = load_ligand_csv(csv)
        assert len(lib) == 2
        assert lib.smiles_column == "SMILES"
        assert lib.name_column == "name"
        assert lib.validated is True
        # blank name cell falls back to the row number, not "nan"
        assert list(lib.frame["name"]) == ["aspirin", "2"]
        assert list(lib.frame["status"]) == ["valid", "valid"]

    def test_smiles_column_autodetected(self, tmp_path: Path) -> None:
        csv = tmp_path / "lib.csv"
        csv.write_text("canonical_smiles\nCCO\nCCC\n", encoding="utf-8")
        lib = load_ligand_csv(csv)
        assert lib.smiles_column == "canonical_smiles"
        # no name column -> 1-based row numbers
        assert list(lib.frame["name"]) == ["1", "2"]

    def test_case_insensitive_detection(self, tmp_path: Path) -> None:
        csv = tmp_path / "lib.csv"
        csv.write_text("Smiles,Drug\nCCO,ethanol\n", encoding="utf-8")
        lib = load_ligand_csv(csv)
        assert lib.smiles_column == "Smiles"
        assert lib.name_column == "Drug"

    def test_tsv_and_sniffing(self, tmp_path: Path) -> None:
        tsv = tmp_path / "lib.tsv"
        tsv.write_text("name\tSMILES\nethanol\tCCO\n", encoding="utf-8")
        lib = load_ligand_csv(tsv)
        assert len(lib) == 1
        assert lib.frame.iloc[0]["smiles"] == "CCO"

    def test_invalid_smiles_accounted(self, tmp_path: Path) -> None:
        csv = tmp_path / "ligands.csv"
        csv.write_text(
            "name,SMILES\nok,CCO\nbad,not_a_smiles\nnone,\n",
            encoding="utf-8",
        )
        lib = load_ligand_csv(csv)
        assert len(lib.invalid_frame) == 2
        assert "parse" in lib.invalid_frame.iloc[0]["error"].lower()
        assert "empty" in lib.invalid_frame.iloc[1]["error"]
        assert len(lib.valid_frame) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            load_ligand_csv(tmp_path / "nope.csv")

    def test_no_smiles_column_raises_with_column_list(self, tmp_path: Path) -> None:
        csv = tmp_path / "lib.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="no SMILES column"):
            load_ligand_csv(csv)

    def test_explicit_column_missing_raises(self, tmp_path: Path) -> None:
        csv = tmp_path / "lib.csv"
        csv.write_text("SMILES\nCCO\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="'xyz' not found"):
            load_ligand_csv(csv, smiles_col="xyz")

    def test_empty_table_raises(self, tmp_path: Path) -> None:
        csv = tmp_path / "lib.csv"
        csv.write_text("name,SMILES\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="no rows"):
            load_ligand_csv(csv)

    def test_write_failures_parity(self, tmp_path: Path) -> None:
        csv = tmp_path / "ligands.csv"
        csv.write_text("name,SMILES\nok,CCO\nbad,zzz\n", encoding="utf-8")
        lib = load_ligand_csv(csv)
        out = tmp_path / "failed_smiles.csv"
        assert lib.write_failures(out) is not None
        failed = pd.read_csv(out)
        assert list(failed["name"]) == ["bad"]
        # clean library -> nothing written
        clean = tmp_path / "clean.csv"
        clean.write_text("name,SMILES\nok,CCO\n", encoding="utf-8")
        assert load_ligand_csv(clean).write_failures(tmp_path / "x.csv") is None

    def test_audit_str(self, tmp_path: Path) -> None:
        csv = tmp_path / "lib.csv"
        csv.write_text("SMILES\nCCO\nzzz\n", encoding="utf-8")
        text = str(load_ligand_csv(csv))
        assert "Ligand library" in text
        assert "rows" in text and "invalid" in text

    def test_numeric_names_lose_trailing_zero(self, tmp_path: Path) -> None:
        """An integer ID column must not become folders '1.0', '2.0'."""
        csv = tmp_path / "lib.csv"
        csv.write_text("id,SMILES\n1,CCO\n2,CCC\n,CCN\n", encoding="utf-8")
        lib = load_ligand_csv(csv)
        assert list(lib.frame["name"]) == ["1", "2", "3"]

    def test_xlsx_input(self, tmp_path: Path) -> None:
        pytest.importorskip("openpyxl")
        xlsx = tmp_path / "lib.xlsx"
        frame = pd.DataFrame({"name": ["ethanol"], "SMILES": ["CCO"]})
        frame.to_excel(xlsx, index=False)
        lib = load_ligand_csv(xlsx)
        assert len(lib) == 1
        assert lib.smiles_column == "SMILES"


# ---------------------------------------------------------------------------
# prepare_ligand_library
# ---------------------------------------------------------------------------


pytest.importorskip("meeko", reason="ligand PDBQT preparation needs the optional meeko extra")


class TestPrepareLigandLibrary:
    @pytest.fixture()
    def ligands_csv(self, tmp_path: Path) -> Path:
        csv = tmp_path / "ligands.csv"
        csv.write_text(
            "name,SMILES\n"
            "aspirin,CC(=O)Oc1ccccc1C(=O)O\n"
            "caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C\n"
            "broken,not_a_smiles\n",
            encoding="utf-8",
        )
        return csv

    def test_campaign_tree_layout(self, ligands_csv: Path, tmp_path: Path) -> None:
        report = prepare_ligand_library(ligands_csv, tmp_path / "campaign")
        assert isinstance(report, LigandPreparationReport)
        assert report.n_prepared == 2
        assert report.n_failed == 1
        folders = sorted(p.name for p in (tmp_path / "campaign").iterdir())
        assert folders == ["01_aspirin", "02_caffeine", "failed_smiles.csv"]
        assert (tmp_path / "campaign" / "01_aspirin" / "l.pdbqt").is_file()
        row = report.frame[report.frame["name"] == "aspirin"].iloc[0]
        assert row["atoms"] > 0
        assert row["torsdof"] == 4  # aspirin has 4 rotatable bonds

    def test_index_naming_matches_legacy_scripts(self, ligands_csv: Path, tmp_path: Path) -> None:
        report = prepare_ligand_library(
            ligands_csv, tmp_path / "legacy", folder_naming="index"
        )
        ok = report.frame[report.frame["ok"]]
        assert list(ok["folder"]) == ["1", "2"]

    def test_name_naming(self, ligands_csv: Path, tmp_path: Path) -> None:
        report = prepare_ligand_library(
            ligands_csv, tmp_path / "named", folder_naming="name"
        )
        ok = report.frame[report.frame["ok"]]
        assert sorted(ok["folder"]) == ["aspirin", "caffeine"]

    def test_invalid_ligand_folders_removed(self, ligands_csv: Path, tmp_path: Path) -> None:
        report = prepare_ligand_library(ligands_csv, tmp_path / "camp")
        assert not (tmp_path / "camp" / "03_broken").exists()
        failed = report.failed_frame
        assert list(failed["name"]) == ["broken"]
        assert (failed.iloc[0]["error"]).strip() != ""

    def test_failed_smiles_csv_written_by_default(self, ligands_csv: Path, tmp_path: Path) -> None:
        report = prepare_ligand_library(ligands_csv, tmp_path / "camp")
        out = report.write_failures_csv()
        assert out is not None and out.name == "failed_smiles.csv"
        failed = pd.read_csv(out)
        assert list(failed["name"]) == ["broken"]

    def test_accepts_library_object(self, tmp_path: Path) -> None:
        csv = tmp_path / "l.csv"
        csv.write_text("name,SMILES\nethanol,CCO\n", encoding="utf-8")
        lib = load_ligand_csv(csv)
        report = prepare_ligand_library(lib, tmp_path / "out")
        assert report.library is lib
        assert report.n_prepared == 1

    def test_bad_folder_naming_rejected(self, tmp_path: Path) -> None:
        csv = tmp_path / "l.csv"
        csv.write_text("SMILES\nCCO\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="folder_naming"):
            prepare_ligand_library(csv, tmp_path / "x", folder_naming="bogus")

    def test_audit_str(self, ligands_csv: Path, tmp_path: Path) -> None:
        report = prepare_ligand_library(ligands_csv, tmp_path / "c")
        text = str(report)
        assert "prepared" in text and "failed_smiles.csv" in text

    def test_duplicate_names_get_suffixed_folders(self, tmp_path: Path) -> None:
        csv = tmp_path / "l.csv"
        csv.write_text(
            "name,SMILES\naspirin,CCO\naspirin,CCC\naspirin,CCN\n",
            encoding="utf-8",
        )
        prepare_ligand_library(csv, tmp_path / "dupes", folder_naming="name")
        folders = sorted(p.name for p in (tmp_path / "dupes").iterdir() if p.is_dir())
        assert len(folders) == 3
        assert len(set(folders)) == 3  # no collision

    def test_name_naming_without_name_column_raises(self, tmp_path: Path) -> None:
        csv = tmp_path / "l.csv"
        csv.write_text("SMILES\nCCO\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="name column"):
            prepare_ligand_library(csv, tmp_path / "x", folder_naming="name")

    def test_clean_run_writes_no_failures_csv(self, tmp_path: Path) -> None:
        csv = tmp_path / "l.csv"
        csv.write_text("name,SMILES\nethanol,CCO\n", encoding="utf-8")
        report = prepare_ligand_library(csv, tmp_path / "clean")
        assert report.n_failed == 0
        assert not (tmp_path / "clean" / "failed_smiles.csv").exists()
        assert report.write_failures_csv() is None

    def test_dependency_error_fails_fast(self, tmp_path: Path, monkeypatch) -> None:
        from pharmaml.docking import ligands as ligands_mod
        from pharmaml.exceptions import DependencyError

        csv = tmp_path / "l.csv"
        csv.write_text("name,SMILES\nethanol,CCO\nmethanol,CO\n", encoding="utf-8")

        def boom(*args, **kwargs):
            raise DependencyError("install meeko first")

        monkeypatch.setattr(ligands_mod, "smiles_to_pdbqt", boom)
        with pytest.raises(DependencyError, match="meeko"):
            prepare_ligand_library(csv, tmp_path / "x")


# ---------------------------------------------------------------------------
# prepare_protein — PDBQT passthrough (no Meeko needed)
# ---------------------------------------------------------------------------


class TestProteinPdbqtPassthrough:
    def _receptor(self, tmp_path: Path, extra: list[str] | None = None) -> Path:
        lines = [
            "REMARK  prepared receptor",
            pdbqt_line(1, "CA", "ALA", "A", 1, 0.24, "C"),
            pdbqt_line(2, "O", "ALA", "A", 1, -0.27, "OA"),
            pdbqt_line(3, "HD", "ALA", "A", 1, 0.42, "HD"),
            "TER",
            "END",
        ]
        if extra:
            lines = lines[:1] + extra + lines[1:]
        path = tmp_path / "r.pdbqt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_passthrough_keeps_types_and_charges(self, tmp_path: Path) -> None:
        src = self._receptor(tmp_path)
        report = prepare_protein(src, tmp_path / "out.pdbqt")
        assert isinstance(report, ProteinReport)
        assert report.mode == "pdbqt"
        assert report.output_atoms == 3
        assert report.atom_types == {"C": 1, "OA": 1, "HD": 1}
        assert report.residues == 1
        assert report.chains == ["A"]
        assert report.charge_min == pytest.approx(-0.27)
        assert report.charge_max == pytest.approx(0.42)
        text = (tmp_path / "out.pdbqt").read_text()
        assert "+0.240" in text  # charges preserved verbatim

    def test_water_removed_from_pdbqt(self, tmp_path: Path) -> None:
        water = pdbqt_line(9, "O", "HOH", "A", 99, -0.5, "OA")
        src = self._receptor(tmp_path, extra=[water])
        report = prepare_protein(src, tmp_path / "out.pdbqt")
        assert report.waters_removed == 1
        assert report.output_atoms == 3

    def test_nonprotein_hetatm_removed_unless_keep(self, tmp_path: Path) -> None:
        ion = atom_line(10, "CL", "CL", "A", 98, 1.0, 1.0, 1.0, "Cl", record="HETATM")
        src = self._receptor(tmp_path, extra=[ion])
        report = prepare_protein(src, tmp_path / "out.pdbqt")
        assert report.hetatm_removed == 1
        kept = prepare_protein(src, tmp_path / "keep.pdbqt", keep_hetatm=True)
        assert kept.hetatm_removed == 0
        assert kept.output_atoms == 4

    def test_modified_residue_always_kept(self, tmp_path: Path) -> None:
        mse = atom_line(11, "SE", "MSE", "A", 50, 1.0, 1.0, 1.0, "SE", record="HETATM")
        src = self._receptor(tmp_path, extra=[mse])
        report = prepare_protein(src, tmp_path / "out.pdbqt")
        assert report.hetatm_removed == 0
        assert report.output_atoms == 4

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            prepare_protein(tmp_path / "ghost.pdb")

    def test_no_atom_records_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.pdb"
        path.write_text("HEADER nothing\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="no ATOM/HETATM"):
            prepare_protein(path)

    def test_default_output_collision_avoided(self, tmp_path: Path) -> None:
        src = self._receptor(tmp_path)  # r.pdbqt in tmp_path
        report = prepare_protein(src)  # default output == input path
        assert report.output_pdbqt.name == "r_clean.pdbqt"
        assert src.read_text().startswith("REMARK")  # original untouched

    def test_audit_dict_and_str(self, tmp_path: Path) -> None:
        report = prepare_protein(self._receptor(tmp_path), tmp_path / "o.pdbqt")
        info = report.as_dict()
        assert info["mode"] == "pdbqt"
        assert info["output_atoms"] == 3
        assert "Protein preparation audit" in str(report)


# ---------------------------------------------------------------------------
# prepare_protein — cleaning rules on synthetic PDB text (no Meeko needed
# for rules that empty the file; Meeko path tested with the real fixture)
# ---------------------------------------------------------------------------


class TestProteinCleaningRules:
    def test_altloc_first_conformer_kept(self, tmp_path: Path) -> None:
        pdb = tmp_path / "alt.pdb"
        pdb.write_text(
            "\n".join(
                [
                    atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
                    atom_line(2, "CB", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
                    atom_line(3, "CB", "ALA", "A", 1, 9.9, 0.0, 0.0, "C", altloc="B"),
                    "END",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        # PDB mode needs Meeko to finish; assert on the cleaning counters
        # by using the passthrough suffix instead.
        pdbqt = tmp_path / "alt.pdbqt"
        pdbqt.write_text(pdb.read_text(), encoding="utf-8")
        report = prepare_protein(pdbqt, tmp_path / "out.pdbqt")
        assert report.altloc_removed == 1
        assert report.output_atoms == 2

    def test_altloc_explicit_letter(self, tmp_path: Path) -> None:
        pdbqt = tmp_path / "alt.pdbqt"
        pdbqt.write_text(
            "\n".join(
                [
                    atom_line(1, "CB", "ALA", "A", 1, 1.0, 0.0, 0.0, "C", altloc="A"),
                    atom_line(2, "CB", "ALA", "A", 1, 9.9, 0.0, 0.0, "C", altloc="B"),
                    atom_line(3, "CA", "ALA", "A", 1, 0.5, 0.0, 0.0, "C"),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report = prepare_protein(pdbqt, tmp_path / "b.pdbqt", default_altloc="B")
        # the A conformer is dropped, the B conformer and the
        # conformer-independent CA are kept
        assert report.altloc_removed == 1
        assert report.output_atoms == 2

    def test_chain_selection(self, tmp_path: Path) -> None:
        pdbqt = tmp_path / "two.pdbqt"
        pdbqt.write_text(
            "\n".join(
                [
                    pdbqt_line(1, "CA", "ALA", "A", 1, 0.1, "C"),
                    pdbqt_line(2, "CA", "ALA", "B", 1, 0.1, "C"),
                    "END",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report = prepare_protein(pdbqt, tmp_path / "a.pdbqt", chain="A")
        assert report.chains == ["A"]
        assert report.output_atoms == 1

    def test_all_filtered_away_raises(self, tmp_path: Path) -> None:
        pdbqt = tmp_path / "onlywater.pdbqt"
        pdbqt.write_text(
            pdbqt_line(1, "O", "HOH", "A", 1, 0.0, "OA") + "\n",
            encoding="utf-8",
        )
        with pytest.raises(DataValidationError, match="all atoms were filtered"):
            prepare_protein(pdbqt, tmp_path / "x.pdbqt")

    def test_no_conformer_mixing_per_residue(self, tmp_path: Path) -> None:
        """'first' altloc must keep ONE conformer per residue, not a
        hybrid of A and B sidechain atoms (a chemically impossible
        receptor)."""
        pdbqt = tmp_path / "alt.pdbqt"
        pdbqt.write_text(
            "\n".join(
                [
                    atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
                    atom_line(2, "CB", "ALA", "A", 1, 1.0, 0.0, 0.0, "C", altloc="A"),
                    atom_line(3, "CB", "ALA", "A", 1, 9.9, 0.0, 0.0, "C", altloc="B"),
                    atom_line(4, "HB", "ALA", "A", 1, 9.8, 1.0, 0.0, "C", altloc="B"),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report = prepare_protein(pdbqt, tmp_path / "out.pdbqt")
        # residue conformer letter is A (first seen): CA + CB-A survive,
        # CB-B and HB-B are dropped -- no A/B hybrid
        assert report.altloc_removed == 2
        assert report.output_atoms == 2

    def test_multi_model_keeps_first_model(self, tmp_path: Path) -> None:
        pdbqt = tmp_path / "nmr.pdbqt"
        pdbqt.write_text(
            "\n".join(
                [
                    "MODEL        1",
                    pdbqt_line(1, "CA", "ALA", "A", 1, 0.1, "C"),
                    pdbqt_line(2, "CB", "ALA", "A", 1, 0.1, "C"),
                    "ENDMDL",
                    "MODEL        2",
                    pdbqt_line(3, "CA", "ALA", "A", 1, 0.2, "C"),
                    pdbqt_line(4, "CB", "ALA", "A", 1, 0.2, "C"),
                    "ENDMDL",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report = prepare_protein(pdbqt, tmp_path / "out.pdbqt")
        assert report.output_atoms == 2  # only MODEL 1
        assert report.altloc_removed == 0  # model 2 is not "alt-locations"
        assert any("MODEL 1 of 2" in note for note in report.notes)

    def test_bad_altloc_argument_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "r.pdbqt"
        src.write_text(pdbqt_line(1, "CA", "ALA", "A", 1, 0.1, "C") + "\n")
        for bad in ("bb", "", "1", "aa"):
            with pytest.raises(DataValidationError, match="default_altloc"):
                prepare_protein(src, tmp_path / "x.pdbqt", default_altloc=bad)

    def test_chain_stripped_and_ter_between_chains(self, tmp_path: Path) -> None:
        pdbqt = tmp_path / "two.pdbqt"
        pdbqt.write_text(
            "\n".join(
                [
                    pdbqt_line(1, "CA", "ALA", "A", 1, 0.1, "C"),
                    pdbqt_line(2, "CA", "GLY", "B", 1, 0.1, "C"),
                    pdbqt_line(3, "CB", "GLY", "B", 1, 0.1, "C"),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        # chain selection with surrounding whitespace works...
        report = prepare_protein(pdbqt, tmp_path / "b.pdbqt", chain=" B ")
        assert report.chains == ["B"]
        assert report.output_atoms == 2
        # ...and keeping both chains emits a TER at the chain break
        both = prepare_protein(pdbqt, tmp_path / "ab.pdbqt")
        assert both.output_atoms == 3
        assert both.chains == ["A", "B"]
        text = (tmp_path / "ab.pdbqt").read_text()
        assert "\nTER" in text, "chain break TER not emitted"


# ---------------------------------------------------------------------------
# prepare_protein — the real Meeko pipeline (needs the 1CRN fixture)
# ---------------------------------------------------------------------------


class TestProteinMeekoPipeline:
    @pytest.fixture(scope="class")
    def crambin_report(self, tmp_path_factory: pytest.TempPathFactory) -> ProteinReport:
        out = tmp_path_factory.mktemp("receptor") / "r.pdbqt"
        return prepare_protein(CRAMBIN, out)

    def test_report_contents(self, crambin_report: ProteinReport) -> None:
        assert crambin_report.mode == "meeko"
        assert crambin_report.input_atoms == 327
        assert crambin_report.residues == 46
        assert crambin_report.chains == ["A"]
        # polar hydrogens added by the templates
        assert crambin_report.atom_types.get("HD", 0) == 67
        assert crambin_report.output_atoms == 394
        assert crambin_report.charge_min < 0 < crambin_report.charge_max

    def test_written_receptor_is_valid_pdbqt(self, crambin_report: ProteinReport) -> None:
        from pharmaml.docking.preparation import extract_pdbqt_atom_types

        types = extract_pdbqt_atom_types(crambin_report.output_pdbqt)
        assert set(types) >= {"A", "C", "HD", "N", "OA", "SA"}
        assert crambin_report.atom_types.keys() == set(types)

    def test_wet_pdb_full_cleaning(self, tmp_path: Path) -> None:
        """1CRN + waters + ion + sulfate + altloc clone -> full audit."""
        lines = CRAMBIN.read_text(encoding="utf-8", errors="ignore").splitlines()
        atoms = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
        wet = list(atoms)
        for i in range(3):  # three waters
            wet.append(
                atom_line(900 + i, "O", "HOH", "A", 90 + i, 1.0, 1.0, 1.0, "O", record="HETATM")
            )
        wet.append(atom_line(950, "CL", "CL", "A", 95, 2.0, 2.0, 2.0, "Cl", record="HETATM"))
        clone = list(atoms[5])
        clone[16] = "B"
        wet.insert(6, "".join(clone))
        src = tmp_path / "wet.pdb"
        src.write_text("\n".join(wet) + "\nTER\nEND\n", encoding="utf-8")

        report = prepare_protein(src, tmp_path / "r.pdbqt")
        assert report.waters_removed == 3
        assert report.hetatm_removed == 1
        assert report.altloc_removed == 1
        assert report.input_atoms == 328
        assert report.output_atoms == 394  # identical to the clean run
        assert report.residues == 46

    def test_input_hydrogens_stripped_then_readded(self, tmp_path: Path) -> None:
        lines = CRAMBIN.read_text(encoding="utf-8", errors="ignore").splitlines()
        atoms = [line for line in lines if line.startswith("ATOM")]
        with_h = atoms + [
            atom_line(900, "HA", "THR", "A", 1, 0.5, 0.5, 0.5, "H"),
        ]
        src = tmp_path / "h.pdb"
        src.write_text("\n".join(with_h) + "\nEND\n", encoding="utf-8")
        report = prepare_protein(src, tmp_path / "r.pdbqt")
        assert report.hydrogens_removed == 1
        assert report.output_atoms == 394  # template hydrogens replace input ones

    def test_digit_leading_hydrogen_names_stripped(self, tmp_path: Path) -> None:
        """Legacy PDBs digit-prefix hydrogen names ('1HB', '2HB1') and omit
        the element column (line stops after the B-factor, length < 78).

        Pre-fix (27-A): the name-based element fallback took the first
        character of the atom name, so '1HB' inferred element '1' -- not
        'H' -- and the atom survived ``strip_hydrogens``.
        """
        lines = CRAMBIN.read_text(encoding="utf-8", errors="ignore").splitlines()
        atoms = [line for line in lines if line.startswith("ATOM")]
        legacy = [
            # name '1HB ' left-justified in cols 13-16, no element column
            "ATOM    900 1HB  THR A   1       0.500   0.500   0.500  1.00  0.00",
            "ATOM    901 2HB1 THR A   1       0.600   0.600   0.600  1.00  0.00",
        ]
        assert all(len(line) < 78 for line in legacy)  # element-column fallback
        src = tmp_path / "legacy_h.pdb"
        src.write_text("\n".join(atoms + legacy) + "\nEND\n", encoding="utf-8")
        report = prepare_protein(src, tmp_path / "r.pdbqt")
        assert report.hydrogens_removed == 2
        assert report.output_atoms == 394  # template hydrogens replace input ones

    def test_broken_residue_raises_docking_error(self, tmp_path: Path) -> None:
        from pharmaml.exceptions import DockingError

        src = tmp_path / "broken.pdb"
        src.write_text(
            atom_line(1, "N", "XXX", "A", 1, 0.0, 0.0, 0.0, "N") + "\nEND\n",
            encoding="utf-8",
        )
        with pytest.raises(DockingError, match="Meeko could not build"):
            prepare_protein(src, tmp_path / "x.pdbqt")


class TestProteinMmcifImport:
    """mmCIF (.cif/.mmcif) receptors are converted via gemmi, then take
    the exact same Meeko cleaning path as .pdb input."""

    @pytest.fixture(scope="class")
    def crambin_cif(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        gemmi = pytest.importorskip("gemmi")
        structure = gemmi.read_structure(str(CRAMBIN))
        structure.setup_entities()
        cif_path = tmp_path_factory.mktemp("cif") / "1crn.cif"
        structure.make_mmcif_document().write_file(str(cif_path))
        return cif_path

    @pytest.fixture(scope="class")
    def crambin_ref(self, tmp_path_factory: pytest.TempPathFactory) -> ProteinReport:
        out = tmp_path_factory.mktemp("cif_ref") / "r.pdbqt"
        return prepare_protein(CRAMBIN, out)

    def test_cif_matches_pdb_pipeline(
        self, crambin_cif: Path, tmp_path: Path, crambin_ref: ProteinReport
    ) -> None:
        report = prepare_protein(crambin_cif, tmp_path / "r.pdbqt")
        assert report.mode == "meeko"
        assert report.input_atoms == 327
        assert report.output_atoms == crambin_ref.output_atoms
        assert report.atom_types == crambin_ref.atom_types
        assert report.residues == crambin_ref.residues
        assert any("mmCIF" in note for note in report.notes)

    def test_mmcif_extension_accepted(self, crambin_cif: Path, tmp_path: Path) -> None:
        renamed = tmp_path / "1crn.mmcif"
        renamed.write_text(crambin_cif.read_text(encoding="utf-8"), encoding="utf-8")
        report = prepare_protein(renamed, tmp_path / "r.pdbqt")
        assert report.mode == "meeko"
        assert report.output_atoms == 394

    def test_cif_without_gemmi_raises_dependency_error(
        self, crambin_cif: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        # 'None' in sys.modules makes any later `import gemmi` fail fast
        monkeypatch.setitem(sys.modules, "gemmi", None)
        with pytest.raises(DependencyError, match="gemmi"):
            prepare_protein(crambin_cif, tmp_path / "r.pdbqt")

    def test_non_cif_content_rejected(self, tmp_path: Path) -> None:
        src = tmp_path / "fake.cif"
        src.write_text(
            "this is definitely not mmCIF content\n", encoding="utf-8"
        )
        with pytest.raises(DataValidationError, match="mmCIF"):
            prepare_protein(src, tmp_path / "r.pdbqt")


# ---------------------------------------------------------------------------
# distribute_receptor — the adder.py upgrade
# ---------------------------------------------------------------------------


class TestDistributeReceptor:
    def _campaign(self, tmp_path: Path) -> Path:
        root = tmp_path / "campaign"
        for name in ("01_aspirin", "02_caffeine"):
            (root / name).mkdir(parents=True)
        return root

    def _receptor(self, tmp_path: Path) -> Path:
        path = tmp_path / "r.pdbqt"
        path.write_text(pdbqt_line(1, "CA", "ALA", "A", 1, 0.1, "C") + "\n")
        return path

    def test_copies_into_every_folder(self, tmp_path: Path) -> None:
        root = self._campaign(tmp_path)
        receptor = self._receptor(tmp_path)
        targets = distribute_receptor(receptor, root)
        assert len(targets) == 2
        assert all(t.name == "r.pdbqt" for t in targets)
        assert (root / "01_aspirin" / "r.pdbqt").is_file()

    def test_idempotent_skip_existing(self, tmp_path: Path) -> None:
        root = self._campaign(tmp_path)
        receptor = self._receptor(tmp_path)
        distribute_receptor(receptor, root)
        # modify one copy; the default second call must NOT overwrite it
        marker = root / "01_aspirin" / "r.pdbqt"
        marker.write_text("MUTATED\n")
        targets = distribute_receptor(receptor, root)
        assert marker.read_text() == "MUTATED\n"
        assert len(targets) == 2
        # overwrite=True refreshes it
        distribute_receptor(receptor, root, overwrite=True)
        assert marker.read_text() != "MUTATED\n"

    def test_custom_filename(self, tmp_path: Path) -> None:
        root = self._campaign(tmp_path)
        receptor = self._receptor(tmp_path)
        targets = distribute_receptor(receptor, root, filename="receptor.pdbqt")
        assert all(t.name == "receptor.pdbqt" for t in targets)

    def test_missing_receptor_raises(self, tmp_path: Path) -> None:
        root = self._campaign(tmp_path)
        with pytest.raises(DataValidationError, match="receptor file not found"):
            distribute_receptor(tmp_path / "ghost.pdbqt", root)

    def test_missing_campaign_raises(self, tmp_path: Path) -> None:
        receptor = self._receptor(tmp_path)
        with pytest.raises(DataValidationError, match="campaign directory not found"):
            distribute_receptor(receptor, tmp_path / "nowhere")

    def test_empty_campaign_returns_empty_list(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        receptor = self._receptor(tmp_path)
        assert distribute_receptor(receptor, root) == []


# ---------------------------------------------------------------------------
# CLI — docking prep-ligands / prep-protein
# ---------------------------------------------------------------------------


class TestDockingPrepCli:
    def test_prep_ligands_cli(self, tmp_path: Path, capsys) -> None:
        from pharmaml.cli import main

        csv = tmp_path / "ligands.csv"
        csv.write_text("name,SMILES\naspirin,CCO\nbad,zzz\n", encoding="utf-8")
        out = tmp_path / "campaign"
        code = main(
            ["docking", "prep-ligands", "--ligands", str(csv), "--out", str(out)]
        )
        assert code == 1  # failures -> exit 1
        stdout = capsys.readouterr().out
        assert "Ligand library" in stdout
        assert "Ligand batch preparation" in stdout
        assert "failed ligands ->" in stdout
        assert (out / "failed_smiles.csv").is_file()
        assert (out / "01_aspirin" / "l.pdbqt").is_file()

    def test_prep_ligands_cli_clean_exit_zero(self, tmp_path: Path, capsys) -> None:
        from pharmaml.cli import main

        csv = tmp_path / "ligands.csv"
        csv.write_text("name,SMILES\nethanol,CCO\n", encoding="utf-8")
        code = main(
            [
                "docking", "prep-ligands", "--ligands", str(csv),
                "--out", str(tmp_path / "ok"), "--naming", "index",
            ]
        )
        assert code == 0
        assert (tmp_path / "ok" / "1" / "l.pdbqt").is_file()

    def test_prep_ligands_cli_bad_naming(self, tmp_path: Path, capsys) -> None:
        from pharmaml.cli import main

        csv = tmp_path / "ligands.csv"
        csv.write_text("SMILES\nCCO\n", encoding="utf-8")
        # argparse validates the choices and exits with code 2
        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "docking", "prep-ligands", "--ligands", str(csv),
                    "--out", str(tmp_path / "x"), "--naming", "nope",
                ]
            )
        assert excinfo.value.code == 2
        assert "invalid choice" in capsys.readouterr().err

    def test_prep_protein_cli(self, tmp_path: Path, capsys) -> None:
        from pharmaml.cli import main

        out = tmp_path / "r.pdbqt"
        code = main(
            [
                "docking", "prep-protein",
                "--protein", str(CRAMBIN),
                "--out", str(out),
                "--chain", "A",
                "--altloc", "first",
            ]
        )
        assert code == 0
        stdout = capsys.readouterr().out
        assert "Protein preparation audit" in stdout
        assert "394 atoms" in stdout
        assert out.is_file()

    def test_prep_protein_cli_lowercase_altloc_normalized(self, tmp_path: Path) -> None:
        from pharmaml.cli import main

        # 'b' should be normalized to 'B' instead of dropping everything
        lines = CRAMBIN.read_text(encoding="utf-8", errors="ignore").splitlines()
        atoms = [l for l in lines if l.startswith("ATOM")]
        clone = list(atoms[5])
        clone[16] = "B"
        src = tmp_path / "alt.pdb"
        src.write_text(
            "\n".join(atoms[:6] + ["".join(clone)] + atoms[6:]) + "\nEND\n",
            encoding="utf-8",
        )
        code = main(
            [
                "docking", "prep-protein",
                "--protein", str(src),
                "--out", str(tmp_path / "r.pdbqt"),
                "--altloc", "b",
            ]
        )
        assert code == 0

    def test_prep_protein_cli_bad_altloc(self, tmp_path: Path, capsys) -> None:
        from pharmaml.cli import main

        code = main(
            [
                "docking", "prep-protein",
                "--protein", str(CRAMBIN),
                "--out", str(tmp_path / "r.pdbqt"),
                "--altloc", "zz",
            ]
        )
        assert code == 1
        assert "default_altloc" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# smiles_to_pdbqt -- charge hint, determinism, unknown-element guard
# ---------------------------------------------------------------------------


def _pdbqt_charges(path: Path) -> list[float]:
    """Partial-charge column (cols 71-76) of every ATOM/HETATM record."""
    return [
        float(line[70:76])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]


class TestSmilesToPdbqt:
    """Direct contracts of the SMILES -> 3D PDBQT converter."""

    def test_charge_hint_warns_and_stays_neutral(self, tmp_path: Path) -> None:
        # Regression: `charge` was dead code -- the docstring promised a
        # protonation-state hint while the neutral state was always built.
        with pytest.warns(UserWarning, match="charge hint is currently ignored"):
            path = smiles_to_pdbqt("CCO", tmp_path, name="ethanol", charge=-1)
        assert path.is_file()
        # the generated ligand is the *neutral* state: partial charges
        # sum to ~0 (PDBQT rounds each to 3 decimals, so allow dust)
        assert sum(_pdbqt_charges(path)) == pytest.approx(0.0, abs=5e-3)

    def test_neutral_call_emits_no_charge_warning(self, tmp_path: Path) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            smiles_to_pdbqt("CCO", tmp_path, name="ethanol")
            smiles_to_pdbqt("CCO", tmp_path, name="ethanol2", charge=0)
        assert not [w for w in caught if "charge hint" in str(w.message)]

    def test_default_seed_is_byte_identical(self, tmp_path: Path) -> None:
        # Regression: ETKDG() had no randomSeed -- two calls produced
        # different conformers, making campaign trees nondeterministic.
        first = smiles_to_pdbqt("CCCCO", tmp_path / "a", name="lig")
        second = smiles_to_pdbqt("CCCCO", tmp_path / "b", name="lig")
        assert first.read_text(encoding="utf-8") == second.read_text(
            encoding="utf-8"
        )

    def test_explicit_seed_reproducible_and_distinct(
        self, tmp_path: Path
    ) -> None:
        one = smiles_to_pdbqt("CCCCO", tmp_path / "one", name="lig", seed=11)
        one_again = smiles_to_pdbqt(
            "CCCCO", tmp_path / "one_again", name="lig", seed=11
        )
        two = smiles_to_pdbqt("CCCCO", tmp_path / "two", name="lig", seed=12)
        assert one.read_text(encoding="utf-8") == one_again.read_text(
            encoding="utf-8"
        )
        assert one.read_text(encoding="utf-8") != two.read_text(
            encoding="utf-8"
        )

    def test_unknown_atom_type_raises_instead_of_dropping(
        self, tmp_path: Path
    ) -> None:
        # Regression: boronic acid OB(O)c1ccccc1 silently lost its boron
        # (net PDBQT partial charge -0.488) while the docstring promised
        # a DockingError for atom-type repair failures.
        with pytest.raises(DockingError, match="'B'"):
            smiles_to_pdbqt("OB(O)c1ccccc1", tmp_path, name="boron")
        assert not (tmp_path / "boron.pdbqt").exists()  # no corrupted ligand


class TestLibraryPreparationGuards:
    """prepare_ligand_library: per-ligand failure recording + seed."""

    def test_unsupported_element_recorded_not_batch_fatal(
        self, tmp_path: Path
    ) -> None:
        # A boron ligand fails with a typed DockingError; the batch keeps
        # going and the report row records ok=False with the reason.
        csv = tmp_path / "ligands.csv"
        csv.write_text(
            "name,SMILES\n"
            "ethanol,CCO\n"
            "boronate,OB(O)c1ccccc1\n"
            "aspirin,CC(=O)Oc1ccccc1C(=O)O\n",
            encoding="utf-8",
        )
        report = prepare_ligand_library(csv, tmp_path / "campaign")
        assert report.n_prepared == 2
        failed = report.failed_frame
        assert list(failed["name"]) == ["boronate"]
        assert not failed.iloc[0]["ok"]
        assert "'B'" in failed.iloc[0]["error"]
        # other ligands are unaffected, the failed folder is cleaned up
        assert (tmp_path / "campaign" / "01_ethanol" / "l.pdbqt").is_file()
        assert (tmp_path / "campaign" / "03_aspirin" / "l.pdbqt").is_file()
        assert not (tmp_path / "campaign" / "02_boronate").exists()

    def test_seed_is_threaded_through_the_batch(self, tmp_path: Path) -> None:
        csv = tmp_path / "ligands.csv"
        csv.write_text("name,SMILES\nflexible,CCCCOCCN\n", encoding="utf-8")
        first = prepare_ligand_library(csv, tmp_path / "run1", seed=42)
        second = prepare_ligand_library(csv, tmp_path / "run2", seed=42)
        assert first.n_prepared == second.n_prepared == 1
        text_a = (tmp_path / "run1" / "01_flexible" / "l.pdbqt").read_text(
            encoding="utf-8"
        )
        text_b = (tmp_path / "run2" / "01_flexible" / "l.pdbqt").read_text(
            encoding="utf-8"
        )
        assert text_a == text_b  # same seed -> identical campaign tree


class TestCoordinateTokenFallback:
    """Whitespace-token fallback of the PDB coordinate reader.

    Chain-less records (no chain ID column) shift every token one slot
    left; fixed offsets read ``(y, z, occupancy)`` instead of
    ``(x, y, z)``.  The fallback must locate the
    ``[resseq, x, y, z]`` window instead.
    """

    def test_chainless_record_gets_true_xyz(self, tmp_path: Path) -> None:
        # single atom: the grid centre IS the atom's (x, y, z)
        ligand = tmp_path / "chainless.pdbqt"
        ligand.write_text(
            "ATOM 1 CA ALA 1 11.100 22.200 33.300 1.00 C\n", encoding="utf-8"
        )
        box = compute_grid_box(ligand, padding=0.0)
        assert (box.center_x, box.center_y, box.center_z) == pytest.approx(
            (11.1, 22.2, 33.3)
        )

    def test_chained_record_unchanged(self, tmp_path: Path) -> None:
        # the chained variant already read (x, y, z); it must stay that way
        ligand = tmp_path / "chained.pdbqt"
        ligand.write_text(
            "ATOM 1 CA ALA A 1 11.100 22.200 33.300 1.00 C\n", encoding="utf-8"
        )
        box = compute_grid_box(ligand, padding=0.0)
        assert (box.center_x, box.center_y, box.center_z) == pytest.approx(
            (11.1, 22.2, 33.3)
        )


class TestLigandStructureGeneration:
    """generate_ligand_structures: the ligprep.py pipeline, library edition.

    The 2026-10 "new folder" upload's ligand generation/ligprep.py: tautomer
    canonicalisation, multi-conformer ETKDG, MMFF-with-UFF-fallback, the
    lowest-energy conformer written as a PDB with CONECT records, the
    metal-centre skip list, and per-row error accounting.
    """

    def test_pipeline_writes_pdb_with_conect(self, tmp_path: Path) -> None:
        from pharmaml.docking.ligands import generate_ligand_structures

        csv = tmp_path / "lib.csv"
        csv.write_text(
            "name,SMILES\nethanol,CCO\naspirin,CC(=O)Oc1ccccc1C(=O)O\n",
            encoding="utf-8",
        )
        out = tmp_path / "structures"
        frame = generate_ligand_structures(
            csv, out, num_confs=4, write_mol2=False
        )
        assert len(frame) == 2
        assert bool(frame["ok"].all())
        for name in ("ethanol", "aspirin"):
            pdb = out / f"{name}.pdb"
            assert pdb.is_file()
            text = pdb.read_text(encoding="utf-8")
            assert text.startswith("HEADER")
            assert "COMPND    " in text
            assert "CONECT" in text and text.rstrip().endswith("END")
            assert "ATOM  " in text
        # the report carries the pipeline's own accounting columns
        assert set(frame["forcefield"]) <= {"RDKit MMFF", "RDKit UFF"}
        assert (frame["conformers"] == 4).all()
        assert (frame["atoms"] > 0).all()
        assert (frame["energy"] != "").all()
        # tautomers were canonicalised
        assert frame["tautomer"].astype(str).ne("").all()

    def test_unsupported_metal_centres_skipped_not_fatal(self, tmp_path: Path) -> None:
        from pharmaml.docking.ligands import (
            UNSUPPORTED_LIGAND_ATOMS,
            generate_ligand_structures,
        )

        assert "Zn" in UNSUPPORTED_LIGAND_ATOMS and "Fe" in UNSUPPORTED_LIGAND_ATOMS
        csv = tmp_path / "lib.csv"
        csv.write_text(
            "name,SMILES\nethanol,CCO\nzinc_complex,[Zn+2]\n",
            encoding="utf-8",
        )
        frame = generate_ligand_structures(
            csv, tmp_path / "out", num_confs=2, write_mol2=False
        )
        assert len(frame) == 2
        assert bool(frame.iloc[0]["ok"])
        assert not bool(frame.iloc[1]["ok"])
        assert "Zn" in frame.iloc[1]["error"]
        assert frame.iloc[1]["pdb"] == ""
        # the valid ligand still landed
        assert (tmp_path / "out" / "ethanol.pdb").is_file()

    def test_invalid_smiles_rides_along_as_failed_row(self, tmp_path: Path) -> None:
        from pharmaml.docking.ligands import generate_ligand_structures

        csv = tmp_path / "lib.csv"
        csv.write_text("name,SMILES\nbroken,C1CC\nethanol,CCO\n", encoding="utf-8")
        frame = generate_ligand_structures(
            csv, tmp_path / "out", num_confs=2, write_mol2=False
        )
        assert len(frame) == 2
        # invalid rows ride along at the end (prepare_ligand_library's
        # convention -- the report covers the whole library)
        broken = frame[frame["ligand"] == "broken"].iloc[0]
        ethanol = frame[frame["ligand"] == "ethanol"].iloc[0]
        assert not bool(broken["ok"])
        assert "SMILES" in broken["error"]
        assert bool(ethanol["ok"])
        assert (tmp_path / "out" / "ethanol.pdb").is_file()

    def test_mol2_graceful_without_openbabel(self, tmp_path: Path) -> None:
        from pharmaml.docking.ligands import generate_ligand_structures

        csv = tmp_path / "lib.csv"
        csv.write_text("name,SMILES\nethanol,CCO\n", encoding="utf-8")
        frame = generate_ligand_structures(csv, tmp_path / "out", num_confs=2)
        # whether OpenBabel is installed or not, the PDB lands and the
        # row tells the truth about the MOL2 twin
        assert bool(frame.iloc[0]["ok"])
        assert (tmp_path / "out" / "ethanol.pdb").is_file()
        mol2 = frame.iloc[0]["mol2"]
        if mol2:
            assert Path(str(mol2)).is_file()
        else:
            # part20 §"Known Findings A": the backend names the tool with its
            # official "Open Babel" spelling; accept both it and the legacy
            # "OpenBabel" form so the assertion checks the semantics (the
            # skip reason names the missing optional dependency) without
            # pinning the whitespace.
            error_text = str(frame.iloc[0]["error"])
            assert "Open Babel" in error_text or "OpenBabel" in error_text

    def test_parameter_guards(self, tmp_path: Path) -> None:
        from pharmaml.docking.ligands import generate_ligand_structures

        csv = tmp_path / "lib.csv"
        csv.write_text("name,SMILES\nethanol,CCO\n", encoding="utf-8")
        with pytest.raises(DataValidationError, match="num_confs"):
            generate_ligand_structures(csv, tmp_path / "a", num_confs=0)
        with pytest.raises(DataValidationError, match="workers"):
            generate_ligand_structures(csv, tmp_path / "b", workers=0)

    def test_workers_pool_writes_the_same_files(self, tmp_path: Path) -> None:
        from pharmaml.docking.ligands import generate_ligand_structures

        csv = tmp_path / "lib.csv"
        csv.write_text("name,SMILES\nethanol,CCO\naspirin,CC(=O)Oc1ccccc1C(=O)O\n")
        frame = generate_ligand_structures(
            csv, tmp_path / "out", num_confs=2, write_mol2=False, workers=2
        )
        assert len(frame) == 2
        assert bool(frame["ok"].all())
        assert (tmp_path / "out" / "ethanol.pdb").is_file()
        assert (tmp_path / "out" / "aspirin.pdb").is_file()


class TestMglToolsConversion:
    """convert_pdb_folder_to_pdbqt: the ligprep_qt.py batch converter.

    The 2026-10 "new folder" upload's ligand generation/ligprep_qt.py:
    prepare_ligand4.py over a folder of PDBs into numbered sub-folders,
    per-ligand timeout, and the conversion log beside the tree.
    """

    def _mgl_stubs(self, tmp_path: Path, *, body: str) -> tuple[Path, Path]:
        """A fake MGLTools pair: executable 'python' + prepare_ligand4.py."""
        mgl_python = tmp_path / "fake_mgl_python.sh"
        mgl_python.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        mgl_python.chmod(0o755)
        adt_script = tmp_path / "prepare_ligand4.py"
        adt_script.write_text("# fake AutoDockTools script\n", encoding="utf-8")
        return mgl_python, adt_script

    def _pdb_folder(self, tmp_path: Path) -> Path:
        src = tmp_path / "pdbs"
        src.mkdir()
        (src / "a.pdb").write_text("ATOM  1  C   LIG A   1      0.000   0.000   0.000\n")
        (src / "b.pdb").write_text("ATOM  1  C   LIG A   1      1.000   0.000   0.000\n")
        (src / "notes.txt").write_text("not a ligand\n")
        return src

    @pytest.mark.skipif(os.name == "nt", reason="POSIX shell fake MGL python")
    def test_numbered_subfolders_and_log(self, tmp_path: Path) -> None:
        from pharmaml.docking.preparation import convert_pdb_folder_to_pdbqt

        stub = (
            'out=""\n'
            'while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift 2;; *) shift;; esac; done\n'
            'printf "REMARK stub pdbqt\\n" > "$out"\n'
        )
        mgl_python, adt_script = self._mgl_stubs(tmp_path, body=stub)
        src = self._pdb_folder(tmp_path)
        out = tmp_path / "pdbqt_out"
        frame = convert_pdb_folder_to_pdbqt(
            src, out, mgl_python, adt_script, timeout_s=10.0
        )
        assert len(frame) == 2  # the .txt is ignored
        assert list(frame["ligand"]) == ["a", "b"]
        assert list(frame["subfolder"]) == ["1", "2"]
        assert list(frame["status"]) == ["converted", "converted"]
        assert (out / "1" / "a.pdbqt").is_file()
        assert (out / "2" / "b.pdbqt").is_file()
        log = out / "conversion_log.txt"
        assert log.is_file()
        text = log.read_text(encoding="utf-8")
        assert "SUCCESSFUL CONVERSIONS:" in text
        assert "a" in text and "b" in text
        assert "All PDB files converted successfully." in text

    @pytest.mark.skipif(os.name == "nt", reason="POSIX shell fake MGL python")
    def test_timeout_marks_the_row_and_cleans_up(self, tmp_path: Path) -> None:
        from pharmaml.docking.preparation import convert_pdb_folder_to_pdbqt

        mgl_python, adt_script = self._mgl_stubs(tmp_path, body="sleep 30")
        src = self._pdb_folder(tmp_path)
        out = tmp_path / "pdbqt_out"
        frame = convert_pdb_folder_to_pdbqt(
            src, out, mgl_python, adt_script, timeout_s=1.0
        )
        assert list(frame["status"]) == ["timeout", "timeout"]
        assert "timed out" in frame.iloc[0]["error"]
        assert frame.iloc[0]["pdbqt"] == ""
        text = (out / "conversion_log.txt").read_text(encoding="utf-8")
        assert "TIMED OUT FILES (1+ seconds):" in text

    @pytest.mark.skipif(os.name == "nt", reason="POSIX shell fake MGL python")
    def test_failing_script_records_nonzero_exit(self, tmp_path: Path) -> None:
        from pharmaml.docking.preparation import convert_pdb_folder_to_pdbqt

        mgl_python, adt_script = self._mgl_stubs(tmp_path, body="exit 3")
        src = self._pdb_folder(tmp_path)
        frame = convert_pdb_folder_to_pdbqt(
            src, tmp_path / "out", mgl_python, adt_script, timeout_s=10.0
        )
        assert list(frame["status"]) == ["failed", "failed"]
        assert "exited with 3" in frame.iloc[0]["error"]
        text = (tmp_path / "out" / "conversion_log.txt").read_text(encoding="utf-8")
        assert "FAILED CONVERSIONS:" in text

    def test_missing_mgl_python_is_a_typed_error(self, tmp_path: Path) -> None:
        from pharmaml.docking.preparation import convert_pdb_folder_to_pdbqt

        src = self._pdb_folder(tmp_path)
        with pytest.raises(DockingError, match="mgl_python"):
            convert_pdb_folder_to_pdbqt(
                src,
                tmp_path / "out",
                tmp_path / "no_such_pythonsh",
                tmp_path / "no_such_script.py",
            )

    def test_missing_source_directory_is_a_typed_error(self, tmp_path: Path) -> None:
        from pharmaml.docking.preparation import convert_pdb_folder_to_pdbqt

        with pytest.raises(DockingError, match="source directory"):
            convert_pdb_folder_to_pdbqt(
                tmp_path / "ghost_src", tmp_path / "out", "/bin/sh", "/bin/sh"
            )

    @pytest.mark.skipif(os.name == "nt", reason="POSIX shell fake MGL python")
    def test_empty_folder_writes_the_honest_log(self, tmp_path: Path) -> None:
        from pharmaml.docking.preparation import convert_pdb_folder_to_pdbqt

        mgl_python, adt_script = self._mgl_stubs(tmp_path, body="true")
        src = tmp_path / "empty"
        src.mkdir()
        frame = convert_pdb_folder_to_pdbqt(
            src, tmp_path / "out", mgl_python, adt_script
        )
        assert len(frame) == 0
        text = (tmp_path / "out" / "conversion_log.txt").read_text(encoding="utf-8")
        assert "No PDB files found." in text
