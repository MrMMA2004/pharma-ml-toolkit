"""Tests for molecular standardization and 3D ligand generation.

Mirrors the upgrade of ``others/Docking/ligand generation/nigga.py``
and ``nigga2.py``: tautomer canonicalization (keto/enol collapse),
unsupported-metal screening, ETKDG multi-conformer embedding with MMFF
(UFF fallback) optimisation, lowest-energy conformer selection and PDB
export with CONECT connectivity records and the AutoDock B-factor
convention.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem

from pharmaml.chem.standardize import (
    NEUTRALIZE_PATTERNS,
    REINVENT_DEFAULT_ELEMENTS,
    UNSUPPORTED_METALS,
    PdbLigandReport,
    canonicalize_tautomer,
    reinvent_standardize_mol,
    reinvent_standardize_smiles,
    smiles_to_pdb,
)
from pharmaml.exceptions import DataValidationError, DependencyError


# (round-249) the module-level ``logging.disable(logging.CRITICAL)``
# this fixture replaces leaked for the WHOLE pytest process and silently
# broke ``caplog`` assertions in sibling modules sharing a pytest process
# (the nine-chunk gate caught it: test_format_converters before
# test_numpy_neural_checkpoints).  Same silence, correct scope: applied
# around each test of THIS module only, previous level restored after.
@pytest.fixture(autouse=True)
def _quiet_module_logging():
    prev = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(prev)

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"


# ---------------------------------------------------------------------------
# canonicalize_tautomer
# ---------------------------------------------------------------------------


class TestCanonicalizeTautomer:
    def test_keto_enol_pair_collapses_to_same_string(self) -> None:
        keto = canonicalize_tautomer("CC(=O)CC(C)=O")
        enol = canonicalize_tautomer("CC(O)=CC(C)=O")
        assert keto == enol
        assert keto == "CC(=O)CC(C)=O"

    def test_roundtrip_is_stable(self) -> None:
        once = canonicalize_tautomer(ASPIRIN)
        twice = canonicalize_tautomer(once)
        assert once == twice

    def test_invalid_smiles_raises(self) -> None:
        with pytest.raises(DataValidationError, match="Cannot parse SMILES"):
            canonicalize_tautomer("not_a_smiles")

    def test_blank_smiles_raises(self) -> None:
        # 100x-19-a family fix: "" parsed as a valid 0-atom molecule, so the
        # tautomer canonicalisation silently returned "" instead of raising.
        with pytest.raises(DataValidationError, match="Cannot parse SMILES"):
            canonicalize_tautomer("")

    def test_non_string_smiles_raises_cleanly(self) -> None:
        # Same guard family: MolFromSmiles(None) escaped as a raw
        # Boost.ArgumentError before the isinstance check.
        with pytest.raises(DataValidationError, match="Cannot parse SMILES"):
            canonicalize_tautomer(None)  # type: ignore[arg-type]

    def test_rdkit_missing_raises_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "rdkit", None)
        monkeypatch.setitem(sys.modules, "rdkit.Chem", None)
        with pytest.raises(DependencyError, match="pip install rdkit"):
            canonicalize_tautomer("CCO")


# ---------------------------------------------------------------------------
# smiles_to_pdb
# ---------------------------------------------------------------------------


class TestSmilesToPdb:
    def test_aspirin_pipeline(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "aspirin.pdb"
        report = smiles_to_pdb(ASPIRIN, out, name="ASPIRIN", num_confs=3)

        assert isinstance(report, PdbLigandReport)
        assert out.is_file()  # parent directory auto-created
        content = out.read_text(encoding="utf-8")
        assert "CONECT" in content
        assert "ATOM" in content or "HETATM" in content  # RDKit writes HETATM for ligands
        assert report.forcefield in {"MMFF", "UFF"}
        assert report.num_conformers == 3
        assert report.num_atoms > 0
        assert report.num_bonds > 0
        assert report.output_path == str(out)
        assert report.name == "ASPIRIN"
        assert "ASPIRIN" in report.status()

    def test_default_b_factor_written(self, tmp_path: Path) -> None:
        out = tmp_path / "b.pdb"
        smiles_to_pdb("CCO", out, num_confs=2)
        atom_lines = [
            line
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.startswith(("ATOM", "HETATM"))
        ]
        assert atom_lines
        for line in atom_lines:  # PDB columns 61-66, the rar default of 20.0
            assert line[60:66] == " 20.00"

    def test_blank_smiles_raises_cleanly(self, tmp_path: Path) -> None:
        # 100x-19-a family fix: "" produced a raw "ValueError: molecule has
        # no atoms" from the embedder instead of the documented parse error.
        with pytest.raises(DataValidationError, match="Cannot parse SMILES"):
            smiles_to_pdb("", tmp_path / "blank.pdb")

    def test_unsupported_metal_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="Fe"):
            smiles_to_pdb("CC(=O)[O-].[Fe+2]", tmp_path / "iron.pdb", name="IRON")

    def test_unsupported_metals_constant(self) -> None:
        assert "Zn" in UNSUPPORTED_METALS
        assert "Fe" in UNSUPPORTED_METALS
        assert len(UNSUPPORTED_METALS) == 8

    def test_invalid_smiles_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="Cannot parse SMILES"):
            smiles_to_pdb("not_a_smiles", tmp_path / "bad.pdb")

    def test_canonicalize_false_keeps_input_tautomer(self, tmp_path: Path) -> None:
        out = tmp_path / "enol.pdb"
        report = smiles_to_pdb(
            "CC(O)=CC(C)=O", out, name="ENOL", num_confs=2, canonicalize=False
        )
        assert out.is_file()
        assert report.forcefield in {"MMFF", "UFF"}
        assert report.num_conformers >= 1

    def test_embedding_failure_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # smiles_to_pdb calls the defining module (rdDistGeom) directly --
        # AllChem only re-exports the same function -- so that is the
        # attribute the failure injection has to replace.
        from rdkit.Chem import rdDistGeom

        monkeypatch.setattr(rdDistGeom, "EmbedMultipleConfs", lambda *args, **kwargs: [])
        with pytest.raises(DataValidationError, match="no conformers"):
            smiles_to_pdb("CCO", tmp_path / "x.pdb")

    def test_rdkit_missing_raises_dependency_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "rdkit", None)
        monkeypatch.setitem(sys.modules, "rdkit.Chem", None)
        with pytest.raises(DependencyError, match="pip install rdkit"):
            smiles_to_pdb("CCO", tmp_path / "x.pdb")


class TestNumConfsValidation:
    """num_confs < 1 must fail up-front with an actionable message."""

    def test_zero_confs_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="num_confs must be at least 1"):
            smiles_to_pdb("CCO", tmp_path / "x.pdb", num_confs=0)

    def test_negative_confs_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="num_confs"):
            smiles_to_pdb("CCO", tmp_path / "x.pdb", num_confs=-5)

    def test_no_file_written_on_rejection(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError):
            smiles_to_pdb("CCO", tmp_path / "never.pdb", num_confs=0)
        assert not (tmp_path / "never.pdb").exists()


class TestReinventStandardize:
    """The REINVENT4 ``default`` chain (round 100x-164).

    Largest fragment -> RemoveHs -> salt removal -> the nine
    neutralisation SMARTS -> cleanup -> unwanted patterns -> the
    size/element window -- each step pinned against its REINVENT4
    semantics.
    """

    def test_salt_becomes_its_parent_acid(self) -> None:
        assert reinvent_standardize_smiles("CC(=O)[O-].[Na+]") == "CC(=O)O"

    def test_largest_fragment_wins(self) -> None:
        assert reinvent_standardize_smiles("CCO.CCCCCCCO") == "CCCCCCCO"

    def test_carboxylate_neutralises(self) -> None:
        assert reinvent_standardize_smiles("CC(=O)[O-]") == "CC(=O)O"

    def test_ammonium_neutralises(self) -> None:
        assert reinvent_standardize_smiles("CCCC[NH3+]") == "CCCCN"

    def test_unparseable_drops_with_warning(self) -> None:
        assert reinvent_standardize_smiles("not-a-smiles") is None

    def test_blank_drops(self) -> None:
        assert reinvent_standardize_smiles("   ") is None

    def test_too_small_rejected(self) -> None:
        assert reinvent_standardize_mol(Chem.MolFromSmiles("C")) is None

    def test_too_big_rejected(self) -> None:
        big = "C" * 71  # 71 heavy atoms: outside the (2, 70) window
        assert reinvent_standardize_mol(Chem.MolFromSmiles(big)) is None

    def test_out_of_vocabulary_element_rejected(self) -> None:
        # iodine is outside the classical Reinvent prior vocabulary
        assert reinvent_standardize_smiles("CI") is None

    def test_unwanted_pattern_rejected(self) -> None:
        # [SH] is on REINVENT4's unwanted-pairs list
        assert reinvent_standardize_mol(Chem.MolFromSmiles("C[SH]")) is None

    def test_custom_element_window(self) -> None:
        iodine_ok = reinvent_standardize_mol(
            Chem.MolFromSmiles("CI"),
            allowed_elements={6, 53},
            min_heavy_atoms=0,
            max_heavy_atoms=100,
        )
        assert iodine_ok is not None

    def test_neutralize_toggle(self) -> None:
        skipped = reinvent_standardize_mol(
            Chem.MolFromSmiles("CCCC[NH3+]"), neutralize=False
        )
        assert skipped is not None
        assert "+" in Chem.MolToSmiles(skipped)

    def test_the_nine_neutralize_patterns(self) -> None:
        assert len(NEUTRALIZE_PATTERNS) == 9
        assert all(
            len(pair) == 2 for pair in NEUTRALIZE_PATTERNS
        )

    def test_default_elements_are_the_prior_vocabulary(self) -> None:
        assert REINVENT_DEFAULT_ELEMENTS == frozenset({6, 7, 8, 9, 16, 17, 35})

    def test_none_mol_returns_none(self) -> None:
        assert reinvent_standardize_mol(None) is None
