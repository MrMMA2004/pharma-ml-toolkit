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

rdkit = pytest.importorskip("rdkit")  # noqa: F841  (gates the whole module)

from pharmaml.chem.standardize import (  # noqa: E402
    UNSUPPORTED_METALS,
    PdbLigandReport,
    canonicalize_tautomer,
    smiles_to_pdb,
)
from pharmaml.exceptions import DataValidationError, DependencyError  # noqa: E402

logging.disable(logging.CRITICAL)

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
        from rdkit.Chem import AllChem

        monkeypatch.setattr(AllChem, "EmbedMultipleConfs", lambda *args, **kwargs: [])
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
