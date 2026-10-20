"""Database-audit regression guards (100x-103).

The shipped example-data inventory is a *database*: 13 CSVs + the docking
campaign + 2 PDB files, every one pinned here.  The full 526-check audit
lives in ``scratch/audit_database_103.py`` (the one-shot auditor); these
tests pin the invariants that must never drift silently:

- structural integrity: shapes, ragged-row freedom, missing-cell budget
  (the *designed* plants only), trailing-newline convention
- chemical integrity: every SMILES RDKit-valid except the documented
  planted-invalid rows; descriptors match RDKit recomputation
- semantic integrity: label censuses, the assay-plausible pIC50 window,
  LN_IC50 physical range, PK AUC pins, campaign audit-case tally
- cross-file coverage: screened cell lines have expression, campaign
  folders match ligands.csv, walkthrough vocabulary == seed molecules

Pre-1.2.46 history this guards against repeating:
- walkthrough_compounds.csv carried ONE invalid SMILES ("CCO(C)C",
  trivalent oxygen) in all 180 rows -- zero validity, zero diversity;
- qsar_regression.csv shipped pIC50 -4.3..14.7 (IC50 of ~21 molar to
  ~20 femtomolar -- chemically impossible; now clipped to [3, 11]);
- pk_profile.csv drifted one repr-digit from its generator after a venv
  reset (the idempotent builder now catches that on every run).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")

from rdkit import Chem

from pharmaml.docking.parsing import parse_dlg_summary
from pharmaml.pharma.pk import auc_trapezoid

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
DATA = TOOLKIT_ROOT / "examples" / "data"

# The planted-invalid design (documented audit cases, NOT errors).
PLANTED_INVALID_SCREEN = {"not_a_smiles", "C1CC"}
PLANTED_INVALID_LIGAND = {"not_a_smiles"}

# The toolkit's 12-molecule seed universe (seed_molecules.csv order).
SEED_VOCAB = [
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "CC(=O)Nc1ccc(O)cc1",             # paracetamol
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",     # ibuprofen
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",     # caffeine
    "CN(C)C(=N)N=C(N)N",              # metformin
    "CN1CCCC1c1cccnc1",               # nicotine
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",  # warfarin
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C",     # theophylline
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",  # diclofenac
    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",   # naproxen
    "CC(C)NCC(O)COc1cccc2ccccc12",    # propranolol
    "CCN(CC)CC(=O)Nc1c(C)cccc1C",     # lidocaine
]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


class TestStructuralIntegrity:
    """Every shipped CSV parses cleanly with the designed missing-cell budget."""

    DESIGNED_NAN: ClassVar[dict[str, dict[str, int]]] = {
        "ligand_library.csv": {"SMILES": 1},      # the planted empty-SMILES row
        "cellline_screen.csv": {"LN_IC50": 1},    # the planted missing potency
    }

    def test_no_ragged_rows_anywhere(self) -> None:
        for path in sorted(DATA.rglob("*.csv")):
            lines = path.read_text(encoding="utf-8").splitlines()
            ncol = len(lines[0].split(","))
            ragged = [
                i + 1
                for i, ln in enumerate(lines[1:])
                if ln.strip() and len(ln.split(",")) != ncol
            ]
            assert not ragged, f"{path.name}: ragged rows {ragged[:5]}"

    def test_missing_cells_match_the_designed_budget(self) -> None:
        for path in sorted(DATA.rglob("*.csv")):
            frame = _read(path.relative_to(DATA).as_posix())
            nan = {c: int(frame[c].isna().sum()) for c in frame.columns if frame[c].isna().any()}
            expected = self.DESIGNED_NAN.get(path.name, {})
            assert nan == expected, f"{path.name}: missing cells {nan} != designed {expected}"

    def test_trailing_newline_convention(self) -> None:
        for path in sorted(DATA.rglob("*.csv")):
            raw = path.read_bytes()
            if path.name == "seed_molecules.csv":
                assert not raw.endswith(b"\n"), "seed_molecules.csv pins NO trailing newline"
            else:
                assert raw.endswith(b"\n") and not raw.endswith(b"\n\n"), path.name

    def test_no_stray_whitespace_or_crlf(self) -> None:
        for path in sorted(DATA.rglob("*.csv")):
            raw = path.read_bytes()
            assert b"\r" not in raw, f"{path.name}: CR bytes present"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            stray = [
                (col, val) for col in frame.columns for val in frame[col] if val != val.strip()
            ]
            assert not stray, f"{path.name}: whitespace in cells {stray[:4]}"


class TestChemicalIntegrity:
    """SMILES validity + descriptor agreement with RDKit recomputation."""

    def test_all_smiles_valid_except_planted(self) -> None:
        cases = [
            ("qsar_classification.csv", "SMILES", set()),
            ("walkthrough_compounds.csv", "SMILES", set()),
            ("library_with_descriptors.csv", "SMILES", set()),
            ("seed_molecules.csv", "smiles", set()),
            ("docking_campaign/ligands.csv", "SMILES", set()),
            ("screening_library.csv", "SMILES", PLANTED_INVALID_SCREEN),
            ("ligand_library.csv", "SMILES", PLANTED_INVALID_LIGAND),
        ]
        for name, col, planted in cases:
            frame = _read(name)
            invalid = {
                str(s) for s in frame[col].dropna() if Chem.MolFromSmiles(str(s)) is None
            }
            assert invalid == planted, f"{name}: invalid SMILES {sorted(invalid)} != {sorted(planted)}"

    def test_walkthrough_vocabulary_is_the_seed_universe(self) -> None:
        frame = _read("walkthrough_compounds.csv")
        assert set(frame["SMILES"]) == set(SEED_VOCAB)
        assert frame["SMILES"].nunique() == 12, "the 12-molecule vocabulary must stay diverse"

    def test_library_descriptors_match_rdkit(self) -> None:
        from rdkit.Chem import Descriptors

        frame = _read("library_with_descriptors.csv")
        for _, row in frame.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            assert mol is not None, row["SMILES"]
            assert abs(Descriptors.MolWt(mol) - row["MolWt"]) < 0.02, row["SMILES"]
            assert abs(Descriptors.MolLogP(mol) - row["MolLogP"]) < 0.02, row["SMILES"]
            assert abs(Descriptors.TPSA(mol) - row["TPSA"]) < 0.1, row["SMILES"]


class TestSemanticIntegrity:
    """Censuses, physical ranges and the PK/campaign pins."""

    def test_qsar_classification_census_and_ids(self) -> None:
        frame = _read("qsar_classification.csv")
        assert frame.shape == (420, 11)
        assert frame["activity"].value_counts().to_dict() == {
            "active": 118, "weak": 155, "inactive": 147,
        }
        assert frame["compound_id"].is_unique
        assert not frame.duplicated().any()

    def test_qsar_regression_pIC50_assay_window(self) -> None:
        frame = _read("qsar_regression.csv")
        assert frame.shape == (260, 21)
        # The assay-plausible window: IC50 between 1 mM and 10 pM.
        assert frame["pIC50"].between(3.0, 11.0).all(), (
            f"pIC50 outside [3, 11]: {frame['pIC50'].min()} .. {frame['pIC50'].max()}"
        )

    def test_qsar_multiclass_census(self) -> None:
        frame = _read("qsar_multiclass.csv")
        assert frame.shape == (360, 21)
        assert frame["Target"].value_counts().to_dict() == {
            "inactive": 125, "weak": 102, "moderate": 79, "strong": 54,
        }

    def test_walkthrough_compounds_census(self) -> None:
        frame = _read("walkthrough_compounds.csv")
        assert frame.shape == (180, 8)
        assert frame["activity"].value_counts().to_dict() == {
            "inactive": 88, "weak": 53, "active": 39,
        }

    def test_cellline_screen_plants_and_ranges(self) -> None:
        screen = _read("cellline_screen.csv")
        assert len(screen) == 185
        assert screen["cell_line"].nunique() == 23
        assert screen["tissue"].nunique() == 11
        # one tissue per line, no contradictions
        assert (screen.groupby("cell_line")["tissue"].nunique() == 1).all()
        # the designed plants: exactly one missing potency + one duplicate pair
        assert int(screen["LN_IC50"].isna().sum()) == 1
        dupe = screen[screen.duplicated(subset=["cell_line", "compound"], keep=False)]
        assert len(dupe) == 2, "the planted assay-repeat pair"
        # physical range: ln(IC50 µM) from 1 nM to 1 mM
        values = screen["LN_IC50"].dropna()
        assert values.between(-7, 7).all(), f"LN_IC50 {values.min()} .. {values.max()}"

    def test_cellline_expression_shape_and_coverage(self) -> None:
        expr = _read("cellline_expression.csv")
        assert expr.shape == (23, 301)  # cell_line + 298 GENE + EGFR + KRT8
        screen = _read("cellline_screen.csv")
        assert set(screen["cell_line"]) == set(expr["cell_line"]), "screen/expression coverage"

    def test_pk_profile_structure_and_auc_pins(self) -> None:
        pk = _read("pk_profile.csv")
        assert len(pk) == 49
        times = pk["time_h"].to_numpy()
        conc = pk["conc_ng_ml"].to_numpy()
        assert (np.diff(times) > 0).all() and times[0] == 0.0 and times[-1] == 24.0
        assert (conc >= 0).all()
        auc_linear = auc_trapezoid(times, conc)
        positive = conc > 0
        auc_log = auc_trapezoid(times[positive], conc[positive], method="log")
        assert abs(auc_linear - 17.9607) < 5e-4, f"AUC linear {auc_linear:.4f}"
        assert abs(auc_log - 17.2536) < 5e-4, f"AUC log {auc_log:.4f}"

    def test_seed_molecules_byte_pin(self) -> None:
        raw = (DATA / "seed_molecules.csv").read_bytes()
        assert len(raw) == 443, f"seed_molecules.csv is {len(raw)} B, pinned 443 B"
        seed = _read("seed_molecules.csv")
        assert len(seed) == 12
        assert list(seed["smiles"]) == SEED_VOCAB


class TestDockingCampaign:
    """14 folders: 12 complete runs + the 2 planted audit cases."""

    def test_campaign_tally(self) -> None:
        camp = DATA / "docking_campaign"
        folders = sorted(p for p in camp.iterdir() if p.is_dir())
        assert len(folders) == 14
        statuses: dict[str, int] = {}
        for path in folders:
            dlg, glg, readme = path / "r.dlg", path / "r.glg", path / "README.md"
            if dlg.is_file() and dlg.stat().st_size == 682 and glg.is_file():
                status = "OK"
            elif readme.is_file() and not dlg.is_file():
                status = "MISSING_LOG"
            elif dlg.is_file() and dlg.stat().st_size == 65:
                status = "INCOMPLETE"
            else:
                status = "UNEXPECTED"
            statuses[status] = statuses.get(status, 0) + 1
        assert statuses == {"OK": 12, "MISSING_LOG": 1, "INCOMPLETE": 1}, statuses

    def test_folders_match_ligands_csv(self) -> None:
        ligands = _read("docking_campaign/ligands.csv")
        assert len(ligands) == 14
        camp = DATA / "docking_campaign"
        folder_names = {p.name.split("_", 1)[1] for p in camp.iterdir() if p.is_dir()}
        assert folder_names == set(ligands["name"])

    def test_aspirin_summary_pin(self) -> None:
        summary = parse_dlg_summary(DATA / "docking_campaign" / "01_aspirin" / "r.dlg")
        got = summary["n_clusters"] if isinstance(summary, dict) else summary.n_clusters
        assert got == 5


class TestPdbFiles:
    """The two shipped PDB structures stay byte-pinned and parseable."""

    @pytest.mark.parametrize(
        ("path", "pin_bytes"),
        [
            (DATA / "lysozyme_1aki.pdb", 116_397),
            (TOOLKIT_ROOT / "tests" / "data" / "1crn.pdb", 49_491),
        ],
        ids=["lysozyme-1aki", "crn-1crn"],
    )
    def test_byte_pin_and_parseable(self, path: Path, pin_bytes: int) -> None:
        assert path.stat().st_size == pin_bytes, f"{path.name}: {path.stat().st_size} B"
        mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
        assert mol is not None and mol.GetNumAtoms() > 0, f"{path.name} unparseable"
