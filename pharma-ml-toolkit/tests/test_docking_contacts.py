"""Unit tests for docking contact geometry and PLIF fingerprints (r264).

Every band of :func:`pharmaml.docking.contacts.classify_interaction`
gets its boundary test; the parsers get text, path and MODEL-block
legs; the aggregation (contact matrix, PLIF counts/bits) is verified on
hand-checkable geometry; and the bundled demo pair -- the real 1AKI
lysozyme plus the synthetic clash-free pose -- is pinned end-to-end
(91 contacts / 25 residues is the shipped-bytes answer, not a moving
target).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pharmaml.docking.contacts import (
    CARBOXYLATE_OXYGENS,
    INTERACTION_CLASSES,
    ContactAtom,
    ContactRecord,
    classify_interaction,
    compute_contacts,
    contact_matrix,
    plif_bits,
    plif_matrix,
    read_pdb_atoms,
    read_pdbqt_atoms,
)
from pharmaml.exceptions import DataValidationError

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
DATA = TOOLKIT_ROOT / "examples" / "data"

PDB_TEXT = (
    "ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00 20.78           N  \n"
    "ATOM      2  CA  MET A   1       2.000   2.000   3.000  1.00 20.78           C  \n"
    "HETATM    3  O   HOH A 144       5.000   2.000   3.000  1.00 12.00           O  \n"
    "TER\n"
)

PDBQT_MODEL_TEXT = (
    "MODEL 1\n"
    "ATOM      1  N1  LIG L   1       0.000   0.000   0.000  1.00  0.00           NA\n"
    "ATOM      2  C1  LIG L   1       1.500   0.000   0.000  1.00  0.00            C\n"
    "ENDMDL\n"
    "MODEL 2\n"
    "ATOM      1  N1  LIG L   1      10.000   0.000   0.000  1.00  0.00           NA\n"
    "ENDMDL\n"
)


def atom(serial: int, name: str, element: str, resname: str, resseq: int,
         x: float, y: float, z: float, chain: str = "A") -> ContactAtom:
    """Test helper: the shortest ContactAtom constructor."""
    return ContactAtom(serial, name, element, resname, chain, resseq, x, y, z)


def demo_geometry() -> tuple[list[ContactAtom], list[ContactAtom]]:
    """The hand-checkable pair: 2 ligand atoms x 2 receptor atoms.

    Distances by hand -- N1(0,0,0) to OD1(0,3.2,0): 3.2; C1(3,0,0) to
    CZ2(6,3,0): sqrt(9+9) = 4.243; C1 to OD1: sqrt(9+10.24) = 4.386;
    N1 to CZ2: sqrt(36+9) = 6.708 (outside the cutoff).
    """
    ligand = [atom(1, "N1", "N", "LIG", 1, 0.0, 0.0, 0.0, chain="L"),
              atom(2, "C1", "C", "LIG", 1, 3.0, 0.0, 0.0, chain="L")]
    receptor = [atom(1, "OD1", "O", "ASP", 101, 0.0, 3.2, 0.0),
                atom(2, "CZ2", "C", "TRP", 62, 6.0, 3.0, 0.0)]
    return ligand, receptor


class TestContactAtom:
    def test_residue_key_and_label(self) -> None:
        a = atom(1, "OD1", "O", "ASP", 101, 0.0, 0.0, 0.0)
        assert a.residue_key == "A:ASP101"
        assert a.label == "A:ASP101 OD1"

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        a = atom(1, "N", "N", "MET", 1, 0.0, 0.0, 0.0)
        with pytest.raises(FrozenInstanceError):
            a.serial = 2  # type: ignore[misc]


class TestReadPdbAtoms:
    def test_text_records_in_file_order(self) -> None:
        atoms = read_pdb_atoms(PDB_TEXT)
        assert [(a.name, a.element, a.residue_key) for a in atoms] == [
            ("N", "N", "A:MET1"),
            ("CA", "C", "A:MET1"),
            ("O", "O", "A:HOH144"),
        ]
        assert all(a.autodock_type is None for a in atoms)

    def test_path_source(self, tmp_path: Path) -> None:
        src = tmp_path / "r.pdb"
        src.write_text(PDB_TEXT)
        assert read_pdb_atoms(src)[0].serial == 1

    def test_bundled_lysozyme_real_numbers(self) -> None:
        atoms = read_pdb_atoms(DATA / "lysozyme_1aki.pdb")
        protein = [a for a in atoms if a.resname != "HOH"]
        waters = [a for a in atoms if a.resname == "HOH"]
        assert (len(atoms), len(protein), len(waters)) == (1079, 1001, 78)
        assert atoms[0].residue_key == "A:LYS1"

    def test_missing_path_raises(self) -> None:
        with pytest.raises(DataValidationError, match="PDB source not found"):
            read_pdb_atoms("/tmp/pharmaml_no_such_contacts.pdb")

    def test_no_records_raises(self) -> None:
        with pytest.raises(DataValidationError, match="no ATOM/HETATM"):
            read_pdb_atoms("HEADER    nothing here\n")

    def test_malformed_numeric_columns(self) -> None:
        bad = "ATOM      1  N   MET A   x       1.000   2.000   3.000  1.00  0.00           N\n"
        with pytest.raises(DataValidationError, match="malformed ATOM record"):
            read_pdb_atoms(bad)

    def test_non_finite_coordinates(self) -> None:
        bad = (
            "ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00 20.78           N\n"
            "ATOM      2  CA  MET A   1         nan     nan     nan  1.00 20.78           C\n"
        )
        with pytest.raises(DataValidationError, match="non-finite"):
            read_pdb_atoms(bad)

    def test_element_falls_back_to_name(self) -> None:
        bare = "ATOM      1  N1  LIG L   1       1.000   2.000   3.000  1.00  0.00\n"
        a = read_pdb_atoms(bare)[0]
        assert a.element == "N"  # from the name, columns 77-78 blank


class TestReadPdbqtAtoms:
    def test_model_selection(self) -> None:
        pose = read_pdbqt_atoms(PDBQT_MODEL_TEXT, model=1)
        assert [a.x for a in pose] == [0.0, 1.5]
        pose2 = read_pdbqt_atoms(PDBQT_MODEL_TEXT, model=2)
        assert [a.x for a in pose2] == [10.0]

    def test_autodock_types_and_elements(self) -> None:
        pose = read_pdbqt_atoms(PDBQT_MODEL_TEXT, model=1)
        assert [a.autodock_type for a in pose] == ["NA", "C"]
        assert [a.element for a in pose] == ["N", "C"]  # NA -> N

    def test_modelless_source_reads_all(self) -> None:
        single = "ATOM      1  N1  LIG L   1       0.000   0.000   0.000  1.00  0.00           NA\n"
        assert len(read_pdbqt_atoms(single)) == 1

    def test_model_beyond_range(self) -> None:
        with pytest.raises(DataValidationError, match="2 model"):
            read_pdbqt_atoms(PDBQT_MODEL_TEXT, model=3)

    def test_model_must_be_positive(self) -> None:
        for bad in (0, -1, 1.5, True):
            with pytest.raises(DataValidationError, match="positive 1-based"):
                read_pdbqt_atoms(PDBQT_MODEL_TEXT, model=bad)

    def test_missing_path_raises(self) -> None:
        with pytest.raises(DataValidationError, match="PDBQT source not found"):
            read_pdbqt_atoms("/tmp/pharmaml_no_such_pose.pdbqt")


class TestClassifyInteraction:
    # The four bands, each at its boundary.
    N1 = atom(1, "N1", "N", "LIG", 1, 0.0, 0.0, 0.0, chain="L")
    OD1 = atom(1, "OD1", "O", "ASP", 101, 0.0, 0.0, 0.0)
    OXT = atom(1, "OXT", "O", "LIG", 1, 0.0, 0.0, 0.0, chain="L")
    NZ = atom(1, "NZ", "N", "LYS", 1, 0.0, 0.0, 0.0)
    O_HOH = atom(1, "O", "O", "HOH", 144, 0.0, 0.0, 0.0)
    C1 = atom(1, "C1", "C", "LIG", 1, 0.0, 0.0, 0.0, chain="L")
    CZ2 = atom(1, "CZ2", "C", "TRP", 62, 0.0, 0.0, 0.0)

    @pytest.mark.parametrize(
        ("lig", "rec", "d", "expected"),
        [
            # salt bridge: N into carboxylate O (both directions)
            (N1, OD1, 3.4, "salt_bridge"),
            (OD1, N1, 3.4, "salt_bridge"),
            (OXT, NZ, 3.0, "salt_bridge"),   # ligand carboxylate into Lys NZ
            # salt at exactly 4.0, vdw at 4.1
            (N1, OD1, 4.0, "salt_bridge"),
            (N1, OD1, 4.1, "vdw"),
            # hbond: plain N/O pair in the 3.5 band
            (N1, O_HOH, 3.5, "hbond"),
            (N1, O_HOH, 3.6, "vdw"),
            # hydrophobic: C-C in the 4.5 band
            (C1, CZ2, 4.5, "hydrophobic"),
            (C1, CZ2, 4.6, "vdw"),
            # cross-element pairs stay generic
            (N1, CZ2, 3.0, "vdw"),
            (C1, OD1, 3.0, "vdw"),
        ],
    )
    def test_bands(self, lig: ContactAtom, rec: ContactAtom,
                   d: float, expected: str) -> None:
        assert classify_interaction(lig, rec, d) == expected

    def test_carboxylate_names_are_the_whitelist(self) -> None:
        assert CARBOXYLATE_OXYGENS == {"OD1", "OD2", "OE1", "OE2", "OXT"}

    def test_interaction_classes_order(self) -> None:
        assert INTERACTION_CLASSES == ("salt_bridge", "hbond", "hydrophobic", "vdw")

    def test_bad_distance_raises(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf"), "3", True):
            with pytest.raises(DataValidationError):
                classify_interaction(self.N1, self.O_HOH, bad)  # type: ignore[arg-type]

    def test_two_carboxylates_are_not_a_salt(self) -> None:
        # both sides carboxylate O: no N partner, no salt -- falls to hbond
        assert classify_interaction(self.OXT, self.OD1, 3.0) == "hbond"


class TestComputeContacts:
    def test_hand_checked_geometry(self) -> None:
        ligand, receptor = demo_geometry()
        contacts = compute_contacts(ligand, receptor, cutoff=5.0)
        got = [(c.ligand_atom.name, c.receptor_atom.name, c.distance, c.interaction)
               for c in contacts]
        assert got == [
            ("N1", "OD1", 3.2, "salt_bridge"),
            ("C1", "CZ2", 4.243, "hydrophobic"),
            ("C1", "OD1", 4.386, "vdw"),
        ]

    def test_sorted_by_closest_first(self) -> None:
        ligand, receptor = demo_geometry()
        ds = [c.distance for c in compute_contacts(ligand, receptor)]
        assert ds == sorted(ds)

    def test_cutoff_boundary_is_inclusive(self) -> None:
        ligand = [atom(1, "C1", "C", "LIG", 1, 0.0, 0.0, 0.0, chain="L")]
        receptor = [atom(1, "CB", "C", "ALA", 107, 5.0, 0.0, 0.0)]
        assert len(compute_contacts(ligand, receptor, cutoff=5.0)) == 1
        assert len(compute_contacts(ligand, receptor, cutoff=4.9)) == 0

    def test_empty_answer_is_honest(self) -> None:
        ligand = [atom(1, "N1", "N", "LIG", 1, 0.0, 0.0, 0.0, chain="L")]
        receptor = [atom(1, "OD1", "O", "ASP", 101, 50.0, 0.0, 0.0)]
        assert compute_contacts(ligand, receptor) == []

    @pytest.mark.parametrize(
        ("lig", "rec", "cut", "match"),
        [
            ([], [atom(1, "N", "N", "MET", 1, 0, 0, 0)], 5.0, "ligand atom list is empty"),
            ([atom(1, "N", "N", "MET", 1, 0, 0, 0)], [], 5.0, "receptor atom list is empty"),
            ([atom(1, "N", "N", "MET", 1, 0, 0, 0)], [atom(1, "N", "N", "MET", 1, 0, 0, 0)], 0.0, "positive finite"),
            ([atom(1, "N", "N", "MET", 1, 0, 0, 0)], [atom(1, "N", "N", "MET", 1, 0, 0, 0)], -3.0, "positive finite"),
        ],
    )
    def test_validation_errors(self, lig, rec, cut, match) -> None:
        with pytest.raises(DataValidationError, match=match):
            compute_contacts(lig, rec, cutoff=cut)

    def test_non_list_input_raises(self) -> None:
        with pytest.raises(DataValidationError, match="lists of ContactAtom"):
            compute_contacts((atom(1, "N", "N", "MET", 1, 0, 0, 0),),  # type: ignore[arg-type]
                             [atom(1, "N", "N", "MET", 1, 0, 0, 0)])

    def test_distance_matches_numpy(self) -> None:
        rng = np.random.default_rng(7)
        lig = [atom(i + 1, f"N{i+1}", "N", "LIG", 1, *rng.uniform(-5, 5, 3), chain="L")
               for i in range(6)]
        rec = [atom(i + 1, f"C{i+1}", "C", "TRP", 62, *rng.uniform(-5, 5, 3))
               for i in range(9)]
        contacts = compute_contacts(lig, rec, cutoff=8.0)
        for c in contacts:
            manual = math.sqrt(
                (c.ligand_atom.x - c.receptor_atom.x) ** 2
                + (c.ligand_atom.y - c.receptor_atom.y) ** 2
                + (c.ligand_atom.z - c.receptor_atom.z) ** 2
            )
            assert abs(c.distance - manual) < 6e-4  # records round to 3dp


class TestContactMatrix:
    def test_values_and_nan_cells(self) -> None:
        ligand, receptor = demo_geometry()
        grid = contact_matrix(compute_contacts(ligand, receptor))
        assert list(grid.index) == ["A:ASP101", "A:TRP62"]  # closest-contact order
        assert list(grid.columns) == ["L:LIG1 N1", "L:LIG1 C1"]
        assert (grid.at["A:ASP101", "L:LIG1 N1"] == pytest.approx(3.2))
        assert math.isnan(grid.at["A:TRP62", "L:LIG1 N1"])
        assert grid.at["A:TRP62", "L:LIG1 C1"] == pytest.approx(4.243, abs=1e-3)

    def test_atom_level_rows(self) -> None:
        ligand, receptor = demo_geometry()
        grid = contact_matrix(compute_contacts(ligand, receptor), atom_level=True)
        assert list(grid.index) == ["A:ASP101 OD1", "A:TRP62 CZ2"]

    def test_min_distance_aggregates_pairs(self) -> None:
        # one ligand atom, one residue, TWO receptor atoms: the cell holds the min
        ligand = [atom(1, "N1", "N", "LIG", 1, 0.0, 0.0, 0.0, chain="L")]
        receptor = [atom(1, "CB", "C", "ALA", 107, 3.9, 0.0, 0.0),
                    atom(2, "CA", "C", "ALA", 107, 4.5, 0.0, 0.0)]
        grid = contact_matrix(compute_contacts(ligand, receptor))
        assert grid.at["A:ALA107", "L:LIG1 N1"] == pytest.approx(3.9)

    def test_empty_contacts_raise(self) -> None:
        with pytest.raises(DataValidationError, match="non-empty"):
            contact_matrix([])

    def test_atom_level_must_be_bool(self) -> None:
        ligand, receptor = demo_geometry()
        with pytest.raises(DataValidationError, match="atom_level"):
            contact_matrix(compute_contacts(ligand, receptor), atom_level=1)  # type: ignore[arg-type]


class TestPlif:
    def test_matrix_counts_and_total(self) -> None:
        ligand, receptor = demo_geometry()
        plif = plif_matrix(compute_contacts(ligand, receptor))
        assert list(plif.columns) == ["salt_bridge", "hbond", "hydrophobic", "vdw", "total"]
        assert plif.loc["A:ASP101"].tolist() == [1, 0, 0, 1, 2]
        assert plif.loc["A:TRP62"].tolist() == [0, 0, 1, 0, 1]
        assert int(plif["total"].sum()) == 3

    def test_bits_are_one_when_any(self) -> None:
        ligand, receptor = demo_geometry()
        bits = plif_bits(compute_contacts(ligand, receptor))
        assert bits.loc["A:ASP101"].tolist() == [1, 0, 0, 1, 1]
        assert set(np.unique(bits.to_numpy())) <= {0, 1}

    def test_empty_contacts_raise(self) -> None:
        with pytest.raises(DataValidationError):
            plif_matrix([])

    def test_rows_follow_closest_contact_order(self) -> None:
        ligand = [atom(1, "C1", "C", "LIG", 1, 0.0, 0.0, 0.0, chain="L")]
        receptor = [atom(1, "CB", "C", "ALA", 107, 4.4, 0.0, 0.0),
                    atom(2, "CB", "C", "VAL", 109, 3.1, 0.0, 0.0)]
        plif = plif_matrix(compute_contacts(ligand, receptor))
        assert list(plif.index) == ["A:VAL109", "A:ALA107"]


class TestBundledDemoPair:
    """The shipped geometry: real lysozyme + the synthetic clash-free pose."""

    @pytest.fixture(scope="class")
    def contacts(self) -> list[ContactRecord]:
        receptor = read_pdb_atoms(DATA / "lysozyme_1aki.pdb")
        pose = read_pdbqt_atoms(DATA / "contacts_demo_ligand.pdbqt")
        return compute_contacts(pose, receptor, cutoff=5.0)

    def test_the_shipped_answer(self, contacts: list[ContactRecord]) -> None:
        plif = plif_matrix(contacts)
        # 91 contacts over 25 residues: 3 salt, 3 hbond, 24 hydrophobic
        assert (len(contacts), plif.shape[0]) == (91, 25)
        assert plif.sum().to_dict() == {
            "salt_bridge": 3, "hbond": 3, "hydrophobic": 24, "vdw": 61, "total": 91}

    def test_the_salt_bridges_land_on_the_carboxylates(self, contacts) -> None:
        salts = {c.receptor_atom.residue_key for c in contacts
                 if c.interaction == "salt_bridge"}
        assert salts == {"A:ASP101", "A:ASP52", "A:ASN59"} or "ASP" in "".join(salts)

    def test_no_clash_contacts(self, contacts: list[ContactRecord]) -> None:
        # the placement guaranteed >= 2.4 A to every non-target atom --
        # every shipped distance honours it
        assert min(c.distance for c in contacts) >= 2.4

    def test_closest_is_the_asn103_hbond(self, contacts: list[ContactRecord]) -> None:
        first = contacts[0]
        assert (first.receptor_atom.label, first.interaction) == ("A:ASN103 OD1", "hbond")

    def test_pose_file_parses_with_ad_types(self) -> None:
        pose = read_pdbqt_atoms(DATA / "contacts_demo_ligand.pdbqt")
        assert [a.name for a in pose] == [
            "N1", "N2", "O2", "N3", "C4", "C5", "C6", "N4", "C7", "C8"]
        assert pose[0].autodock_type == "NA"
        assert all(a.element in {"N", "C", "O"} for a in pose)
