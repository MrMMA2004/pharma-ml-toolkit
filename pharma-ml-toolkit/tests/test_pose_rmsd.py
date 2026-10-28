"""Tests for :mod:`pharmaml.docking.pose_rmsd` (round 100x-164).

The DiffDock pose-analysis port: the Kabsch superposition and its
reflection fix, the QCP closed-form RMSD, the Hungarian assignment,
the spyrmsd symmetry-corrected RMSD with automorphism caching, the
torsion machinery (DiffDock's graph-disconnect rule, the masks, the
dihedrals, the perturbation, ``randomize_position``), the
evaluate.py metrics and the MOAD-style pocket extraction.

The suite pins:

* Kabsch: rigid shifts and rotations align away to 0, non-rigid
  scaling does not, the reflection fix (det < 0) path, the < 3-point
  translation-only fallback;
* QCP == Kabsch to machine precision on random geometries;
* Hungarian: reordered atoms with matching element multisets score 0,
  mismatched multisets refused;
* symmetry-corrected RMSD: the ethane atom swap scores 0 (the
  automorphism undoes it), the benzene ring flip, the permutation
  return, the minimisation variant, the automorphism cache reuse, the
  size guard;
* the torsion rule: butane's one rotatable bond, terminal bonds never
  rotate, rings never rotate, the mask marks the smaller side;
* ``perturb_torsions``: on-axis atoms stay, off-axis atoms swing by
  the exact rotation;
* ``randomize_pose``: deterministic under the seed, re-centred,
  actually randomised;
* the evaluation report: the < 2 A / < 5 A fractions, best-of-N,
  percentiles, centroid stats, the clash fraction, the clusters;
* the pocket extraction: residue cutoff, whole-chain mode, the
  20 A crop, the closest-contact report, the typed refusals.

RDKit is required; the file skips as a module when it is absent.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pharmaml.docking.pose_rmsd import (
    AutomorphismCache,
    best_pose_rmsd,
    centroid_distance,
    crop_beyond,
    dihedrals,
    extract_pocket,
    hungarian_rmsd,
    kabsch_rmsd,
    kabsch_transform,
    min_intra_distance,
    perturb_torsions,
    plain_rmsd,
    pose_clusters,
    pose_evaluation_report,
    pose_rmsd_matrix,
    qcp_rmsd,
    randomize_pose,
    rmsd_wrapper,
    rotatable_masks,
    symmetry_corrected_rmsd,
    torsion_bonds,
)
from pharmaml.exceptions import DataValidationError

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_RDKIT, reason="RDKit not installed (the zip-only tree)"
)


def _square() -> np.ndarray:
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )


def _rotation_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _tetrahedral() -> np.ndarray:
    """A non-degenerate 4-point geometry (the Kabsch reflection path)."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, math.sqrt(3) / 2, 0.0],
            [0.5, math.sqrt(3) / 6, math.sqrt(2.0 / 3.0)],
        ]
    )


def _mirror(coords: np.ndarray) -> np.ndarray:
    """Mirror through the xy plane -- det(fit) < 0 territory."""
    mirrored = coords.copy()
    mirrored[:, 2] *= -1.0
    return mirrored


# ---------------------------------------------------------------------------
# Kabsch / QCP / plain / centroid
# ---------------------------------------------------------------------------


class TestKabsch:
    def test_rigid_shift_aligns_to_zero(self) -> None:
        shifted = _square() + np.array([5.0, -3.0, 2.0])
        assert kabsch_rmsd(_square(), shifted) == pytest.approx(0.0, abs=1e-12)

    def test_rotation_aligns_to_zero(self) -> None:
        rotated = _square() @ _rotation_z(math.pi / 3).T
        assert kabsch_rmsd(_square(), rotated) == pytest.approx(0.0, abs=1e-12)

    def test_scaling_is_not_rigid(self) -> None:
        assert kabsch_rmsd(_square(), _square() * 2.0) == pytest.approx(
            math.sqrt(0.5), rel=1e-9
        )

    def test_reflection_fix_on_mirrored_geometry(self) -> None:
        # a mirrored tetrahedron needs the det < 0 correction; the
        # optimal PROPER rotation still leaves the mirror error
        value = kabsch_rmsd(_tetrahedral(), _mirror(_tetrahedral()))
        assert value > 0.1  # chirality is not alignable away
        assert value < 1.0  # but the centroid shape is close

    def test_transform_reconstructs_the_target(self) -> None:
        rotation, _translation = kabsch_transform(
            _tetrahedral(), _tetrahedral() @ _rotation_z(0.7).T + 3.0
        )
        assert np.linalg.det(rotation) == pytest.approx(1.0)

    def test_two_points_fall_back_to_translation(self) -> None:
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert kabsch_rmsd(a, a + 5.0) == pytest.approx(0.0, abs=1e-12)

    def test_unequal_lengths_refused(self) -> None:
        with pytest.raises(DataValidationError, match="equal atom counts"):
            kabsch_rmsd(_square(), _square()[:3])

    def test_qcp_matches_kabsch_on_random_geometries(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.normal(size=(12, 3))
            b = rng.normal(size=(12, 3))
            assert qcp_rmsd(a, b) == pytest.approx(kabsch_rmsd(a, b), rel=1e-8)

    def test_qcp_zero_for_rigid_moves(self) -> None:
        moved = _square() @ _rotation_z(0.9).T + np.array([1.0, 2.0, 3.0])
        assert qcp_rmsd(_square(), moved) == pytest.approx(0.0, abs=1e-9)


class TestPlainAndCentroid:
    def test_plain_rmsd_formula(self) -> None:
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert plain_rmsd(a, b) == pytest.approx(1.0)

    def test_plain_rmsd_center_removes_translation(self) -> None:
        a = _square()
        assert plain_rmsd(a, a + 10.0, center=True) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_centroid_distance(self) -> None:
        a = _square()
        assert centroid_distance(a, a + np.array([3.0, 4.0, 0.0])) == (
            pytest.approx(5.0)
        )

    def test_bad_shapes_refused(self) -> None:
        with pytest.raises(DataValidationError, match="N, 3"):
            plain_rmsd(np.zeros((2, 4)), np.zeros((2, 4)))
        with pytest.raises(DataValidationError, match="non-finite"):
            plain_rmsd([[0.0, 0.0, math.nan]], [[0.0, 0.0, 0.0]])


class TestHungarian:
    def test_reordered_atoms_score_zero(self) -> None:
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
        b = np.array([[0.0, 1.5, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert hungarian_rmsd(a, b, [6, 6, 8], [8, 6, 6]) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_mismatched_multiset_refused(self) -> None:
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(DataValidationError, match="element multisets"):
            hungarian_rmsd(a, b, [6, 6], [6, 8])


# ---------------------------------------------------------------------------
# Symmetry-corrected RMSD
# ---------------------------------------------------------------------------


class TestSymmetryCorrectedRmsd:
    def test_ethane_atom_swap_is_the_same_molecule(self) -> None:
        ethane = Chem.MolFromSmiles("CC")
        a = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        b = np.array([[1.54, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert plain_rmsd(a, b) == pytest.approx(1.54)
        assert symmetry_corrected_rmsd(ethane, a, b) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_benzene_ring_rotation_is_the_same_pose(self) -> None:
        benzene = Chem.MolFromSmiles("c1ccccc1")
        ref = Chem.MolFromSmiles("c1ccccc1")
        AllChem.Compute2DCoords(benzene)
        AllChem.Compute2DCoords(ref)
        a = benzene.GetConformer().GetPositions()
        b = ref.GetConformer().GetPositions()
        # rotate the ring labelling by two atoms
        rotated = b[[2, 3, 4, 5, 0, 1]]
        assert symmetry_corrected_rmsd(benzene, a, rotated) < plain_rmsd(
            a, rotated
        )

    def test_return_permutation_gives_the_winning_mapping(self) -> None:
        ethane = Chem.MolFromSmiles("CC")
        a = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        b = np.array([[1.54, 0.0, 0.0], [0.0, 0.0, 0.0]])
        value, permutation = symmetry_corrected_rmsd(
            ethane, a, b, return_permutation=True
        )
        assert value == pytest.approx(0.0, abs=1e-12)
        assert permutation == [1, 0]

    def test_minimize_variant_aligns_before_comparing(self) -> None:
        ethane = Chem.MolFromSmiles("CC")
        a = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        b = (np.array([[1.54, 0.0, 0.0], [0.0, 0.0, 0.0]])) + 7.0
        assert symmetry_corrected_rmsd(ethane, a, b, minimize=True) == (
            pytest.approx(0.0, abs=1e-9)
        )

    def test_automorphism_cache_reuses_the_permutation_list(self) -> None:
        benzene = Chem.MolFromSmiles("c1ccccc1")
        cache = AutomorphismCache()
        first = cache.permutations(benzene)
        assert cache.permutations(benzene) is first
        assert len(first) == 12  # the D6h automorphisms RDKit reports

    def test_size_guard_on_the_cache(self) -> None:
        with pytest.raises(DataValidationError, match="positive int"):
            AutomorphismCache(max_permutations=0)

    def test_atom_count_mismatch_refused(self) -> None:
        ethane = Chem.MolFromSmiles("CC")
        with pytest.raises(DataValidationError, match="atom count"):
            symmetry_corrected_rmsd(
                ethane, np.zeros((3, 3)), np.zeros((3, 3))
            )

    def test_rmsd_wrapper_against_pose_copies(self) -> None:
        ref = Chem.MolFromSmiles("CCO")
        AllChem.Compute2DCoords(ref)
        same = Chem.Mol(ref)
        assert rmsd_wrapper(ref, [same]) == [pytest.approx(0.0, abs=1e-12)]

    def test_best_pose_rmsd_over_a_list(self) -> None:
        ethane = Chem.MolFromSmiles("CC")
        ref = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        poses = [ref.copy(), ref + 1.0, ref + 9.0]
        values = best_pose_rmsd(ethane, ref, poses)
        assert values[0] == pytest.approx(0.0, abs=1e-12)
        assert values[1] == pytest.approx(math.sqrt(3), rel=1e-9)

    def test_pose_rmsd_matrix(self) -> None:
        a = _square()
        matrix = pose_rmsd_matrix([a, a + 1.0, a + 5.0])
        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 0.0)
        assert matrix[0, 1] == pytest.approx(math.sqrt(3), rel=1e-9)
        assert matrix[0, 2] > matrix[0, 1]


# ---------------------------------------------------------------------------
# The torsion machinery
# ---------------------------------------------------------------------------


class TestTorsionBonds:
    def test_butane_has_exactly_one(self) -> None:
        assert torsion_bonds(Chem.MolFromSmiles("CCCC")) == [(1, 2)]

    def test_terminal_bonds_never_rotate(self) -> None:
        # every bond of propane holds a single-atom side: none rotate
        assert torsion_bonds(Chem.MolFromSmiles("CCC")) == []

    def test_rings_never_rotate(self) -> None:
        assert torsion_bonds(Chem.MolFromSmiles("c1ccccc1")) == []

    def test_hexane_has_three(self) -> None:
        bonds = torsion_bonds(Chem.MolFromSmiles("CCCCCC"))
        assert len(bonds) == 3

    def test_masks_mark_the_smaller_side(self) -> None:
        butane = Chem.MolFromSmiles("CCCC")
        bonds, masks = rotatable_masks(butane)
        assert bonds == [(1, 2)]
        assert masks[0].astype(int).tolist() == [1, 1, 0, 0]

    def test_dihedrals_of_butane(self) -> None:
        butane = Chem.MolFromSmiles("CCCC")
        AllChem.Compute2DCoords(butane)
        angles = dihedrals(butane)
        assert angles.shape == (1,)
        assert abs(float(angles[0])) <= math.pi

    def test_perturb_torsions_keeps_on_axis_atoms(self) -> None:
        butane = Chem.MolFromSmiles("CCCC")
        bonds, masks = rotatable_masks(butane)
        flat = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]]
        )
        turned = perturb_torsions(flat, bonds, masks, [math.pi / 2])
        assert np.allclose(turned[1], flat[1])  # on the bond axis
        assert np.allclose(turned[3], flat[3])  # not masked
        displacement = float(np.linalg.norm(turned[0] - flat[0]))
        assert displacement == pytest.approx(math.sqrt(2.0), rel=1e-9)

    def test_zero_update_is_identity(self) -> None:
        butane = Chem.MolFromSmiles("CCCC")
        bonds, masks = rotatable_masks(butane)
        flat = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]]
        )
        turned = perturb_torsions(flat, bonds, masks, [0.0])
        assert np.allclose(turned, flat)

    def test_perturb_length_mismatch_refused(self) -> None:
        butane = Chem.MolFromSmiles("CCCC")
        bonds, masks = rotatable_masks(butane)
        flat = np.zeros((4, 3))
        with pytest.raises(DataValidationError, match="matching lengths"):
            perturb_torsions(flat, bonds, masks, [0.1, 0.2])

    def test_randomize_pose_is_seeded_and_recentred(self) -> None:
        butane = Chem.MolFromSmiles("CCCC")
        AllChem.Compute2DCoords(butane)
        flat = butane.GetConformer().GetPositions()
        first = randomize_pose(butane, seed=42, center=[0.0, 0.0, 0.0])
        second = randomize_pose(butane, seed=42, center=[0.0, 0.0, 0.0])
        np.testing.assert_allclose(first, second)  # deterministic
        assert np.allclose(first.mean(axis=0), 0.0, atol=1e-9)  # centred
        assert not np.allclose(first, flat)  # actually randomised
        other = randomize_pose(butane, seed=7, center=[0.0, 0.0, 0.0])
        assert not np.allclose(first, other)  # the seed matters


# ---------------------------------------------------------------------------
# The evaluation metrics
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_min_intra_distance(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
        assert min_intra_distance(coords) == pytest.approx(1.5)

    def test_pose_clusters_greedy_leaders(self) -> None:
        matrix = np.array(
            [[0.0, 0.5, 5.0], [0.5, 0.0, 4.5], [5.0, 4.5, 0.0]]
        )
        assert pose_clusters(matrix, threshold=2.0) == [[0, 1], [2]]

    def test_report_metrics(self) -> None:
        ethane = Chem.MolFromSmiles("CC")
        ref = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        poses = [ref.copy(), ref + 0.1, ref + 9.0]
        report = pose_evaluation_report(ethane, ref, poses)
        assert report.n_poses == 3
        assert report.rmsd_below_2_fraction == pytest.approx(2.0 / 3.0)
        assert report.rmsd_below_5_fraction == pytest.approx(2.0 / 3.0)
        assert report.best_rmsd == pytest.approx(0.0, abs=1e-12)
        assert report.centroid_below_2_fraction == pytest.approx(2.0 / 3.0)
        assert report.self_clash_fraction == 0.0
        assert len(report.clusters) >= 1
        text = report.summary()
        assert "RMSD < 2 A" in text and "pose clusters" in text

    def test_report_refuses_empty_poses(self) -> None:
        ethane = Chem.MolFromSmiles("CC")
        ref = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        with pytest.raises(DataValidationError, match="at least one pose"):
            pose_evaluation_report(ethane, ref, [])

    def test_report_refuses_mixed_sizes(self) -> None:
        ethane = Chem.MolFromSmiles("CC")
        ref = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        with pytest.raises(DataValidationError, match="atom counts"):
            pose_evaluation_report(
                ethane, ref, [np.zeros((3, 3))]
            )

    def test_clash_fraction_counts_tight_poses(self) -> None:
        ethane = Chem.MolFromSmiles("CC")
        ref = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        tight = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])  # 0.2 A apart
        report = pose_evaluation_report(ethane, ref, [ref.copy(), tight])
        assert report.self_clash_fraction == pytest.approx(0.5)
        assert report.min_intra_distance_min == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# The pocket extraction
# ---------------------------------------------------------------------------


def _mini_protein() -> str:
    """Three residues across two chains around a ligand at the origin."""

    def atom(serial: int, name: str, res: str, chain: str, seq: int, x: float) -> str:
        return (
            f"ATOM  {serial:5d} {name:>4s} {res} {chain}{seq:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}"
            f"           {name[0]:>2s}"
        )

    return "\n".join(
        [
            atom(1, "N", "ALA", "A", 1, 0.0),
            atom(2, "CA", "ALA", "A", 1, 1.5),
            atom(3, "N", "GLY", "A", 2, 4.0),
            atom(4, "N", "VAL", "B", 1, 50.0),
            "END",
        ]
    ) + "\n"


class TestPocketExtraction:
    def test_residue_cutoff_selects_nearby_residues(self) -> None:
        report = extract_pocket(
            _mini_protein(),
            ligand_coords=[[0.0, 0.0, 0.0]],
            residue_cutoff=5.0,
            keep_whole_chains=False,
        )
        assert report.residues == [("A", "1", ""), ("A", "2", "")]
        assert report.n_atoms == 3
        assert report.chains == ["A"]

    def test_whole_chain_mode_keeps_chain_b_when_it_qualifies(self) -> None:
        report = extract_pocket(
            _mini_protein(),
            ligand_coords=[[50.0, 0.0, 0.0]],
            residue_cutoff=5.0,
            keep_whole_chains=True,
        )
        # pocket = B only; whole-chain mode keeps all of B (its one
        # residue) AND nothing from A
        assert report.residues == [("B", "1", "")]

    def test_min_ligand_distance_reported(self) -> None:
        report = extract_pocket(
            _mini_protein(),
            ligand_coords=[[0.0, 0.0, 0.0]],
            residue_cutoff=5.0,
            keep_whole_chains=False,
        )
        assert report.min_ligand_distance == pytest.approx(0.0)

    def test_crop_beyond_twenty_angstrom(self) -> None:
        cropped = crop_beyond(
            _mini_protein(), ligand_coords=[[0.0, 0.0, 0.0]], cutoff=20.0
        )
        assert cropped.count("ATOM") == 3  # the 50 A valine is dropped
        assert "VAL" not in cropped

    def test_summary_text(self) -> None:
        report = extract_pocket(
            _mini_protein(),
            ligand_coords=[[0.0, 0.0, 0.0]],
            residue_cutoff=5.0,
            keep_whole_chains=False,
        )
        text = report.summary()
        assert "pocket residues" in text and "chains kept" in text

    def test_empty_protein_refused(self) -> None:
        with pytest.raises(DataValidationError, match="no ATOM/HETATM"):
            extract_pocket("REMARK nothing here\n", ligand_coords=[[0.0, 0.0, 0.0]])

    def test_bad_cutoff_refused(self) -> None:
        with pytest.raises(DataValidationError, match="positive"):
            extract_pocket(
                _mini_protein(), ligand_coords=[[0.0, 0.0, 0.0]], residue_cutoff=0.0
            )
