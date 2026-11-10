"""Tests for :mod:`pharmaml.md` — the round-249 molecular-dynamics package.

Every module of the package is exercised: the uploaded pipeline's
``.xvg``/``.ndx`` parsing and unit conversions, the trajectory core
and its selection mini-language, Kabsch alignment, the metric suite,
geometric H-bonds, PCA/tICA, the elastic network models, the ProLIF
interaction fingerprints and the ligand-study reader — with known-value
geometries wherever one exists (the honest-test rule: a rotation must
recover exactly, a two-atom Rg must be the separation, a Q-value must
be the designed fractions).

The synthetic helix (:func:`pharmaml.md.trajectory.synthetic_trajectory`)
carries the statistical assertions — seeded, so deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pharmaml.exceptions import MDAnalysisError, PharmaMLError
from pharmaml.md import (
    LigandResult,
    MDStudy,
    XvgSeries,
    anm_fluctuations,
    anm_hessian,
    anm_modes,
    build_study,
    center_of_mass,
    classify_metric,
    contact_map,
    cross_correlation,
    dihedral,
    find_hbonds,
    gnm_fluctuations,
    hbond_timeline,
    interaction_fingerprint,
    kabsch_rotation,
    kirchhoff_matrix,
    metric_correlations,
    metric_stats,
    moving_average,
    native_contact_fraction,
    pairwise_distances,
    parse_ndx_hbnum,
    parse_xvg,
    ps_to_ns,
    radial_distribution_function,
    radius_of_gyration,
    radius_of_gyration_trajectory,
    read_ligand_folder,
    rmsd,
    rmsd_trajectory,
    rmsf,
    salt_bridges,
    superpose,
    synthetic_hbond_trajectory,
    synthetic_trajectory,
    tica,
    to_kcal,
    trajectory_pca,
)
from pharmaml.md import (
    salt_bridges as salt_bridges_alias,
)
from pharmaml.md.trajectory import Topology, Trajectory, resolve_atom_indices

_CHAIN = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
_TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.7, 1.3, 0.0]])


# ---------------------------------------------------------------- xvg ---
class TestXvgParsing:
    def test_parse_xvg_skips_comments_and_directives(self) -> None:
        text = "# header\n@ s0 legend\n0.0 1.0\nskip me\n0.5 2.0 9.0\n\n"
        series = parse_xvg(text)
        assert len(series) == 2
        assert list(series.time) == [0.0, 0.5]
        assert list(series.values) == [1.0, 2.0]

    def test_parse_xvg_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / "ligand_rmsd.xvg"
        path.write_text("# gmx rms\n0 0.1\n1000 0.2\n")
        series = parse_xvg(path)
        assert series.source == "ligand_rmsd.xvg"
        assert len(series) == 2

    def test_parse_xvg_empty_raises(self) -> None:
        with pytest.raises(MDAnalysisError, match="no numeric data rows"):
            parse_xvg("# nothing\n@ at all\n")

    def test_parse_ndx_hbnum_finds_time_and_integer_columns(self) -> None:
        # Column 0 is a non-monotone decoy; column 1 (time) is monotone;
        # column 2 is the integer H-bond count; column 3 a float decoy.
        text = (
            "0.5 0 2 0.1\n"
            "0.9 10 2 0.2\n"
            "0.7 20 3 0.3\n"
            "1.1 30 3 0.4\n"
        )
        series = parse_ndx_hbnum(text)
        assert list(series.time) == [0.0, 10.0, 20.0, 30.0]
        assert list(series.values) == [2.0, 2.0, 3.0, 3.0]

    def test_parse_ndx_hbnum_empty_raises(self) -> None:
        with pytest.raises(MDAnalysisError, match="no numeric rows"):
            parse_ndx_hbnum("[ empty ]\nwords only\n")

    def test_metric_stats(self) -> None:
        stats = metric_stats(np.array([2.0, 4.0]))
        assert stats["mean"] == 3.0
        assert stats["std"] == 1.0
        assert stats["cv"] == pytest.approx(1.0 / 3.0)
        empty = metric_stats(np.array([]))
        assert empty == {"mean": None, "std": None, "cv": None}
        zero_mean = metric_stats(np.array([-1.0, 1.0]))
        assert zero_mean["cv"] is None

    def test_moving_average_centered(self) -> None:
        smoothed = moving_average(np.array([1.0, 2.0, 3.0]))
        assert smoothed[1] == 2.0
        assert smoothed[0] == pytest.approx(1.0)
        assert smoothed[2] == pytest.approx(5.0 / 3.0)
        with pytest.raises(MDAnalysisError, match="window"):
            moving_average(np.array([1.0]), window=2)

    def test_unit_conversions(self) -> None:
        kcal = to_kcal(np.array([4.184]))
        assert kcal[0] == pytest.approx(1.0)
        ns = ps_to_ns(np.array([1500.0]))
        assert ns[0] == pytest.approx(1.5)

    def test_classify_metric(self) -> None:
        exact = classify_metric("protein_backbone_Rg.xvg")
        assert exact is not None and exact.kind == "rg"
        twin = classify_metric("hbnum_intramolecular.ndx")
        assert twin is not None and twin.kind == "hbond"
        inferred = classify_metric("my_rmsf_per_atom.xvg")
        assert inferred is not None and inferred.kind == "rmsf"
        # "energy" must win over the "rg" substring inside it.
        energy = classify_metric("replica_energy.xvg")
        assert energy is not None and energy.kind == "energy"
        assert classify_metric("random.txt") is None

    def test_xvg_series_validation(self) -> None:
        with pytest.raises(MDAnalysisError, match="lengths differ"):
            XvgSeries(time=np.zeros(3), values=np.zeros(2))
        with pytest.raises(MDAnalysisError, match="empty"):
            XvgSeries(time=np.zeros(0), values=np.zeros(0))
        series = XvgSeries(time=np.arange(4.0), values=np.arange(4.0))
        assert series.to_dict()["stats"]["mean"] == 1.5


# --------------------------------------------------------- trajectory ---
class TestTopologyAndTrajectory:
    def test_topology_validation(self) -> None:
        with pytest.raises(MDAnalysisError, match="at least one atom"):
            Topology(atom_names=[])
        with pytest.raises(MDAnalysisError, match="residue_names has"):
            Topology(atom_names=["CA"], residue_names=["ALA", "GLY"])
        with pytest.raises(MDAnalysisError, match="masses"):
            Topology(atom_names=["CA", "N"], masses=np.array([1.0, -1.0]))

    def test_select_atoms_language(self) -> None:
        topo = Topology(
            atom_names=["N", "CA", "C", "H", "HA"],
            residue_names=["ALA", "ALA", "GLY", "GLY", "GLY"],
            residue_ids=[1, 1, 2, 2, 2],
        )
        assert list(topo.select_atoms("name CA")) == [1]
        assert list(topo.select_atoms("name N C")) == [0, 2]
        assert list(topo.select_atoms("name H*")) == [3, 4]
        assert list(topo.select_atoms("resname ALA")) == [0, 1]
        assert list(topo.select_atoms("resid 2")) == [2, 3, 4]
        assert list(topo.select_atoms("resid 1:2")) == [0, 1, 2, 3, 4]
        assert list(topo.select_atoms("resid 2-2")) == [2, 3, 4]
        assert list(topo.select_atoms("not name H*")) == [0, 1, 2]
        assert list(topo.select_atoms("name H* and resname ALA")) == []
        assert list(topo.select_atoms("name N or name H*")) == [0, 3, 4]
        assert list(topo.select_atoms("(name N or name C) and resid 1")) == [0]
        assert list(topo.select_atoms("(name N or name C) and (resid 1 or resid 2)")) == [0, 2]
        with pytest.raises(MDAnalysisError, match="unknown selection field"):
            topo.select_atoms("element C")
        with pytest.raises(MDAnalysisError, match="needs residue names"):
            Topology(atom_names=["CA"]).select_atoms("resname ALA")

    def test_trajectory_validation_and_indexing(self) -> None:
        topo = Topology(atom_names=["A", "B"])
        with pytest.raises(MDAnalysisError, match="n_frames, n_atoms, 3"):
            Trajectory(np.zeros((2, 2, 2)), topo)
        with pytest.raises(MDAnalysisError, match="topology has"):
            Trajectory(np.zeros((2, 3, 3)), topo)
        traj = Trajectory(
            np.arange(12.0).reshape(2, 2, 3), topo, time=np.array([0.0, 1.0])
        )
        assert (len(traj), traj.n_frames, traj.n_atoms) == (2, 2, 2)
        assert traj[1].coords.shape == (1, 2, 3)
        assert traj[1].time is not None and list(traj[1].time) == [1.0]
        assert traj[0:2].coords.shape == (2, 2, 3)
        assert traj[[1, 1]].coords.shape == (2, 2, 3)
        assert traj[-1].coords.shape == (1, 2, 3)
        with pytest.raises(MDAnalysisError, match="out of range"):
            traj[5]
        with pytest.raises(MDAnalysisError, match="no frames"):
            traj[2:2]

    def test_center_geometric_and_weighted(self) -> None:
        topo = Topology(atom_names=["A", "B"], masses=np.array([1.0, 3.0]))
        traj = Trajectory(
            np.array([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]), topo
        )
        geo = traj.center()
        assert np.allclose(geo.coords[0, 0], [-2.0, 0.0, 0.0])
        weighted = traj.center(weighted=True)
        assert np.allclose(weighted.coords[0, 0], [-3.0, 0.0, 0.0])
        bare = Trajectory(
            np.array([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]),
            Topology(atom_names=["A", "B"]),
        )
        with pytest.raises(MDAnalysisError, match="weighted centering"):
            bare.center(weighted=True)

    def test_synthetic_trajectory_deterministic(self) -> None:
        first = synthetic_trajectory(n_frames=6, seed=11)
        second = synthetic_trajectory(n_frames=6, seed=11)
        assert np.array_equal(first.coords, second.coords)
        loop = synthetic_trajectory(n_frames=30, seed=11, noise=0.5)
        loop_fl = rmsf(loop, atom_indices=list(range(12, 21)))
        helix_fl = rmsf(loop, atom_indices=[0, 1, 2, 3])
        assert float(np.mean(loop_fl)) > float(np.mean(helix_fl))
        with pytest.raises(MDAnalysisError, match="n_frames"):
            synthetic_trajectory(n_frames=0)

    def test_synthetic_hbond_timeline_matches_design(self) -> None:
        # The designed schedule: two phase-shifted distance oscillations
        # across the 3.5 A cutoff — frame 0 breaks both, frame 6 keeps
        # only bond 2, frame 9 only bond 1, frames 7-8 keep both.
        demo = synthetic_hbond_trajectory(n_frames=12, seed=0)
        counts = hbond_timeline(demo, donors=[(0, 1), (3, 4)], acceptors=[2, 5])
        assert [int(c) for c in counts] == [0, 0, 0, 0, 1, 1, 1, 2, 2, 1, 1, 1]

    def test_synthetic_hbond_deterministic_and_self_consistent(self) -> None:
        first = synthetic_hbond_trajectory(n_frames=30, seed=5)
        second = synthetic_hbond_trajectory(n_frames=30, seed=5)
        assert np.array_equal(first.coords, second.coords)
        # The counts re-derive from the coordinates with the public API.
        counts = hbond_timeline(first, donors=[(0, 1), (3, 4)], acceptors=[2, 5])
        again = hbond_timeline(first, donors=[(0, 1), (3, 4)], acceptors=[2, 5])
        assert np.array_equal(counts, again)
        # Both bonds work for a living and neither is ever stuck.
        assert 5 <= int((counts > 0).sum()) <= 25
        assert int(counts.max()) == 2
        # Topology: the six demo atoms, donors before acceptors.
        assert first.topology.atom_names == ["D1", "H1", "A1", "D2", "H2", "A2"]

    def test_synthetic_hbond_validation(self) -> None:
        with pytest.raises(MDAnalysisError, match="n_frames"):
            synthetic_hbond_trajectory(n_frames=0)
        with pytest.raises(MDAnalysisError, match="noise"):
            synthetic_hbond_trajectory(noise=-0.1)


# -------------------------------------------------------------- align ---
class TestAlignment:
    def test_kabsch_recovers_rotation(self) -> None:
        rng = np.random.default_rng(7)
        pts = rng.normal(size=(50, 3))
        c, s = np.cos(0.7), np.sin(0.7)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        recovered = kabsch_rotation(pts @ rotation, pts)
        assert np.allclose(recovered, rotation.T, atol=1e-10)

    def test_rmsd_invariance(self) -> None:
        rng = np.random.default_rng(8)
        pts = rng.normal(size=(40, 3))
        c, s = np.cos(1.1), np.sin(1.1)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        moved = pts @ rotation + np.array([10.0, -5.0, 3.0])
        assert rmsd(moved, pts) < 1e-10
        assert rmsd(pts, pts) < 1e-12

    def test_rmsd_noise_magnitude(self) -> None:
        rng = np.random.default_rng(9)
        pts = rng.normal(size=(90, 3))
        sigma = 0.2
        noisy = pts + rng.normal(0.0, sigma, size=pts.shape)
        # RMSD is the per-atom 3-D norm: sigma per component gives
        # sigma * sqrt(3) per atom (minus the 6 superposition dof).
        expected = sigma * np.sqrt(3)
        assert 0.9 * expected < rmsd(noisy, pts) < 1.1 * expected

    def test_superpose_translation_and_rotation(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        fitted = superpose(pts + 5.0, pts)
        assert rmsd(fitted, pts) < 1e-10

    def test_alignment_shape_errors(self) -> None:
        with pytest.raises(MDAnalysisError, match="must match"):
            rmsd(np.zeros((3, 3)), np.zeros((4, 3)))
        with pytest.raises(MDAnalysisError, match="n_atoms, 3"):
            kabsch_rotation(np.zeros((2, 2)), np.zeros((2, 2)))


# ------------------------------------------------------------ metrics ---
class TestMetrics:
    def test_rmsd_trajectory(self) -> None:
        traj = synthetic_trajectory(n_frames=15, seed=3, noise=0.45)
        series = rmsd_trajectory(traj)
        assert series.shape == (15,)
        assert series[0] < 1e-12
        assert np.all(series[1:] > 0)
        ca = traj.topology.select_atoms("name CA")
        backbone = rmsd_trajectory(traj, atom_indices=ca)
        assert backbone.shape == (15,)

    def test_rmsf_loop_more_flexible_than_helix(self) -> None:
        traj = synthetic_trajectory(n_frames=40, seed=5, noise=0.4)
        fl = rmsf(traj)
        assert fl.shape == (traj.n_atoms,)
        assert np.all(fl >= 0)
        assert float(np.mean(fl[12:21])) > float(np.mean(fl[0:12]))

    def test_center_of_mass_and_radius_of_gyration(self) -> None:
        two = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert np.allclose(center_of_mass(two), [0.0, 0.0, 0.0])
        assert radius_of_gyration(two) == pytest.approx(1.0)
        weighted = radius_of_gyration(two, np.array([1.0, 3.0]))
        assert weighted == pytest.approx(np.sqrt(0.75))
        single = np.array([[5.0, 5.0, 5.0]])
        assert radius_of_gyration(single) == pytest.approx(0.0)

    def test_radius_of_gyration_trajectory(self) -> None:
        traj = synthetic_trajectory(n_frames=8, seed=6, noise=0.2)
        rg = radius_of_gyration_trajectory(traj)
        assert rg.shape == (8,)
        assert np.all(rg > 0)

    def test_pairwise_distances(self) -> None:
        d = pairwise_distances(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
        assert d.shape == (2, 2)
        assert d[0, 1] == pytest.approx(3.0)
        assert d[1, 1] == pytest.approx(0.0)

    def test_contact_map_designed_frequencies(self) -> None:
        base = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        mid = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        apart = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [18.0, 0.0, 0.0]])
        topo = Topology(atom_names=["A", "B", "C"])
        cm = contact_map(
            Trajectory(np.stack([base, mid, apart]), topo), cutoff=4.5
        )
        assert cm[0, 1] == pytest.approx(2.0 / 3.0)
        assert cm[0, 2] == pytest.approx(0.0)
        assert cm[1, 2] == pytest.approx(1.0 / 3.0)
        assert np.allclose(np.diag(cm), 0.0)

    def test_native_contact_fraction(self) -> None:
        base = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0]])
        half = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-4.4, 0.0, 0.0]])
        apart = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
        topo = Topology(atom_names=["A", "B", "C"])
        traj = Trajectory(np.stack([base, half, apart]), topo)
        q = native_contact_fraction(traj, cutoff=4.5)
        assert q[0] == pytest.approx(1.0)
        assert q[1] == pytest.approx(2.0 / 3.0)
        assert q[2] == pytest.approx(0.0)

    def test_rdf_peak_location(self) -> None:
        centers, g = radial_distribution_function(
            np.zeros((1, 3)), np.array([[3.0, 0.0, 0.0]]), r_max=5.0, bins=5
        )
        assert int(np.argmax(g)) == 3
        assert np.all(g[:3] == 0.0)
        assert centers.shape == g.shape == (5,)

    def test_dihedral_knowns(self) -> None:
        p1, p2 = np.zeros(3), np.array([1.0, 0.0, 0.0])
        cis = dihedral(np.array([0.0, 1.0, 0.0]), p1, p2, np.array([1.0, 1.0, 0.0]))
        trans = dihedral(
            np.array([0.0, 1.0, 0.0]), p1, p2, np.array([1.0, -1.0, 0.0])
        )
        assert abs(cis) == pytest.approx(0.0, abs=1e-12)
        assert abs(trans) == pytest.approx(180.0, abs=1e-12)
        with pytest.raises(MDAnalysisError, match="3-vectors"):
            dihedral(np.zeros(2), p1, p2, np.array([1.0, 1.0, 0.0]))

    def test_atom_indices_validation(self) -> None:
        assert list(resolve_atom_indices(None, 2)) == [0, 1]
        with pytest.raises(MDAnalysisError, match="out of range"):
            resolve_atom_indices([2], 2)
        with pytest.raises(MDAnalysisError, match="no atoms"):
            resolve_atom_indices([], 5)


# ------------------------------------------------------------- hbond ---
class TestHydrogenBonds:
    def test_find_hbonds_criteria(self) -> None:
        # Linear D-H...A at 2 Å: passes both criteria.
        frame = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
        assert find_hbonds(frame, donors=[(0, 1)], acceptors=[2]) == [(0, 2)]
        # Right-angle H: fails the angle criterion.
        bent = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        )
        assert find_hbonds(bent, donors=[(0, 1)], acceptors=[2]) == []
        # Distance beyond the cutoff.
        far = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        assert find_hbonds(far, donors=[(0, 1)], acceptors=[2]) == []
        # The donor's own hydrogen is never an acceptor.
        assert find_hbonds(frame, donors=[(0, 1)], acceptors=[0, 1, 2]) == [(0, 2)]

    def test_hbond_timeline(self) -> None:
        base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        far = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        topo = Topology(atom_names=["D", "H", "A"])
        traj = Trajectory(np.stack([base, far, base]), topo)
        assert list(hbond_timeline(traj, donors=[(0, 1)], acceptors=[2])) == [
            1.0,
            0.0,
            1.0,
        ]

    def test_salt_bridges(self) -> None:
        charged = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        assert salt_bridges_alias(charged, positive=[0], negative=[1, 2]) == [(0, 1)]
        assert salt_bridges(charged, positive=[1], negative=[0]) == [(1, 0)]
        assert salt_bridges(charged, positive=[2], negative=[0]) == []

    def test_hbond_validation(self) -> None:
        frame = np.zeros((3, 3))
        with pytest.raises(MDAnalysisError, match="donor"):
            find_hbonds(frame, donors=[(0,)], acceptors=[2])
        with pytest.raises(MDAnalysisError, match="out of range"):
            find_hbonds(frame, donors=[(0, 1)], acceptors=[9])
        with pytest.raises(MDAnalysisError, match="angle_cutoff"):
            find_hbonds(frame, donors=[(0, 1)], acceptors=[2], angle_cutoff=200.0)


# ------------------------------------------------------------ dimred ---
class TestDimensionalityReduction:
    def test_pca_line_dominant_axis(self) -> None:
        data = np.zeros((20, 6))
        data[:, 0] = np.arange(20.0)
        result = trajectory_pca(
            Trajectory(data.reshape(20, 2, 3), Topology(atom_names=["A", "B"])),
            n_components=1,
            align=False,
        )
        assert result.explained_variance_ratio[0] == pytest.approx(1.0)
        assert abs(result.components[0, 0]) == pytest.approx(1.0)
        assert np.all(np.abs(result.components[0, 1:]) < 1e-8)

    def test_pca_on_synthetic_trajectory(self) -> None:
        traj = synthetic_trajectory(n_frames=40, seed=3, noise=0.4)
        result = trajectory_pca(traj, n_components=3)
        assert result.components.shape == (3, 3 * traj.n_atoms)
        assert np.all(np.diff(result.explained_variance_ratio) <= 1e-12)
        projection = result.transform_trajectory(traj)
        assert projection.shape == (40, 3)
        assert np.all(np.var(projection, axis=0) > 0)

    def test_tica_ranks_slow_channel(self) -> None:
        steps = np.arange(400.0)
        slow = np.sin(steps * 2 * np.pi / 200.0)
        fast = np.sin(steps * 2 * np.pi / 3.7)
        result = tica(np.column_stack([slow, fast]), lag=8, n_components=1)
        assert abs(result.components[0, 0]) > abs(result.components[0, 1])
        assert result.explained_variance[0] > 0.5

    def test_dimred_validation(self) -> None:
        with pytest.raises(MDAnalysisError, match="lag"):
            tica(np.zeros((5, 2)), lag=0)
        with pytest.raises(MDAnalysisError, match="time-lagged"):
            tica(np.zeros((5, 2)), lag=5)
        traj = synthetic_trajectory(n_frames=5, seed=1)
        with pytest.raises(MDAnalysisError, match="two frames"):
            trajectory_pca(traj[0:1])


# --------------------------------------------------------------- enm ---
class TestElasticNetworks:
    def test_kirchhoff_matrix(self) -> None:
        g = kirchhoff_matrix(_CHAIN, cutoff=1.5, gamma=2.0)
        assert np.allclose(g.sum(axis=1), 0.0)
        assert g[0, 0] == 2.0 and g[1, 1] == 4.0
        assert g[0, 1] == -2.0 and g[0, 2] == 0.0
        assert np.allclose(g, g.T)

    def test_gnm_fluctuations(self) -> None:
        fl = gnm_fluctuations(_CHAIN, cutoff=1.5)
        assert np.all(fl > 0)
        assert fl[1] < fl[0]  # the middle atom is the most constrained

    def test_cross_correlation(self) -> None:
        corr = cross_correlation(_CHAIN, cutoff=1.5)
        assert np.allclose(np.diag(corr), 1.0)
        assert np.all(np.abs(corr) <= 1.0 + 1e-9)
        assert np.allclose(corr, corr.T)

    def test_anm_hessian_six_zero_modes(self) -> None:
        h = anm_hessian(_TRIANGLE, cutoff=2.0)
        assert h.shape == (9, 9)
        assert np.allclose(h.sum(axis=1), 0.0, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(h)
        assert np.sum(eigenvalues < 1e-8) == 6

    def test_anm_modes(self) -> None:
        w, modes = anm_modes(_TRIANGLE, n_modes=3, cutoff=2.0)
        assert w.shape == (3,) and modes.shape == (3, 9)
        assert np.all(w > 0)
        # A collinear chain supplies fewer non-trivial modes than six
        # zero modes leave — the honest refusal.
        with pytest.raises(MDAnalysisError, match="too disconnected"):
            anm_modes(_CHAIN, n_modes=3, cutoff=1.5)

    def test_anm_fluctuations(self) -> None:
        fl = anm_fluctuations(_TRIANGLE, cutoff=2.0)
        assert fl.shape == (3,)
        assert np.all(fl > 0)


# -------------------------------------------------------------- plif ---
class TestInteractionFingerprints:
    def test_all_interaction_families(self) -> None:
        lig = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0]]
        )
        lig_types = ["HYD", "HBD", "POS", "PI"]
        prot = np.array(
            [[3.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [16.0, 0.0, 0.0]]
        )
        prot_types = ["HYD", "HBA", "NEG", "PI"]
        fp = interaction_fingerprint(lig, lig_types, prot, prot_types)
        assert fp.counts["hydrophobic"] == 1
        assert fp.counts["hbond_donor"] == 1
        assert fp.counts["hbond_acceptor"] == 0
        assert fp.counts["ionic"] == 1
        assert fp.counts["pi_stack"] == 1  # distance-only, no normals
        assert list(fp.bits) == [True, True, False, True, True]
        assert len(fp) == 4
        payload = fp.to_dict()
        assert payload["pairs"]["hydrophobic"] == [[0, 0]]

    def test_pi_stack_orientation_guard(self) -> None:
        lig = np.array([[0.0, 0.0, 0.0]])
        prot = np.array([[1.0, 0.0, 0.0]])
        parallel = interaction_fingerprint(
            lig,
            ["PI"],
            prot,
            ["PI"],
            ligand_normals=np.array([[0.0, 0.0, 1.0]]),
            protein_normals=np.array([[0.0, 0.0, 1.0]]),
        )
        perpendicular = interaction_fingerprint(
            lig,
            ["PI"],
            prot,
            ["PI"],
            ligand_normals=np.array([[0.0, 0.0, 1.0]]),
            protein_normals=np.array([[1.0, 0.0, 0.0]]),
        )
        assert parallel.counts["pi_stack"] == 1
        assert perpendicular.counts["pi_stack"] == 0

    def test_plif_validation(self) -> None:
        lig = np.zeros((1, 3))
        with pytest.raises(MDAnalysisError, match="unknown ligand"):
            interaction_fingerprint(lig, ["XX"], lig, ["HYD"])
        with pytest.raises(MDAnalysisError, match="entries but"):
            interaction_fingerprint(lig, ["HYD", "HBA"], lig, ["HYD"])
        with pytest.raises(MDAnalysisError, match="both"):
            interaction_fingerprint(
                lig, ["HYD"], lig, ["HYD"], ligand_normals=lig.copy()
            )


# ------------------------------------------------------------- study ---
class TestLigandStudy:
    def _write_campaign(self, root: Path) -> None:
        lig_a = root / "ligA"
        lig_a.mkdir()
        (lig_a / "ligand_rmsd.xvg").write_text("# t rmsd\n0 0.1\n1000 0.2\n2000 0.3\n")
        (lig_a / "binding_free_energy.xvg").write_text("# dg\n0 -41.84\n1000 -20.92\n")
        (lig_a / "hbnum_intermolecular.ndx").write_text(
            "0 1 0.5\n1000 2 0.6\n2000 3 0.7\n"
        )
        lig_b = root / "ligB"
        lig_b.mkdir()
        (lig_b / "ligand_rmsd.xvg").write_text("# t rmsd\n0 0.3\n1000 0.9\n2000 1.5\n")
        (root / "Global_plots").mkdir()
        (root / "Global_plots" / "stale.txt").write_text("skip\n")

    def test_read_ligand_folder(self, tmp_path: Path) -> None:
        folder = tmp_path / "ligA"
        folder.mkdir()
        (folder / "ligand_rmsd.xvg").write_text("# t rmsd\n0 0.1\n1000 0.2\n2000 0.3\n")
        (folder / "binding_free_energy.xvg").write_text("# dg\n0 -41.84\n1000 -20.92\n")
        (folder / "hbnum_intermolecular.ndx").write_text(
            "0 1 0.5\n1000 2 0.6\n2000 3 0.7\n"
        )
        (folder / "ignored.txt").write_text("not a metric\n")
        result = read_ligand_folder(folder)
        assert isinstance(result, LigandResult)
        assert sorted(result.series) == [
            "binding_free_energy",
            "hbnum_intermolecular",
            "ligand_rmsd",
        ]
        rmsd = result.series["ligand_rmsd"]
        assert list(rmsd.time) == [0.0, 1.0, 2.0]  # ps -> ns conversion
        energy = result.series["binding_free_energy"]
        assert energy.values[0] == pytest.approx(-10.0)  # kJ -> kcal
        hbonds = result.series["hbnum_intermolecular"]
        assert hbonds.values[1] == pytest.approx(2.0)  # smoothed centre

    def test_read_ligand_folder_skips_broken_export(self, tmp_path: Path) -> None:
        folder = tmp_path / "ligA"
        folder.mkdir()
        (folder / "ligand_rmsd.xvg").write_text("# t rmsd\n0 0.1\n1000 0.2\n")
        (folder / "ligand_rmsf.xvg").write_text("# no data rows at all\n")
        result = read_ligand_folder(folder)
        assert list(result.series) == ["ligand_rmsd"]

    def test_read_ligand_folder_requires_one_metric(self, tmp_path: Path) -> None:
        folder = tmp_path / "empty"
        folder.mkdir()
        with pytest.raises(MDAnalysisError, match="no parsable"):
            read_ligand_folder(folder)
        with pytest.raises(MDAnalysisError, match="not a directory"):
            read_ligand_folder(tmp_path / "missing")

    def test_build_study(self, tmp_path: Path) -> None:
        self._write_campaign(tmp_path)
        study = build_study(tmp_path)
        assert isinstance(study, MDStudy)
        assert study.ligand_names == ["ligA", "ligB"]
        global_rmsd = study.metric_series("ligand_rmsd")
        assert sorted(global_rmsd) == ["ligA", "ligB"]
        assert study.ligand("ligB").series["ligand_rmsd"].values[0] == pytest.approx(
            0.3
        )
        with pytest.raises(MDAnalysisError, match="no ligand named"):
            study.ligand("ligC")

    def test_summary_csv_contract(self, tmp_path: Path) -> None:
        self._write_campaign(tmp_path)
        study = build_study(tmp_path)
        lines = study.summary_csv().splitlines()
        assert lines[0] == "Folder,File,Mean,Stdev,Coefficient_of_Variation"
        row = lines[1].split(",")
        assert row[0] == "ligA" and row[1] == "binding_free_energy"
        assert float(row[2]) == pytest.approx(np.mean([-10.0, -5.0]))

    def test_flat_folder_single_ligand_study(self, tmp_path: Path) -> None:
        (tmp_path / "ligand_rmsd.xvg").write_text("# t\n0 0.1\n1000 0.2\n")
        study = build_study(tmp_path)
        assert len(study) == 1
        assert study.ligands[0].name == tmp_path.name

    def test_metric_correlations(self) -> None:
        rising = XvgSeries(
            time=np.arange(4.0), values=np.array([1.0, 2.0, 3.0, 4.0])
        )
        doubling = XvgSeries(
            time=np.arange(4.0), values=np.array([2.0, 4.0, 6.0, 8.0])
        )
        labels, corr = metric_correlations({"a": rising, "b": doubling})
        assert labels == ["a", "b"]
        assert np.allclose(corr, 1.0)
        assert metric_correlations({"a": rising}) is None
        shorts = {
            "a": rising,
            "b": XvgSeries(time=np.zeros(1), values=np.zeros(1)),
        }
        assert metric_correlations(shorts) is None

    def test_study_is_part_of_the_pharmaml_error_family(self) -> None:
        assert issubclass(MDAnalysisError, PharmaMLError)
