"""Tests for :mod:`pharmaml.md.cv` — the PLUMED-style collective variables.

The honest-test rule carries: every CV is checked against a designed
geometry (a 3-4-5 triangle's angle is the arccos of 0.6, an eclipsed
dihedral is exactly 0°, a two-atom Rg is half the separation), the
``COORDINATION`` fraction against a hand-counted numerator, the parser
against PLUMED's real semantics (1-based indices, ``a-b`` range
expansion, duplicate-label refusal), and the ``COLVAR`` text byte for
byte.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pharmaml.exceptions import MDAnalysisError
from pharmaml.md import (
    CVTable,
    cv_angle,
    cv_coordination,
    cv_dihedral,
    cv_distance,
    cv_gyration,
    cv_rmsd,
    cv_table,
    parse_plumed_cvs,
)
from pharmaml.md.trajectory import Topology, Trajectory, synthetic_trajectory
from pharmaml.md.xvg import XvgSeries


def _line_traj(frames: list[np.ndarray]) -> Trajectory:
    """A trajectory from explicit frames with generic atom names."""
    n = frames[0].shape[0]
    return Trajectory(np.stack(frames), Topology(atom_names=[f"A{i}" for i in range(n)]))


@pytest.fixture()
def two_atom_traj() -> Trajectory:
    """Atom 0 at the origin, atom 1 sliding 2 Å -> 3 Å -> 4 Å away."""
    return _line_traj(
        [
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        ]
    )


class TestDistance:
    def test_known_separations(self, two_atom_traj: Trajectory) -> None:
        assert list(cv_distance(two_atom_traj, atoms=[0, 1])) == pytest.approx(
            [2.0, 3.0, 4.0]
        )

    def test_order_invariant(self, two_atom_traj: Trajectory) -> None:
        assert np.array_equal(
            cv_distance(two_atom_traj, atoms=[0, 1]),
            cv_distance(two_atom_traj, atoms=[1, 0]),
        )

    def test_three_d_geometry(self) -> None:
        traj = _line_traj(
            [np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 12.0]])]
        )
        assert cv_distance(traj, atoms=[0, 1])[0] == pytest.approx(13.0)

    @pytest.mark.parametrize(
        ("atoms", "message"),
        [
            ([0], "DISTANCE needs exactly 2 atoms"),
            ([0, 1, 2], "DISTANCE needs exactly 2 atoms"),
            ([0, 0], "DISTANCE needs 2 distinct atoms"),
            ([0, 9], "out of range for 2 atoms"),
            ([-1, 1], "out of range for 2 atoms"),
        ],
    )
    def test_validation(
        self, two_atom_traj: Trajectory, atoms: list[int], message: str
    ) -> None:
        with pytest.raises(MDAnalysisError, match=message):
            cv_distance(two_atom_traj, atoms=atoms)


class TestAngle:
    def test_right_angle(self) -> None:
        traj = _line_traj(
            [
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                )
            ]
        )
        assert cv_angle(traj, atoms=[0, 1, 2])[0] == pytest.approx(90.0)

    def test_three_four_five_triangle(self) -> None:
        # 3-4-5: the angle at the vertex on the length-3 leg has
        # cos = 9 / (3 * 5) = 0.6 — the apex is atom 1 (cv_angle's middle
        # index), atoms 0 and 2 the flanks.
        traj = _line_traj(
            [np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])]
        )
        assert cv_angle(traj, atoms=[0, 1, 2])[0] == pytest.approx(
            float(np.degrees(np.arccos(0.6)))
        )

    def test_collinear_is_defined(self) -> None:
        traj = _line_traj(
            [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])]
        )
        assert cv_angle(traj, atoms=[0, 1, 2])[0] == pytest.approx(180.0)

    def test_coincident_atoms_refused(self) -> None:
        traj = _line_traj(
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])]
        )
        with pytest.raises(MDAnalysisError, match="coincident"):
            cv_angle(traj, atoms=[0, 1, 2])

    @pytest.mark.parametrize(
        ("atoms", "message"),
        [
            ([0, 1], "ANGLE needs exactly 3 atoms"),
            ([0, 1, 2, 3], "ANGLE needs exactly 3 atoms"),
            ([0, 1, 1], "ANGLE needs 3 distinct atoms"),
        ],
    )
    def test_validation(
        self, two_atom_traj: Trajectory, atoms: list[int], message: str
    ) -> None:
        with pytest.raises(MDAnalysisError, match=message):
            cv_angle(two_atom_traj, atoms=atoms)


class TestDihedral:
    def test_eclipsed_and_trans(self) -> None:
        # The metrics.dihedral doctest geometry: same-side flank -> 0°,
        # opposite-side flank -> 180°.
        base = [np.zeros(3), np.array([1.0, 0.0, 0.0])]
        traj = _line_traj(
            [
                np.array([np.array([0.0, 1.0, 0.0]), *base, np.array([1.0, 1.0, 0.0])]),
                np.array([np.array([0.0, 1.0, 0.0]), *base, np.array([1.0, -1.0, 0.0])]),
            ]
        )
        values = cv_dihedral(traj, atoms=[0, 1, 2, 3])
        assert values[0] == pytest.approx(0.0)
        assert abs(values[1]) == pytest.approx(180.0)

    def test_degenerate_central_bond(self) -> None:
        # Four distinct indices, but the central-bond atoms (1 and 2)
        # sit at the same point — indices pass the distinctness check,
        # geometry does not.
        traj = _line_traj(
            [np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])]
        )
        with pytest.raises(MDAnalysisError, match="central bond"):
            cv_dihedral(traj, atoms=[0, 1, 2, 3])

    def test_validation(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="DIHEDRAL needs exactly 4 atoms"):
            cv_dihedral(two_atom_traj, atoms=[0, 1])


class TestCoordination:
    def test_hand_counted_fraction(self) -> None:
        # One A atom, two B atoms at 1 Å and 5 Å: with R_0 = 3.5 Å the
        # fraction is 1/2.
        traj = _line_traj(
            [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])]
        )
        assert cv_coordination(traj, group_a=[0], group_b=[1, 2], r_0=3.5)[
            0
        ] == pytest.approx(0.5)

    def test_full_and_empty(self) -> None:
        frames = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        ]
        traj = _line_traj(frames)
        values = cv_coordination(traj, group_a=[0], group_b=[1, 2], r_0=3.0)
        assert list(values) == pytest.approx([1.0, 0.0])

    def test_cutoff_edge_is_inclusive(self) -> None:
        traj = _line_traj([np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])])
        assert cv_coordination(traj, group_a=[0], group_b=[1], r_0=3.0)[0] == 1.0

    def test_empty_group_refused(self) -> None:
        traj = _line_traj([np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])])
        with pytest.raises(MDAnalysisError, match="GROUPA selects no atoms"):
            cv_coordination(traj, group_a=[], group_b=[1], r_0=3.0)

    @pytest.mark.parametrize("r_0", [0.0, -1.0, float("inf"), float("nan")])
    def test_bad_r0_refused(self, two_atom_traj: Trajectory, r_0: float) -> None:
        with pytest.raises(MDAnalysisError, match="r_0 must be finite and positive"):
            cv_coordination(two_atom_traj, group_a=[0], group_b=[1], r_0=r_0)

    def test_index_out_of_range(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="GROUPB index 7 out of range"):
            cv_coordination(two_atom_traj, group_a=[0], group_b=[7], r_0=3.0)


class TestRmsdAndGyration:
    def test_rmsd_first_frame_zero(self) -> None:
        traj = synthetic_trajectory(n_frames=5, seed=3, noise=0.4)
        values = cv_rmsd(traj, atoms=[0, 1, 2, 3, 4])
        assert values.shape == (5,)
        assert values[0] < 1e-12
        assert np.all(values[1:] > 0.0)

    def test_rmsd_matches_metrics(self) -> None:
        from pharmaml.md.metrics import rmsd_trajectory

        traj = synthetic_trajectory(n_frames=6, seed=1, noise=0.5)
        atoms = list(range(10))
        assert np.array_equal(
            cv_rmsd(traj, atoms=atoms), rmsd_trajectory(traj, atom_indices=atoms)
        )

    def test_gyration_two_atoms(self) -> None:
        traj = _line_traj([np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])])
        assert cv_gyration(traj, atoms=[0, 1])[0] == pytest.approx(1.0)

    def test_gyration_mass_weighting(self) -> None:
        traj = _line_traj([np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])])
        masses = np.array([1.0, 3.0])
        assert cv_gyration(traj, atoms=[0, 1], masses=masses)[0] == pytest.approx(
            np.sqrt(3.0) / 2.0
        )

    def test_empty_atoms_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="RMSD needs at least one atom"):
            cv_rmsd(two_atom_traj, atoms=[])
        with pytest.raises(MDAnalysisError, match="GYRATION needs at least one atom"):
            cv_gyration(two_atom_traj, atoms=[])


class TestCVTable:
    def test_to_colvar_exact_bytes(self, two_atom_traj: Trajectory) -> None:
        table = cv_table(
            two_atom_traj, {"d": cv_distance(two_atom_traj, atoms=[0, 1])}
        )
        assert table.to_colvar() == (
            "#! FIELDS time d\n"
            "0.000000 2.000000\n"
            "1.000000 3.000000\n"
            "2.000000 4.000000\n"
        )

    def test_uses_trajectory_time_axis(self) -> None:
        frames = [
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        ]
        traj = Trajectory(
            np.stack(frames),
            Topology(atom_names=["A", "B"]),
            time=np.array([10.0, 20.0]),
        )
        table = cv_table(traj, {"d": cv_distance(traj, atoms=[0, 1])})
        assert list(table.time) == [10.0, 20.0]

    def test_explicit_time_override(self, two_atom_traj: Trajectory) -> None:
        table = cv_table(
            two_atom_traj,
            {"d": cv_distance(two_atom_traj, atoms=[0, 1])},
            time=np.array([0.5, 1.0, 1.5]),
        )
        assert list(table.time) == [0.5, 1.0, 1.5]

    def test_axes_must_match(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="carries 4 values but the trajectory has 3"):
            cv_table(two_atom_traj, {"d": np.zeros(4)})

    def test_empty_cvs_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="at least one named CV"):
            cv_table(two_atom_traj, {})

    def test_nan_series_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="must be finite"):
            cv_table(two_atom_traj, {"d": np.array([np.nan, 1.0, 2.0])})

    def test_direct_construction_validation(self) -> None:
        with pytest.raises(MDAnalysisError, match="duplicate CV labels"):
            CVTable(
                time=np.array([0.0]),
                names=("d", "d"),
                series=np.zeros((1, 2)),
            )
        with pytest.raises(MDAnalysisError, match="2 CV names but the series carries"):
            CVTable(time=np.array([0.0]), names=("d", "x"), series=np.zeros((1, 3)))

    def test_to_dict_roundtrip(self, two_atom_traj: Trajectory) -> None:
        table = cv_table(
            two_atom_traj, {"d": cv_distance(two_atom_traj, atoms=[0, 1])}
        )
        payload = table.to_dict()
        assert payload["names"] == ["d"]
        assert payload["series"] == [[2.0], [3.0], [4.0]]
        assert payload["time"] == [0.0, 1.0, 2.0]

    def test_xvg_series_bridge(self, two_atom_traj: Trajectory) -> None:
        table = cv_table(
            two_atom_traj, {"d": cv_distance(two_atom_traj, atoms=[0, 1])}
        )
        series = table.xvg_series("d")
        assert isinstance(series, XvgSeries)
        assert series.label == "d"
        assert list(series.values) == pytest.approx([2.0, 3.0, 4.0])
        with pytest.raises(MDAnalysisError, match="no CV named 'nope'"):
            table.xvg_series("nope")

    def test_len_and_n_cvs(self, two_atom_traj: Trajectory) -> None:
        table = cv_table(
            two_atom_traj,
            {
                "d": cv_distance(two_atom_traj, atoms=[0, 1]),
                "rg": cv_gyration(two_atom_traj, atoms=[0, 1]),
            },
        )
        assert len(table) == 3
        assert table.n_cvs == 2


class TestPlumedParser:
    def test_all_six_actions(self) -> None:
        traj = synthetic_trajectory(n_frames=4, seed=0)
        source = (
            "# a real-ish plumed.dat\n"
            "d: DISTANCE ATOMS=1,2\n"
            "ang: ANGLE ATOMS=1,2,3\n"
            "dihe: DIHEDRAL ATOMS=1,2,3,4\n"
            "coord: COORDINATION GROUPA=1-3 GROUPB=4-6 R_0=5.0\n"
            "rms: RMSD ATOMS=1-10\n"
            "rg: GYRATION ATOMS=1-10"
        )
        table = parse_plumed_cvs(traj, source)
        assert table.names == ("d", "ang", "dihe", "coord", "rms", "rg")
        assert table.series.shape == (4, 6)
        assert np.all(np.isfinite(table.series))

    def test_one_based_conversion(self, two_atom_traj: Trajectory) -> None:
        # ATOMS=1,2 must mean the toolkit's 0-based (0, 1) — the
        # distances prove the offset is exactly one, not a silent
        # reinterpretation.
        table = parse_plumed_cvs(two_atom_traj, "d: DISTANCE ATOMS=1,2")
        assert list(table.xvg_series("d").values) == pytest.approx([2.0, 3.0, 4.0])

    def test_range_expansion(self) -> None:
        traj = synthetic_trajectory(n_frames=3, seed=0)
        table = parse_plumed_cvs(traj, "rg: GYRATION ATOMS=1-30")
        assert np.array_equal(
            table.series[:, 0],
            cv_gyration(traj, atoms=list(range(30))),
        )
        mixed = parse_plumed_cvs(traj, "rms: RMSD ATOMS=1,3-5")
        assert np.array_equal(
            mixed.series[:, 0], cv_rmsd(traj, atoms=[0, 2, 3, 4])
        )

    def test_mass_flag(self) -> None:
        frames = [np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
        topo = Topology(atom_names=["A", "B"], masses=np.array([1.0, 3.0]))
        traj = Trajectory(np.stack(frames), topo)
        table = parse_plumed_cvs(traj, "rg: GYRATION ATOMS=1-2 MASS")
        assert table.series[0, 0] == pytest.approx(np.sqrt(3.0) / 2.0)

    def test_mass_without_topology_masses(self) -> None:
        frames = [np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
        traj = Trajectory(np.stack(frames), Topology(atom_names=["A", "B"]))
        with pytest.raises(MDAnalysisError, match="needs topology masses"):
            parse_plumed_cvs(traj, "rg: GYRATION ATOMS=1-2 MASS")

    def test_unknown_action_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(
            MDAnalysisError, match="unsupported action 'METAD'"
        ):
            parse_plumed_cvs(two_atom_traj, "m: METAD ARG=d PACE=100")
        with pytest.raises(
            MDAnalysisError, match="supported: DISTANCE, ANGLE, DIHEDRAL, COORDINATION, RMSD, GYRATION"
        ):
            parse_plumed_cvs(two_atom_traj, "x: NOPE ATOMS=1,2")

    def test_unknown_keyword_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="takes ATOMS only"):
            parse_plumed_cvs(two_atom_traj, "d: DISTANCE ATOMS=1,2 NOPHYS=3")

    def test_bare_flag_keyword_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="expected KEY=VALUE"):
            parse_plumed_cvs(two_atom_traj, "d: DISTANCE ATOMS=1,2 COMPONENTS")

    def test_rmsd_reference_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="REFERENCE= files are not supported"):
            parse_plumed_cvs(two_atom_traj, "r: RMSD ATOMS=1,2 REFERENCE=ref.gro")

    def test_coordination_switching_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="rational switching"):
            parse_plumed_cvs(
                two_atom_traj,
                "c: COORDINATION GROUPA=1 GROUPB=2 R_0=3.5 NN=6 MM=12",
            )

    def test_coordination_missing_r0(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(
            MDAnalysisError, match="takes GROUPA, GROUPB and R_0 exactly"
        ):
            parse_plumed_cvs(
                two_atom_traj, "c: COORDINATION GROUPA=1 GROUPB=2"
            )

    def test_zero_based_index_is_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="there is no atom 0"):
            parse_plumed_cvs(two_atom_traj, "d: DISTANCE ATOMS=0,1")

    def test_out_of_range_plumed_atom(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="PLUMED atom out of range"):
            parse_plumed_cvs(two_atom_traj, "d: DISTANCE ATOMS=1,3")

    def test_duplicate_label_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="duplicate CV label 'd'"):
            parse_plumed_cvs(
                two_atom_traj, "d: DISTANCE ATOMS=1,2\nd: ANGLE ATOMS=1,2,1"
            )

    def test_unlabelled_line_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="needs a label"):
            parse_plumed_cvs(two_atom_traj, "DISTANCE ATOMS=1,2")

    def test_bad_label_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="invalid CV label"):
            parse_plumed_cvs(two_atom_traj, "1d: DISTANCE ATOMS=1,2")

    def test_no_action_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="has no action"):
            parse_plumed_cvs(two_atom_traj, "d:")

    def test_empty_input_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="no CV declarations"):
            parse_plumed_cvs(two_atom_traj, "# only comments\n\n")

    def test_duplicate_keyword_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="duplicate keyword 'ATOMS'"):
            parse_plumed_cvs(
                two_atom_traj, "d: DISTANCE ATOMS=1,2 ATOMS=1,2"
            )

    def test_bad_range_refused(self, two_atom_traj: Trajectory) -> None:
        with pytest.raises(MDAnalysisError, match="ranges must be ascending"):
            parse_plumed_cvs(two_atom_traj, "d: DISTANCE ATOMS=2-1")

    def test_deterministic(self) -> None:
        traj = synthetic_trajectory(n_frames=6, seed=7)
        source = "d: DISTANCE ATOMS=1,5\nrg: GYRATION ATOMS=1-20"
        first = parse_plumed_cvs(traj, source)
        second = parse_plumed_cvs(traj, source)
        assert np.array_equal(first.series, second.series)


class TestHouseStyle:
    def test_module_never_imports_matplotlib(self) -> None:
        # The source-level guard (the md/ honesty rule: matplotlib is
        # viz/ business only) — checked statically because a shared
        # pytest process may already carry matplotlib from other suites.
        import pharmaml.md.cv as cv_module

        source = Path(cv_module.__file__).read_text(encoding="utf-8")
        assert "matplotlib" not in source

    def test_viz_twin_consumes_cv_series(self, two_atom_traj: Trajectory) -> None:
        # The release tie-in: a CV column flows into the shipped
        # plot_md_series exactly like any .xvg metric series.
        from pharmaml.viz.md_plots import plot_md_series

        table = cv_table(
            two_atom_traj, {"d": cv_distance(two_atom_traj, atoms=[0, 1])}
        )
        out_path = Path("/tmp") / "cv_series_twin.png"
        result = plot_md_series(table.xvg_series("d"), out_path, close=True)
        assert result["n"] == 3
        assert out_path.exists()
