"""Unit tests for docking preparation, parsing, and scoring."""

from __future__ import annotations

import logging
from contextlib import contextmanager

import pandas as pd
import pytest

from pharmaml.docking.parsing import (
    check_docking_completion,
    parse_dlg_cluster_table,
    parse_dlg_summary,
    parse_vina_output,
    scan_docking_directory,
    scan_vina_directory,
)
from pharmaml.docking.preparation import (
    GridBox,
    compute_grid_box,
    extract_pdbqt_atom_types,
    write_gpf,
)
from pharmaml.docking.scoring import (
    consensus_rescore,
    ligand_efficiency,
    ligand_efficiency_series,
)
from pharmaml.exceptions import DataValidationError, DockingParseError


@contextmanager
def _pharmaml_logs_visible():
    """Re-enable propagation on the ``pharmaml`` logger for caplog.

    The CLI tests call :func:`configure_logging`, which sets
    ``propagate=False`` on the ``pharmaml`` root logger; pytest's
    caplog handler lives on the true root logger, so records only
    reach it while propagation is on (restored on exit).  Same dance
    as ``tests/test_ml_pipeline.py``.
    """
    pharma_root = logging.getLogger("pharmaml")
    prev_propagate = pharma_root.propagate
    pharma_root.propagate = True
    try:
        yield
    finally:
        pharma_root.propagate = prev_propagate


class TestGridBox:
    def test_size_property(self) -> None:
        box = GridBox(1.0, 2.0, 3.0, npts=(60, 60, 60), spacing=0.375)
        assert box.size == pytest.approx((22.5, 22.5, 22.5))

    def test_vina_kwargs(self) -> None:
        box = GridBox(0, 0, 0, npts=(40, 40, 40), spacing=0.5)
        kwargs = box.as_vina_kwargs()
        assert kwargs["size_x"] == pytest.approx(20.0)
        assert kwargs["center_y"] == 0.0


class TestGridComputation:
    def test_center_is_bounding_box_midpoint(self, tmp_path, sample_ligand_pdbqt) -> None:
        """Grid centre follows the AutoDock convention: midpoint of the
        reference ligand's bounding box (symmetric padding on all sides)."""
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        box = compute_grid_box(ligand, padding=4.0)
        # x range [0.987, 3.456], y [-1.234, 1.012], z [-1.234, 1.789].
        assert box.center_x == pytest.approx((0.987 + 3.456) / 2, abs=1e-3)
        assert box.center_y == pytest.approx((-1.234 + 1.012) / 2, abs=1e-3)
        assert box.center_z == pytest.approx((-1.234 + 1.789) / 2, abs=1e-3)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(DockingParseError, match="not found"):
            compute_grid_box(tmp_path / "missing.pdbqt")

    def test_negative_padding_rejected(self, tmp_path, sample_ligand_pdbqt) -> None:
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        with pytest.raises(DataValidationError):
            compute_grid_box(ligand, padding=-1.0)


class TestAtomTypesAndGPF:
    def test_extract_atom_types(self, tmp_path, sample_ligand_pdbqt) -> None:
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        assert extract_pdbqt_atom_types(ligand) == ["C", "HD", "N", "OA"]

    def test_write_gpf_content(self, tmp_path, sample_ligand_pdbqt) -> None:
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        box = GridBox(2.0, -0.2, 0.2)
        gpf = write_gpf(ligand, receptor_file="r.pdbqt", box=box)
        assert gpf.is_file()
        content = gpf.read_text()
        assert content.startswith("npts 60 60 60")
        assert "ligand_types C HD N OA" in content
        assert "receptor r.pdbqt" in content
        assert "gridcenter 2.0 -0.2 0.2" in content
        # one affinity-map line per ligand atom type + elec + dsolv
        assert content.count("map r.") == 4 + 2

    def test_write_gpf_requires_existing_ligand(self, tmp_path) -> None:
        with pytest.raises(Exception, match="not found"):
            write_gpf(tmp_path / "ghost.pdbqt")


class TestDLGParsing:
    def test_cluster_table(self, sample_dlg) -> None:
        clusters = parse_dlg_cluster_table(sample_dlg)
        assert len(clusters) == 4
        assert clusters[0].rank == 1
        assert clusters[0].lowest_energy == pytest.approx(-11.5)
        assert clusters[0].mean_energy == pytest.approx(-11.2)
        assert clusters[3].num_in_cluster == 12

    def test_summary_most_populated(self, sample_dlg) -> None:
        summary = parse_dlg_summary(sample_dlg)
        assert summary["best_rank"] == 3       # 52 poses, the largest cluster
        assert summary["best_num_in_cluster"] == 52
        assert summary["top_rank"] == 1
        assert summary["top_lowest_energy"] == pytest.approx(-11.5)
        assert summary["n_clusters"] == 4

    def test_no_table_raises(self) -> None:
        with pytest.raises(DockingParseError, match="no cluster table"):
            parse_dlg_cluster_table("DPF> nothing to see here")

    def test_parse_from_file(self, tmp_path, sample_dlg) -> None:
        log = tmp_path / "r.dlg"
        log.write_text(sample_dlg, encoding="utf-8")
        assert parse_dlg_summary(log)["n_clusters"] == 4


class TestVinaParsing:
    def test_mode_table(self, sample_vina_output) -> None:
        poses = parse_vina_output(sample_vina_output)
        assert len(poses) == 4
        assert poses[0].mode == 1
        assert poses[0].affinity == pytest.approx(-10.3)
        assert poses[1].rmsd_lb == pytest.approx(1.842)
        assert poses[1].rmsd_ub == pytest.approx(2.301)
        assert [p.mode for p in poses] == [1, 2, 3, 4]

    def test_remark_format(self) -> None:
        text = "REMARK VINA RESULT:    -9.5      1.234      2.345\nREMARK VINA RESULT:    -8.9      2.000      3.000"
        poses = parse_vina_output(text)
        assert poses[0].affinity == pytest.approx(-9.5)
        assert poses[0].rmsd_lb == pytest.approx(1.234)

    def test_empty_raises(self) -> None:
        with pytest.raises(DockingParseError, match="no Vina pose"):
            parse_vina_output("nothing useful\n")

    def test_scientific_notation_rows_are_parsed(self) -> None:
        # Regression: affinity/RMSD values in exponent form (e.g. from
        # post-processed tables) were silently dropped by the row regex.
        text = "   1      -1.0e+01          0          0\n   2      -9.6e+00      1.842      2.301\n   3       -9.4      2.105      2.777"
        poses = parse_vina_output(text)
        assert len(poses) == 3
        assert poses[0].affinity == pytest.approx(-10.0)
        assert poses[1].affinity == pytest.approx(-9.6)
        assert poses[1].rmsd_lb == pytest.approx(1.842)

    def test_malformed_remark_line_warns_and_keeps_good_poses(self, caplog) -> None:
        # Regression: a corrupted REMARK used to crash with a raw
        # ValueError ('could not convert string to float: abc').
        text = "REMARK VINA RESULT:    abc      1.234      2.345\nREMARK VINA RESULT:    -9.5      1.234      2.345\nREMARK VINA RESULT:    -8.9      2.000      3.000"
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.docking.parsing"):
                poses = parse_vina_output(text)
        assert [p.affinity for p in poses] == pytest.approx([-9.5, -8.9])
        assert "malformed Vina RESULT" in caplog.text

    def test_short_remark_line_warns(self, caplog) -> None:
        # A truncated REMARK (affinity only) must not vanish silently.
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.docking.parsing"):
                with pytest.raises(DockingParseError, match="no Vina pose"):
                    parse_vina_output("REMARK VINA RESULT:    -9.5")
        assert "malformed Vina RESULT" in caplog.text


class TestDirectoryScans:
    def _make_tree(self, tmp_path, sample_dlg) -> None:
        good = tmp_path / "ligand_01"
        good.mkdir()
        (good / "r.dlg").write_text(sample_dlg, encoding="utf-8")
        incomplete = tmp_path / "ligand_02"
        incomplete.mkdir()
        (incomplete / "r.dlg").write_text("DPF> interrupted run\n", encoding="utf-8")
        (tmp_path / "ligand_03").mkdir()  # no log at all
        (tmp_path / "notes.txt").write_text("stray file", encoding="utf-8")

    def test_completion_audit(self, tmp_path, sample_dlg) -> None:
        self._make_tree(tmp_path, sample_dlg)
        audit = check_docking_completion(tmp_path)
        statuses = dict(zip(audit["folder"], audit["status"]))
        assert statuses["ligand_01"] == "OK"
        assert statuses["ligand_02"] == "INCOMPLETE"
        assert statuses["ligand_03"] == "MISSING_LOG"

    def test_scan_summary_table(self, tmp_path, sample_dlg) -> None:
        self._make_tree(tmp_path, sample_dlg)
        summary = scan_docking_directory(tmp_path)
        assert len(summary) == 3
        ok_row = summary[summary["folder"] == "ligand_01"].iloc[0]
        assert ok_row["status"] == "OK"
        assert ok_row["n_clusters"] == 4
        missing = summary[summary["folder"] == "ligand_03"].iloc[0]
        assert missing["status"] == "MISSING_LOG"

    def test_scan_requires_directory(self, tmp_path) -> None:
        with pytest.raises(DockingParseError):
            scan_docking_directory(tmp_path / "nowhere")

    def test_completion_audit_requires_directory(self, tmp_path) -> None:
        # Regression: a typo'd path used to return an empty audit table,
        # which the CLI reads as "all runs OK" (exit code 0).
        with pytest.raises(DockingParseError, match="not found"):
            check_docking_completion(tmp_path / "nowhere")

    def test_scan_vina_directory_parses_and_sorts(self, tmp_path) -> None:
        (tmp_path / "ligand_a.log").write_text(
            "   1      -1.05e+01          0          0\n", encoding="utf-8"
        )
        (tmp_path / "ligand_b.log").write_text(
            "   1       -9.7          0          0\n", encoding="utf-8"
        )
        (tmp_path / "ligand_c.log").write_text("garbage\n", encoding="utf-8")
        report = scan_vina_directory(tmp_path)
        assert list(report["ligand"]) == ["ligand_a", "ligand_b", "ligand_c"]
        assert report["best_affinity"].iloc[0] == pytest.approx(-10.5)
        assert report["best_affinity"].iloc[1] == pytest.approx(-9.7)
        assert pd.isna(report["best_affinity"].iloc[2])  # unparseable sorts last
        assert list(report["n_poses"]) == [1, 1, 0]
        assert report["log"].iloc[0] == "ligand_a.log"


class TestScoring:
    def test_ligand_efficiency(self) -> None:
        assert ligand_efficiency(-11.5, 23) == pytest.approx(0.5)
        assert ligand_efficiency(-8.0, 40) == pytest.approx(0.2)

    def test_ligand_efficiency_validation(self) -> None:
        with pytest.raises(DataValidationError):
            ligand_efficiency(-10.0, 0)
        with pytest.raises(DataValidationError):
            ligand_efficiency(-10.0, 12.5)  # non-integer atom count

    def test_consensus_prefers_low_energy_and_big_clusters(self) -> None:
        frame = pd.DataFrame(
            {
                "ligand": ["a", "b", "c"],
                "best_lowest_energy": [-11.0, -9.0, -10.0],
                "best_num_in_cluster": [50, 50, 10],
            }
        )
        scores = consensus_rescore(
            frame,
            columns=["best_lowest_energy", "best_num_in_cluster"],
            weights=[0.7, 0.3],
        )
        assert scores.iloc[0] == scores.max()
        assert ((scores >= 0) & (scores <= 1)).all()

    def test_consensus_missing_column_raises(self) -> None:
        with pytest.raises(DataValidationError, match="missing columns"):
            consensus_rescore(pd.DataFrame({"x": [1]}), columns=["nope"])

    def test_consensus_negative_weights_raise(self) -> None:
        # Regression: negative/zero-sum weights used to slip through and
        # silently produce a NaN Series (pandas divides by zero without
        # raising), which clip() happily propagated.
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
        with pytest.raises(DataValidationError, match="non-negative"):
            consensus_rescore(frame, columns=["a", "b"], weights=[1.0, -1.0])

    def test_consensus_all_zero_weights_raise(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(DataValidationError, match="all be zero"):
            consensus_rescore(frame, columns=["a"], weights=[0.0])

    def test_consensus_zero_weight_column_ok(self) -> None:
        # A zero weight is fine as long as others keep the sum positive.
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
        scores = consensus_rescore(frame, columns=["a", "b"], weights=[1.0, 0.0])
        assert scores.notna().all()
        assert ((scores >= 0) & (scores <= 1)).all()

    def test_consensus_zero_variance_handled(self) -> None:
        frame = pd.DataFrame({"e": [-9.0, -9.0, -9.0], "n": [1, 5, 9]})
        scores = consensus_rescore(frame, columns=["e", "n"])
        assert scores.notna().all()

    def test_ligand_efficiency_series_atom_count_validation(self) -> None:
        # Regression: 12.5 atoms used to be silently truncated to 12
        # (LE computed on the wrong count), diverging from the scalar
        # ligand_efficiency contract; whole-number floats stay accepted.
        bad = pd.DataFrame(
            {"best_lowest_energy": [-10.0, -10.0], "num_heavy_atoms": [12.5, 30.0]}
        )
        with pytest.raises(DataValidationError, match="12.5"):
            ligand_efficiency_series(bad)
        good = pd.DataFrame({"best_lowest_energy": [-10.0], "num_heavy_atoms": [30.0]})
        assert ligand_efficiency_series(good).iloc[0] == pytest.approx(10.0 / 30.0)

    def test_ligand_efficiency_series_rejects_string_and_nan_atoms(self) -> None:
        # Regression: 'n/a'/NaN atom counts surfaced as raw ValueError /
        # "cannot convert float NaN to integer" from inside apply().
        for bad_value in ("n/a", float("nan")):
            frame = pd.DataFrame(
                {"best_lowest_energy": [-10.0], "num_heavy_atoms": [bad_value]}
            )
            with pytest.raises(DataValidationError, match="atom counts"):
                ligand_efficiency_series(frame)

    def test_ligand_efficiency_series_empty_frame_returns_series(self) -> None:
        # Regression: apply() on an empty frame returned a DataFrame, so
        # the standard ok["le"] = ligand_efficiency_series(ok) flow
        # crashed with "Cannot set a DataFrame with multiple columns".
        frame = pd.DataFrame(columns=["best_lowest_energy", "num_heavy_atoms"])
        result = ligand_efficiency_series(frame)
        assert isinstance(result, pd.Series)
        assert len(result) == 0
        ok = pd.DataFrame(columns=["folder"])
        ok["ligand_efficiency"] = result
        assert len(ok) == 0

    def test_ligand_efficiency_series_warns_on_missing_affinity(self, caplog) -> None:
        # Regression: NaN affinities produced silently-NaN efficiencies.
        frame = pd.DataFrame(
            {"best_lowest_energy": [float("nan"), -10.0], "num_heavy_atoms": [23, 30]}
        )
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.docking.scoring"):
                result = ligand_efficiency_series(frame)
        assert result.iloc[0] != result.iloc[0]  # NaN row stays NaN ...
        assert result.iloc[1] == pytest.approx(10.0 / 30.0)
        assert "missing/non-numeric affinity" in caplog.text  # ... but not silently

    def test_consensus_direction_defaults_for_energy_and_affinity_columns(self) -> None:
        # Regression: scan_vina_directory's 'best_affinity' (and
        # parse_dlg_summary's 'top_lowest_energy') defaulted to
        # higher-is-better, inverting the ranking: the WORST binder
        # scored 1.0.  Docstring promises binding energies default to
        # False (more negative = better).
        vina = pd.DataFrame(
            {"ligand": ["weak", "medium", "strong"], "best_affinity": [-5.0, -8.0, -11.0]}
        )
        scores = consensus_rescore(vina, columns=["best_affinity"])
        assert scores.iloc[2] == pytest.approx(1.0)  # strongest binder wins
        assert scores.iloc[0] == pytest.approx(0.0)
        dlg = pd.DataFrame({"top_lowest_energy": [-11.5, -10.0, -9.0]})
        scores_dlg = consensus_rescore(dlg, columns=["top_lowest_energy"])
        assert scores_dlg.iloc[0] == pytest.approx(1.0)

    def test_consensus_all_nan_column_raises(self) -> None:
        # Regression: the docstring promised DataValidationError on
        # all-NaN inputs, but the code warned "zero variance" and
        # scored the column a neutral 0.5.
        frame = pd.DataFrame({"e": [float("nan")] * 3, "n": [1, 5, 9]})
        with pytest.raises(DataValidationError, match="no numeric values"):
            consensus_rescore(frame, columns=["e", "n"])

    def test_consensus_non_numeric_column_raises(self) -> None:
        # Regression: strings were coerced to NaN by pd.to_numeric and
        # then masqueraded as a "zero variance" column scored 0.5.
        frame = pd.DataFrame({"e": ["a", "b", "c"], "n": [1, 5, 9]})
        with pytest.raises(DataValidationError, match="no numeric values"):
            consensus_rescore(frame, columns=["e", "n"])

    def test_consensus_partial_nan_warns_and_keeps_row_nan(self, caplog) -> None:
        frame = pd.DataFrame({"e": [-11.0, float("nan"), -9.0], "n": [1, 5, 9]})
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.docking.scoring"):
                scores = consensus_rescore(frame, columns=["e", "n"])
        assert scores.iloc[1] != scores.iloc[1]  # NaN row stays NaN ...
        assert scores.iloc[[0, 2]].notna().all()
        assert "NaN/non-numeric" in caplog.text  # ... but not silently

    def test_consensus_inf_values_score_nan_with_warning(self, caplog) -> None:
        # Regression: +-inf made the span non-finite, which tripped the
        # "zero variance" branch and flattened the whole column to 0.5.
        frame = pd.DataFrame({"e": [float("-inf"), -9.0, -8.0], "n": [1, 5, 9]})
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.docking.scoring"):
                scores = consensus_rescore(frame, columns=["e", "n"])
        assert scores.iloc[0] != scores.iloc[0]
        assert scores.iloc[[1, 2]].notna().all()
        assert "NaN/non-numeric" in caplog.text

    def test_consensus_empty_columns_list_raises(self) -> None:
        # Regression: used to fall through to the confusing
        # "consensus weights must not all be zero, got []".
        with pytest.raises(DataValidationError, match="columns list is empty"):
            consensus_rescore(pd.DataFrame({"a": [1.0]}), columns=[])

    def test_consensus_empty_frame_raises(self) -> None:
        frame = pd.DataFrame({"e": pd.Series(dtype=float), "n": pd.Series(dtype=float)})
        with pytest.raises(DataValidationError, match="frame is empty"):
            consensus_rescore(frame, columns=["e", "n"])
