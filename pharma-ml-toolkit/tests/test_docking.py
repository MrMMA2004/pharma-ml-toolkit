"""Unit tests for docking preparation, parsing, and scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.docking.parsing import (
    check_docking_completion,
    parse_dlg_cluster_table,
    parse_dlg_summary,
    parse_vina_output,
    scan_docking_directory,
)
from pharmaml.docking.preparation import (
    GridBox,
    compute_grid_box,
    extract_pdbqt_atom_types,
    write_gpf,
)
from pharmaml.docking.scoring import consensus_rescore, ligand_efficiency
from pharmaml.exceptions import DataValidationError, DockingParseError


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
        text = "\n".join(
            [
                "REMARK VINA RESULT:    -9.5      1.234      2.345",
                "REMARK VINA RESULT:    -8.9      2.000      3.000",
            ]
        )
        poses = parse_vina_output(text)
        assert poses[0].affinity == pytest.approx(-9.5)
        assert poses[0].rmsd_lb == pytest.approx(1.234)

    def test_empty_raises(self) -> None:
        with pytest.raises(DockingParseError, match="no Vina pose"):
            parse_vina_output("nothing useful\n")


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
