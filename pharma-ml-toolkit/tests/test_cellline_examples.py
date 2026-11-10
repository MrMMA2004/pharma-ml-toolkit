"""Synthetic cell line data generator + demo script tests.

``make_synthetic_cellline_data.py`` produces the biology domain's demo
tables (response screen + expression matrix) with a **planted EGFR
biomarker**: EGFR expression (high in lung lines) drives sensitivity to
the EGFR inhibitors.  These tests pin the generator's structure, its
determinism, and -- the headline -- that the planted biomarker stays
recoverable through the real ``pharmaml.biology`` workflow.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

from pharmaml.biology import (
    collapse_replicates,
    load_response_table,
    sensitivity_associations,
)
from pharmaml.biology.omics import normalize_expression

# The generators live in examples/ (not an installed package).
_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from make_synthetic_cellline_data import (
    CELL_LINES,
    COMPOUNDS,
    make_expression_matrix,
    make_response_screen,
)


@pytest.fixture(autouse=True)
def _restore_script_logging_state():
    """(round 263) the script-main tests below call ``configure_logging``.

    ``run_cellline_analysis.main`` and the generator CLIs configure the
    ``pharmaml`` root logger (``propagate=False`` + a console handler
    bound to THIS test's captured stderr).  Both leak past the test:
    later ``caplog`` assertions in sibling modules sharing the pytest
    process capture nothing (records stop at the pharmaml handler; the
    stale stream then prints ``--- Logging error ---`` noise).  The
    ten-chunk gate caught this exact shape in round 249b for
    ``test_format_converters``; this module got the same leak through
    the example scripts.  The at-import propagate/handlers/level/disable
    state is restored after every test now.
    """
    prev_disable = logging.root.manager.disable
    pharma_root = logging.getLogger("pharmaml")
    prev_propagate = pharma_root.propagate
    prev_handlers = list(pharma_root.handlers)
    prev_level = pharma_root.level
    yield
    logging.disable(prev_disable)
    pharma_root.propagate = prev_propagate
    pharma_root.handlers[:] = prev_handlers
    pharma_root.setLevel(prev_level)


@pytest.fixture(scope="module")
def screen_frame() -> pd.DataFrame:
    return make_response_screen(seed=42, plant_issues=True)


@pytest.fixture(scope="module")
def expression_frame() -> pd.DataFrame:
    return make_expression_matrix(seed=43)


class TestSyntheticScreen:
    def test_shape_and_columns(self, screen_frame: pd.DataFrame) -> None:
        # 24 cell lines x 8 compounds = 192 pairs + 1 planted duplicate.
        assert len(screen_frame) == 193
        assert list(screen_frame.columns) == [
            "cell_line", "compound", "tissue", "LN_IC50", "AUC",
        ]
        assert screen_frame["cell_line"].nunique() == 24
        assert screen_frame["compound"].nunique() == 8
        assert screen_frame["tissue"].nunique() == 10

    def test_panel_matches_constant(self) -> None:
        assert len(CELL_LINES) == 24
        assert len(COMPOUNDS) == 8
        assert {tissue for _, tissue in CELL_LINES}.__len__() == 10
        assert "erlotinib" in COMPOUNDS and "gefitinib" in COMPOUNDS

    def test_planted_issues(self, screen_frame: pd.DataFrame) -> None:
        screen = load_response_table(screen_frame, tissue_column="tissue")
        pairs = screen.frame.groupby(["cell_line", "compound"]).size()
        assert int((pairs > 1).sum()) == 1  # one duplicate pair
        assert int(screen.frame["LN_IC50"].isna().sum()) == 1  # one missing value

    def test_clean_mode_has_no_issues(self) -> None:
        clean = make_response_screen(seed=42, plant_issues=False)
        assert len(clean) == 192
        assert not clean["LN_IC50"].isna().any()
        assert not clean.duplicated(["cell_line", "compound"]).any()

    def test_determinism(self) -> None:
        first = make_response_screen(seed=7)
        second = make_response_screen(seed=7)
        pd.testing.assert_frame_equal(first, second)
        other = make_response_screen(seed=8)
        assert not first.equals(other)

    def test_auc_tracks_potency(self, screen_frame: pd.DataFrame) -> None:
        # Lower LN_IC50 (more sensitive) -> lower AUC: a positive
        # rank correlation between the two metrics.
        rho = screen_frame["LN_IC50"].corr(screen_frame["AUC"], method="spearman")
        assert rho > 0.5

    def test_n_cell_lines_bounds(self) -> None:
        with pytest.raises(ValueError, match="n_cell_lines must be >= 6"):
            make_response_screen(n_cell_lines=5)
        with pytest.raises(ValueError, match="n_cell_lines must be <= 24"):
            make_response_screen(n_cell_lines=25)
        small = make_response_screen(n_cell_lines=6, plant_issues=False)
        assert small["cell_line"].nunique() == 6


class TestSyntheticExpression:
    def test_shape_and_index(self, expression_frame: pd.DataFrame) -> None:
        assert expression_frame.shape == (24, 300)
        assert list(expression_frame.index) == [name for name, _ in CELL_LINES]
        assert "EGFR" in expression_frame.columns
        assert "KRT8" in expression_frame.columns
        assert "GENE_001" in expression_frame.columns

    def test_egfr_lung_lines_run_hot(self, expression_frame: pd.DataFrame) -> None:
        lung = [n for n, t in CELL_LINES if t == "lung"]
        other = [n for n, t in CELL_LINES if t != "lung"]
        lung_mean = expression_frame.loc[lung, "EGFR"].mean()
        other_mean = expression_frame.loc[other, "EGFR"].mean()
        assert lung_mean > other_mean + 1.5

    def test_egfr_stable_across_seeds(self) -> None:
        # EGFR levels are a property of the cell line (own fixed seed),
        # so both tables agree on them regardless of the caller's seed.
        a = make_expression_matrix(seed=1)
        b = make_expression_matrix(seed=99)
        pd.testing.assert_series_equal(a["EGFR"], b["EGFR"])
        assert not a["GENE_001"].equals(b["GENE_001"])  # noise still varies

    def test_determinism(self) -> None:
        pd.testing.assert_frame_equal(
            make_expression_matrix(seed=5), make_expression_matrix(seed=5)
        )

    def test_gene_count_bounds(self) -> None:
        with pytest.raises(ValueError, match="n_genes must be >= 3"):
            make_expression_matrix(n_genes=2)


class TestPlantedBiomarker:
    """The headline: EGFR must surface through the real workflow."""

    def test_egfr_top_biomarker_for_erlotinib(
        self, screen_frame: pd.DataFrame, expression_frame: pd.DataFrame
    ) -> None:
        screen = collapse_replicates(
            load_response_table(screen_frame, tissue_column="tissue")
        )
        expr_norm = normalize_expression(expression_frame)
        response = (
            screen.frame[screen.frame["compound"] == "erlotinib"]
            .set_index("cell_line")["LN_IC50"]
        )
        associations = sensitivity_associations(
            expr_norm.loc[response.index], response
        )
        egfr = associations.set_index("gene").loc["EGFR"]
        # Higher EGFR -> lower LN_IC50 -> strong negative association ...
        assert egfr["rho"] < -0.8
        # ... significant after BH-FDR over all 300 genes.
        assert egfr["significant"] is True or egfr["significant"] == True
        # And it is the strongest association in the whole panel.
        strongest = associations.iloc[0]["gene"]
        assert strongest == "EGFR"

    def test_non_egfr_compound_has_no_egfr_signal(
        self, screen_frame: pd.DataFrame, expression_frame: pd.DataFrame
    ) -> None:
        screen = collapse_replicates(
            load_response_table(screen_frame, tissue_column="tissue")
        )
        expr_norm = normalize_expression(expression_frame)
        response = (
            screen.frame[screen.frame["compound"] == "cisplatin"]
            .set_index("cell_line")["LN_IC50"]
        )
        associations = sensitivity_associations(
            expr_norm.loc[response.index], response
        )
        egfr = associations.set_index("gene").loc["EGFR"]
        assert abs(egfr["rho"]) < 0.5  # no planted signal for cisplatin


class TestDemoScript:
    def test_importable(self) -> None:
        import run_cellline_analysis

        assert callable(run_cellline_analysis.main)

    def test_screen_and_expression_must_come_together(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import run_cellline_analysis

        monkeypatch.setattr(
            sys, "argv", ["run_cellline_analysis.py", "--screen", "only.csv"]
        )
        with pytest.raises(SystemExit) as excinfo:
            run_cellline_analysis.main()
        assert excinfo.value.code == 2
        assert "together" in capsys.readouterr().err

    def test_generator_cli_writes_tables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        import make_synthetic_cellline_data

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "make_synthetic_cellline_data.py",
                "--out-dir", str(tmp_path),
                "--seed", "3",
                "--clean",
            ],
        )
        make_synthetic_cellline_data.main()
        out = capsys.readouterr().out
        assert "response screen" in out and "expression" in out
        screen = pd.read_csv(tmp_path / "synthetic_screen.csv")
        assert len(screen) == 192  # --clean: no planted duplicate
        expression = pd.read_csv(tmp_path / "synthetic_expression.csv", index_col=0)
        assert expression.shape == (24, 300)
