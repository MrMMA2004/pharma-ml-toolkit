"""Tests for the ``pharmaml.biology`` subpackage (cell line data).

Covers the three modules end to end on synthetic data built with fixed
numpy seeds: drug-response screen loading/auditing/reshaping
(:mod:`pharmaml.biology.response`), expression-table normalisation and
correlation (:mod:`pharmaml.biology.omics`) and the FDR-controlled
biomarker discovery flow (:mod:`pharmaml.biology.biomarkers`).
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from pharmaml.biology import (
    ResponseScreen,
    audit_response_table,
    call_sensitive_resistant,
    collapse_replicates,
    correlate_with_response,
    direction_of_association,
    impute_expression,
    ln_ic50_to_pic50,
    load_response_table,
    most_variable_genes,
    normalize_expression,
    pic50_to_ln_ic50,
    response_matrix,
    sensitivity_associations,
    summarize_compounds,
    top_biomarkers,
)
from pharmaml.exceptions import DataValidationError


@contextmanager
def _pharmaml_logs_visible():
    """Re-enable propagation on the ``pharmaml`` logger for caplog.

    Same dance as ``tests/test_ml_pipeline.py`` /
    ``tests/test_docking.py``: CLI tests set ``propagate=False`` on the
    ``pharmaml`` root logger, and pytest's caplog handler lives on the
    true root logger.
    """
    pharma_root = logging.getLogger("pharmaml")
    prev_propagate = pharma_root.propagate
    pharma_root.propagate = True
    try:
        yield
    finally:
        pharma_root.propagate = prev_propagate


def _screen_frame() -> pd.DataFrame:
    """Clean 4x3 screen (4 cell lines, 3 compounds, no issues)."""
    rows = []
    for cell in ("A549", "HCT116", "MCF7", "HL60"):
        for compound in ("cisplatin", "doxorubicin", "erlotinib"):
            rows.append(
                {
                    "cell_line": cell,
                    "compound": compound,
                    "tissue": "pancreas" if cell == "A549" else "colon",
                    "LN_IC50": 1.0 + 0.5 * len(cell),
                    "AUC": 0.5,
                }
            )
    return pd.DataFrame(rows)


class TestLoadResponseTable:
    def test_dataframe_input_with_autodetected_metrics(self) -> None:
        frame = _screen_frame()
        frame["study"] = "GDSC1"  # non-numeric column must NOT become a metric
        screen = load_response_table(frame, tissue_column="tissue")
        assert isinstance(screen, ResponseScreen)
        assert screen.metrics == ["LN_IC50", "AUC"]
        assert screen.tissue_col == "tissue"
        assert "study" in screen.frame.columns  # passthrough columns are kept

    def test_csv_path_input(self, tmp_path) -> None:
        path = tmp_path / "screen.csv"
        _screen_frame().to_csv(path, index=False)
        screen = load_response_table(path)
        assert screen.cell_lines == ["A549", "HCT116", "HL60", "MCF7"]
        assert screen.compounds == ["cisplatin", "doxorubicin", "erlotinib"]
        assert screen.metrics == ["LN_IC50", "AUC"]

    def test_missing_identifier_column_lists_available(self) -> None:
        frame = pd.DataFrame({"cell": ["A"], "compound": ["d"], "LN_IC50": [1.0]})
        with pytest.raises(DataValidationError, match="Available columns") as excinfo:
            load_response_table(frame)
        assert "cell" in str(excinfo.value)
        assert "compound" in str(excinfo.value)

    def test_missing_explicit_metric_column_raises(self) -> None:
        frame = _screen_frame()
        with pytest.raises(DataValidationError, match="AUC"):
            load_response_table(frame, metrics=["LN_IC50", "bogus"])

    def test_no_metric_columns_at_all_raises(self) -> None:
        frame = pd.DataFrame({"cell_line": ["A"], "compound": ["d"]})
        with pytest.raises(DataValidationError, match="no metric columns"):
            load_response_table(frame)

    def test_non_numeric_metric_values_coerce_with_warning(self, caplog) -> None:
        frame = pd.DataFrame(
            {
                "cell_line": ["A", "B", "C"],
                "compound": ["d", "d", "d"],
                "LN_IC50": ["1.5", "2.5", "not-a-number"],
            }
        )
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.biology.response"):
                screen = load_response_table(frame)
        assert "1 non-numeric value(s) in metric column 'LN_IC50'" in caplog.text
        assert screen.frame["LN_IC50"].tolist()[:2] == [1.5, 2.5]
        assert pd.isna(screen.frame["LN_IC50"].iloc[2])
        # the junk string also keeps the column auto-detectable
        assert screen.metrics == ["LN_IC50"]

    def test_tissue_column_is_optional(self) -> None:
        screen = load_response_table(_screen_frame())  # no tissue_column
        assert screen.tissue_col is None
        assert "tissue" not in screen.metrics  # string column, not a metric


class TestResponseScreen:
    def test_properties(self) -> None:
        frame = _screen_frame()
        screen = load_response_table(frame)
        assert screen.cell_lines == sorted(set(frame["cell_line"]))
        assert screen.compounds == sorted(set(frame["compound"]))
        assert screen.n_pairs == 12
        assert len(screen.frame) == 12

    def test_summary_contents(self) -> None:
        frame = _screen_frame()
        frame.loc[0, "LN_IC50"] = math.nan  # one missing value
        extra = frame.iloc[0].copy()  # duplicate (A549, cisplatin)
        extra["LN_IC50"] = 2.0
        screen = load_response_table(
            pd.concat([frame, pd.DataFrame([extra])], ignore_index=True),
            tissue_column="tissue",
        )
        text = screen.summary()
        assert "4 cell line(s) x 3 compound(s) -> 12 (cell, compound) pair(s)" in text
        assert "13 (1 duplicate row(s))" in text
        assert "LN_IC50" in text and "1 missing" in text
        assert "dup. pairs" in text and "(A549 + cisplatin)" in text
        assert "pancreas" in text  # tissue values are listed


class TestAuditResponseTable:
    def test_clean_table_reports_ok(self) -> None:
        screen = load_response_table(_screen_frame(), tissue_column="tissue")
        report = audit_response_table(screen)
        assert len(report) == 1  # exactly the "ok" row
        assert report.iloc[0]["issue"] == "ok"
        assert int(report.iloc[0]["count"]) == 0

    def test_planted_issues_are_all_reported(self) -> None:
        frame = _screen_frame()
        frame.loc[0, "LN_IC50"] = math.nan          # missing metric value
        frame.loc[5, "AUC"] = math.nan              # ... in the other metric too
        extra = frame.iloc[0].copy()                 # duplicate (A549, cisplatin)
        extra["LN_IC50"] = 2.0
        frame = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
        a549_rows = frame.index[frame["cell_line"] == "A549"][:3]
        frame.loc[a549_rows, "tissue"] = ["lung", "breast", "pancreas"]
        # A549 is now claimed by three tissues
        # one compound measured on only two cell lines
        rare = pd.DataFrame(
            {
                "cell_line": ["A549", "HCT116"],
                "compound": ["rare", "rare"],
                "tissue": ["lung", "colon"],
                "LN_IC50": [1.0, 1.5],
                "AUC": [0.4, 0.6],
            }
        )
        frame = pd.concat([frame, rare], ignore_index=True)
        screen = load_response_table(frame, tissue_column="tissue")
        report = audit_response_table(screen)

        missing = report[report["issue"] == "missing metric values"]
        assert set(missing["detail"]) == {"metric 'LN_IC50'", "metric 'AUC'"}
        assert set(missing["count"]) == {1}

        duplicates = report[report["issue"] == "duplicate pairs"]
        assert duplicates["detail"].tolist() == ["A549 / cisplatin"]
        assert duplicates["count"].tolist() == [2]

        ambiguous = report[report["issue"] == "ambiguous tissue"]
        assert ambiguous["detail"].tolist() == ["A549: breast, lung, pancreas"]
        assert ambiguous["count"].tolist() == [3]

        undertested = report[report["issue"] == "under-tested compound"]
        assert undertested["detail"].tolist() == ["rare"]
        assert undertested["count"].tolist() == [2]


class TestCollapseReplicates:
    def test_duplicate_pairs_collapse_to_median(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_line": ["A", "A", "B"],
                "compound": ["d", "d", "d"],
                "v": [1.0, 3.0, 10.0],
                "note": ["one", "two", "kept"],
            }
        )
        screen = load_response_table(frame, metrics=["v"])
        collapsed = collapse_replicates(screen)
        assert len(collapsed.frame) == 2  # row count drops
        pair = collapsed.frame[collapsed.frame["cell_line"] == "A"]
        assert pair["v"].tolist() == [2.0]  # median of [1.0, 3.0]
        assert pair["note"].tolist() == ["one"]  # first annotation kept
        # non-duplicate rows untouched
        assert collapsed.frame[collapsed.frame["cell_line"] == "B"]["v"].tolist() == [
            10.0
        ]
        assert collapsed.metrics == ["v"]
        # the original screen is not mutated
        assert len(screen.frame) == 3


class TestLnIc50Conversion:
    def test_known_values(self) -> None:
        assert ln_ic50_to_pic50(0.0) == pytest.approx(6.0)  # IC50 = 1 uM
        assert ln_ic50_to_pic50(math.log(2.0)) == pytest.approx(
            6.0 - math.log10(2.0)
        )  # IC50 = 2 uM
        assert ln_ic50_to_pic50(math.log(0.001)) == pytest.approx(9.0)  # 1 nM

    def test_round_trip(self) -> None:
        values = np.linspace(-8.0, 5.0, 27)
        round_trip = pic50_to_ln_ic50(ln_ic50_to_pic50(values))
        np.testing.assert_allclose(round_trip, values)

    def test_series_in_series_out_with_nan_passthrough(self) -> None:
        series = pd.Series([0.0, math.log(2.0), np.nan], index=["a", "b", "c"])
        result = ln_ic50_to_pic50(series)
        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b", "c"]
        assert result["a"] == pytest.approx(6.0)
        assert result["b"] == pytest.approx(6.0 - math.log10(2.0))
        assert pd.isna(result["c"])
        # and the inverse keeps its index too
        back = pic50_to_ln_ic50(result)
        assert isinstance(back, pd.Series)
        np.testing.assert_allclose(back[:2], series[:2])

    def test_scalar_and_array_kinds(self) -> None:
        assert isinstance(ln_ic50_to_pic50(1.5), float)
        array = ln_ic50_to_pic50(np.array([0.0, math.log(10.0)]))
        assert isinstance(array, np.ndarray)
        assert array[1] == pytest.approx(5.0)  # IC50 = 10 uM


class TestResponseMatrix:
    def test_shape_values_and_missing_pairs(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_line": ["B", "A", "A", "B"],
                "compound": ["y", "x", "y", "x"],
                "v": [2.0, 1.0, 4.0, 3.0],
            }
        )
        screen = load_response_table(frame, metrics=["v"])
        matrix = response_matrix(screen, "v")
        assert matrix.shape == (2, 2)
        assert list(matrix.index) == ["A", "B"]  # sorted
        assert list(matrix.columns) == ["x", "y"]  # sorted
        assert matrix.loc["A", "x"] == 1.0
        assert matrix.loc["B", "y"] == 2.0
        untested = pd.DataFrame(
            {"cell_line": ["C"], "compound": ["z"], "v": [9.0]}
        )
        wider = load_response_table(
            pd.concat([frame, untested], ignore_index=True), metrics=["v"]
        )
        wide_matrix = response_matrix(wider, "v")
        assert wide_matrix.loc["A", "z"] != wide_matrix.loc["A", "z"]  # NaN
        assert wide_matrix.loc["C", "z"] == 9.0

    def test_unknown_metric_raises_listing_metrics(self) -> None:
        screen = load_response_table(_screen_frame())
        with pytest.raises(DataValidationError, match="available metrics") as excinfo:
            response_matrix(screen, "bogus")
        assert "LN_IC50" in str(excinfo.value)

    def test_duplicate_pairs_refuse_to_pivot(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_line": ["A", "A"],
                "compound": ["d", "d"],
                "v": [1.0, 2.0],
            }
        )
        screen = load_response_table(frame, metrics=["v"])
        with pytest.raises(DataValidationError, match="collapse_replicates"):
            response_matrix(screen, "v")


class TestSummarizeCompounds:
    def test_hand_computed_statistics(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_line": ["A", "B", "C", "D", "A", "B", "C", "D"],
                "compound": ["d1"] * 4 + ["d2"] * 4,
                "v": [1.0, 2.0, 3.0, 5.0, 4.0, 4.0, 4.0, 4.0],
            }
        )
        screen = load_response_table(frame, metrics=["v"])
        summary = summarize_compounds(screen, "v")
        assert list(summary.index) == ["d1", "d2"]
        d1 = summary.loc["d1"]
        assert d1["n_cell_lines"] == 4
        assert d1["median"] == pytest.approx(2.5)
        assert d1["min"] == 1.0
        assert d1["max"] == 5.0
        assert d1["std"] == pytest.approx(1.707825127659933)  # sample std, ddof=1
        assert d1["sensitive_spread"] == pytest.approx(4.0)
        d2 = summary.loc["d2"]
        assert d2["std"] == 0.0
        assert d2["sensitive_spread"] == 0.0

    def test_direction_aware_most_sensitive_cell_line(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_line": ["A", "B", "C"],
                "compound": ["d", "d", "d"],
                "v": [3.0, 1.0, 2.0],
            }
        )
        screen = load_response_table(frame, metrics=["v"])
        lower = summarize_compounds(screen, "v")
        assert lower.loc["d", "most_sensitive_cell_line"] == "B"  # min = potent
        higher = summarize_compounds(screen, "v", higher_is_more_sensitive=True)
        assert higher.loc["d", "most_sensitive_cell_line"] == "A"  # max

    def test_unknown_metric_raises(self) -> None:
        screen = load_response_table(_screen_frame())
        with pytest.raises(DataValidationError, match="available metrics"):
            summarize_compounds(screen, "bogus")


class TestCallSensitiveResistant:
    def test_twelve_values_quantile_labels(self) -> None:
        values = pd.Series(np.arange(12, dtype=float))  # 0..11
        labels = call_sensitive_resistant(values)
        assert (labels == "sensitive").sum() == 3      # 0, 1, 2
        assert (labels == "resistant").sum() == 3      # 9, 10, 11
        assert (labels == "intermediate").sum() == 6
        assert labels.iloc[0] == "sensitive"
        assert labels.iloc[-1] == "resistant"
        assert labels.iloc[6] == "intermediate"

    def test_nan_entries_are_not_tested(self) -> None:
        values = pd.Series([np.nan] + list(range(11)), dtype=float)
        labels = call_sensitive_resistant(values)
        assert labels.iloc[0] == "not tested"
        assert (labels == "not tested").sum() == 1

    def test_higher_is_sensitive_flips_the_calls(self) -> None:
        values = pd.Series(np.arange(12, dtype=float))
        flipped = call_sensitive_resistant(values, lower_is_sensitive=False)
        assert flipped.iloc[0] == "resistant"   # low value, pIC50-style
        assert flipped.iloc[-1] == "sensitive"  # high value
        assert (flipped == "sensitive").sum() == 3
        assert (flipped == "resistant").sum() == 3

    def test_quantile_ordering_is_validated(self) -> None:
        values = pd.Series(np.arange(12, dtype=float))
        for bad in ((0.8, 0.75), (0.0, 0.75), (0.25, 1.0), (0.5, 0.5)):
            with pytest.raises(DataValidationError, match="quantiles must satisfy"):
                call_sensitive_resistant(
                    values, sensitive_quantile=bad[0], resistant_quantile=bad[1]
                )


class TestNormalizeExpression:
    def test_log2_then_zscore_hand_checked(self) -> None:
        frame = pd.DataFrame(
            {"G1": [1.0, 3.0], "G2": [0.0, 7.0]}, index=["c1", "c2"]
        )
        result = normalize_expression(frame)
        # log2(x+1) -> G1 [1, 2], G2 [0, 3]; both z-score to exactly [-1, +1]
        assert result["G1"].tolist() == pytest.approx([-1.0, 1.0])
        assert result["G2"].tolist() == pytest.approx([-1.0, 1.0])
        assert list(result.index) == ["c1", "c2"]
        # input untouched
        assert frame["G1"].tolist() == [1.0, 3.0]

    def test_nan_passthrough_and_partial_zscore(self) -> None:
        frame = pd.DataFrame({"G": [5.0, np.nan, 7.0]})
        result = normalize_expression(frame)
        assert pd.isna(result["G"].iloc[1])
        assert result["G"].iloc[0] == pytest.approx(-1.0)  # over the 2 observed
        assert result["G"].iloc[2] == pytest.approx(1.0)

    def test_negative_values_raise_with_count(self) -> None:
        frame = pd.DataFrame({"G": [1.0, -0.5, -2.0]})
        with pytest.raises(DataValidationError, match="2 negative"):
            normalize_expression(frame)

    def test_constant_gene_scores_zero_with_warning(self, caplog) -> None:
        frame = pd.DataFrame({"G": [4.0, 4.0, 4.0], "V": [1.0, 2.0, 3.0]})
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.biology.omics"):
                result = normalize_expression(frame)
        assert result["G"].tolist() == [0.0, 0.0, 0.0]
        assert "constant gene(s)" in caplog.text and "G" in caplog.text

    def test_flags_can_be_disabled(self) -> None:
        frame = pd.DataFrame({"G": [1.0, 3.0]})
        log_only = normalize_expression(frame, z_score=False)
        assert log_only["G"].tolist() == pytest.approx([1.0, 2.0])
        z_only = normalize_expression(frame, log_transform=False)
        assert z_only["G"].tolist() == pytest.approx([-1.0, 1.0])


class TestImputeExpression:
    def test_median_fill_per_gene(self) -> None:
        frame = pd.DataFrame(
            {"G1": [1.0, np.nan, 3.0], "G2": [10.0, 20.0, np.nan]}
        )
        result = impute_expression(frame)
        assert result["G1"].iloc[1] == pytest.approx(2.0)   # median of [1, 3]
        assert result["G2"].iloc[2] == pytest.approx(15.0)  # median of [10, 20]

    def test_mean_strategy(self) -> None:
        frame = pd.DataFrame({"G": [1.0, np.nan, 3.0]})
        result = impute_expression(frame, strategy="mean")
        assert result["G"].iloc[1] == pytest.approx(2.0)

    def test_all_nan_gene_stays_nan_with_warning(self, caplog) -> None:
        frame = pd.DataFrame({"G": [1.0, 2.0], "EMPTY": [np.nan, np.nan]})
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.biology.omics"):
                result = impute_expression(frame)
        assert result["EMPTY"].isna().all()
        assert "no observed values" in caplog.text and "EMPTY" in caplog.text

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(DataValidationError, match="strategy"):
            impute_expression(pd.DataFrame({"G": [1.0]}), strategy="mode")


class TestMostVariableGenes:
    def test_variable_beats_flat_genes(self) -> None:
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            {
                "flat": [5.0] * 10,
                "wild": rng.normal(0, 10, size=10),
                "mild": rng.normal(0, 1, size=10),
            }
        )
        assert most_variable_genes(frame, n=2) == ["wild", "mild"]
        assert most_variable_genes(frame, n=1) == ["wild"]

    def test_n_above_gene_count_returns_all_with_warning(self, caplog) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [5.0, 5.0]})
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.biology.omics"):
                genes = most_variable_genes(frame, n=50)
        assert genes == ["a", "b"]
        assert "only 2" in caplog.text

    def test_ties_are_deterministic(self) -> None:
        frame = pd.DataFrame(
            {"z": [1.0, 3.0], "a": [1.0, 3.0], "m": [1.0, 3.0]}
        )  # identical MAD -> alphabetical order
        assert most_variable_genes(frame, n=3) == ["a", "m", "z"]
        assert most_variable_genes(frame, n=3) == ["a", "m", "z"]  # stable

    def test_n_below_one_raises(self) -> None:
        with pytest.raises(DataValidationError, match="n must be"):
            most_variable_genes(pd.DataFrame({"a": [1.0]}), n=0)


class TestCorrelateWithResponse:
    @staticmethod
    def _panel(n: int = 12) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(9)
        response = pd.Series(rng.normal(size=n), index=[f"c{i}" for i in range(n)])
        frame = pd.DataFrame(
            rng.normal(50, 10, size=(n, 3)),
            index=response.index,
            columns=["planted", "random", "constant"],
        )
        frame["planted"] = 10.0 + 3.0 * response  # noise-free association
        frame["constant"] = 7.0
        return frame, response

    def test_planted_association_is_perfect(self) -> None:
        frame, response = self._panel()
        result = correlate_with_response(frame, response)
        planted = result[result["gene"] == "planted"].iloc[0]
        assert abs(planted["rho"]) == pytest.approx(1.0, abs=1e-12)
        assert planted["p_value"] < 1e-8
        assert list(result.columns) == ["gene", "rho", "p_value"]
        assert set(result["gene"]) == {"planted", "random", "constant"}

    def test_unassociated_gene_is_weak(self) -> None:
        frame, response = self._panel()
        result = correlate_with_response(frame, response)
        random_row = result[result["gene"] == "random"].iloc[0]
        assert abs(random_row["rho"]) < 0.7  # pinned with the fixed seed
        assert random_row["p_value"] > 0.05

    def test_constant_gene_stays_nan(self) -> None:
        frame, response = self._panel()
        result = correlate_with_response(frame, response)
        constant_row = result[result["gene"] == "constant"].iloc[0]
        assert pd.isna(constant_row["rho"])
        assert pd.isna(constant_row["p_value"])

    def test_short_gene_warns_and_stays_nan(self, caplog) -> None:
        frame, response = self._panel()
        frame["short"] = np.nan
        frame.loc[frame.index[:2], "short"] = 1.0  # only 2 joint values
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.biology.omics"):
                result = correlate_with_response(frame, response)
        short_row = result[result["gene"] == "short"].iloc[0]
        assert pd.isna(short_row["rho"])
        assert "fewer than 3 joint non-NaN" in caplog.text

    def test_pearson_method_and_unknown_method(self) -> None:
        frame, response = self._panel()
        pearson = correlate_with_response(frame, response, method="pearson")
        planted = pearson[pearson["gene"] == "planted"].iloc[0]
        assert planted["rho"] == pytest.approx(1.0)
        with pytest.raises(DataValidationError, match="method"):
            correlate_with_response(frame, response, method="kendall")

    def test_disjoint_indices_raise(self) -> None:
        frame, _ = self._panel()
        stranger = pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"])
        with pytest.raises(DataValidationError, match="no overlap"):
            correlate_with_response(frame, stranger)


class TestSensitivityAssociations:
    @staticmethod
    def _associations(n: int = 14):
        rng = np.random.default_rng(11)
        response = pd.Series(rng.normal(size=n), index=[f"c{i}" for i in range(n)])
        genes = ["BRCA1", "EGFR", "KRAS", "TP53", "MYC", "ALK", "PTEN", "RB1"]
        frame = pd.DataFrame(
            rng.normal(0, 1, size=(n, len(genes))),
            index=response.index,
            columns=genes,
        )
        frame["BRCA1"] = -2.0 * response + rng.normal(scale=0.05, size=n)
        # one gene with only 2 joint values -> NaN p must survive the FDR step
        frame["SHORTY"] = np.nan
        frame.loc[frame.index[:2], "SHORTY"] = 1.0
        return sensitivity_associations(frame, response), frame, response

    def test_planted_biomarker_survives_fdr(self) -> None:
        associations, _, _ = self._associations()
        brca1 = associations[associations["gene"] == "BRCA1"].iloc[0]
        assert bool(brca1["significant"]) is True
        assert brca1["rho"] < 0  # expression high tracks low response values
        assert brca1["fdr"] < 0.05
        assert list(associations.columns) == [
            "gene",
            "rho",
            "p_value",
            "fdr",
            "significant",
        ]

    def test_sorted_by_absolute_rho(self) -> None:
        associations, _, _ = self._associations()
        rho = associations["rho"].abs().to_numpy()
        assert (np.diff(rho[~np.isnan(rho)]) <= 1e-12).all()

    def test_fdr_monotonic_with_p(self) -> None:
        associations, _, _ = self._associations()
        paired = associations[["p_value", "fdr"]].dropna()
        by_p = paired.sort_values("p_value")
        assert (by_p["fdr"].diff().dropna() >= -1e-12).all()

    def test_nan_p_values_keep_nan_fdr(self) -> None:
        associations, _, _ = self._associations()
        shorty = associations[associations["gene"] == "SHORTY"].iloc[0]
        assert pd.isna(shorty["p_value"])
        assert pd.isna(shorty["fdr"])
        assert not shorty["significant"]

    def test_bad_alpha_raises(self) -> None:
        _, frame, response = self._associations()
        with pytest.raises(DataValidationError, match="fdr_alpha"):
            sensitivity_associations(frame, response, fdr_alpha=1.5)


class TestTopBiomarkers:
    def test_clamps_to_available_rows(self) -> None:
        associations, _, _ = TestSensitivityAssociations._associations()
        hits = top_biomarkers(associations, n=3, only_significant=False)
        assert len(hits) == 3
        assert hits.iloc[0]["gene"] == "BRCA1"

    def test_fewer_significant_than_n_falls_back_with_warning(self, caplog) -> None:
        associations, _, _ = TestSensitivityAssociations._associations()
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.biology.biomarkers"):
                hits = top_biomarkers(associations, n=50)
        assert len(hits) == len(associations)  # full table, ranked by |rho|
        assert "only 1 significant association(s)" in caplog.text
        assert "returning the top" in caplog.text

    def test_rejects_frames_without_association_columns(self) -> None:
        bogus = pd.DataFrame({"gene": ["a"], "rho": [0.1]})
        with pytest.raises(DataValidationError, match="Available columns"):
            top_biomarkers(bogus, n=5)

    def test_n_below_one_raises(self) -> None:
        associations, _, _ = TestSensitivityAssociations._associations()
        with pytest.raises(DataValidationError, match="n must be"):
            top_biomarkers(associations, n=0)


class TestDirectionOfAssociation:
    def test_both_signs_and_zero(self) -> None:
        assert direction_of_association(-0.9) == "expression high -> more sensitive"
        assert direction_of_association(0.9) == "expression high -> more resistant"
        assert direction_of_association(0.0) == "no association"

    def test_lower_is_sensitive_flip(self) -> None:
        # pIC50-style response: positive rho means expression tracks potency
        assert direction_of_association(0.9, lower_is_sensitive=False) == (
            "expression high -> more sensitive"
        )
        assert direction_of_association(-0.9, lower_is_sensitive=False) == (
            "expression high -> more resistant"
        )

    def test_nan_and_junk_are_no_association(self) -> None:
        assert direction_of_association(float("nan")) == "no association"
        assert direction_of_association("junk") == "no association"
        assert direction_of_association(None) == "no association"


class TestMetricAutoDetectionExcludesBooleans:
    """Boolean annotation flags must never be auto-detected as metrics.

    ``pandas`` counts ``bool`` dtype as numeric, so without an explicit
    exclusion a screen's ``msi_status``-style flag column would sneak
    into ``screen.metrics`` and pollute every per-metric statistic.
    """

    def test_bool_dtype_column_is_not_a_metric(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_line": ["A549", "HCT116", "MCF7"],
                "compound": ["cisplatin"] * 3,
                "msi_status": [True, False, True],
                "LN_IC50": [1.0, 2.0, 3.0],
            }
        )
        screen = load_response_table(frame)
        assert screen.metrics == ["LN_IC50"]
        # the flag column survives as passthrough data, just not as a metric
        assert "msi_status" in screen.frame.columns

    def test_nullable_boolean_object_column_is_not_a_metric(self) -> None:
        # [True, False, None] lands in object dtype -- the case a csv
        # round-trip of a partially-missing flag column produces
        frame = pd.DataFrame(
            {
                "cell_line": ["A549", "HCT116", "MCF7"],
                "compound": ["cisplatin"] * 3,
                "msi_status": [True, False, None],
                "LN_IC50": [1.0, 2.0, 3.0],
            }
        )
        assert frame["msi_status"].dtype == object
        screen = load_response_table(frame)
        assert screen.metrics == ["LN_IC50"]

    def test_explicit_bool_metric_is_still_allowed(self) -> None:
        # explicit metrics are the caller's call: True/False act as 1/0
        frame = pd.DataFrame(
            {
                "cell_line": ["A549", "HCT116"],
                "compound": ["cisplatin"] * 2,
                "flag": [True, False],
            }
        )
        screen = load_response_table(frame, metrics=["flag"])
        assert screen.metrics == ["flag"]
        summary = summarize_compounds(screen, "flag")
        assert summary.loc["cisplatin", "median"] == 0.5


class TestNaNIdentifierRows:
    """Rows with a missing cell/compound identifier: warn + keep, never drop.

    ``pandas`` ``groupby`` drops NaN-key rows by default; every reshaping
    site in :mod:`pharmaml.biology.response` must keep them (as their own
    NaN-keyed group / row) and the audit must make them visible instead.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cell_line": ["A", "A", None, "B"],
                "compound": ["d", "d", "d", None],
                "v": [1.0, 3.0, 99.0, 42.0],
            }
        )

    def test_cell_lines_and_compounds_never_stringify_nan(self) -> None:
        screen = load_response_table(self._frame(), metrics=["v"])
        assert screen.cell_lines == ["A", "B"]  # no "nan"/"None" junk entries
        assert screen.compounds == ["d"]
        assert "nan" not in screen.cell_lines + screen.compounds
        assert "None" not in screen.cell_lines + screen.compounds

    def test_n_pairs_counts_nan_key_pairs(self) -> None:
        screen = load_response_table(self._frame(), metrics=["v"])
        # 4 rows: (A, d) twice, (nan, d), (B, nan) -> 3 distinct pairs
        assert screen.n_pairs == 3
        assert len(screen.frame) == 4

    def test_collapse_replicates_keeps_nan_key_rows(self, caplog) -> None:
        screen = load_response_table(self._frame(), metrics=["v"])
        with _pharmaml_logs_visible():
            with caplog.at_level(
                logging.WARNING, logger="pharmaml.biology.response"
            ):
                collapsed = collapse_replicates(screen)
        assert len(collapsed.frame) == 3  # (A,d) median + (nan,d) + (B,nan)
        assert collapsed.frame[collapsed.frame["cell_line"] == "A"][
            "v"
        ].tolist() == [2.0]
        kept = collapsed.frame[collapsed.frame["cell_line"].isna()]
        assert kept["v"].tolist() == [99.0]
        kept_b = collapsed.frame[collapsed.frame["compound"].isna()]
        assert kept_b["v"].tolist() == [42.0]
        assert (
            "2 row(s) have a missing cell-line/compound identifier" in caplog.text
        )

    def test_audit_reports_missing_identifiers(self) -> None:
        screen = load_response_table(self._frame(), metrics=["v"])
        report = audit_response_table(screen)
        missing = report[report["issue"] == "missing identifier"]
        assert missing["detail"].tolist() == [
            "column 'cell_line'",
            "column 'compound'",
        ]
        assert missing["count"].tolist() == [1, 1]

    def test_audit_sees_nan_key_duplicate_pairs(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_line": ["A", "A", None, None],
                "compound": ["d", "d", "d", "d"],
                "v": [1.0, 2.0, 3.0, 4.0],
            }
        )
        screen = load_response_table(frame, metrics=["v"])
        report = audit_response_table(screen)
        duplicates = report[report["issue"] == "duplicate pairs"]
        assert set(duplicates["detail"]) == {"A / d", "nan / d"}
        assert set(duplicates["count"]) == {2}

    def test_summarize_compounds_keeps_nan_compound_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_line": ["A", "B", "C", "A", "B"],
                "compound": ["d", "d", "d", None, None],
                "v": [1.0, 2.0, 3.0, 7.0, 9.0],
            }
        )
        screen = load_response_table(frame, metrics=["v"])
        summary = summarize_compounds(screen, "v")
        assert len(summary) == 2  # the NaN compound is kept, not dropped
        nan_row = summary.loc[summary.index.isna()]
        assert len(nan_row) == 1
        assert nan_row["n_cell_lines"].iloc[0] == 2
        assert nan_row["median"].iloc[0] == 8.0  # median of [7, 9]
        # lower-is-sensitive default: the min (7.0, on cell line A) wins
        assert nan_row["most_sensitive_cell_line"].iloc[0] == "A"

    def test_response_matrix_keeps_nan_keys_after_collapse(self) -> None:
        frame = pd.DataFrame(
            {
                "cell_line": ["A", "A", None],
                "compound": ["x", "x", "y"],
                "v": [1.0, 3.0, 9.0],
            }
        )
        screen = load_response_table(frame, metrics=["v"])
        matrix = response_matrix(collapse_replicates(screen), "v")
        assert matrix.loc["A", "x"] == 2.0  # median of the duplicate pair
        nan_row = matrix.loc[matrix.index.isna()]
        assert nan_row["y"].iloc[0] == 9.0  # NaN-key row survives the pivot


class TestCorrelateAlignmentEdges:
    """Index alignment edge cases for ``correlate_with_response``.

    The docstring promises partial-overlap support (warn + correlate on
    the shared cell lines); duplicate labels make the pairing ambiguous
    and must raise a typed error instead of crashing downstream.
    """

    def test_expression_rows_without_response_warn_and_skip(
        self, caplog
    ) -> None:
        rng = np.random.default_rng(31)
        response = pd.Series(rng.normal(size=3), index=["c0", "c1", "c2"])
        frame = pd.DataFrame(
            rng.normal(size=(5, 2)),
            index=["c0", "c1", "c2", "c3", "c4"],
            columns=["planted", "random"],
        )
        frame.loc[frame.index[:3], "planted"] = 10.0 + 3.0 * response.to_numpy()
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.biology.omics"):
                result = correlate_with_response(frame, response)
        planted = result[result["gene"] == "planted"].iloc[0]
        assert abs(planted["rho"]) == pytest.approx(1.0, abs=1e-12)
        assert "3 shared cell line(s) (2 expression row(s)" in caplog.text

    def test_response_rows_without_expression_warn(self, caplog) -> None:
        rng = np.random.default_rng(33)
        response = pd.Series(rng.normal(size=4), index=["c0", "c1", "c2", "zzz"])
        frame = pd.DataFrame(
            rng.normal(size=(3, 1)), index=["c0", "c1", "c2"], columns=["planted"]
        )
        frame["planted"] = 10.0 + 3.0 * response.loc[frame.index].to_numpy()
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.biology.omics"):
                result = correlate_with_response(frame, response)
        planted = result[result["gene"] == "planted"].iloc[0]
        assert abs(planted["rho"]) == pytest.approx(1.0, abs=1e-12)
        assert "1 response value(s) unmatched" in caplog.text

    def test_duplicate_expression_labels_raise(self) -> None:
        rng = np.random.default_rng(35)
        frame = pd.DataFrame(
            rng.normal(size=(3, 1)), index=["a", "a", "b"], columns=["G"]
        )
        response = pd.Series(rng.normal(size=2), index=["a", "b"])
        with pytest.raises(DataValidationError, match="duplicate cell-line label"):
            correlate_with_response(frame, response)

    def test_duplicate_response_labels_raise(self) -> None:
        rng = np.random.default_rng(37)
        frame = pd.DataFrame(
            rng.normal(size=(2, 1)), index=["a", "b"], columns=["G"]
        )
        response = pd.Series(rng.normal(size=3), index=["a", "a", "b"])
        with pytest.raises(DataValidationError, match="duplicate cell-line label"):
            correlate_with_response(frame, response)

    def test_nan_labelled_rows_never_match(self, caplog) -> None:
        rng = np.random.default_rng(39)
        frame = pd.DataFrame(
            rng.normal(size=(4, 1)), index=["a", "b", "c", np.nan], columns=["G"]
        )
        response = pd.Series(rng.normal(size=3), index=["a", "b", "c"])
        with _pharmaml_logs_visible():
            with caplog.at_level(logging.WARNING, logger="pharmaml.biology.omics"):
                result = correlate_with_response(frame, response)
        assert len(result) == 1  # the gene itself still gets its row
        assert "1 expression row(s) and 0 response value(s) unmatched" in caplog.text
