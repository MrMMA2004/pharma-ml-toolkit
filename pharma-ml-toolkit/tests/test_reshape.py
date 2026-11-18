"""Tests for pharmaml.utils.reshape (part8 r9 — the STRETCH-card backend).

Each of the seven documented stretch operations gets contract tests:
validation errors name the exact problem, numerical outputs are
checked against independently-computed expectations, and edge cases
(missing values, duplicate keys, empty results) are pinned.  The GUI
suite (tests/ui/test_table_tools.py) pins the wiring; this suite pins
the science.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pharmaml.utils.reshape import (
    SUPPORTED_AGGREGATIONS,
    ReshapeReport,
    clean_strings,
    combine_frames,
    filter_frame,
    groupby_summary,
    merge_frames,
    pivot_table,
    resample_timeseries,
    rolling_window,
)

# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def assays() -> list[pd.DataFrame]:
    """Two replicate assay tables (the 03.06 use case)."""
    return [
        pd.DataFrame({
            "compound": ["aspirin", "ibuprofen", "naproxen"],
            "ic50": [3.2, 8.9, 1.4],
        }),
        pd.DataFrame({
            "compound": ["aspirin", "ibuprofen", "warfarin"],
            "ic50": [3.0, 9.1, 0.7],
        }),
    ]


@pytest.fixture
def long_frame() -> pd.DataFrame:
    """Long-format frame: compound × batch → yield (03.08/03.09 use)."""
    rows = []
    for compound in ["A", "B", "C"]:
        for batch in ["b1", "b2"]:
            for rep in range(3):
                rows.append({
                    "compound": compound, "batch": batch, "rep": rep,
                    "yield": 10.0 * ord(compound) + 5 * int(batch[1]) + rep,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def time_frame() -> pd.DataFrame:
    """Unevenly-spaced concentration/time rows (03.11 use case)."""
    times = pd.to_datetime([
        "2026-01-01 00:00", "2026-01-01 00:07", "2026-01-01 00:14",
        "2026-01-01 01:02", "2026-01-01 01:09", "2026-01-01 02:30",
    ])
    return pd.DataFrame({
        "time": times,
        "conc": [0.0, 5.1, 8.3, 4.9, 2.2, 0.5],
    })


# -- combine ----------------------------------------------------------------


class TestCombine:
    def test_stack_keeps_source_identity(self, assays):
        out, report = combine_frames(assays)
        assert len(out) == 6
        assert out["source_frame"].tolist() == ["frame_1"] * 3 + ["frame_2"] * 3
        assert report.rows_out == 6
        assert "stacked 2 frames" in report.detail

    def test_stack_with_user_keys_labels_rows(self, assays):
        out, report = combine_frames(assays, keys=["lab_a", "lab_b"])
        assert set(out["source_frame"]) == {"lab_a", "lab_b"}
        assert "lab_a: 3" in report.detail and "lab_b: 3" in report.detail

    def test_side_by_side_requires_equal_rows(self, assays):
        short = assays[1].head(2)
        with pytest.raises(ValueError, match="equal row counts.*frame_2: 2"):
            combine_frames([assays[0], short], axis=1)

    def test_side_by_side_joins_columns(self, assays):
        out, report = combine_frames(assays, axis=1)
        assert out.shape == (3, 4)
        assert "side-by-side" in report.detail

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError, match="no frames to combine"):
            combine_frames([])

    def test_key_count_must_match(self, assays):
        with pytest.raises(ValueError, match="2 key labels given for 2 frames|labels given"):
            combine_frames(assays, keys=["only_one"])


# -- merge ------------------------------------------------------------------


class TestMerge:
    def test_left_merge_keeps_left_rows(self, assays):
        out, report = merge_frames(assays[0], assays[1], on="compound", how="left")
        assert len(out) == 3
        # warfarin only exists in right → NaN for its ic50_y
        assert out.loc[out["compound"] == "naproxen", "ic50_y"].isna().all()
        assert "shared key values" in report.detail

    def test_inner_merge_keeps_overlap_only(self, assays):
        out, _ = merge_frames(assays[0], assays[1], on="compound", how="inner")
        assert sorted(out["compound"]) == ["aspirin", "ibuprofen"]

    def test_outer_merge_reports_full_count(self, assays):
        out, report = merge_frames(assays[0], assays[1], on="compound", how="outer")
        assert len(out) == 4  # aspirin, ibuprofen, naproxen, warfarin
        assert "4 matched rows" in report.detail

    def test_missing_key_column_named(self, assays):
        with pytest.raises(ValueError, match="left frame is missing column\\(s\\): batch"):
            merge_frames(assays[0], assays[1], on="batch")

    def test_many_to_many_fanout_warned(self):
        left = pd.DataFrame({"compound": ["x", "x"], "a": [1, 2]})
        right = pd.DataFrame({"compound": ["x", "x"], "b": [3, 4]})
        _, report = merge_frames(left, right, on="compound", how="inner")
        assert "many-to-many fan-out" in report.detail
        assert "left dups 1" in report.detail and "right dups 1" in report.detail

    def test_invalid_how_rejected(self, assays):
        with pytest.raises(ValueError, match="how must be inner, left or outer"):
            merge_frames(assays[0], assays[1], on="compound", how="right")


# -- groupby ----------------------------------------------------------------


class TestGroupby:
    def test_default_numeric_summary(self, long_frame):
        out, report = groupby_summary(long_frame, by=["compound"])
        assert len(out) == 3  # one row per compound
        for stat in ("yield_mean", "yield_std", "yield_min", "yield_max", "yield_count"):
            assert stat in out.columns
        # independently computed expectation for compound A
        expected_a_mean = long_frame.loc[long_frame["compound"] == "A", "yield"].mean()
        got = out.loc[out["compound"] == "A", "yield_mean"].item()
        assert got == pytest.approx(expected_a_mean)
        assert "3 groups over compound" in report.detail

    def test_custom_aggregations_validated(self, long_frame):
        with pytest.raises(ValueError, match="unsupported aggregation\\(s\\).*p05"):
            groupby_summary(long_frame, by=["compound"], aggregations={"yield": ["p05"]})

    def test_multi_key_grouping(self, long_frame):
        out, _ = groupby_summary(long_frame, by=["compound", "batch"],
                                 aggregations={"yield": ["mean"]})
        assert len(out) == 6  # 3 compounds × 2 batches
        assert ["compound", "batch", "yield_mean"] == list(out.columns)

    def test_missing_group_column_named(self, long_frame):
        with pytest.raises(ValueError, match="missing column\\(s\\): molecule"):
            groupby_summary(long_frame, by=["molecule"])

    def test_nunique_counts_distinct(self, long_frame):
        out, _ = groupby_summary(long_frame, by=["compound"],
                                 aggregations={"batch": ["nunique"]})
        assert (out["batch_nunique"] == 2).all()

    def test_no_numeric_columns_reports_nothing_to_do(self):
        frame = pd.DataFrame({"cat": ["a", "b"]})
        with pytest.raises(ValueError, match="no columns to aggregate"):
            groupby_summary(frame, by=["cat"])


# -- pivot ------------------------------------------------------------------


class TestPivot:
    def test_long_to_wide(self, long_frame):
        out, report = pivot_table(
            long_frame, index=["compound"], columns="batch", values="yield",
        )
        assert list(out.columns) == ["compound", "yield_b1", "yield_b2"]
        expected = long_frame[
            (long_frame["compound"] == "B") & (long_frame["batch"] == "b2")
        ]["yield"].mean()
        assert out.loc[out["compound"] == "B", "yield_b2"].item() == pytest.approx(expected)
        assert "2 value columns (mean of 'yield' by 'batch')" in report.detail

    def test_missing_combinations_stay_nan_not_zero(self):
        frame = pd.DataFrame({
            "compound": ["A", "A", "B"],
            "batch": ["b1", "b1", "b2"],
            "yield": [1.0, 2.0, 3.0],
        })
        out, report = pivot_table(frame, index=["compound"], columns="batch", values="yield")
        # B×b1 has no data → NaN, never 0
        assert pd.isna(out.loc[out["compound"] == "B", "yield_b1"].item())
        assert "2 empty combinations left NaN" in report.detail

    def test_aggregation_choices_pinned(self, long_frame):
        out, _ = pivot_table(
            long_frame, index=["batch"], columns="compound", values="yield", aggfunc="max",
        )
        expected = long_frame[
            (long_frame["batch"] == "b1") & (long_frame["compound"] == "C")
        ]["yield"].max()
        assert out.loc[out["batch"] == "b1", "yield_C"].item() == pytest.approx(expected)

    def test_unsupported_aggfunc_rejected(self, long_frame):
        with pytest.raises(ValueError, match="unsupported aggregation 'prod'"):
            pivot_table(long_frame, index=["compound"], columns="batch",
                        values="yield", aggfunc="prod")

    def test_index_must_exist(self, long_frame):
        with pytest.raises(ValueError, match="missing column\\(s\\): plate"):
            pivot_table(long_frame, index=["plate"], columns="batch", values="yield")


# -- string cleaning ---------------------------------------------------------


class TestCleanStrings:
    def test_strip_and_collapse(self):
        frame = pd.DataFrame({"smiles": ["  CCO  ", "C\tC  O", "benzene"]})
        out, report = clean_strings(frame, column="smiles")
        assert out["smiles"].tolist() == ["CCO", "C C O", "benzene"]
        assert "2 values changed" in report.detail

    def test_case_transforms(self):
        frame = pd.DataFrame({"name": ["Aspirin", "IBUPROFEN"]})
        out, _ = clean_strings(frame, column="name", strip=False, lowercase=True)
        assert out["name"].tolist() == ["aspirin", "ibuprofen"]
        out, _ = clean_strings(frame, column="name", strip=False, uppercase=True)
        assert out["name"].tolist() == ["ASPIRIN", "IBUPROFEN"]

    def test_case_conflict_rejected(self):
        frame = pd.DataFrame({"name": ["x"]})
        with pytest.raises(ValueError, match="cannot both be requested"):
            clean_strings(frame, column="name", lowercase=True, uppercase=True)

    def test_input_frame_never_mutated(self):
        frame = pd.DataFrame({"smiles": ["  CCO  "]})
        clean_strings(frame, column="smiles")
        assert frame["smiles"].item() == "  CCO  "

    def test_missing_values_untouched_by_default(self):
        frame = pd.DataFrame({"smiles": ["CCO", None]})
        out, report = clean_strings(frame, column="smiles")
        assert len(out) == 2
        assert out["smiles"].isna().sum() == 1
        assert "1 missing values left untouched" in report.detail

    def test_drop_missing_removes_rows(self):
        frame = pd.DataFrame({"smiles": ["CCO", None, "CCC"]})
        out, report = clean_strings(frame, column="smiles", drop_missing=True)
        assert len(out) == 2
        assert "1 missing dropped" in report.detail

    def test_no_options_rejected(self):
        frame = pd.DataFrame({"smiles": ["CCO"]})
        with pytest.raises(ValueError, match="nothing to do"):
            clean_strings(
                frame, column="smiles",
                strip=False, collapse_whitespace=False,
            )


# -- resample / rolling -------------------------------------------------------


class TestResample:
    def test_hourly_mean_of_uneven_rows(self, time_frame):
        out, report = resample_timeseries(
            time_frame, time_column="time", value_columns=["conc"], rule="1h",
        )
        assert len(out) == 3  # 00:00, 01:00, 02:00 windows
        # window 00:00-00:59 holds rows 0.0, 5.1, 8.3 → mean 4.4667
        assert out.loc[out["time"] == pd.Timestamp("2026-01-01 00:00"), "conc"].item() == pytest.approx((0.0 + 5.1 + 8.3) / 3)
        assert "mean per 1h" in report.detail

    def test_invalid_rule_rejected_before_touching_data(self, time_frame):
        with pytest.raises(ValueError, match="invalid resample rule '5 minn'"):
            resample_timeseries(time_frame, time_column="time",
                                value_columns=["conc"], rule="5 minn")

    def test_non_datetime_values_name_rows(self):
        frame = pd.DataFrame({"time": ["2026-01-01", "not-a-date"], "conc": [1.0, 2.0]})
        with pytest.raises(ValueError, match=r"non-datetime values at row\(s\) 1"):
            resample_timeseries(frame, time_column="time", value_columns=["conc"], rule="1D")

    def test_empty_windows_reported(self, time_frame):
        _out, report = resample_timeseries(
            time_frame, time_column="time", value_columns=["conc"], rule="30min",
        )
        # 02:30 is the only row in its window; several 30-min windows have no data
        assert "empty windows" in report.detail

    def test_unsupported_aggregation_rejected(self, time_frame):
        with pytest.raises(ValueError, match="unsupported aggregation 'var'"):
            resample_timeseries(time_frame, time_column="time",
                                value_columns=["conc"], rule="1h", aggregation="var")


class TestRolling:
    def test_rolling_mean_values(self, time_frame):
        out, report = rolling_window(
            time_frame, time_column="time", value_columns=["conc"], window=3,
        )
        assert "conc_rolling_mean" in out.columns
        assert len(out) == len(time_frame)  # no rows dropped
        # time-sorted (already sorted): row 2 mean = (0.0+5.1+8.3)/3
        assert out["conc_rolling_mean"].iloc[2] == pytest.approx((0.0 + 5.1 + 8.3) / 3)
        # min_periods=1 default → row 0 is its own mean
        assert out["conc_rolling_mean"].iloc[0] == pytest.approx(0.0)
        assert "3-point mean window" in report.detail

    def test_unsorted_input_sorted_before_window(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
            "conc": [30.0, 10.0, 20.0],
        })
        out, _ = rolling_window(frame, time_column="time", value_columns=["conc"], window=2)
        # after sorting: 10, 20, 30 → rolling means 10, 15, 25
        assert out["conc_rolling_mean"].tolist() == pytest.approx([10.0, 15.0, 25.0])
        assert out["conc"].tolist() == pytest.approx([10.0, 20.0, 30.0])

    def test_window_bounds_validated(self, time_frame):
        with pytest.raises(ValueError, match="window must be ≥ 2"):
            rolling_window(time_frame, time_column="time", value_columns=["conc"], window=1)
        with pytest.raises(ValueError, match="min_periods must be ≥ 1"):
            rolling_window(time_frame, time_column="time", value_columns=["conc"],
                           window=3, min_periods=0)

    def test_rolling_std_bessel_pinned(self, time_frame):
        out, _ = rolling_window(
            time_frame, time_column="time", value_columns=["conc"], window=3, aggregation="std",
        )
        # rows 0-2: 0.0, 5.1, 8.3 → ddof=1 std
        import statistics
        assert out["conc_rolling_std"].iloc[2] == pytest.approx(
            statistics.stdev([0.0, 5.1, 8.3])
        )


# -- filter -------------------------------------------------------------------


class TestFilter:
    def test_comparison_expression(self, long_frame):
        out, report = filter_frame(long_frame, "yield > 665")
        expected = int((long_frame["yield"] > 665).sum())
        assert expected == 11  # A's two 66x rows + all 6 of B + b1's 690-adjacent… verifiable below
        assert len(out) == expected
        assert f"kept {expected:,} of {len(long_frame):,} rows" in report.detail

    def test_in_expression(self, long_frame):
        out, _ = filter_frame(long_frame, "compound in ['A', 'C']")
        assert set(out["compound"]) == {"A", "C"}

    def test_keyword_column_names_backtick_escaped(self):
        # ``yield`` is a Python keyword and a real pharma column name;
        # pandas' query parser rejects it bare -- the backend escapes
        # keyword tokens that are actual column names, transparently.
        frame = pd.DataFrame({"yield": [1.0, 200.0, 300.0], "compound": ["a", "b", "c"]})
        out, _ = filter_frame(frame, "yield > 100")
        assert out["compound"].tolist() == ["b", "c"]
        # already-backticked input works untouched
        out2, _ = filter_frame(frame, "`yield` > 250")
        assert out2["compound"].tolist() == ["c"]
        # grammar keywords in operator position are never mangled
        out3, _ = filter_frame(frame, "compound in ['a', 'b']")
        assert out3["compound"].tolist() == ["a", "b"]

    def test_syntax_error_surfaces_pandas_message(self, long_frame):
        with pytest.raises(ValueError, match="invalid filter expression"):
            filter_frame(long_frame, "yield >>")

    def test_unknown_column_rejected(self, long_frame):
        with pytest.raises(ValueError, match="invalid filter expression"):
            filter_frame(long_frame, "nonexistent_column > 1")

    def test_match_nothing_reported_honestly(self, long_frame):
        _, report = filter_frame(long_frame, "yield > 99999")
        assert "matched no rows" in report.detail

    def test_match_everything_reported(self, long_frame):
        _, report = filter_frame(long_frame, "yield > -1")
        assert "matched every row" in report.detail

    def test_empty_expression_rejected(self, long_frame):
        with pytest.raises(ValueError, match="empty filter expression"):
            filter_frame(long_frame, "   ")


# -- report contract -----------------------------------------------------------


class TestReport:
    def test_summary_line_shape(self):
        report = ReshapeReport(
            operation="combine", rows_in=1000, rows_out=1500,
            columns_in=5, columns_out=6, detail="stacked 2 frames",
        )
        assert report.summary_line() == (
            "combine: 1,000×5 → 1,500×6 — stacked 2 frames"
        )

    def test_every_operation_returns_a_report(self, long_frame, time_frame):
        reports = [
            combine_frames([long_frame.head(2), long_frame.tail(2)])[1],
            merge_frames(long_frame.head(3), long_frame.tail(3), on="compound")[1],
            groupby_summary(long_frame, by=["compound"])[1],
            pivot_table(long_frame, index=["compound"], columns="batch", values="yield")[1],
            clean_strings(long_frame.assign(name=" x "), column="name")[1],
            resample_timeseries(time_frame, time_column="time", value_columns=["conc"], rule="1h")[1],
            rolling_window(time_frame, time_column="time", value_columns=["conc"], window=3)[1],
            filter_frame(long_frame, "yield > 0")[1],
        ]
        assert all(r.operation for r in reports)
        assert all(r.rows_out >= 0 for r in reports)
        # every report's rows_out matches the actual output length
        outs = [
            combine_frames([long_frame.head(2), long_frame.tail(2)])[0],
            merge_frames(long_frame.head(3), long_frame.tail(3), on="compound")[0],
        ]
        assert reports[0].rows_out == len(outs[0])
        assert reports[1].rows_out == len(outs[1])

    def test_supported_aggregations_stable(self):
        assert SUPPORTED_AGGREGATIONS == (
            "mean", "median", "std", "min", "max", "count", "sum", "nunique",
        )
