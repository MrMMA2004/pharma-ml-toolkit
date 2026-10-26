"""Unit tests for bioactivity converters and table utilities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pharmaml.chem.converters import (
    EndpointDatasetReport,
    EndpointValue,
    add_pic50_column,
    ic50_to_pic50,
    ki_to_pki,
    parse_endpoint,
    prepare_pic50_dataset,
    split_table_into_chunks,
)
from pharmaml.exceptions import DataValidationError


class TestParseEndpoint:
    def test_plain_number(self) -> None:
        parsed = parse_endpoint(123.4)
        assert parsed == EndpointValue(value=123.4)
        assert not parsed.is_censored

    def test_censored_values(self) -> None:
        assert parse_endpoint("<0.5") == EndpointValue(value=0.5, qualifier="<")
        assert parse_endpoint(">10000").is_censored

    def test_unit_suffix_stripped(self) -> None:
        assert parse_endpoint("250 nM") == EndpointValue(value=250.0)

    def test_unit_suffix_stripped_regardless_of_case(self) -> None:
        """Unit tokens are lexed case-insensitively (27-A).

        Pre-fix, a pure case variance ('250 NM', '0.5 um', '10 MM') made
        the entry unparseable -- the row silently became NaN (or was
        dropped by ``add_pic50_column``) even though the caller's
        ``unit`` parameter, not the stripped token, is the authoritative
        scale, so no interpretation was at risk.
        """
        assert parse_endpoint("250 NM") == EndpointValue(value=250.0)
        assert parse_endpoint("0.5 um") == EndpointValue(value=0.5)
        assert parse_endpoint("0.5 UM") == EndpointValue(value=0.5)
        assert parse_endpoint("10 MM") == EndpointValue(value=10.0)
        assert parse_endpoint("2 pm") == EndpointValue(value=2.0)

    def test_micro_sign_and_greek_mu_both_accepted(self) -> None:
        # U+00B5 (micro sign) and U+03BC (Greek small mu) spell the same
        # unit; casefold() unifies them.
        assert parse_endpoint("10 \u00b5M") == EndpointValue(value=10.0)
        assert parse_endpoint("10 \u03bcM") == EndpointValue(value=10.0)

    def test_invalid_inputs_return_none(self) -> None:
        for bad in (None, np.nan, "", "activity", "n/a", 0, -5, float("inf")):
            assert parse_endpoint(bad) is None


class TestIC50ToPIC50:
    def test_known_conversions(self) -> None:
        assert ic50_to_pic50(1000) == pytest.approx(6.0)      # 1 uM
        assert ic50_to_pic50(10) == pytest.approx(8.0)        # 10 nM
        assert ic50_to_pic50(1) == pytest.approx(9.0)         # 1 nM

    def test_censored_value_uses_magnitude(self) -> None:
        assert ic50_to_pic50("<100") == pytest.approx(7.0)

    def test_unit_parameter(self) -> None:
        assert ic50_to_pic50(1.0, unit="uM") == pytest.approx(6.0)
        assert ic50_to_pic50(1.0, unit="M") == pytest.approx(0.0)

    def test_invalid_returns_none(self) -> None:
        assert ic50_to_pic50("garbage") is None
        assert ic50_to_pic50(0) is None

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(DataValidationError, match="unit"):
            ic50_to_pic50(1.0, unit="spoons")

    def test_pki_alias(self) -> None:
        assert ki_to_pki(100) == pytest.approx(7.0)

    def test_one_molar_endpoint_is_positive_zero(self) -> None:
        """``-log10(1.0)`` is IEEE ``-0.0``; never surface it (27-A).

        Pre-fix, ``ic50_to_pic50(1.0, unit="M")`` returned ``-0.0`` and
        ``add_pic50_column`` wrote it into the table, displaying as
        "-0.0" where "0.0" is correct.
        """
        for value, unit in [(1.0, "M"), (1e9, "nM"), (1000.0, "mM"), (1e6, "uM")]:
            result = ic50_to_pic50(value, unit=unit)
            assert math.copysign(1.0, result) == 1.0
            assert str(result) == "0.0"
        assert str(ki_to_pki(1.0, unit="M")) == "0.0"

    def test_add_pic50_column_writes_zero_not_negative_zero(self) -> None:
        frame = pd.DataFrame({"IC50 (M)": ["1", "0.1"]})
        result = add_pic50_column(frame, source_col="IC50 (M)", unit="M")
        assert str(result.loc[0, "pIC50"]) == "0.0"
        assert result.loc[1, "pIC50"] == pytest.approx(1.0)


class TestAddPIC50Column:
    def test_end_to_end(self) -> None:
        frame = pd.DataFrame(
            {
                "compound": ["A", "B", "C", "D"],
                "IC50 (nM)": [1000, 10, "invalid", "<50"],
            }
        )
        result = add_pic50_column(frame)
        assert "pIC50" in result.columns
        assert len(result) == 3  # invalid row dropped
        assert result.loc[0, "pIC50"] == pytest.approx(6.0)
        assert result.loc[1, "pIC50"] == pytest.approx(8.0)
        assert result.loc[2, "pIC50"] == pytest.approx(7.3, abs=0.01)  # log10(50)~1.7

    def test_keep_invalid_rows_option(self) -> None:
        frame = pd.DataFrame({"IC50 (nM)": [100.0, "bad"]})
        result = add_pic50_column(frame, drop_invalid=False)
        assert len(result) == 2
        assert result["pIC50"].isna().sum() == 1

    def test_missing_column_raises(self) -> None:
        with pytest.raises(DataValidationError, match="Missing required"):
            add_pic50_column(pd.DataFrame({"foo": [1]}), source_col="IC50 (nM)")

    def test_source_frame_not_mutated(self) -> None:
        frame = pd.DataFrame({"IC50 (nM)": [100.0]})
        add_pic50_column(frame)
        assert "pIC50" not in frame.columns


class TestPreparePic50Dataset:
    """D2.py's process_data, upgraded (100x-145)."""

    @pytest.mark.parametrize(
        "reserved",
        ["SMILES", "n_measurements", "endpoint_median"],
    )
    def test_out_col_collision_with_reserved_columns_refused(self, reserved: str) -> None:
        # (100x-146) a colliding out_col used to silently clobber its
        # twin in the result dict (the round-146 audit's duplicate-key
        # finding) -- the pipeline refuses it up front now
        frame = pd.DataFrame({"Ligand SMILES": ["CCO"], "EC50 (nM)": [100.0]})
        with pytest.raises(DataValidationError, match="reserved result column"):
            prepare_pic50_dataset(frame, out_col=reserved)

    def test_end_to_end_with_d2_spelling(self) -> None:
        frame = pd.DataFrame(
            {
                "Ligand SMILES": ["CCO", "CCO", "C", "", "CCN"],
                "EC50 (nM)": ["<50", 200.0, "junk", 10.0, 1e10],
            }
        )
        result, report = prepare_pic50_dataset(frame)
        assert result["SMILES"].tolist() == ["CCO"]
        assert result["n_measurements"].tolist() == [2]
        # median(50, 200) nM -> pEC50 = 9 - log10(125)
        assert result["pEC50"].iloc[0] == pytest.approx(9 - math.log10(125.0))
        assert report.rows_in == 5
        assert report.rows_valid == 2
        assert report.rows_out == 1
        assert report.rows_censored == 1
        assert report.rows_unparsable == 1
        assert report.rows_missing_smiles == 1
        assert report.rows_out_of_range == 1
        assert report.duplicates_collapsed == 1
        # The accounting identity: every input row lands in exactly one bucket.
        assert (
            report.rows_valid
            + report.rows_missing_smiles
            + report.rows_unparsable
            + report.rows_out_of_range
            == report.rows_in
        )
        assert report.rows_out == report.rows_valid - report.duplicates_collapsed

    def test_aggregate_modes(self) -> None:
        frame = pd.DataFrame(
            {"SMILES": ["CCO", "CCO", "CCO"], "EC50 (nM)": [100.0, 200.0, 1000.0]}
        )
        median_result, median_report = prepare_pic50_dataset(frame, aggregate="median")
        assert median_result["endpoint_median"].iloc[0] == pytest.approx(200.0)
        assert median_report.duplicates_collapsed == 2
        mean_result, _ = prepare_pic50_dataset(frame, aggregate="mean")
        assert mean_result["endpoint_median"].iloc[0] == pytest.approx(1300.0 / 3)
        first_result, _ = prepare_pic50_dataset(frame, aggregate="first")
        assert first_result["endpoint_median"].iloc[0] == pytest.approx(100.0)
        keep_result, keep_report = prepare_pic50_dataset(frame, aggregate="keep")
        assert len(keep_result) == 3
        assert keep_report.duplicates_collapsed == 0
        assert keep_result["n_measurements"].tolist() == [1, 1, 1]

    def test_explicit_columns_and_unit(self) -> None:
        frame = pd.DataFrame({"smi": ["CCO"], "activity": [1000.0]})
        result, report = prepare_pic50_dataset(
            frame, smiles_col="smi", endpoint_col="activity", unit="uM"
        )
        # 1000 uM = 1e-3 M -> pEC50 = 3
        assert result["pEC50"].iloc[0] == pytest.approx(3.0)
        assert (report.smiles_col, report.endpoint_col, report.unit) == (
            "smi",
            "activity",
            "uM",
        )

    def test_ic50_candidate_fallback(self) -> None:
        frame = pd.DataFrame({"SMILES": ["CCO"], "IC50 (nM)": [100.0]})
        _result, report = prepare_pic50_dataset(frame)
        assert report.endpoint_col == "IC50 (nM)"

    def test_upper_limit_filter_is_inclusive(self) -> None:
        frame = pd.DataFrame({"SMILES": ["CCO", "CCN"], "EC50 (nM)": [100.0, 101.0]})
        result, report = prepare_pic50_dataset(frame, upper_limit=100.0)
        assert result["SMILES"].tolist() == ["CCO"]
        assert report.rows_out_of_range == 1

    def test_missing_columns_raise_with_candidates(self) -> None:
        with pytest.raises(DataValidationError, match="no SMILES column found"):
            prepare_pic50_dataset(pd.DataFrame({"compound": ["A"]}))
        with pytest.raises(DataValidationError, match="no endpoint column found"):
            prepare_pic50_dataset(pd.DataFrame({"SMILES": ["CCO"]}))
        with pytest.raises(DataValidationError, match="'typo' not found"):
            prepare_pic50_dataset(
                pd.DataFrame({"SMILES": ["CCO"], "EC50 (nM)": [1.0]}),
                smiles_col="typo",
            )

    def test_bad_aggregate_and_limit_raise(self) -> None:
        frame = pd.DataFrame({"SMILES": ["CCO"], "EC50 (nM)": [1.0]})
        with pytest.raises(DataValidationError, match="aggregate must be"):
            prepare_pic50_dataset(frame, aggregate="sum")
        with pytest.raises(DataValidationError, match="upper_limit must be"):
            prepare_pic50_dataset(frame, upper_limit=0.0)
        with pytest.raises(DataValidationError, match="upper_limit must be"):
            prepare_pic50_dataset(frame, upper_limit=float("inf"))

    def test_bad_unit_raises_early(self) -> None:
        frame = pd.DataFrame({"SMILES": ["CCO"], "EC50 (nM)": [1.0]})
        with pytest.raises(DataValidationError, match="Unknown concentration unit"):
            prepare_pic50_dataset(frame, unit="furlongs")

    def test_no_usable_rows_raises(self) -> None:
        frame = pd.DataFrame({"SMILES": ["CCO"], "EC50 (nM)": ["junk"]})
        with pytest.raises(DataValidationError, match="no usable rows"):
            prepare_pic50_dataset(frame)

    def test_report_str_and_as_dict(self) -> None:
        frame = pd.DataFrame(
            {"Ligand SMILES": ["CCO", "CCO"], "EC50 (nM)": ["<50", 200.0]}
        )
        _result, report = prepare_pic50_dataset(frame)
        text = str(report)
        assert "smiles='Ligand SMILES'" in text
        assert "2 in -> 1 out" in text
        assert "censored magnitudes used: 1" in text
        as_dict = report.as_dict()
        assert as_dict["rows_in"] == 2
        assert as_dict["aggregate"] == "median"
        assert isinstance(report, EndpointDatasetReport)

    def test_source_frame_not_mutated(self) -> None:
        frame = pd.DataFrame({"SMILES": ["CCO"], "EC50 (nM)": [100.0]})
        prepare_pic50_dataset(frame)
        assert "pEC50" not in frame.columns
        assert len(frame) == 1


class TestSplitTable:
    def test_chunking(self, tmp_path) -> None:
        source = tmp_path / "input.csv"
        pd.DataFrame({"x": range(10)}).to_csv(source, index=False)
        parts = split_table_into_chunks(source, tmp_path / "out", chunk_rows=4)
        assert len(parts) == 3
        rebuilt = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        assert list(rebuilt["x"]) == list(range(10))

    def test_invalid_chunk_size(self, tmp_path) -> None:
        source = tmp_path / "input.csv"
        pd.DataFrame({"x": [1]}).to_csv(source, index=False)
        with pytest.raises(DataValidationError):
            split_table_into_chunks(source, tmp_path, chunk_rows=0)
