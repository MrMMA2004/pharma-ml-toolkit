"""Unit tests for bioactivity converters and table utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.chem.converters import (
    EndpointValue,
    add_pic50_column,
    ic50_to_pic50,
    ki_to_pki,
    parse_endpoint,
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
