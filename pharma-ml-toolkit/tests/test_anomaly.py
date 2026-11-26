"""Tests for :mod:`pharmaml.ml.anomaly` -- the part18 §4.5 lane.

The contract pins: planted outliers are flagged, the contamination
parameter drives the threshold honestly, the export carries per-
sample verdicts, the feature report is descriptive-only, the seed
pin reproduces verdicts exactly, and the guards refuse every bad
input with the typed error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.anomaly import AnomalyResult, isolation_forest_anomaly


@pytest.fixture()
def frame() -> pd.DataFrame:
    """Normal data with three planted far outliers (rows 0-2)."""
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("wxyz"))
    data.iloc[0] = data.iloc[0] + 18.0
    data.iloc[1] = data.iloc[1] - 20.0
    data.iloc[2] = data.iloc[2] + 21.0
    return data


class TestIsolationForest:
    def test_planted_outliers_flagged(self, frame: pd.DataFrame) -> None:
        result = isolation_forest_anomaly(frame, contamination=0.03)
        assert isinstance(result, AnomalyResult)
        assert set(result.outlier_indices[:3]) == {0, 1, 2}
        assert result.n_outliers >= 3
        assert result.n_outliers <= 8

    def test_threshold_signs(self, frame: pd.DataFrame) -> None:
        result = isolation_forest_anomaly(frame, contamination=0.03)
        labels = np.asarray(result.labels)
        scores = np.asarray(result.scores)
        # the reported cutoff sits between the outliers and the inliers:
        # every outlier at-or-below, every inlier at-or-above
        assert (scores[labels == -1] <= result.threshold + 1e-12).all()
        assert (scores[labels == 1] >= result.threshold - 1e-12).all()
        # and the flagged set are the LOWEST scores (most anomalous first)
        assert scores[labels == -1].max() <= scores[labels == 1].min() + 1e-12

    def test_anomaly_fraction_matches_contamination(self, frame: pd.DataFrame) -> None:
        result = isolation_forest_anomaly(frame, contamination=0.05)
        # 0.05 x 150 = 7.5 -> 8 flagged (the ceil the forest applies)
        assert result.n_outliers in (7, 8)
        assert result.anomaly_fraction == pytest.approx(result.n_outliers / 150)

    def test_auto_contamination(self, frame: pd.DataFrame) -> None:
        result = isolation_forest_anomaly(frame, contamination="auto")
        assert 1 <= result.n_outliers < 20
        assert result.contamination == "auto"

    def test_seed_reproduces_verdicts(self, frame: pd.DataFrame) -> None:
        first = isolation_forest_anomaly(frame, contamination=0.04, seed=7)
        second = isolation_forest_anomaly(frame, contamination=0.04, seed=7)
        np.testing.assert_array_equal(
            np.asarray(first.labels), np.asarray(second.labels)
        )
        np.testing.assert_allclose(
            np.asarray(first.scores), np.asarray(second.scores)
        )

    def test_as_frame_export(self, frame: pd.DataFrame) -> None:
        result = isolation_forest_anomaly(frame, contamination=0.03)
        out = result.as_frame()
        assert list(out.columns) == ["anomaly_score", "is_outlier", "outlier_rank"]
        assert out["is_outlier"].sum() == result.n_outliers
        assert out["outlier_rank"].max() == result.n_outliers

    def test_feature_report_descriptive(self, frame: pd.DataFrame) -> None:
        result = isolation_forest_anomaly(frame, contamination=0.03)
        report = result.feature_report(frame)
        assert list(report.columns) == ["feature", "inlier_mean", "outlier_mean", "shift_sigma"]
        assert list(report["feature"]) == ["w", "x", "y", "z"]
        # the planted shifts are large -> |shift_sigma| clearly above 2
        big = report.loc[
            report["feature"].isin(["w", "x", "y"]), "shift_sigma"
        ].abs()
        assert (big > 2.0).all()

    def test_str_uses_honest_wording(self, frame: pd.DataFrame) -> None:
        text = str(isolation_forest_anomaly(frame, contamination=0.03))
        assert "flagged" in text
        assert "trees" in text

    @pytest.mark.parametrize(
        "bad",
        [-0.1, 0.0, 0.55, 1.0, "auto0", None],
    )
    def test_bad_contamination_refused(self, frame: pd.DataFrame, bad) -> None:
        with pytest.raises(DataValidationError):
            isolation_forest_anomaly(frame, contamination=bad)

    def test_bad_estimators_refused(self, frame: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError):
            isolation_forest_anomaly(frame, n_estimators=0)

    def test_too_few_rows_refused(self) -> None:
        tiny = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        with pytest.raises(DataValidationError):
            isolation_forest_anomaly(tiny)

    def test_nan_refused(self, frame: pd.DataFrame) -> None:
        holed = frame.copy()
        holed.loc[0, "w"] = np.nan
        with pytest.raises(DataValidationError):
            isolation_forest_anomaly(holed)
