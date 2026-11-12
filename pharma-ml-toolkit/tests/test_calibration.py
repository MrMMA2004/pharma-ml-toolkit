"""Tests for probability calibration (:mod:`pharmaml.ml.calibration`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.calibration import (
    CalibrationReport,
    brier_score,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    recalibrate,
    reliability_bins,
)
from pharmaml.ml.persistence import save_model


class TestBrierScore:
    def test_perfect_predictions_score_zero(self):
        y = np.array([0, 0, 1, 1])
        assert brier_score(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0

    def test_constant_half_on_balanced_is_a_quarter(self):
        y = np.array([0, 0, 1, 1])
        assert brier_score(y, np.full(4, 0.5)) == 0.25

    def test_accepts_arbitrary_two_labels(self):
        # The larger label is the positive class.
        assert brier_score(["no", "yes"], np.array([0.0, 1.0])) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(DataValidationError, match="same length"):
            brier_score([0, 1, 0], [0.1, 0.2])

    def test_probability_out_of_range_raises(self):
        with pytest.raises(DataValidationError, match=r"\[0, 1\]"):
            brier_score([0, 1], [0.5, 1.4])


class TestReliabilityBins:
    def test_bins_are_right_closed_and_sum_to_n(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.05, 0.15, 0.85, 0.95])
        bins = reliability_bins(y, p, n_bins=10)
        # Empty bins are omitted; the four samples land in four distinct bins.
        assert sum(b.count for b in bins) == 4
        assert [b.count for b in bins] == [1, 1, 1, 1]
        assert [b.lower for b in bins] == pytest.approx([0.0, 0.1, 0.8, 0.9])

    def test_probability_one_lands_in_the_last_bin(self):
        bins = reliability_bins([1, 1], [1.0, 1.0], n_bins=5)
        assert len(bins) == 1
        assert bins[0].lower == pytest.approx(0.8)
        assert bins[0].upper == pytest.approx(1.0)

    def test_gap_sign_names_over_confidence(self):
        # bin [0.8, 0.9): predicted 0.85 but only half positive -> over-confident
        bins = reliability_bins([0, 1], [0.85, 0.85], n_bins=10)
        assert bins[0].gap == pytest.approx(0.85 - 0.5)

    def test_too_few_bins_raises(self):
        with pytest.raises(DataValidationError, match="n_bins must be"):
            reliability_bins([0, 1], [0.2, 0.8], n_bins=1)


class TestExpectedAndMaximumError:
    def test_perfectly_calibrated_bins_give_zero_ece(self):
        y = np.array([0, 0, 1, 1])
        assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0

    def test_constant_half_is_calibrated_but_useless(self):
        y = np.array([0, 0, 1, 1])
        assert expected_calibration_error(y, np.full(4, 0.5)) == 0.0

    def test_overconfidence_raises_ece(self):
        y = np.array([0, 0, 1, 1])
        assert expected_calibration_error(y, np.full(4, 0.9)) == pytest.approx(0.4)

    def test_mce_is_the_worst_single_bin(self):
        y = np.array([0, 1, 1, 1])
        p = np.array([0.85, 0.85, 0.85, 0.85])
        # all four in [0.8, 0.9): predicted 0.85, actual 0.75
        assert maximum_calibration_error(y, p, n_bins=10) == pytest.approx(0.10)


def _fit_logistic(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 4))
    logits = X[:, 0] * 1.5 - X[:, 1]
    y = (rng.random(400) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    model = LogisticRegression().fit(X, y)
    return model, X, y


class TestCalibrationReport:
    def test_report_fields_are_consistent(self):
        model, X, y = _fit_logistic()
        report = calibration_report(model, X, y, n_bins=10)
        assert isinstance(report, CalibrationReport)
        assert 0.0 <= report.brier <= 1.0
        assert 0.0 <= report.ece <= 1.0
        assert report.mce >= report.ece
        assert report.n_samples == 400
        assert report.base_rate == pytest.approx(float(y.mean()))
        assert sum(b.count for b in report.bins) == 400

    def test_str_names_the_worst_bin(self):
        model, X, y = _fit_logistic()
        text = str(calibration_report(model, X, y))
        assert "calibration: brier=" in text
        assert "worst bin" in text

    def test_to_frame_has_one_row_per_populated_bin(self):
        model, X, y = _fit_logistic()
        report = calibration_report(model, X, y, n_bins=10)
        frame = report.to_frame()
        assert list(frame.columns) == [
            "lower", "upper", "count", "mean_predicted", "fraction_positive", "gap",
        ]
        assert len(frame) == len(report.bins)

    def test_verdict_thresholds(self):
        model, X, y = _fit_logistic()
        report = calibration_report(model, X, y)
        assert report.verdict in {"well-calibrated", "fair", "miscalibrated"}

    def test_regressor_is_refused(self):
        from sklearn.linear_model import Ridge

        X = np.zeros((5, 2))
        model = Ridge().fit(X, np.zeros(5))
        with pytest.raises(ModelError, match="predict_proba"):
            calibration_report(model, X, np.zeros(5))


class TestRecalibrate:
    def test_recalibration_returns_a_probabilistic_estimator(self):
        model, X, y = _fit_logistic()
        calibrated = recalibrate(model, X, y, method="sigmoid")
        proba = calibrated.predict_proba(X)
        assert proba.shape == (400, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_unknown_method_raises(self):
        model, X, y = _fit_logistic()
        with pytest.raises(DataValidationError, match="isotonic.*sigmoid"):
            recalibrate(model, X, y, method="platt")


def _write_binary_csv(path, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, 4)) + shift
    logits = X[:, 0] * 1.2
    y = (rng.random(300) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    frame["active"] = y
    frame.to_csv(path, index=False)
    return path


class TestCalibrateCli:
    def test_fresh_training_flow_reports_calibration(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        csv = _write_binary_csv(tmp_path / "data.csv")
        code = main(["ml", "calibrate", "--data", str(csv), "--target", "active",
                     "--estimator", "logistic_regression", "--bins", "8"])
        assert code == 0
        out = capsys.readouterr().out
        assert "calibration: brier=" in out
        assert "held-out" in out

    def test_fresh_flow_writes_recalibrated_bundle(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        csv = _write_binary_csv(tmp_path / "data.csv")
        out_model = tmp_path / "cal.joblib"
        code = main(["ml", "calibrate", "--data", str(csv), "--target", "active",
                     "--estimator", "logistic_regression",
                     "--out-model", str(out_model)])
        assert code == 0
        assert out_model.exists()
        assert "held-out split" in capsys.readouterr().out

    def test_bundle_flow_reports_and_writes_table(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        csv = _write_binary_csv(tmp_path / "data.csv")
        model, _X, _y = _fit_logistic()
        bundle = save_model(model, tmp_path / "clf.joblib", name="clf",
                            task="classification", features=[f"f{i}" for i in range(4)],
                            target="active")
        table = tmp_path / "rel.csv"
        code = main(["ml", "calibrate", "--model", str(bundle), "--data", str(csv),
                     "--out", str(table)])
        assert code == 0
        out = capsys.readouterr().out
        assert "calibration: brier=" in out
        assert table.exists()
        assert len(pd.read_csv(table)) > 0

    def test_bundle_flow_rejects_a_regressor(self, tmp_path, capsys):
        from sklearn.linear_model import Ridge

        model = Ridge().fit(np.zeros((5, 2)), np.zeros(5))
        bundle = save_model(model, tmp_path / "reg.joblib", name="reg",
                            task="regression", features=["a", "b"])
        csv = _write_binary_csv(tmp_path / "data.csv")
        code = main(["ml", "calibrate", "--model", str(bundle), "--data", str(csv)])
        assert code == 1
        assert "classification bundle" in capsys.readouterr().err

    def test_missing_target_column_errors(self, tmp_path, capsys):
        csv = _write_binary_csv(tmp_path / "data.csv")
        code = main(["ml", "calibrate", "--data", str(csv), "--target", "nope"])
        assert code == 1
        assert "error" in capsys.readouterr().err

    def test_plot_writes_a_reliability_diagram(self, tmp_path, capsys, monkeypatch):
        import matplotlib

        matplotlib.use("Agg")
        monkeypatch.chdir(tmp_path)
        csv = _write_binary_csv(tmp_path / "data.csv")
        plot = tmp_path / "rel.png"
        code = main(["ml", "calibrate", "--data", str(csv), "--target", "active",
                     "--estimator", "logistic_regression", "--plot", str(plot)])
        assert code == 0
        assert plot.exists()
        assert "reliability diagram" in capsys.readouterr().out
