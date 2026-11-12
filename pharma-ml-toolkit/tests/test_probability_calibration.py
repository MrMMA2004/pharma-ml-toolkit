"""Independent binary recalibration, leakage guards, and held-out scoring."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.probability_calibration import (
    binary_probability_calibration_from_files,
    binary_probability_calibration_report,
)


def partitions():
    frames = []
    rng = np.random.default_rng(123)
    for name, count in (("train", 60), ("calibration", 40), ("test", 30)):
        x = rng.normal(size=count)
        labels = np.where(np.arange(count) % 2, "01", "1")
        frames.append(pd.DataFrame({"id": [f"{name}_{i}" for i in range(count)], "x": x,
                                    "y": labels}))
    return frames


def run(frames, **kwargs):
    return binary_probability_calibration_report(*frames, features=["x"], target="y", key="id",
                                                  positive_label=kwargs.get("positive_label", "01"))


def test_baseline_and_calibration_match_manual_fit_and_scores():
    from sklearn.pipeline import Pipeline

    from pharmaml.ml.preprocessing import build_preprocessor

    train, calibration, test = partitions()
    report = run([train, calibration, test])
    base = Pipeline([("prep", build_preprocessor()),
                     ("estimator", LogisticRegression(solver="lbfgs", C=1, max_iter=2000, tol=1e-4,
                                                      random_state=0))])
    base.fit(train[["x"]], train.y)
    # Positive class "01" is sorted first, so negate sklearn's decision scores.
    calibrator = LogisticRegression(solver="lbfgs", C=1, max_iter=2000, tol=1e-4, random_state=0)
    calibrator.fit(-base.decision_function(calibration[["x"]]).reshape(-1, 1), (calibration.y == "01").astype(int))
    p_base = base.predict_proba(test[["x"]])[:, 0]
    p_cal = calibrator.predict_proba(-base.decision_function(test[["x"]]).reshape(-1, 1))[:, 1]
    actual = (test.y == "01").astype(int)
    assert report["uncalibrated"]["binary_brier"] == pytest.approx(brier_score_loss(actual, p_base))
    assert report["recalibrated"]["binary_brier"] == pytest.approx(brier_score_loss(actual, p_cal))
    assert report["recalibrated"]["log_loss"] == pytest.approx(log_loss(actual, p_cal, labels=[0, 1]))
    assert report["recalibrated"]["reliability"]["multiclass_brier_sum"] == pytest.approx(
        2 * report["recalibrated"]["binary_brier"])
    assert report["partition_class_counts"]["test"] == {"01": 15, "1": 15}
    assert "not conformal" in report["limitations"]


def test_second_sorted_positive_class_works():
    first = run(partitions(), positive_label="01")
    second = run(partitions(), positive_label="1")
    assert second["uncalibrated"]["binary_brier"] == pytest.approx(first["uncalibrated"]["binary_brier"])
    assert second["recalibrated"]["binary_brier"] == pytest.approx(first["recalibrated"]["binary_brier"])


@pytest.mark.parametrize("case", ["overlap", "one_train_class", "three_train_classes", "one_cal_class",
                                  "unknown_test_class", "bad_positive", "missing_id", "inf_feature"])
def test_invalid_inputs_rejected_before_any_fit(case, monkeypatch):
    frames = partitions()
    train, calibration, test = frames
    positive = "01"
    if case == "overlap":
        test.loc[0, "id"] = train.loc[0, "id"]
    elif case == "one_train_class":
        train.y = "01"
    elif case == "three_train_classes":
        train.loc[0, "y"] = "other"
    elif case == "one_cal_class":
        calibration.y = "01"
    elif case == "unknown_test_class":
        test.loc[0, "y"] = "other"
    elif case == "bad_positive":
        positive = "not-a-class"
    elif case == "missing_id":
        test.loc[0, "id"] = " "
    elif case == "inf_feature":
        calibration.loc[0, "x"] = np.inf
    fits = []
    monkeypatch.setattr(LogisticRegression, "fit", lambda *args, **kwargs: fits.append(True))
    with pytest.raises(DataValidationError):
        run(frames, positive_label=positive)
    assert not fits


def test_test_data_never_used_for_fitting(monkeypatch):
    train, calibration, test = partitions()
    test.x += 1000
    from sklearn.pipeline import Pipeline

    real_pipeline_fit, real_logistic_fit = Pipeline.fit, LogisticRegression.fit
    observed = []

    def pipeline_spy(self, X, y, *args, **kwargs):
        assert X.index.equals(train.index) and np.array_equal(X.x, train.x)
        observed.append("base")
        return real_pipeline_fit(self, X, y, *args, **kwargs)

    def logistic_spy(self, X, y, *args, **kwargs):
        observed.append(len(y))
        return real_logistic_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", pipeline_spy)
    monkeypatch.setattr(LogisticRegression, "fit", logistic_spy)
    run([train, calibration, test])
    assert observed == ["base", len(train), len(calibration)]


def test_cli_files_hash_and_exact_class_labels(tmp_path, capsys):
    paths = []
    for n, frame in zip(("train", "cal", "test"), partitions(), strict=True):
        path = tmp_path / (n + ".csv")
        frame.to_csv(path, index=False)
        paths.append(path)
    result = binary_probability_calibration_from_files(*paths, features=["x"], target="y", key="id",
                                                       positive_label="01")
    assert result["positive_label"] == "01" and result["negative_label"] == "1"
    assert len(result["provenance"]["inputs"]["probability_calibration"]["sha256"]) == 64
    assert main(["ml", "calibrate-binary", "--train", str(paths[0]), "--calibration", str(paths[1]),
                 "--test", str(paths[2]), "--features", "x", "--target", "y", "--key", "id",
                 "--positive-label", "01"]) == 0
    assert json.loads(capsys.readouterr().out)["recalibrated"]["binary_brier"] == result["recalibrated"]["binary_brier"]


def test_changed_input_refused(tmp_path, monkeypatch):
    from pharmaml.utils import identifier_table

    paths = []
    for index, frame in enumerate(partitions()):
        path = tmp_path / f"{index}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    original = identifier_table.read_identifier_table

    def changed(path, delim):
        frame = original(path, delim)
        path.write_text(path.read_text() + "\n")
        return frame

    monkeypatch.setattr(identifier_table, "read_identifier_table", changed)
    with pytest.raises(DataValidationError, match="changed while reading"):
        binary_probability_calibration_from_files(*paths, features=["x"], target="y", key="id", positive_label="01")


@pytest.mark.parametrize("failed_fit", [1, 2])
def test_nonconvergence_fails_report(monkeypatch, failed_fit):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    original = LogisticRegression.fit
    fits = []

    def warn_on_fit(self, X, y, *args, **kwargs):
        fits.append(1)
        if len(fits) == failed_fit:
            warnings.warn("not converged", ConvergenceWarning)
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", warn_on_fit)
    with pytest.raises(DataValidationError, match="did not converge"):
        run(partitions())
    assert len(fits) == failed_fit


def test_test_single_class_is_reported_not_used_for_training():
    frames = partitions()
    frames[2]["y"] = "01"
    result = run(frames)
    assert result["partition_class_counts"]["test"] == {"01": 30, "1": 0}
    assert np.isfinite(result["recalibrated"]["log_loss"])


def test_both_calibration_test_overlap_is_refused():
    frames = partitions()
    frames[2].loc[0, "id"] = frames[1].loc[0, "id"]
    with pytest.raises(DataValidationError, match="overlap"):
        run(frames)
