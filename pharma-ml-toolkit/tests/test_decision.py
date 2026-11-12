"""Tests for decision-curve analysis (``pharmaml.ml.decision`` and ``ml decide``).

Every number is checked against a hand-computed net benefit, and the CLI is
exercised through :func:`pharmaml.cli.main` on a real ``ml train`` bundle —
no mocks, the same way a shell would run it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.decision import (
    DecisionCurve,
    DecisionPoint,
    decision_curve,
    decision_curve_from_scores,
    net_benefit,
    optimal_threshold,
    standardized_net_benefit,
    treat_all_net_benefit,
)

logging.disable(logging.CRITICAL)


# --------------------------------------------------------------------------- #
# net benefit
# --------------------------------------------------------------------------- #
class TestNetBenefit:
    def test_perfect_separation_has_no_false_positives(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        # 2 TP, 0 FP -> 2/4 = 0.5 net true positives per subject.
        assert net_benefit(y, p, threshold=0.5) == pytest.approx(0.5)

    def test_false_positives_are_charged_at_the_threshold_odds(self):
        y = np.array([0, 1])
        p = np.array([0.9, 0.9])
        # threshold 0.5 -> odds 1: 1 TP and 1 FP over 2 rows -> 0.5 - 0.5 = 0.
        assert net_benefit(y, p, threshold=0.5) == pytest.approx(0.0)

    def test_cost_ratio_scales_the_false_positive_penalty(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.9, 0.9, 0.9, 0.1])  # 1 TP, 2 FP at t=0.5
        base = net_benefit(y, p, threshold=0.5)  # 0.25 - 0.5 = -0.25
        costly = net_benefit(y, p, threshold=0.5, cost_ratio=2.0)
        assert base == pytest.approx(-0.25)
        assert costly == pytest.approx(0.25 - 0.5 * 2.0)

    def test_never_beats_treat_none_when_always_wrong(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.9, 0.9, 0.1, 0.1])  # inverted ranking
        assert net_benefit(y, p, threshold=0.5) < 0.0

    def test_treat_all_matches_net_benefit_of_calling_everyone_positive(self):
        y = np.array([0, 0, 1, 1])
        p = np.ones(4)
        assert treat_all_net_benefit(y, threshold=0.4) == pytest.approx(
            net_benefit(y, p, threshold=0.4)
        )

    def test_standardized_is_net_benefit_over_prevalence(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        assert standardized_net_benefit(y, p, threshold=0.5) == pytest.approx(1.0)

    @pytest.mark.parametrize("threshold", [1.0, 1.5, -0.1])
    def test_threshold_out_of_range_is_rejected(self, threshold):
        with pytest.raises(DataValidationError, match="threshold"):
            net_benefit([0, 1], [0.4, 0.6], threshold=threshold)

    def test_cost_ratio_must_be_positive(self):
        with pytest.raises(DataValidationError, match="cost_ratio"):
            net_benefit([0, 1], [0.4, 0.6], threshold=0.5, cost_ratio=0.0)

    def test_non_binary_target_is_rejected(self):
        with pytest.raises(DataValidationError, match="distinct labels"):
            net_benefit([0, 1, 2], [0.1, 0.5, 0.9], threshold=0.5)

    def test_probability_out_of_range_is_rejected(self):
        with pytest.raises(DataValidationError, match=r"probabilities in \[0, 1\]"):
            net_benefit([0, 1], [0.1, 1.4], threshold=0.5)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(DataValidationError, match="same length"):
            net_benefit([0, 1], [0.1, 0.5, 0.9], threshold=0.5)

    def test_single_class_target_is_rejected(self):
        with pytest.raises(DataValidationError, match="only one class"):
            net_benefit([0, 0], [0.1, 0.2], threshold=0.5)


# --------------------------------------------------------------------------- #
# decision curve
# --------------------------------------------------------------------------- #
class TestDecisionCurve:
    def test_explicit_grid_keeps_every_threshold(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        curve = decision_curve_from_scores(y, p, thresholds=[0.3, 0.5, 0.7])
        assert [pt.threshold for pt in curve.points] == [0.3, 0.5, 0.7]
        assert curve.n_samples == 4
        assert curve.base_rate == pytest.approx(0.5)

    def test_max_net_benefit_and_optimal_threshold(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        curve = decision_curve_from_scores(y, p, thresholds=[0.3, 0.5, 0.7])
        assert curve.max_net_benefit.net_benefit == pytest.approx(0.5)
        assert curve.optimal_threshold == 0.5

    def test_useful_region_reports_where_it_beats_treat_all(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        curve = decision_curve_from_scores(y, p, thresholds=[0.3, 0.5, 0.7])
        low, high = curve.useful_region
        assert low == 0.3 and high == 0.7

    def test_useful_region_is_none_when_inverted(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.9, 0.9, 0.1, 0.1])
        curve = decision_curve_from_scores(y, p, thresholds=[0.3, 0.5, 0.7])
        assert curve.useful_region is None
        assert "never beats 'treat all'" in str(curve)

    def test_beats_default_flags(self):
        good = DecisionPoint(0.5, net_benefit=0.3, standardized_net_benefit=0.6,
                             treat_all_net_benefit=0.0)
        assert good.beats_treat_all and good.beats_treat_none
        bad = DecisionPoint(0.5, net_benefit=-0.1, standardized_net_benefit=-0.2,
                            treat_all_net_benefit=0.0)
        assert not bad.beats_treat_all and not bad.beats_treat_none

    def test_default_grid_is_interior_and_sized(self):
        y = np.array([0] * 10 + [1] * 10)
        p = np.linspace(0.05, 0.95, 20)
        curve = decision_curve_from_scores(y, p, n_points=25)
        assert len(curve.points) == 25
        assert all(0.0 < pt.threshold < 1.0 for pt in curve.points)

    def test_to_frame_round_trips(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        curve = decision_curve_from_scores(y, p, thresholds=[0.3, 0.5])
        frame = curve.to_frame()
        assert list(frame.columns) == [
            "threshold", "net_benefit", "standardized_net_benefit",
            "treat_all_net_benefit",
        ]
        assert len(frame) == 2

    def test_net_benefit_at_uses_nearest_grid_point(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        curve = decision_curve_from_scores(y, p, thresholds=[0.3, 0.5, 0.7])
        assert curve.net_benefit_at(0.45) == pytest.approx(
            curve.net_benefit_at(0.5)
        )

    def test_invalid_thresholds_are_rejected(self):
        with pytest.raises(DataValidationError, match=r"\[0, 1\)"):
            decision_curve_from_scores([0, 1], [0.4, 0.6], thresholds=[1.0])
        with pytest.raises(DataValidationError, match="empty"):
            decision_curve_from_scores([0, 1], [0.4, 0.6], thresholds=[])

    def test_empty_curve_accessors_raise(self):
        empty = DecisionCurve(cost_ratio=1.0, n_samples=0, base_rate=0.0, points=())
        with pytest.raises(ValueError, match="empty decision curve"):
            _ = empty.max_net_benefit

    def test_model_overload_matches_scores(self):
        sklearn = pytest.importorskip("sklearn")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = sklearn.linear_model.LogisticRegression(max_iter=1000).fit(X, y)
        curve = decision_curve(model, X, y, thresholds=[0.3, 0.5, 0.7])
        proba = model.predict_proba(X)[:, 1]
        direct = decision_curve_from_scores(y, proba, thresholds=[0.3, 0.5, 0.7])
        assert curve.max_net_benefit.net_benefit == pytest.approx(
            direct.max_net_benefit.net_benefit
        )

    def test_model_without_predict_proba_is_rejected(self):
        class _Bare:
            def predict(self, X):  # pragma: no cover - never called
                return X

        with pytest.raises(ModelError, match="predict_proba"):
            decision_curve(_Bare(), [[0.0], [1.0]], [0, 1])


# --------------------------------------------------------------------------- #
# optimal threshold
# --------------------------------------------------------------------------- #
class TestOptimalThreshold:
    def test_youden_matches_a_hand_checked_boundary(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        # Any threshold in (0.4, 0.6] gives J = 1; the sweep returns the score.
        assert optimal_threshold(y, p, criterion="youden") == pytest.approx(0.6)

    def test_f1_agrees_with_youden_on_a_separable_case(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        assert optimal_threshold(y, p, criterion="f1") == pytest.approx(0.6)

    def test_cost_criterion_matches_a_brute_force_search(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            n = 80
            y = (rng.random(n) < 0.4).astype(int)
            if y.sum() in (0, n):
                continue
            p = rng.random(n)
            got = optimal_threshold(y, p, criterion="cost", cost_ratio=3.0)
            cands = np.unique(p)
            best = max(
                cands,
                key=lambda c: net_benefit(y, p, threshold=float(c), cost_ratio=3.0),
            )
            assert net_benefit(y, p, threshold=got, cost_ratio=3.0) == pytest.approx(
                net_benefit(y, p, threshold=float(best), cost_ratio=3.0)
            )

    def test_unknown_criterion_is_rejected(self):
        with pytest.raises(DataValidationError, match="criterion"):
            optimal_threshold([0, 1], [0.4, 0.6], criterion="mcc")

    def test_single_class_target_is_rejected(self):
        with pytest.raises(DataValidationError, match="only one class"):
            optimal_threshold([1, 1, 1], [0.2, 0.6, 0.9])


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
class TestPlot:
    def test_plot_returns_a_figure(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from pharmaml.viz import close_figure
        from pharmaml.viz.ml_plots import plot_decision_curve

        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.6, 0.9])
        curve = decision_curve_from_scores(y, p, thresholds=[0.3, 0.5, 0.7])
        figure = plot_decision_curve(curve)
        try:
            assert figure is not None
            assert figure.axes
        finally:
            close_figure(figure)

    def test_plot_rejects_an_empty_curve(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from pharmaml.viz.ml_plots import plot_decision_curve

        empty = DecisionCurve(cost_ratio=1.0, n_samples=0, base_rate=0.0, points=())
        with pytest.raises(DataValidationError, match="no points"):
            plot_decision_curve(empty)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@pytest.fixture()
def clf_csv(tmp_path) -> str:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(160, 4)), columns=[f"d{i}" for i in range(4)])
    frame["Target"] = (frame["d0"] + frame["d1"] > 0).astype(int)
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def reg_csv(tmp_path) -> str:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"d{i}" for i in range(4)])
    frame["pIC50"] = 5 + frame["d0"] * 2 + rng.normal(scale=0.3, size=120)
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _train(csv: str, model: str, tmp_path: Path, target: str | None = None) -> str:
    argv = ["ml", "train", "--data", csv, "--model", model, "--no-tune",
            "--cv", "3", "--out", str(tmp_path / "art")]
    if target is not None:
        argv += ["--target", target]
    assert main(argv) == 0
    bundle = next((tmp_path / "artifacts").glob("*.joblib"))
    return str(bundle)


class TestDecideCli:
    def test_reports_curve_table_and_plot(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        code = main(["ml", "decide", "--model", bundle, "--data", clf_csv,
                     "--target", "Target", "--out", str(tmp_path / "nb.csv"),
                     "--plot", str(tmp_path / "dec.png")])
        assert code == 0
        out = capsys.readouterr().out
        assert "decision:" in out
        assert "table:" in out
        assert (tmp_path / "nb.csv").exists()
        assert (tmp_path / "dec.png").exists()

    def test_single_threshold_reports_one_point(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        code = main(["ml", "decide", "--model", bundle, "--data", clf_csv,
                     "--target", "Target", "--threshold", "0.5",
                     "--out", str(tmp_path / "nb.csv")])
        assert code == 0
        frame = pd.read_csv(tmp_path / "nb.csv")
        assert len(frame) == 1
        assert frame["threshold"].iloc[0] == pytest.approx(0.5)

    def test_criterion_prints_an_optimal_threshold(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        code = main(["ml", "decide", "--model", bundle, "--data", clf_csv,
                     "--target", "Target", "--criterion", "youden"])
        assert code == 0
        assert "optimal:" in capsys.readouterr().out

    def test_both_threshold_flags_conflict(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        code = main(["ml", "decide", "--model", bundle, "--data", clf_csv,
                     "--target", "Target", "--threshold", "0.5",
                     "--thresholds", "0.3,0.6"])
        assert code == 1
        assert "not both" in capsys.readouterr().err

    def test_regression_bundle_is_rejected(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        code = main(["ml", "decide", "--model", bundle, "--data", reg_csv])
        assert code == 1
        assert "classification bundle" in capsys.readouterr().err

    def test_missing_label_column_is_rejected(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        query = pd.read_csv(clf_csv).drop(columns=["Target"])
        query_path = tmp_path / "query.csv"
        query.to_csv(query_path, index=False)
        code = main(["ml", "decide", "--model", bundle, "--data", str(query_path)])
        assert code == 1
        assert "label column" in capsys.readouterr().err
