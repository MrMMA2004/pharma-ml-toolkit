"""Tests for stratified subgroup performance (``pharmaml.ml.subgroups`` and ``ml subgroup``).

Every subgroup number is checked against a hand-computed value, the
worst-group ranking is exercised in both directions, and the CLI is driven
through :func:`pharmaml.cli.main` on a real ``ml train`` bundle — no mocks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.evaluation import evaluate_model
from pharmaml.ml.subgroups import (
    SubgroupMetrics,
    subgroup_gap_scan,
    subgroup_gap_test,
    subgroup_report,
)

logging.disable(logging.CRITICAL)


class _EchoClassifier:
    """Predicts the class stored in the first feature column (slice-safe)."""

    def predict(self, X):
        return np.asarray(X)[:, 0].astype(int)

    def predict_proba(self, X):
        pos = np.asarray(X)[:, 0].astype(float)
        return np.column_stack([1 - pos, pos])


class _EchoRegressor:
    """Predicts the value stored in the first feature column (slice-safe)."""

    def predict(self, X):
        return np.asarray(X)[:, 0].astype(float)


def _clf_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 4 rows in "A" (correct), 2 rows in "B" (both wrong), 1 row in "C" (single class).
    y = np.array([0, 1, 0, 1, 0, 1, 0])
    pred = np.array([0, 1, 0, 1, 1, 0, 0])
    groups = np.array(["A", "A", "A", "A", "B", "B", "C"])
    X = pred.reshape(-1, 1).astype(float)
    return X, y, groups


# --------------------------------------------------------------------------- #
# metrics per subgroup
# --------------------------------------------------------------------------- #
class TestSubgroupMetrics:
    def test_classification_accuracy_is_hand_checked(self):
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=2)
        assert report.group("A").metrics["accuracy"] == pytest.approx(1.0)
        assert report.group("B").metrics["accuracy"] == pytest.approx(0.0)
        assert report.group("A").n == 4
        assert report.group("B").n == 2

    def test_overall_matches_evaluate_model(self):
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=2)
        expected = evaluate_model(_EchoClassifier(), X, y, "classification")
        assert report.overall["accuracy"] == pytest.approx(expected.accuracy)
        assert report.overall["mcc"] == pytest.approx(expected.mcc)
        assert report.overall["n_samples"] == 7

    def test_overall_is_repeated_on_every_group(self):
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=2)
        for group in report.subgroups:
            assert group.overall["accuracy"] == pytest.approx(report.overall["accuracy"])

    def test_regression_rmse_is_hand_checked(self):
        y = np.array([0.0, 0.0, 5.0, 5.0])
        pred = np.zeros(4)  # error 0, 0, 5, 5
        X = pred.reshape(-1, 1)
        batch = np.array(["b1", "b1", "b2", "b2"])
        report = subgroup_report(_EchoRegressor(), X, y, batch,
                                 task="regression", min_group_size=2)
        assert report.group("b1").metrics["rmse"] == pytest.approx(0.0)
        assert report.group("b2").metrics["rmse"] == pytest.approx(5.0)
        assert report.overall["rmse"] == pytest.approx(np.sqrt(12.5))

    def test_dataframe_features_are_sliced_positionally(self):
        X, y, groups = _clf_frame()
        frame = pd.DataFrame({"pred": X[:, 0]})
        report = subgroup_report(_EchoClassifier(), frame, y, groups,
                                 task="classification", min_group_size=2)
        assert report.group("B").metrics["accuracy"] == pytest.approx(0.0)

    def test_undefined_metric_for_a_slice_is_none_not_dropped(self):
        # Group "C" is a single class -> ROC-AUC undefined for that slice.
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=1)
        assert report.group("C").metrics["roc_auc"] is None
        assert "roc_auc" in report.metric_names  # still a column

    def test_small_group_is_flagged_insufficient(self):
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=3)
        assert report.group("A").sufficient is True
        assert report.group("B").sufficient is False
        # The metrics are still computed for transparency.
        assert report.group("B").metrics["accuracy"] == pytest.approx(0.0)

    def test_missing_labels_are_dropped_and_counted(self):
        y = np.array([0, 1, 0, 1])
        pred = np.array([0, 1, 0, 1])
        X = pred.reshape(-1, 1).astype(float)
        groups = np.array(["A", "A", None, "B"])
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=1)
        assert {g.value for g in report.subgroups} == {"A", "B"}
        assert report.overall["rows_dropped"] == 1
        assert report.overall["n_samples"] == 3


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_unknown_task_is_rejected(self):
        X, y, groups = _clf_frame()
        with pytest.raises(DataValidationError, match="task must be"):
            subgroup_report(_EchoClassifier(), X, y, groups, task="ranking")

    def test_negative_min_group_size_is_rejected(self):
        X, y, groups = _clf_frame()
        with pytest.raises(DataValidationError, match="min_group_size"):
            subgroup_report(_EchoClassifier(), X, y, groups, min_group_size=-1)

    def test_x_and_y_length_mismatch_is_rejected(self):
        X, y, groups = _clf_frame()
        with pytest.raises(DataValidationError, match="row"):
            subgroup_report(_EchoClassifier(), X[:-1], y, groups)

    def test_groups_and_y_length_mismatch_is_rejected(self):
        X, y, groups = _clf_frame()
        with pytest.raises(DataValidationError, match="groups has"):
            subgroup_report(_EchoClassifier(), X, y, groups[:-1])

    def test_empty_target_is_rejected(self):
        with pytest.raises(DataValidationError, match="empty"):
            subgroup_report(_EchoClassifier(), np.zeros((0, 1)), np.array([]),
                            np.array([]))

    def test_entirely_missing_groups_is_rejected(self):
        X, y, _ = _clf_frame()
        with pytest.raises(DataValidationError, match="entirely missing"):
            subgroup_report(_EchoClassifier(), X, y, np.array([None] * 7))


# --------------------------------------------------------------------------- #
# worst-group ranking
# --------------------------------------------------------------------------- #
class TestWorstGroup:
    def test_higher_is_better_picks_the_weakest_accuracy(self):
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=2)
        assert report.worst_group("accuracy").value == "B"

    def test_lower_is_better_picks_the_worst_rmse(self):
        y = np.array([0.0, 0.0, 5.0, 5.0])
        X = np.zeros((4, 1))
        batch = np.array(["b1", "b1", "b2", "b2"])
        report = subgroup_report(_EchoRegressor(), X, y, batch,
                                 task="regression", min_group_size=2)
        assert report.worst_group("rmse").value == "b2"

    def test_ties_break_toward_the_smaller_group(self):
        # "big" (n=4) and "small" (n=2) both score accuracy 1.0 -> small wins.
        y = np.array([0, 1, 0, 1, 0, 1])
        X = y.reshape(-1, 1).astype(float)  # perfect predictions everywhere
        groups = np.array(["big", "big", "big", "big", "small", "small"])
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=2)
        assert report.worst_group("accuracy").value == "small"

    def test_unknown_metric_is_rejected(self):
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=2)
        with pytest.raises(DataValidationError, match="not in this report"):
            report.worst_group("nope")

    def test_all_insufficient_returns_a_flagged_group(self):
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=100)
        worst = report.worst_group("accuracy")
        assert worst.sufficient is False

    def test_metric_undefined_everywhere_is_rejected(self):
        # Single-class group on each side -> roc_auc None for every slice.
        y = np.array([0, 0, 1, 1])
        X = np.array([[0.0], [0.0], [1.0], [1.0]])
        groups = np.array(["neg", "neg", "pos", "pos"])
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=1)
        with pytest.raises(DataValidationError, match="numeric"):
            report.worst_group("roc_auc")

    def test_group_accessor_and_missing(self):
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=2)
        assert isinstance(report.group("A"), SubgroupMetrics)
        with pytest.raises(DataValidationError, match="no subgroup"):
            report.group("Z")


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
class TestToFrame:
    def test_frame_has_one_row_per_group_plus_overall(self):
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=2)
        frame = report.to_frame()
        assert list(frame["subgroup"]) == ["A", "B", "C", "overall"]
        assert set(report.metric_names).issubset(frame.columns)
        overall = frame[frame["subgroup"] == "overall"].iloc[0]
        assert overall["n"] == 7

    def test_frame_preserves_none_for_undefined_metric(self):
        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=1)
        frame = report.to_frame()
        c_row = frame[frame["subgroup"] == "C"].iloc[0]
        assert pd.isna(c_row["roc_auc"])


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
class TestPlot:
    def test_plot_returns_a_figure(self):
        from pharmaml.viz import close_figure
        from pharmaml.viz.ml_plots import plot_subgroup_performance

        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=2)
        fig = plot_subgroup_performance(report, "accuracy")
        assert len(fig.axes) == 1
        assert fig.axes[0].get_xlabel() == "accuracy"
        close_figure(fig)

    def test_plot_rejects_an_unknown_metric(self):
        from pharmaml.viz.ml_plots import plot_subgroup_performance

        X, y, groups = _clf_frame()
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=2)
        with pytest.raises(DataValidationError, match="not in this report"):
            plot_subgroup_performance(report, "nope")

    def test_plot_rejects_a_metric_with_no_values(self):
        from pharmaml.viz.ml_plots import plot_subgroup_performance

        y = np.array([0, 0, 1, 1])
        X = np.array([[0.0], [0.0], [1.0], [1.0]])
        groups = np.array(["neg", "neg", "pos", "pos"])
        report = subgroup_report(_EchoClassifier(), X, y, groups,
                                 task="classification", min_group_size=1)
        with pytest.raises(DataValidationError, match="no subgroup carries"):
            plot_subgroup_performance(report, "roc_auc")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@pytest.fixture()
def clf_csv(tmp_path) -> str:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(160, 4)), columns=[f"d{i}" for i in range(4)])
    frame["Target"] = (frame["d0"] + frame["d1"] > 0).astype(int)
    frame["series"] = np.where(frame["d2"] > 0.0, "A", "B")
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def reg_csv(tmp_path) -> str:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"d{i}" for i in range(4)])
    frame["pIC50"] = 5 + frame["d0"] * 2 + rng.normal(scale=0.3, size=120)
    frame["series"] = np.where(frame["d1"] > 0.0, "A", "B")
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def scan_csv(tmp_path) -> str:
    """A three-series classification set where the series really differ.

    Series "A" and "B" are badly predicted (B worst), "C" is only mildly
    off — significant on its own but not after a family correction, which is
    exactly the point of the scan.
    """
    rng = np.random.default_rng(7)
    n = 300
    frame = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"d{i}" for i in range(4)])
    series = np.repeat(["A", "B", "C"], n // 3)
    base = (frame["d0"] > 0).to_numpy().astype(int)
    flip = {"A": 0.0, "B": 0.45, "C": 0.12}
    y = base.copy()
    for i, label in enumerate(series):
        if rng.random() < flip[label]:
            y[i] = 1 - y[i]
    frame["Target"] = y
    frame["series"] = series
    path = tmp_path / "scan.csv"
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


class TestSubgroupGap:
    def test_worse_group_is_flagged_and_delta_is_negative(self):
        X, y, groups = _clf_frame()
        gap = subgroup_gap_test(_EchoClassifier(), X, y, groups, "B",
                                task="classification", min_group_size=2)
        assert gap.direction == "worse"
        assert gap.delta == pytest.approx(0.0 - 1.0)
        assert gap.n == 2 and gap.n_rest == 5

    def test_better_group_is_flagged(self):
        X, y, groups = _clf_frame()
        # Group "A" (perfect) vs the rest (imperfect) -> better.
        gap = subgroup_gap_test(_EchoClassifier(), X, y, groups, "A",
                                task="classification", min_group_size=2)
        assert gap.direction == "better"
        assert gap.delta is not None and gap.delta > 0

    def test_regression_gap_uses_rmse_direction(self):
        y = np.array([0.0, 0.0, 5.0, 5.0])
        X = np.zeros((4, 1))
        batch = np.array(["b1", "b1", "b2", "b2"])
        gap = subgroup_gap_test(_EchoRegressor(), X, y, batch, "b2",
                                task="regression", min_group_size=2)
        # b2 has the larger RMSE -> worse despite a positive delta.
        assert gap.metric == "rmse"
        assert gap.delta is not None and gap.delta > 0
        assert gap.direction == "worse"

    def test_significant_when_the_gap_is_total_and_large(self):
        n = 200
        # "good" rows are all predicted right, "bad" rows all wrong.
        groups = np.array(["good"] * n + ["bad"] * n)
        y = np.tile([0, 1], n)
        pred = np.concatenate([y[:n], 1 - y[n:]])
        X = pred.reshape(-1, 1).astype(float)
        gap = subgroup_gap_test(_EchoClassifier(), X, y, groups, "bad",
                                task="classification", min_group_size=10)
        assert gap.test is not None
        assert gap.test.significant
        assert gap.direction == "worse"

    def test_too_few_rows_skips_the_test(self):
        X, y, groups = _clf_frame()
        gap = subgroup_gap_test(_EchoClassifier(), X, y, groups, "B",
                                task="classification", min_group_size=10)
        assert gap.test is None
        assert "too few" in str(gap)

    def test_nonparametric_uses_mann_whitney(self):
        rng = np.random.default_rng(11)
        n = 60
        groups = np.array(["g"] * n + ["h"] * n)
        y = rng.integers(0, 2, size=2 * n)
        pred = y.copy()
        pred[n:] = 1 - pred[n:]
        X = pred.reshape(-1, 1).astype(float)
        gap = subgroup_gap_test(_EchoClassifier(), X, y, groups, "h",
                                task="classification", min_group_size=10,
                                nonparametric=True)
        assert gap.test is not None
        assert gap.test.test == "Mann-Whitney U"

    def test_unknown_metric_is_rejected(self):
        X, y, groups = _clf_frame()
        with pytest.raises(DataValidationError, match="not defined for task"):
            subgroup_gap_test(_EchoClassifier(), X, y, groups, "B", metric="rmse")

    def test_absent_value_is_rejected(self):
        X, y, groups = _clf_frame()
        with pytest.raises(DataValidationError, match="no row has group"):
            subgroup_gap_test(_EchoClassifier(), X, y, groups, "Z")

    def test_length_mismatch_is_rejected(self):
        X, y, groups = _clf_frame()
        with pytest.raises(DataValidationError, match="row"):
            subgroup_gap_test(_EchoClassifier(), X[:-1], y, groups, "B")


def _multi_group_frame(n: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Six equal groups; five identical near-perfect and one half-wrong.

    Each good group gets six wrong rows (accuracy 0.925) — a gap that clears
    alpha on its own (p ≈ 0.016) but not after a family correction — while
    "bad" is half wrong.  The family-wide verdict is what decides.
    """
    total = 6 * n
    y = np.tile([0, 1], total // 2)
    series = np.repeat(["g0", "g1", "g2", "g3", "g4", "bad"], n)
    correct = np.ones(total, dtype=bool)
    for start in range(0, 5 * n, n):
        correct[start + n - 6:start + n] = False
    correct[5 * n + n // 2:] = False
    pred = np.where(correct, y, 1 - y)
    return pred.reshape(-1, 1).astype(float), y, series


class TestSubgroupGapScan:
    def test_scan_covers_every_subgroup_in_first_seen_order(self):
        X, y, groups = _multi_group_frame()
        gaps = subgroup_gap_scan(_EchoClassifier(), X, y, groups,
                                 min_group_size=10)
        assert [g.value for g in gaps] == ["g0", "g1", "g2", "g3", "g4", "bad"]

    def test_only_the_bad_group_survives_bonferroni(self):
        X, y, groups = _multi_group_frame()
        gaps = subgroup_gap_scan(_EchoClassifier(), X, y, groups,
                                 min_group_size=10, correction="bonferroni")
        flagged = [g.value for g in gaps if g.fdr_significant]
        assert flagged == ["bad"]
        bad = next(g for g in gaps if g.value == "bad")
        assert bad.test is not None
        # The family correction never lowers a p-value below its raw one.
        assert bad.fdr_p_value >= bad.test.p_value

    def test_correction_is_stricter_than_uncorrected(self):
        X, y, groups = _multi_group_frame()
        gaps = subgroup_gap_scan(_EchoClassifier(), X, y, groups,
                                 min_group_size=10, correction="bonferroni")
        raw = [g.value for g in gaps
               if g.test is not None and g.test.significant]
        corrected = [g.value for g in gaps if g.fdr_significant]
        # Each good group clears alpha on its own but not after correction.
        assert "g0" in raw
        assert "g0" not in corrected
        assert set(corrected) <= set(raw)

    def test_bonferroni_holm_and_bh_are_ordered(self):
        X, y, groups = _multi_group_frame()
        kw = {"min_group_size": 10}

        def qs(method: str) -> dict[str, float]:
            gaps = subgroup_gap_scan(_EchoClassifier(), X, y, groups,
                                     correction=method, **kw)
            out: dict[str, float] = {}
            for gap in gaps:
                assert gap.fdr_p_value is not None
                out[gap.value] = gap.fdr_p_value
            return out

        bh, holm, bonf = qs("fdr_bh"), qs("holm"), qs("bonferroni")
        for value in bh:
            assert bonf[value] >= holm[value] >= bh[value] - 1e-9

    def test_small_groups_are_left_out_of_the_family(self):
        # "tiny" has 2 rows: no test, no q-value, but still reported.
        X, y, groups = _clf_frame()
        groups = np.append(groups, ["tiny", "tiny"])
        X = np.vstack([X, np.array([[1.0], [1.0]])])
        y = np.append(y, [1, 1])
        gaps = subgroup_gap_scan(_EchoClassifier(), X, y, groups,
                                 min_group_size=10)
        tiny = next(g for g in gaps if g.value == "tiny")
        assert tiny.test is None
        assert tiny.fdr_p_value is None
        assert tiny.fdr_significant is None

    def test_missing_group_labels_are_dropped(self):
        X, y, groups = _multi_group_frame()
        labels = groups.astype(object).copy()
        labels[0] = None
        gaps = subgroup_gap_scan(_EchoClassifier(), X, y, labels,
                                 min_group_size=10)
        assert [g.value for g in gaps] == ["g0", "g1", "g2", "g3", "g4", "bad"]
        assert sum(g.n for g in gaps) == y.size - 1

    def test_regression_scan_defaults_to_rmse(self):
        rng = np.random.default_rng(3)
        n = 40
        groups = np.repeat(["a", "b"], n)
        y = rng.normal(size=2 * n)
        pred = y + rng.normal(scale=0.1, size=2 * n)
        pred[n:] = y[n:] + 5.0  # 'b' carries the error
        X = pred.reshape(-1, 1).astype(float)
        gaps = subgroup_gap_scan(_EchoRegressor(), X, y, groups,
                                 task="regression", min_group_size=10)
        assert all(g.metric == "rmse" for g in gaps)
        bad = next(g for g in gaps if g.value == "b")
        assert bad.direction == "worse"
        assert bad.fdr_significant

    def test_nan_p_value_is_left_out_of_the_family(self):
        # Both groups are perfect, so the t-test has no variance and returns
        # a NaN p-value — no q-value is invented for it.
        n = 20
        y = np.tile([0, 1], n)
        groups = np.array(["a"] * n + ["b"] * n)
        X = y.reshape(-1, 1).astype(float)
        gaps = subgroup_gap_scan(_EchoClassifier(), X, y, groups,
                                 min_group_size=10)
        assert all(g.fdr_p_value is None for g in gaps)
        assert all(g.fdr_significant is None for g in gaps)

    def test_unknown_correction_is_rejected(self):
        X, y, groups = _clf_frame()
        with pytest.raises(DataValidationError, match="correction must be one of"):
            subgroup_gap_scan(_EchoClassifier(), X, y, groups,
                              min_group_size=2, correction="magic")

    def test_unknown_metric_is_rejected(self):
        X, y, groups = _clf_frame()
        with pytest.raises(DataValidationError, match="not defined for task"):
            subgroup_gap_scan(_EchoClassifier(), X, y, groups, metric="rmse")

    def test_length_mismatch_is_rejected(self):
        X, y, groups = _clf_frame()
        with pytest.raises(DataValidationError, match="row"):
            subgroup_gap_scan(_EchoClassifier(), X[:-1], y, groups)


class TestSubgroupCli:
    def test_reports_table_worst_group_and_plot(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        code = main(["ml", "subgroup", "--model", bundle, "--data", clf_csv,
                     "--by", "series", "--metric", "accuracy",
                     "--out", str(tmp_path / "sub.csv"),
                     "--plot", str(tmp_path / "sub.png")])
        assert code == 0
        out = capsys.readouterr().out
        assert "worst:" in out
        assert "table:" in out
        assert (tmp_path / "sub.csv").exists()
        assert (tmp_path / "sub.png").exists()
        frame = pd.read_csv(tmp_path / "sub.csv")
        assert "overall" in set(frame["subgroup"])

    def test_default_metric_is_accuracy_for_classification(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        code = main(["ml", "subgroup", "--model", bundle, "--data", clf_csv,
                     "--by", "series"])
        assert code == 0
        assert "by accuracy" in capsys.readouterr().out

    def test_regression_bundle_uses_rmse(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        code = main(["ml", "subgroup", "--model", bundle, "--data", reg_csv,
                     "--by", "series", "--out", str(tmp_path / "sub.csv")])
        assert code == 0
        out = capsys.readouterr().out
        assert "by rmse" in out
        frame = pd.read_csv(tmp_path / "sub.csv")
        assert "rmse" in frame.columns

    def test_metric_absent_from_task_is_rejected(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        code = main(["ml", "subgroup", "--model", bundle, "--data", clf_csv,
                     "--by", "series", "--metric", "rmse"])
        assert code == 1
        assert "not in this report" in capsys.readouterr().err

    def test_unknown_by_column_is_rejected(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        code = main(["ml", "subgroup", "--model", bundle, "--data", clf_csv,
                     "--by", "nope"])
        assert code == 1
        assert "is not a column" in capsys.readouterr().err

    def test_missing_label_column_is_rejected(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        query = pd.read_csv(clf_csv).drop(columns=["Target"])
        query_path = tmp_path / "query.csv"
        query.to_csv(query_path, index=False)
        code = main(["ml", "subgroup", "--model", bundle, "--data", str(query_path),
                     "--by", "series"])
        assert code == 1
        assert "label column" in capsys.readouterr().err

    def test_scan_reports_corrected_p_values(self, scan_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(scan_csv, "logistic_regression", tmp_path, target="Target")
        code = main(["ml", "subgroup", "--model", bundle, "--data", scan_csv,
                     "--by", "series", "--scan", "--correction", "bonferroni"])
        assert code == 0
        out = capsys.readouterr().out
        assert "scan:" in out
        assert "bonferroni correction" in out
        assert "FDR q=" in out
        assert "significant after correction:" in out

    def test_scan_defaults_to_bh(self, scan_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(scan_csv, "logistic_regression", tmp_path, target="Target")
        code = main(["ml", "subgroup", "--model", bundle, "--data", scan_csv,
                     "--by", "series", "--scan"])
        assert code == 0
        assert "fdr_bh correction" in capsys.readouterr().out

    def test_scan_rejects_unknown_correction(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic_regression", tmp_path, target="Target")
        with pytest.raises(SystemExit):
            main(["ml", "subgroup", "--model", bundle, "--data", clf_csv,
                  "--by", "series", "--scan", "--correction", "magic"])
