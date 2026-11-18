"""Tests for leaderboard uncertainty (``pharmaml.ml.comparison`` and
``ml compare --significance``), the shared ``multiplicity`` module, and the
paired Wilcoxon test.

Every verdict is checked against a hand-built fold vector with a known
answer, the multiplicity correction is verified to actually demote a raw
winner, and the CLI is driven through :func:`pharmaml.cli.main` on a real
CSV — no mocks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.comparison import (
    COMPARISON_TESTS,
    ModelComparisonRow,
    model_comparison_test,
)
from pharmaml.ml.cv_results import METRIC_LABELS, CVResult, MetricSummary
from pharmaml.ml.multiplicity import (
    CORRECTION_METHODS,
    correct_p_values,
)
from pharmaml.ml.stats import paired_wilcoxon_test

logging.disable(logging.CRITICAL)


def _result(name: str, folds, *, metric="roc_auc_ovo", task="classification"):
    folds = [float(value) for value in folds]
    lower_is_better = METRIC_LABELS.get(metric, (metric, False))[1]
    summary = MetricSummary(
        name=metric,
        mean=sum(folds) / len(folds),
        std=0.0,
        folds=folds,
        higher_is_better=not lower_is_better,
        lower_is_better=lower_is_better,
    )
    return CVResult(
        model=name,
        task=task,
        strategy="stratified" if task == "classification" else "kfold",
        n_folds=len(folds),
        seed=42,
        metrics={metric: summary},
        primary=metric,
        duration_s=0.0,
    )


# --------------------------------------------------------------------------- #
# shared multiplicity module
# --------------------------------------------------------------------------- #
class TestMultiplicity:
    def test_bonferroni_is_linear_scaling(self):
        p = np.array([0.01, 0.02, 0.5])
        assert np.allclose(correct_p_values(p, "bonferroni"), np.array([0.03, 0.06, 1.0]))

    def test_holm_is_step_down_and_monotone(self):
        # Holm never exceeds Bonferroni and stays non-decreasing when sorted.
        p = np.array([0.001, 0.04, 0.03, 0.5])
        adjusted = correct_p_values(p, "holm")
        assert np.all(adjusted <= correct_p_values(p, "bonferroni") + 1e-12)
        order = np.argsort(p)
        assert np.all(np.diff(adjusted[order]) >= -1e-12)

    def test_fdr_bh_known_vector(self):
        # Benjamini-Hochberg on the classic hand example.
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
        adjusted = correct_p_values(p, "fdr_bh")
        assert np.all(adjusted >= p - 1e-12)  # never decreases a p-value
        assert adjusted[0] == pytest.approx(0.008, abs=1e-9)

    def test_dispatch_matches_method_aliases(self):
        p = np.array([0.01, 0.2, 0.6])
        # Default is fdr_bh; explicit methods agree with the default call.
        assert np.allclose(correct_p_values(p), correct_p_values(p, "fdr_bh"))
        assert np.all(correct_p_values(p, "holm") <= correct_p_values(p, "bonferroni") + 1e-12)

    def test_unknown_correction_rejected(self):
        with pytest.raises(DataValidationError):
            correct_p_values(np.array([0.1, 0.2]), "sidak")

    def test_empty_family_rejected(self):
        with pytest.raises(DataValidationError):
            correct_p_values(np.array([]), "bonferroni")

    def test_out_of_range_p_value_rejected(self):
        with pytest.raises(DataValidationError):
            correct_p_values(np.array([0.5, 1.5]), "bonferroni")

    def test_correction_methods_tuple_is_the_public_menu(self):
        assert set(CORRECTION_METHODS) == {"fdr_bh", "bonferroni", "holm"}


# --------------------------------------------------------------------------- #
# paired Wilcoxon test
# --------------------------------------------------------------------------- #
class TestPairedWilcoxon:
    def test_uniform_direction_gives_rank_biserial_one(self):
        before = [0.90, 0.88, 0.92, 0.85, 0.91, 0.87]
        after = [0.80, 0.79, 0.83, 0.74, 0.82, 0.78]
        result = paired_wilcoxon_test(before, after)
        assert result.significant
        assert result.effect_value == pytest.approx(1.0)

    def test_opposite_direction_flips_the_sign(self):
        before = [0.80, 0.79, 0.83, 0.74, 0.82, 0.78]
        after = [0.90, 0.88, 0.92, 0.85, 0.91, 0.87]
        assert paired_wilcoxon_test(before, after).effect_value == pytest.approx(-1.0)

    def test_all_identical_pairs_are_rejected_not_crashed(self):
        with pytest.raises(DataValidationError):
            paired_wilcoxon_test([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])

    def test_length_mismatch_rejected(self):
        with pytest.raises(DataValidationError):
            paired_wilcoxon_test([0.1, 0.2], [0.1])


# --------------------------------------------------------------------------- #
# model_comparison_test
# --------------------------------------------------------------------------- #
class TestModelComparisonTest:
    def test_clear_loser_is_marked_worse(self):
        leader = _result("rf", [0.90, 0.91, 0.89, 0.92, 0.90, 0.91])
        weak = _result("tree", [0.70, 0.72, 0.68, 0.75, 0.69, 0.71])
        comparison = model_comparison_test([leader, weak])
        assert comparison.leader == "rf"
        row = comparison.rows[0]
        assert row.model == "tree"
        assert row.verdict == "worse"
        assert row.difference < 0.0
        assert row.adjusted_p is not None and row.adjusted_p < comparison.alpha

    def test_close_model_is_on_par(self):
        leader = _result("rf", [0.900, 0.910, 0.890, 0.905, 0.898, 0.902])
        # differences alternate sign, tiny magnitude -> no consistent edge
        close = _result("logi", [0.902, 0.908, 0.892, 0.901, 0.900, 0.899])
        comparison = model_comparison_test([leader, close])
        assert comparison.rows[0].verdict == "on par"

    def test_leader_is_the_best_mean(self):
        a = _result("a", [0.50, 0.52, 0.48, 0.51, 0.49, 0.50])
        b = _result("b", [0.90, 0.86, 0.92, 0.88, 0.91, 0.89])
        comparison = model_comparison_test([a, b])
        assert comparison.leader == "b"

    def test_lower_is_better_metric_orientation(self):
        strong = _result(
            "good", [-0.5, -0.6, -0.55, -0.52, -0.58, -0.51],
            metric="neg_root_mean_squared_error", task="regression",
        )
        weak = _result(
            "bad", [-1.5, -1.7, -1.45, -1.62, -1.55, -1.48],
            metric="neg_root_mean_squared_error", task="regression",
        )
        # Stored fold scores are sklearn-style: neg RMSE, so the less
        # negative (smaller error) model is the leader.
        comparison = model_comparison_test([strong, weak])
        assert comparison.leader == "good"
        assert comparison.rows[0].difference < 0.0  # challenger worse
        assert comparison.rows[0].verdict == "worse"

    def test_leader_selection_on_real_neg_rmse_cv(self):
        # Guards the sklearn convention end to end: on a noisy regression
        # panel the leader must be the model with the smaller RMSE (the
        # larger stored neg-RMSE), never the reverse.
        from pharmaml.ml.cv_results import compare_models_cv

        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"d{i}" for i in range(4)])
        y = 5 + X["d0"] * 2 - X["d1"] + rng.normal(scale=0.2, size=120)
        results, _ = compare_models_cv(
            X, y, "regression", ["random_forest", "linear_regression"],
            n_folds=5, primary="neg_root_mean_squared_error",
        )
        by_model = {
            r.model: -r.metrics["neg_root_mean_squared_error"].mean for r in results
        }
        comparison = model_comparison_test(
            results, primary="neg_root_mean_squared_error"
        )
        best_rmse = min(by_model.values())
        assert by_model[comparison.leader] == pytest.approx(best_rmse)

    def test_correction_demotes_a_raw_winner(self):
        # A hand-picked fold vector where one challenger ("rf") would be
        # called significantly worse than the leader on its raw paired
        # p-value (0.0438 < 0.05) but is demoted to "on par" once the
        # sibling challenger is counted (q = 0.0877).  This is the whole
        # point of the module: the corrected verdict differs from the raw.
        leader = _result("c1", [0.7572, 0.8690, 0.8062, 0.8210, 0.7948, 0.8239])
        challengers = [
            _result("c2", [0.7729, 0.8477, 0.8081, 0.8011, 0.8099, 0.8221]),
            _result("rf", [0.7602, 0.8238, 0.7844, 0.7871, 0.7917, 0.8057]),
        ]
        comparison = model_comparison_test([leader, *challengers], correction="fdr_bh")
        rf = next(row for row in comparison.rows if row.model == "rf")
        assert rf.test is not None
        assert rf.test.p_value < 0.05  # significant on its own ...
        assert rf.adjusted_p >= 0.05  # ... demoted by the family correction
        assert rf.verdict == "on par"

    def test_identical_model_is_untested_with_a_note(self):
        leader = _result("rf", [0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        twin = _result("twin", [0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        comparison = model_comparison_test([leader, twin])
        row = comparison.rows[0]
        assert row.test is None and row.adjusted_p is None
        assert row.verdict == "untested"
        assert any("identical" in note for note in comparison.notes)

    def test_wilcoxon_floor_note_at_five_folds(self):
        leader = _result("rf", [0.90, 0.91, 0.89, 0.92, 0.90])
        weak = _result("tree", [0.70, 0.72, 0.68, 0.75, 0.69])
        comparison = model_comparison_test([leader, weak], test="wilcoxon")
        assert any("floors at" in note for note in comparison.notes)
        # The floor (2 * 2**-5 = 0.0625) sits just above alpha 0.05.
        assert comparison.rows[0].verdict == "on par"

    def test_single_model_has_no_challengers(self):
        comparison = model_comparison_test([_result("rf", [0.9, 0.91, 0.89])])
        assert comparison.rows == ()
        assert "0/0" in comparison.summary()

    def test_nan_folds_are_skipped_pairwise(self):
        leader = _result("rf", [0.90, float("nan"), 0.89, 0.92, 0.90, 0.91])
        weak = _result("tree", [0.70, 0.72, float("nan"), 0.75, 0.69, 0.71])
        comparison = model_comparison_test([leader, weak])
        # 4 fold pairs survive -> still testable, and still a clear loss.
        assert comparison.rows[0].test is not None
        assert comparison.rows[0].verdict == "worse"

    def test_empty_panel_rejected(self):
        with pytest.raises(ModelError):
            model_comparison_test([])

    def test_bad_correction_and_test_rejected(self):
        panel = [_result("rf", [0.9, 0.91, 0.89])]
        with pytest.raises(DataValidationError):
            model_comparison_test(panel, correction="sidak")
        with pytest.raises(DataValidationError):
            model_comparison_test(panel, test="anova")

    def test_bad_alpha_rejected(self):
        panel = [_result("rf", [0.9, 0.91, 0.89])]
        with pytest.raises(DataValidationError):
            model_comparison_test(panel, alpha=1.5)

    def test_missing_primary_metric_rejected(self):
        panel = [_result("rf", [0.9, 0.91, 0.89])]
        with pytest.raises(ModelError):
            model_comparison_test(panel, primary="not_a_metric")

    def test_comparison_tests_tuple_is_the_public_menu(self):
        assert set(COMPARISON_TESTS) == {"paired_t", "wilcoxon"}

    def test_row_line_is_single_line(self):
        row = ModelComparisonRow(
            model="tree",
            mean=0.8,
            leader_mean=0.9,
            difference=-0.1,
            test=None,
            adjusted_p=None,
        )
        assert "\n" not in row.line()

    def test_to_frame_has_the_documented_columns(self):
        panel = [
            _result("rf", [0.92, 0.93, 0.91, 0.94, 0.92, 0.93]),
            _result("tree", [0.80, 0.83, 0.78, 0.85, 0.79, 0.82]),
        ]
        frame = model_comparison_test(panel).to_frame()
        assert list(frame.columns) == [
            "model", "mean", "leader_mean", "difference",
            "p_value", "adjusted_p", "verdict", "note",
        ]
        assert list(frame["model"]) == ["tree"]
        assert frame["verdict"].iloc[0] == "worse"


# --------------------------------------------------------------------------- #
# CLI: ml compare --significance
# --------------------------------------------------------------------------- #
@pytest.fixture()
def clf_csv(tmp_path) -> str:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"d{i}" for i in range(4)])
    frame["Target"] = (frame["d0"] - frame["d1"] > 0).astype(int)
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return str(path)


class TestCompareSignificanceCLI:
    def test_reports_leader_and_verdicts(self, clf_csv, capsys):
        code = main([
            "ml", "compare", "--data", clf_csv, "--target", "Target",
            "--models", "rf", "logistic", "--cv", "5", "--significance",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "leader:" in out
        assert "verdict" in out
        assert "significantly ahead of" in out

    def test_wilcoxon_and_correction_flags(self, clf_csv, capsys):
        code = main([
            "ml", "compare", "--data", clf_csv, "--target", "Target",
            "--models", "rf", "logistic", "--cv", "5", "--significance",
            "--significance-test", "wilcoxon", "--correction", "holm",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "holm" in out and "wilcoxon" in out

    def test_significance_writes_leaderboard(self, clf_csv, tmp_path, capsys):
        out_csv = tmp_path / "board.csv"
        code = main([
            "ml", "compare", "--data", clf_csv, "--target", "Target",
            "--models", "rf", "logistic", "--cv", "5", "--significance",
            "--out", str(out_csv),
        ])
        assert code == 0
        assert out_csv.is_file()
        # the corrected verdicts ride beside the leaderboard
        assert (tmp_path / "verdicts.csv").is_file()
        assert "leaderboard ->" in capsys.readouterr().out

    def test_bad_alpha_is_rejected(self, clf_csv, capsys):
        code = main([
            "ml", "compare", "--data", clf_csv, "--target", "Target",
            "--models", "rf", "logistic", "--cv", "3", "--significance",
            "--alpha", "2.0",
        ])
        assert code == 1
        assert "error:" in capsys.readouterr().err
