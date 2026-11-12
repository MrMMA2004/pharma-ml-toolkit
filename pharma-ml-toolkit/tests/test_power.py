"""Fold-count power analysis (`ml compare --significance --power`).

The paired test answers "is the lead real?"; this module answers the
follow-up an honest reader always has — "if not, was the panel even big
enough to tell?".  These tests pin the exact power arithmetic against
the classic textbook values and the round-trips between the three
functions.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.comparison import model_comparison_test
from pharmaml.ml.cv_results import CVResult, MetricSummary
from pharmaml.ml.power import (
    MAX_FOLDS,
    PowerReport,
    folds_for_power,
    minimum_detectable_effect,
    paired_power,
    power_report,
)


# --------------------------------------------------------------------------- #
# paired_power
# --------------------------------------------------------------------------- #
class TestPairedPower:
    def test_matches_the_classic_table(self):
        # Cohen's d = 1.0 at n = 10 folds -> ~0.80 power (the textbook row).
        assert round(paired_power(1.0, 10), 3) == 0.803
        # d = 0.5 needs the larger panel (n = 34) for 80% power.
        assert round(paired_power(0.5, 34), 3) == 0.808

    def test_zero_effect_gives_the_false_positive_rate(self):
        # With no true effect the "power" is exactly the two-sided alpha.
        assert paired_power(0.0, 10, alpha=0.05) == pytest.approx(0.05, abs=1e-9)
        assert paired_power(0.0, 7, alpha=0.10) == pytest.approx(0.10, abs=1e-9)

    def test_power_is_monotone_in_folds_and_effect(self):
        rising_folds = [paired_power(0.6, n) for n in range(2, 40)]
        assert all(b > a for a, b in itertools.pairwise(rising_folds))
        rising_effect = [paired_power(d, 12) for d in np.linspace(0.1, 1.5, 15)]
        assert all(b > a for a, b in itertools.pairwise(rising_effect))

    def test_sign_of_the_effect_does_not_matter(self):
        assert paired_power(-0.7, 15) == paired_power(0.7, 15)

    def test_large_effect_saturates_near_one(self):
        assert paired_power(5.0, 2000) == pytest.approx(1.0, abs=1e-9)

    def test_validation(self):
        with pytest.raises(DataValidationError):
            paired_power(0.5, 1)
        with pytest.raises(DataValidationError):
            paired_power(0.5, 5, alpha=0.0)
        with pytest.raises(DataValidationError):
            paired_power(0.5, 5, alpha=1.5)


# --------------------------------------------------------------------------- #
# minimum_detectable_effect
# --------------------------------------------------------------------------- #
class TestMinimumDetectableEffect:
    def test_round_trips_through_paired_power(self):
        for n in (3, 5, 8, 20, 50):
            mde = minimum_detectable_effect(n)
            assert paired_power(mde, n) == pytest.approx(0.8, abs=1e-6)

    def test_shrinks_as_folds_grow(self):
        mdes = [minimum_detectable_effect(n) for n in (3, 5, 10, 20, 100)]
        assert all(b < a for a, b in itertools.pairwise(mdes))

    def test_five_folds_can_only_see_a_large_effect(self):
        assert round(minimum_detectable_effect(5), 3) == 1.682

    def test_higher_target_power_demands_a_larger_effect(self):
        assert minimum_detectable_effect(10, target_power=0.9) > minimum_detectable_effect(
            10, target_power=0.8
        )

    def test_validation(self):
        with pytest.raises(DataValidationError):
            minimum_detectable_effect(1)
        with pytest.raises(DataValidationError):
            minimum_detectable_effect(10, target_power=1.0)


# --------------------------------------------------------------------------- #
# folds_for_power
# --------------------------------------------------------------------------- #
class TestFoldsForPower:
    def test_matches_the_classic_requirement(self):
        assert folds_for_power(0.5) == 34
        assert folds_for_power(1.0) == 10

    def test_returned_count_actually_reaches_the_target(self):
        for d in (0.3, 0.5, 0.8, 1.2):
            n = folds_for_power(d)
            assert n is not None
            assert paired_power(d, n) >= 0.8
            assert paired_power(d, n - 1) < 0.8

    def test_bigger_effect_needs_no_more_folds(self):
        counts = [folds_for_power(d) for d in (0.2, 0.4, 0.6, 0.9, 1.5)]
        assert all(b <= a for a, b in itertools.pairwise(counts))

    def test_non_positive_effect_is_none(self):
        assert folds_for_power(0.0) is None
        assert folds_for_power(-0.5) is None

    def test_vanishing_effect_cannot_be_powered_by_folds(self):
        assert folds_for_power(1e-4) is None

    def test_never_exceeds_the_cap(self):
        for d in (0.05, 0.1, 0.2):
            n = folds_for_power(d)
            if n is not None:
                assert n <= MAX_FOLDS

    def test_validation(self):
        with pytest.raises(DataValidationError):
            folds_for_power(0.5, alpha=2.0)
        with pytest.raises(DataValidationError):
            folds_for_power(0.5, target_power=0.0)


# --------------------------------------------------------------------------- #
# power_report
# --------------------------------------------------------------------------- #
class TestPowerReport:
    def test_without_an_observed_effect_only_the_mde_is_known(self):
        report = power_report(5)
        assert isinstance(report, PowerReport)
        assert report.mde == pytest.approx(minimum_detectable_effect(5))
        assert report.observed_effect is None
        assert report.observed_power is None
        assert report.sufficient is None

    def test_sufficient_flags_an_effect_above_the_floor(self):
        report = power_report(5, observed_effect=1.7)
        assert report.sufficient is True
        assert report.observed_power >= 0.8

    def test_insufficient_flags_an_effect_below_the_floor(self):
        report = power_report(5, observed_effect=0.3)
        assert report.sufficient is False
        assert report.observed_power < 0.8

    def test_line_is_single_line_and_names_the_mde(self):
        line = power_report(5, observed_effect=0.3).line()
        assert "\n" not in line
        assert "MDE" in line and "underpowered" in line


# --------------------------------------------------------------------------- #
# ModelComparison.power — the CLI-facing integration
# --------------------------------------------------------------------------- #
def _result(name: str, folds: list[float]) -> CVResult:
    summary = MetricSummary(
        name="r2",
        mean=sum(folds) / len(folds),
        std=0.0,
        folds=list(folds),
        higher_is_better=True,
        lower_is_better=False,
    )
    return CVResult(
        model=name,
        task="regression",
        strategy="kfold",
        n_folds=len(folds),
        seed=42,
        metrics={"r2": summary},
        primary="r2",
        duration_s=0.0,
    )


class TestComparisonPower:
    def test_observed_effect_is_the_strongest_tested_challenger(self):
        panel = [
            _result("gbr", [0.90, 0.91, 0.89, 0.92, 0.90, 0.91]),
            _result("ridge", [0.83, 0.88, 0.82, 0.90, 0.85, 0.89]),
            _result("linear", [0.50, 0.55, 0.48, 0.60, 0.52, 0.54]),
        ]
        comparison = model_comparison_test(panel)
        report = comparison.power()
        tested = [abs(r.test.effect_value) for r in comparison.rows if r.test is not None]
        assert report.observed_effect == pytest.approx(max(tested))
        assert report.n_folds == 6
        assert report.sufficient is True

    def test_underpowered_panel_reads_as_insufficient(self):
        # A tiny, noisy gap: the observed effect is far below a 3-fold floor.
        panel = [
            _result("gbr", [0.800, 0.810, 0.790]),
            _result("ridge", [0.799, 0.808, 0.791]),
        ]
        comparison = model_comparison_test(panel)
        report = comparison.power()
        assert report.n_folds == 3
        assert report.sufficient is False

    def test_no_tested_challenger_leaves_the_effect_unset(self):
        panel = [
            _result("a", [0.5, 0.5, 0.5]),
            _result("b", [0.5, 0.5, 0.5]),  # identical -> untestable
        ]
        report = model_comparison_test(panel).power()
        assert report.observed_effect is None
        assert report.sufficient is None


# --------------------------------------------------------------------------- #
# CLI: ml compare --significance --power
# --------------------------------------------------------------------------- #
@pytest.fixture()
def clf_csv(tmp_path) -> str:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"d{i}" for i in range(4)])
    frame["Target"] = (frame["d0"] - frame["d1"] > 0).astype(int)
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return str(path)


class TestCLIPowerFlag:
    def test_power_prints_the_mde_line(self, clf_csv, capsys, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        code = main(
            [
                "ml",
                "compare",
                "--data",
                clf_csv,
                "--models",
                "rf",
                "logistic",
                "--cv",
                "5",
                "--significance",
                "--power",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "power:" in out
        assert "MDE" in out and "folds" in out

    def test_power_names_the_underpowered_reading(self, clf_csv, capsys, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        # A tiny separation with five folds cannot be resolved: the note must
        # say so rather than let "on par" read as "equal".
        code = main(
            [
                "ml",
                "compare",
                "--data",
                clf_csv,
                "--models",
                "rf",
                "logistic",
                "--cv",
                "5",
                "--significance",
                "--power",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "underpowered" in out or "adequate" in out

    def test_target_power_moves_the_floor(self, clf_csv, capsys, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        main(
            [
                "ml",
                "compare",
                "--data",
                clf_csv,
                "--models",
                "rf",
                "logistic",
                "--cv",
                "5",
                "--significance",
                "--power",
                "--target-power",
                "0.8",
            ]
        )
        floor_80 = capsys.readouterr().out
        main(
            [
                "ml",
                "compare",
                "--data",
                clf_csv,
                "--models",
                "rf",
                "logistic",
                "--cv",
                "5",
                "--significance",
                "--power",
                "--target-power",
                "0.95",
            ]
        )
        floor_95 = capsys.readouterr().out
        assert "MDE" in floor_80 and "MDE" in floor_95
        assert "95% power" in floor_95 and "80% power" in floor_80

    def test_wilcoxon_power_points_back_at_paired_t(self, clf_csv, capsys, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        code = main(
            [
                "ml",
                "compare",
                "--data",
                clf_csv,
                "--models",
                "rf",
                "logistic",
                "--cv",
                "5",
                "--significance",
                "--power",
                "--significance-test",
                "wilcoxon",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "power figures assume the paired t-test" in out
