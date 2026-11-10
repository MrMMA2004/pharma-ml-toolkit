"""Unit tests for the section 24.3/24.4 SA surface on GenerationResult (r265).

The raw Ertl SA score ([1, 10], lower = easier) is kept strictly
distinct from the SAScoreObjective's [0, 1] optimization reward: the
results-table, sorting, filtering and distribution pieces are pinned
against hand-checked values (aspirin 1.58, CCO 1.98, caffeine 2.298),
the nothing-discarded filter contract (kept + rejected == total) is
verified, and the reward-column collision path is exercised both on a
hand-built result and on a real seeded GA run.
"""

from __future__ import annotations

import math

import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.generative import (
    GenerationResult,
    GeneticOptimizer,
    MultiObjective,
    QedObjective,
    SAFilteredCandidates,
    SAScoreObjective,
)
from pharmaml.viz import close_figure, plot_sa_distribution

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"          # raw SA 1.58
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"    # raw SA 2.298
ETHANOL = "CCO"                            # raw SA 1.98
ETHYLAMINE = "CCN"                          # raw SA 2.174

BAD = "not-a-smiles"


def _result() -> GenerationResult:
    return GenerationResult(
        smiles=[CAFFEINE, ETHANOL, BAD],
        scores=[0.9, 0.5, 0.0],
        breakdown=[{"qed": 0.7}, {"qed": 0.4}, {}],
        metadata={"method": "genetic"},
    )


# -- sa_frame -----------------------------------------------------------


class TestSaFrame:
    def test_default_columns_and_rank_order(self):
        frame = _result().sa_frame()
        assert list(frame.columns) == [
            "rank", "smiles", "score", "qed", "sa_score", "sa_valid",
        ]
        assert list(frame["smiles"]) == [CAFFEINE, ETHANOL, BAD]
        assert list(frame["rank"]) == [1, 2, 3]

    def test_raw_scores_are_the_ertl_values(self):
        frame = _result().sa_frame()
        by_smiles = dict(zip(frame["smiles"], frame["sa_score"]))
        assert by_smiles[CAFFEINE] == pytest.approx(2.298, abs=0.02)
        assert by_smiles[ETHANOL] == pytest.approx(1.98, abs=0.02)

    def test_invalid_row_is_nan_not_a_fake_number(self):
        frame = _result().sa_frame()
        assert math.isnan(frame.loc[2, "sa_score"])
        assert not bool(frame.loc[2, "sa_valid"])

    @pytest.mark.parametrize("sort,expected,expected_rank", [
        ("easiest", [ETHANOL, CAFFEINE, BAD], [2, 1, 3]),
        ("hardest", [CAFFEINE, ETHANOL, BAD], [1, 2, 3]),
    ])
    def test_sorting_by_raw_sa_with_invalid_last(self, sort, expected, expected_rank):
        frame = _result().sa_frame(sort=sort)
        assert list(frame["smiles"]) == expected
        # the rank column keeps the objective rank (provenance, not order)
        assert list(frame["rank"]) == expected_rank

    def test_unknown_sort_key_rejected(self):
        with pytest.raises(DataValidationError, match="sort must be one of"):
            _result().sa_frame(sort="nope")

    def test_reward_breakdown_column_is_renamed_not_clobbered(self):
        run = GenerationResult(
            smiles=[ETHANOL], scores=[0.8],
            breakdown=[{"qed": 0.6, "sa_score": 1.0}],  # the 0-1 reward
        )
        frame = run.sa_frame()
        assert list(frame.columns) == [
            "rank", "smiles", "score", "qed", "sa_score_reward",
            "sa_score", "sa_valid",
        ]
        assert frame.loc[0, "sa_score_reward"] == pytest.approx(1.0)
        assert frame.loc[0, "sa_score"] == pytest.approx(1.98, abs=0.02)

    def test_empty_result_stays_typed(self):
        frame = GenerationResult().sa_frame()
        assert len(frame) == 0
        assert "sa_score" in frame.columns and "sa_valid" in frame.columns


# -- filter_by_sa -------------------------------------------------------


class TestFilterBySa:
    def test_both_sides_preserved(self):
        picked = _result().filter_by_sa(max_sa=2.0)
        assert isinstance(picked, SAFilteredCandidates)
        assert picked.kept.smiles == [ETHANOL]
        assert picked.n_kept == 1 and picked.n_rejected == 2
        assert picked.n_total == 3  # kept + rejected == every candidate

    def test_rejected_table_carries_reasons_and_original_score(self):
        rejected = _result().filter_by_sa(max_sa=2.0).rejected
        assert list(rejected.columns) == [
            "rank", "smiles", "score", "sa_score", "reason",
        ]
        assert rejected.loc[0, "smiles"] == CAFFEINE
        assert rejected.loc[0, "score"] == pytest.approx(0.9)
        assert rejected.loc[0, "reason"] == "sa_score 2.298 above max_sa 2.0"
        assert rejected.loc[1, "reason"].startswith("invalid SMILES")

    def test_kept_result_keeps_breakdown_and_records_provenance(self):
        picked = _result().filter_by_sa(max_sa=2.0)
        assert picked.kept.breakdown == [{"qed": 0.4}]
        assert picked.kept.metadata["sa_filter"] == {
            "min_sa": None, "max_sa": 2.0, "kept": 1, "rejected": 2,
        }
        assert picked.kept.metadata["method"] == "genetic"

    def test_min_bound_rejects_below_with_reason(self):
        picked = _result().filter_by_sa(min_sa=2.1)
        assert picked.kept.smiles == [CAFFEINE]
        assert picked.rejected.loc[0, "reason"] == (
            "sa_score 1.98 below min_sa 2.1"
        )

    def test_inclusive_bounds(self):
        picked = _result().filter_by_sa(min_sa=1.98, max_sa=2.298)
        # both boundary molecules survive: bounds are inclusive
        assert picked.kept.smiles == [CAFFEINE, ETHANOL]
        assert picked.n_rejected == 1  # only the invalid row

    def test_no_constraints_is_a_no_op(self):
        result = _result()
        picked = result.filter_by_sa()
        assert picked.kept is result
        assert picked.n_kept == 3 and picked.n_rejected == 0
        assert list(picked.rejected.columns) == [
            "rank", "smiles", "score", "sa_score", "reason",
        ]

    def test_invalid_rows_survive_when_no_constraint(self):
        # without a constraint nothing is filtered -- not even the invalid
        picked = GenerationResult(smiles=[BAD], scores=[0.0]).filter_by_sa()
        assert picked.kept.smiles == [BAD]

    @pytest.mark.parametrize("min_sa,max_sa", [(3.0, 2.0), (2.9, 2.8)])
    def test_empty_or_inverted_range_rejected(self, min_sa, max_sa):
        with pytest.raises(DataValidationError, match="must not exceed"):
            _result().filter_by_sa(min_sa=min_sa, max_sa=max_sa)

    @pytest.mark.parametrize("bound", [float("nan"), float("inf"), "4", True])
    def test_non_finite_or_non_numeric_bounds_rejected(self, bound):
        with pytest.raises(DataValidationError, match="finite number"):
            _result().filter_by_sa(max_sa=bound)

    def test_empty_result_filter(self):
        picked = GenerationResult().filter_by_sa(max_sa=4.0)
        assert picked.n_total == 0 and picked.kept.is_empty()


# -- sa_summary ---------------------------------------------------------


class TestSaSummary:
    def test_stats_over_valid_subset_only(self):
        summary = _result().sa_summary()
        assert summary["n_total"] == 3
        assert summary["n_scored"] == 2
        assert summary["n_invalid"] == 1
        assert summary["min"] == pytest.approx(1.98, abs=0.02)
        assert summary["max"] == pytest.approx(2.298, abs=0.02)
        assert summary["mean"] == pytest.approx((1.98 + 2.298) / 2, abs=0.02)
        assert summary["n_druglike_band"] == 2

    def test_direction_and_limitations_always_ride(self):
        summary = GenerationResult(smiles=[ETHANOL], scores=[0.5]).sa_summary()
        assert summary["direction"] == (
            "lower = estimated easier to synthesise"
        )
        assert "computational estimate" in summary["limitations"]
        assert "Ertl-Schuffenhauer" in summary["method"]

    def test_empty_summary_has_no_fake_statistics(self):
        summary = GenerationResult().sa_summary()
        assert summary["n_total"] == 0
        for key in ("min", "max", "mean", "n_druglike_band"):
            assert key not in summary


# -- plot_sa_distribution -----------------------------------------------


class TestPlotSaDistribution:
    def test_figure_and_payload(self):
        payload = plot_sa_distribution(_result())
        assert payload["n"] == 2
        assert payload["n_invalid"] == 1
        assert payload["direction"] == "lower = estimated easier to synthesise"
        assert "Ertl-Schuffenhauer" in payload["method"]
        fig = payload["figure"]
        ax = fig.axes[0]
        assert ax.get_xlabel() == "SA score (lower = estimated easier to synthesise)"
        assert "SA score distribution" in ax.get_title(loc="left")
        legend = {t.get_text() for t in ax.get_legend().get_texts()}
        assert "molecules" in legend
        assert "conventional drug-like band (1-4)" in legend
        close_figure(fig)

    def test_path_save_keeps_figure_open_for_caller(self, tmp_path):
        target = tmp_path / "sa_dist.png"
        payload = plot_sa_distribution(_result(), path=target)
        assert target.exists() and target.stat().st_size > 0
        assert len(payload["figure"].axes) == 1
        close_figure(payload["figure"])

    def test_too_few_valid_molecules_rejected(self):
        with pytest.raises(DataValidationError, match="fewer than two valid"):
            plot_sa_distribution(
                GenerationResult(smiles=[ETHANOL, BAD], scores=[0.5, 0.0])
            )

    def test_empty_result_rejected(self):
        with pytest.raises(DataValidationError, match="fewer than two valid"):
            plot_sa_distribution(GenerationResult())


# -- end-to-end: a real seeded GA run with the SA objective -------------


class TestGaRunWithSaObjective:
    def test_sa_frame_distinguishes_reward_from_raw_on_a_real_run(self):
        objective = MultiObjective(
            [QedObjective(), SAScoreObjective()], weights=[1.0, 1.0]
        )
        run = GeneticOptimizer(
            objective, population_size=8, generations=2, seed=7
        ).optimize(corpus=[ASPIRIN, ETHANOL, "c1ccccc1"])
        frame = run.sa_frame()
        # the reward column the optimiser actually maximised ...
        assert "sa_score_reward" in frame.columns
        assert ((frame["sa_score_reward"] >= 0.0) & (frame["sa_score_reward"] <= 1.0)).all()
        # ... and the raw Ertl score the results table reports
        assert ((frame["sa_score"] >= 1.0) & (frame["sa_score"] <= 10.0)).all()
        assert frame["sa_valid"].all()  # the GA only keeps valid SMILES
        # sorting by raw SA reorders away from the objective rank
        easiest = run.sa_frame(sort="easiest")
        assert list(easiest["sa_score"]) == sorted(frame["sa_score"])
        # the filter contract survives a real run
        picked = run.filter_by_sa(max_sa=4.0)
        assert picked.n_total == len(run)
        assert picked.n_kept + picked.n_rejected == len(run)

    def test_ga_history_and_summary_agree_on_invalid_count(self):
        run = GeneticOptimizer(
            QedObjective(), population_size=8, generations=2, seed=7
        ).optimize(corpus=[ETHANOL, "c1ccccc1"])
        summary = run.sa_summary()
        assert summary["n_total"] == len(run)
        assert summary["n_scored"] == len(run)  # GA emits valid SMILES only
