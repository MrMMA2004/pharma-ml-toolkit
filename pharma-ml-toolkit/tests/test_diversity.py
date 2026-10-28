"""Tests for :mod:`pharmaml.generative.diversity` (round 100x-164).

The REINVENT4 diversity-filter port: the BucketCounter's off-by-one,
the five penalty functions' exact formulas, the five filters'
``score_scaffolds`` flow, the factory's name normalisation, and the
:class:`DiversityFilteredObjective` wrapper that hooks the mechanism
into the toolkit's GA / RL / MCTS optimisers.

The suite pins:

* ``BucketCounter.bucket_full`` is *strictly* greater than max_size
  (REINVENT4's own off-by-one, held here deliberately);
* every penalty function's closed form (step / linear / sigmoid /
  tanh / erf) against hand-computed values;
* the ``score_scaffolds`` order of operations: the duplicate rule
  fires *before* the bucket logic; minscore gates memory entry; the
  bucket count feeds the penalty (``step`` reproduces the classic
  zero-when-full bit-exactly, the soft functions decay smoothly);
* IdenticalMurckoScaffold vs the topological (generic-scaffold)
  variant -- a pyridine and a benzene share the generic bucket;
* ScaffoldSimilarity collapses near-duplicate scaffolds into one
  bucket (atom-pair Dice >= minsimilarity);
* PenalizeSameSmiles: the multiplier, no buckets;
* NoFilter records the memory but never touches a score;
* the factory: TOML spelling, class-name spelling, dashed spelling,
  and the unknown-name refusal that names the registry;
* the objective wrapper: single-molecule and batch paths, the
  scaffold-stats surface, reset_stats vs reset_filter, and the
  GA end-to-end integration (a wrapped objective actually steers a
  GeneticOptimizer run without changing its API).

RDKit is required; the file skips as a module when it is absent.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.generative import GeneticOptimizer, QedObjective, ScoringObjective
from pharmaml.generative.diversity import (
    DIVERSITY_FILTERS,
    SCAFFOLD_PENALTY_FUNCTIONS,
    BucketCounter,
    DiversityFilteredObjective,
    IdenticalMurckoScaffoldFilter,
    IdenticalTopologicalScaffoldFilter,
    NoFilter,
    PenalizeSameSmilesFilter,
    ScaffoldSimilarityFilter,
    make_diversity_filter,
    scaffold_penalty_erf,
    scaffold_penalty_linear,
    scaffold_penalty_sigmoid,
    scaffold_penalty_step,
    scaffold_penalty_tanh,
)

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_RDKIT, reason="RDKit not installed (the zip-only tree)"
)


class TestBucketCounter:
    def test_bucket_full_is_strictly_greater(self) -> None:
        counter = BucketCounter(max_size=2)
        counter.add("scaffold")
        counter.add("scaffold")
        assert counter.bucket_full("scaffold") is False  # 2 <= 2: not full
        counter.add("scaffold")
        assert counter.bucket_full("scaffold") is True  # 3 > 2: full

    def test_unseen_bucket_is_not_full(self) -> None:
        counter = BucketCounter(max_size=1)
        assert counter.bucket_full("nothing") is False
        assert counter.bucket_count("nothing") == 0

    def test_count_full(self) -> None:
        counter = BucketCounter(max_size=1, A=2, B=1)
        assert counter.count_full() == 1

    def test_bad_bucket_size_refused(self) -> None:
        with pytest.raises(DataValidationError, match="positive int"):
            BucketCounter(max_size=0)

    def test_round_trips_through_pickle(self) -> None:
        import pickle

        counter = BucketCounter(max_size=3, scaffold=2)
        clone = pickle.loads(pickle.dumps(counter))
        assert clone.max_size == 3 and clone.bucket_count("scaffold") == 2


class TestPenaltyFunctions:
    def test_step_is_the_classic_zero_when_full(self) -> None:
        assert scaffold_penalty_step(1, 3) == 1.0
        assert scaffold_penalty_step(4, 3) == 0.0

    def test_linear_formula(self) -> None:
        assert scaffold_penalty_linear(0, 4) == 1.0
        assert scaffold_penalty_linear(2, 4) == 0.5
        assert scaffold_penalty_linear(8, 4) == 0.0  # clamped

    def test_sigmoid_formula(self) -> None:
        value = scaffold_penalty_sigmoid(2, 4)
        exponent = -(((2 / 4) * 2 - 1) / 0.15)
        expected = 1.0 - 1.0 / (1.0 + math.exp(exponent))
        assert value == pytest.approx(expected)

    def test_tanh_formula(self) -> None:
        assert scaffold_penalty_tanh(1, 5) == pytest.approx(1.0)  # tanh(0)
        assert scaffold_penalty_tanh(4, 5) == pytest.approx(
            1.0 - math.tanh(3.0 * 3 / 5)
        )

    def test_erf_formula(self) -> None:
        value = scaffold_penalty_erf(2, 4)
        expected = (
            1.0
            + math.erf(math.sqrt(math.pi) / 4)
            - math.erf(math.sqrt(math.pi) * 2 / 4)
        )
        assert value == pytest.approx(expected)

    def test_registry_keys(self) -> None:
        assert set(SCAFFOLD_PENALTY_FUNCTIONS) == {
            "step",
            "linear",
            "sigmoid",
            "tanh",
            "erf",
        }


class TestScoreScaffoldsFlow:
    def test_duplicate_rule_fires_before_bucket_logic(self) -> None:
        f = IdenticalMurckoScaffoldFilter(bucket_size=2, minscore=0.5)
        updated = f.update_score([0.9, 0.9, 0.9], ["CCO", "CCO", "CCO"])
        assert updated.tolist() == [0.9, 0.0, 0.0]

    def test_full_bucket_zeroes_with_step_penalty(self) -> None:
        f = IdenticalMurckoScaffoldFilter(bucket_size=1, minscore=0.5)
        kids = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"]
        updated = f.update_score([0.9, 0.9, 0.9], kids)
        assert updated.tolist() == [0.9, 0.0, 0.0]

    def test_erf_penalty_decays_smoothly(self) -> None:
        # three benzene children, bucket_size=2, the erf penalty: n=1
        # is exactly 1.0 (the two erf terms cancel), n=2 and n=3 decay
        f = IdenticalMurckoScaffoldFilter(
            bucket_size=2, minscore=0.5, penalty_function="erf"
        )
        kids = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"]
        updated = f.update_score([0.8, 0.8, 0.8], kids)
        assert updated[0] == pytest.approx(0.8)  # erf(1, 2) == 1.0
        assert 0.0 < updated[1] < 0.8
        assert 0.0 < updated[2] < updated[1]

    def test_minscore_gates_memory_entry(self) -> None:
        f = IdenticalMurckoScaffoldFilter(bucket_size=1, minscore=0.5)
        # scores below minscore never enter the memory ...
        updated = f.update_score([0.1, 0.1], ["CCO", "CCO"])
        assert updated.tolist() == [0.1, 0.1]
        # ... so the second molecule is NOT a duplicate yet
        assert f.scaffold_stats()["n_memorized_smiles"] == 0

    def test_invalid_smiles_untouched(self) -> None:
        f = IdenticalMurckoScaffoldFilter(bucket_size=1, minscore=0.5)
        updated = f.update_score([0.0, 0.9], ["nonsense", "CCO"])
        assert updated.tolist() == [0.0, 0.9]

    def test_length_mismatch_refused(self) -> None:
        f = IdenticalMurckoScaffoldFilter()
        with pytest.raises(DataValidationError, match="must match"):
            f.update_score([0.5], ["CCO", "CCN"])


class TestConcreteFilters:
    def test_murcko_vs_topological_buckets(self) -> None:
        # benzene vs pyridine: DIFFERENT Murcko scaffolds, the SAME
        # generic (all-carbon) scaffold
        murcko = IdenticalMurckoScaffoldFilter(bucket_size=1, minscore=0.5)
        topological = IdenticalTopologicalScaffoldFilter(
            bucket_size=1, minscore=0.5
        )
        pair = ["c1ccccc1O", "c1ccncc1O"]
        assert murcko.update_score([0.9, 0.9], pair).tolist() == [0.9, 0.9]
        assert topological.update_score([0.9, 0.9], pair).tolist() == [0.9, 0.0]

    def test_scaffold_similarity_collapses_near_duplicates(self) -> None:
        # benzene and toluene scaffolds: atom-pair Dice >= 0.4 -- the
        # second molecule joins the FIRST bucket instead of opening one
        f = ScaffoldSimilarityFilter(bucket_size=1, minscore=0.5)
        pair = ["c1ccccc1O", "Cc1ccccc1O"]
        updated = f.update_score([0.9, 0.9], pair)
        # bucket count 2 > 1 -> the second molecule is penalised
        assert updated[0] == 0.9
        assert updated[1] == 0.0
        assert f.scaffold_stats()["n_scaffolds"] == 1

    def test_penalize_same_smiles_multiplies_repeats(self) -> None:
        f = PenalizeSameSmilesFilter(penalty_multiplier=0.5)
        updated = f.update_score([0.8, 0.8, 0.8], ["CCO", "CCO", "CCN"])
        assert updated.tolist() == [0.8, 0.4, 0.8]

    def test_penalize_same_smiles_never_touches_scaffolds(self) -> None:
        f = PenalizeSameSmilesFilter(penalty_multiplier=0.5)
        f.update_score([0.9, 0.9], ["c1ccccc1O", "c1ccccc1N"])
        assert f.scaffold_stats()["n_scaffolds"] == 0

    def test_no_filter_records_but_never_penalises(self) -> None:
        f = NoFilter(minscore=0.5)
        updated = f.update_score([0.9, 0.9, 0.9], ["CCO", "CCO", "CCN"])
        assert updated.tolist() == [0.9, 0.9, 0.9]
        assert f.scaffold_stats()["n_memorized_smiles"] == 2

    def test_acyclic_molecules_share_the_empty_scaffold(self) -> None:
        # acyclic molecules have the empty Murcko scaffold "" -- they
        # share one bucket (REINVENT4's own behaviour)
        f = IdenticalMurckoScaffoldFilter(bucket_size=1, minscore=0.5)
        updated = f.update_score([0.9, 0.9], ["CCO", "CCC"])
        assert updated.tolist() == [0.9, 0.0]

    def test_reset_purges_the_memory(self) -> None:
        f = IdenticalMurckoScaffoldFilter(bucket_size=1, minscore=0.5)
        f.update_score([0.9], ["CCO"])
        f.reset()
        assert f.scaffold_stats()["n_memorized_smiles"] == 0
        assert f.update_score([0.9], ["CCO"]).tolist() == [0.9]

    def test_constructor_refusals(self) -> None:
        with pytest.raises(DataValidationError, match="penalty_function"):
            IdenticalMurckoScaffoldFilter(penalty_function="nope")
        with pytest.raises(DataValidationError, match="minscore"):
            IdenticalMurckoScaffoldFilter(minscore=1.5)
        with pytest.raises(DataValidationError, match="minsimilarity"):
            ScaffoldSimilarityFilter(minsimilarity=-0.1)
        with pytest.raises(DataValidationError, match="penalty_multiplier"):
            PenalizeSameSmilesFilter(penalty_multiplier=0.0)


class TestFactory:
    def test_registry_holds_the_reinvent4_keys(self) -> None:
        assert set(DIVERSITY_FILTERS) == {
            "identical_murcko_scaffold",
            "identical_topological_scaffold",
            "scaffold_similarity",
            "penalize_same_smiles",
            "no_filter",
        }

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("identical_murcko_scaffold", IdenticalMurckoScaffoldFilter),
            ("IdenticalMurckoScaffold", IdenticalMurckoScaffoldFilter),
            ("identical-murcko-scaffold", IdenticalMurckoScaffoldFilter),
            ("ScaffoldSimilarity", ScaffoldSimilarityFilter),
            ("penalize_same_smiles", PenalizeSameSmilesFilter),
            ("no_filter", NoFilter),
        ],
    )
    def test_spelling_normalisation(self, kind: str, expected: type) -> None:
        assert type(make_diversity_filter(kind)) is expected

    def test_defaults_match_reinvent4(self) -> None:
        f = make_diversity_filter("ScaffoldSimilarity")
        assert f.bucket_size == 25
        assert f.minscore == pytest.approx(0.4)
        assert f.minsimilarity == pytest.approx(0.4)
        assert f.penalty_multiplier == pytest.approx(0.5)
        assert f.penalty_function == "step"

    def test_unknown_name_names_the_registry(self) -> None:
        with pytest.raises(DataValidationError, match="registry holds"):
            make_diversity_filter("no-such-filter")


class _Lucky(ScoringObjective):
    """A deterministic 0.9-everything objective for wrapper tests."""

    name = "lucky"

    def score(self, smiles: str) -> float:
        return 0.9


class TestDiversityFilteredObjective:
    def test_single_scores_flow_through_the_filter(self) -> None:
        wrapped = DiversityFilteredObjective(
            _Lucky(), "identical_murcko_scaffold", bucket_size=1, minscore=0.5
        )
        assert wrapped.score("CCO") == pytest.approx(0.9)
        assert wrapped.score("CCO") == 0.0  # the duplicate rule

    def test_batch_path_penalises_within_the_batch(self) -> None:
        wrapped = DiversityFilteredObjective(
            _Lucky(), "identical_murcko_scaffold", bucket_size=1, minscore=0.5
        )
        scores = wrapped.score_many(["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"])
        assert scores[0] == pytest.approx(0.9)
        assert scores[1] == 0.0
        assert scores[2] == 0.0

    def test_scaffold_stats_surface(self) -> None:
        wrapped = DiversityFilteredObjective(
            _Lucky(), "identical_murcko_scaffold", bucket_size=1, minscore=0.5
        )
        wrapped.score_many(["CCO", "CCC"])
        stats = wrapped.scaffold_stats()
        assert stats["n_scaffolds"] == 1  # the shared empty scaffold
        assert stats["n_memorized_smiles"] == 2
        assert stats["n_zeroed"] == 1  # the second acyclic hit the full bucket

    def test_reset_stats_keeps_the_filter_memory(self) -> None:
        wrapped = DiversityFilteredObjective(
            QedObjective(), "penalize_same_smiles", penalty_multiplier=0.5
        )
        wrapped.score("CCO")
        wrapped.reset_stats()
        # the objective counters reset, the filter memory deliberately
        # does NOT (mid-run amnesia would forget what was already seen)
        assert wrapped.score("CCO") == pytest.approx(0.406808 * 0.5)
        assert wrapped.n_scored == 1  # one score since the reset

    def test_reset_filter_purges(self) -> None:
        wrapped = DiversityFilteredObjective(
            _Lucky(), "penalize_same_smiles", penalty_multiplier=0.5
        )
        wrapped.score("CCO")
        wrapped.reset_filter()
        assert wrapped.score("CCO") == pytest.approx(0.9)

    def test_name_and_n_invalid_delegate_to_the_wrapped(self) -> None:
        wrapped = DiversityFilteredObjective(
            QedObjective(), "no_filter"
        )
        assert wrapped.name == "qed"
        wrapped.score("nonsense")
        assert wrapped.n_invalid == 1

    def test_non_objective_refused(self) -> None:
        with pytest.raises(DataValidationError, match="callable score"):
            DiversityFilteredObjective("CCO", "no_filter")  # type: ignore[arg-type]

    def test_scores_stay_in_the_unit_interval(self) -> None:
        wrapped = DiversityFilteredObjective(
            _Lucky(), "penalize_same_smiles", penalty_multiplier=0.9
        )
        for _ in range(3):
            value = wrapped.score("CCO")
            assert 0.0 <= value <= 1.0


class TestGeneticIntegration:
    def test_a_wrapped_objective_runs_end_to_end(self) -> None:
        # the honest integration claim: wrapping the objective is ALL
        # it takes -- the GA scores through it like any other objective
        objective = DiversityFilteredObjective(
            QedObjective(),
            "identical_murcko_scaffold",
            bucket_size=2,
            minscore=0.4,
        )
        result = GeneticOptimizer(
            objective, population_size=8, generations=3, seed=42
        ).optimize()
        assert result.best_score >= 0.0
        assert len(result.smiles) > 0
        stats = objective.scaffold_stats()
        assert stats["n_memorized_smiles"] >= 1
        # determinism: the wrapper holds filter state, but a fresh
        # wrapper + same seed reproduces the same leaderboard
        objective2 = DiversityFilteredObjective(
            QedObjective(),
            "identical_murcko_scaffold",
            bucket_size=2,
            minscore=0.4,
        )
        result2 = GeneticOptimizer(
            objective2, population_size=8, generations=3, seed=42
        ).optimize()
        assert result2.smiles == result.smiles
