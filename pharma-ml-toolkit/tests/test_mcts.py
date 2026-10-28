"""Tests for :mod:`pharmaml.generative.mcts` (round 100x-163).

The ChemTSv2 programming model ported to the toolkit's stack -- UCB1 /
PUCT tree policies over SMILES prefixes, the cumulative-probability
expansion, rollouts, the dedup cache and the extension mode.

The suite pins (an injected stub policy throughout -- no torch, no RNN
training, so the search logic itself is what is tested):

* the constructor's typed refusals (bad tree policy, non-callable
  policy, out-of-range numerics, a bare-string corpus, an empty
  corpus list);
* the search end-to-end: a :class:`GenerationResult` comes back with
  unique ranked molecules, one ``history`` record per cycle, and the
  search-statistics metadata;
* determinism: two same-seed runs produce identical leaderboards;
* the end token: it terminates molecules but NEVER appears in a
  generated SMILES;
* the cumulative-probability expansion rule: a top-heavy distribution
  still expands two children; the threshold cut; the renormalisation
  of a sloppy non-unit-mass custom policy;
* the extension mode (``prefix=``): the root's continuations only, the
  prefix echoed in metadata, and the typed refusal when the prefix
  leaves no room to grow;
* the PUCT tree policy variant and the breakdown path (a duck-typed
  ``score_with_breakdown`` objective lands per-objective columns);
* the empty-policy degenerate case: a policy with no moves yields the
  honest empty result, never a crash;
* the ``run_mcts_optimization`` convenience wrapper.

RDKit is required (the evaluation canonicalises rollouts through it);
the file skips as a module when it is absent, mirroring the GA tests.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from pharmaml.exceptions import DataValidationError

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_RDKIT, reason="RDKit not installed (the zip-only tree)"
)


# --------------------------------------------------------------------------- #
# fixtures: a deterministic stub policy + a stub objective
# --------------------------------------------------------------------------- #
def _flat_policy(prefix: str) -> list[tuple[str, float]]:
    """A stand-in for the RNN prior: the same distribution everywhere."""
    return [("C", 0.4), ("N", 0.2), ("O", 0.2), ("<end>", 0.2)]


def _top_heavy_policy(prefix: str) -> list[tuple[str, float]]:
    """A confident policy: one token carries 99.9% of the mass."""
    return [("C", 0.999), ("N", 0.0005), ("<end>", 0.0005)]


class _CarbonCounter:
    """A rdkit-free objective: the C-fraction of the (valid) molecule."""

    name = "carbon_fraction"

    def __init__(self) -> None:
        self.n_scored = 0
        self.n_invalid = 0

    def score(self, smiles: str) -> float:
        from pharmaml.generative._utils import canonical_smiles

        canonical = canonical_smiles(smiles)
        if canonical is None:
            self.n_invalid += 1
            return 0.0
        self.n_scored += 1
        carbons = sum(1 for char in canonical if char == "C")
        return carbons / max(len(canonical), 1)


class _BreakdownObjective(_CarbonCounter):
    """Duck-typed ``score_with_breakdown`` (the MultiObjective path)."""

    name = "multi_stub"

    def score_with_breakdown(self, smiles: str) -> tuple[float, dict[str, float]]:
        total = self.score(smiles)
        return total, {"carbon_fraction": total, "constant": 0.5}


# --------------------------------------------------------------------------- #
# constructor refusals (no search runs -- these need no RDKit either)
# --------------------------------------------------------------------------- #
def test_constructor_refusals() -> None:
    from pharmaml.generative import MctsOptimizer

    objective = _CarbonCounter()
    with pytest.raises(DataValidationError, match="tree_policy must be"):
        MctsOptimizer(objective, tree_policy="greedy")
    with pytest.raises(DataValidationError, match="policy must be a callable"):
        MctsOptimizer(objective, policy="not callable")
    with pytest.raises(DataValidationError, match="iterations must be >= 1"):
        MctsOptimizer(objective, iterations=0)
    with pytest.raises(DataValidationError, match="simulation_num must be >= 1"):
        MctsOptimizer(objective, simulation_num=0)
    with pytest.raises(DataValidationError, match="max_len must be >= 4"):
        MctsOptimizer(objective, max_len=3)
    with pytest.raises(DataValidationError, match="expansion_threshold must be in"):
        MctsOptimizer(objective, expansion_threshold=1.5)
    with pytest.raises(DataValidationError, match="expansion_threshold must be in"):
        MctsOptimizer(objective, expansion_threshold=0.0)
    with pytest.raises(DataValidationError, match="c_val must be >= 0"):
        MctsOptimizer(objective, c_val=-1)
    with pytest.raises(DataValidationError, match="puct_c must be >= 0"):
        MctsOptimizer(objective, tree_policy="puct", puct_c=-0.5)
    with pytest.raises(DataValidationError, match="prior_epochs must be >= 1"):
        MctsOptimizer(objective, prior_epochs=0)
    with pytest.raises(DataValidationError, match="corpus must be a sequence"):
        MctsOptimizer(objective, corpus="CCO")
    with pytest.raises(DataValidationError, match="corpus is empty"):
        MctsOptimizer(objective, corpus=[])
    with pytest.raises(DataValidationError, match="callable score"):
        MctsOptimizer(objective="not an objective")


# --------------------------------------------------------------------------- #
# the search itself
# --------------------------------------------------------------------------- #
def test_search_returns_ranked_unique_result() -> None:
    from pharmaml.generative import MctsOptimizer

    result = MctsOptimizer(
        _CarbonCounter(), policy=_flat_policy, iterations=6, seed=42
    ).optimize()
    assert result.metadata["method"] == "mcts"
    assert result.metadata["policy"] == "custom"
    assert result.metadata["corpus_size"] is None
    assert result.smiles, "the flat policy must find valid molecules"
    assert len(result.smiles) == len(set(result.smiles))  # deduplicated
    assert result.scores == sorted(result.scores, reverse=True)  # ranked
    assert all(0.0 <= score <= 1.0 for score in result.scores)
    assert result.best == result.smiles[0]
    assert result.best_score == result.scores[0]
    # one history record per cycle, each carrying the search fields
    assert len(result.history) == 6
    for cycle, record in enumerate(result.history):
        assert record["cycle"] == cycle
        for key in ("best_score", "mean_score", "n_unique", "n_nodes", "cache_hits"):
            assert key in record
    # the search statistics
    assert result.metadata["n_nodes"] > 1
    assert result.metadata["n_rollouts"] > 0
    assert result.metadata["iterations"] == 6
    assert result.metadata["elapsed_s"] >= 0.0


def test_search_deterministic_same_seed() -> None:
    from pharmaml.generative import MctsOptimizer

    def run() -> Any:
        return MctsOptimizer(
            _CarbonCounter(), policy=_flat_policy, iterations=5, seed=7
        ).optimize()

    first, second = run(), run()
    assert first.smiles == second.smiles
    assert first.scores == second.scores
    assert first.metadata["n_nodes"] == second.metadata["n_nodes"]


def test_end_token_never_enters_a_smiles() -> None:
    from pharmaml.generative import MctsOptimizer

    result = MctsOptimizer(
        _CarbonCounter(), policy=_flat_policy, iterations=4, seed=1
    ).optimize()
    assert result.smiles
    assert all("<end>" not in smiles for smiles in result.smiles)


def test_extension_mode_roots_at_prefix() -> None:
    from pharmaml.generative import MctsOptimizer

    result = MctsOptimizer(
        _CarbonCounter(), policy=_flat_policy, iterations=4, seed=42
    ).optimize(prefix="CC(=O)N")
    assert result.metadata["prefix"] == "CC(=O)N"
    assert result.smiles
    with pytest.raises(DataValidationError, match="no room to grow"):
        MctsOptimizer(
            _CarbonCounter(), policy=_flat_policy, max_len=8
        ).optimize(prefix="CCCCCCCCCCCC")
    with pytest.raises(DataValidationError, match="prefix must be a str"):
        MctsOptimizer(_CarbonCounter(), policy=_flat_policy).optimize(prefix=123)


def test_puct_variant_runs() -> None:
    from pharmaml.generative import MctsOptimizer

    result = MctsOptimizer(
        _CarbonCounter(),
        policy=_flat_policy,
        iterations=4,
        tree_policy="puct",
        puct_c=2.5,
        seed=42,
    ).optimize()
    assert result.metadata["tree_policy"] == "puct"
    assert result.metadata["puct_c"] == 2.5
    assert result.smiles


def test_breakdown_objective_lands_per_objective_columns() -> None:
    from pharmaml.generative import MctsOptimizer

    result = MctsOptimizer(
        _BreakdownObjective(), policy=_flat_policy, iterations=4, seed=42
    ).optimize()
    assert result.smiles
    assert all(
        set(entry) == {"carbon_fraction", "constant"} for entry in result.breakdown
    )
    frame = result.to_dataframe()
    assert "carbon_fraction" in frame.columns
    assert "constant" in frame.columns


# --------------------------------------------------------------------------- #
# expansion rule internals
# --------------------------------------------------------------------------- #
def test_expansion_set_min_two_children() -> None:
    from pharmaml.generative.mcts import MctsOptimizer

    optimizer = MctsOptimizer(_CarbonCounter(), policy=_flat_policy)
    # a top-heavy distribution: the top token alone exceeds 0.995, but
    # ChemTSv2's rule still returns two children
    chosen = optimizer._expansion_set(_top_heavy_policy(""))
    assert len(chosen) == 2
    assert chosen[0] == ("C", 0.999)
    # the flat distribution: cumulative 0.4 + 0.2 + 0.2 + 0.2 = 1.0 --
    # the threshold is crossed only at the fourth token, so all four
    chosen = optimizer._expansion_set(_flat_policy(""))
    assert len(chosen) == 4
    # a lower threshold cuts earlier (but never below two)
    optimizer = MctsOptimizer(
        _CarbonCounter(), policy=_flat_policy, expansion_threshold=0.5
    )
    chosen = optimizer._expansion_set(_flat_policy(""))
    assert len(chosen) == 2


def test_sloppy_policy_is_renormalised() -> None:
    from pharmaml.generative import MctsOptimizer

    def sloppy(prefix: str) -> list[tuple[str, float]]:
        return [("C", 0.8), ("N", 0.4), ("<end>", 0.4)]  # sums to 1.6

    result = MctsOptimizer(
        _CarbonCounter(), policy=sloppy, iterations=4, seed=42
    ).optimize()
    assert result.smiles
    assert all(0.0 <= score <= 1.0 for score in result.scores)


def test_degenerate_empty_policy_yields_empty_result() -> None:
    from pharmaml.generative import MctsOptimizer

    result = MctsOptimizer(
        _CarbonCounter(), policy=lambda prefix: [], iterations=2, seed=42
    ).optimize()
    assert result.smiles == []
    assert result.scores == []
    assert result.best == ""
    assert result.best_score == 0.0
    assert result.metadata["n_nodes"] == 1  # the root, never expanded
    assert result.metadata["n_valid"] == 0


def test_run_mcts_convenience_wrapper() -> None:
    from pharmaml.generative import run_mcts_optimization

    result = run_mcts_optimization(
        _CarbonCounter(), policy=_flat_policy, iterations=3, seed=42
    )
    assert result.metadata["method"] == "mcts"
    assert result.smiles


# --------------------------------------------------------------------------- #
# the real path: the default SmilesRnn policy (torch-gated)
# --------------------------------------------------------------------------- #
_HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed (the zip-only tree)")
def test_default_rnn_policy_search_runs() -> None:
    from pharmaml.generative import MctsOptimizer

    result = MctsOptimizer(
        _CarbonCounter(),
        iterations=2,
        simulation_num=1,
        prior_epochs=1,
        prior_hidden_dim=48,
        seed=42,
    ).optimize()
    # the default policy is the trained RNN over the bundled corpus
    assert result.metadata["policy"] == "smiles-rnn"
    assert result.metadata["corpus_size"] == 244
    assert result.metadata["n_nodes"] > 1
    assert result.metadata["n_rollouts"] > 0
    # a 1-epoch prior writes mostly invalid chemistry -- the run still
    # completes honestly (whatever it found, ranked or empty)
    assert all(0.0 <= score <= 1.0 for score in result.scores)
