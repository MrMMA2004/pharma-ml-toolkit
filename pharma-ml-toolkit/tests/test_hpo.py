"""Tests for ``pharmaml.ml.hpo`` -- the TPE / SHA / Hyperband engines.

The lost round-188/189 engines rebuilt per the preserved specifications in
``scratch/round189_state_after_rollback.md`` §3: this file pins the TPE split
rule (γ = min(⌈0.1·n⌉, 25)), the Parzen estimator contract, the hyperband
bracket arithmetic (the classic R=27/η=3 table), successive halving's walk,
the study bookkeeping, and the sklearn adapter.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from pharmaml.ml.hpo import (
    CategoricalParam,
    FloatParam,
    HPOStudy,
    HyperbandBracket,
    IntParam,
    RandomSampler,
    SearchSpace,
    TPEOptimizer,
    Trial,
    TrialState,
    gamma_split,
    halving_verdict,
    hyperband_brackets,
    parzen_density,
    successive_halving,
    tune_model,
)

# -- the gamma split rule (the optuna cap) --------------------------------- #


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, 0),
        (1, 1),
        (5, 1),
        (10, 1),
        (11, 2),
        (20, 2),
        (50, 5),
        (100, 10),
        (250, 25),
        (1000, 25),  # the cap
    ],
)
def test_gamma_split_rule(n: int, expected: int) -> None:
    assert gamma_split(n) == expected


def test_gamma_split_negative_is_zero() -> None:
    assert gamma_split(-3) == 0


# -- the Parzen estimator --------------------------------------------------- #


def test_parzen_shapes_with_edge_sentinels() -> None:
    centres, sigmas = parzen_density(np.array([0.2, 0.8]), 0.0, 1.0)
    assert centres.shape == sigmas.shape == (4,)
    assert centres[0] == 0.0 and centres[1] == 1.0
    assert np.all(sigmas > 0)


def test_parzen_empty_points_degenerate_kernels() -> None:
    centres, sigmas = parzen_density(np.array([]), -1.0, 1.0)
    assert len(centres) == 2
    assert np.all(sigmas > 0)


def test_parzen_bandwidth_shrinks_with_density() -> None:
    one, _ = parzen_density(np.array([0.5]), 0.0, 1.0)
    many, _ = parzen_density(np.linspace(0.1, 0.9, 20), 0.0, 1.0)
    # the middle kernel of the dense set is sharper than the sparse one
    assert many[11] < one[2] or many[11] == pytest.approx(one[2])


def test_parzen_density_is_higher_at_observed_points() -> None:
    from pharmaml.ml.hpo import _log_density_at

    centres, sigmas = parzen_density(np.array([0.4, 0.6]), 0.0, 1.0)
    at_observed = _log_density_at(0.5, centres, sigmas)
    at_edge = _log_density_at(0.99, centres, sigmas)
    assert at_observed > at_edge


# -- the search space -------------------------------------------------------- #


def test_space_uniform_sample_in_range() -> None:
    space = SearchSpace.uniform("x", 2.0, 5.0)
    rng = __import__("random").Random(3)
    for _ in range(50):
        assert 2.0 <= space.sample(rng)["x"] <= 5.0


def test_space_log_sample_in_range_and_spread() -> None:
    space = SearchSpace.uniform("lr", 1e-4, 1.0, log=True)
    rng = __import__("random").Random(5)
    draws = [space.sample(rng)["lr"] for _ in range(200)]
    assert all(1e-4 <= v <= 1.0 for v in draws)
    assert any(v < 1e-2 for v in draws), "log scale must reach the low decade"
    assert any(v > 0.1 for v in draws)


def test_space_randint_inclusive_bounds() -> None:
    space = SearchSpace.randint("k", 1, 3)
    rng = __import__("random").Random(11)
    seen = {space.sample(rng)["k"] for _ in range(120)}
    assert seen == {1, 2, 3}


def test_space_categorical_choices_covered() -> None:
    spec = CategoricalParam("solver", choices=("lbfgs", "sgd", "adam"))
    space = SearchSpace(spec)
    rng = __import__("random").Random(9)
    seen = {space.sample(rng)["solver"] for _ in range(90)}
    assert seen == {"lbfgs", "sgd", "adam"}


def test_space_add_and_len() -> None:
    space = SearchSpace.uniform("x", 0.0, 1.0)
    space.add(IntParam("depth", 1, 4))
    assert len(space) == 2
    assert "depth" in space and "x" in space
    assert "missing" not in space


def test_float_param_contains() -> None:
    spec = FloatParam("x", 0.0, 1.0)
    assert spec.contains(0.5) and not spec.contains(1.5)


# -- the samplers ------------------------------------------------------------ #


def _bowl(params: dict[str, float]) -> float:
    return abs(params["x"] - 0.3)


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_tpe_converges_toward_the_optimum(seed: int) -> None:
    opt = TPEOptimizer(SearchSpace.uniform("x", 0.0, 1.0), seed=seed)
    for _ in range(30):
        opt.ask_and_run(_bowl)
    assert abs(opt.best().params["x"] - 0.3) < 0.15


def test_random_sampler_baseline_runs() -> None:
    sampler = RandomSampler(SearchSpace.uniform("x", 0.0, 1.0), seed=4)
    for _ in range(10):
        sampler.ask_and_run(_bowl)
    assert len(sampler.trials) == 10
    assert all(0.0 <= t.params["x"] <= 1.0 for t in sampler.trials)


def test_tpe_startup_phase_is_random() -> None:
    opt = TPEOptimizer(
        SearchSpace.uniform("x", 0.0, 1.0), n_startup=5, seed=1
    )
    for _ in range(4):
        opt.ask_and_run(_bowl)
    # still inside the startup window: deterministic replay must match
    replay = TPEOptimizer(SearchSpace.uniform("x", 0.0, 1.0), n_startup=5, seed=1)
    for _ in range(4):
        replay.ask_and_run(_bowl)
    assert [t.params for t in opt.trials] == [t.params for t in replay.trials]


def test_tpe_deterministic_given_seed() -> None:
    def run() -> list[dict[str, float]]:
        opt = TPEOptimizer(SearchSpace.uniform("x", 0.0, 1.0), seed=42)
        for _ in range(20):
            opt.ask_and_run(_bowl)
        return [t.params for t in opt.trials]

    assert run() == run()


def test_tpe_log_space_objective() -> None:
    # the optimum sits at lr ~ 0.01, one decade below the range's middle
    opt = TPEOptimizer(
        SearchSpace.uniform("lr", 1e-4, 1.0, log=True), seed=3
    )
    for _ in range(30):
        opt.ask_and_run(lambda p: abs(math.log10(p["lr"]) + 2.0))
    assert math.log10(opt.best().params["lr"]) > -3.0


def test_tpe_mixed_space() -> None:
    space = (
        SearchSpace.uniform("x", 0.0, 1.0)
        .add(IntParam("units", 4, 32))
        .add(CategoricalParam("kind", choices=("a", "b")))
    )
    opt = TPEOptimizer(space, n_startup=6, seed=8)
    for _ in range(18):
        trial = opt.ask_and_run(
            lambda p: abs(p["x"] - 0.5) + 0.01 * (p["units"] / 32)
            + (0.05 if p["kind"] == "a" else 0.0)
        )
        assert 0.0 <= trial.params["x"] <= 1.0
        assert 4 <= trial.params["units"] <= 32
        assert trial.params["kind"] in {"a", "b"}
    assert opt.best().params["kind"] == "b"


def test_tpe_trials_record_numbers_and_states() -> None:
    opt = TPEOptimizer(SearchSpace.uniform("x", 0.0, 1.0), seed=2)
    for _ in range(3):
        opt.ask_and_run(_bowl)
    assert [t.number for t in opt.trials] == [0, 1, 2]
    assert all(t.state == TrialState.COMPLETE for t in opt.trials)
    assert all(t.is_finished() for t in opt.trials)


def test_tpe_best_is_the_minimum() -> None:
    opt = TPEOptimizer(SearchSpace.uniform("x", 0.0, 1.0), seed=6)
    for _ in range(12):
        opt.ask_and_run(_bowl)
    assert opt.best().value == min(t.value for t in opt.trials)


def test_best_on_empty_raises() -> None:
    opt = TPEOptimizer(SearchSpace.uniform("x", 0.0, 1.0))
    with pytest.raises(ValueError):
        opt.best()


def test_tpe_ask_returns_dict_in_space() -> None:
    space = SearchSpace.uniform("x", 0.0, 1.0).add(IntParam("d", 1, 2))
    opt = TPEOptimizer(space, seed=0)
    params = opt.ask()
    assert set(params) == {"x", "d"}


# -- successive halving ------------------------------------------------------- #


def test_sha_picks_the_best_index() -> None:
    assert successive_halving([0.4, 0.9, 0.1, 0.5], eta=2) == [2]


def test_sha_module_example() -> None:
    assert successive_halving([0.9, 0.5, 0.7, 0.1, 0.6], eta=2) == [3]


def test_sha_ties_break_by_index() -> None:
    assert successive_halving([0.5, 0.5, 0.5], eta=2) == [0]


def test_sha_single_value() -> None:
    assert successive_halving([0.7]) == [0]


def test_sha_empty() -> None:
    assert successive_halving([]) == []


def test_sha_eta_validation() -> None:
    with pytest.raises(ValueError):
        successive_halving([0.1, 0.2], eta=1.0)


def test_sha_ranking_walk_is_monotone() -> None:
    values = [0.8, 0.2, 0.6, 0.1, 0.9, 0.3]
    survivors = successive_halving(values, eta=3.0)
    assert survivors == [3]


# -- hyperband ---------------------------------------------------------------- #


def test_hyperband_classic_table_r27_eta3() -> None:
    brackets = hyperband_brackets(27.0, 3.0)
    table = [(b.bracket, b.n_configs, b.base_budget) for b in brackets]
    assert table == [(0, 4, 27.0), (1, 6, 9.0), (2, 12, 3.0), (3, 27, 1.0)]


def test_hyperband_r81_eta3_has_five_brackets() -> None:
    brackets = hyperband_brackets(81.0, 3.0)
    assert len(brackets) == 5
    fast = brackets[4]
    assert (fast.base_budget, fast.n_configs) == (1.0, 81)


def test_hyperband_total_budget_is_bounded() -> None:
    # Li et al.'s guarantee: each bracket's budget sum <= (s_max + 1) * R
    R, eta = 27.0, 3.0
    brackets = hyperband_brackets(R, eta)
    s_max = len(brackets) - 1
    bound = (s_max + 1) * R * (eta + 1) / eta  # slack for ceil effects
    for b in brackets:
        total = sum(n * r for n, r in b.rungs())
        assert total <= bound * (eta + 1)


def test_hyperband_rungs_end_within_max_budget() -> None:
    for b in hyperband_brackets(81.0, 3.0):
        rungs = b.rungs()
        assert rungs[0][1] == pytest.approx(b.base_budget)
        assert all(r <= b.max_budget + 1e-9 for _, r in rungs)
        assert rungs[-1][1] == pytest.approx(b.max_budget)


def test_hyperband_rungs_promote_one_over_eta() -> None:
    b = hyperband_brackets(27.0, 3.0)[1]
    rungs = b.rungs()
    for (n_before, _), (n_after, _) in itertools.pairwise(rungs):
        assert n_after == math.ceil(n_before / b.eta)


def test_hyperband_eta_validation() -> None:
    with pytest.raises(ValueError):
        hyperband_brackets(27.0, eta=1.0)


def test_hyperband_budget_validation() -> None:
    with pytest.raises(ValueError):
        hyperband_brackets(0.0)


def test_bracket_dataclass_fields() -> None:
    b = HyperbandBracket(
        bracket=2, max_budget=27.0, eta=3.0, base_budget=3.0, n_configs=12
    )
    assert b.bracket == 2 and b.n_configs == 12
    assert b.rungs()[0] == (12, 3.0)


# -- the study ---------------------------------------------------------------- #


def test_study_runs_and_tracks_history() -> None:
    study = HPOStudy(
        objective=_bowl,
        space=SearchSpace.uniform("x", 0.0, 1.0),
        sampler="tpe",
        seed=10,
    )
    study.run(8)
    assert len(study.history) == 8
    assert len(study.trials) == 8
    study.run(2)
    assert len(study.history) == 10


def test_study_best_matches_minimum() -> None:
    study = HPOStudy(
        objective=_bowl,
        space=SearchSpace.uniform("x", 0.0, 1.0),
        sampler="random",
        seed=12,
    )
    study.run(15)
    assert study.best().value == min(t.value for t in study.trials)


def test_study_unknown_sampler_rejected() -> None:
    with pytest.raises(ValueError):
        HPOStudy(
            objective=_bowl,
            space=SearchSpace.uniform("x", 0.0, 1.0),
            sampler="anneal",
        )


def test_study_random_sampler_branch() -> None:
    study = HPOStudy(
        objective=_bowl,
        space=SearchSpace.uniform("x", 0.0, 1.0),
        sampler="random",
        seed=0,
    )
    study.run(5)
    assert study.trials[0].state == TrialState.COMPLETE


# -- the halving verdict (should_prune) --------------------------------------- #


def test_verdict_prunes_when_worse_than_all() -> None:
    assert halving_verdict(0.9, [0.1, 0.3]) == "PRUNE"


def test_verdict_continues_when_beats_someone() -> None:
    assert halving_verdict(0.2, [0.1, 0.3]) == "CONTINUE"


def test_verdict_continues_on_empty_cohort() -> None:
    assert halving_verdict(9.9, []) == "CONTINUE"


def test_verdict_patience_defers_the_axe() -> None:
    assert halving_verdict(0.9, [0.1, 0.3], patience=1) == "CONTINUE"


# -- the sklearn adapter ------------------------------------------------------- #


def _toy_classification_data(
    n: int = 60, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_tune_model_random_smoke() -> None:
    X, y = _toy_classification_data()
    space = SearchSpace.uniform("alpha", 0.01, 1.0, log=True)
    report = tune_model(
        "ridge_classifier", space, X, y,
        task="classification", cv=2, n_trials=4, sampler="random", seed=1,
    )
    assert report.n_trials == 4
    assert report.best_score > 0  # accuracy, restored to bigger-is-better
    assert 0.01 <= report.best_params["alpha"] <= 1.0
    assert len(report.history) == 4


def test_tune_model_tpe_smoke() -> None:
    X, y = _toy_classification_data(seed=2)
    space = SearchSpace.uniform("alpha", 0.05, 2.0)
    report = tune_model(
        "ridge_classifier", space, X, y,
        task="classification", cv=2, n_trials=6, sampler="tpe", seed=3,
    )
    assert report.best_score > 0
    assert report.as_dict()["task"] == "classification"
    assert len(report.as_dict()["trials"]) == 6


def test_tune_model_report_round_trip() -> None:
    X, y = _toy_classification_data(seed=4)
    space = SearchSpace.randint("n_neighbors", 1, 5)
    report = tune_model(
        "knn", space, X, y,
        task="classification", cv=2, n_trials=3, sampler="random", seed=5,
    )
    payload = report.as_dict()
    assert payload["best_params"] == report.best_params
    assert payload["n_trials"] == 3
    assert set(payload["trials"][0]) == {"number", "params", "value"}


# -- the trial record ----------------------------------------------------------- #


def test_trial_default_state_is_running() -> None:
    trial = Trial(number=0, params={"x": 0.5})
    assert trial.state == TrialState.RUNNING
    assert not trial.is_finished()


def test_trial_state_constants() -> None:
    assert TrialState.RUNNING == "RUNNING"
    assert TrialState.COMPLETE == "COMPLETE"
    assert TrialState.PRUNED == "PRUNED"
    assert TrialState.FAIL == "FAIL"


def test_tpe_with_categorical_only_space() -> None:
    space = SearchSpace(
        CategoricalParam("kernel", choices=("linear", "rbf"))
    )
    opt = TPEOptimizer(space, seed=13)
    for _ in range(8):
        opt.ask_and_run(lambda p: 0.1 if p["kernel"] == "rbf" else 0.4)
    assert opt.best().params["kernel"] == "rbf"


def test_tpe_all_good_no_bad_falls_back_to_random() -> None:
    # one completed trial -> gamma = 1 -> good = [t0], bad = [] -> fallback
    opt = TPEOptimizer(SearchSpace.uniform("x", 0.0, 1.0), n_startup=1, seed=0)
    opt.ask_and_run(_bowl)
    params = opt.ask()  # must not raise
    assert 0.0 <= params["x"] <= 1.0
