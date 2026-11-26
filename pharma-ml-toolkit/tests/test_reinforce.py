"""Tests for :mod:`pharmaml.ml.reinforce` -- the part18 §6 lane.

The contract pins (the spec's own requirements): deterministic test
environments, reproducible seeds, episode-level metrics, reward
curves, policy evaluation, checkpointing + model persistence, action
and state validation, resource limits, and clear training status.
The RL lane must also STAY SEPARATED from the supervised registry
(the §6 opening rule -- asserted here so a future merge mistake
fails loudly).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.models import MODEL_REGISTRY
from pharmaml.ml.reinforce import (
    GridWorldEnv,
    MDPSpec,
    RLRunResult,
    actor_critic_train,
    dqn_train,
    evaluate_policy,
    load_policy,
    q_learning,
    reinforce_train,
    sarsa,
    save_policy,
)


@pytest.fixture()
def env() -> GridWorldEnv:
    return GridWorldEnv(rows=4, cols=5, seed=42)


class TestEnvironment:
    def test_reset_returns_state_in_range(self, env: GridWorldEnv) -> None:
        state = env.reset()
        assert 0 <= state < env.n_states

    def test_step_contract(self, env: GridWorldEnv) -> None:
        env.reset()
        result = env.step(0)  # right, deterministic
        assert len(result) == 5
        obs, reward, term, trunc, _info = result
        assert 0 <= obs < env.n_states
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)

    def test_goal_reward(self, env: GridWorldEnv) -> None:
        env.reset()
        # walk to the goal deterministically: the goal is bottom-right
        for _ in range(env.cols):
            env.step(0)  # right
        for _ in range(env.rows - 1):
            env.step(2)  # down
        _obs, reward, term, _trunc, _info = env.step(0)
        assert term is True
        assert reward == 1.0

    def test_bad_action_refused(self, env: GridWorldEnv) -> None:
        env.reset()
        with pytest.raises(DataValidationError):
            env.step(9)

    def test_bad_grid_refused(self) -> None:
        with pytest.raises(DataValidationError):
            GridWorldEnv(rows=1, cols=5)

    def test_mdp_spec_validates_indices(self) -> None:
        with pytest.raises(DataValidationError):
            MDPSpec(
                transitions=[[5, 0], [0, 0]],
                rewards=[[0.0, 0.0], [0.0, 0.0]],
                terminal=[0, 0],
            )

    def test_mdp_spec_step(self) -> None:
        mdp = MDPSpec(
            transitions=[[1, 1], [0, 0]],
            rewards=[[1.0, 1.0], [0.0, 0.0]],
            terminal=[0, 1],
        )
        mdp.reset()
        state, reward, term, _trunc, _info = mdp.step(0)
        assert (state, reward, term) == (1, 1.0, True)


class TestQLearning:
    def test_learns_the_grid(self, env: GridWorldEnv) -> None:
        run = q_learning(env, episodes=150, seed=42)
        assert isinstance(run, RLRunResult)
        assert len(run.episode_rewards) == 150
        assert run.success_rate >= 0.5
        # the learning curve improves: last-20 mean >> first-20 mean
        assert np.mean(run.episode_rewards[-20:]) > np.mean(
            run.episode_rewards[:20]
        )

    def test_reproducible_seed(self, env: GridWorldEnv) -> None:
        first = q_learning(env, episodes=80, seed=7)
        second = q_learning(env, episodes=80, seed=7)
        assert first.episode_rewards == second.episode_rewards

    def test_checkpoints_recorded(self, env: GridWorldEnv) -> None:
        run = q_learning(env, episodes=60, checkpoint_every=30)
        assert set(run.checkpoints) == {30, 60}

    def test_policy_evaluated(self, env: GridWorldEnv) -> None:
        run = q_learning(env, episodes=150, seed=42)
        verdict = evaluate_policy(env, run.policy, episodes=25)
        assert verdict["episodes"] == 25
        assert verdict["mean_reward"] > 0.5
        assert verdict["termination_rate"] == 1.0

    def test_bad_episodes_refused(self, env: GridWorldEnv) -> None:
        with pytest.raises(DataValidationError):
            q_learning(env, episodes=0)

    def test_env_needs_state_space(self) -> None:
        class _Bare:
            pass

        with pytest.raises(DataValidationError):
            q_learning(_Bare(), episodes=10)  # type: ignore[arg-type]


class TestSarsa:
    def test_learns_the_grid(self, env: GridWorldEnv) -> None:
        run = sarsa(env, episodes=150, seed=42)
        assert run.method == "sarsa"
        assert run.success_rate >= 0.4
        assert run.policy  # non-empty

    def test_reproducible_seed(self, env: GridWorldEnv) -> None:
        assert sarsa(env, episodes=80, seed=7).episode_rewards == sarsa(
            env, episodes=80, seed=7
        ).episode_rewards


class TestDQN:
    def test_compact_dqn_trains(self, env: GridWorldEnv) -> None:
        run = dqn_train(env, episodes=120, seed=42, batch=16, replay_size=256)
        assert run.method == "dqn"
        assert len(run.episode_rewards) == 120
        assert "final_weights" in run.checkpoints
        # the hyperparams carry the full evidence set
        for key in ("gamma", "epsilon", "lr", "hidden", "batch", "seed"):
            assert key in run.hyperparams

    def test_reproducible_seed(self, env: GridWorldEnv) -> None:
        first = dqn_train(env, episodes=40, seed=3, batch=8, replay_size=64)
        second = dqn_train(env, episodes=40, seed=3, batch=8, replay_size=64)
        assert first.episode_rewards == second.episode_rewards


class TestPolicyGradient:
    def test_reinforce_learns(self, env: GridWorldEnv) -> None:
        run = reinforce_train(env, episodes=300, seed=42)
        assert run.method == "reinforce"
        assert run.success_rate >= 0.4

    def test_actor_critic_learns(self, env: GridWorldEnv) -> None:
        run = actor_critic_train(env, episodes=300, seed=42)
        assert run.method == "actor_critic"
        assert run.success_rate >= 0.4


class TestPersistence:
    def test_round_trip(self, env: GridWorldEnv, tmp_path) -> None:
        run = q_learning(env, episodes=50, seed=42, checkpoint_every=25)
        path = save_policy(run, tmp_path / "policy.json")
        reloaded = load_policy(path)
        assert reloaded.method == run.method
        assert reloaded.episode_rewards == run.episode_rewards
        assert reloaded.policy == run.policy
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["hyperparams"]["seed"] == 42


class TestSeparationRule:
    def test_rl_not_in_supervised_registry(self) -> None:
        """part18 §6: these are not ordinary supervised models."""
        for forbidden in (
            "q_learning",
            "sarsa",
            "dqn",
            "reinforce",
            "actor_critic",
            "md_p",
            "markov",
        ):
            assert forbidden not in MODEL_REGISTRY

    def test_module_docstring_labels_experimental(self) -> None:
        from pharmaml.ml import reinforce as module

        assert "Experimental reinforcement learning" in module.__doc__
