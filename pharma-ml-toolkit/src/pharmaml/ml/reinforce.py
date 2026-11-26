"""Experimental reinforcement learning / sequential optimization.

The part18 §6 placement rule, stated once at the top: these methods
are **not** supervised classification or regression and do NOT
appear in the ordinary model registry.  They live here as a clearly
labelled experimental subsystem with the standard abstraction:

    Environment -> State -> Action -> Reward -> Transition -> Policy -> Evaluation

Everything is NumPy-only (no Gymnasium / stable-baselines dependency
in the core install; the environments follow the Gymnasium reset /
step / observation / action convention so a future port is trivial),
deterministic under ``seed=42`` (the toolkit pin), and bounded:
episodes, steps per episode and the Q-table size are all explicit
limits the caller controls.

Agents:

* :class:`QLearningAgent` / :func:`q_learning` -- off-policy TD(0).
* :class:`SarsaAgent` / :func:`sarsa` -- on-policy TD(0).
* :func:`dqn_train` -- a compact DQN: an experience-replay buffer,
  a frozen target network and a 2-layer NumPy MLP Q-network trained
  with hand-derived gradients (the honest scope: the state space is
  one-hot encoded, so this is DQN-as-teaching-lab, not a
  production-scale deep RL engine).
* :func:`reinforce_train` -- REINFORCE policy gradient with a softmax
  tabular policy and a moving-average baseline.
* :func:`actor_critic_train` -- tabular actor (softmax) + critic
  (state-value) with one-step TD error.

Every trainer returns the same :class:`RLRunResult` (episode reward
curve, convergence episode, the greedy policy, the value table, the
checkpoints) and every result round-trips through :meth:`to_dict` /
:meth:`from_dict` for persistence.  Policies are exported via
:func:`save_policy` / :func:`load_policy` as plain JSON.

Scientific honesty: no pharmaceutical RL environment is invented
here.  A molecule-design or experimental-design RL loop needs a
defensible state/action/reward/transition model before it deserves
the name; until then the environments are deterministic toy grids
and explicit MDP tables -- labelled experimental, measured, and
reproducible.

Examples
--------
>>> from pharmaml.ml.reinforce import GridWorldEnv, q_learning
>>> env = GridWorldEnv(rows=4, cols=5, seed=42)
>>> run = q_learning(env, episodes=150, seed=42)
>>> run.success_rate >= 0.5
True
>>> len(run.episode_rewards) == 150
True
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.ml.unsupervised import RANDOM_STATE

_LOG = get_logger(__name__)

#: The toolkit-wide seed pin threaded through every agent.
RL_SEED = RANDOM_STATE

#: Hard bounds that keep a runaway training loop from eating the box.
MAX_EPISODES = 10_000
MAX_STEPS = 2_000

_ACTIONS: tuple[tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))
_ACTION_NAMES: tuple[str, ...] = ("right", "left", "down", "up")


class GridWorldEnv:
    """A deterministic grid world: reach the goal (+1), avoid the pit (-1).

    Gymnasium-shaped API (``reset`` -> observation, ``step(action)``
    -> ``(observation, reward, terminated, truncated, info)``) without
    the dependency.  The start cell is fixed by ``seed`` so a training
    run is exactly reproducible; walls are absent by default (an empty
    grid) and the episode truncates at ``max_steps``.

    Parameters
    ----------
    rows, cols:
        Grid shape; the state id is ``row * cols + col``.
    seed:
        Picks the start cell deterministically.
    step_reward:
        The living cost per move (the classic -0.01 shaping).
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 5,
        *,
        seed: int = RL_SEED,
        step_reward: float = -0.01,
        max_steps: int = 200,
    ) -> None:
        if rows < 2 or cols < 2:
            raise DataValidationError(f"grid needs rows>=2 and cols>=2, got {rows}x{cols}")
        self.rows = int(rows)
        self.cols = int(cols)
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        self.step_reward = float(step_reward)
        self.max_steps = min(int(max_steps), MAX_STEPS)
        self.goal = (self.rows - 1, self.cols - 1)
        self.pit = (self.rows // 2, self.cols // 2) if self.rows > 2 and self.cols > 2 else None
        rng = np.random.default_rng(seed)
        self.start = (int(rng.integers(0, self.rows)), 0)
        if self.start == self.goal:
            self.start = (0, 0)
        self._pos = self.start
        self._steps = 0

    def reset(self, *, seed: int | None = None) -> int:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.start = (int(rng.integers(0, self.rows)), 0)
            if self.start == self.goal:
                self.start = (0, 0)
        self._pos = self.start
        self._steps = 0
        return self._observation()

    def _observation(self) -> int:
        return self._pos[0] * self.cols + self._pos[1]

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise DataValidationError(f"action must be 0..{self.n_actions - 1}, got {action}")
        dr, dc = _ACTIONS[action]
        row = min(max(self._pos[0] + dr, 0), self.rows - 1)
        col = min(max(self._pos[1] + dc, 0), self.cols - 1)
        self._pos = (row, col)
        self._steps += 1
        if self._pos == self.goal:
            return self._observation(), 1.0, True, False, {"pos": self._pos}
        if self.pit is not None and self._pos == self.pit:
            return self._observation(), -1.0, True, False, {"pos": self._pos}
        truncated = self._steps >= self.max_steps
        return self._observation(), self.step_reward, False, truncated, {"pos": self._pos}


class MDPSpec:
    """An explicit discrete Markov decision process (the §6 abstraction).

    ``transitions[s, a]`` is the next state, ``rewards[s, a]`` the
    immediate reward, ``terminal`` the absorbing-state mask.  All
    tables are validated on construction -- an MDP whose indices
    escape the state/action space is a defect, not a runtime surprise.
    """

    def __init__(
        self,
        transitions: Any,
        rewards: Any,
        terminal: Any,
        *,
        start_state: int = 0,
        max_steps: int = 100,
    ) -> None:
        transitions = np.asarray(transitions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        terminal = np.asarray(terminal, dtype=int)
        if transitions.ndim != 2 or rewards.ndim != 2 or terminal.ndim != 1:
            raise DataValidationError("transitions/rewards must be 2-D, terminal 1-D")
        if transitions.shape != rewards.shape:
            raise DataValidationError(
                f"transitions {transitions.shape} and rewards {rewards.shape} disagree"
            )
        n_states, n_actions = transitions.shape
        if terminal.shape[0] != n_states:
            raise DataValidationError(f"terminal must hold {n_states} states")
        if n_states == 0 or n_actions == 0:
            raise DataValidationError("empty MDP")
        if transitions.min() < 0 or transitions.max() >= n_states:
            raise DataValidationError("transition indices escape the state space")
        self.transitions = transitions
        self.rewards = rewards
        self.terminal = terminal.astype(bool)
        self.n_states = n_states
        self.n_actions = n_actions
        self.start_state = int(start_state)
        self.max_steps = min(int(max_steps), MAX_STEPS)

    def reset(self, *, seed: int | None = None) -> int:
        return self.start_state

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise DataValidationError(f"action must be 0..{self.n_actions - 1}, got {action}")
        state = self.start_state if self._current is None else self._current
        next_state = int(self.transitions[state, action])
        reward = float(self.rewards[state, action])
        self._current = next_state
        terminated = bool(self.terminal[next_state])
        truncated = False
        return next_state, reward, terminated, truncated, {}

    _current: int | None = None

    def as_env(self) -> MDPSpec:
        self._current = self.start_state
        return self


@dataclass
class RLRunResult:
    """The uniform trainer report: evidence, policy, value, checkpoints.

    Attributes
    ----------
    method:
        ``'q_learning' | 'sarsa' | 'dqn' | 'reinforce' | 'actor_critic'``.
    env:
        A description of the environment (its repr string).
    episode_rewards:
        Total reward per episode (the learning curve).
    success_rate:
        Fraction of episodes that reached the terminal goal (envs
        without a goal concept report the termination rate).
    policy:
        The greedy policy: state -> action id (or action array for
        grid worlds rendered as a string).
    value_table:
        The state-value estimate under the greedy policy.
    convergence_episode:
        First episode after which every remaining episode's reward
        stayed within ``tol`` of the final mean (``None`` when the run
        never settled -- reported honestly, not zeroed).
    checkpoints:
        ``{episode: Q snapshot}`` at the recorded cadence.
    hyperparams:
        The exact hyperparameters used (reproducibility evidence).
    """

    method: str
    env: str
    episode_rewards: list[float]
    success_rate: float
    policy: Any
    value_table: Any
    convergence_episode: int | None
    checkpoints: dict[int, Any]
    hyperparams: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = (
            f"{self.method} over {self.env} -- {len(self.episode_rewards)} episodes, "
            f"success {self.success_rate:.1%}, "
            f"convergence episode {self.convergence_episode}\n"
            f"  final mean reward (last 20): {np.mean(self.episode_rewards[-20:]):.3f}"
        )
        return head

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe round-trip (persistence for the trained policy)."""

        def _safe(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                # policy tables carry int state keys: preserve them so
                # the round-trip stays comparable (str keys would lie)
                keys = list(value.keys())
                keep_int = keys and all(isinstance(k, (int, np.integer)) for k in keys)
                return {
                    (int(k) if keep_int else str(k)): _safe(v) for k, v in value.items()
                }
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            return value

        return {
            "method": self.method,
            "env": self.env,
            "episode_rewards": self.episode_rewards,
            "success_rate": self.success_rate,
            "policy": _safe(self.policy),
            "value_table": _safe(self.value_table),
            "convergence_episode": self.convergence_episode,
            "checkpoints": {str(k): _safe(v) for k, v in self.checkpoints.items()},
            "hyperparams": _safe(self.hyperparams),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RLRunResult:
        known = {f for f in cls.__dataclass_fields__ if f in payload} - {"notes"}
        kwargs = {f: payload[f] for f in known}
        # JSON forces dict keys to strings: the policy table's state ids
        # come back as "0", "1", ... -- restore the int domain
        policy = kwargs.get("policy")
        if isinstance(policy, dict) and policy and all(
            isinstance(k, str) and k.isdigit() for k in policy
        ):
            kwargs["policy"] = {int(k): int(v) for k, v in policy.items()}
        return cls(**kwargs)


def save_policy(run: RLRunResult, path: str | Path) -> Path:
    """Persist a run (policy + curve + hyperparams) as JSON."""
    target = Path(path)
    target.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
    return target


def load_policy(path: str | Path) -> RLRunResult:
    """Reload a persisted run (the checkpoint round-trip)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RLRunResult.from_dict(payload)


def evaluate_policy(env: Any, policy: dict[int, int], *, episodes: int = 50) -> dict[str, Any]:
    """Run the greedy policy without exploration and report honestly.

    Returns the mean/total reward and the termination rate over
    ``episodes`` deterministic rollouts -- the §6 "policy evaluation"
    step.  No learning happens here.
    """
    if episodes < 1 or episodes > MAX_EPISODES:
        raise DataValidationError(f"episodes must be 1..{MAX_EPISODES}, got {episodes}")
    totals: list[float] = []
    terminated = 0
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for _step in range(env.max_steps if hasattr(env, "max_steps") else MAX_STEPS):
            action = int(policy.get(int(state), 0))
            state, reward, term, trunc, _info = env.step(action)
            total += float(reward)
            if term or trunc:
                terminated += int(bool(term))
                break
        totals.append(total)
    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(totals)),
        "total_reward": float(np.sum(totals)),
        "termination_rate": terminated / episodes,
    }


def _convergence_episode(rewards: list[float], *, window: int = 20, tol: float = 0.05) -> int | None:
    """First episode after which the running mean stops moving > ``tol``."""
    if len(rewards) < 2 * window:
        return None
    means = np.convolve(rewards, np.ones(window) / window, mode="valid")
    final_mean = float(means[-1])
    for offset in range(len(means) - 1, -1, -1):
        if abs(float(means[offset]) - final_mean) > tol:
            episode = offset + window
            return episode if episode < len(rewards) - 1 else None
    return window


def _rollout_metrics(episode_rewards: list[float], terminals: list[bool]) -> tuple[float, list[str]]:
    notes: list[str] = []
    if not any(terminals):
        notes.append("no episode reached a terminal state (truncation only)")
    success = sum(1 for t in terminals if t) / max(1, len(terminals))
    return float(success), notes


def _greedy_policy_from_q(q: np.ndarray) -> dict[int, int]:
    return {int(s): int(np.argmax(q[s])) for s in range(q.shape[0])}


def _values_from_q(q: np.ndarray) -> np.ndarray:
    return np.max(q, axis=1)


def q_learning(
    env: Any,
    *,
    episodes: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int = RL_SEED,
    checkpoint_every: int = 0,
) -> RLRunResult:
    """Off-policy Q-learning (Watkins) with epsilon-greedy exploration."""
    episodes = int(episodes)
    if not 1 <= episodes <= MAX_EPISODES:
        raise DataValidationError(f"episodes must be 1..{MAX_EPISODES}, got {episodes}")
    n_states = getattr(env, "n_states", None)
    if n_states is None:
        raise DataValidationError("environment must expose n_states / n_actions")
    q = np.zeros((env.n_states, env.n_actions))
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    terminals: list[bool] = []
    checkpoints: dict[int, Any] = {}
    for episode in range(episodes):
        state = env.reset()
        total = 0.0
        terminated = False
        for _step in range(env.max_steps if hasattr(env, "max_steps") else MAX_STEPS):
            if rng.random() < epsilon:
                action = int(rng.integers(0, env.n_actions))
            else:
                action = int(np.argmax(q[state]))
            next_state, reward, term, trunc, _info = env.step(action)
            best_next = float(np.max(q[next_state]))
            q[state, action] += alpha * (reward + gamma * best_next - q[state, action])
            total += float(reward)
            state = int(next_state)
            if term or trunc:
                terminated = terminated or bool(term)
                break
        rewards.append(total)
        terminals.append(terminated)
        if checkpoint_every and (episode + 1) % checkpoint_every == 0:
            checkpoints[episode + 1] = q.copy()
    success, notes = _rollout_metrics(rewards, terminals)
    return RLRunResult(
        method="q_learning",
        env=repr(env),
        episode_rewards=rewards,
        success_rate=success,
        policy=_greedy_policy_from_q(q),
        value_table=_values_from_q(q),
        convergence_episode=_convergence_episode(rewards),
        checkpoints=checkpoints,
        hyperparams={"alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed},
        notes=notes,
    )


def sarsa(
    env: Any,
    *,
    episodes: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int = RL_SEED,
    checkpoint_every: int = 0,
) -> RLRunResult:
    """On-policy SARSA: the update uses the action actually taken."""
    episodes = int(episodes)
    if not 1 <= episodes <= MAX_EPISODES:
        raise DataValidationError(f"episodes must be 1..{MAX_EPISODES}, got {episodes}")
    q = np.zeros((env.n_states, env.n_actions))
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    terminals: list[bool] = []
    checkpoints: dict[int, Any] = {}
    for episode in range(episodes):
        state = env.reset()
        if rng.random() < epsilon:
            action = int(rng.integers(0, env.n_actions))
        else:
            action = int(np.argmax(q[state]))
        total = 0.0
        terminated = False
        for _step in range(env.max_steps if hasattr(env, "max_steps") else MAX_STEPS):
            next_state, reward, term, trunc, _info = env.step(action)
            if rng.random() < epsilon:
                next_action = int(rng.integers(0, env.n_actions))
            else:
                next_action = int(np.argmax(q[next_state]))
            target = reward + (0.0 if term else gamma * q[next_state, next_action])
            q[state, action] += alpha * (target - q[state, action])
            total += float(reward)
            state, action = int(next_state), int(next_action)
            if term or trunc:
                terminated = terminated or bool(term)
                break
        rewards.append(total)
        terminals.append(terminated)
        if checkpoint_every and (episode + 1) % checkpoint_every == 0:
            checkpoints[episode + 1] = q.copy()
    success, notes = _rollout_metrics(rewards, terminals)
    return RLRunResult(
        method="sarsa",
        env=repr(env),
        episode_rewards=rewards,
        success_rate=success,
        policy=_greedy_policy_from_q(q),
        value_table=_values_from_q(q),
        convergence_episode=_convergence_episode(rewards),
        checkpoints=checkpoints,
        hyperparams={"alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed},
        notes=notes,
    )


def dqn_train(
    env: Any,
    *,
    episodes: int = 200,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    lr: float = 0.01,
    hidden: int = 32,
    replay_size: int = 512,
    batch: int = 32,
    target_sync: int = 20,
    seed: int = RL_SEED,
) -> RLRunResult:
    """A compact DQN: NumPy MLP Q-network, replay buffer, frozen target net.

    The honest scope: one-hot state encoding, a two-layer MLP with
    hand-derived gradients, uniform replay sampling and a target
    network synced every ``target_sync`` episodes.  This is
    DQN-as-teaching-lab over a tabular problem -- the §6 "use
    optional dependencies" rule keeps torch out of the core install,
    and nothing here pretends otherwise.
    """
    episodes = int(episodes)
    if not 1 <= episodes <= MAX_EPISODES:
        raise DataValidationError(f"episodes must be 1..{MAX_EPISODES}, got {episodes}")
    n_states, n_actions = env.n_states, env.n_actions
    rng = np.random.default_rng(seed)
    weights = {
        "w1": rng.normal(0, 0.1, (n_states, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0, 0.1, (hidden, n_actions)),
        "b2": np.zeros(n_actions),
    }
    target = {k: v.copy() for k, v in weights.items()}

    def _forward(net: dict[str, np.ndarray], state: int) -> tuple[np.ndarray, np.ndarray]:
        onehot = np.zeros(n_states)
        onehot[state] = 1.0
        hidden_out = np.tanh(onehot @ net["w1"] + net["b1"])
        q_values = hidden_out @ net["w2"] + net["b2"]
        return hidden_out, q_values

    def _backward(state: int, action: int, delta: float) -> None:
        onehot = np.zeros(n_states)
        onehot[state] = 1.0
        hidden_out, _q = _forward(weights, state)
        grad_q = np.zeros(n_actions)
        grad_q[action] = delta
        grad_h = grad_q @ weights["w2"].T
        weights["w2"] += lr * np.outer(hidden_out, grad_q)
        weights["b2"] += lr * grad_q
        grad_h_pre = grad_h * (1.0 - hidden_out**2)
        weights["w1"] += lr * np.outer(onehot, grad_h_pre)
        weights["b1"] += lr * grad_h_pre

    replay: list[tuple[int, int, float, int, bool]] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    for episode in range(episodes):
        state = env.reset()
        total = 0.0
        terminated = False
        for _step in range(env.max_steps if hasattr(env, "max_steps") else MAX_STEPS):
            if rng.random() < epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                _h, q_values = _forward(weights, state)
                action = int(np.argmax(q_values))
            next_state, reward, term, trunc, _info = env.step(action)
            replay.append((int(state), action, float(reward), int(next_state), bool(term)))
            if len(replay) > replay_size:
                replay.pop(0)
            if len(replay) >= batch:
                sample_ids = rng.integers(0, len(replay), size=batch)
                for sid in sample_ids:
                    s0, a0, r0, s1, done = replay[int(sid)]
                    _h, q_target = _forward(target, s1)
                    best = 0.0 if done else float(np.max(q_target))
                    _h2, q_now = _forward(weights, s0)
                    delta = r0 + gamma * best - float(q_now[a0])
                    _backward(s0, a0, delta)
            total += float(reward)
            state = int(next_state)
            if term or trunc:
                terminated = terminated or bool(term)
                break
        rewards.append(total)
        terminals.append(terminated)
        if (episode + 1) % target_sync == 0:
            target = {k: v.copy() for k, v in weights.items()}
    q_table = np.zeros((n_states, n_actions))
    for s_id in range(n_states):
        _h, q_values = _forward(weights, s_id)
        q_table[s_id] = q_values
    success, notes = _rollout_metrics(rewards, terminals)
    return RLRunResult(
        method="dqn",
        env=repr(env),
        episode_rewards=rewards,
        success_rate=success,
        policy=_greedy_policy_from_q(q_table),
        value_table=_values_from_q(q_table),
        convergence_episode=_convergence_episode(rewards),
        # the DQN checkpoint is keyed by name (not an episode number);
        # the int-keyed type is the episode-ladder convention
        checkpoints={"final_weights": {k: v.tolist() for k, v in weights.items()}},  # type: ignore[dict-item]
        hyperparams={
            "gamma": gamma, "epsilon": epsilon, "lr": lr, "hidden": hidden,
            "replay_size": replay_size, "batch": batch, "target_sync": target_sync,
            "seed": seed,
        },
        notes=notes,
    )


def reinforce_train(
    env: Any,
    *,
    episodes: int = 300,
    gamma: float = 0.99,
    lr: float = 0.05,
    baseline: bool = True,
    seed: int = RL_SEED,
) -> RLRunResult:
    """REINFORCE: a softmax tabular policy with a moving-average baseline."""
    episodes = int(episodes)
    if not 1 <= episodes <= MAX_EPISODES:
        raise DataValidationError(f"episodes must be 1..{MAX_EPISODES}, got {episodes}")
    preferences = np.zeros((env.n_states, env.n_actions))
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    terminals: list[bool] = []
    avg_return = 0.0
    for episode in range(episodes):
        state = env.reset()
        # (state, action, reward, the softmax probabilities at choice
        # time) -- the true gradient needs the full action distribution
        trajectory: list[tuple[int, int, float, np.ndarray]] = []
        total = 0.0
        terminated = False
        for _step in range(env.max_steps if hasattr(env, "max_steps") else MAX_STEPS):
            prefs = preferences[state] - np.max(preferences[state])
            probs = np.exp(prefs)
            probs = probs / probs.sum()
            action = int(rng.choice(env.n_actions, p=probs))
            next_state, reward, term, trunc, _info = env.step(action)
            trajectory.append((state, action, float(reward), probs.copy()))
            total += float(reward)
            state = int(next_state)
            if term or trunc:
                terminated = terminated or bool(term)
                break
        returns: list[float] = []
        g = 0.0
        for _s, _a, r, _p in reversed(trajectory):
            g = r + gamma * g
            returns.append(g)
        returns.reverse()
        baseline_value = avg_return if baseline else 0.0
        for (s_id, a_id, _r, probs_at_choice), g_val in zip(trajectory, returns):
            # the honest REINFORCE gradient: (G - b) * d log pi(a|s)
            # with the softmax log-derivative  (e_a - pi(s))
            onehot = np.zeros(env.n_actions)
            onehot[a_id] = 1.0
            preferences[s_id] += lr * (g_val - baseline_value) * (onehot - probs_at_choice)
        avg_return = 0.9 * avg_return + 0.1 * total
        rewards.append(total)
        terminals.append(terminated)
    success, notes = _rollout_metrics(rewards, terminals)
    policy = _greedy_policy_from_q(preferences)
    values = np.max(preferences, axis=1)
    return RLRunResult(
        method="reinforce",
        env=repr(env),
        episode_rewards=rewards,
        success_rate=success,
        policy=policy,
        value_table=values,
        convergence_episode=_convergence_episode(rewards),
        checkpoints={},
        hyperparams={"gamma": gamma, "lr": lr, "baseline": baseline, "seed": seed},
        notes=notes,
    )


def actor_critic_train(
    env: Any,
    *,
    episodes: int = 300,
    gamma: float = 0.99,
    alpha_actor: float = 0.05,
    alpha_critic: float = 0.1,
    seed: int = RL_SEED,
) -> RLRunResult:
    """Tabular actor-critic: softmax actor + state-value critic, one-step TD."""
    episodes = int(episodes)
    if not 1 <= episodes <= MAX_EPISODES:
        raise DataValidationError(f"episodes must be 1..{MAX_EPISODES}, got {episodes}")
    preferences = np.zeros((env.n_states, env.n_actions))
    values = np.zeros(env.n_states)
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    terminals: list[bool] = []
    for episode in range(episodes):
        state = env.reset()
        total = 0.0
        terminated = False
        for _step in range(env.max_steps if hasattr(env, "max_steps") else MAX_STEPS):
            prefs = preferences[state] - np.max(preferences[state])
            probs = np.exp(prefs)
            probs = probs / probs.sum()
            action = int(rng.choice(env.n_actions, p=probs))
            next_state, reward, term, trunc, _info = env.step(action)
            next_value = 0.0 if term else float(values[next_state])
            td_error = reward + gamma * next_value - float(values[state])
            values[state] += alpha_critic * td_error
            preferences[state, action] += alpha_actor * td_error
            total += float(reward)
            state = int(next_state)
            if term or trunc:
                terminated = terminated or bool(term)
                break
        rewards.append(total)
        terminals.append(terminated)
    success, notes = _rollout_metrics(rewards, terminals)
    return RLRunResult(
        method="actor_critic",
        env=repr(env),
        episode_rewards=rewards,
        success_rate=success,
        policy=_greedy_policy_from_q(preferences),
        value_table=values,
        convergence_episode=_convergence_episode(rewards),
        checkpoints={},
        hyperparams={
            "gamma": gamma, "alpha_actor": alpha_actor, "alpha_critic": alpha_critic,
            "seed": seed,
        },
        notes=notes,
    )


__all__ = [
    "RL_SEED",
    "GridWorldEnv",
    "MDPSpec",
    "RLRunResult",
    "actor_critic_train",
    "dqn_train",
    "evaluate_policy",
    "load_policy",
    "q_learning",
    "reinforce_train",
    "sarsa",
    "save_policy",
]
