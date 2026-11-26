"""``/api/v1/reinforce`` — the part18 §6 experimental RL lane.

EXPERIMENTAL, and labelled so everywhere: these are sequential-
optimization teaching agents (Q-learning, SARSA, a compact DQN,
REINFORCE, actor-critic) over deterministic grid environments and
explicit MDP tables — NOT pharmaceutical models and NOT supervised
registry entries (the §6 separation rule is enforced by a test).
The routes run the real :mod:`pharmaml.ml.reinforce` engine with
seeded reproducibility, bounded episodes/steps, episode-level
metrics, the greedy policy + value table, convergence reporting and
the JSON round-trip persistence (the trained policy downloads as
an artifact through the standard artifacts lane).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.ml.reinforce import (
    GridWorldEnv,
    actor_critic_train,
    dqn_train,
    evaluate_policy,
    q_learning,
    reinforce_train,
    sarsa,
)

from ..jobs import CancelledJob
from ..stores import json_safe

router = APIRouter(prefix="/reinforce", tags=["reinforcement-learning"])

_Method = Literal["q_learning", "sarsa", "dqn", "reinforce", "actor_critic"]

#: The resource bounds mirrored from the engine (§6's own limits).
_MAX_EPISODES_REQUEST = 2000


class ReinforceTrainRequest(BaseModel):
    """POST /api/v1/reinforce/train — the experimental lane body."""

    method: _Method = "q_learning"
    rows: int = Field(default=4, ge=2, le=20)
    cols: int = Field(default=5, ge=2, le=20)
    episodes: int = Field(default=200, ge=1, le=_MAX_EPISODES_REQUEST)
    seed: int = 42
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    # TD learners
    alpha: float = Field(default=0.1, gt=0.0, le=1.0)
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    # DQN
    lr: float = Field(default=0.01, gt=0.0, le=1.0)
    hidden: int = Field(default=32, ge=4, le=128)
    batch: int = Field(default=32, ge=4, le=128)
    replay_size: int = Field(default=512, ge=32, le=4096)
    # policy-gradient learners
    alpha_actor: float = Field(default=0.05, gt=0.0, le=1.0)
    alpha_critic: float = Field(default=0.1, gt=0.0, le=1.0)
    baseline: bool = True
    mode: Literal["job", "sync"] = "job"


class ReinforceEvaluateRequest(BaseModel):
    """POST /api/v1/reinforce/evaluate — the greedy-policy rollout."""

    method: _Method = "q_learning"
    rows: int = Field(default=4, ge=2, le=20)
    cols: int = Field(default=5, ge=2, le=20)
    seed: int = 42
    episodes: int = Field(default=50, ge=1, le=500)
    policy: dict[str, int] = Field(
        description="the trained run's policy table (state id -> action id)"
    )


def _run_view(run) -> dict[str, Any]:
    rewards = np.asarray(run.episode_rewards, dtype=float)
    curve = {
        "episodes": int(rewards.shape[0]),
        "first_20_mean": json_safe(float(rewards[:20].mean())),
        "last_20_mean": json_safe(float(rewards[-20:].mean())),
        "max": json_safe(float(rewards.max())),
        "min": json_safe(float(rewards.min())),
    }
    return {
        "method": run.method,
        "env": run.env,
        "success_rate": json_safe(run.success_rate),
        "reward_curve": curve,
        "convergence_episode": run.convergence_episode,
        "hyperparams": json_safe(run.hyperparams),
        "policy": {str(int(k)): int(v) for k, v in run.policy.items()},
        "value_table_preview": json_safe(
            np.asarray(run.value_table)[:16].tolist()
        ),
        "notes": list(run.notes),
        "experimental": True,
    }


def _train_body(
    method: str,
    env_args: dict,
    kwargs: dict,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL trainer over the deterministic grid."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the RL training loop.")
    env = GridWorldEnv(seed=env_args["seed"], rows=env_args["rows"], cols=env_args["cols"])
    if method == "q_learning":
        run = q_learning(env, **kwargs)
    elif method == "sarsa":
        run = sarsa(env, **kwargs)
    elif method == "dqn":
        run = dqn_train(env, **kwargs)
    elif method == "reinforce":
        run = reinforce_train(env, **kwargs)
    elif method == "actor_critic":
        run = actor_critic_train(env, **kwargs)
    else:  # pragma: no cover - the Literal binds the set
        raise HTTPException(status_code=422, detail=f"unknown method {method!r}")
    return {"reinforcement": _run_view(run)}


def _method_kwargs(req: ReinforceTrainRequest) -> dict:
    shared = {"episodes": req.episodes, "gamma": req.gamma, "seed": req.seed}
    if req.method in ("q_learning", "sarsa"):
        return {**shared, "alpha": req.alpha, "epsilon": req.epsilon}
    if req.method == "dqn":
        return {
            "episodes": req.episodes, "gamma": req.gamma, "seed": req.seed,
            "epsilon": req.epsilon, "lr": req.lr, "hidden": req.hidden,
            "batch": req.batch, "replay_size": req.replay_size,
        }
    if req.method == "reinforce":
        return {
            "episodes": req.episodes, "gamma": req.gamma, "seed": req.seed,
            "baseline": req.baseline, "lr": req.lr,
        }
    return {  # actor_critic
        "episodes": req.episodes, "gamma": req.gamma, "seed": req.seed,
        "alpha_actor": req.alpha_actor, "alpha_critic": req.alpha_critic,
    }


@router.get("/methods")
def methods() -> dict:
    """The capability listing with the experimental label front and centre."""
    return {
        "status": "experimental",
        "separation_rule": (
            "not supervised classification/regression; never in the "
            "MODEL_REGISTRY -- the §6 rule, enforced by test_reinforce.py"
        ),
        "agents": [
            {
                "method": "q_learning",
                "python": "pharmaml.ml.reinforce.q_learning",
                "route": "/api/v1/reinforce/train",
                "type": "value-based, off-policy TD(0)",
            },
            {
                "method": "sarsa",
                "python": "pharmaml.ml.reinforce.sarsa",
                "route": "/api/v1/reinforce/train",
                "type": "value-based, on-policy TD(0)",
            },
            {
                "method": "dqn",
                "python": "pharmaml.ml.reinforce.dqn_train",
                "route": "/api/v1/reinforce/train",
                "type": "compact DQN: NumPy MLP + replay + frozen target",
            },
            {
                "method": "reinforce",
                "python": "pharmaml.ml.reinforce.reinforce_train",
                "route": "/api/v1/reinforce/train",
                "type": "policy gradient (REINFORCE, softmax, baseline)",
            },
            {
                "method": "actor_critic",
                "python": "pharmaml.ml.reinforce.actor_critic_train",
                "route": "/api/v1/reinforce/train",
                "type": "tabular actor + one-step TD critic",
            },
        ],
        "environments": [
            {
                "env": "GridWorldEnv",
                "python": "pharmaml.ml.reinforce.GridWorldEnv",
                "note": "deterministic grid; Gymnasium-shaped API, no dependency",
            },
            {
                "env": "MDPSpec",
                "python": "pharmaml.ml.reinforce.MDPSpec",
                "note": "explicit transition/reward/terminal tables",
            },
        ],
        "scientific_honesty": (
            "no pharmaceutical RL environment is invented; molecular "
            "optimization RL needs a defensible MDP definition first"
        ),
    }


@router.post("/train")
def train(req: ReinforceTrainRequest, request: Request) -> dict:
    env_args = {"rows": req.rows, "cols": req.cols, "seed": req.seed}
    kwargs = _method_kwargs(req)
    if req.mode == "sync":
        return {"mode": "sync", **_train_body(req.method, env_args, kwargs)}
    manager = request.app.state.job_manager
    record = manager.submit(
        "reinforce.train",
        "machine-learning",
        _train_body,
        configuration={
            "method": req.method, **json_safe(env_args), **json_safe(kwargs),
        },
        method=req.method,
        env_args=env_args,
        kwargs=kwargs,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/evaluate")
def evaluate(req: ReinforceEvaluateRequest, request: Request) -> dict:
    """Deterministic greedy rollouts of a trained policy (no learning)."""
    env = GridWorldEnv(rows=req.rows, cols=req.cols, seed=req.seed)
    policy: dict[int, int] = {}
    for key, action in req.policy.items():
        try:
            policy[int(key)] = int(action)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"policy keys/values must be ints, got {key!r}->{action!r}",
            ) from None
    verdict = evaluate_policy(env, policy, episodes=req.episodes)
    return {
        "evaluation": {
            "method": req.method,
            "env": repr(env),
            "episodes": verdict["episodes"],
            "mean_reward": json_safe(verdict["mean_reward"]),
            "termination_rate": json_safe(verdict["termination_rate"]),
            "experimental": True,
        }
    }
