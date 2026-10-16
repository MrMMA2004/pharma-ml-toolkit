"""Reinforcement-learning molecule generation (REINVENT-style fine-tuning).

Takes a character-level SMILES "prior" (:class:`~pharmaml.generative.SmilesRnn`
trained on a corpus of drug-like molecules) and fine-tunes it toward a user
scoring function with policy gradients -- the REINVENT family recipe,
re-implemented small enough to run on a laptop CPU.

**Chosen variant (documented honestly).**  This is *REINFORCE with a
running baseline*, not the original augmented-likelihood formulation:

* each iteration samples a batch from the agent, scores every string
  (invalid SMILES score 0.0, so their probability is actively pushed
  *down* -- validity is learned, not assumed);
* the advantage is ``(score - baseline) / running_std`` where the
  baseline is an exponential moving average of *previous* batches' mean
  scores and ``running_std`` an EMA of their return spread (classic
  variance reduction).  Both statistics are folded in *after* each
  gradient, so the current batch cannot cancel itself out of its own
  advantage (a per-batch mean subtraction would make ANY baseline
  mathematically inert);
* an **entropy bonus** keeps exploration alive;
* a **prior anchor** (the mode-collapse brake) adds
  ``anchor_weight * mean((logP_agent - logP_prior)^2)`` -- a soft
  KL-style penalty pulling the agent's sequence likelihood back toward
  the frozen prior, so the policy cannot cheaply collapse onto a single
  high-scoring string.  ``anchor_weight=0`` recovers vanilla
  REINFORCE-with-baseline;
* an **experience replay buffer** keeps the best-scoring unique samples
  and mixes them into every training batch (REINVENT-style replay);
* gradients are clipped and the run stops early when the mean score
  plateaus.

Torch is optional at package level (the :mod:`pharmaml.ml.neural`
contract): importing this module never requires it; running
:meth:`ReinventOptimizer.optimize` raises an actionable
:class:`~pharmaml.exceptions.DependencyError` when it is missing.

Example
-------
>>> from pharmaml.generative import QedObjective, run_rl_optimization
>>> result = run_rl_optimization(QedObjective(), iterations=10, seed=42)
>>> result.best_score >= 0.5
True
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

from ._utils import canonical_smiles
from .corpus import DEFAULT_CORPUS
from .result import GenerationResult
from .rnn import SmilesRnn, _require_torch, _validate_torch_seed

__all__ = ["ReinventOptimizer", "run_rl_optimization"]

_LOG = get_logger(__name__)

#: Initial value of the REINFORCE running baseline.  ``None`` (default)
#: seeds the baseline from the first batch's mean score; a number seeds
#: it directly.  Module-level so tests can inject a fixed baseline and
#: observe its effect on the run (the seam that proved the baseline was
#: once mathematically inert -- see ``_BASELINE_INIT``-titled regression
#: test in tests/test_generative.py).
_BASELINE_INIT: float | None = None

#: Decay of the baseline / return-std exponential moving averages (the
#: classic 0.9/0.1 smoothing; the gradient sees statistics from previous
#: batches only).
_BASELINE_DECAY = 0.9


def _clamped_score(objective: Any, smiles: str) -> float:
    """Score ``smiles`` clamped to ``[0, 1]`` (NaN/inf -> 0.0).

    A non-numeric return value raises the same typed error as
    :class:`~pharmaml.generative.MultiObjective` -- the objective contract
    (``score(smiles) -> float``) is enforced consistently everywhere.
    """
    try:
        value = float(objective.score(smiles))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"objective {getattr(objective, 'name', objective)!r} returned a "
            f"non-numeric score for {smiles!r} -- objectives must map SMILES "
            "to floats"
        ) from exc
    if not math.isfinite(value):
        return 0.0
    return min(max(value, 0.0), 1.0)


class ReinventOptimizer:
    """Policy-gradient fine-tuning of a SMILES RNN prior.

    Parameters
    ----------
    objective:
        Anything with ``score(smiles) -> float`` (any
        :class:`~pharmaml.generative.ScoringObjective`).
    prior:
        A **trained** :class:`~pharmaml.generative.SmilesRnn`.  ``None``
        (default) trains a fresh prior from ``corpus`` inside
        :meth:`optimize`.
    corpus:
        Training molecules for the on-the-fly prior (default:
        :data:`~pharmaml.generative.DEFAULT_CORPUS`); ignored when
        ``prior`` is given.  A bare string is rejected (it would be
        iterated as single-atom characters).
    iterations:
        Maximum RL iterations (default 30).
    batch_size:
        Molecules sampled per iteration (default 32).
    temperature:
        Sampling temperature of the agent (default 0.9).
    prior_epochs:
        Epochs used when training the on-the-fly prior (default 8).
    lr:
        Adam learning rate of the agent (default 0.003).
    anchor_weight:
        Strength of the squared-drift prior anchor (default 0.1; 0
        disables it -- vanilla REINFORCE with baseline).
    entropy_bonus:
        Entropy regularisation strength (default 0.02).
    replay_size:
        Capacity of the experience-replay buffer (default 64).
    patience:
        Early-stop patience: stop when the best mean score has not
        improved by ``tol`` for this many iterations (default 5).
    tol:
        Minimum mean-score improvement that resets ``patience``.
    grad_clip:
        Gradient-norm clip (default 1.0).
    max_smiles_length:
        Hard length cap for sampled molecules (default 120).
    seed:
        Master seed (default 42).  Two runs with the same seed, corpus
        and prior produce identical results.  Must lie in
        ``[-2**63, 2**64 - 1]`` (the interval torch's RNG can encode).

    Raises
    ------
    DataValidationError
        On out-of-range hyperparameters, non-finite float
        hyperparameters (NaN slips sign-only guards), a ``seed`` outside
        the torch-encodable interval, a bare-string ``corpus`` (it would
        be iterated as single-atom characters), or an explicitly passed
        ``prior`` that is untrained.
    DependencyError
        When torch is not installed (at :meth:`optimize` time).
    """

    def __init__(
        self,
        objective: Any,
        *,
        prior: SmilesRnn | None = None,
        corpus: Sequence[str] | None = None,
        iterations: int = 30,
        batch_size: int = 32,
        temperature: float = 0.9,
        prior_epochs: int = 8,
        lr: float = 0.003,
        anchor_weight: float = 0.1,
        entropy_bonus: float = 0.02,
        replay_size: int = 64,
        patience: int = 5,
        tol: float = 1e-3,
        grad_clip: float = 1.0,
        max_smiles_length: int = 120,
        seed: int = 42,
    ) -> None:
        if not callable(getattr(objective, "score", None)):
            raise DataValidationError(
                "objective must expose a callable score(smiles) method"
            )
        if iterations < 1:
            raise DataValidationError(f"iterations must be >= 1, got {iterations}")
        if batch_size < 4:
            raise DataValidationError(f"batch_size must be >= 4, got {batch_size}")
        if not 0 < temperature <= 5.0:
            raise DataValidationError(
                f"temperature must be in (0, 5], got {temperature}"
            )
        if prior_epochs < 1:
            raise DataValidationError(f"prior_epochs must be >= 1, got {prior_epochs}")
        # sign-only guards let NaN through (NaN compares False to everything)
        # and inf past the upper-unbounded ones: empirically lr=inf NaN-ed
        # the agent in one SGD step, anchor_weight=NaN silently disabled the
        # anchor, tol=NaN made every iteration look non-improving.  Every
        # float hyperparameter must be finite.
        for name, value in (
            ("lr", lr),
            ("anchor_weight", anchor_weight),
            ("entropy_bonus", entropy_bonus),
            ("tol", tol),
            ("grad_clip", grad_clip),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise DataValidationError(
                    f"{name} must be a finite number, got {value!r}"
                )
        if lr <= 0:
            raise DataValidationError(f"lr must be positive, got {lr!r}")
        if anchor_weight < 0:
            raise DataValidationError(
                f"anchor_weight must be >= 0, got {anchor_weight!r}"
            )
        if entropy_bonus < 0:
            raise DataValidationError(
                f"entropy_bonus must be >= 0, got {entropy_bonus!r}"
            )
        if replay_size < 0:
            raise DataValidationError(f"replay_size must be >= 0, got {replay_size}")
        if patience < 1:
            raise DataValidationError(f"patience must be >= 1, got {patience}")
        if tol < 0:
            raise DataValidationError(f"tol must be >= 0, got {tol!r}")
        if grad_clip <= 0:
            raise DataValidationError(f"grad_clip must be positive, got {grad_clip!r}")
        if max_smiles_length < 20:
            raise DataValidationError(
                f"max_smiles_length must be >= 20, got {max_smiles_length}"
            )
        _validate_torch_seed(seed, "seed")
        if corpus is not None and isinstance(corpus, str):
            # iterating a bare string would treat CHARACTERS as molecules
            # ("CCO" -> a corpus of C, C, O atoms) -- the NoveltyObjective
            # guard, same typed DataValidationError.
            raise DataValidationError(
                "corpus must be a sequence of SMILES strings, not a single "
                f"string (got {corpus!r} of length {len(corpus)})"
            )
        self.objective = objective
        self.prior = prior
        self.corpus = list(corpus) if corpus is not None else None
        self.iterations = iterations
        self.batch_size = batch_size
        self.temperature = temperature
        self.prior_epochs = prior_epochs
        self.lr = lr
        self.anchor_weight = anchor_weight
        self.entropy_bonus = entropy_bonus
        self.replay_size = replay_size
        self.patience = patience
        self.tol = tol
        self.grad_clip = grad_clip
        self.max_smiles_length = max_smiles_length
        self.seed = seed

    # -- public API ------------------------------------------------------
    def optimize(self) -> GenerationResult:
        """Run the fine-tuning loop; returns the ranked molecules.

        Raises
        ------
        DataValidationError
            If an explicit ``prior`` is untrained, or the corpus (used only
            when no prior is given) holds no parseable molecule.
        DependencyError
            If torch is not installed.
        """
        torch = _require_torch()
        started = time.perf_counter()

        if self.prior is not None:
            if not self.prior.is_trained:
                raise DataValidationError(
                    "the provided prior is untrained -- call prior.train_prior() "
                    "first (or pass prior=None to train one from the corpus)"
                )
            prior = self.prior
            prior_source = "user prior"
        else:
            corpus = list(self.corpus) if self.corpus is not None else list(DEFAULT_CORPUS)
            # canonicalize + drop unparseable entries: the prior should learn
            # the corpus's chemistry, not its typos (invalid strings would
            # otherwise be trained on verbatim).  All-valid corpora are
            # untouched by this pass.
            canonical_corpus = [
                canonical
                for canonical in (
                    canonical_smiles(smiles) if isinstance(smiles, str) else None
                    for smiles in corpus
                )
                if canonical is not None
            ]
            if not canonical_corpus:
                raise DataValidationError(
                    "corpus is empty -- nothing to train the prior on"
                )
            if len(canonical_corpus) < len(corpus):
                _LOG.warning(
                    "dropping %d unparseable corpus entr(ies) before prior training",
                    len(corpus) - len(canonical_corpus),
                )
            prior = SmilesRnn(seed=self.seed)
            losses = prior.train_prior(canonical_corpus, epochs=self.prior_epochs)
            prior_source = (
                f"corpus prior ({len(canonical_corpus)} molecules, {len(losses)} epochs)"
            )
            _LOG.info("trained %s, final loss %.4f", prior_source, losses[-1])

        agent = prior.copy()
        agent._model.train()
        optimizer = torch.optim.Adam(agent._model.parameters(), lr=self.lr)

        # REINFORCE running statistics: the baseline is the EMA of
        # PREVIOUS batches' mean scores, the normaliser the EMA of their
        # return spread.  ``None`` seeds each from the first batch (the
        # ``_BASELINE_INIT`` seam lets tests inject a fixed baseline).
        baseline: float | None = _BASELINE_INIT
        std_ema: float | None = None
        best_mean = -1.0
        stale = 0
        stopped_early = False
        replay: dict[str, float] = {}
        produced: dict[str, float] = {}
        history: list[dict[str, Any]] = []
        n_sampled = 0
        n_valid = 0

        for iteration in range(self.iterations):
            # deterministic per-iteration sampling seed: independent of any
            # sampling the caller did with the prior before the run
            sample_seed = (self.seed * 1_000_003 + iteration) % (2**31)
            raw = agent.sample(
                self.batch_size,
                temperature=self.temperature,
                seed=sample_seed,
                max_length=self.max_smiles_length,
            )
            n_sampled += len(raw)
            scores = [_clamped_score(self.objective, smiles) for smiles in raw]
            canonical = [canonical_smiles(smiles) for smiles in raw]
            for smiles, canonical_form, score in zip(raw, canonical, scores):
                if canonical_form is not None:
                    n_valid += 1
                    produced[canonical_form] = max(
                        produced.get(canonical_form, 0.0), score
                    )
                    if score > 0:
                        replay[canonical_form] = max(
                            replay.get(canonical_form, 0.0), score
                        )

            mean_score = sum(scores) / len(scores)

            # experience replay: keep the best unique samples, mix them in
            replay = dict(
                sorted(replay.items(), key=lambda kv: (-kv[1], kv[0]))[: self.replay_size]
            )
            train_batch = list(zip(raw, scores))
            train_batch += list(replay.items())[: self.batch_size // 2]

            loss_value = float("nan")
            if train_batch:
                sequences = [agent._encoded(smiles) for smiles, _ in train_batch]
                returns = torch.tensor([score for _, score in train_batch])
                with torch.no_grad():
                    prior_lp, _ = prior._sequence_logprob_and_entropy(
                        torch, sequences
                    )
                agent_lp, entropy = agent._sequence_logprob_and_entropy(
                    torch, sequences
                )
                # Advantage = (returns - baseline) / running std.  NO
                # per-batch mean subtraction: subtracting the batch mean
                # would cancel the baseline exactly ((r - b) -
                # mean(r - b) = r - mean(r) for ANY b), leaving the
                # advertised running baseline mathematically inert.
                batch_std = float(returns.std())
                if baseline is None:
                    baseline = mean_score  # seed: no previous batch yet
                if std_ema is None:
                    std_ema = batch_std
                advantage = (returns - baseline) / (std_ema + 1e-6)
                anchor_loss = ((agent_lp - prior_lp) ** 2).mean()
                loss = -(advantage * agent_lp).mean() - self.entropy_bonus * entropy
                if self.anchor_weight > 0:
                    loss = loss + self.anchor_weight * anchor_loss
                agent._apply_sgd_step(torch, optimizer, loss, self.grad_clip)
                loss_value = float(loss.detach())
                # Fold the current batch into the running statistics only
                # AFTER the gradient: this iteration's advantage saw
                # previous batches, the next one's sees this one too.
                baseline = _BASELINE_DECAY * baseline + (
                    1.0 - _BASELINE_DECAY
                ) * mean_score
                std_ema = _BASELINE_DECAY * std_ema + (
                    1.0 - _BASELINE_DECAY
                ) * batch_std

            best_index = scores.index(max(scores))
            history.append(
                {
                    "iteration": iteration,
                    "best_score": max(scores),
                    "mean_score": mean_score,
                    "best_smiles": canonical[best_index] or raw[best_index],
                    "valid_fraction": sum(c is not None for c in canonical)
                    / len(canonical),
                    "replay_size": len(replay),
                    "loss": loss_value,
                }
            )
            _LOG.debug(
                "rl iteration %d: mean %.3f best %.3f valid %.2f",
                iteration,
                mean_score,
                max(scores),
                history[-1]["valid_fraction"],
            )

            # early stop on mean-score plateau
            if mean_score > best_mean + self.tol:
                best_mean = mean_score
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    stopped_early = True
                    _LOG.info(
                        "mean score plateaued at %.3f -- stopping after %d "
                        "iteration(s)",
                        best_mean,
                        iteration + 1,
                    )
                    break

        ranked = sorted(produced.items(), key=lambda kv: (-kv[1], kv[0]))
        smiles_list = [smiles for smiles, _ in ranked]
        score_list = [score for _, score in ranked]
        breakdown: list[dict[str, float]] = []
        score_with_breakdown = getattr(self.objective, "score_with_breakdown", None)
        for smiles in smiles_list:
            if callable(score_with_breakdown):
                breakdown.append(score_with_breakdown(smiles)[1])
            else:
                breakdown.append({})
        elapsed = time.perf_counter() - started
        return GenerationResult(
            smiles=smiles_list,
            scores=score_list,
            breakdown=breakdown,
            best=smiles_list[0] if smiles_list else "",
            best_score=score_list[0] if score_list else 0.0,
            history=history,
            metadata={
                "method": "reinforce_rl",
                "seed": self.seed,
                "iterations": len(history),
                "elapsed_s": elapsed,
                "valid_generated": n_valid,
                "unique_generated": len(smiles_list),
                "sampled": n_sampled,
                "early_stopped": stopped_early,
                "prior": prior_source,
                "batch_size": self.batch_size,
                "anchor_weight": self.anchor_weight,
            },
        )


def run_rl_optimization(
    objective: Any,
    *,
    corpus: Sequence[str] | None = None,
    prior: SmilesRnn | None = None,
    prior_epochs: int = 8,
    **kwargs: Any,
) -> GenerationResult:
    """One-call RL optimisation (trains a prior from the corpus if needed).

    Example
    -------
    >>> from pharmaml.generative import QedObjective, run_rl_optimization
    >>> result = run_rl_optimization(QedObjective(), iterations=5, seed=1)
    >>> result.metadata["method"]
    'reinforce_rl'
    """
    optimizer = ReinventOptimizer(
        objective,
        prior=prior,
        corpus=corpus,
        prior_epochs=prior_epochs,
        **kwargs,
    )
    return optimizer.optimize()
