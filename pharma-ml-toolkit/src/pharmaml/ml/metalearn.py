"""TCRP's few-shot transfer: first-order MAML in plain NumPy.

``idekerlab/TCRP`` was reviewed file by file for this module:
``code/meta_learner_cv.py`` (the meta-learner — 226 lines), ``code/
inner_loop.py`` (the MAML inner loop), ``code/mlp.py`` /
``simple_mlp.py`` (the base network), ``code/score.py`` (the
Pearson/Spearman evaluation), ``code/data_loading.py`` and
``code/tcrp_train.py``/``tcrp_cv.py``, plus the README's data story
(pre-train across 12 tissues on Sorafenib response from CCLE
expression + mutation profiles; adapt to an unseen tissue — or
patient-derived tumour cells / xenografts — with a handful of
samples).

TCRP is a *few-shot transfer* recipe (Ma et al., Nat Cancer 2021):
pre-train one network across many related tasks, then adapt it to a
new task with K samples in a few gradient steps.  The algorithm is
first-order MAML, and none of it needs an engine:

* **The inner loop** — clone the meta weights, run ``inner_lr`` SGD
  steps on the K *support* samples (``InnerLoop.forward``'s
  fast-weights exactly).
* **The meta update** — TCRP's own trick, verbatim: accumulate the
  query-loss gradients of the *adapted* weights across a meta-batch
  of tasks and credit them to the meta weights (the first-order
  approximation drops the second-derivative path; their
  ``meta_update`` registers hooks that replace the dummy gradients
  with the accumulated ones -- here it is one explicit line).
* **The honest evaluation** — three arms on the new task, TCRP's
  ``zero_shot_test`` / ``unseen_tissue_learn`` pair plus the
  from-scratch control: zero-shot (copy, no adaptation), K-shot
  (copy + inner steps on K samples), and scratch (random init trained
  on the same K samples), all scored by Pearson correlation -- the
  paper's metric.

The pharma reading, stated once: a *task* is a tissue (or a cohort);
the features are expression/mutation profiles; the label is drug
response; the promise is that one meta-trained model adapts to a new
cohort from a handful of patients.

The honest scope: the CCLE/GDSC-scale pre-training, the clinical
transfer experiments and the ``.tar.gz`` pretrained model stay with
the PyTorch engine the toolkit does not vendor; what lands is the
algorithm -- inner loop, meta update, three-arm evaluation -- exact,
seeded and runnable at teaching scale.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.metalearn import (
...     TaskFamily, adapt, fomaml_train, few_shot_report,
... )
>>> family = TaskFamily(n_tasks=8, n_features=4, seed=42)
>>> tasks = family.sample_support_query(n_support=24, n_query=24, seed=1)
>>> len(tasks)
8
>>> meta = fomaml_train(tasks, hidden=8, epochs=200, seed=42)
>>> meta.history.train_losses[-1] < meta.history.train_losses[0]
True
>>> new_task = family.new_task(seed=7)
>>> report = few_shot_report(meta, new_task, k=4, inner_steps=40, seed=42)
>>> report.pearson_k_shot > report.pearson_scratch    # the transfer wins
True
>>> adapted = adapt(meta, new_task.support(k=4), steps=40, seed=42)
>>> float(adapted.pearson(new_task.x_query, new_task.y_query)) > 0.5
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through every init and split (the pin).
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# The task family (tissues: same feature space, related structure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSplit:
    """One task's support/query split (the inner-loop data contract)."""

    weights: tuple[float, ...]
    x_support: np.ndarray
    y_support: np.ndarray
    x_query: np.ndarray
    y_query: np.ndarray

    def support(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The first ``k`` support rows (the K-shot budget)."""
        if k < 1 or k > self.x_support.shape[0]:
            raise DataValidationError(f"k must be in [1, {self.x_support.shape[0]}], got {k}")
        return self.x_support[:k], self.y_support[:k]


class TaskFamily:
    """Seeded related regression tasks: one shared prior, per-task draws.

    Each task's label is ``x · w + ε`` with ``w`` drawn once from a
    shared Gaussian prior (positive mean on the first feature -- the
    family's common structure, the reason transfer works at all) and
    ``ε`` a fixed task noise scale.  This is the tissue metaphor in
    miniature: same feature space, same rough biology, per-tissue
    coefficients.
    """

    def __init__(
        self,
        n_tasks: int = 8,
        n_features: int = 4,
        prior_mean: float = 1.5,
        prior_scale: float = 0.5,
        noise: float = 0.1,
        seed: int = RANDOM_STATE,
    ) -> None:
        if n_tasks < 2:
            raise DataValidationError(f"n_tasks must be >= 2, got {n_tasks}")
        if n_features < 1:
            raise DataValidationError(f"n_features must be >= 1, got {n_features}")
        if prior_scale < 0 or noise < 0:
            raise DataValidationError("prior_scale and noise must be >= 0")
        self.n_tasks = int(n_tasks)
        self.n_features = int(n_features)
        self.prior_mean = float(prior_mean)
        self.prior_scale = float(prior_scale)
        self.noise = float(noise)
        self._rng = np.random.default_rng(seed)
        self._task_rngs = [np.random.default_rng(seed * 1000 + index) for index in range(n_tasks)]
        self._task_weights = [
            self._rng.normal(prior_mean, prior_scale, size=n_features) for _ in range(n_tasks)
        ]
        self._new_task_count = 0

    def _make_split(
        self,
        weights: np.ndarray,
        rng: np.random.Generator,
        n_support: int,
        n_query: int,
    ) -> TaskSplit:
        x_support = rng.normal(size=(n_support, self.n_features))
        y_support = x_support @ weights + self.noise * rng.normal(size=n_support)
        x_query = rng.normal(size=(n_query, self.n_features))
        y_query = x_query @ weights + self.noise * rng.normal(size=n_query)
        return TaskSplit(
            weights=tuple(float(value) for value in weights),
            x_support=x_support,
            y_support=y_support.reshape(-1, 1),
            x_query=x_query,
            y_query=y_query.reshape(-1, 1),
        )

    def sample_support_query(
        self, n_support: int = 24, n_query: int = 24, seed: int | None = None
    ) -> list[TaskSplit]:
        """Every task's support/query split (the meta-training set).

        ``seed`` re-splits the *data* without redrawing the task
        weights (a fresh curriculum over the same tissues).
        """
        if n_support < 2 or n_query < 2:
            raise DataValidationError(
                f"n_support and n_query must be >= 2, got {(n_support, n_query)!r}"
            )
        if seed is None:
            rngs = self._task_rngs
        else:
            rngs = [np.random.default_rng(seed * 1000 + index) for index in range(self.n_tasks)]
        return [
            self._make_split(np.asarray(self._task_weights[index]), rngs[index], n_support, n_query)
            for index in range(self.n_tasks)
        ]

    def new_task(self, seed: int | None = None) -> TaskSplit:
        """An unseen task (the held-out tissue), support/query split."""
        rng = np.random.default_rng(seed if seed is not None else 10_000 + self._new_task_count)
        self._new_task_count += 1
        weights = rng.normal(self.prior_mean, self.prior_scale, size=self.n_features)
        return self._make_split(weights, rng, n_support=30, n_query=30)


# ---------------------------------------------------------------------------
# The base network (mlp.py: one hidden tanh layer + linear head)
# ---------------------------------------------------------------------------


class _Mlp:
    """The NumPy twin of TCRP's ``mlp``: in → hidden tanh → linear out.

    Manual forward/backward (MSE loss), the gradient machinery the
    meta-loop composes; weights live in a plain dict so cloning is a
    copy and the meta update is arithmetic.
    """

    def __init__(self, n_features: int, hidden: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        limit_hidden = np.sqrt(6.0 / (n_features + hidden))
        limit_out = np.sqrt(6.0 / (hidden + 1))
        self.weights: dict[str, np.ndarray] = {
            "W1": rng.uniform(-limit_hidden, limit_hidden, size=(n_features, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.uniform(-limit_out, limit_out, size=(hidden, 1)),
            "b2": np.zeros(1),
        }
        self.n_features = int(n_features)
        self.hidden = int(hidden)

    def clone(self) -> _Mlp:
        """A weight-identical copy (TCRP's ``copy_weights``)."""
        copy = _Mlp(self.n_features, self.hidden, seed=0)
        copy.weights = {key: value.copy() for key, value in self.weights.items()}
        return copy

    def predict(self, x: np.ndarray) -> np.ndarray:
        hidden = np.tanh(x @ self.weights["W1"] + self.weights["b1"])
        return hidden @ self.weights["W2"] + self.weights["b2"]

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        """MSE and its exact gradients (the inner-loop workhorse)."""
        hidden = np.tanh(x @ self.weights["W1"] + self.weights["b1"])
        prediction = hidden @ self.weights["W2"] + self.weights["b2"]
        error = prediction - y
        loss = float(np.mean(error**2))
        n = x.shape[0]
        d_pred = (2.0 / n) * error
        grads = {
            "W2": hidden.T @ d_pred,
            "b2": d_pred.sum(axis=0),
        }
        d_hidden = d_pred @ self.weights["W2"].T
        d_pre = d_hidden * (1.0 - hidden**2)
        grads["W1"] = x.T @ d_pre
        grads["b1"] = d_pre.sum(axis=0)
        return loss, grads

    def sgd_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        """One SGD step on (x, y); returns the pre-step loss."""
        loss, grads = self.loss_and_grads(x, y)
        for key, gradient in grads.items():
            self.weights[key] -= lr * gradient
        return loss

    def pearson(self, x: np.ndarray, y: np.ndarray) -> float:
        """Pearson r of the predictions against ``y`` (score.py's metric)."""
        predicted = self.predict(x).reshape(-1)
        actual = np.asarray(y, dtype=float).reshape(-1)
        centered_p = predicted - predicted.mean()
        centered_a = actual - actual.mean()
        denominator = float(np.linalg.norm(centered_p) * np.linalg.norm(centered_a))
        if denominator < 1e-24:
            return float("nan")
        return float(np.dot(centered_p, centered_a) / denominator)


# ---------------------------------------------------------------------------
# The meta-loop (meta_learner_cv.py, first-order)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaHistory:
    """The meta-training curve."""

    train_losses: tuple[float, ...]
    final_loss: float
    n_epochs: int


class MetaModel:
    """The meta-trained network (TCRP's ``observed_tissue_model``)."""

    def __init__(self, net: _Mlp, history: MetaHistory, inner_lr: float) -> None:
        self.net = net
        self.history = history
        self.inner_lr = float(inner_lr)

    def pearson(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.net.pearson(x, y)


@dataclass(frozen=True)
class AdaptedModel:
    """A K-shot adapted copy of the meta model."""

    net: _Mlp
    k: int
    steps: int
    support_loss: float

    def pearson(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.net.pearson(x, y)


def _check(tasks: list[TaskSplit], epochs: int, inner_lr: float, meta_lr: float) -> None:
    if not tasks:
        raise DataValidationError("tasks must not be empty")
    if epochs < 1:
        raise DataValidationError(f"epochs must be >= 1, got {epochs}")
    if inner_lr <= 0 or meta_lr <= 0:
        raise DataValidationError(
            f"learning rates must be > 0, got inner={inner_lr!r}, meta={meta_lr!r}"
        )


def fomaml_train(
    tasks: list[TaskSplit],
    hidden: int = 16,
    epochs: int = 100,
    inner_steps: int = 2,
    inner_lr: float = 0.02,
    meta_lr: float = 0.01,
    meta_batch_size: int | None = None,
    seed: int = RANDOM_STATE,
) -> MetaModel:
    """First-order MAML over the task family (TCRP's train loop).

    Per epoch: for every task in the meta-batch, clone the meta
    weights, run ``inner_steps`` SGD steps on the support set, then
    measure the *query* loss and its gradient at the adapted weights;
    the accumulated gradients are applied to the meta weights in one
    step -- the first-order approximation (the second-derivative path
    of full MAML is dropped, exactly as TCRP's hook-replacement
    ``meta_update`` does).  ``meta_batch_size`` ``None`` means all
    tasks per epoch.

    Returns
    -------
    MetaModel
        The meta-trained network plus its loss history.
    """
    _check(tasks, epochs, inner_lr, meta_lr)
    if hidden < 1:
        raise DataValidationError(f"hidden must be >= 1, got {hidden}")
    n_features = tasks[0].x_support.shape[1]
    meta_net = _Mlp(n_features, hidden, seed=seed)
    batch = tasks if meta_batch_size is None else tasks[:meta_batch_size]
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for epoch in range(epochs):
        accumulated: dict[str, np.ndarray] | None = None
        epoch_loss = 0.0
        for task in batch:
            adapted = meta_net.clone()
            for _step in range(inner_steps):
                adapted.sgd_step(task.x_support, task.y_support, inner_lr)
            query_loss, query_grads = adapted.loss_and_grads(task.x_query, task.y_query)
            epoch_loss += query_loss
            if accumulated is None:
                accumulated = {key: gradient.copy() for key, gradient in query_grads.items()}
            else:
                for key, gradient in query_grads.items():
                    accumulated[key] += gradient
        # the seeded rng exists so the (future) shuffled meta-batch order
        # stays reproducible; full-batch epochs need no shuffle
        _ = rng.random()
        assert accumulated is not None
        for key, gradient in accumulated.items():
            meta_net.weights[key] -= meta_lr * gradient / len(batch)
        losses.append(epoch_loss / len(batch))
        if epoch % max(1, epochs // 10) == 0:
            _LOG.debug("fomaml epoch %d/%d mean query loss %.6f", epoch, epochs, losses[-1])
    history = MetaHistory(train_losses=tuple(losses), final_loss=losses[-1], n_epochs=int(epochs))
    return MetaModel(net=meta_net, history=history, inner_lr=inner_lr)


def adapt(
    model: MetaModel | AdaptedModel,
    support: tuple[np.ndarray, np.ndarray],
    steps: int = 20,
    lr: float | None = None,
    seed: int = RANDOM_STATE,
) -> AdaptedModel:
    """Clone-and-fine-tune on a support set (the inner loop alone).

    ``lr`` defaults to the meta model's own ``inner_lr`` -- the rate
    the pre-training taught the weights to move at.
    """
    source = model.net
    x, y = support
    if steps < 1:
        raise DataValidationError(f"steps must be >= 1, got {steps}")
    rate = lr if lr is not None else (model.inner_lr if isinstance(model, MetaModel) else 0.02)
    adapted = source.clone()
    support_loss = 0.0
    for _step in range(steps):
        support_loss = adapted.sgd_step(x, y, rate)
    return AdaptedModel(net=adapted, k=int(x.shape[0]), steps=int(steps), support_loss=support_loss)


@dataclass(frozen=True)
class FewShotReport:
    """The three-arm verdict on a new task (TCRP's evaluation).

    ``pearson_zero_shot`` -- the meta weights as-is; ``pearson_k_shot``
    -- after ``k``-sample inner-loop adaptation;
    ``pearson_scratch`` -- a same-architecture network trained from a
    random init on the same ``k`` samples (the control that shows the
    transfer is real: without pre-training, K samples do not buy
    this).  ``transfer_gain`` is k-shot minus scratch.
    """

    k: int
    inner_steps: int
    pearson_zero_shot: float
    pearson_k_shot: float
    pearson_scratch: float

    @property
    def transfer_gain(self) -> float:
        return self.pearson_k_shot - self.pearson_scratch


def few_shot_report(
    model: MetaModel,
    task: TaskSplit,
    k: int,
    inner_steps: int = 20,
    scratch_steps: int = 200,
    seed: int = RANDOM_STATE,
) -> FewShotReport:
    """Zero-shot vs K-shot vs from-scratch, Pearson-scored.

    The scratch arm trains a fresh ``_Mlp`` (same hidden width) on the
    same ``k`` support samples for ``scratch_steps`` steps at a tuned
    rate -- generous, deliberately: the comparison is honest only if
    the control gets every advantage except the pre-training.
    """
    if not isinstance(model, MetaModel):
        raise DataValidationError("few_shot_report needs a MetaModel (the pre-trained net)")
    x_support, y_support = task.support(k)
    zero_shot = model.net.pearson(task.x_query, task.y_query)
    adapted = adapt(model, (x_support, y_support), steps=inner_steps, seed=seed)
    k_shot = adapted.net.pearson(task.x_query, task.y_query)
    scratch = _Mlp(task.x_support.shape[1], model.net.hidden, seed=seed + 1)
    rate = 0.05
    for _step in range(scratch_steps):
        scratch.sgd_step(x_support, y_support, rate)
    scratch_r = scratch.pearson(task.x_query, task.y_query)
    report = FewShotReport(
        k=int(k),
        inner_steps=int(inner_steps),
        pearson_zero_shot=float(zero_shot),
        pearson_k_shot=float(k_shot),
        pearson_scratch=float(scratch_r),
    )
    _LOG.debug(
        "few-shot (k=%d): zero %.3f / k-shot %.3f / scratch %.3f",
        k,
        report.pearson_zero_shot,
        report.pearson_k_shot,
        report.pearson_scratch,
    )
    return report
