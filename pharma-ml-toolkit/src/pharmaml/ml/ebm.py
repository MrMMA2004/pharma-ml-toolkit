"""Explainable Boosting Machines — the interpret EBM concept on numpy.

Absorbed in round 100x-174 from **interpretml/interpret** (v0.7.x,
reviewed file by file): the glass-box model that cyclically boosts
*one feature at a time* into piecewise term functions, keeping the
additive form ``f(x) = intercept + Σ_j f_j(x_j) + Σ_{jk} f_{jk}(x_j,
x_k)`` directly readable.  What was transcribed:

* the **quantile binning** of ``_preprocessor.py`` (equal-density
  cuts, never inside a run of equal values — ``np.unique`` here);
* the **Newton leaf update** of ``ebm_stats.hpp`` /
  ``GenerateTermUpdate.cpp`` — per boosting step the term's bins are
  partitioned by the single best split (``max_leaves = 2``), and each
  side receives ``Δ = −learning_rate·Σg/(Σh + λ)`` with the binary
  log-loss gradients ``g = σ(s) − y``, ``h = p(1−p)`` and the RMSE
  constants normalised to ``g = s − y``, ``h = 1`` (the report's
  2/2-cancellation);
* the split gain ``G₁²/(H₁+λ) + G₂²/(H₂+λ) − G²/(H+λ)``;
* the closed-form intercepts (RMSE: the mean of y; log-loss: the
  class base-rate log-odds) and the post-fit **re-centering** that
  moves each term's weighted mean into the intercept so the term
  graphs are zero-mean (``_utils.py::process_bag_terms``);
* the ``avg_weight`` term importance — the training-weight-weighted
  mean of ``|term scores|``;
* the learning-rate defaults (0.04 regression / 0.015 classification).

Honest boundaries: the 0.7.x engine's greedy steps (``greedy_ratio``
10), smoothing rounds (random splits), 14 outer bags, inner bags,
early stopping and the FAST pairwise-interaction ranking (the 2×2
quadrant gain sweep) are not ported — pairs are fit only when
explicitly requested, without split search on the pair tensor; the
missing/unseen sentinel bins are folded into the ordinary bins.
``max_bins`` defaults to 64 rather than 1024 to keep the term graphs
readable at playground scale.

Examples
--------
>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> X = rng.normal(size=(400, 3))
>>> y = 2.0 * X[:, 0] + (X[:, 1] ** 2 - 0.5)          # additive truth
>>> ebm = ExplainableBoostingModel(loss='rmse', learning_rate=0.05,
...                                max_rounds=150, random_state=0)
>>> _ = ebm.fit(X, y)
>>> scores = ebm.predict(X)
>>> ss = 1.0 - ((y - scores) ** 2).sum() / ((y - y.mean()) ** 2).sum()
>>> bool(ss > 0.9)                                     # additive recovery
True
>>> noise = ebm.term_importance()['x3'] / ebm.term_importance()['x1']
>>> bool(noise < 0.25)                                 # the noise term stays flat
True
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = ["ExplainableBoostingModel"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


class ExplainableBoostingModel:
    """A cyclically-boosted additive model — interpret's EBM, lightly.

    Parameters
    ----------
    loss:
        ``'rmse'`` (regression) or ``'log_loss'`` (binary
        classification — ``predict_proba`` returns the sigmoid).
    max_bins:
        Quantile bins per feature (EBM's 1024 is trimmed to 64 for
        readable term graphs).
    learning_rate:
        Per-step Newton shrinkage — the interpret defaults are 0.04
        (rmse) and 0.015 (log_loss).
    max_rounds:
        Cyclic passes over every term.
    reg_lambda:
        The Newton leaf's ``λ`` (0 in interpret's defaults).
    pairs:
        Explicit ``(i, j)`` index pairs to fit as 2-D terms.
    random_state:
        Seeds the round-order permutation (interpret's cyclic order is
        itself randomized).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(4)
    >>> X = rng.normal(size=(300, 2))
    >>> p = 1.0 / (1.0 + np.exp(-(3.0 * X[:, 0] + 2.0 * (X[:, 1] > 0))))
    >>> y = (rng.uniform(size=300) < p).astype(int)
    >>> ebm = ExplainableBoostingModel(loss='log_loss', learning_rate=0.05,
    ...                                max_rounds=120, random_state=1)
    >>> _ = ebm.fit(X, y)
    >>> from sklearn.metrics import roc_auc_score
    >>> auc = roc_auc_score(y, ebm.predict_proba(X))
    >>> bool(auc > 0.9)
    True
    """

    def __init__(
        self,
        loss: str = "rmse",
        *,
        max_bins: int = 64,
        learning_rate: float | None = None,
        max_rounds: int = 200,
        reg_lambda: float = 0.0,
        pairs: tuple[tuple[int, int], ...] = (),
        random_state: int = 42,
    ) -> None:
        if loss not in ("rmse", "log_loss"):
            raise ValueError("loss must be 'rmse' or 'log_loss'")
        if max_bins < 4:
            raise ValueError("max_bins must be >= 4")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.loss = loss
        self.max_bins = int(max_bins)
        self.learning_rate = (
            learning_rate if learning_rate is not None
            else (0.04 if loss == "rmse" else 0.015)
        )
        self.max_rounds = int(max_rounds)
        self.reg_lambda = float(reg_lambda)
        self.pairs = tuple(tuple(sorted(pair)) for pair in pairs)
        self.random_state = int(random_state)
        self.cuts_: list[np.ndarray] = []
        self.term_scores_: dict[int, np.ndarray] = {}
        self.pair_scores_: dict[tuple[int, int], np.ndarray] = {}
        self.intercept_: float = 0.0
        self.bin_weights_: dict[int, np.ndarray] = {}
        self.pair_weights_: dict[tuple[int, int], np.ndarray] = {}
        self.n_features_: int = 0

    # -- binning -----------------------------------------------------------
    def _bin_columns(self, X: np.ndarray) -> list[np.ndarray]:
        """Quantile cuts per column (never inside equal-value runs)."""
        binned: list[np.ndarray] = []
        self.cuts_ = []
        for j in range(X.shape[1]):
            quantiles = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
            cuts = np.unique(np.quantile(X[:, j], quantiles))
            self.cuts_.append(cuts)
            binned.append(np.asarray(np.searchsorted(cuts, X[:, j]), dtype=int))
        return binned

    # -- fitting -----------------------------------------------------------
    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> ExplainableBoostingModel:
        """Cyclically boost every term's bin scores (the EBM core)."""
        X = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if target.shape[0] != X.shape[0]:
            raise ValueError("X and y lengths differ")
        if self.loss == "log_loss" and not np.all(np.isin(np.unique(target), (0.0, 1.0))):
            raise ValueError("log_loss needs binary 0/1 targets")
        self.n_features_ = X.shape[1]
        for a, b in self.pairs:
            if not (0 <= a < X.shape[1] and 0 <= b < X.shape[1]) or a == b:
                raise ValueError(f"invalid pair {(a, b)}")
        binned = self._bin_columns(X)
        n_bins = [len(self.cuts_[j]) + 1 for j in range(X.shape[1])]

        # closed-form intercepts (the _boost.py special cases)
        if self.loss == "rmse":
            self.intercept_ = float(target.mean())
        else:
            rate = float(np.clip(target.mean(), 1e-6, 1 - 1e-6))
            self.intercept_ = math.log(rate / (1.0 - rate))
        self.term_scores_ = {j: np.zeros(n_bins[j]) for j in range(X.shape[1])}
        self.pair_scores_ = {}
        self.bin_weights_ = {}
        self.pair_weights_ = {}
        for j in range(X.shape[1]):
            counts = np.bincount(binned[j], minlength=n_bins[j]).astype(float)
            self.bin_weights_[j] = counts / counts.sum()
        for a, b in self.pairs:
            grid = n_bins[a] * n_bins[b]
            joint = binned[a] * n_bins[b] + binned[b]
            counts = np.bincount(joint, minlength=grid).astype(float)
            self.pair_weights_[(a, b)] = counts / counts.sum()
            self.pair_scores_[(a, b)] = np.zeros(grid)

        scores = np.full(X.shape[0], self.intercept_)
        rng = np.random.default_rng(self.random_state)
        terms: list[object] = list(range(X.shape[1])) + list(self.pairs)
        for round_index in range(self.max_rounds):
            order = np.array(terms, dtype=object)
            if round_index:
                order = rng.permutation(order)
            for term in order:
                if isinstance(term, tuple):
                    joint = binned[term[0]] * n_bins[term[1]] + binned[term[1]]
                    scores = self._boost_pair(term, joint, target, scores)
                else:
                    scores = self._boost_main(int(term), binned[int(term)], target, scores)

        self._recenter()
        return self

    def _boost_main(
        self, j: int, bins: np.ndarray, target: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """One Newton boosting step on feature ``j`` (``max_leaves = 2``).

        The bin sums of the gradients/hessians are split once at the
        best adjacent-bin cut (the gain ``G_l²/(H_l+λ) + G_r²/(H_r+λ)
        − G²/(H+λ)``) and each side takes the XGBoost-style Newton
        leaf ``−lr·G/(H+λ)`` — ``GenerateTermUpdate.cpp`` on numpy.
        """
        grad, hess = self._grad_hess(scores, target)
        n_bins = len(self.term_scores_[j])
        g_sum = np.bincount(bins, weights=grad, minlength=n_bins)
        h_sum = np.bincount(bins, weights=hess, minlength=n_bins)
        counts = np.bincount(bins, minlength=n_bins).astype(float)
        old = self.term_scores_[j].copy()
        lam = self.reg_lambda
        total_g, total_h = g_sum.sum(), h_sum.sum()
        base = total_g**2 / (total_h + lam + 1e-12)
        best_gain, best_cut = 0.0, -1
        for cut in range(1, n_bins):
            if counts[:cut].sum() < 1 or counts[cut:].sum() < 1:
                continue
            gl, hl = g_sum[:cut].sum(), h_sum[:cut].sum()
            gain = (
                gl**2 / (hl + lam + 1e-12)
                + (total_g - gl) ** 2 / (total_h - hl + lam + 1e-12)
                - base
            )
            if gain > best_gain:
                best_gain, best_cut = gain, cut
        if best_cut < 0:
            self.term_scores_[j] += (
                -self.learning_rate * g_sum / (h_sum + lam + 1e-12)
            )
        else:
            for selection in (slice(0, best_cut), slice(best_cut, None)):
                g_side = float(g_sum[selection].sum())
                h_side = float(h_sum[selection].sum())
                self.term_scores_[j][selection] += (
                    -self.learning_rate * g_side / (h_side + lam + 1e-12)
                )
        return scores + (self.term_scores_[j][bins] - old[bins])

    def _boost_pair(
        self, pair: tuple[int, int], joint: np.ndarray, target: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """One Newton step on a pair tensor (no split search — honest)."""
        grad, hess = self._grad_hess(scores, target)
        grid = len(self.pair_scores_[pair])
        g_sum = np.bincount(joint, weights=grad, minlength=grid)
        h_sum = np.bincount(joint, weights=hess, minlength=grid)
        previous = self.pair_scores_[pair].copy()
        self.pair_scores_[pair] += (
            -self.learning_rate * g_sum / (h_sum + self.reg_lambda + 1e-12)
        )
        return scores + self.pair_scores_[pair][joint] - previous[joint]

    def _grad_hess(
        self, scores: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.loss == "rmse":
            return scores - target, np.ones_like(scores)
        p = _sigmoid(scores)
        return p - target, p * (1.0 - p) + 1e-12

    def _recenter(self) -> None:
        """Move each term's weighted mean into the intercept (EBM post-fit)."""
        for j, weights in self.bin_weights_.items():
            mean = float((weights * self.term_scores_[j]).sum())
            self.term_scores_[j] -= mean
            self.intercept_ += mean
        for pair, weights in self.pair_weights_.items():
            mean = float((weights * self.pair_scores_[pair]).sum())
            self.pair_scores_[pair] -= mean
            self.intercept_ += mean

    # -- prediction -----------------------------------------------------------
    def _bin_row(self, X: np.ndarray) -> list[np.ndarray]:
        return [
            np.asarray(np.searchsorted(self.cuts_[j], X[:, j]), dtype=int)
            for j in range(self.n_features_)
        ]

    def decision_function(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """The additive score ``intercept + Σ terms``."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"X must have {self.n_features_} columns")
        binned = self._bin_row(X)
        total = np.full(X.shape[0], self.intercept_)
        for j in range(self.n_features_):
            total += self.term_scores_[j][binned[j]]
        for (a, b), scores in self.pair_scores_.items():
            n_bins_b = len(self.term_scores_[b])
            joint = binned[a] * n_bins_b + binned[b]
            total += scores[joint]
        return total

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Regression values (rmse) or binary classes (log_loss)."""
        if self.loss == "rmse":
            return self.decision_function(X)
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Class-1 probabilities (log_loss only)."""
        if self.loss != "log_loss":
            raise ValueError("predict_proba needs loss='log_loss'")
        return _sigmoid(self.decision_function(X))

    # -- explanation -----------------------------------------------------------
    def term_importance(self) -> dict[str, float]:
        """The ``avg_weight`` importances — weighted mean of |term scores|."""
        out: dict[str, float] = {}
        for j in range(self.n_features_):
            weights = self.bin_weights_[j]
            out[f"x{j + 1}"] = float((weights * np.abs(self.term_scores_[j])).sum())
        for (a, b) in self.pair_scores_:
            weights = self.pair_weights_[(a, b)]
            out[f"x{a + 1}:x{b + 1}"] = float(
                (weights * np.abs(self.pair_scores_[(a, b)])).sum()
            )
        return out

    def term_profile(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Feature ``j``'s graph — bin centers and zero-mean scores.

        The centers sit between the cut points (the outermost bins get
        one-bin-width anchors beyond the extreme cuts); the scores are
        the re-centered term values — EBM's readable graphs.
        """
        if not 0 <= j < self.n_features_:
            raise ValueError(f"feature index {j} out of range")
        cuts = self.cuts_[j]
        width = (cuts[-1] - cuts[0]) / max(1, len(cuts)) if len(cuts) else 1.0
        lo = cuts[0] - width if len(cuts) else -1.0
        hi = cuts[-1] + width if len(cuts) else 1.0
        edges = np.concatenate([[lo], cuts, [hi]])
        centers = (edges[:-1] + edges[1:]) / 2.0
        return centers, self.term_scores_[j].copy()
