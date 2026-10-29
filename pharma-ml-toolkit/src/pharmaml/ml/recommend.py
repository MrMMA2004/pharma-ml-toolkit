"""Collaborative filtering in plain NumPy: the recommender's core.

The best-of-ml-python *Recommender Systems* category (18 projects:
implicit, LightFM, spotlight, librecommender, RecBole ...) was
reviewed entry by entry for this module.  Strip the factorisation
machines, the GPU samplers and the ANN serving tiers and every one of
them reduces to the same three ideas -- and none of the three needs an
engine:

* **The similarity** -- cosine or Pearson over *co-rated* items only
  (the pairwise-complete correlation, the statistic the neighbourhood
  methods actually estimate; rows with no overlap get similarity 0,
  not NaN propagation).
* **The neighbourhood prediction** -- user-based (a weighted average
  over the ``k`` most similar users that rated the item) and
  item-based (a weighted sum over the ``k`` most similar items the
  user already rated) -- the two Chapter-9 classics the category's
  libraries all grew out of.
* **The honest evaluation** -- leave-one-out: hide one rated item per
  user, rebuild, and ask whether the top-N finds it -- *against the
  popularity baseline*, because a recommender that cannot beat "rank
  by count" has learned nothing a hit-rate can see.

The pharma reading, stated once: a "user" is a screening campaign, an
"item" is a compound series, a "rating" is a measured activity grade
-- the same arithmetic ranks which untested series a campaign with
known preferences should screen next.

The honest scope: matrix factorisation with SGD (spotlight/LightFM's
core) needs iterative training loops the toolkit's NumPy-first surface
does not vendor; the neighbourhood methods here are the exact,
closed-form half of the field, and the leave-one-out protocol is how
they are honestly judged.

Examples
--------
>>> from pharmaml.ml.recommend import (
...     evaluate_recommender, recommend_itembased, recommend_userbased,
...     similarity_matrix, synthetic_ratings,
... )
>>> matrix = synthetic_ratings(seed=42)
>>> matrix.shape
(24, 20)
>>> sims = similarity_matrix(matrix, metric="cosine")
>>> round(float(sims[0, 0]), 6)
1.0
>>> recs = recommend_userbased(matrix, user=0, k=8, top_n=3)
>>> len(recs.recommendations)
3
>>> recs.recommendations[0].item < 20
True
>>> item_recs = recommend_itembased(matrix, user=0, k=8, top_n=3)
>>> len(item_recs.recommendations)
3
>>> report = evaluate_recommender(matrix, kind="user", k=8, top_n=5)
>>> report.n_eval_users > 15
True
>>> report.hit_rate_popularity < 1.0     # not a free lunch
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through the synthetic data and the held-out
#: split (the pin).
RANDOM_STATE = 42

#: The two neighbourhood methods.
SIMILARITY_METRICS = ("cosine", "pearson")


# ---------------------------------------------------------------------------
# The rating matrix
# ---------------------------------------------------------------------------


def _check_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise DataValidationError(
            f"the rating matrix must be 2-D (users x items), got {array.ndim}-D"
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise DataValidationError(f"the rating matrix must be non-empty, got shape {array.shape}")
    observed = ~np.isnan(array)
    if not observed.any():
        raise DataValidationError("the rating matrix has no observed ratings")
    values = array[observed]
    if not np.all(np.isfinite(values)):
        raise DataValidationError("observed ratings must all be finite (NaN marks missing)")
    return array


def synthetic_ratings(
    n_users: int = 24,
    n_items: int = 20,
    density: float = 0.4,
    n_factors: int = 3,
    noise: float = 0.25,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """A seeded rating matrix with genuine latent structure.

    Users and items get ``n_factors``-dimensional latent vectors; the
    rating is the inner product (rescaled to the 1-5 grade grid) plus
    Gaussian noise, observed on a ``density`` fraction of the cells.
    The structure is what makes collaborative filtering *workable* --
    similar users genuinely share preferences -- so the leave-one-out
    hit-rate below has something real to find.

    Raises
    ------
    DataValidationError
        On non-positive sizes, density outside (0, 1], or non-positive
        factors/noise.
    """
    if n_users < 2 or n_items < 2:
        raise DataValidationError(f"need at least 2 users and 2 items, got {(n_users, n_items)!r}")
    if not (0.0 < density <= 1.0):
        raise DataValidationError(f"density must be in (0, 1], got {density!r}")
    if n_factors < 1:
        raise DataValidationError(f"n_factors must be >= 1, got {n_factors}")
    if noise < 0:
        raise DataValidationError(f"noise must be >= 0, got {noise!r}")
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(size=(n_users, n_factors))
    item_factors = rng.normal(size=(n_items, n_factors))
    raw = user_factors @ item_factors.T + noise * rng.normal(size=(n_users, n_items))
    ratings = np.clip(np.round(raw * 1.5 + 3.0), 1.0, 5.0)
    keep = rng.random(size=(n_users, n_items)) < density
    # every user and every item keeps at least one rating, or CF has no
    # foothold at all
    keep[np.arange(n_users), rng.integers(0, n_items, size=n_users)] = True
    keep[rng.integers(0, n_users, size=n_items), np.arange(n_items)] = True
    matrix = np.where(keep, ratings, np.nan)
    _LOG.debug(
        "synthetic ratings: %d users x %d items, %d observed (%.0f%%)",
        n_users,
        n_items,
        int(keep.sum()),
        100.0 * keep.mean(),
    )
    return matrix


# ---------------------------------------------------------------------------
# The similarities (pairwise-complete: co-rated items only)
# ---------------------------------------------------------------------------


def similarity_matrix(matrix: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """User-user similarities over co-rated items only.

    Pairwise-complete statistics: two users are compared on the items
    *both* rated (cosine: the dot product over that subset; Pearson:
    the mean-centred correlation, which needs at least two co-rated
    items).  Users with no overlap get similarity exactly 0 -- the
    neighbourhood machinery downstream must not see NaN -- and the
    diagonal is 1 by construction.  The item-based twin is the same
    function on the transpose: ``similarity_matrix(matrix.T)``.

    Raises
    ------
    DataValidationError
        On an unknown metric or a malformed matrix.
    """
    if metric not in SIMILARITY_METRICS:
        raise DataValidationError(
            f"unknown metric {metric!r}; valid: {', '.join(SIMILARITY_METRICS)}"
        )
    array = _check_matrix(matrix)
    n = array.shape[0]
    sims = np.zeros((n, n), dtype=float)
    np.fill_diagonal(sims, 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            both = ~np.isnan(array[i]) & ~np.isnan(array[j])
            if not both.any():
                continue
            a, b = array[i, both], array[j, both]
            if metric == "cosine":
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                value = float(np.dot(a, b)) / denom if denom > 1e-12 else 0.0
            else:
                if both.sum() < 2:
                    continue
                sa, sb = a.std(), b.std()
                if sa < 1e-12 or sb < 1e-12:
                    continue
                value = float(np.corrcoef(a, b)[0, 1])
            value = float(np.clip(value, -1.0, 1.0))
            sims[i, j] = sims[j, i] = value
    return sims


# ---------------------------------------------------------------------------
# The two neighbourhood methods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    """One ranked item: index, predicted rating, neighbourhood score."""

    item: int
    predicted_rating: float
    neighbour_weight: float


@dataclass(frozen=True)
class UserRecommendations:
    """A user's top-N list, with the method and neighbourhood recorded."""

    user: int
    method: str
    k: int
    user_mean: float
    n_rated: int
    recommendations: tuple[Recommendation, ...]


def recommend_userbased(
    matrix: np.ndarray, user: int, k: int = 10, top_n: int = 5
) -> UserRecommendations:
    """User-based collaborative filtering: the k-neighbour weighted mean.

    For every item the user has *not* rated, the prediction is the
    weighted average of the neighbours' ratings on it -- weights are
    the similarities, neighbours are the ``k`` most similar users with
    a rating on that item (per-item neighbourhoods, the classic
    GroupLens recipe).  Negative similarities are dropped: a strongly
    dissimilar twin is information, but not a usable vote.

    Raises
    ------
    DataValidationError
        On an out-of-range user, non-positive k/top_n, or a user with
        no ratings.
    """
    array = _check_matrix(matrix)
    n_users, n_items = array.shape
    if not (0 <= user < n_users):
        raise DataValidationError(f"user index {user} out of range for {n_users} users")
    if k < 1:
        raise DataValidationError(f"k must be >= 1, got {k}")
    if top_n < 1:
        raise DataValidationError(f"top_n must be >= 1, got {top_n}")
    rated = ~np.isnan(array[user])
    if not rated.any():
        raise DataValidationError(f"user {user} has no ratings to build a neighbourhood from")
    user_mean = float(np.nanmean(array[user]))
    sims = similarity_matrix(array, metric="cosine")
    sims[user, user] = 0.0  # never vote for yourself
    candidates = [i for i in range(n_items) if not rated[i]]
    scored: list[Recommendation] = []
    for item in candidates:
        raters = ~np.isnan(array[:, item])
        weights = np.where(raters, sims[user], 0.0)
        positive = weights > 0.0
        if not positive.any():
            continue
        order = np.argsort(-weights[positive])
        top = np.where(positive)[0][order[:k]]
        w = sims[user, top]
        r = array[top, item]
        denom = float(w.sum())
        if denom < 1e-12:
            continue
        prediction = float(np.dot(w, r) / denom)
        scored.append(
            Recommendation(item=item, predicted_rating=prediction, neighbour_weight=denom)
        )
    scored.sort(key=lambda rec: (-rec.predicted_rating, rec.item))
    _LOG.debug(
        "user-based CF: user=%d k=%d scored %d of %d unseen items",
        user,
        k,
        len(scored),
        len(candidates),
    )
    return UserRecommendations(
        user=int(user),
        method="user-based",
        k=int(k),
        user_mean=user_mean,
        n_rated=int(rated.sum()),
        recommendations=tuple(scored[:top_n]),
    )


def recommend_itembased(
    matrix: np.ndarray, user: int, k: int = 10, top_n: int = 5
) -> UserRecommendations:
    """Item-based collaborative filtering: the similarity-weighted sum.

    The Amazon/Sarwar et al. recipe: for every unseen item, take the
    ``k`` items the user *did* rate with the highest item-item cosine
    similarity to it, and sum similarity × rating (normalised by the
    similarity mass).  Denser and more stable than user-based on wide
    catalogues -- the reason the industry standardised on it first.

    Raises
    ------
    DataValidationError
        On an out-of-range user, non-positive k/top_n, or a user with
        no ratings.
    """
    array = _check_matrix(matrix)
    n_users, n_items = array.shape
    if not (0 <= user < n_users):
        raise DataValidationError(f"user index {user} out of range for {n_users} users")
    if k < 1:
        raise DataValidationError(f"k must be >= 1, got {k}")
    if top_n < 1:
        raise DataValidationError(f"top_n must be >= 1, got {top_n}")
    rated = ~np.isnan(array[user])
    if not rated.any():
        raise DataValidationError(f"user {user} has no ratings to build a profile from")
    user_mean = float(np.nanmean(array[user]))
    item_sims = similarity_matrix(array.T, metric="cosine")
    rated_items = np.where(rated)[0]
    candidates = [i for i in range(n_items) if not rated[i]]
    scored: list[Recommendation] = []
    for item in candidates:
        sims_to_rated = item_sims[item, rated_items]
        order = np.argsort(-sims_to_rated)[:k]
        top = rated_items[order]
        w = item_sims[item, top]
        positive = w > 0.0
        if not positive.any():
            continue
        w = np.where(positive, w, 0.0)
        denom = float(w.sum())
        if denom < 1e-12:
            continue
        prediction = float(np.dot(w, array[user, top]) / denom)
        scored.append(
            Recommendation(item=item, predicted_rating=prediction, neighbour_weight=denom)
        )
    scored.sort(key=lambda rec: (-rec.predicted_rating, rec.item))
    _LOG.debug(
        "item-based CF: user=%d k=%d scored %d of %d unseen items",
        user,
        k,
        len(scored),
        len(candidates),
    )
    return UserRecommendations(
        user=int(user),
        method="item-based",
        k=int(k),
        user_mean=user_mean,
        n_rated=int(rated.sum()),
        recommendations=tuple(scored[:top_n]),
    )


# ---------------------------------------------------------------------------
# The honest evaluation: leave-one-out against the popularity baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecommenderEvaluation:
    """Leave-one-out hit-rate, CF versus the popularity baseline.

    ``hit_rate_cf`` is the fraction of evaluable users whose hidden
    item appears in the CF top-N; ``hit_rate_popularity`` the same for
    "rank unseen items by global rating count"; ``lift`` their ratio.
    A lift below 1 means the neighbourhood learned nothing the hit-rate
    can see -- the report says so rather than burying it.
    """

    kind: str
    k: int
    top_n: int
    n_eval_users: int
    hits_cf: int
    hits_popularity: int
    hit_rate_cf: float
    hit_rate_popularity: float

    @property
    def lift(self) -> float:
        """CF hit-rate over popularity hit-rate (1.0 = no lift)."""
        if self.hit_rate_popularity <= 0.0:
            return float("inf") if self.hit_rate_cf > 0 else 1.0
        return self.hit_rate_cf / self.hit_rate_popularity


def evaluate_recommender(
    matrix: np.ndarray,
    kind: str = "user",
    k: int = 10,
    top_n: int = 5,
    seed: int = RANDOM_STATE,
) -> RecommenderEvaluation:
    """Leave-one-out: hide one rating per user, rebuild, hit@N.

    The protocol, run exactly as the category's benchmarks do it: for
    every user with at least two observed ratings, one held-out item is
    hidden (seeded -- the pin); the recommender rebuilds on the pruned
    matrix and must place that item in its top-N; the popularity
    baseline ranks by training rating count.  Only users with at least
    one *unseen* item after hiding are evaluable.

    Raises
    ------
    DataValidationError
        On an unknown kind, non-positive k/top_n, or a matrix where no
        user can be evaluated.
    """
    if kind not in ("user", "item"):
        raise DataValidationError(f"unknown kind {kind!r}; valid: user, item")
    if k < 1:
        raise DataValidationError(f"k must be >= 1, got {k}")
    if top_n < 1:
        raise DataValidationError(f"top_n must be >= 1, got {top_n}")
    array = _check_matrix(matrix).copy()
    n_users, _n_items = array.shape
    rng = np.random.default_rng(seed)
    hidden: dict[int, int] = {}
    for user in range(n_users):
        rated = np.where(~np.isnan(array[user]))[0]
        if rated.size < 2:
            continue
        hidden[user] = int(rng.choice(rated))
        array[user, hidden[user]] = np.nan
    evaluable = [
        user
        for user, item in hidden.items()
        if np.isnan(array[user]).sum() > 0  # something left to recommend
    ]
    if not evaluable:
        raise DataValidationError("no user is evaluable: everyone rated everything")
    train_counts = np.nansum(~np.isnan(array), axis=0).astype(float)
    recommend = recommend_userbased if kind == "user" else recommend_itembased
    hits_cf = 0
    hits_pop = 0
    for user in evaluable:
        recs = recommend(array, user=user, k=k, top_n=top_n)
        cf_items = {r.item for r in recs.recommendations}
        if hidden[user] in cf_items:
            hits_cf += 1
        unseen = np.where(np.isnan(array[user]))[0]
        pop_order = sorted(unseen, key=lambda i: (-train_counts[i], i))
        if hidden[user] in set(pop_order[:top_n]):
            hits_pop += 1
    n = len(evaluable)
    _LOG.debug(
        "leave-one-out (%s, k=%d, top_n=%d): %d/%d CF vs %d/%d popularity",
        kind,
        k,
        top_n,
        hits_cf,
        n,
        hits_pop,
        n,
    )
    return RecommenderEvaluation(
        kind=kind,
        k=int(k),
        top_n=int(top_n),
        n_eval_users=int(n),
        hits_cf=int(hits_cf),
        hits_popularity=int(hits_pop),
        hit_rate_cf=hits_cf / n,
        hit_rate_popularity=hits_pop / n,
    )
