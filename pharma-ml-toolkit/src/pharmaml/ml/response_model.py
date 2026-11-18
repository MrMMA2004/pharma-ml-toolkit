"""The drug-response recommender: CaDRReS's model, DeepResponse's harness.

Three repositories were reviewed file by file for this module, none of
whose engines had a native home in the toolkit before this round:

* ``CSB5/CaDRReS`` (``scripts/CaDRReS_train_and_test.py`` and
  ``CaDRReS_test.py`` -- pure NumPy/pandas, Python 2 vintage) is a
  matrix-factorisation recommender over the drug-response matrix:
  sensitivity ``S`` approximated as ``mu + b_p + b_q + P·Qᵀ`` where the
  cell-half ``P = X·WP`` is *driven by cell-line features* (the
  pharmacogenomic space the paper interprets drug mechanisms through)
  and the drug-half ``Q = WQ`` is a free embedding (their ``Y = I``
  identity trick).  Full-batch gradient descent with the stop-before-
  diverging guard, the exact-rank nDCG evaluation, and the unseen-cell
  bias estimation are all absorbed here -- the training loop mirrors
  the original's update order (biases, then ``WP`` with the old
  ``WQ``, then ``WQ`` with the *new* ``WP``) so the optimisation
  dynamics are the paper's, not a paraphrase.
* ``burakcan-izmirli/DeepResponse`` contributed the *evaluation* half:
  its five split strategies (``random``, ``cell_stratified``,
  ``drug_stratified``, ``drug_cell_stratified``, ``cross_domain`` --
  the honest-leakage ladder every drug-response paper should climb)
  and its thresholded metric contract (regression metrics plus
  sensitivity-thresholded classification, ``pic50 >= 6`` sensitive).
  The PyTorch hybrid itself (SELFormer drug encoder, CNN cell encoder,
  FiLM fusion head) stays external -- the encoder stack is not
  reimplemented; what lands is the evaluation harness and the
  repurposing-inference concept.
* ``SmritiChawla/Precily`` supplied the grouped-split discipline (its
  ``cell_line_split`` -- cell lines never straddle train/test), which
  is exactly DeepResponse's ``cell_stratified`` arm.

Sign conventions, stated once: the response matrix reads in the IC50
direction (lower = more sensitive, matching
:mod:`pharmaml.biology.response`'s ``LN_IC50`` tables); the model
internally flips to the sensitivity score ``S = -IC50`` exactly as
CaDRReS's ``ss_df *= -1`` does, and every prediction frame is flipped
back to IC50 before it leaves this module.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from pharmaml.ml.response_model import (
...     cross_domain_split, exact_rank, fit_response_recommender,
...     ndcg_score, pharmacogenomic_space, predict_responses,
...     repurposing_candidates, response_metrics, response_split,
...     synthetic_response_screen,
... )
>>> response, features, truth = synthetic_response_screen(
...     10, 6, 5, n_factors=2, seed=42)
>>> response.shape, features.shape, truth.shape
((10, 6), (10, 5), (10, 6))
>>> model = fit_response_recommender(
...     response, features, n_factors=2, lr=0.05, max_epochs=200, seed=42)
>>> round(model.observed_rmse, 3) < round(model.history[0][1], 3)
True
>>> pred = predict_responses(model, features)
>>> pred.shape
(10, 6)
>>> p_space, q_space = pharmacogenomic_space(model, features)
>>> p_space.shape, q_space.shape
((10, 2), (6, 2))
>>> long = pd.DataFrame({
...     "cell_line": [f"C{i}" for i in range(6)] * 4,
...     "drug": sum([[f"D{j}"] * 6 for j in range(4)], []),
...     "value": np.arange(24.0),
... })
>>> split = response_split(long, strategy="cell_stratified", seed=42)
>>> set(split.train["cell_line"]).isdisjoint(split.test["cell_line"])
True
>>> metrics = response_metrics(long["value"], long["value"])
>>> round(metrics["rmse"], 6), metrics["accuracy"]
(0.0, 1.0)
>>> ranks = exact_rank(np.array([3.0, 1.0, 2.0]))
>>> ranks.tolist()
[0, 2, 1]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError

__all__ = [
    "SPLIT_STRATEGIES",
    "ResponseRecommender",
    "ResponseSplit",
    "cross_domain_split",
    "exact_rank",
    "fit_response_recommender",
    "ndcg_score",
    "pharmacogenomic_space",
    "predict_responses",
    "repurposing_candidates",
    "response_metrics",
    "response_split",
    "synthetic_response_screen",
]

#: The four within-table split strategies (DeepResponse's fifth --
#: ``cross_domain`` -- takes two tables and has its own entry point).
SPLIT_STRATEGIES: tuple[str, ...] = (
    "random",
    "cell_stratified",
    "drug_stratified",
    "drug_cell_stratified",
)

_MIN_SPLIT_CELLS = 3  # DeepResponse's MIN_UNIQUE_CELLS_FOR_SPLIT
_RIDGE_ALPHA = 1.0  # unseen-cell bias estimation


# --------------------------------------------------------------------- #
# The model (CaDRReS)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResponseRecommender:
    """A fitted CaDRReS-style response recommender.

    Attributes
    ----------
    wp:
        ``(n_features, n_factors)`` cell-feature loadings -- the map
        from the feature space into the pharmacogenomic space.
    wq:
        ``(n_drugs, n_factors)`` drug embeddings (CaDRReS's ``Y = I``
        identity trick makes ``Q = WQ`` exactly).
    mu / b_p / b_q:
        Global mean and per-cell / per-drug bias terms of the
        sensitivity score ``S = -IC50``.
    x_train:
        The ``(n_cells, n_features)`` training feature block, kept so
        :func:`predict_responses` can ridge-estimate biases for cell
        lines the response matrix never saw.
    cells / drugs / features:
        Name axes, preserved for every downstream frame.
    history:
        ``(epoch, observed_rmse, ndcg)`` per epoch -- the exact triple
        CaDRReS prints during training.  As upstream, the recorded
        loss lags the parameter update by one epoch.
    observed_rmse:
        Root-mean-square error on the observed entries at the final
        parameters.
    stopped_early:
        True when the divergence guard fired and the failing epoch's
        update was discarded (CaDRReS's "stop before diverging").
    """

    wp: np.ndarray
    wq: np.ndarray
    mu: float
    b_p: np.ndarray
    b_q: np.ndarray
    x_train: np.ndarray
    cells: tuple[str, ...]
    drugs: tuple[str, ...]
    features: tuple[str, ...]
    history: tuple[tuple[int, float, float], ...]
    observed_rmse: float
    stopped_early: bool
    n_factors: int


def _numeric_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Coerce a frame to float, refusing non-numeric junk."""
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    bad = [
        str(column)
        for column in frame.columns
        if numeric[column].isna().all() and frame[column].notna().any()
    ]
    if bad:
        raise DataValidationError(
            f"{name} has non-numeric columns: {', '.join(bad[:5])}"
        )
    return numeric.astype(float)


def fit_response_recommender(
    response: pd.DataFrame,
    features: pd.DataFrame,
    *,
    n_factors: int = 10,
    lr: float = 0.01,
    max_epochs: int = 500,
    seed: int = 42,
) -> ResponseRecommender:
    """Fit CaDRReS's matrix-factorisation recommender.

    The model (all in the flipped sensitivity space ``S = -IC50``)::

        pred[u, i] = mu + b_p[u] + b_q[i] + (X·WP)[u, :] @ WQ[i, :]

    trained by CaDRReS's full-batch gradient descent: the masked
    squared error over the ``n_K`` observed entries, biases first,
    then ``WP`` with the current ``WQ``, then ``WQ`` with the *just
    updated* ``WP`` -- and the guard that discards an epoch's update
    the first time the observed RMSE fails to improve (the original's
    "stop before diverging"; as upstream, the recorded loss lags the
    update by one epoch, and the parameters kept are the pre-update
    ones).

    Parameters
    ----------
    response:
        Cell lines x drugs response matrix in the IC50 direction
        (lower = more sensitive); unmeasured entries are NaN.  Rows
        or columns with *no* observed entry are dropped (CaDRReS's
        ``dropna(how='all')``).
    features:
        Cell lines x features table (the ML ``X`` convention).  Must
        cover every surviving response row; NaN features are refused
        (impute first, e.g. :func:`pharmaml.biology.omics.impute_expression`).
    n_factors:
        Dimensionality of the pharmacogenomic space (the paper's
        ``f``, default 10 in their runs).
    lr / max_epochs / seed:
        Learning rate, epoch ceiling and RNG seed; ``seed`` drives the
        CaDRReS-style ``U(-0.05, 0.05)`` initialisation of both
        loading matrices.

    Returns
    -------
    ResponseRecommender
        The fitted model; see :func:`predict_responses` and
        :func:`pharmacogenomic_space` for the two things one does
        with it.

    Examples
    --------
    >>> response, features, _truth = synthetic_response_screen(
    ...     8, 5, 4, n_factors=2, seed=7)
    >>> model = fit_response_recommender(
    ...     response, features, n_factors=2, lr=0.05, max_epochs=120,
    ...     seed=7)
    >>> model.wp.shape, model.wq.shape
    ((4, 2), (5, 2))
    >>> model.history[0][0], len(model.history) >= 1
    (1, True)
    """
    if not isinstance(response, pd.DataFrame) or not isinstance(features, pd.DataFrame):
        raise DataValidationError("response and features must be pandas DataFrames")
    if n_factors < 1:
        raise DataValidationError(f"n_factors must be >= 1, got {n_factors}")
    if lr <= 0.0:
        raise DataValidationError(f"lr must be > 0, got {lr}")
    if max_epochs < 1:
        raise DataValidationError(f"max_epochs must be >= 1, got {max_epochs}")

    matrix = _numeric_frame(response, "response")
    dropped_rows = [str(i) for i in matrix.index if not matrix.loc[i].notna().any()]
    matrix = matrix.dropna(how="all")
    dropped_cols = [str(c) for c in matrix.columns if not matrix[c].notna().any()]
    matrix = matrix.dropna(axis=1, how="all")
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise DataValidationError(
            "response needs at least 2 cell lines and 2 drugs with "
            f"observed entries (got {matrix.shape[0]} x {matrix.shape[1]}"
            + (f"; dropped rows {dropped_rows[:3]}" if dropped_rows else "")
            + (f"; dropped columns {dropped_cols[:3]}" if dropped_cols else "")
            + ")"
        )

    cells = [str(i) for i in matrix.index]
    drugs = [str(c) for c in matrix.columns]
    feature_index = [str(i) for i in features.index]
    missing = [cell for cell in cells if cell not in feature_index]
    if missing:
        raise DataValidationError(
            f"features is missing {len(missing)} response cell lines "
            f"(first: {missing[:3]}) -- index the features by cell line"
        )
    feat = _numeric_frame(features, "features").loc[cells]
    if feat.isna().any().any():
        raise DataValidationError(
            "features contains NaN -- impute before fitting "
            "(biology.omics.impute_expression / biology.enrichment.knn_impute)"
        )
    feature_names = [str(c) for c in feat.columns]

    x = feat.to_numpy(dtype=float)
    r = matrix.to_numpy(dtype=float)
    sensitivity = -r  # CaDRReS's ``ss_df *= -1``: higher = more sensitive
    observed = np.isfinite(sensitivity)
    n_observed = float(observed.sum())

    rng = np.random.default_rng(seed)
    wp = (rng.random((x.shape[1], n_factors)) - 0.5) / 10.0
    wq = (rng.random((len(drugs), n_factors)) - 0.5) / 10.0
    mu = float(np.mean(np.where(observed, sensitivity, 0.0)))
    b_p = np.zeros(len(cells))
    b_q = np.zeros(len(drugs))

    history: list[tuple[int, float, float]] = []
    best_rmse = np.inf
    stopped_early = False
    for epoch in range(1, max_epochs + 1):
        pred = mu + b_p[:, None] + b_q[None, :] + (x @ wp) @ wq.T
        error = np.where(observed, sensitivity - pred, 0.0)
        rmse = float(np.sqrt((error**2).sum() / n_observed))
        ndcg = _ndcg_rows(sensitivity, pred, observed)
        history.append((epoch, rmse, ndcg))
        if rmse >= best_rmse:
            # CaDRReS's guard: this epoch's update is discarded and the
            # loop stops with the pre-update parameters.
            stopped_early = True
            break
        best_rmse = rmse
        # the update order mirrors the original script exactly
        b_p = b_p + lr * error.sum(axis=1) / n_observed
        b_q = b_q + lr * error.sum(axis=0) / n_observed
        wp = wp + lr * (x.T @ error @ wq) / n_observed
        wq = wq + lr * (error.T @ x @ wp) / n_observed  # NEW wp, as upstream

    return ResponseRecommender(
        wp=wp,
        wq=wq,
        mu=mu,
        b_p=b_p,
        b_q=b_q,
        x_train=x,
        cells=tuple(cells),
        drugs=tuple(drugs),
        features=tuple(feature_names),
        history=tuple(history),
        observed_rmse=float(history[-1][1]),
        stopped_early=stopped_early,
        n_factors=n_factors,
    )


def predict_responses(
    model: ResponseRecommender, features: pd.DataFrame
) -> pd.DataFrame:
    """Predict the full cell x drug IC50 matrix for any cell lines.

    Seen cell lines carry their learned ``b_p`` bias; unseen ones get
    a bias estimated by ridge regression of the trained ``b_p`` on the
    stored training features -- the honest generalisation of
    CaDRReS's ``b_p_test = b_p @ X_test.T`` line (which silently
    requires the feature count to equal the cell count; the ridge
    works at any shape and says so).

    Examples
    --------
    >>> response, features, _truth = synthetic_response_screen(
    ...     8, 5, 4, n_factors=2, seed=3)
    >>> model = fit_response_recommender(
    ...     response, features, n_factors=2, lr=0.05, max_epochs=150,
    ...     seed=3)
    >>> pred = predict_responses(model, features)
    >>> list(pred.columns) == list(response.columns)
    True
    >>> held_out = features.iloc[:4].copy()
    >>> held_out.index = [f"NEW_{i}" for i in held_out.index]
    >>> fresh = predict_responses(model, held_out)
    >>> fresh.shape
    (4, 5)
    """
    if not isinstance(features, pd.DataFrame):
        raise DataValidationError("features must be a pandas DataFrame")
    names = [str(c) for c in features.columns]
    missing = [name for name in model.features if name not in names]
    if missing:
        raise DataValidationError(
            f"features is missing {len(missing)} model features "
            f"(first: {missing[:3]})"
        )
    feat = _numeric_frame(features, "features")[list(model.features)]
    if feat.isna().any().any():
        raise DataValidationError("features contains NaN -- impute before predicting")
    x = feat.to_numpy(dtype=float)

    seen = np.array([str(i) in model.cells for i in feat.index])
    b_p = np.zeros(len(feat))
    if seen.any():
        position = {cell: k for k, cell in enumerate(model.cells)}
        b_p[seen] = [
            float(model.b_p[position[str(i)]]) for i in feat.index[seen]
        ]
    if (~seen).any():
        gram = model.x_train.T @ model.x_train
        gram = gram + _RIDGE_ALPHA * np.eye(gram.shape[0])
        weights = np.linalg.solve(gram, model.x_train.T @ model.b_p)
        b_p[~seen] = x[~seen] @ weights

    sensitivity = (
        model.mu + b_p[:, None] + model.b_q[None, :] + (x @ model.wp) @ model.wq.T
    )
    return pd.DataFrame(-sensitivity, index=feat.index, columns=list(model.drugs))


def pharmacogenomic_space(
    model: ResponseRecommender, features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ``(P, Q)`` pharmacogenomic space as DataFrames.

    ``P`` = cell lines x factors (``X·WP``), ``Q`` = drugs x factors
    (``WQ``) -- the two matrices CaDRReS exports as ``CaDRReS_P.csv``
    and ``CaDRReS_Q.csv`` for drug-mechanism interpretation.  ``P``
    needs the feature frame the model was trained on (any superset of
    its cell lines works).

    Examples
    --------
    >>> response, features, _truth = synthetic_response_screen(
    ...     6, 4, 3, n_factors=2, seed=5)
    >>> model = fit_response_recommender(
    ...     response, features, n_factors=2, lr=0.05, max_epochs=60,
    ...     seed=5)
    >>> p_space, q_space = pharmacogenomic_space(model, features)
    >>> p_space.shape, q_space.shape
    ((6, 2), (4, 2))
    >>> list(q_space.index) == list(response.columns)
    True
    """
    names = [str(c) for c in features.columns]
    missing = [name for name in model.features if name not in names]
    if missing:
        raise DataValidationError(f"features is missing: {missing[:3]}")
    feat = _numeric_frame(features, "features")
    x = feat.loc[list(model.cells), list(model.features)].to_numpy(dtype=float)
    p_frame = pd.DataFrame(
        x @ model.wp,
        index=list(model.cells),
        columns=[f"f{k + 1}" for k in range(model.n_factors)],
    )
    q_frame = pd.DataFrame(
        model.wq,
        index=list(model.drugs),
        columns=[f"f{k + 1}" for k in range(model.n_factors)],
    )
    return p_frame, q_frame


# --------------------------------------------------------------------- #
# The evaluation (CaDRReS's nDCG + DeepResponse's metrics)
# --------------------------------------------------------------------- #


def exact_rank(scores: np.ndarray) -> np.ndarray:
    """CaDRReS's ``score_to_exact_rank``: 0-based rank, high = rank 0.

    Examples
    --------
    >>> exact_rank(np.array([3.0, 1.0, 2.0])).tolist()
    [0, 2, 1]
    >>> exact_rank(np.array([-1.0, -1.0])).tolist()  # ties: stable order
    [0, 1]
    """
    values = np.asarray(scores, dtype=float)
    return (-values).argsort(kind="stable").argsort(kind="stable")


def _ndcg_rows(
    sensitivity: np.ndarray, pred: np.ndarray, observed: np.ndarray
) -> float:
    """Mean nDCG over rows (CaDRReS's ``cal_exact_avg_ndcg``)."""
    total = 0.0
    count = 0
    for u in range(sensitivity.shape[0]):
        mask = observed[u, :]
        if not mask.any():
            continue
        s_u = sensitivity[u, :][mask]
        r_u = exact_rank(s_u)
        r_pred = exact_rank(pred[u, :][mask])
        gains = np.power(2.0, s_u)
        ideal = float((gains / np.log(r_u + 2.0)).sum())
        if ideal <= 0.0:
            continue
        total += float((gains / np.log(r_pred + 2.0)).sum()) / ideal
        count += 1
    return total / count if count else float("nan")


def ndcg_score(response: pd.DataFrame, predicted: pd.DataFrame) -> float:
    """Mean exact-rank nDCG of ``predicted`` against ``response``.

    Both frames are cell x drug in the IC50 direction (lower = more
    sensitive); only entries observed (finite) in ``response`` are
    scored, row by row, with CaDRReS's gain ``2 ** sensitivity`` and
    discount ``log(rank + 2)`` (the log base cancels in the ratio --
    the original used ``np.log``; any base gives the same number).

    Examples
    --------
    >>> response, features, _truth = synthetic_response_screen(
    ...     6, 4, 3, n_factors=2, seed=11)
    >>> model = fit_response_recommender(
    ...     response, features, n_factors=2, lr=0.05, max_epochs=80,
    ...     seed=11)
    >>> pred = predict_responses(model, features)
    >>> value = ndcg_score(response, pred)
    >>> 0.0 <= value <= 1.0000001
    True
    >>> round(ndcg_score(response, response.copy()), 6)
    1.0
    """
    if not isinstance(response, pd.DataFrame) or not isinstance(predicted, pd.DataFrame):
        raise DataValidationError("response and predicted must be DataFrames")
    drugs = [str(c) for c in response.columns]
    missing = [d for d in drugs if d not in [str(c) for c in predicted.columns]]
    if missing:
        raise DataValidationError(f"predicted is missing drugs: {missing[:3]}")
    cells = [str(i) for i in response.index]
    pred_aligned = predicted.loc[cells, drugs]
    truth = _numeric_frame(response, "response").to_numpy(dtype=float)
    pred_np = pred_aligned.to_numpy(dtype=float)
    observed = np.isfinite(truth)
    return _ndcg_rows(-truth, -pred_np, observed)


#: The metric keys :func:`response_metrics` always returns.
METRIC_KEYS: tuple[str, ...] = (
    "rmse", "mae", "r2", "pcc", "scc",
    "accuracy", "precision", "recall", "f1", "mcc",
)


def response_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    threshold: float = 6.0,
) -> dict[str, float]:
    """DeepResponse's metric contract on a pair of response vectors.

    Regression metrics (``rmse``/``mae``/``r2``/``pcc``/``scc``) plus
    the six thresholded classification metrics (``accuracy``,
    ``precision``, ``recall``, ``f1``, ``mcc``) at the sensitivity
    threshold -- DeepResponse's ``BINARY_THRESHOLD = 6.0`` on pIC50,
    where ``value >= threshold`` reads *sensitive*.  Non-finite pairs
    are dropped pairwise, ``n <= 2`` undefined correlations read NaN,
    and a constant ``y_true`` yields ``r2 = NaN`` rather than
    sklearn's silent ``0.0`` (stated, not silent).  The upstream's
    scipy p-values (``pcc_p``/``scc_p``) are the one piece not
    carried over -- the toolkit has no scipy dependency by policy.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([4.0, 5.0, 6.5, 7.0, 8.0])
    >>> perfect = response_metrics(y, y.copy())
    >>> round(perfect["rmse"], 6), round(perfect["r2"], 6)
    (0.0, 1.0)
    >>> perfect["accuracy"], perfect["mcc"]
    (1.0, 1.0)
    >>> flipped = response_metrics(y, y[::-1].copy())
    >>> round(flipped["pcc"], 3)
    -0.961
    >>> empty = response_metrics(np.array([np.nan]), np.array([1.0]))
    >>> bool(np.isnan(empty["rmse"]))
    True
    """
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.size != pred.size:
        raise DataValidationError(
            f"y_true and y_pred length mismatch: {true.size} != {pred.size}"
        )
    keep = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[keep], pred[keep]
    n = int(true.size)

    def _nan() -> float:
        return float("nan")

    if n == 0:
        return {key: _nan() for key in METRIC_KEYS}
    error = pred - true
    rmse = float(np.sqrt((error**2).mean()))
    mae = float(np.abs(error).mean())
    ss_res = float((error**2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else _nan()
    pcc = float(np.corrcoef(true, pred)[0, 1]) if (n > 2 and ss_tot > 0.0) else _nan()
    scc = _spearman(true, pred) if n > 2 else _nan()

    true_binary = (true >= threshold).astype(int)
    pred_binary = (pred >= threshold).astype(int)
    tp = int(((true_binary == 1) & (pred_binary == 1)).sum())
    tn = int(((true_binary == 0) & (pred_binary == 0)).sum())
    fp = int(((true_binary == 0) & (pred_binary == 1)).sum())
    fn = int(((true_binary == 1) & (pred_binary == 0)).sum())
    accuracy = float((tp + tn) / n)
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = float((tp * tn - fp * fn) / np.sqrt(mcc_denom)) if mcc_denom else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pcc": pcc,
        "scc": scc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation via tie-averaged ranks."""
    rank_a = _average_ranks(a)
    rank_b = _average_ranks(b)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks (1-based), the ``scipy.stats.rankdata`` rule."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


# --------------------------------------------------------------------- #
# The split harness (DeepResponse + Precily)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResponseSplit:
    """A three-way split of a long-format response table.

    Attributes
    ----------
    train / val / test:
        The three frames (original columns preserved).
    strategy:
        One of :data:`SPLIT_STRATEGIES` or ``"cross_domain"``.
    notes:
        The honest-leakage sentences each strategy earns -- the
        reason this dataclass exists at all.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    notes: tuple[str, ...] = field(default_factory=tuple)


def _split_axes(
    labels: list[str],
    test_size: float,
    val_size: float,
    rng: np.random.Generator,
    *,
    what: str,
) -> tuple[list[str], list[str], list[str]]:
    """Permute an axis and carve test/val/train label lists."""
    unique = list(dict.fromkeys(labels))
    if what == "cells" and len(unique) < _MIN_SPLIT_CELLS:
        raise DataValidationError(
            f"cell_stratified split needs >= {_MIN_SPLIT_CELLS} unique "
            f"cell lines (found {len(unique)})"
        )
    if len(unique) < 2:
        raise DataValidationError(
            f"cannot stratify by {what}: only {len(unique)} unique value(s)"
        )
    shuffled = list(rng.permutation(unique))
    n = len(shuffled)
    n_test = max(1, round(test_size * n))
    n_val = max(1, round(val_size * n))
    if n_test + n_val >= n:
        raise DataValidationError(
            f"{what} split leaves no training {what}: "
            f"{n} unique values, test {n_test}, val {n_val}"
        )
    return (
        shuffled[n_test + n_val :],
        shuffled[n_test : n_test + n_val],
        shuffled[:n_test],
    )


def _axis_split(
    frame: pd.DataFrame,
    column: str,
    train_labels: list[str],
    val_labels: list[str],
    test_labels: list[str],
    strategy: str,
    note: str,
) -> ResponseSplit:
    series = frame[column].astype(str)
    return ResponseSplit(
        train=frame[series.isin(train_labels)].reset_index(drop=True),
        val=frame[series.isin(val_labels)].reset_index(drop=True),
        test=frame[series.isin(test_labels)].reset_index(drop=True),
        strategy=strategy,
        notes=(note,),
    )


def response_split(
    frame: pd.DataFrame,
    *,
    strategy: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    cell_col: str = "cell_line",
    drug_col: str = "drug",
    value_col: str = "value",
) -> ResponseSplit:
    """Split a long-format response table the honest-leakage way.

    DeepResponse's four within-table strategies, each with its
    leakage sentence baked into ``notes``:

    * ``"random"`` -- row-level shuffle (cell lines *and* drugs leak
      across train/test; the optimistic number every paper quotes).
    * ``"cell_stratified"`` -- whole cell lines held out (the
      cell-blind regime; Precily's ``cell_line_split`` is this).
    * ``"drug_stratified"`` -- whole drugs held out (cold-start
      candidates; the TCRP / Precily-Fig4h regime).
    * ``"drug_cell_stratified"`` -- both axes held out disjointly
      (the strictest in-table split; crossed pairs are discarded and
      counted, never silently kept).

    The fifth strategy, ``"cross_domain"`` (train on one panel, test
    on another), needs two tables -- see :func:`cross_domain_split`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> long = pd.DataFrame({
    ...     "cell_line": [f"C{i}" for i in range(8)] * 3,
    ...     "drug": sum([[f"D{j}"] * 8 for j in range(3)], []),
    ...     "value": np.arange(24.0),
    ... })
    >>> split = response_split(long, strategy="drug_stratified", seed=42)
    >>> set(split.train["drug"]).isdisjoint(split.test["drug"])
    True
    >>> len(split.train) + len(split.val) + len(split.test) == len(long)
    True
    >>> strict = response_split(long, strategy="drug_cell_stratified", seed=42)
    >>> bool(strict.notes)
    True
    """
    if strategy not in SPLIT_STRATEGIES:
        raise DataValidationError(
            f"strategy must be one of {SPLIT_STRATEGIES} (cross_domain: "
            "use cross_domain_split), got {strategy!r}"
        )
    for column in (cell_col, drug_col, value_col):
        if column not in frame.columns:
            raise DataValidationError(f"frame is missing the {column!r} column")
    if not 0.0 < test_size < 1.0 or not 0.0 <= val_size < 1.0:
        raise DataValidationError(
            f"test_size/val_size must be fractions, got {test_size}/{val_size}"
        )
    if len(frame) < 3:
        raise DataValidationError(f"frame needs >= 3 rows, got {len(frame)}")

    rng = np.random.default_rng(seed)
    cells = [str(v) for v in frame[cell_col].tolist()]
    drugs = [str(v) for v in frame[drug_col].tolist()]

    if strategy == "random":
        order = rng.permutation(len(frame))
        n_test = max(1, round(test_size * len(frame)))
        n_val = max(1, round(val_size * len(frame)))
        if n_test + n_val >= len(frame):
            raise DataValidationError("random split leaves no training rows")
        return ResponseSplit(
            train=frame.iloc[order[n_test + n_val :]].reset_index(drop=True),
            val=frame.iloc[order[n_test : n_test + n_val]].reset_index(drop=True),
            test=frame.iloc[order[:n_test]].reset_index(drop=True),
            strategy=strategy,
            notes=(
                (
                    "row-level split: cell lines AND drugs appear on both "
                    "sides -- the optimistic (leaky) reference number, "
                    "reported only to be beaten"
                ),
            ),
        )

    if strategy == "cell_stratified":
        train_c, val_c, test_c = _split_axes(cells, test_size, val_size, rng, what="cells")
        return _axis_split(
            frame,
            cell_col,
            train_c,
            val_c,
            test_c,
            strategy,
            (
                "held-out CELL LINES: every test row's cell line is unseen -- "
                "the cell-blind regime a clinical application faces"
            ),
        )

    if strategy == "drug_stratified":
        train_d, val_d, test_d = _split_axes(drugs, test_size, val_size, rng, what="drugs")
        return _axis_split(
            frame,
            drug_col,
            train_d,
            val_d,
            test_d,
            strategy,
            (
                "held-out DRUGS: every test row's drug is unseen -- the "
                "cold-start candidate regime (a feature-free model cannot "
                "be trusted here, only a feature-driven one)"
            ),
        )

    # drug_cell_stratified: both axes disjoint, crossed pairs dropped
    train_c, val_c, test_c = _split_axes(cells, test_size, val_size, rng, what="cells")
    train_d, val_d, test_d = _split_axes(drugs, test_size, val_size, rng, what="drugs")
    cell_series = frame[cell_col].astype(str)
    drug_series = frame[drug_col].astype(str)
    train_mask = cell_series.isin(train_c) & drug_series.isin(train_d)
    val_mask = cell_series.isin(val_c) & drug_series.isin(val_d)
    test_mask = cell_series.isin(test_c) & drug_series.isin(test_d)
    used = int((train_mask | val_mask | test_mask).sum())
    dropped = len(frame) - used
    return ResponseSplit(
        train=frame[train_mask].reset_index(drop=True),
        val=frame[val_mask].reset_index(drop=True),
        test=frame[test_mask].reset_index(drop=True),
        strategy=strategy,
        notes=(
            (
                "held-out cells AND drugs: the strictest within-table split "
                "-- no test row shares a cell line or a drug with any train "
                "row"
            ),
            (
                f"{dropped} crossed pair(s) (train-cell x test-drug and the "
                "mirror) discarded -- counted, never silently kept"
            ),
        ),
    )


def cross_domain_split(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    val_size: float = 0.1,
    seed: int = 42,
    cell_col: str = "cell_line",
    drug_col: str = "drug",
    value_col: str = "value",
) -> ResponseSplit:
    """DeepResponse's fifth strategy: train on one panel, test on another.

    ``source`` (e.g. GDSC) provides train + a drug-blind validation
    carve; ``target`` (e.g. CCLE) is the whole test set -- the
    panel-transfer regime, the hardest of the five.

    Examples
    --------
    >>> import pandas as pd
    >>> source = pd.DataFrame({
    ...     "cell_line": [f"S{i}" for i in range(6)] * 3,
    ...     "drug": sum([[f"D{j}"] * 6 for j in range(3)], []),
    ...     "value": [float(i % 5) for i in range(18)],
    ... })
    >>> target = pd.DataFrame({
    ...     "cell_line": ["T0", "T1"], "drug": ["D9", "D8"], "value": [2.0, 3.0],
    ... })
    >>> split = cross_domain_split(source, target, seed=42)
    >>> split.strategy, len(split.test)
    ('cross_domain', 2)
    """
    for column in (cell_col, drug_col, value_col):
        if column not in source.columns or column not in target.columns:
            raise DataValidationError(
                f"source/target must both carry the {column!r} column"
            )
    if not 0.0 < val_size < 1.0:
        raise DataValidationError(f"val_size must be a fraction, got {val_size}")
    rng = np.random.default_rng(seed)
    drugs = [str(v) for v in source[drug_col].tolist()]
    unique = list(dict.fromkeys(drugs))
    if len(unique) < 3:
        raise DataValidationError(
            f"cross_domain needs >= 3 unique source drugs (found {len(unique)})"
        )
    shuffled = list(rng.permutation(unique))
    n_val = max(1, round(val_size * len(shuffled)))
    val_drugs = set(shuffled[:n_val])
    val_mask = source[drug_col].astype(str).isin(val_drugs)
    return ResponseSplit(
        train=source[~val_mask].reset_index(drop=True),
        val=source[val_mask].reset_index(drop=True),
        test=target.reset_index(drop=True),
        strategy="cross_domain",
        notes=(
            (
                "trained on the SOURCE panel, tested on the TARGET panel -- "
                "the transfer regime (GDSC -> CCLE in DeepResponse's runs)"
            ),
            (
                f"validation carved drug-blind from the source "
                f"({len(val_drugs)} held-out source drugs)"
            ),
        ),
    )


# --------------------------------------------------------------------- #
# Repurposing + the seeded demo screen
# --------------------------------------------------------------------- #


def repurposing_candidates(
    model: ResponseRecommender, features: pd.DataFrame
) -> pd.DataFrame:
    """Rank every modelled drug for one (or more) cell-line profile(s).

    DeepResponse's ``RepurposingInferenceEngine`` scores a configured
    list of (drug, cell line) candidates after every training fold and
    exports the ranked table; here the candidates are the model's
    whole drug axis for whatever profiles are handed in -- the same
    product, minus the JSON config plumbing.  Rows come out
    most-sensitive first (lowest predicted IC50 = rank 1).

    Examples
    --------
    >>> response, features, _truth = synthetic_response_screen(
    ...     6, 4, 3, n_factors=2, seed=9)
    >>> model = fit_response_recommender(
    ...     response, features, n_factors=2, lr=0.05, max_epochs=60,
    ...     seed=9)
    >>> ranked = repurposing_candidates(model, features.iloc[[0]])
    >>> list(ranked.columns)
    ['cell_line', 'drug', 'predicted_ic50', 'rank']
    >>> len(ranked)
    4
    >>> ranked["rank"].tolist()
    [1, 2, 3, 4]
    """
    pred = predict_responses(model, features)
    rows: list[dict[str, object]] = []
    for cell in pred.index:
        order = pred.loc[cell].sort_values()
        for rank, (drug, value) in enumerate(order.items(), start=1):
            rows.append(
                {
                    "cell_line": str(cell),
                    "drug": str(drug),
                    "predicted_ic50": float(value),
                    "rank": rank,
                }
            )
    return pd.DataFrame(
        rows, columns=["cell_line", "drug", "predicted_ic50", "rank"]
    )


def synthetic_response_screen(
    n_cells: int = 12,
    n_drugs: int = 8,
    n_features: int = 6,
    *,
    n_factors: int = 3,
    noise: float = 0.3,
    observed: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A seeded IC50 screen: observed matrix, features, full truth.

    The honest label, stated once: this is a SYNTHETIC screen (latent
    factor structure + Gaussian noise + a masked observation pattern),
    the demo/test stand-in for a GDSC-sized table -- not real
    pharmacology.  ``IC50 = 4 - latent`` keeps values in a
    textbook-ish range where lower = more sensitive.

    Examples
    --------
    >>> response, features, truth = synthetic_response_screen(
    ...     10, 6, 5, n_factors=2, seed=42)
    >>> response.shape, features.shape, truth.shape
    ((10, 6), (10, 5), (10, 6))
    >>> bool(response.isna().any().any())
    True
    >>> bool(truth.notna().all().all())
    True
    >>> list(features.columns)
    ['F1', 'F2', 'F3', 'F4', 'F5']
    """
    if n_cells < 4 or n_drugs < 3 or n_features < 2:
        raise DataValidationError(
            "need >= 4 cells, >= 3 drugs and >= 2 features "
            f"(got {n_cells}/{n_drugs}/{n_features})"
        )
    if not 0.3 <= observed <= 1.0:
        raise DataValidationError(f"observed must be in [0.3, 1.0], got {observed}")
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n_cells, n_features))
    loadings = rng.normal(0.0, 1.0, (n_features, n_factors))
    drug_embed = rng.normal(0.0, 1.0, (n_drugs, n_factors))
    drug_bias = rng.normal(0.0, 0.5, n_drugs)
    latent = (x @ loadings) @ drug_embed.T + drug_bias[None, :]
    latent = latent + rng.normal(0.0, noise, latent.shape)
    ic50 = 4.0 - latent

    mask = np.asarray(rng.random(ic50.shape) < observed, dtype=np.bool_)
    for row in range(n_cells):  # every cell observed somewhere
        if not mask[row].any():
            mask[row, int(rng.integers(0, n_drugs))] = True
    for col in range(n_drugs):  # every drug observed somewhere
        if not mask[:, col].any():
            mask[int(rng.integers(0, n_cells)), col] = True

    cells = [f"C{index + 1:02d}" for index in range(n_cells)]
    drugs = [f"D{index + 1:02d}" for index in range(n_drugs)]
    feature_names = [f"F{k + 1}" for k in range(n_features)]
    response = pd.DataFrame(np.where(mask, ic50, np.nan), index=cells, columns=drugs)
    features = pd.DataFrame(x, index=cells, columns=feature_names)
    truth = pd.DataFrame(ic50, index=cells, columns=drugs)
    return response, features, truth
