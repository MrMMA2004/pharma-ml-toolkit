"""Out-of-fold (OOF) stacking for regression endpoints.

Upgrades look.ipynb (the OOF stacking + residual-correction cell): the
notebook's own three syntax slips (``odel.predict``,
``tuned_meta_models].predict``, ``in(...)`` for ``min(...)``) and its
duplicate physchem descriptor columns (MolWt/LogP/TPSA/HBA/HBD/ROTB
already inside Descriptors.descList) are FIXED here, not ported.

The notebook's pattern, made deterministic and library-shaped:

1. :class:`~sklearn.model_selection.KFold` (shuffled, seeded)
   out-of-fold predictions from the registry base models build an
   honest meta-feature matrix -- no blender ever scores a row it was
   trained on;
2. the base models are refitted on the full training set to produce
   the test meta-features;
3. meta-learners compete on a leaderboard ranked by test R^2;
4. the winning blender gets a Ridge residual-corrector fitted on its
   own OOF residuals; the refined prediction is blender + corrector.

``target_transform="quantile"`` threads the notebook's
``QuantileTransformer(output_distribution="normal")`` through the
TARGET, fitted on the training target only.  HONEST SCALE NOTE: every
score this module reports (``oof_r2``, ``test_r2``,
``residual_test_r2``) is computed on the TRANSFORMED scale when the
transform is active -- the notebook's own convention (look.ipynb never
inverse-transforms); do not compare those numbers against raw-scale
pIC50 R^2 values.  With ``target_transform="none"`` (the default) all
scores are on the raw scale.

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_regression
>>> from pharmaml.ml.stacking import stack_oof_regressors
>>> X, y = make_regression(n_samples=60, n_features=8, noise=4.0, random_state=42)
>>> result = stack_oof_regressors(
...     X[:45], y[:45], X[45:], y[45:],
...     base_models=["ridge", "knn"], meta_models=["ridge"], residual=False,
... )
>>> result.oof_frame.shape                      # 45 train rows, 2 base models
(45, 2)
>>> result.leaderboard["model"].tolist()        # residual=False: metas only
['ridge']
>>> result.residual_test_r2 is None
True
>>> bool(np.isfinite(result.best_test_r2))
True

>>> result = stack_oof_regressors(
...     X[:45], y[:45], X[45:], y[45:],
...     base_models=["ridge"], meta_models=["ridge", "huber_regression"],
...     target_transform="quantile",
... )
>>> result.target_transform, len(result.y_test_trans)
('quantile', 15)
>>> result.leaderboard["model"].tolist()[-1].endswith("+residual")   # appended last
True
>>> result.best_meta in {"ridge", "huber_regression"}
True

>>> stack_oof_regressors(X[:45], y[:45], X[45:], y[45:],
...                      base_models=["ridge"], n_splits=1)
Traceback (most recent call last):
...
pharmaml.exceptions.ModelError: n_splits must be >= 2 folds, got 1
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from pharmaml.exceptions import ModelError
from pharmaml.logger import get_logger
from pharmaml.ml.models import MODEL_REGISTRY, build_model, resolve_model_name

_LOG = get_logger(__name__)

__all__ = ["OOFStackingResult", "stack_oof_regressors"]

#: Meta-learners compared when ``meta_models`` is not given -- the
#: notebook's Ridge / Huber / LightGBM trio.  Entries whose optional
#: backend is missing are dropped (not errored) inside
#: :func:`stack_oof_regressors`.
_DEFAULT_META_MODELS: tuple[str, ...] = ("ridge", "huber_regression", "lightgbm")


@dataclass(slots=True)
class OOFStackingResult:
    """Outcome of one out-of-fold stacking run.

    Attributes
    ----------
    oof_frame:
        Out-of-fold base predictions -- one row per training sample
        (index ``0..n_train-1``), one column per base model, in
        ``base_models`` order.  A blender trained on this matrix never
        scores a fold it was fitted on.
    leaderboard:
        ``model`` / ``oof_r2`` / ``test_r2`` rows, one per
        meta-learner, sorted by ``test_r2`` descending (NaN rows sink;
        the sort is stable, so canonical order breaks ties), plus a
        final ``"<best>+residual"`` row when ``residual=True``.
    best_meta:
        Registry name of the winning meta-learner.
    best_test_r2:
        The winning meta-learner's plain test R^2 (the residual-refined
        score lives in ``residual_test_r2``).
    residual_test_r2:
        Test R^2 of the residual-refined prediction; ``None`` when
        ``residual=False``.
    target_transform:
        ``'none'`` or ``'quantile'`` -- the target scale every reported
        score is computed on (see the module's HONEST SCALE NOTE).
    base_models:
        Canonical registry names of the base (level-0) models actually
        used -- aliases resolved, duplicates collapsed.
    meta_models:
        Canonical registry names of the meta-learners that ran --
        unavailable optional backends filtered out, duplicates
        collapsed.
    n_splits:
        Fold count used for the OOF loop.
    y_test_trans:
        The test target on the reported scale (quantile-transformed
        values when ``target_transform='quantile'``, else raw floats).
    best_test_prediction:
        Test predictions of the final model -- residual-refined when
        ``residual=True``, else the winning meta-learner's own
        predictions.
    """

    oof_frame: pd.DataFrame
    leaderboard: pd.DataFrame
    best_meta: str
    best_test_r2: float
    residual_test_r2: float | None
    target_transform: str
    base_models: list[str]
    meta_models: list[str]
    n_splits: int
    y_test_trans: np.ndarray
    best_test_prediction: np.ndarray


def _notify(
    on_step: Callable[[dict[str, Any]], None] | None,
    *,
    step: str,
    index: int,
    total: int,
    label: str,
) -> None:
    """Invoke the progress callback, swallowing anything it raises.

    A progress callback must never break the run, so its exceptions are
    logged (debug) and dropped -- the caller-side contract the GUI
    already relies on for ``compare_models``' ``on_model`` heartbeat.
    """
    if on_step is None:
        return
    try:
        on_step({"step": step, "index": index, "total": total, "label": label})
    except Exception:  # noqa: BLE001 — a progress callback must never break the run
        _LOG.debug("on_step callback raised during %s; ignored", step, exc_info=True)


def stack_oof_regressors(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    base_models: list[str],
    meta_models: list[str] | None = None,
    n_splits: int = 5,
    target_transform: str = "none",
    residual: bool = True,
    random_state: int = 42,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> OOFStackingResult:
    """Stack registry regressors out-of-fold and rank the blenders.

    Parameters
    ----------
    X_train, y_train:
        Training features and regression target (the OOF matrix, the
        full base refits and every meta fit come from these rows only).
    X_test, y_test:
        Held-out features and target -- the leaderboard's ``test_r2``
        scale.  Never used for fitting anything.
    base_models:
        Registry names of the level-0 learners (aliases resolved,
        duplicates collapsed).  Unknown names raise before any work is
        spent; unavailable optional backends raise through
        :func:`~pharmaml.ml.models.build_model`, also before any
        fitting.
    meta_models:
        Registry names of the level-1 blenders; defaults to
        ``['ridge', 'huber_regression', 'lightgbm']`` filtered to the
        *available* registry entries, so a missing lightgbm degrades
        instead of erroring.  A meta-learner that fails to build or fit
        keeps its row with NaN scores and sinks to the bottom of the
        leaderboard (the ``compare_models`` convention).
    n_splits:
        OOF fold count (at least 2, at most ``len(X_train)``).
    target_transform:
        ``'none'`` or ``'quantile'``.  With ``'quantile'`` a
        :class:`~sklearn.preprocessing.QuantileTransformer`
        (``output_distribution='normal'``, seeded) is fitted on the
        TRAINING target only and applied to both targets; every
        reported score is then on the TRANSFORMED scale (the
        notebook's own convention -- see the module's HONEST SCALE
        NOTE).
    residual:
        Fit a ``Ridge(alpha=1.0)`` corrector on the best blender's OOF
        residuals and append the ``"<best>+residual"`` leaderboard row;
        ``best_test_prediction`` carries the refined values.
    random_state:
        Seed for the KFold shuffle, the quantile transform and every
        estimator that accepts one -- the whole run is deterministic.
    on_step:
        Optional progress callback receiving
        ``{'step': 'oof'|'base'|'meta'|'residual', 'index': i,
        'total': n, 'label': str}`` -- one call per OOF fold, per
        full-train base refit, per meta-learner and (when active) one
        for the residual step.  It must never raise; if it does, the
        exception is swallowed and the run continues.

    Returns
    -------
    OOFStackingResult
        With ``residual=False`` the leaderboard holds exactly one row
        per healthy-or-not meta-learner and no residual row.

    Raises
    ------
    ModelError
        When ``base_models`` is empty (the message lists the registered
        names), a base or meta name is unknown, ``n_splits`` is below 2
        or exceeds the training row count, ``target_transform`` is not
        ``'none'``/``'quantile'``, the X/y inputs are non-numeric or
        shape-mismatched, a target contains NaN, ``y_test`` is empty,
        or every meta-learner failed on the OOF matrix.
    """
    if target_transform not in {"none", "quantile"}:
        raise ModelError(
            f"target_transform must be 'none' or 'quantile', got {target_transform!r}"
        )
    if not base_models:
        raise ModelError(
            "base_models list is empty -- pass at least one registry name "
            f"(registered models: {', '.join(sorted(MODEL_REGISTRY))})"
        )
    # Unknown base names raise here, through resolve_model_name's own
    # ModelError, BEFORE any estimator is built or fitted.
    base_keys = list(dict.fromkeys(resolve_model_name(name) for name in base_models))

    try:
        Xtr = np.asarray(X_train, dtype=float)
        ytr = np.asarray(y_train, dtype=float)
        Xte = np.asarray(X_test, dtype=float)
        yte = np.asarray(y_test, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelError(f"X/y inputs must be numeric tables: {exc}") from exc

    if Xtr.ndim != 2 or Xte.ndim != 2:
        raise ModelError(
            f"X_train/X_test must be 2-D feature tables, got shapes {Xtr.shape} "
            f"and {Xte.shape}"
        )
    if ytr.ndim != 1:
        raise ModelError(f"y_train must be 1-D, got shape {ytr.shape}")
    if yte.ndim != 1:
        raise ModelError(f"y_test must be 1-D, got shape {yte.shape}")
    n_train = int(Xtr.shape[0])
    n_test = int(Xte.shape[0])
    if n_train != ytr.shape[0]:
        raise ModelError(
            f"X_train has {n_train} row(s) but y_train has {ytr.shape[0]} value(s)"
        )
    if n_test != yte.shape[0]:
        raise ModelError(
            f"X_test has {n_test} row(s) but y_test has {yte.shape[0]} value(s)"
        )
    if Xte.shape[1] != Xtr.shape[1]:
        raise ModelError(
            f"X_test has {Xte.shape[1]} column(s) but X_train has {Xtr.shape[1]} -- "
            "the test meta-features need the same feature columns"
        )
    for label, values in (("y_train", ytr), ("y_test", yte)):
        n_missing = int(np.isnan(values).sum())
        if n_missing:
            raise ModelError(
                f"{label} contains {n_missing} NaN value(s) -- drop or impute the "
                "target before stacking"
            )
    if yte.size == 0:
        raise ModelError(
            "y_test is empty -- the stacking leaderboard needs at least one "
            "held-out sample"
        )
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise ModelError(f"n_splits must be an int >= 2, got {n_splits!r}")
    if n_splits < 2:
        raise ModelError(f"n_splits must be >= 2 folds, got {n_splits}")
    if n_splits > n_train:
        raise ModelError(
            f"n_splits={n_splits} exceeds the number of training rows "
            f"(n_train={n_train}) -- no fold can be built; lower n_splits or "
            "supply more data"
        )

    # Build every base estimator up front so an unavailable optional
    # backend (DependencyError) or a task-incompatible name (ModelError)
    # fails BEFORE any OOF fitting work is spent -- the registry
    # convention compare_models and build_stacking already follow.
    base_estimators = [build_model(key, "regression") for key in base_keys]

    if meta_models is None:
        requested_meta = list(_DEFAULT_META_MODELS)
    else:
        requested_meta = list(meta_models)
    # Unknown meta names raise the same typed error as base names ...
    resolved_meta = [resolve_model_name(name) for name in requested_meta]
    # ... but KNOWN-but-unavailable optional backends degrade instead of
    # erroring, so the default list survives a missing lightgbm.
    meta_keys = [key for key in resolved_meta if MODEL_REGISTRY[key].available]
    dropped_meta = [key for key in resolved_meta if key not in meta_keys]
    if dropped_meta:
        _LOG.warning(
            "Dropping unavailable meta-learner(s) %s -- install the optional "
            "backend or pass meta_models explicitly to override the default",
            sorted(set(dropped_meta)),
        )
    if not meta_keys:
        _LOG.warning("no available meta-learner left; falling back to 'ridge'")
        meta_keys = ["ridge"]
    meta_keys = list(dict.fromkeys(meta_keys))

    if target_transform == "quantile":
        transformer = QuantileTransformer(
            output_distribution="normal",
            random_state=random_state,
            n_quantiles=int(min(1000, n_train)),
        )
        # fitted on the TRAINING target only; y_test is transformed with
        # the train-fitted quantiles, never refit.
        ytr_trans = transformer.fit_transform(ytr.reshape(-1, 1)).ravel()
        yte_trans = transformer.transform(yte.reshape(-1, 1)).ravel()
    else:
        ytr_trans = ytr
        yte_trans = yte

    _LOG.info(
        "OOF stacking: %d base model(s) x %d folds, %d meta-learner(s) "
        "(target_transform=%s, residual=%s)",
        len(base_keys), n_splits, len(meta_keys), target_transform, residual,
    )

    # -- OOF matrix (the notebook's step 10) ---------------------------
    oof = np.zeros((n_train, len(base_estimators)), dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_no, (fit_idx, val_idx) in enumerate(splitter.split(Xtr), start=1):
        X_fit, X_val = Xtr[fit_idx], Xtr[val_idx]
        y_fit = ytr_trans[fit_idx]
        for column, estimator in enumerate(base_estimators):
            fold_estimator = clone(estimator)
            fold_estimator.fit(X_fit, y_fit)
            oof[val_idx, column] = fold_estimator.predict(X_val)
        _notify(
            on_step,
            step="oof",
            index=fold_no,
            total=n_splits,
            label=f"fold {fold_no}/{n_splits}",
        )

    # -- full-train refits -> test meta-features (notebook step 12/13) --
    test_meta = np.zeros((n_test, len(base_estimators)), dtype=float)
    for column, (key, estimator) in enumerate(zip(base_keys, base_estimators)):
        full_estimator = clone(estimator)
        full_estimator.fit(Xtr, ytr_trans)
        test_meta[:, column] = full_estimator.predict(Xte)
        _notify(on_step, step="base", index=column + 1, total=len(base_keys), label=key)

    # The notebook scales the meta-feature matrix itself -- mirrored.
    meta_scaler = StandardScaler()
    oof_scaled = meta_scaler.fit_transform(oof)
    test_meta_scaled = meta_scaler.transform(test_meta)

    # -- meta-learner leaderboard (notebook step 11 + 14) --------------
    rows: list[dict[str, Any]] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for position, key in enumerate(meta_keys, start=1):
        try:
            meta = clone(build_model(key, "regression"))
            meta.fit(oof_scaled, ytr_trans)
            oof_pred = np.asarray(meta.predict(oof_scaled), dtype=float).ravel()
            test_pred = np.asarray(meta.predict(test_meta_scaled), dtype=float).ravel()
            rows.append(
                {
                    "model": key,
                    "oof_r2": float(r2_score(ytr_trans, oof_pred)),
                    "test_r2": float(r2_score(yte_trans, test_pred)),
                }
            )
            predictions[key] = (oof_pred, test_pred)
        except Exception as exc:  # noqa: BLE001 — keep the leaderboard alive on failures
            _LOG.warning("Meta-learner %s failed during stacking: %s", key, exc)
            rows.append({"model": key, "oof_r2": np.nan, "test_r2": np.nan})
        _notify(on_step, step="meta", index=position, total=len(meta_keys), label=key)

    if not predictions:
        raise ModelError(
            "all meta-learners failed on the OOF matrix -- no blender could be "
            "trained; the per-learner exceptions were logged as warnings "
            "(degenerate OOF predictions or task-incompatible meta names are "
            "the usual causes)"
        )

    board = pd.DataFrame(rows, columns=["model", "oof_r2", "test_r2"])
    board = board.sort_values(
        "test_r2", ascending=False, na_position="last", kind="stable"
    ).reset_index(drop=True)
    best_meta = str(board.iloc[0]["model"])
    best_oof_pred, best_test_pred = predictions[best_meta]
    best_test_r2 = float(board.iloc[0]["test_r2"])

    # -- residual correction (notebook step 16, honestly fitted) --------
    residual_test_r2: float | None = None
    best_prediction = best_test_pred
    if residual:
        # The Ridge corrector learns the best blender's OOF residuals on
        # the TRAINING side only, then adds its meta-feature-driven
        # estimate to both the blender's OOF and test predictions.
        residual_model = Ridge(alpha=1.0)
        residual_model.fit(oof_scaled, ytr_trans - best_oof_pred)
        refined_oof = best_oof_pred + np.asarray(
            residual_model.predict(oof_scaled), dtype=float
        ).ravel()
        refined_test = best_test_pred + np.asarray(
            residual_model.predict(test_meta_scaled), dtype=float
        ).ravel()
        residual_test_r2 = float(r2_score(yte_trans, refined_test))
        board = pd.concat(
            [
                board,
                pd.DataFrame(
                    [
                        {
                            "model": f"{best_meta}+residual",
                            "oof_r2": float(r2_score(ytr_trans, refined_oof)),
                            "test_r2": residual_test_r2,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        best_prediction = refined_test
        _notify(
            on_step,
            step="residual",
            index=1,
            total=1,
            label=f"{best_meta}+residual",
        )

    return OOFStackingResult(
        oof_frame=pd.DataFrame(oof, columns=list(base_keys)),
        leaderboard=board,
        best_meta=best_meta,
        best_test_r2=best_test_r2,
        residual_test_r2=residual_test_r2,
        target_transform=target_transform,
        base_models=list(base_keys),
        meta_models=list(meta_keys),
        n_splits=n_splits,
        y_test_trans=np.asarray(yte_trans, dtype=float),
        best_test_prediction=np.asarray(best_prediction, dtype=float),
    )
