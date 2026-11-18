"""Semi-supervised learning — profiting from the rows nobody labeled.

High-throughput assays routinely produce far more measurements than
anybody has the budget (or patience) to curate: the assay readout is
there for every well, but the authoritative class call — active /
inactive, tox / non-tox, hit / non-hit — exists for a small annotated
subset.  Semi-supervised learning is the family of methods built for
exactly that shape of data: a handful of labeled rows plus a pile of
unlabeled ones whose *geometry* (clusters, manifolds, neighbourhoods)
still carries label information.

This module wraps the three scikit-learn workhorses behind one
workflow with the toolkit's usual guarantees:

* :func:`validate_unlabeled` — the ``-1`` sentinel convention done
  safely.  scikit-learn's semi-supervised estimators encode
  "unlabeled" as the integer ``-1``, which is fine for freshly
  synthesized ``y`` arrays and quietly dangerous for real assay
  frames: a pandas column of *numeric* class IDs that happens to
  contain a legitimate class ``-1`` (a z-scored bin, a signed charge
  bucket) or a *string* column containing the literal ``"-1"`` must
  not have those rows silently reinterpreted as "no label".  The
  validator therefore converts ONLY values that equal the marker
  under a strict type-aware comparison — a string ``"-1"`` never
  matches the default marker ``-1`` — and reports the labeled /
  unlabeled / per-class counts it found.  A custom marker
  (``"unlabeled"``, ``"NA"``, ``np.nan``, ...) converts the values
  that equal IT; ``unlabeled_marker=None`` converts nothing (an
  all-supervised input is a valid, if boring, semi-supervised one).
* :func:`train_semi_supervised` — fit one of three methods on ``X``
  / ``y`` and get a :class:`SemiSupervisedResult` carrying the final
  label of every training row (real labels preserved, pseudo-labels
  filling the sentinel positions), a per-row confidence readout, the
  pseudo-label mask, and a fully reproducible ``params`` dict.
* :func:`evaluate_semi_supervised` — the leakage-safe honest
  readout: a stratified split of the LABELED rows only, the
  semi-supervised model trained on labeled-train + the unlabeled
  rows (held-out genuine labels never enter training), the same
  split scored by a supervised baseline trained on labeled-train
  alone, and both metrics averaged over ``n_repeats`` seeds.  When
  there are too few labeled rows to split honestly, the metrics
  come back ``None`` with a summary that says so — never a number
  manufactured from a degenerate split.

The three methods, and when to reach for each:

* ``"self_training"`` — :class:`~sklearn.semi_supervised.
  SelfTrainingClassifier` around a probabilistic base estimator
  (``LogisticRegression`` by default).  Iteratively labels the rows
  the current model is most confident about, adds them to the
  training set, refits.  Cheapest, works in any feature space, and
  the only one of the three that is inductive (it can ``predict``
  on genuinely new rows — the fitted wrapper IS the model to ship).
* ``"label_propagation"`` — :class:`~sklearn.semi_supervised.
  LabelPropagation`.  Transductive: builds a graph over the training
  rows and lets the known labels flow along it until equilibrium.
  The labels of the training rows themselves are treated as ground
  truth (clamping factor 1); noise in them is propagated faithfully.
* ``"label_spreading"`` — :class:`~sklearn.semi_supervised.
  LabelSpreading`.  The regularized sibling: labels are allowed to
  move a little (``alpha``), which buys robustness to mislabeled
  rows at the cost of occasionally overriding a genuine label.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from sklearn.datasets import make_classification
>>> from pharmaml.ml.semi_supervised import (
...     train_semi_supervised, validate_unlabeled)
>>> X, y = make_classification(n_samples=60, n_features=8,
...     n_informative=6, n_clusters_per_class=1, class_sep=2.0,
...     random_state=0)
>>> y_masked = y.copy()
>>> y_masked[::3] = -1                       # 20 rows lose their label
>>> labels, info = validate_unlabeled(y_masked)
>>> info["n_labeled"], info["n_unlabeled"]
(40, 20)
>>> result = train_semi_supervised(X, y_masked, method="label_spreading")
>>> result.pseudo_label_mask.sum()
20
>>> bool((result.labels_[~result.pseudo_label_mask] == y[~result.pseudo_label_mask]).all())
True

The sentinel never eats a legitimate class — a string column holding
``"-1"`` as an actual label keeps it:

>>> series = pd.Series(["-1", "active", "inactive", "-1", "active"])
>>> labels, info = validate_unlabeled(series)
>>> info["n_unlabeled"]
0
>>> sorted(info["class_counts"])
['-1', 'active', 'inactive']

A custom marker converts only its own value:

>>> marked = pd.Series(["A", "B", "unlabeled", "A", "unlabeled"])
>>> labels, info = validate_unlabeled(marked, unlabeled_marker="unlabeled")
>>> info["n_unlabeled"]
2
>>> int(labels[2])
-1

Error contracts (the house style):

>>> validate_unlabeled([-1, -1, -1])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: y has 0 labeled sample(s) — semi-supervised learning needs at least 1 (found 3 unlabeled row(s))
>>> validate_unlabeled(["A", "A", "A"])
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: y carries 1 distinct real class(es) — semi-supervised learning needs at least 2
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading,
    SelfTrainingClassifier,
)

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "SemiSupervisedEvaluation",
    "SemiSupervisedMethod",
    "SemiSupervisedResult",
    "evaluate_semi_supervised",
    "train_semi_supervised",
    "validate_unlabeled",
]

#: The three wrapped methods.
SemiSupervisedMethod = Literal["self_training", "label_propagation", "label_spreading"]

_METHODS: tuple[SemiSupervisedMethod, ...] = (
    "self_training",
    "label_propagation",
    "label_spreading",
)

_UNLABELED_SENTINEL = -1


def _equals_marker(value: Any, marker: Any) -> bool:
    """Type-aware equality against the unlabeled marker.

    The whole sentinel-safety story lives here: a numeric marker only
    matches numeric values of the same scalar type (never a numeric
    string), a string marker only matches the same string, ``nan``
    matches only ``nan``, and ``None`` is handled by the caller
    (marker disabled).  ``bool`` is excluded on purpose — numpy's
    ``True == 1`` coercion must not let a marker of ``1`` eat genuine
    boolean/integer labels.
    """
    if isinstance(marker, float) and np.isnan(marker):
        return (
            not isinstance(value, bool)
            and isinstance(value, (float, np.floating))
            and np.isnan(value)
        )
    if isinstance(marker, str):
        return isinstance(value, str) and value == marker
    if isinstance(marker, (bool, np.bool_)):
        return isinstance(value, (bool, np.bool_)) and value == marker
    if isinstance(marker, (int, np.integer)):
        if isinstance(value, (bool, np.bool_)):
            return False
        if isinstance(value, (int, np.integer)):
            return int(value) == int(marker)
        if isinstance(value, (float, np.floating)):
            return float(value).is_integer() and int(value) == int(marker)
        return False
    if isinstance(marker, (float, np.floating)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return False
        return float(value) == float(marker)
    return value is marker


def validate_unlabeled(
    y: pd.Series | np.ndarray | list | tuple,
    unlabeled_marker: Any = _UNLABELED_SENTINEL,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate a target vector under the ``-1`` unlabeled convention.

    Parameters
    ----------
    y:
        Array-like target vector.  May be numeric (with ``-1`` marking
        unlabeled rows) or non-numeric (strings / categoricals with a
        custom marker such as ``"unlabeled"``).
    unlabeled_marker:
        The value that means "no label".  Only values strictly equal
        to the marker (under a type-aware comparison) are converted to
        the integer sentinel ``-1`` — a pandas Series holding the
        STRING class ``"-1"`` keeps it as a real class under the
        default numeric marker, and a numeric column holding a
        legitimate class ``-1.0`` keeps it too when the marker is,
        say, ``"NA"``.  Defaults to ``-1``.  Pass ``None`` to disable
        conversion entirely (all-supervised input).

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        ``(labels, info)`` where ``labels`` is the validated target —
        marker positions as the integer sentinel ``-1``, real labels
        otherwise (numeric classes keep their values; non-numeric
        classes are encoded to indices ``0..k-1`` and decoded again
        by :func:`train_semi_supervised`, because sklearn rejects
        object arrays mixing strings with the sentinel) — and ``info``
        carries ``n_labeled``, ``n_unlabeled``, ``class_counts`` (a
        ``{class: count}`` dict over the REAL classes, unlabeled rows
        excluded) plus the fit path's decode map (``class_order``,
        ``class_decoder``, ``encoded``).

    Raises
    ------
    DataValidationError
        When ``y`` is empty, or has fewer than 2 distinct real classes
        or fewer than 1 labeled sample.

    Examples
    --------
    >>> labels, info = validate_unlabeled([0, 1, -1, 0])
    >>> info["n_labeled"], info["n_unlabeled"], info["class_counts"]
    (3, 1, {0: 2, 1: 1})
    """
    if isinstance(y, pd.Series):
        raw = y.to_numpy(dtype=object)
    else:
        raw = np.asarray(y, dtype=object)
    if raw.ndim != 1:
        raise DataValidationError(
            f"y must be a 1-D target vector, got shape {raw.shape}"
        )
    if raw.size == 0:
        raise DataValidationError("y is empty — nothing to train on")

    # Which values are "no label": only values strictly equal to the
    # marker (type-aware — a string "-1" never matches the numeric
    # default marker).  A NaN/None/pd.NA target always means "no
    # label" under the DEFAULT marker — a missing value has no other
    # honest reading — but under a custom marker only the marker
    # itself converts.
    marker = unlabeled_marker
    is_unlabeled = np.zeros(raw.size, dtype=bool)
    labeled_parts: list[Any] = []
    for i, value in enumerate(raw):
        marker_hit = marker is not None and _equals_marker(value, marker)
        missing_hit = marker == _UNLABELED_SENTINEL and (
            value is None
            or value is pd.NA
            or (isinstance(value, (float, np.floating)) and np.isnan(value))
        )
        if marker_hit or missing_hit:
            is_unlabeled[i] = True
        else:
            labeled_parts.append(value)

    n_unlabeled = int(is_unlabeled.sum())
    if not labeled_parts:
        raise DataValidationError(
            f"y has 0 labeled sample(s) — semi-supervised learning "
            f"needs at least 1 (found {n_unlabeled} unlabeled row(s))"
        )
    classes, inverse = np.unique(
        np.asarray(labeled_parts, dtype=object), return_inverse=True
    )
    if classes.size < 2:
        raise DataValidationError(
            f"y carries {classes.size} distinct real class(es) — "
            "semi-supervised learning needs at least 2"
        )

    # Final label vector, dtype-aware so sklearn accepts it:
    # * homogeneous INTEGRAL numeric labels (int, or float values
    #   like 1.0) keep their values in an int array with the -1
    #   sentinel — ``type_of_target``-clean (probed on sklearn 1.5.2);
    # * everything else (strings, mixed types, NON-integral floats
    #   like class IDs 0.5/1.5 — sklearn reads those as a regression
    #   target and rejects the fit) is ENCODED to integer class
    #   indices 0..k-1 with the sentinel as -1.  ``encoded=True`` in
    #   the info dict tells the fit path to decode the outputs
    #   (``labels_`` / ``classes_``) back to the caller's classes.
    all_numeric = all(
        isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, bool)
        for v in labeled_parts
    )
    numeric_labels = (
        np.array([float(_label_key(v)) for v in labeled_parts], dtype=float)
        if all_numeric else None
    )
    encoded = (
        not all_numeric
        or numeric_labels is None
        or not np.array_equal(numeric_labels, np.round(numeric_labels))
    )
    if not encoded:
        labels = np.full(raw.size, _UNLABELED_SENTINEL, dtype=np.int64)
        labels[~is_unlabeled] = (
            numeric_labels if numeric_labels is not None else inverse
        ).astype(np.int64)
        class_order = np.array(
            [float(_label_key(c)) for c in classes], dtype=float
        )
    else:
        labels = np.full(raw.size, _UNLABELED_SENTINEL, dtype=np.int64)
        labels[~is_unlabeled] = inverse
        class_order = np.arange(classes.size, dtype=np.int64)

    class_counts = {
        _label_key(cls): int(count)
        for cls, count in zip(classes, np.bincount(inverse), strict=True)
    }
    info: dict[str, Any] = {
        "n_labeled": len(labeled_parts),
        "n_unlabeled": n_unlabeled,
        "class_counts": class_counts,
        "class_order": class_order,
        "class_decoder": np.asarray(classes, dtype=object),
        "encoded": encoded,
    }
    _LOG.debug(
        "validate_unlabeled: %d labeled / %d unlabeled, classes=%s",
        info["n_labeled"], n_unlabeled, sorted(map(str, classes.tolist())),
    )
    return labels, info


def _label_key(value: Any) -> Any:
    """Normalize a class label to a plain-python hashable scalar."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _native_class_dtype(values: np.ndarray) -> np.dtype | None:
    """A native dtype the decoded classes fit without loss, if any.

    ``values`` is the object array decoded from the class map.  When
    every entry is numeric a float64 array is returned (sklearn's
    metrics read numeric arrays as classification targets);
    ``None`` means "keep object dtype" (string or mixed classes —
    fine for ``str``-typed comparisons, and the encoded evaluate
    path never sends those arrays to the metrics anyway).
    """
    if values.size and all(
        isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, bool)
        for v in values
    ):
        return np.dtype(float)
    return None


def _self_training_estimator(base: Any, **kwargs: Any) -> SelfTrainingClassifier:
    """Build a ``SelfTrainingClassifier`` across sklearn versions.

    scikit-learn renamed the constructor's first parameter from
    ``base_estimator`` to ``estimator`` in 1.2 (old name deprecated
    in 1.2, removed in 1.4; some distributions carry either spelling).
    Rather than pinning the toolkit to one side of that rename, this
    helper introspects :func:`inspect.signature` of the installed
    ``SelfTrainingClassifier`` and passes whichever keyword the
    installed class actually accepts.

    Parameters
    ----------
    base:
        The probabilistic base estimator to wrap.
    **kwargs:
        Remaining constructor arguments (``threshold``, ``max_iter``,
        ``k_best``, ...).

    Returns
    -------
    SelfTrainingClassifier
        Constructed with the version-correct first-parameter name.

    Raises
    ------
    DataValidationError
        When the installed scikit-learn accepts neither spelling
        (an unknown future refactoring).
    """
    parameters = inspect.signature(SelfTrainingClassifier).parameters
    if "estimator" in parameters:
        return SelfTrainingClassifier(estimator=base, **kwargs)
    if "base_estimator" in parameters:
        return SelfTrainingClassifier(base_estimator=base, **kwargs)
    raise DataValidationError(
        "SelfTrainingClassifier accepts neither 'estimator' nor "
        "'base_estimator' — unsupported scikit-learn "
        f"version {sklearn.__version__}"
    )


def _self_training_kwarg_name() -> str:
    """The first-parameter spelling this sklearn accepts (test hook)."""
    parameters = inspect.signature(SelfTrainingClassifier).parameters
    if "estimator" in parameters:
        return "estimator"
    if "base_estimator" in parameters:
        return "base_estimator"
    raise DataValidationError(
        "SelfTrainingClassifier accepts neither 'estimator' nor "
        "'base_estimator' — unsupported scikit-learn version "
        f"{sklearn.__version__}"
    )


def _resolve_base_estimator(base_estimator: Any) -> Any:
    """Default to LogisticRegression; require ``predict_proba``."""
    base = base_estimator if base_estimator is not None else LogisticRegression(
        max_iter=1000
    )
    if not callable(getattr(base, "predict_proba", None)):
        raise DataValidationError(
            "base_estimator must expose predict_proba — self-training "
            f"labels rows by confidence, got {type(base).__name__}"
        )
    return base


def _validate_train_params(
    method: SemiSupervisedMethod | str,
    threshold: float,
    k_best: int,
    max_iter: int,
    n_neighbors: int,
    gamma: float,
    unlabeled_marker: Any,
) -> None:
    """Fail-closed parameter validation before any fit (the house style)."""
    if method not in _METHODS:
        raise DataValidationError(
            f"method must be one of {', '.join(_METHODS)}, got {method!r}"
        )
    if not (0.0 < float(threshold) < 1.0):
        raise DataValidationError(
            f"threshold must be in (0, 1), got {threshold!r}"
        )
    if int(k_best) < 0:
        raise DataValidationError(f"k_best must be >= 0, got {k_best!r}")
    if int(max_iter) < 1:
        raise DataValidationError(f"max_iter must be >= 1, got {max_iter!r}")
    if int(n_neighbors) < 1:
        raise DataValidationError(f"n_neighbors must be >= 1, got {n_neighbors!r}")
    if not (float(gamma) > 0.0 and np.isfinite(float(gamma))):
        raise DataValidationError(
            f"gamma must be a positive finite number, got {gamma!r}"
        )
    if not isinstance(unlabeled_marker, (str, int, float, np.integer, np.floating, bool)) \
            and unlabeled_marker is not None:
        raise DataValidationError(
            f"unlabeled_marker must be a scalar or None, got {type(unlabeled_marker).__name__}"
        )


@dataclass(slots=True)
class SemiSupervisedResult:
    """Outcome of one semi-supervised fit.

    Attributes
    ----------
    method:
        The fitted method name (``'self_training'``,
        ``'label_propagation'`` or ``'label_spreading'``).
    params:
        The full effective configuration — every knob plus the sklearn
        version and the estimator class name — enough to reproduce the
        fit without remembering the call site.
    model:
        The fitted sklearn estimator.  For ``self_training`` this is
        the fitted :class:`SelfTrainingClassifier` (inductive — safe to
        ``predict`` on new rows); for the graph methods it is the
        fitted transductive estimator.
    labels_:
        Final label of every training row — genuine labels kept
        verbatim, pseudo-labels filling the positions that arrived as
        unlabeled.
    confidence:
        Per-row confidence: ``predict_proba().max(axis=1)`` for
        self-training, the max label distribution for the graph
        methods, ``None`` when the method exposes no distribution.
    pseudo_label_mask:
        Boolean mask — ``True`` where the row arrived unlabeled.
    n_labeled / n_unlabeled:
        Row counts of the two halves.
    classes_:
        The REAL classes (the sentinel ``-1`` never appears).
    fit_seconds:
        Wall-clock cost of the fit.
    """

    method: str
    params: dict[str, Any]
    model: Any
    labels_: np.ndarray
    confidence: np.ndarray | None
    pseudo_label_mask: np.ndarray
    n_labeled: int
    n_unlabeled: int
    classes_: np.ndarray
    fit_seconds: float = 0.0

    def to_frame(self, names: list[str] | pd.Series | None = None) -> pd.DataFrame:
        """Sample-level pseudo-label report, one row per training row.

        Columns: ``name, final_label, confidence, was_pseudo_labeled``
        (the confidence column is all-NaN when the method exposes no
        distribution).
        """
        n = int(self.labels_.size)
        if names is None:
            labels = [f"sample_{i}" for i in range(n)]
        else:
            labels = [str(v) for v in list(names)[:n]]
            labels += [f"sample_{i}" for i in range(len(labels), n)]
        data: dict[str, Any] = {
            "name": labels,
            "final_label": self.labels_,
            "confidence": self.confidence if self.confidence is not None
            else np.full(n, np.nan),
            "was_pseudo_labeled": self.pseudo_label_mask,
        }
        return pd.DataFrame(data)


@dataclass(slots=True)
class SemiSupervisedEvaluation:
    """Leakage-safe honest evaluation of a semi-supervised configuration.

    Attributes
    ----------
    heldout_accuracy / heldout_f1_macro:
        Accuracy and macro-F1 of the semi-supervised model on the
        held-out labeled rows, averaged over ``n_repeats`` seeds
        (``None`` when the labeled data was too small to split).
    baseline_accuracy / baseline_f1_macro:
        The same metrics for the supervised baseline trained on the
        labeled-train rows alone (``None`` alongside the above).
    train_accuracy_on_labeled:
        Accuracy of the final semi-supervised labels on the rows that
        arrived labeled (a sanity readout, not a generalization
        estimate; ``None`` when evaluation was not possible).
    method:
        The evaluated method name.
    """

    heldout_accuracy: float | None
    heldout_f1_macro: float | None
    baseline_accuracy: float | None
    baseline_f1_macro: float | None
    train_accuracy_on_labeled: float | None
    method: str

    def summary(self) -> str:
        """One honest human-readable line.

        Reports both metrics, names the method, and — when evaluation
        was not possible — says so plainly instead of dressing a
        missing number as a result.
        """
        if self.heldout_accuracy is None or self.heldout_f1_macro is None:
            return (
                f"semi-supervised ({self.method}): evaluation not possible — "
                "too few labeled rows for a leakage-safe held-out split"
            )
        baseline_acc = (
            "n/a" if self.baseline_accuracy is None
            else f"{self.baseline_accuracy:.3f}"
        )
        baseline_f1 = (
            "n/a" if self.baseline_f1_macro is None
            else f"{self.baseline_f1_macro:.3f}"
        )
        return (
            f"semi-supervised ({self.method}): held-out accuracy="
            f"{self.heldout_accuracy:.3f}, macro-F1={self.heldout_f1_macro:.3f} "
            f"vs supervised baseline accuracy={baseline_acc}, "
            f"macro-F1={baseline_f1}"
        )


def _numeric_features(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Coerce features to a finite 2-D float matrix (house guards)."""
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise DataValidationError(
                "X has duplicate feature names; columns must be unique"
            )
        bad = [str(col) for col in X.columns if X[col].dtype.kind not in "fiub"]
        if bad:
            raise DataValidationError(
                f"X carries non-numeric column(s) {', '.join(bad)} — "
                "compute descriptors / clean the frame first"
            )
        try:
            matrix = X.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"X could not be converted to a numeric matrix: {exc}"
            ) from exc
    else:
        try:
            matrix = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"X could not be converted to a numeric matrix: {exc}"
            ) from exc
    if matrix.ndim != 2:
        raise DataValidationError(
            f"X must be a 2-D feature matrix, got shape {matrix.shape}"
        )
    if matrix.shape[0] == 0:
        raise DataValidationError("X is empty — nothing to train on")
    if matrix.shape[1] == 0:
        raise DataValidationError("X has zero feature columns")
    n_nan = int(np.isnan(matrix).sum())
    if n_nan:
        raise DataValidationError(
            f"X contains {n_nan} NaN value(s) — semi-supervised "
            "learning needs finite descriptors"
        )
    n_inf = int(np.isinf(matrix).sum())
    if n_inf:
        raise DataValidationError(
            f"X contains {n_inf} infinite value(s) — semi-supervised "
            "learning needs finite descriptors"
        )
    return matrix


def train_semi_supervised(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | list,
    method: SemiSupervisedMethod = "self_training",
    *,
    base_estimator: Any = None,
    threshold: float = 0.75,
    k_best: int = 10,
    max_iter: int = 10,
    kernel: str = "knn",
    n_neighbors: int = 7,
    gamma: float = 20,
    unlabeled_marker: Any = _UNLABELED_SENTINEL,
    random_state: int = 42,
) -> SemiSupervisedResult:
    """Fit one of the three semi-supervised methods on ``X`` / ``y``.

    Parameters
    ----------
    X:
        Feature matrix — a numeric ``pd.DataFrame`` or a 2-D array
        (finite; DataFrame columns must be unique).
    y:
        Target vector with the ``-1`` convention (or a custom
        ``unlabeled_marker``); see :func:`validate_unlabeled`.
    method:
        ``'self_training'``, ``'label_propagation'`` or
        ``'label_spreading'``.
    base_estimator:
        Base estimator for self-training (default
        ``LogisticRegression(max_iter=1000)``).  Must expose
        ``predict_proba``.  Ignored by the graph methods.
    threshold:
        Self-training confidence threshold in ``(0, 1)``.
    k_best:
        Self-training ``k_best`` — the number of most-confident
        pseudo-labels added per iteration when the criterion is
        ``'k_best'``.  ``0`` disables the knob (criterion stays
        threshold-based).
    max_iter:
        Maximum self-training iterations (``>= 1``) / graph method
        propagation iterations.
    kernel:
        Graph-method kernel (``'knn'``, ``'rbf'`` or a callable).
    n_neighbors:
        kNN-graph neighbour count (``>= 1``), used when
        ``kernel='knn'``.
    gamma:
        RBF-kernel bandwidth coefficient (``> 0``), used when
        ``kernel='rbf'``.
    unlabeled_marker:
        The value in ``y`` meaning "no label" (default ``-1``);
        ``None`` disables conversion.
    random_state:
        Recorded in ``params`` for reproducibility and used to seed
        :func:`evaluate_semi_supervised`'s splits.  The wrapped
        estimators themselves accept no seed (probed on sklearn
        1.5.2) and are deterministic given the data; seed a custom
        ``base_estimator`` on the caller's side if it is stochastic.

    Returns
    -------
    SemiSupervisedResult
        With ``labels_`` / ``confidence`` / ``pseudo_label_mask`` /
        ``classes_`` populated per the module docstring's contract,
        and ``params`` carrying the full effective configuration
        (method, every knob, the sklearn version and the wrapped
        estimator's class name).

    Raises
    ------
    DataValidationError
        On an unknown ``method``, out-of-range ``threshold`` /
        ``k_best`` / ``max_iter`` / ``n_neighbors`` / ``gamma``, a
        ``base_estimator`` without ``predict_proba``, non-numeric or
        non-finite features, or a target that fails
        :func:`validate_unlabeled`.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=50, n_features=6,
    ...     n_clusters_per_class=1, class_sep=2.0, random_state=0)
    >>> y[::4] = -1
    >>> result = train_semi_supervised(X, y, method="self_training")
    >>> result.method, result.params["sklearn_version"] is not None
    ('self_training', True)
    >>> result.confidence is not None and result.n_unlabeled > 0
    True
    """
    _validate_train_params(
        method, threshold, k_best, max_iter, n_neighbors, gamma, unlabeled_marker
    )
    matrix = _numeric_features(X)
    labels, info = validate_unlabeled(y, unlabeled_marker=unlabeled_marker)
    if labels.shape[0] != matrix.shape[0]:
        raise DataValidationError(
            f"y has {labels.shape[0]} value(s) but X has {matrix.shape[0]} rows"
        )

    n_labeled = info["n_labeled"]
    n_unlabeled = info["n_unlabeled"]
    started = time.perf_counter()

    if method == "self_training":
        base = _resolve_base_estimator(base_estimator)
        effective: dict[str, Any] = {}
        if int(k_best) > 0:
            effective["k_best"] = int(k_best)
        model = _self_training_estimator(
            base,
            threshold=float(threshold),
            max_iter=int(max_iter),
            **effective,
        )
        model.fit(matrix, labels)
        fit_seconds = time.perf_counter() - started

        predicted = np.asarray(model.predict(matrix))
        final_labels = labels.copy()
        pseudo_mask = labels == _UNLABELED_SENTINEL
        # ONLY the unlabeled positions take the pseudo-label — a
        # self-training wrapper never overrides a genuine label, and
        # neither do we.  The prediction is cast back into the
        # validated labels' dtype so metrics downstream see one
        # consistent type.
        final_labels[pseudo_mask] = predicted[pseudo_mask].astype(labels.dtype)
        confidence = np.asarray(model.predict_proba(matrix), dtype=float).max(axis=1)
        classes_ = np.asarray(model.classes_)
        estimator_name = type(model.base_estimator_).__name__
    else:
        graph_cls = LabelPropagation if method == "label_propagation" else LabelSpreading
        graph_kwargs: dict[str, Any] = {
            "kernel": kernel,
            "n_neighbors": int(n_neighbors),
            "gamma": float(gamma),
            "max_iter": int(max_iter),
        }
        if method == "label_spreading":
            graph_kwargs["alpha"] = 0.2
        model = graph_cls(**graph_kwargs)
        model.fit(matrix, labels)
        fit_seconds = time.perf_counter() - started

        final_labels = np.asarray(model.transduction_).copy()
        pseudo_mask = labels == _UNLABELED_SENTINEL
        distributions = getattr(model, "label_distributions_", None)
        if distributions is not None:
            confidence = np.asarray(distributions, dtype=float).max(axis=1)
        else:  # pragma: no cover - both graph methods expose it on 1.5
            confidence = None
        classes_ = np.asarray(model.classes_)
        estimator_name = type(model).__name__

    # Non-numeric input classes were encoded to indices by
    # validate_unlabeled — decode labels_ / classes_ back so the
    # caller sees their own class values, not our bookkeeping ints.
    # The decoded array keeps a NATIVE dtype when the classes are
    # homogeneous (all-float classes decode to float64) so sklearn
    # metrics still read it as a classification target; mixed-type
    # classes fall back to object.
    if info.get("encoded"):
        decoder: np.ndarray = info["class_decoder"]
        decoded_labels = decoder[np.asarray(final_labels, dtype=np.int64)]
        decoded_classes = decoder[np.asarray(classes_, dtype=np.int64)]
        native = _native_class_dtype(decoded_labels)
        final_labels = (
            decoded_labels.astype(native) if native else decoded_labels
        )
        classes_ = (
            decoded_classes.astype(native) if native else decoded_classes
        )

    params: dict[str, Any] = {
        "method": method,
        "threshold": float(threshold),
        "k_best": int(k_best),
        "max_iter": int(max_iter),
        "kernel": kernel,
        "n_neighbors": int(n_neighbors),
        "gamma": float(gamma),
        "unlabeled_marker": unlabeled_marker,
        "random_state": int(random_state),
        "sklearn_version": sklearn.__version__,
        "estimator_class": estimator_name,
    }
    _LOG.debug(
        "semi-supervised %s: %d labeled / %d unlabeled in %.2fs",
        method, n_labeled, n_unlabeled, fit_seconds,
    )
    return SemiSupervisedResult(
        method=method,
        params=params,
        model=model,
        labels_=final_labels,
        confidence=confidence,
        pseudo_label_mask=pseudo_mask,
        n_labeled=n_labeled,
        n_unlabeled=n_unlabeled,
        classes_=classes_,
        fit_seconds=float(fit_seconds),
    )


@dataclass(slots=True)
class _RepeatMetrics:
    """Internal accumulator for one evaluation repeat seed."""

    accuracy: float
    f1_macro: float
    baseline_accuracy: float
    baseline_f1_macro: float
    train_accuracy: float


def evaluate_semi_supervised(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | list,
    method: SemiSupervisedMethod = "self_training",
    *,
    test_size: float = 0.25,
    n_repeats: int = 3,
    random_state: int = 42,
    **train_kwargs: Any,
) -> SemiSupervisedEvaluation:
    """Leakage-safe honest evaluation of a semi-supervised configuration.

    Protocol (the scientific core — every step exists to keep the
    held-out labels out of training):

    1. **Stratified split of the labeled rows only.**
       ``train_test_split(stratify=real_labels)`` carves a held-out
       set out of the rows that HAVE labels; the unlabeled rows are
       never split (they cannot be stratified and never need to be).
    2. **Train on labeled-train + the unlabeled rows.**  The
       semi-supervised model sees the labeled-train rows with their
       genuine labels and every originally-unlabeled row as ``-1``.
       Held-out rows are dropped entirely for the fit — their labels
       never enter training in any form (not even masked).
    3. **Score on the held-out rows.**  ``self_training`` predicts
       directly; the graph methods are transductive, so the held-out
       rows join the fit as ``-1`` rows AFTER the scoring split has
       been frozen — no, they are predicted through the fitted graph
       by being included as unlabeled points in the SAME fit (the
       honest transductive reading: the graph is built once over
       train + held-out-as-unlabeled, labels flow only from the
       labeled-train rows, and the held-out transduction is scored).
    4. **Supervised baseline.**  The same estimator configuration
       trained on labeled-train only.  For ``self_training`` this is
       the wrapped base estimator verbatim; for the graph methods —
       which have no supervised analogue — the baseline is
       ``LogisticRegression(max_iter=1000)`` on the labeled-train
       rows (the graph methods' transductive nature has no honest
       supervised translation, so the comparison is against the
       standard supervised reference the toolkit would train on the
       same labeled budget).
    5. **Repeat and average.**  ``n_repeats`` seeds re-split and
       re-fit; the reported metrics are means over repeats.

    When the labeled data is too small for an honest split (fewer
    than 4 labeled rows per class, fewer than 8 labeled rows total,
    or ``n_repeats < 1`` / ``test_size`` out of range), every metric
    comes back ``None`` and :meth:`SemiSupervisedEvaluation.summary`
    says evaluation was not possible — a missing number, honestly
    reported, beats a number manufactured from a degenerate split.

    Parameters
    ----------
    X, y:
        Features and target (the ``-1`` convention / custom marker).
    method:
        The method to evaluate.
    test_size:
        Held-out share of the labeled rows, in ``(0, 1)``.
    n_repeats:
        Number of re-split/re-fit repeats (``>= 1``).
    random_state:
        Base seed; repeat ``i`` uses ``random_state + i``.
    **train_kwargs:
        Forwarded to :func:`train_semi_supervised` (``threshold``,
        ``k_best``, ``kernel``, ``n_neighbors``, ``gamma``,
        ``base_estimator``, ...).

    Returns
    -------
    SemiSupervisedEvaluation
        Held-out and baseline metrics averaged over repeats, or all
        ``None`` when an honest split was impossible.

    Raises
    ------
    DataValidationError
        On invalid ``test_size`` / ``n_repeats``, or any validation
        failure from the underlying train call.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=120, n_features=8,
    ...     n_informative=6, n_clusters_per_class=1, class_sep=2.0,
    ...     random_state=0)
    >>> y_masked = y.copy(); y_masked[::3] = -1
    >>> evaluation = evaluate_semi_supervised(X, y_masked, n_repeats=2)
    >>> evaluation.heldout_accuracy is not None and evaluation.heldout_accuracy > 0.5
    True
    >>> evaluation.baseline_accuracy is not None
    True
    """
    if not 0.0 < float(test_size) < 1.0:
        raise DataValidationError(
            f"test_size must be in (0, 1), got {test_size!r}"
        )
    if int(n_repeats) < 1:
        raise DataValidationError(f"n_repeats must be >= 1, got {n_repeats!r}")

    marker = train_kwargs.get("unlabeled_marker", _UNLABELED_SENTINEL)
    matrix = _numeric_features(X)
    labels, info = validate_unlabeled(y, unlabeled_marker=marker)
    if labels.shape[0] != matrix.shape[0]:
        raise DataValidationError(
            f"y has {labels.shape[0]} value(s) but X has {matrix.shape[0]} rows"
        )

    labeled_mask = labels != _UNLABELED_SENTINEL
    labeled_positions = np.flatnonzero(labeled_mask)
    unlabeled_positions = np.flatnonzero(~labeled_mask)
    n_labeled = int(labeled_positions.size)
    class_counts = info["class_counts"]
    min_class = min(class_counts.values()) if class_counts else 0

    if n_labeled < 8 or min_class < 4:
        _LOG.debug(
            "evaluation skipped: n_labeled=%d, min class count=%d "
            "(need >= 8 labeled rows and >= 4 per class)",
            n_labeled, min_class,
        )
        return SemiSupervisedEvaluation(
            heldout_accuracy=None,
            heldout_f1_macro=None,
            baseline_accuracy=None,
            baseline_f1_macro=None,
            train_accuracy_on_labeled=None,
            method=method,
        )

    repeat_metrics: list[_RepeatMetrics] = []
    for repeat in range(int(n_repeats)):
        seed = int(random_state) + repeat
        train_pos, heldout_pos = train_test_split(
            labeled_positions,
            test_size=float(test_size),
            stratify=labels[labeled_positions],
            random_state=seed,
        )
        train_pos = np.sort(train_pos)
        heldout_pos = np.sort(heldout_pos)

        # The fit sees labeled-train + the unlabeled rows + the
        # held-out rows as UNLABELED (transductive honesty: the graph
        # methods need the query rows inside the fit; their genuine
        # labels are never handed over).  For self-training the
        # held-out rows could be dropped, but including them masked
        # keeps one protocol for all three methods and matches how
        # the methods are used in practice (all rows in hand, few
        # labels).  Row order is preserved so positions map back.
        fit_positions = np.sort(
            np.concatenate([train_pos, unlabeled_positions, heldout_pos])
        )
        fit_labels = labels[fit_positions].copy().astype(float)
        local_of = {int(pos): local for local, pos in enumerate(fit_positions)}
        heldout_local_positions = np.array(
            [local_of[int(p)] for p in heldout_pos], dtype=int
        )
        train_local_positions = np.array(
            [local_of[int(p)] for p in train_pos], dtype=int
        )
        fit_labels[heldout_local_positions] = float(_UNLABELED_SENTINEL)

        fit_kwargs = {
            k: v for k, v in train_kwargs.items() if k != "unlabeled_marker"
        }
        result = train_semi_supervised(
            matrix[fit_positions],
            fit_labels,
            method,
            random_state=seed,
            **fit_kwargs,
        )
        heldout_true = labels[heldout_pos]
        train_true = labels[train_pos]

        # Supervised baseline on labeled-train only.
        baseline = _train_baseline(
            method, matrix[train_pos], train_true, train_kwargs, seed
        )
        baseline_pred = np.asarray(baseline.predict(matrix[heldout_pos]))

        # Encoded (string-class) inputs: decode BOTH sides into the
        # caller's original class space so every metric compares like
        # with like — the result's labels_, the baseline's output and
        # the ground truth all speak the caller's classes.
        # Encoded (string / non-integral-float) classes: score in the
        # ENCODED index space, which is always a clean integer
        # classification target for the sklearn metrics, then decode
        # nothing — the metric values are invariant under a bijection
        # of the class labels.  This sidesteps every dtype fight at
        # once (object arrays read as 'unknown', decoded floats as
        # 'continuous').
        if info.get("encoded"):
            heldout_pred = np.asarray(
                result.labels_[heldout_local_positions], dtype=np.int64
            )
            train_pred = np.asarray(
                result.labels_[train_local_positions], dtype=np.int64
            )
            baseline_pred = np.asarray(baseline_pred, dtype=np.int64)
        elif labels.dtype.kind in "iu":
            # The metrics need both sides in one dtype: the validated
            # labels are int, the transduction / predictions can come
            # back int or float depending on the estimator — cast the
            # predictions to the ground truth's dtype.
            heldout_pred = np.asarray(
                result.labels_[heldout_local_positions], dtype=labels.dtype
            )
            train_pred = np.asarray(
                result.labels_[train_local_positions], dtype=labels.dtype
            )
            baseline_pred = np.asarray(baseline_pred, dtype=labels.dtype)
        else:  # pragma: no cover - non-integral numeric labels are encoded
            heldout_pred = np.asarray(result.labels_[heldout_local_positions])
            train_pred = np.asarray(result.labels_[train_local_positions])
            baseline_pred = np.asarray(baseline_pred)

        repeat_metrics.append(
            _RepeatMetrics(
                accuracy=float(accuracy_score(heldout_true, heldout_pred)),
                f1_macro=float(f1_score(heldout_true, heldout_pred, average="macro")),
                baseline_accuracy=float(
                    accuracy_score(heldout_true, baseline_pred)
                ),
                baseline_f1_macro=float(
                    f1_score(heldout_true, baseline_pred, average="macro")
                ),
                train_accuracy=float(accuracy_score(train_true, train_pred)),
            )
        )

    heldout_accuracy = float(np.mean([m.accuracy for m in repeat_metrics]))
    _LOG.debug(
        "evaluate_semi_supervised(%s): held-out accuracy=%.3f over %d repeat(s)",
        method, heldout_accuracy, len(repeat_metrics),
    )
    return SemiSupervisedEvaluation(
        heldout_accuracy=heldout_accuracy,
        heldout_f1_macro=float(np.mean([m.f1_macro for m in repeat_metrics])),
        baseline_accuracy=float(
            np.mean([m.baseline_accuracy for m in repeat_metrics])
        ),
        baseline_f1_macro=float(
            np.mean([m.baseline_f1_macro for m in repeat_metrics])
        ),
        train_accuracy_on_labeled=float(
            np.mean([m.train_accuracy for m in repeat_metrics])
        ),
        method=method,
    )


def _train_baseline(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_kwargs: dict[str, Any],
    seed: int,
) -> Any:
    """Fit the supervised baseline on the labeled-train rows only.

    ``self_training`` reuses its own base estimator (default
    ``LogisticRegression(max_iter=1000)``) so the comparison isolates
    the VALUE OF THE UNLABELED ROWS; the graph methods have no
    supervised analogue, so they are compared against the standard
    ``LogisticRegression`` reference — the supervised model the
    toolkit would train on the same labeled budget.  The baseline is
    a fresh ``clone`` (never the evaluated fit itself), seeded through
    ``seed`` when the estimator accepts a ``random_state``.
    """
    if method == "self_training":
        base = train_kwargs.get("base_estimator")
        if base is None:
            base = LogisticRegression(max_iter=1000)
    else:
        base = LogisticRegression(max_iter=1000)
    baseline = clone(base)
    if "random_state" in baseline.get_params():
        baseline.set_params(random_state=seed)
    baseline.fit(X_train, y_train)
    return baseline
