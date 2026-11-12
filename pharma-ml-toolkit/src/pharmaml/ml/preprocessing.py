"""Data preprocessing for QSAR tables.

Upgrades the preprocessing fragments that were copy-pasted into every
legacy model script (median imputation + standard scaling, the
``Non-numerical data remover.py`` script and the split helper) into one
composable, reusable module.  The returned sklearn objects are fitted
on training data only by convention of :func:`build_preprocessor` --
the caller fits inside a Pipeline to avoid train/test leakage.

Examples
--------
>>> import numpy as np
>>> import pandas as pd
>>> from pharmaml.ml.preprocessing import has_categorical_columns
>>> has_categorical_columns(pd.DataFrame({"logp": [1.2, 2.4], "assay": ["A", "B"]}))
True
>>> has_categorical_columns(pd.DataFrame({"logp": [1.2, 2.4]}))
False
>>> has_categorical_columns(np.array([[1.0, 2.0]]))    # arrays are never categorical
False

>>> from pharmaml.ml.preprocessing import clean_numeric_frame, split_features_target
>>> raw = pd.DataFrame({"SMILES": ["CCO", "CCC"], "logp": [1.2, 2.4],
...                     "assay": ["A", "B"], "pIC50": [5.0, 4.0]})
>>> clean_numeric_frame(raw, target_col="pIC50", keep_cols=["SMILES"]).columns.tolist()
['SMILES', 'logp', 'pIC50']
>>> X, y = split_features_target(raw, "pIC50", drop_cols=["SMILES"])
>>> (X.columns.tolist(), y.tolist())
(['logp', 'assay'], [5.0, 4.0])

>>> from pharmaml.ml.preprocessing import build_preprocessor
>>> [step[0] for step in build_preprocessor().steps]        # impute -> scale
['impute', 'scale']
>>> [step[0] for step in build_preprocessor(scale=False).steps]
['impute']
>>> X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]})
>>> [[round(v, 6) for v in row]
...  for row in build_preprocessor().fit_transform(X).tolist()]   # median + z-score
[[-1.224745, -1.224745], [0.0, 0.0], [1.224745, 1.224745]]

>>> from pharmaml.ml.preprocessing import remove_outliers_iqr
>>> vals = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
>>> remove_outliers_iqr(vals)["x"].tolist()                # 100 is beyond Q3 + 1.5*IQR
[1.0, 2.0, 3.0, 4.0]
>>> remove_outliers_iqr(vals, factor=0.1)["x"].tolist()    # tighter fence
[2.0, 3.0, 4.0]
>>> remove_outliers_iqr(vals, factor=0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: factor must be > 0, got 0
>>> remove_outliers_iqr(vals, factor=float("nan"))
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: factor must be > 0, got nan

>>> from pharmaml.ml.preprocessing import make_train_test_split
>>> Xs = pd.DataFrame({"f": np.arange(10.0)})
>>> ys = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
>>> X_train, X_test, y_train, y_test = make_train_test_split(Xs, ys, random_state=42)
>>> (X_train.shape, X_test.shape, sorted(y_test.tolist()))  # stratified 8/2
((8, 1), (2, 1), [0, 1])
>>> make_train_test_split(Xs, ys, test_size=1.5)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: test_size must be in (0, 1), got 1.5
>>> make_train_test_split(Xs.iloc[:0], pd.Series([], dtype=float))
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: y is empty -- make_train_test_split needs at least one labeled sample
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pharmaml.exceptions import (
    DataValidationError,
    JobCancelled,
    PersistenceError,
)
from pharmaml.logger import get_logger
from pharmaml.utils.validation import looks_continuous, require_columns

_LOG = get_logger(__name__)

__all__ = [
    "apply_preprocessor_chunked",
    "build_preprocessor",
    "clean_numeric_frame",
    "has_categorical_columns",
    "make_train_test_split",
    "remove_outliers_iqr",
    "select_kbest_union",
    "split_features_target",
]


def has_categorical_columns(X: pd.DataFrame | np.ndarray) -> bool:
    """Return True when ``X`` is a DataFrame with non-numeric columns.

    Used by :func:`~pharmaml.ml.training.train_model` and
    :func:`~pharmaml.ml.training.compare_models` to switch the default
    preprocessor to one-hot mode automatically, so a mixed feature
    table (numeric descriptors + e.g. assay/cell-line strings) trains
    without the caller writing any ColumnTransformer code.
    """
    if not isinstance(X, pd.DataFrame):
        return False
    return any(not pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)


def clean_numeric_frame(
    frame: pd.DataFrame,
    target_col: str | None = None,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Drop non-numeric columns (except the target/keep list).

    The legacy script removed every non-numeric column unconditionally;
    here the target and explicitly whitelisted metadata columns (e.g.
    ``SMILES``) survive, and a warning names what was dropped so
    accidental loss of a descriptor column is visible immediately.
    """
    protect = {target_col, *(keep_cols or [])} - {None}
    numeric_cols = [
        col
        for col in frame.columns
        if pd.api.types.is_numeric_dtype(frame[col]) or col in protect
    ]
    dropped = [col for col in frame.columns if col not in numeric_cols]
    if dropped:
        _LOG.warning("Dropping %d non-numeric columns: %s", len(dropped), dropped[:10])
    cleaned = frame.loc[:, numeric_cols].copy()
    if target_col is not None:
        require_columns(cleaned, [target_col])
    return cleaned


def build_preprocessor(
    impute_strategy: str = "median",
    scale: bool = True,
    encode_categorical: bool = False,
) -> Pipeline:
    """Build the (impute -> scale) preprocessing pipeline.

    Parameters
    ----------
    impute_strategy:
        ``'median'`` (default, robust for descriptors), ``'mean'`` or
        ``'most_frequent'``.
    scale:
        Apply :class:`~sklearn.preprocessing.StandardScaler` after
        imputation.  Keep True for distance/linear models; tree/boosting
        models are scale-invariant but rarely hurt by it.
    encode_categorical:
        Handle **non-numeric feature columns** with a
        :class:`~sklearn.compose.ColumnTransformer`: numeric columns
        get the usual impute(+scale) branch, string/category columns
        get ``most_frequent`` imputation + one-hot encoding with
        ``handle_unknown='ignore'`` (unseen categories at scoring time
        become all-zero instead of crashing).  ``train_model``/
        ``compare_models`` enable this automatically whenever the
        feature table contains non-numeric columns.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline to be composed with an estimator, e.g.
        ``Pipeline([("prep", build_preprocessor()), ("clf", model)])``.
        With ``encode_categorical=True`` the returned pipeline wraps a
        ColumnTransformer (``get_feature_names_out`` yields the expanded
        one-hot names, so feature-importance plots stay readable).
    """
    if impute_strategy not in {"median", "mean", "most_frequent"}:
        raise DataValidationError(
            f"impute_strategy must be median/mean/most_frequent, got {impute_strategy!r}"
        )

    if not encode_categorical:
        steps = [("impute", SimpleImputer(strategy=impute_strategy))]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.preprocessing import OneHotEncoder

    numeric_steps = [("impute", SimpleImputer(strategy=impute_strategy))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return Pipeline(
        steps=[
            (
                "columns",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(numeric_steps),
                            make_column_selector(dtype_include=np.number),
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "onehot",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            sparse_output=False,
                                        ),
                                    ),
                                ]
                            ),
                            make_column_selector(dtype_exclude=np.number),
                        ),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            )
        ]
    )


def split_features_target(
    frame: pd.DataFrame,
    target_col: str,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target column.

    Parameters
    ----------
    frame:
        Input table.
    target_col:
        Column holding the endpoint (class label or pIC50).
    drop_cols:
        Extra columns to exclude from features (IDs, SMILES, splits...).
        Entries missing from the frame are logged and ignored -- a typo
        would otherwise silently leave e.g. a SMILES column among the
        features.

    Returns
    -------
    tuple
        ``(X, y)`` where ``X`` carries every remaining column, numeric
        and non-numeric alike (string columns are one-hot encoded
        downstream by ``train_model`` / ``compare_models``; pass
        metadata like SMILES via ``drop_cols``).
    """
    require_columns(frame, [target_col])
    requested = drop_cols or []
    missing = [c for c in requested if c not in frame.columns]
    if missing:
        _LOG.warning("drop_cols not present in frame, kept as features: %s", missing)
    drop = {target_col, *requested}
    X = frame.drop(columns=[c for c in drop if c in frame.columns])
    y = frame[target_col]
    return X, y


def make_train_test_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    min_class_count: int = 2,
):
    """Stratified train/test split with automatic fallback.

    Stratification is attempted for class-label targets when every
    class has at least ``min_class_count`` members.  Regression-style
    targets (numeric with mostly unique values, see
    :func:`pharmaml.utils.validation.looks_continuous`) skip
    stratification quietly — it is the expected path for them, not a
    fallback.  For categorical targets with a rare class a plain split
    is used and a warning is logged (the legacy scripts crashed
    outright on single-member classes).

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.

    Raises
    ------
    DataValidationError
        When ``test_size`` is outside (0, 1), ``y`` is empty, ``y``
        contains NaN values (the stratified splitter used to surface
        a raw sklearn ``ValueError: Input y contains NaN``), or the
        split configuration cannot produce a valid split for the data
        shape (e.g. ``test_size`` so close to 1 that the training set
        would be empty).
    """
    if not 0.0 < test_size < 1.0:
        raise DataValidationError(f"test_size must be in (0, 1), got {test_size}")
    # 100x-41-F: an empty y used to fall through every guard and die
    # inside sklearn with a raw ``ValueError: With n_samples=0, ...``
    # (the stratify bookkeeping cannot even count classes) -- name it
    # up front instead.
    if len(np.asarray(y)) == 0:
        raise DataValidationError(
            "y is empty -- make_train_test_split needs at least one labeled sample"
        )
    n_missing = int(pd.isna(np.asarray(y)).sum())
    if n_missing:
        raise DataValidationError(
            f"y contains {n_missing} NaN value(s) -- drop or impute the target "
            "before splitting (load_dataset drops missing-target rows for you)"
        )
    use_stratify = stratify
    if stratify:
        if looks_continuous(y):
            # Regression-style target: stratification is not a thing --
            # skip it quietly rather than warning about 1-member "classes".
            _LOG.debug("continuous target -- splitting without stratification")
            use_stratify = False
        else:
            counts = pd.Series(np.asarray(y)).value_counts()
            if counts.min() < min_class_count:
                _LOG.warning(
                    "Class with %d member(s) < %d required; falling back to a "
                    "non-stratified split.",
                    int(counts.min()),
                    min_class_count,
                )
                use_stratify = False
    try:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if use_stratify else None,
        )
    except ValueError as exc:
        # 100x-41-F: in-(0,1)-boundary corners still belong to sklearn
        # -- test_size=0.999 with n=10 ("the resulting train set will
        # be empty"), 1e-9 stratified ("The test_size = 1 should be
        # greater or equal to the number of classes = 2") -- wrap them
        # so the module's typed-error contract holds for the whole
        # split, not just the pre-checked inputs.
        raise DataValidationError(f"invalid split configuration: {exc}") from exc


def remove_outliers_iqr(
    frame: pd.DataFrame,
    factor: float = 1.5,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """Remove rows containing IQR outliers in numeric columns.

    A row is kept only if, for *every* considered column, its value lies
    within ``[Q1 − factor·IQR, Q3 + factor·IQR]``.  ``subset=None``
    considers all numeric columns.  ``subset`` entries that are missing
    from the frame or non-numeric are ignored with a warning (a typo
    used to silently disable the filter for that column).

    Raises
    ------
    DataValidationError
        When ``factor`` is not strictly positive -- including NaN,
        which used to slip past ``factor <= 0`` and silently disable
        the filter entirely (NaN fences compare False against every
        value, so all rows were kept without any warning).
    """
    # 100x-41-F: ``factor <= 0`` let NaN through (every NaN comparison
    # is False) -- NaN fences then disabled the filter silently.
    # ``not factor > 0`` rejects NaN alongside non-positive factors;
    # +inf stays allowed (an infinitely wide fence is a legitimate
    # keep-everything choice).
    if not factor > 0:
        raise DataValidationError(f"factor must be > 0, got {factor}")
    numeric = frame.select_dtypes(include=[np.number])
    if subset is None:
        columns = list(numeric.columns)
    else:
        columns = [c for c in subset if c in numeric.columns]
        ignored = [c for c in subset if c not in numeric.columns]
        if ignored:
            _LOG.warning(
                "Ignoring subset column(s) not found among numeric columns: %s",
                ignored,
            )
    if not columns:
        return frame.copy()

    Q1 = numeric[columns].quantile(0.25)
    Q3 = numeric[columns].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~(
        (numeric[columns] < (Q1 - factor * IQR)) | (numeric[columns] > (Q3 + factor * IQR))
    ).any(axis=1)
    kept = frame.loc[mask].reset_index(drop=True)
    _LOG.info("IQR outlier filter: kept %d of %d rows", len(kept), len(frame))
    return kept


#: Score functions :func:`select_kbest_union` accepts, mapped to the
#: sklearn implementations.  ``mutual_info_regression`` is pinned with
#: ``random_state=42`` (via :func:`functools.partial`) so its k-nearest
#: neighbour density estimates -- and therefore the selection -- are
#: reproducible run to run.
_KBEST_METHODS: dict[str, Callable[..., Any]] = {
    "f_regression": f_regression,
    "mutual_info_regression": partial(mutual_info_regression, random_state=42),
}


def select_kbest_union(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    k: int = 256,
    methods: tuple[str, ...] = ("f_regression", "mutual_info_regression"),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Select the UNION of per-method top-k columns (upgrades look.ipynb).

    The notebook's step-6 selector pair -- one
    :class:`~sklearn.feature_selection.SelectKBest` per scoring function
    (default: ``f_regression`` and ``mutual_info_regression``), each
    keeping its own top-``k`` columns -- with the two kept-sets UNIONed
    (``np.unique`` of the concatenated ``get_support(indices=True)``
    positions, so the union is deduplicated and returned in the
    ORIGINAL ``X_train`` column order).  A union keeps the columns only
    MI likes (nonlinear signal an F-test misses) alongside the F-test
    favourites -- the notebook's own rationale.

    BOTH selectors are fitted on ``(X_train, y_train)`` only and are
    then merely *applied* to ``X_test`` as a pure column subset -- the
    notebook's leakage fix ("feature selection on training data only")
    kept as the contract: nothing here ever refits on the test frame.

    Parameters
    ----------
    X_train, X_test:
        Feature tables sharing the same columns (order-insensitive;
        both outputs follow ``X_train``'s original column order).
    y_train:
        Regression target aligned with ``X_train`` rows.
    k:
        Columns each method keeps (so the union has between ``k`` and
        ``k * len(methods)`` columns).
    methods:
        Scoring functions to union over; ``mutual_info_regression`` is
        seeded at 42, making the whole selection deterministic.

    Returns
    -------
    tuple
        ``(train_selected, test_selected, columns)`` -- the two column-
        subset frames plus the selected column labels in original
        ``X_train`` order.

    Raises
    ------
    TypeError
        When ``X_train``/``X_test`` are not pandas DataFrames (the
        selection is by column NAME).
    ValueError
        When ``k`` is not a positive int (bools rejected), ``k``
        exceeds the feature count, ``methods`` is empty or names an
        unknown scoring function, or ``X_train``/``X_test`` do not
        carry the same columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_regression
    >>> from pharmaml.ml.preprocessing import select_kbest_union
    >>> X, y = make_regression(n_samples=50, n_features=12, n_informative=4, random_state=42)
    >>> names = [f"descr_{i:02d}" for i in range(12)]
    >>> X_train = pd.DataFrame(X[:40], columns=names)
    >>> X_test = pd.DataFrame(X[40:], columns=names)
    >>> train_sel, test_sel, columns = select_kbest_union(X_train, y[:40], X_test, k=3)
    >>> 3 <= len(columns) <= 6                     # union of two top-3 lists
    True
    >>> list(train_sel.columns) == columns == list(test_sel.columns)
    True
    >>> _, _, again = select_kbest_union(X_train, y[:40], X_test, k=3)
    >>> again == columns                           # mutual_info pinned at seed 42
    True

    >>> select_kbest_union(X_train, y[:40], X_test, k=99)
    Traceback (most recent call last):
    ...
    ValueError: k=99 exceeds the 12 feature columns
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive int, got {k!r}")
    if not methods:
        raise ValueError(
            "methods tuple is empty -- pass at least one of: "
            + ", ".join(sorted(_KBEST_METHODS))
        )
    unknown = [method for method in methods if method not in _KBEST_METHODS]
    if unknown:
        raise ValueError(
            f"unknown feature-selection method(s) {unknown}; valid methods: "
            + ", ".join(sorted(_KBEST_METHODS))
        )
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "select_kbest_union selects columns by NAME and needs pandas "
            f"DataFrames, got {type(X_train).__name__} and "
            f"{type(X_test).__name__}"
        )

    train_columns = list(X_train.columns)
    test_columns = list(X_test.columns)
    if set(train_columns) != set(test_columns) or len(train_columns) != len(test_columns):
        missing = sorted(set(train_columns) - set(test_columns), key=str)[:5]
        extra = sorted(set(test_columns) - set(train_columns), key=str)[:5]
        raise ValueError(
            "X_train and X_test column mismatch -- missing from X_test: "
            f"{missing}; extra in X_test: {extra}"
        )

    n_features = len(train_columns)
    if k > n_features:
        raise ValueError(f"k={k} exceeds the {n_features} feature columns")

    target = np.asarray(y_train)
    positions: list[np.ndarray] = []
    for method in methods:
        selector = SelectKBest(score_func=_KBEST_METHODS[method], k=k)
        selector.fit(X_train, target)
        positions.append(selector.get_support(indices=True))
    union = np.unique(np.concatenate(positions))
    columns = [train_columns[int(position)] for position in union]

    _LOG.info(
        "SelectKBest union: %d column(s) kept from %d method(s) x top-%d "
        "(of %d feature columns)",
        len(columns), len(methods), k, n_features,
    )
    # Pure column subsets of the caller's frames -- never refit on the
    # test side (the notebook's leakage discipline).
    return X_train.loc[:, columns].copy(), X_test.loc[:, columns].copy(), columns


# ---------------------------------------------------------------------------
# (part5-continuation round 11, §13) the resumable chunked preprocessing
# ---------------------------------------------------------------------------
def apply_preprocessor_chunked(
    frame: pd.DataFrame,
    *,
    impute_strategy: str = "median",
    scale: bool = True,
    encode_categorical: bool = False,
    chunk_rows: int = 500,
    progress: Callable[..., Any] | None = None,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    checkpoint_every: int = 1,
    checkpoint_resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the standard preprocessor and apply it CHUNK-WISE, resumably.

    The last of the six §13 operation classes: for a table too large
    to transform in one holding (the "large preprocessing jobs" of the
    schema header), the standard :func:`build_preprocessor` pipeline
    is fitted once on the full table, then applied ``chunk_rows`` rows
    at a time.  The exactness contract rests on two verified
    properties: the fit is DETERMINISTIC given the table (so a resume
    refits and lands on the same state -- the fitted statistics are
    never serialized, they are recomputed), and ``transform`` is
    ROW-PURE (the transform of a slice equals the slice of the
    transform -- verified in the round's recon).  The checkpoint is
    therefore only the completed chunks' transformed rows: a resume
    skips them bit-exactly and continues with the tail.

    Parameters
    ----------
    frame:
        The table to preprocess.  Non-numeric columns require
        ``encode_categorical=True`` (otherwise they are, per
        :func:`build_preprocessor`'s ColumnTransformer contract, only
        passable when the flag is on -- the same validation the
        pipeline itself applies).
    impute_strategy, scale, encode_categorical:
        Passed verbatim to :func:`build_preprocessor` -- they are the
        config fingerprint's core (a different strategy refuses to
        resume: exact restore or nothing).
    chunk_rows:
        Rows per chunk (default 500; >= 1 enforced).  The chunk is the
        resume unit.
    progress:
        Optional ``report(done, total, note)`` callback per finished
        chunk; returning ``False`` stops at the next chunk boundary
        with :class:`~pharmaml.exceptions.JobCancelled` carrying the
        partial ``(frame, report)`` on ``partial_results``.
    checkpoint_dir:
        Optional directory for the v2-envelope checkpoints.  One
        checkpoint per ``checkpoint_every`` chunk boundaries carrying
        every completed row's transformed vector (JSON float
        round-trip is bit-exact in CPython), the column list, and the
        config/dataset fingerprint (strategy knobs + column list +
        dtypes + shape + a content checksum).  ``None`` keeps the
        historical one-shot behaviour available (call
        ``build_preprocessor().fit_transform`` directly).
    checkpoint_every:
        Chunk boundaries between checkpoint saves (default 1; >= 1
        enforced).
    checkpoint_resume:
        When ``True`` (default) and ``checkpoint_dir`` is given,
        resume from the newest valid fingerprint-matching checkpoint.
        ``False`` still writes checkpoints but always computes from
        scratch -- the fresh-run twin the proof suite compares
        against.

    Returns
    -------
    tuple
        ``(frame, report)`` -- the transformed float frame (same row
        order, ``get_feature_names_out`` columns) and a report dict
        (``chunks``, ``rows``, ``columns``, ``resumed_from_chunks``,
        ``fitted_steps``, ``checkpoint``).

    Raises
    ------
    DataValidationError
        Empty frame, a bad ``chunk_rows``/``checkpoint_every``, or a
        non-numeric table without ``encode_categorical`` (the
        pipeline's own contract).
    JobCancelled
        Cooperative stop: ``partial_results`` carries the partial
        ``(frame, report)`` -- the completed rows transformed, the
        pending rows absent.

    Example
    -------
    >>> import numpy as np, pandas as pd
    >>> from pharmaml.ml.preprocessing import apply_preprocessor_chunked
    >>> table = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]})
    >>> out, report = apply_preprocessor_chunked(table, chunk_rows=2)
    >>> report["rows"], report["chunks"]
    (3, 2)
    >>> bool(np.isclose(out["a"].iloc[1], 0.0))     # the imputed median z-scores to 0
    True
    """
    if frame.empty:
        raise DataValidationError("the frame is empty -- nothing to preprocess")
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        raise ValueError(f"chunk_rows must be a positive int, got {chunk_rows!r}")
    if (
        not isinstance(checkpoint_every, int)
        or isinstance(checkpoint_every, bool)
        or checkpoint_every < 1
    ):
        raise ValueError(
            f"checkpoint_every must be a positive int, got {checkpoint_every!r}"
        )

    from pharmaml.utils.checkpoint_schema import OperationSpec
    from pharmaml.utils.checkpoints import CheckpointManager

    # -- the pipeline: fitted ONCE, deterministically ------------------- #
    # (a resume refits rather than deserializing sklearn state -- the
    # fit is deterministic given the table, which the recon verified)
    pipeline = build_preprocessor(
        impute_strategy=impute_strategy,
        scale=scale,
        encode_categorical=encode_categorical,
    )
    fitted = pipeline.fit(frame)
    columns_out = [str(c) for c in fitted.get_feature_names_out()]
    total_rows = len(frame)
    total_chunks = (total_rows + chunk_rows - 1) // chunk_rows

    # -- the §13 checkpoint machinery ------------------------------------ #
    manager: Any = None
    spec: Any = None
    config_fp = ""
    if checkpoint_dir is not None:
        try:
            manager = CheckpointManager(
                checkpoint_dir, name="preprocessing"
            )
            spec = OperationSpec(
                operation_type="preprocessing",
                operation_id=f"table-{total_rows}x{len(frame.columns)}",
            )
        except PersistenceError as exc:
            _LOG.warning(
                "checkpoint home unusable (%s); preprocessing without "
                "checkpoints",
                exc,
            )
            manager = None
            spec = None

        # the fingerprint: the strategy knobs (the config), the shape +
        # column list + dtypes + a content checksum (the dataset) --
        # any edit to the table refuses the resume (exact restore or
        # nothing).  pandas' hash_pandas_object covers mixed dtypes
        # (including category/string columns) and the index.
        content = hashlib.sha256(
            pd.util.hash_pandas_object(
                frame, index=True
            ).to_numpy().tobytes()
        ).hexdigest()
        config_fp = hashlib.sha256(
            json.dumps(
                {
                    "impute_strategy": impute_strategy,
                    "scale": scale,
                    "encode_categorical": encode_categorical,
                    "chunk_rows": chunk_rows,
                    "columns": [str(c) for c in frame.columns],
                    "dtypes": [str(frame[c].dtype) for c in frame.columns],
                    "n_rows": total_rows,
                    "content_sha256": content,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    # -- the resume: completed chunks' rows are restored verbatim ------- #
    done_rows: list[list[float]] = []
    start_chunk = 0
    resumed_chunks = 0
    if manager is not None and checkpoint_resume:
        try:
            latest = manager.latest()
        except OSError:  # pragma: no cover - scan failure
            latest = None
            _LOG.warning("checkpoint scan failed; starting fresh")
        if latest is not None and latest.resumable:
            st = latest.state
            if st.get("config_fp") != config_fp:
                _LOG.warning(
                    "preprocessing checkpoint at step %d is for a different "
                    "table/config; starting fresh (exact restore or nothing)",
                    latest.step,
                )
            else:
                try:
                    done_rows = [
                        [float(v) for v in row] for row in st["rows"]
                    ]
                    if len(done_rows) > total_rows:
                        raise ValueError(
                            f"ledger holds {len(done_rows)} rows for a "
                            f"{total_rows}-row table"
                        )
                    if len(done_rows) == total_rows:
                        # the run was already COMPLETE -- every chunk
                        # restored, nothing left to transform
                        start_chunk = total_chunks
                    elif len(done_rows) % chunk_rows:
                        # a partial ledger must be chunk-aligned (the
                        # checkpoint fires on chunk boundaries only)
                        raise ValueError(
                            f"ledger holds {len(done_rows)} rows -- not "
                            "chunk-aligned"
                        )
                    else:
                        start_chunk = len(done_rows) // chunk_rows
                    resumed_chunks = start_chunk
                    if start_chunk and start_chunk < total_chunks:
                        _LOG.info(
                            "preprocessing resumed from checkpoint step %d "
                            "(%d chunk(s) / %d row(s) already transformed)",
                            latest.step,
                            start_chunk,
                            len(done_rows),
                        )
                    elif start_chunk == total_chunks:
                        _LOG.info(
                            "preprocessing checkpoint at step %d already "
                            "holds the complete table (%d row(s)) -- "
                            "nothing left to transform",
                            latest.step,
                            len(done_rows),
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    _LOG.warning(
                        "checkpoint payload unreadable (%s); starting fresh",
                        exc,
                    )
                    done_rows = []
                    start_chunk = 0
                    resumed_chunks = 0

    def _save(step: int, note: str) -> None:
        if manager is None or spec is None:
            return
        try:
            manager.save(
                step=step,
                total=total_chunks,
                state={"config_fp": config_fp, "rows": done_rows},
                note=note,
                operation=spec,
            )
        except (PersistenceError, OSError, TypeError, ValueError) as exc:
            _LOG.warning(
                "checkpoint write failed at %s (%s); continuing without it",
                note,
                exc,
            )

    # -- the chunk loop --------------------------------------------------- #
    boundaries_since_save = 0
    for chunk_index in range(start_chunk, total_chunks):
        lo = chunk_index * chunk_rows
        hi = min(lo + chunk_rows, total_rows)
        block = fitted.transform(frame.iloc[lo:hi])
        done_rows.extend(
            [[float(v) for v in row] for row in np.asarray(block).tolist()]
        )
        if (
            progress is not None
            and progress(
                chunk_index + 1 - start_chunk,
                total_chunks - start_chunk,
                "preprocessing chunks",
            )
            is False
        ):
            partial_frame = pd.DataFrame(done_rows, columns=columns_out)
            partial_report = {
                "chunks": len(done_rows) // chunk_rows,
                "rows": len(done_rows),
                "columns": columns_out,
                "resumed_from_chunks": resumed_chunks,
                "fitted_steps": [s for s, _ in fitted.steps],
                "checkpoint": manager is not None,
                "partial": True,
            }
            raise JobCancelled(
                f"preprocessing cancelled after chunk "
                f"{chunk_index + 1}/{total_chunks} -- stopped at a "
                "chunk boundary by the caller",
                partial_results=(partial_frame, partial_report),
            )
        boundaries_since_save += 1
        if (
            manager is not None
            and spec is not None
            and (
                chunk_index >= total_chunks - 1
                or boundaries_since_save >= checkpoint_every
            )
        ):
            boundaries_since_save = 0
            _save(
                chunk_index + 1,
                f"preprocessing {chunk_index + 1}/{total_chunks} chunks",
            )

    # -- the closing frame ------------------------------------------------ #
    out = pd.DataFrame(done_rows, columns=columns_out)
    report = {
        "chunks": total_chunks,
        "rows": total_rows,
        "columns": columns_out,
        "resumed_from_chunks": resumed_chunks,
        "fitted_steps": [s for s, _ in fitted.steps],
        "checkpoint": manager is not None,
    }
    return out, report
