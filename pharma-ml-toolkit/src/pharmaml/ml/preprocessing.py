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

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.utils.validation import looks_continuous, require_columns

_LOG = get_logger(__name__)

__all__ = [
    "build_preprocessor",
    "clean_numeric_frame",
    "has_categorical_columns",
    "make_train_test_split",
    "remove_outliers_iqr",
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
