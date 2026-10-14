"""Data preprocessing for QSAR tables.

Upgrades the preprocessing fragments that were copy-pasted into every
legacy model script (median imputation + standard scaling, the
``Non-numerical data remover.py`` script and the split helper) into one
composable, reusable module.  The returned sklearn objects are fitted
on training data only by convention of :func:`build_preprocessor` --
the caller fits inside a Pipeline to avoid train/test leakage.
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
from pharmaml.utils.validation import require_columns

_LOG = get_logger(__name__)

__all__ = [
    "clean_numeric_frame",
    "build_preprocessor",
    "split_features_target",
    "make_train_test_split",
    "remove_outliers_iqr",
]


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

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline to be composed with an estimator, e.g.
        ``Pipeline([("prep", build_preprocessor()), ("clf", model)])``.
    """
    if impute_strategy not in {"median", "mean", "most_frequent"}:
        raise DataValidationError(
            f"impute_strategy must be median/mean/most_frequent, got {impute_strategy!r}"
        )
    steps = [("impute", SimpleImputer(strategy=impute_strategy))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


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

    Returns
    -------
    tuple
        ``(X, y)`` where ``X`` keeps only numeric feature columns.
    """
    require_columns(frame, [target_col])
    drop = {target_col, *(drop_cols or [])}
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

    Stratification is attempted when every class has at least
    ``min_class_count`` members; otherwise a plain split is used and a
    warning is logged (the legacy scripts crashed outright on
    single-member classes).

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.
    """
    if not 0.0 < test_size < 1.0:
        raise DataValidationError(f"test_size must be in (0, 1), got {test_size}")
    use_stratify = stratify
    if stratify:
        counts = pd.Series(np.asarray(y)).value_counts()
        if counts.min() < min_class_count:
            _LOG.warning(
                "Class with %d member(s) < %d required; falling back to a "
                "non-stratified split.",
                int(counts.min()),
                min_class_count,
            )
            use_stratify = False
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if use_stratify else None,
    )


def remove_outliers_iqr(
    frame: pd.DataFrame,
    factor: float = 1.5,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """Remove rows containing IQR outliers in numeric columns.

    A row is kept only if, for *every* considered column, its value lies
    within ``[Q1 − factor·IQR, Q3 + factor·IQR]``.  ``subset=None``
    considers all numeric columns.
    """
    if factor <= 0:
        raise DataValidationError(f"factor must be > 0, got {factor}")
    numeric = frame.select_dtypes(include=[np.number])
    if subset is None:
        columns = list(numeric.columns)
    else:
        columns = [c for c in subset if c in numeric.columns]
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
