"""Model persistence with provenance metadata.

A saved artifact is a joblib dump of a *bundle* dict::

    {
      "model": <fitted estimator>,
      "metadata": {
         "name": ..., "task": ..., "pharmaml_version": ...,
         "sklearn_version": ..., "numpy_version": ...,
         "trained_at": <UTC ISO timestamp>,
         "features": [...], "best_params": {...},
         "metrics": {...},
      },
    }

so a model loaded two years later still knows what it is, which
feature order it expects, and how well it scored.  The legacy code
saved bare ``.joblib``/``.pkl`` blobs with zero provenance.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import joblib
import numpy
import sklearn

from pharmaml import __version__
from pharmaml.exceptions import PersistenceError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = ["save_model", "load_model", "bundle_metadata"]

_BUNDLE_KEY_MODEL = "model"
_BUNDLE_KEY_META = "metadata"


def bundle_metadata(
    name: str,
    task: str,
    features: list[str] | None = None,
    best_params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance metadata block for a model bundle."""
    return {
        "name": name,
        "task": task,
        "pharmaml_version": __version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": numpy.__version__,
        "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "features": list(features) if features is not None else None,
        "best_params": best_params or {},
        "metrics": metrics or {},
    }


def save_model(
    model: Any,
    path: str | Path,
    name: str,
    task: str,
    features: list[str] | None = None,
    best_params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Persist a fitted model plus its metadata bundle.

    Parameters
    ----------
    model:
        Fitted estimator (or Pipeline).
    path:
        Destination ``.joblib`` file; parent directories are created.
    name, task:
        Registry name and task labels for provenance.
    features:
        Ordered feature names expected at inference time.
    best_params, metrics:
        Optional hyperparameters and held-out metrics.

    Returns
    -------
    Path
        Resolved path of the written artifact.
    """
    target = Path(path)
    if target.suffix.lower() not in {".joblib", ".pkl"}:
        target = target.with_suffix(".joblib")
    target.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        _BUNDLE_KEY_MODEL: model,
        _BUNDLE_KEY_META: bundle_metadata(name, task, features, best_params, metrics),
    }
    try:
        joblib.dump(bundle, target, compress=3)
    except Exception as exc:
        raise PersistenceError(f"Failed to save model to {target}: {exc}") from exc
    _LOG.info("Saved model '%s' (%s) to %s", name, task, target)
    return target


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a model bundle saved by :func:`save_model`.

    Returns
    -------
    tuple
        ``(model, metadata)``.

    Raises
    ------
    PersistenceError
        If the file is missing, not a PharmaML bundle, or was written
        by an incompatible joblib/sklearn generation.
    """
    source = Path(path)
    if not source.is_file():
        raise PersistenceError(f"Model file not found: {source}")

    try:
        bundle = joblib.load(source)
    except Exception as exc:
        raise PersistenceError(f"Cannot load '{source}' ({exc})") from exc

    if not isinstance(bundle, dict) or _BUNDLE_KEY_MODEL not in bundle:
        raise PersistenceError(
            f"'{source}' is not a PharmaML model bundle "
            "(legacy bare .joblib/.pkl files are not supported; retrain and "
            "save with pharmaml.ml.persistence.save_model)"
        )

    metadata = bundle.get(_BUNDLE_KEY_META, {})
    _LOG.info(
        "Loaded model '%s' trained at %s (pharmaml %s, sklearn %s)",
        metadata.get("name", "<unknown>"),
        metadata.get("trained_at", "<unknown>"),
        metadata.get("pharmaml_version", "?"),
        metadata.get("sklearn_version", "?"),
    )
    if metadata.get("sklearn_version") not in {None, sklearn.__version__}:
        _LOG.warning(
            "Model was trained with sklearn %s but runtime is %s -- "
            "re-validate before drawing conclusions.",
            metadata.get("sklearn_version"),
            sklearn.__version__,
        )
    return bundle[_BUNDLE_KEY_MODEL], metadata


def write_metadata_sidecar(model_path: str | Path) -> Path:
    """Dump the metadata block of a bundle as a sibling ``.json`` file.

    Handy for audits and version control diffs of model provenance.
    """
    source = Path(model_path)
    _, metadata = load_model(source)
    sidecar = source.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return sidecar
