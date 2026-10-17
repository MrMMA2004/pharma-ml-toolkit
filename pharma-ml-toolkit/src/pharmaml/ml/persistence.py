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

Examples
--------
>>> from pharmaml.ml.persistence import bundle_metadata, load_model, save_model
>>> metadata = bundle_metadata("y=2x", "regression", features=["x"], metrics={"rmse": 0.0})
>>> sorted(metadata)[:4]
['best_params', 'features', 'metrics', 'name']
>>> "sklearn_version" in metadata        # provenance pins the training stack
True
>>> (metadata["name"], metadata["task"], metadata["features"], metadata["metrics"])
('y=2x', 'regression', ['x'], {'rmse': 0.0})
>>> metadata["trained_at"].endswith("+00:00")     # UTC ISO timestamp
True
>>> bundle_metadata("demo", "classification")["features"] is None
True

>>> from sklearn.linear_model import LinearRegression
>>> model = LinearRegression().fit([[0.0], [1.0], [2.0]], [0.0, 2.0, 4.0])
>>> path = save_model(model, "/tmp/pharmaml_doctest_model/lin.joblib",
...                   name="y=2x", task="regression", features=["x"],
...                   metrics={"rmse": 0.0})
>>> path.name
'lin.joblib'
>>> restored, metadata = load_model(path)
>>> [round(float(v), 6) for v in restored.predict([[3.0], [4.0]])]
[6.0, 8.0]
>>> restored.predict([[3.0]]).tolist() == model.predict([[3.0]]).tolist()
True
>>> (metadata["name"], metadata["task"], metadata["features"], metadata["metrics"])
('y=2x', 'regression', ['x'], {'rmse': 0.0})
>>> metadata["trained_at"].endswith("+00:00")
True

>>> from pharmaml.ml.persistence import write_metadata_sidecar
>>> coerced = save_model(model, "/tmp/pharmaml_doctest_model/legacy.txt",
...                      name="y=2x", task="regression")   # unknown suffix -> .joblib
>>> coerced.name
'legacy.joblib'
>>> pkl = save_model(model, "/tmp/pharmaml_doctest_model/legacy.pkl",
...                 name="y=2x", task="regression")        # .pkl passes through
>>> pkl.name
'legacy.pkl'
>>> sidecar = write_metadata_sidecar(path)
>>> sidecar.name
'lin.json'
>>> import json
>>> payload = json.loads(sidecar.read_text())
>>> (payload["name"], payload["task"], payload["features"], payload["metrics"])
('y=2x', 'regression', ['x'], {'rmse': 0.0})
>>> load_model("/tmp/pharmaml_doctest_missing.joblib")
Traceback (most recent call last):
...
pharmaml.exceptions.PersistenceError: Model file not found: /tmp/pharmaml_doctest_missing.joblib
>>> for artifact in (path, coerced, pkl, sidecar):
...     artifact.unlink()
>>> import shutil
>>> shutil.rmtree("/tmp/pharmaml_doctest_model")
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

__all__ = ["bundle_metadata", "load_model", "save_model", "write_metadata_sidecar"]

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

    Raises
    ------
    PersistenceError
        When the parent directory cannot be created or the artifact
        cannot be written (permissions, unwritable location, joblib
        serialisation failures).
    """
    target = Path(path)
    if target.suffix.lower() not in {".joblib", ".pkl"}:
        target = target.with_suffix(".joblib")

    bundle = {
        _BUNDLE_KEY_MODEL: model,
        _BUNDLE_KEY_META: bundle_metadata(name, task, features, best_params, metrics),
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
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
        If the file is missing, not a PharmaML bundle (a malformed
        metadata block included), or was written by an incompatible
        joblib/sklearn generation.
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
    if not isinstance(metadata, dict):
        raise PersistenceError(
            f"'{source}' has a malformed metadata block "
            f"({type(metadata).__name__}) -- re-save the model with "
            "pharmaml.ml.persistence.save_model"
        )
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

    Raises
    ------
    PersistenceError
        When ``model_path`` cannot be loaded as a PharmaML bundle.
    """
    source = Path(model_path)
    _, metadata = load_model(source)
    sidecar = source.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return sidecar
