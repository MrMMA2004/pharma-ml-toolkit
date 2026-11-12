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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy
import sklearn

from pharmaml import __version__
from pharmaml.exceptions import DataValidationError, PersistenceError
from pharmaml.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

_LOG = get_logger(__name__)

__all__ = [
    "ModelInputReport",
    "bundle_metadata",
    "feature_ranges_from_frame",
    "load_model",
    "save_model",
    "validate_bundle_input",
    "write_metadata_sidecar",
]

def feature_ranges_from_frame(
    frame: Any, features: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Per-feature min/max box of a training frame (the applicability domain).

    Records the range each numeric feature was fitted inside, so a saved
    bundle can flag out-of-domain queries at inference time.  Non-numeric
    columns are skipped (they are one-hot encoded inside the pipeline, so
    their raw range is not a meaningful box).  When ``features`` is given,
    only those columns are considered and their order is preserved.
    """
    columns = list(features) if features else list(frame.columns)
    box: dict[str, dict[str, float]] = {}
    for name in columns:
        if name not in frame.columns:
            continue
        series = frame[name]
        if getattr(series.dtype, "kind", "") in "iufc":
            box[str(name)] = {"min": float(series.min()), "max": float(series.max())}
    return box


_BUNDLE_KEY_MODEL = "model"
_BUNDLE_KEY_META = "metadata"


def bundle_metadata(
    name: str,
    task: str,
    features: list[str] | None = None,
    best_params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    target: str | None = None,
    feature_ranges: dict[str, dict[str, float]] | None = None,
    feature_distributions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance metadata block for a model bundle.

    ``target`` records the training target column name so inference
    callers (the CLI check, the playground's predict card) can drop it
    automatically when the scored CSV still carries it -- the same
    convention the toolkit's own check flow documents.  ``feature_ranges``
    records the per-feature training box (see
    :func:`feature_ranges_from_frame`) so inference can flag rows outside
    it without re-reading the training data; ``feature_distributions``
    records the per-feature quantile sketch (see
    :func:`pharmaml.ml.drift.feature_distributions_from_frame`) so a later
    batch can be checked for distribution shift with no training frame.
    """
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
        "target": target,
        "feature_ranges": feature_ranges or None,
        "feature_distributions": feature_distributions or None,
    }


def save_model(
    model: Any,
    path: str | Path,
    name: str,
    task: str,
    features: list[str] | None = None,
    best_params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    target: str | None = None,
    feature_ranges: dict[str, dict[str, float]] | None = None,
    feature_distributions: dict[str, dict[str, Any]] | None = None,
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
    target:
        Training target column name; inference-time audits drop it
        automatically when the scored CSV still carries it.
    feature_ranges:
        Per-feature training min/max box (see
        :func:`feature_ranges_from_frame`); inference flags rows outside
        it without re-reading the training data.
    feature_distributions:
        Per-feature quantile sketch (see
        :func:`pharmaml.ml.drift.feature_distributions_from_frame`) so a
        later batch can be drift-checked with no training frame.

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
    target_path = Path(path)
    if target_path.suffix.lower() not in {".joblib", ".pkl"}:
        target_path = target_path.with_suffix(".joblib")

    bundle = {
        _BUNDLE_KEY_MODEL: model,
        _BUNDLE_KEY_META: bundle_metadata(
            name,
            task,
            features,
            best_params,
            metrics,
            target=target,
            feature_ranges=feature_ranges,
            feature_distributions=feature_distributions,
        ),
    }
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, target_path, compress=3)
    except Exception as exc:
        raise PersistenceError(
            f"Failed to save model to {target_path}: {exc}"
        ) from exc
    _LOG.info("Saved model '%s' (%s) to %s", name, task, target_path)
    return target_path


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


@dataclass(frozen=True)
class ModelInputReport:
    """Verdict of :func:`validate_bundle_input` — the round-trip audit.

    ``ok`` is ``True`` only when the frame satisfies the bundle's
    feature contract exactly (names *and* order, or the feature count
    when names were never recorded).  Version drift is reported as
    warnings — it never flips ``ok`` by itself, because a drifted stack
    may still predict fine; the audit names it so the caller decides.

    ``__str__`` renders the one-line verdict plus every problem, the
    format ``pharmaml ml check`` prints.
    """

    ok: bool
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    reordered: bool = False
    count_only: bool = False
    version_warnings: tuple[str, ...] = ()
    messages: tuple[str, ...] = field(default=())

    def __str__(self) -> str:
        head = "OK" if self.ok else "FAIL"
        lines = [f"audit: {head}"]
        lines.extend(f"  - {message}" for message in self.messages)
        return "\n".join(lines)


def validate_bundle_input(path: str | Path, frame: pd.DataFrame) -> ModelInputReport:
    """Audit a new dataset against a saved model bundle before inference.

    The production failure mode this closes: a model saved months ago,
    a regenerated CSV, and a crash deep inside sklearn with a cryptic
    feature-count error.  The bundle's metadata already records the
    ordered feature names — this function checks the new frame against
    that contract up front and names every difference:

    * **missing** — features the model expects but the frame lacks;
    * **extra** — columns the model has never seen;
    * **reordered** — the same names in a different order (sklearn
      rejects those even when the *set* matches);
    * **count contract** — when the bundle was saved without feature
      names, the model's ``n_features_in_`` is compared with the
      frame's width instead;
    * **version warnings** — sklearn/pharmaml drift between training
      and load time (the same conditions :func:`load_model` only logs,
      surfaced here so CLI and GUI callers can show them).

    Parameters
    ----------
    path:
        A bundle written by :func:`save_model`.
    frame:
        The new data (anything with ``.columns``, i.e. a DataFrame).

    Returns
    -------
    ModelInputReport
        ``ok`` plus the column buckets; ``messages`` carries every
        problem in a human-readable form.

    Raises
    ------
    DataValidationError
        When ``frame`` has no ``columns`` attribute.
    PersistenceError
        When the bundle cannot be loaded (see :func:`load_model`).

    Examples
    --------
    >>> from pharmaml.ml.persistence import save_model, validate_bundle_input
    >>> from sklearn.linear_model import LinearRegression
    >>> import pandas as pd
    >>> model = LinearRegression().fit([[0.0], [1.0], [2.0]], [0.0, 2.0, 4.0])
    >>> path = save_model(model, "/tmp/pharmaml_doctest_model/audit.joblib",
    ...                   name="y=2x", task="regression", features=["x"])
    >>> report = validate_bundle_input(path, pd.DataFrame({"x": [3.0]}))
    >>> report.ok
    True
    >>> drifted = validate_bundle_input(path, pd.DataFrame({"x": [1.0], "z": [2.0]}))
    >>> (drifted.ok, drifted.extra)
    (False, ('z',))
    >>> import shutil
    >>> shutil.rmtree("/tmp/pharmaml_doctest_model")
    """
    if not hasattr(frame, "columns"):
        raise DataValidationError(
            "validate_bundle_input needs a DataFrame (with .columns), "
            f"got {type(frame).__name__}"
        )
    model, metadata = load_model(path)

    messages: list[str] = []
    version_warnings: list[str] = []
    trained_sklearn = metadata.get("sklearn_version")
    if trained_sklearn is not None and trained_sklearn != sklearn.__version__:
        warning = (
            f"sklearn drift: trained with {trained_sklearn}, runtime is "
            f"{sklearn.__version__} -- re-validate before drawing conclusions"
        )
        version_warnings.append(warning)
        messages.append(f"warning: {warning}")
    trained_pharmaml = metadata.get("pharmaml_version")
    if trained_pharmaml is not None and trained_pharmaml != __version__:
        warning = (
            f"pharmaml drift: trained with {trained_pharmaml}, runtime is "
            f"{__version__}"
        )
        version_warnings.append(warning)
        messages.append(f"warning: {warning}")

    expected = metadata.get("features")
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    reordered = False
    count_only = False

    if isinstance(expected, list) and expected:
        columns = [str(column) for column in frame.columns]
        wanted = [str(column) for column in expected]
        missing = tuple(name for name in wanted if name not in columns)
        extra = tuple(name for name in columns if name not in wanted)
        if missing:
            messages.append(
                f"missing {len(missing)} feature(s): {', '.join(missing)}"
            )
        if extra:
            messages.append(
                f"{len(extra)} extra column(s) the model does not know: "
                f"{', '.join(extra)}"
            )
        if not missing and not extra and columns != wanted:
            reordered = True
            messages.append(
                "feature columns are in a different order than trained on "
                f"(expected: {', '.join(wanted[:5])}"
                + ("…" if len(wanted) > 5 else "") + ")"
            )
    else:
        count_only = True
        n_in = getattr(model, "n_features_in_", None)
        if n_in is not None and int(n_in) != len(frame.columns):
            messages.append(
                f"feature-count contract: model expects {int(n_in)} "
                f"feature(s), frame has {len(frame.columns)} column(s)"
            )

    ok = not messages or all(m.startswith("warning:") for m in messages)
    return ModelInputReport(
        ok=ok,
        missing=missing,
        extra=extra,
        reordered=reordered,
        count_only=count_only,
        version_warnings=tuple(version_warnings),
        messages=tuple(messages),
    )


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
