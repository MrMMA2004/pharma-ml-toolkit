"""Session-scoped stores: datasets, trained models, downloadable
artifacts (part10's queued row after the opening foundation).

Everything here was session-scoped exactly like the session token
under part10 — in-memory only, never persisted, fresh per
:func:`~pharmaml.api.create_app`.  **part11 Phase Eleven replaced
that contract**: when a store is constructed with a
:class:`~pharmaml.api.persistence.Persistence` backend (the
``create_app(data_dir=...)`` / ``PHARMAML_STUDIO_DATA`` path, which
the shipped host always takes), every register/evict writes through
to SQLite + the file vault and ``create_app`` recovers the prior
registry on startup — durability across restarts, backend-owned.

Bounds stay honest and *enforced*: each store evicts oldest-first
when its cap is crossed (a local GUI session, not a server, so the
newest work always wins; eviction also removes the persisted row +
vault payload — the retention rule), and a single artifact larger
than the session byte cap is refused outright rather than silently
evicting everything else.
"""

from __future__ import annotations

import hashlib
import pickle
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pharmaml.data import Dataset

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .persistence import Persistence

__all__ = [
    "MAX_ARTIFACTS",
    "MAX_ARTIFACT_TOTAL_BYTES",
    "MAX_DATASETS",
    "MAX_PRIORS",
    "MAX_TRAINED_MODELS",
    "MAX_UPLOAD_BYTES",
    "ArtifactRecord",
    "ArtifactStore",
    "DatasetRecord",
    "DatasetStore",
    "ModelStore",
    "PriorRecord",
    "PriorStore",
    "TrainedModelRecord",
    "json_safe",
]

#: Honest session caps (documented in the OpenAPI descriptions).
MAX_DATASETS = 32
MAX_TRAINED_MODELS = 16
MAX_ARTIFACTS = 64
MAX_ARTIFACT_TOTAL_BYTES = 256 * 1024 * 1024  # 256 MiB of live artifacts
MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64 MiB per dataset upload
MAX_PRIORS = 16  # trained generative priors (SmilesRnn instances)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(value: Any) -> Any:
    """Recursively convert backend values to strict-JSON-safe ones.

    NaN/Infinity become ``null`` (JSON has no NaN); numpy scalars and
    arrays become their Python shapes; timestamps become ISO strings;
    anything exotic falls back to ``str`` so a result never crashes
    the serializer mid-payload.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return None if (np.isnan(value) or np.isinf(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if hasattr(value, "item"):  # remaining numpy 0-d/extension scalars
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):  # pragma: no cover - exotic
            return str(value)
    return str(value)


def frame_payload(frame: pd.DataFrame, max_rows: int = 10) -> dict:
    """A JSON-safe head-view of a frame (NaN cells become ``null``)."""
    head = frame.head(max_rows)
    return {
        "columns": [str(c) for c in frame.columns],
        "dtypes": {
            str(c): str(t) for c, t in zip(frame.columns, frame.dtypes)
        },
        "n_rows": len(frame),
        "returned": len(head),
        "rows": [
            [json_safe(v) for v in row] for row in head.itertuples(index=False)
        ],
    }


class DatasetRecord:
    """One registered dataset (the parsed upload, in memory)."""

    def __init__(
        self,
        dataset_id: str,
        dataset: Dataset,
        name: str,
        source_bytes: int,
        source_filename: str,
    ) -> None:
        self.dataset_id = dataset_id
        self.name = name
        self.frame = dataset.frame
        self.X = dataset.X
        self.y = dataset.y
        self.target = dataset.target
        self.task = dataset.task
        self.classes = dataset.classes
        self.class_counts = dataset.class_counts
        self.dropped_columns = list(dataset.dropped_columns)
        self.categorical_columns = list(dataset.categorical_columns)
        self.summary = dataset.summary().as_dict()
        self.source_bytes = source_bytes
        self.source_filename = source_filename
        self.created_at = _now_iso()

    def view(self, preview_rows: int = 0) -> dict:
        payload = {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source_filename": self.source_filename,
            "size_bytes": self.source_bytes,
            "shape": [int(self.frame.shape[0]), int(self.frame.shape[1])],
            "columns": [str(c) for c in self.frame.columns],
            "target": self.target,
            "task": self.task,
            "n_features": int(self.X.shape[1]),
            "feature_names": [str(c) for c in self.X.columns],
            "classes": json_safe(self.classes),
            "class_counts": json_safe(self.class_counts),
            "dropped_columns": self.dropped_columns,
            "categorical_columns": self.categorical_columns,
            "summary": json_safe(self.summary),
            "created_at": self.created_at,
        }
        if preview_rows:
            payload["preview"] = frame_payload(self.frame, preview_rows)
        return payload


class DatasetStore:
    """Bounded registry of the session's datasets (oldest-first evict)."""

    def __init__(
        self,
        max_items: int = MAX_DATASETS,
        persistence: Persistence | None = None,
    ) -> None:
        self._max_items = max_items
        self._items: dict[str, DatasetRecord] = {}
        self._persistence = persistence

    def register(
        self,
        dataset: Dataset,
        name: str,
        source_bytes: int,
        source_filename: str,
        raw_bytes: bytes | None = None,
    ) -> DatasetRecord:
        dataset_id = f"ds-{uuid.uuid4().hex[:12]}"
        record = DatasetRecord(
            dataset_id, dataset, name, source_bytes, source_filename
        )
        self._items[dataset_id] = record
        if self._persistence is not None and raw_bytes is not None:
            self._persistence.save_dataset(record.view(), raw_bytes)
        while len(self._items) > self._max_items:
            oldest = next(iter(self._items))
            self._forget(oldest)
        return record

    def _forget(self, dataset_id: str) -> None:
        record = self._items.pop(dataset_id, None)
        if record is not None and self._persistence is not None:
            self._persistence.delete_dataset(dataset_id)

    def get(self, dataset_id: str) -> DatasetRecord | None:
        return self._items.get(dataset_id)

    def list(self) -> list[dict]:
        views = [r.view() for r in self._items.values()]
        return sorted(views, key=lambda v: v["created_at"])

    def evict(self, dataset_id: str) -> bool:
        if dataset_id not in self._items:
            return False
        self._forget(dataset_id)
        return True


class TrainedModelRecord:
    """One trained model kept for prediction/artifact export."""

    def __init__(
        self,
        model_id: str,
        model: Any,
        name: str,
        task: str,
        dataset_id: str,
        target: str,
        features: list[str],
        n_train_rows: int,
        best_params: dict,
        cv_scores: Any,
        cv_mean: Any,
        cv_std: Any,
        fit_seconds: float,
    ) -> None:
        self.model_id = model_id
        self.model = model  # the fitted estimator / pipeline (never JSON'd)
        self.name = name
        self.task = task
        self.dataset_id = dataset_id
        self.target = target
        self.features = features
        self.n_train_rows = n_train_rows
        self.best_params = dict(best_params or {})
        self.cv_scores = cv_scores
        self.cv_mean = cv_mean
        self.cv_std = cv_std
        self.fit_seconds = fit_seconds
        self.created_at = _now_iso()

    def view(self) -> dict:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "task": self.task,
            "dataset_id": self.dataset_id,
            "target": self.target,
            "features": self.features,
            "n_train_rows": self.n_train_rows,
            "best_params": json_safe(self.best_params),
            "cv_scores": json_safe(self.cv_scores),
            "cv_mean": json_safe(self.cv_mean),
            "cv_std": json_safe(self.cv_std),
            "fit_seconds": self.fit_seconds,
            "created_at": self.created_at,
        }


class ModelStore:
    """Bounded registry of the session's trained models."""

    def __init__(
        self,
        max_items: int = MAX_TRAINED_MODELS,
        persistence: Persistence | None = None,
    ) -> None:
        self._max_items = max_items
        self._items: dict[str, TrainedModelRecord] = {}
        self._persistence = persistence

    def register(self, record: TrainedModelRecord) -> TrainedModelRecord:
        self._items[record.model_id] = record
        if self._persistence is not None:
            payload = pickle.dumps(record.model, protocol=pickle.HIGHEST_PROTOCOL)
            self._persistence.save_model(record.view(), payload)
        while len(self._items) > self._max_items:
            oldest = next(iter(self._items))
            self._forget(oldest)
        return record

    def _forget(self, model_id: str) -> None:
        record = self._items.pop(model_id, None)
        if record is not None and self._persistence is not None:
            self._persistence.delete_model(model_id)

    def get(self, model_id: str) -> TrainedModelRecord | None:
        return self._items.get(model_id)

    def list(self) -> list[dict]:
        views = [r.view() for r in self._items.values()]
        return sorted(views, key=lambda v: v["created_at"])

    def evict(self, model_id: str) -> bool:
        if model_id not in self._items:
            return False
        self._forget(model_id)
        return True


class PriorRecord:
    """One trained generative prior (a fitted ``SmilesRnn``).

    The model object stays in memory only (never JSON'd, like the
    trained estimators); the view carries the training telemetry so
    the WinUI page can plot the loss curve without the weights.
    """

    def __init__(
        self,
        prior_id: str,
        model: Any,
        name: str,
        params: dict,
        n_train_smiles: int,
        losses: list[float],
        corpus_source: str,
    ) -> None:
        self.prior_id = prior_id
        self.model = model  # the fitted SmilesRnn (never JSON'd)
        self.name = name
        self.params = dict(params or {})
        self.n_train_smiles = int(n_train_smiles)
        self.losses = [float(x) for x in (losses or [])]
        self.corpus_source = corpus_source
        self.created_at = _now_iso()

    def view(self) -> dict:
        return {
            "prior_id": self.prior_id,
            "name": self.name,
            "kind": "smiles_rnn",
            "params": json_safe(self.params),
            "n_train_smiles": self.n_train_smiles,
            "n_epochs": len(self.losses),
            "final_loss": self.losses[-1] if self.losses else None,
            "losses": json_safe(self.losses),
            "corpus_source": self.corpus_source,
            "created_at": self.created_at,
            "sample_url": "/api/v1/generative/rnn/sample",
        }


class PriorStore:
    """Bounded registry of the session's trained generative priors."""

    def __init__(
        self,
        max_items: int = MAX_PRIORS,
        persistence: Persistence | None = None,
    ) -> None:
        self._max_items = max_items
        self._items: dict[str, PriorRecord] = {}
        self._persistence = persistence

    def register(self, record: PriorRecord) -> PriorRecord:
        self._items[record.prior_id] = record
        if self._persistence is not None:
            payload = pickle.dumps(record.model, protocol=pickle.HIGHEST_PROTOCOL)
            self._persistence.save_prior(record.view(), payload)
        while len(self._items) > self._max_items:
            oldest = next(iter(self._items))
            self._forget(oldest)
        return record

    def _forget(self, prior_id: str) -> None:
        record = self._items.pop(prior_id, None)
        if record is not None and self._persistence is not None:
            self._persistence.delete_prior(prior_id)

    def get(self, prior_id: str) -> PriorRecord | None:
        return self._items.get(prior_id)

    def list(self) -> list[dict]:
        views = [r.view() for r in self._items.values()]
        return sorted(views, key=lambda v: v["created_at"])

    def evict(self, prior_id: str) -> bool:
        if prior_id not in self._items:
            return False
        self._forget(prior_id)
        return True


class ArtifactRecord:
    """One downloadable artifact: exact bytes + digest + provenance."""

    def __init__(
        self,
        artifact_id: str,
        kind: str,
        filename: str,
        content_type: str,
        data: bytes,
        meta: dict | None = None,
    ) -> None:
        self.artifact_id = artifact_id
        self.kind = kind
        self.filename = filename
        self.content_type = content_type
        self.data = data
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.size = len(data)
        self.meta = json_safe(dict(meta or {}))
        self.created_at = _now_iso()

    def view(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "meta": self.meta,
            "download_url": f"/api/v1/artifacts/{self.artifact_id}/download",
        }


class ArtifactStore:
    """Bounded byte-verified artifact registry (the download source)."""

    def __init__(
        self,
        max_items: int = MAX_ARTIFACTS,
        max_total_bytes: int = MAX_ARTIFACT_TOTAL_BYTES,
        persistence: Persistence | None = None,
    ) -> None:
        self._max_items = max_items
        self._max_total_bytes = max_total_bytes
        self._items: dict[str, ArtifactRecord] = {}
        self._persistence = persistence

    @property
    def total_bytes(self) -> int:
        return sum(r.size for r in self._items.values())

    def register(
        self,
        kind: str,
        filename: str,
        content_type: str,
        data: bytes,
        meta: dict | None = None,
    ) -> ArtifactRecord:
        if len(data) > self._max_total_bytes:
            raise ValueError(
                f"artifact of {len(data)} bytes exceeds the session cap "
                f"of {self._max_total_bytes} bytes"
            )
        artifact_id = f"art-{uuid.uuid4().hex[:12]}"
        record = ArtifactRecord(
            artifact_id, kind, filename, content_type, data, meta
        )
        self._items[artifact_id] = record
        if self._persistence is not None:
            self._persistence.save_artifact(record.view(), data)
        # oldest-first eviction; the artifact just registered is never
        # evicted by its own registration (guarded by len > 1)
        while len(self._items) > 1 and (
            len(self._items) > self._max_items
            or self.total_bytes > self._max_total_bytes
        ):
            oldest = next(iter(self._items))
            if oldest == artifact_id:  # pragma: no cover - guard order
                break
            self._forget(oldest)
        return record

    def _forget(self, artifact_id: str) -> None:
        record = self._items.pop(artifact_id, None)
        if record is not None and self._persistence is not None:
            self._persistence.delete_artifact(artifact_id)

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        return self._items.get(artifact_id)

    def list(self) -> list[dict]:
        views = [r.view() for r in self._items.values()]
        return sorted(views, key=lambda v: v["created_at"])

    def evict(self, artifact_id: str) -> bool:
        if artifact_id not in self._items:
            return False
        self._forget(artifact_id)
        return True
