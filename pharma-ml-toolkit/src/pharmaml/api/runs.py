"""part16 §15.2 — experiment tracking / run lineage (the Phase-12 lane).

One first-class, queryable record per scientific computation, joining
the §15.2 reproducibility checklist in a single view: the dataset
identifier (the DatasetRecord's identifying fields), the feature
configuration, the preprocessing pipeline, the algorithm + toolkit
version, the hyperparameters (requested AND tuned), the random seed,
the cross-validation configuration, the evaluation metrics, the
environment snapshot, the model-artifact reference, the execution
timestamps, and the job reference whose JobManager history carries the
full stage/error log (``GET /api/v1/jobs/{job_id}``).

Integration rule (§8 — no parallel architecture): the records are
BUILT at the two points the ml.train route already holds every
ingredient (success + failure), from the very objects the route
already built — the dataset view, the TrainRequest echo, the
TrainingResult, the TrainedModelRecord — never re-computed, never
re-derived from a second source.  The store follows the sibling-store
shape: a bounded in-memory registry (oldest-first evict) plus one
SQLite row per run in durable mode (MIGRATIONS v3), recovered on
startup.

Failed runs are recorded too: a doomed configuration is scientific
information ("do not re-run this"), and the record carries the typed
error the job surfaced.  Cancelled runs record their own status for
the same reason.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .stores import json_safe

__all__ = [
    "MAX_RUNS",
    "ExperimentRunRecord",
    "RunStore",
    "build_ml_train_failed_run",
    "build_ml_train_run",
]

#: Bounded in-memory registry (the sibling-store convention).
MAX_RUNS = 64

#: The §15.2 preprocessing description — the fitted estimator embeds
#: the toolkit's ``build_preprocessor`` when ``preprocess`` is on.
_PREPROCESS_PIPELINE = (
    "median imputation -> StandardScaler (non-numeric columns: "
    "most_frequent imputation + one-hot encoding, handle_unknown=ignore)"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentRunRecord:
    """One experiment run: the §15.2 checklist as a single lineage view."""

    def __init__(
        self,
        run_id: str,
        family: str,
        mode: str,
        status: str,
        job_id: str | None,
        dataset: dict[str, Any],
        feature_configuration: dict[str, Any],
        preprocessing: dict[str, Any],
        algorithm: dict[str, Any],
        hyperparameters: dict[str, Any],
        random_seed: int,
        cross_validation: dict[str, Any],
        evaluation: dict[str, Any],
        model_artifact: dict[str, Any] | None,
        environment: dict[str, Any],
        logs: dict[str, Any],
        started_at: str,
        completed_at: str,
        error: str | None = None,
        created_at: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.family = family
        self.mode = mode
        self.status = status
        self.job_id = job_id
        self.dataset = dataset
        self.feature_configuration = feature_configuration
        self.preprocessing = preprocessing
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.random_seed = random_seed
        self.cross_validation = cross_validation
        self.evaluation = evaluation
        self.model_artifact = model_artifact
        self.environment = environment
        self.logs = logs
        self.started_at = started_at
        self.completed_at = completed_at
        self.error = error
        self.created_at = created_at or _now_iso()

    def summary(self) -> dict[str, Any]:
        """The list-view projection (the lineage table's row)."""
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "family": self.family,
            "mode": self.mode,
            "status": self.status,
            "job_id": self.job_id,
            "dataset_id": self.dataset.get("dataset_id"),
            "dataset_name": self.dataset.get("name"),
            "model_name": self.algorithm.get("name"),
            "task": self.algorithm.get("task"),
            "metric": self.evaluation.get("cv_mean"),
            "cv_folds": self.cross_validation.get("cv_folds"),
            "random_seed": self.random_seed,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload

    def view(self) -> dict[str, Any]:
        """The full §15.2 record (the detail endpoint's payload)."""
        payload = self.summary()
        payload.update(
            {
                "dataset": self.dataset,
                "feature_configuration": self.feature_configuration,
                "preprocessing": self.preprocessing,
                "algorithm": self.algorithm,
                "hyperparameters": self.hyperparameters,
                "cross_validation": self.cross_validation,
                "evaluation": self.evaluation,
                "model_artifact": self.model_artifact,
                "environment": self.environment,
                "logs": self.logs,
                "schema": "pharmaml-experiment-run/1",
            }
        )
        return json_safe(payload)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExperimentRunRecord:
        """Rehydrate from the persisted full view (the durable lane's
        startup recovery — JSON only, nothing to unpickle)."""
        return cls(
            run_id=payload["run_id"],
            family=payload["family"],
            mode=payload["mode"],
            status=payload["status"],
            job_id=payload.get("job_id"),
            dataset=payload.get("dataset") or {},
            feature_configuration=payload.get("feature_configuration") or {},
            preprocessing=payload.get("preprocessing") or {},
            algorithm=payload.get("algorithm") or {},
            hyperparameters=payload.get("hyperparameters") or {},
            random_seed=int(payload.get("random_seed") or 0),
            cross_validation=payload.get("cross_validation") or {},
            evaluation=payload.get("evaluation") or {},
            model_artifact=payload.get("model_artifact"),
            environment=payload.get("environment") or {},
            logs=payload.get("logs") or {},
            started_at=payload.get("started_at") or payload["created_at"],
            completed_at=payload.get("completed_at") or payload["created_at"],
            error=payload.get("error"),
            created_at=payload.get("created_at"),
        )


def _dataset_lineage(dataset_view: dict[str, Any]) -> dict[str, Any]:
    """The dataset identifier block (§15.2 'dataset identifiers and
    versions') — the DatasetRecord's own identifying fields, no
    re-derivation."""
    return {
        "dataset_id": dataset_view.get("dataset_id"),
        "name": dataset_view.get("name"),
        "source_filename": dataset_view.get("source_filename"),
        "size_bytes": dataset_view.get("size_bytes"),
        "shape": dataset_view.get("shape"),
        "target": dataset_view.get("target"),
        "task": dataset_view.get("task"),
        "n_features": dataset_view.get("n_features"),
        "created_at": dataset_view.get("created_at"),
    }


def build_ml_train_run(
    *,
    mode: str,
    job_id: str | None,
    dataset_view: dict[str, Any],
    model_name: str,
    task: str,
    tune: bool,
    cv: int,
    n_jobs_requested: int,
    n_jobs_effective: int,
    parallelism_advisory: dict[str, Any] | None,
    preprocess: bool,
    random_state: int,
    model_params: dict[str, Any] | None,
    split: dict[str, Any] | None = None,
    training_result: Any,
    model_view: dict[str, Any],
    environment: dict[str, Any],
    toolkit_version: str,
    started_at: str,
    completed_at: str,
) -> ExperimentRunRecord:
    """The completed-run record, from the objects the route already built.

    ``split`` (ML-3) is the strategy-aware held-out carve block the
    train route computed before the fit — strategy, the two sides'
    row counts, the group census and the ``shared_groups`` verdict —
    riding the checklist's own cross-validation slot so the lineage
    states WHICH frame the cv scores were computed over.
    """
    return ExperimentRunRecord(
        run_id=f"run-{uuid_hex12()}",
        family="ml.train",
        mode=mode,
        status="completed",
        job_id=job_id,
        dataset=_dataset_lineage(dataset_view),
        feature_configuration={
            "feature_names": dataset_view.get("feature_names"),
            "n_features": dataset_view.get("n_features"),
            "target": dataset_view.get("target"),
        },
        preprocessing={
            "enabled": bool(preprocess),
            "pipeline": _PREPROCESS_PIPELINE if preprocess else "none (raw features)",
        },
        algorithm={
            "name": model_name,
            "task": task,
            "toolkit_version": toolkit_version,
        },
        hyperparameters={
            "requested": dict(model_params or {}),
            "tuned": bool(tune),
            "best_params": model_view.get("best_params"),
        },
        random_seed=random_state,
        cross_validation={
            "cv_folds": cv,
            "tuning": bool(tune),
            "n_jobs_requested": n_jobs_requested,
            "n_jobs_effective": n_jobs_effective,
            "parallelism_advisory": parallelism_advisory,
            "split": split,
        },
        evaluation={
            "cv_scores": model_view.get("cv_scores"),
            "cv_mean": model_view.get("cv_mean"),
            "cv_std": model_view.get("cv_std"),
            "primary_metric": training_result_primary_metric(task),
        },
        model_artifact={
            "model_id": model_view.get("model_id"),
            "name": model_view.get("name"),
            "task": model_view.get("task"),
            "n_train_rows": model_view.get("n_train_rows"),
            "fit_seconds": model_view.get("fit_seconds"),
            "created_at": model_view.get("created_at"),
            "fetch": f"GET /api/v1/ml/trained/{model_view.get('model_id')}",
        },
        environment=environment,
        logs={
            "job_id": job_id,
            "stage": "completed",
            "history": (
                f"GET /api/v1/jobs/{job_id}"
                if job_id
                else "sync mode (no job history)"
            ),
        },
        started_at=started_at,
        completed_at=completed_at,
    )


def build_ml_train_failed_run(
    *,
    mode: str,
    job_id: str | None,
    dataset_view: dict[str, Any],
    model_name: str,
    task: str,
    tune: bool,
    cv: int,
    n_jobs_requested: int,
    preprocess: bool,
    random_state: int,
    model_params: dict[str, Any] | None,
    split: dict[str, Any] | None = None,
    environment: dict[str, Any],
    toolkit_version: str,
    started_at: str,
    completed_at: str,
    error: str,
    status: str = "failed",
) -> ExperimentRunRecord:
    """The failed/cancelled-run record — the config that did not land.

    Everything the §15.2 checklist can state pre-fit is recorded so a
    doomed configuration is traceable and avoidable; the fields only
    knowable after a successful fit stay absent (never invented).
    ``split`` (ML-3) carries the carve as it stood: the computed block
    when the carve ran and the fit failed, or the requested echo plus
    its ``error`` when the carve itself was refused.
    """
    return ExperimentRunRecord(
        run_id=f"run-{uuid_hex12()}",
        family="ml.train",
        mode=mode,
        status=status,
        job_id=job_id,
        dataset=_dataset_lineage(dataset_view),
        feature_configuration={
            "feature_names": dataset_view.get("feature_names"),
            "n_features": dataset_view.get("n_features"),
            "target": dataset_view.get("target"),
        },
        preprocessing={
            "enabled": bool(preprocess),
            "pipeline": _PREPROCESS_PIPELINE if preprocess else "none (raw features)",
        },
        algorithm={
            "name": model_name,
            "task": task,
            "toolkit_version": toolkit_version,
        },
        hyperparameters={
            "requested": dict(model_params or {}),
            "tuned": bool(tune),
            "best_params": None,
        },
        random_seed=random_state,
        cross_validation={
            "cv_folds": cv,
            "tuning": bool(tune),
            "n_jobs_requested": n_jobs_requested,
            "n_jobs_effective": None,
            "parallelism_advisory": None,
            "split": split,
        },
        evaluation={
            "cv_scores": None,
            "cv_mean": None,
            "cv_std": None,
            "primary_metric": None,
        },
        model_artifact=None,
        environment=environment,
        logs={
            "job_id": job_id,
            "stage": status,
            "history": (
                f"GET /api/v1/jobs/{job_id}"
                if job_id
                else "sync mode (no job history)"
            ),
        },
        started_at=started_at,
        completed_at=completed_at,
        error=error,
    )


def training_result_primary_metric(task: str) -> str:
    """The CV scorer the trainer used for this task (training.py pins it)."""
    if task == "classification":
        return "roc_auc_ovo (balanced_accuracy without predict_proba)"
    return "neg_root_mean_squared_error"


def uuid_hex12() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


class RunStore:
    """Bounded registry of the session's experiment runs (the sibling shape)."""

    def __init__(
        self,
        persistence: Any | None = None,
        max_items: int = MAX_RUNS,
    ) -> None:
        self._max_items = max_items
        self._items: dict[str, ExperimentRunRecord] = {}
        self._persistence = persistence

    def register(self, record: ExperimentRunRecord) -> None:
        self._items[record.run_id] = record
        if self._persistence is not None:
            self._persistence.record_run(record.view())
        while len(self._items) > self._max_items:
            oldest = next(iter(self._items))
            self._items.pop(oldest, None)

    def get(self, run_id: str) -> dict[str, Any] | None:
        record = self._items.get(run_id)
        return None if record is None else record.view()

    def list(
        self,
        family: str | None = None,
        dataset_id: str | None = None,
        model_name: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        views = [r.summary() for r in self._items.values()]
        if family is not None:
            views = [v for v in views if v["family"] == family]
        if dataset_id is not None:
            views = [v for v in views if v["dataset_id"] == dataset_id]
        if model_name is not None:
            views = [v for v in views if v["model_name"] == model_name]
        if status is not None:
            views = [v for v in views if v["status"] == status]
        views = sorted(views, key=lambda v: (v["created_at"], v["run_id"]))
        return list(reversed(views))[: int(limit)]
