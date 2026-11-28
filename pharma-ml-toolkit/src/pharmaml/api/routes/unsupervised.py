"""``/api/v1/unsupervised`` — the part18 §4.3/§4.5 lane (extended
by ML-4, the WORKFLOW_ENHANCEMENT_ROADMAP's row 11): the clustering
families (DBSCAN / Mean Shift / Spectral + k-means / GMM /
hierarchical), PCA reduction, and anomaly detection (Isolation
Forest) over registered datasets.

Every route runs the REAL backend modules
(:mod:`pharmaml.ml.clustering`, :mod:`pharmaml.ml.unsupervised`,
:mod:`pharmaml.ml.anomaly`) on the dataset store's ``X`` matrix with
typed request models, sync/job modes, honest degenerate reporting
(all-noise DBSCAN, single-cluster mean-shift return their states,
never fabricated scores; a k-means scan whose silhouette is not
computable reports the None), and json-safe previews of the
per-sample output.  Clustering, PCA and anomaly detection are
UNSUPERVISED surfaces: no target column is consulted, no train/test
split applies, and the responses never imply one.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from pharmaml.ml.anomaly import AnomalyResult, isolation_forest_anomaly
from pharmaml.ml.clustering import (
    ClusterDiagnostics,
    DbscanResult,
    MeanShiftClusterResult,
    SpectralResult,
    _cluster_diagnostics,
    dbscan_cluster,
    meanshift_cluster,
    spectral_cluster,
)
from pharmaml.ml.unsupervised import (
    GmmResult,
    HierarchicalResult,
    KMeansResult,
    PcaResult,
    _as_numeric_matrix,
    gmm_cluster,
    hierarchical_cluster,
    kmeans_cluster,
    pca_reduce,
)

from ..jobs import CancelledJob
from ..stores import json_safe

router = APIRouter(prefix="/unsupervised", tags=["unsupervised"])

_ClusterMethod = Literal[
    "dbscan", "meanshift", "spectral", "kmeans", "gmm", "hierarchical"
]


class ClusterRequest(BaseModel):
    """POST /api/v1/unsupervised/cluster — the shared request body."""

    dataset_id: str
    method: _ClusterMethod = "dbscan"
    # DBSCAN controls
    eps: float | None = Field(
        default=None, gt=0.0,
        description="DBSCAN neighbourhood radius; omit to use the "
        "k-distance estimate (reported as eps_source)",
    )
    min_samples: int = Field(default=5, ge=1, le=500)
    metric: Literal["euclidean", "manhattan", "cosine"] = "euclidean"
    # Mean Shift controls
    bandwidth: float | None = Field(
        default=None, gt=0.0,
        description="Mean Shift kernel bandwidth; omit to estimate",
    )
    bin_seeding: bool = False
    # Spectral controls
    n_clusters: int | None = Field(
        default=None, ge=2,
        description="Spectral Clustering cluster count (required for "
        "the spectral method)",
    )
    affinity: Literal["rbf", "nearest_neighbors"] = "rbf"
    gamma: float = Field(default=1.0, gt=0.0)
    n_neighbors: int = Field(default=10, ge=1)
    # k-means controls (ML-4)
    k: int | None = Field(
        default=None, ge=2,
        description="k-means cluster count; omit to scan k in the "
        "engine's own range and keep the best silhouette",
    )
    # GMM controls (ML-4)
    n_components: int | None = Field(
        default=None, ge=1,
        description="GMM component count; omit to sweep the "
        "engine's own range and keep the criterion minimum",
    )
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    criterion: Literal["bic", "aic"] = "bic"
    # hierarchical controls (ML-4)
    linkage: Literal["ward", "single", "complete", "average"] = "ward"
    # shared
    mode: Literal["job", "sync"] = "job"


class AnomalyRequest(BaseModel):
    """POST /api/v1/unsupervised/anomaly — Isolation Forest."""

    dataset_id: str
    contamination: float | str = "auto"
    n_estimators: int = Field(default=200, ge=1, le=1000)
    seed: int = 42
    mode: Literal["job", "sync"] = "job"


class UnsupervisedPcaRequest(BaseModel):
    """POST /api/v1/unsupervised/pca — the ML-4 PCA sibling.

    The unique component name follows the family's anti-collision
    rule (the ADAnalyzeRequest precedent): routes/md.py carries its
    own ``PcaRequest`` for ``POST /api/v1/md/pca``, and FastAPI's
    collision fallback picks the plain-name winner nondeterministically
    (hash-seed-dependent set iteration) — unique class names keep the
    exported contract byte-stable across processes.
    """

    dataset_id: str
    n_components: int | None = Field(
        default=None, ge=1,
        description="component count; omit to keep the engine's own "
        "cap min(10, n_samples, n_features)",
    )
    scale: bool = True
    mode: Literal["job", "sync"] = "job"


def _diagnostics_view(diag: ClusterDiagnostics) -> dict[str, Any]:
    return {
        "n_clusters": int(diag.n_clusters),
        "silhouette": json_safe(diag.silhouette),
        "davies_bouldin": json_safe(diag.davies_bouldin),
        "calinski_harabasz": json_safe(diag.calinski_harabasz),
        "noise_count": int(diag.noise_count),
        "noise_ratio": json_safe(diag.noise_ratio),
        "cluster_sizes": json_safe(
            {str(k): int(v) for k, v in diag.sizes.items()}
        ),
    }


def _labels_preview(labels: Any, limit: int = 12) -> dict[str, Any]:
    array = np.asarray(labels)
    return {
        "n": int(array.shape[0]),
        "preview": json_safe(array[:limit].tolist()),
        "unique": json_safe(sorted({int(v) for v in array})),
    }


def _cluster_body(
    dataset_id: str,
    dataset_X,
    method: str,
    params: dict,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL clustering fit over dataset.X.

    The ML-4 additions (kmeans / gmm / hierarchical) ride the same
    shared block; their diagnostics come from the SAME
    ``_cluster_diagnostics`` metric builder the original three
    families' engines call internally (single-cluster honesty: the
    similarity metrics report None, never a fabricated score).
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the clustering fit.")
    result: DbscanResult | MeanShiftClusterResult | SpectralResult | KMeansResult | GmmResult | HierarchicalResult
    diagnostics: ClusterDiagnostics
    if method == "dbscan":
        result = dbscan_cluster(
            dataset_X, params.get("eps"), min_samples=params["min_samples"],
            metric=params["metric"],
        )
        diagnostics = result.diagnostics
        payload = {
            "method": "dbscan",
            "eps": json_safe(result.eps),
            "eps_source": result.eps_source,
            "min_samples": result.min_samples,
        }
    elif method == "meanshift":
        result = meanshift_cluster(
            dataset_X, params.get("bandwidth"),
            bin_seeding=params["bin_seeding"],
        )
        diagnostics = result.diagnostics
        payload = {
            "method": "meanshift",
            "bandwidth": json_safe(result.bandwidth),
            "bandwidth_source": result.bandwidth_source,
            "n_modes": result.n_clusters,
            "centers_preview": json_safe(result.centers.head(8)),
        }
    elif method == "spectral":
        if params.get("n_clusters") is None:
            raise HTTPException(
                status_code=422,
                detail="spectral clustering needs an explicit n_clusters",
            )
        result = spectral_cluster(
            dataset_X, params["n_clusters"], affinity=params["affinity"],
            gamma=params["gamma"], n_neighbors=params["n_neighbors"],
        )
        diagnostics = result.diagnostics
        payload = {
            "method": "spectral",
            "n_clusters": result.n_clusters,
            "affinity": result.affinity,
            "gamma": json_safe(result.gamma),
            "n_neighbors": json_safe(result.n_neighbors),
        }
    elif method == "kmeans":
        matrix, _names, _index = _as_numeric_matrix(
            dataset_X, caller="kmeans_cluster"
        )
        result = kmeans_cluster(dataset_X, params.get("k"))
        diagnostics = _cluster_diagnostics(
            matrix, np.asarray(result.labels), caller="unsupervised.kmeans"
        )
        payload = {
            "method": "kmeans",
            "k": result.k,
            "k_source": (
                "manual"
                if params.get("k") is not None
                else "the silhouette scan over the engine's own k range"
            ),
            "kmeans_method": result.method,
            "inertia": json_safe(result.inertia),
            "kmeans_silhouette": json_safe(result.silhouette),
            "scan": json_safe(
                []
                if result.scan.empty
                else result.scan.head(12).to_dict(orient="records")
            ),
            "sizes": json_safe(
                {str(key): int(count) for key, count in result.sizes.items()}
            ),
        }
    elif method == "gmm":
        matrix, _names, _index = _as_numeric_matrix(
            dataset_X, caller="gmm_cluster"
        )
        result = gmm_cluster(
            dataset_X,
            params.get("n_components"),
            covariance_type=params["covariance_type"],
            criterion=params["criterion"],
        )
        diagnostics = _cluster_diagnostics(
            matrix, np.asarray(result.labels), caller="unsupervised.gmm"
        )
        payload = {
            "method": "gmm",
            "n_components": result.n_components,
            "covariance_type": result.covariance_type,
            "chosen_by": result.chosen_by,
            "criterion": params["criterion"],
            "converged": bool(result.converged),
            "log_likelihood": json_safe(result.log_likelihood),
            "weights": json_safe(
                {str(key): float(weight) for key, weight in result.weights.items()}
            ),
            "bic_table": json_safe(
                []
                if result.scan.empty
                else result.scan.head(12).to_dict(orient="records")
            ),
        }
    elif method == "hierarchical":
        matrix, _names, _index = _as_numeric_matrix(
            dataset_X, caller="hierarchical_cluster"
        )
        k = params.get("n_clusters")
        result = hierarchical_cluster(
            dataset_X, k if k is not None else 3, linkage=params["linkage"]
        )
        diagnostics = _cluster_diagnostics(
            matrix, np.asarray(result.labels), caller="unsupervised.hierarchical"
        )
        payload = {
            "method": "hierarchical",
            "k": result.k,
            "k_source": (
                "manual"
                if k is not None
                else "the engine's own default (3)"
            ),
            "linkage": result.linkage,
            "inertia": json_safe(result.inertia),
            "merge_heights": json_safe(list(result.merge_heights)),
        }
    else:  # pragma: no cover - the Literal already binds the set
        raise HTTPException(status_code=422, detail=f"unknown method {method!r}")
    payload.update(
        {
            "dataset_id": dataset_id,
            "n_samples": result.n_samples,
            "diagnostics": _diagnostics_view(diagnostics),
            "labels": _labels_preview(result.labels),
            "summary": str(result),
            "download_hint": (
                "the per-sample assignment table is available as the "
                "labels column of the dataset view joined via "
                "as_frame(); export via the artifacts lane"
            ),
        }
    )
    return {"unsupervised": payload}


def _anomaly_body(
    dataset_id: str,
    dataset_X,
    contamination: float | str,
    n_estimators: int,
    seed: int,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL Isolation Forest fit."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the anomaly fit.")
    result: AnomalyResult = isolation_forest_anomaly(
        dataset_X, contamination=contamination,
        n_estimators=n_estimators, seed=seed,
    )
    return {
        "anomaly": {
            "method": "isolation_forest",
            "dataset_id": dataset_id,
            "contamination": (
                json_safe(result.contamination)
                if isinstance(result.contamination, float)
                else str(result.contamination)
            ),
            "n_estimators": result.n_estimators,
            "seed": result.seed,
            "n_samples": result.n_samples,
            "n_features": result.n_features,
            "n_outliers": int(result.n_outliers),
            "anomaly_fraction": json_safe(result.anomaly_fraction),
            "score_threshold": json_safe(result.threshold),
            "outlier_positions": json_safe(result.outlier_indices),
            "scores_preview": json_safe(
                np.asarray(result.scores)[:10].tolist()
            ),
            "labels": _labels_preview(result.labels),
            "feature_report": json_safe(
                result.feature_report(dataset_X).to_dict(orient="records")
            ),
            "summary": str(result),
            "honesty_note": (
                "anomaly labels are contamination-threshold verdicts, not "
                "ground truth; the feature report is descriptive, not causal"
            ),
        }
    }


def _resolve_dataset(request: Request, dataset_id: str):
    dataset = request.app.state.dataset_store.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown dataset_id.")
    return dataset


@router.get("/methods")
def methods() -> dict:
    """The capability listing (the §9 "complete integration path" door)."""
    return {
        "clustering": [
            {
                "method": "dbscan",
                "python": "pharmaml.ml.clustering.dbscan_cluster",
                "route": "/api/v1/unsupervised/cluster",
                "controls": ["eps", "min_samples", "metric"],
                "notes": "noise-point reporting first class (label -1)",
            },
            {
                "method": "meanshift",
                "python": "pharmaml.ml.clustering.meanshift_cluster",
                "route": "/api/v1/unsupervised/cluster",
                "controls": ["bandwidth"],
                "notes": "modes discovered, no k parameter",
            },
            {
                "method": "spectral",
                "python": "pharmaml.ml.clustering.spectral_cluster",
                "route": "/api/v1/unsupervised/cluster",
                "controls": ["n_clusters", "affinity", "gamma", "n_neighbors"],
                "notes": "graph clustering, distinct from the spectral embedding",
            },
            {
                "method": "kmeans",
                "python": "pharmaml.ml.unsupervised.kmeans_cluster",
                "route": "/api/v1/unsupervised/cluster",
                "controls": ["k"],
                "notes": "elbow AND silhouette: k omitted scans the engine's own range",
            },
            {
                "method": "gmm",
                "python": "pharmaml.ml.unsupervised.gmm_cluster",
                "route": "/api/v1/unsupervised/cluster",
                "controls": ["n_components", "covariance_type", "criterion"],
                "notes": "AIC/BIC sweep; soft responsibilities are the honest membership",
            },
            {
                "method": "hierarchical",
                "python": "pharmaml.ml.unsupervised.hierarchical_cluster",
                "route": "/api/v1/unsupervised/cluster",
                "controls": ["n_clusters", "linkage"],
                "notes": "agglomerative k-cut over the full distance matrix",
            },
        ],
        "anomaly_detection": [
            {
                "method": "isolation_forest",
                "python": "pharmaml.ml.anomaly.isolation_forest_anomaly",
                "route": "/api/v1/unsupervised/anomaly",
                "controls": ["contamination", "n_estimators", "seed"],
            }
        ],
        "dimensionality_reduction": [
            {
                "method": "pca",
                "python": "pharmaml.ml.unsupervised.pca_reduce",
                "route": "/api/v1/unsupervised/pca",
                "controls": ["n_components", "scale"],
                "notes": "explained variance + loadings (the what-does-PC1-mean table)",
            }
        ],
        "unsupervised_rule": (
            "no target column consulted; no train/test split; degenerate "
            "outcomes reported honestly (None metrics, noise rows)"
        ),
    }


@router.post("/cluster")
def cluster(req: ClusterRequest, request: Request) -> dict:
    dataset = _resolve_dataset(request, req.dataset_id)
    params = {
        "eps": req.eps,
        "min_samples": req.min_samples,
        "metric": req.metric,
        "bandwidth": req.bandwidth,
        "bin_seeding": req.bin_seeding,
        "n_clusters": req.n_clusters,
        "affinity": req.affinity,
        "gamma": req.gamma,
        "n_neighbors": req.n_neighbors,
        "k": req.k,
        "n_components": req.n_components,
        "covariance_type": req.covariance_type,
        "criterion": req.criterion,
        "linkage": req.linkage,
    }
    if req.method == "spectral" and req.n_clusters is None:
        raise HTTPException(
            status_code=422,
            detail="spectral clustering needs an explicit n_clusters",
        )
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_cluster_body(req.dataset_id, dataset.X, req.method, params),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "unsupervised.cluster",
        "machine-learning",
        _cluster_body,
        configuration={
            "dataset_id": req.dataset_id,
            "method": req.method,
            **json_safe(params),
        },
        dataset_id=req.dataset_id,
        dataset_X=dataset.X,
        method=req.method,
        params=params,
    )
    return {"mode": "job", "job": record.public()}


@router.post("/anomaly")
def anomaly(req: AnomalyRequest, request: Request) -> dict:
    dataset = _resolve_dataset(request, req.dataset_id)
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_anomaly_body(
                req.dataset_id, dataset.X, req.contamination,
                req.n_estimators, req.seed,
            ),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "unsupervised.anomaly",
        "machine-learning",
        _anomaly_body,
        configuration={
            "dataset_id": req.dataset_id,
            "contamination": req.contamination,
            "n_estimators": req.n_estimators,
            "seed": req.seed,
        },
        dataset_id=req.dataset_id,
        dataset_X=dataset.X,
        contamination=req.contamination,
        n_estimators=req.n_estimators,
        seed=req.seed,
    )
    return {"mode": "job", "job": record.public()}


def _pca_body(
    dataset_id: str,
    dataset_X,
    n_components: int | None,
    scale: bool,
    cancel_event=None,
) -> dict:
    """Job/sync body: the REAL pca_reduce fit (ML-4's PCA sibling)."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJob("Cancelled before the PCA fit.")
    result: PcaResult = pca_reduce(dataset_X, n_components, scale=scale)
    payload = {
        "method": "pca",
        "dataset_id": dataset_id,
        "n_components": int(result.n_components),
        "scaled": bool(result.scaled),
        "n_samples": int(result.n_samples),
        "n_features": int(result.n_features),
        "explained_variance_ratio": json_safe(
            [float(value) for value in result.explained_variance_ratio]
        ),
        "cumulative_variance": json_safe(
            [float(value) for value in result.cumulative_variance]
        ),
        "total_variance_explained": json_safe(result.total_variance_explained),
        "column_names": [str(name) for name in result.column_names],
        "loadings_preview": json_safe(
            result.loadings.head(12).to_dict(orient="records")
        ),
        "scores_preview": json_safe(
            result.transformed.head(12).to_dict(orient="records")
        ),
        "summary": str(result),
        "honesty_note": (
            "the components are the axes of maximum retained variance, "
            "not causal directions; scaled=True standardises the columns "
            "first (the handbook's own choice whenever column scales "
            "differ, which for molecular descriptors they always do)"
        ),
    }
    return {"pca": payload}


@router.post("/pca")
def pca(req: UnsupervisedPcaRequest, request: Request) -> dict:
    """ML-4's PCA sibling — the first plot every chemist asks for on a
    descriptor matrix, served over the real ``pca_reduce`` engine."""
    dataset = _resolve_dataset(request, req.dataset_id)
    if req.mode == "sync":
        return {
            "mode": "sync",
            **_pca_body(req.dataset_id, dataset.X, req.n_components, req.scale),
        }
    manager = request.app.state.job_manager
    record = manager.submit(
        "unsupervised.pca",
        "machine-learning",
        _pca_body,
        configuration={
            "dataset_id": req.dataset_id,
            "n_components": req.n_components,
            "scale": req.scale,
        },
        dataset_id=req.dataset_id,
        dataset_X=dataset.X,
        n_components=req.n_components,
        scale=req.scale,
    )
    return {"mode": "job", "job": record.public()}
