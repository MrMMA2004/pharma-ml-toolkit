"""The ML-4 unsupervised-families drift guard (the roadmap's row 11).

The lane spans three sources that must never drift apart:

* the route -- ``routes/unsupervised.py``'s extended ``_ClusterMethod``
  literal (dbscan/meanshift/spectral + kmeans/gmm/hierarchical) wired
  one-to-one to the engines, plus the ``POST /unsupervised/pca``
  sibling over ``pca_reduce``;
* the lane -- ``UnsupervisedLearningLane.tsx``'s six-method select +
  the per-method controls + the PCA card, and the client's extended
  parsers (``ClusterResultView``'s new families + ``PcaResultView``);
* the lane's own vitest file -- ``UnsupervisedMethodsLane.test.tsx``
  (the chip test, the three method round trips, the pca round trip,
  the degenerate single-cluster honesty).

Plus the roadmap's own acceptance clauses:

* the methods literal equals the engine list (the cross-check this
  row opens with);
* the GMM BIC table and the k-means scan ride the responses;
* a degenerate single-cluster answer reports its honest nulls, never
  a fabricated score;
* the contract-count pins ride ``test_frontend_registry.py`` /
  ``test_frontend_runs.py`` (145 -> 146 by this row's pca path).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROUTE = REPO / "src" / "pharmaml" / "api" / "routes" / "unsupervised.py"
LANE = (
    REPO / "frontend" / "src" / "features" / "ml" / "UnsupervisedLearningLane.tsx"
)
CLIENT = REPO / "frontend" / "src" / "services" / "ml" / "unsupervised.ts"
VITEST = (
    REPO / "frontend" / "src" / "features" / "ml" / "__tests__"
    / "UnsupervisedMethodsLane.test.tsx"
)
ROADMAP = REPO / "docs" / "QA" / "WORKFLOW_ENHANCEMENT_ROADMAP.md"


def _route() -> str:
    return ROUTE.read_text(encoding="utf-8")


def _lane() -> str:
    return LANE.read_text(encoding="utf-8")


def test_the_methods_literal_equals_the_engine_list() -> None:
    """The route's Literal mirrors the engines wired one-to-one."""
    src = _route()
    m = re.search(
        r"_ClusterMethod = Literal\[(.*?)\]",
        src,
        re.DOTALL,
    )
    assert m, "the _ClusterMethod literal drifted or was removed"
    literals = {name.strip().strip('"') for name in m.group(1).split(",")}
    assert literals == {
        "dbscan", "meanshift", "spectral", "kmeans", "gmm", "hierarchical",
    }, f"the literal set drifted: {sorted(literals)}"
    # every literal's engine is imported and branched
    for engine in ("kmeans_cluster", "gmm_cluster", "hierarchical_cluster"):
        assert engine in src, f"the engine {engine} left the route imports"
    for method in ("kmeans", "gmm", "hierarchical"):
        assert f'elif method == "{method}"' in src, (
            f"the {method} branch left _cluster_body"
        )
    # the original three stay wired
    for engine in ("dbscan_cluster", "meanshift_cluster", "spectral_cluster"):
        assert engine in src, f"the engine {engine} left the route imports"


def test_the_new_method_controls_ride_the_request_model() -> None:
    """k / n_components / covariance_type / criterion / linkage ride
    the ClusterRequest, and the cluster route forwards them."""
    src = _route()
    for field in (
        "k: int | None",
        "n_components: int | None",
        'covariance_type: Literal["full", "tied", "diag", "spherical"]',
        'criterion: Literal["bic", "aic"]',
        'linkage: Literal["ward", "single", "complete", "average"]',
    ):
        assert field in src, f"the ClusterRequest lost its {field} control"


def test_the_pca_sibling_route_exists() -> None:
    """``POST /unsupervised/pca`` over the real pca_reduce engine."""
    src = _route()
    assert "@router.post(\"/pca\")" in src, "the pca route was removed"
    assert "pca_reduce" in src, "the route stopped calling the real engine"
    assert 'return {"pca": payload}' in _route() or "return {\"pca\": payload}" in src
    assert "honesty_note" in src, "the pca answer lost its honesty note"
    assert "explained_variance_ratio" in src, (
        "the pca answer lost the variance ladder"
    )
    assert "loadings_preview" in src, "the pca answer lost the loadings preview"


def test_the_methods_listing_carries_the_six_plus_pca() -> None:
    """The capability listing mirrors the route's own families."""
    src = _route()
    for method in ("kmeans", "gmm", "hierarchical"):
        assert f'"method": "{method}"' in src, (
            f"the methods listing lost its {method} row"
        )
    assert '"method": "pca"' in src, "the listing lost the pca family row"
    assert "dimensionality_reduction" in src, (
        "the listing lost its dimensionality_reduction family"
    )


def test_the_lane_carries_the_six_chips_and_the_pca_card() -> None:
    """The method select carries six options; the PCA card renders."""
    src = _lane()
    for option in (
        '<option value="kmeans">k-means</option>',
        '<option value="gmm">Gaussian Mixture</option>',
        '<option value="hierarchical">Hierarchical</option>',
    ):
        assert option in src, f"the method select lost its option {option}"
    for label in (
        '"the clustering method"',
        '"the k-means k"',
        '"the GMM components"',
        '"the GMM covariance type"',
        '"the GMM criterion"',
        '"the hierarchical linkage"',
        '"the PCA components"',
    ):
        assert f"aria-label={label}" in src, (
            f"the lane lost its aria-labelled control {label}"
        )
    assert "Reduce (PCA)" in src, "the pca card lost its run button"
    assert "PCA (dimensionality reduction)" in src, (
        "the pca card title was lost"
    )


def test_the_client_carries_the_new_families_and_pca() -> None:
    """ClusterResultView's new keys + PcaResultView + the pca operation."""
    src = CLIENT.read_text(encoding="utf-8")
    for key in (
        "kmeans_silhouette",
        "bic_table",
        "merge_heights",
        "covariance_type",
        "chosen_by",
        "log_likelihood",
    ):
        assert key in src, f"the client view lost its {key} key"
    assert "export interface PcaResultView" in src, (
        "the pca view was removed"
    )
    assert "async pca(" in src, "the pca client operation was removed"
    assert '"/api/v1/unsupervised/pca"' in src, (
        "the pca route literal was lost"
    )


def test_the_degenerate_answer_is_honest() -> None:
    """A single-cluster answer renders its nulls, never a fabricated score."""
    src = _lane()
    assert 'fmtNumber' in src and '"n/a"' in src, (
        "the honest n/a renderer was lost"
    )
    route = _route()
    assert "_cluster_diagnostics" in route, (
        "the shared honest-degenerate metric builder left the route"
    )


def test_vitest_file_pins_the_acceptance_clauses() -> None:
    """The lane's vitest carries the roadmap's clauses."""
    assert VITEST.exists(), (
        "the ML-4 vitest file is gone — the acceptance clauses lost "
        "their fail-first pins"
    )
    src = VITEST.read_text(encoding="utf-8")
    for clause in (
        "the engine list mirrored",
        "the engine's own scan",
        "the AIC/BIC table",
        "the merge-height line",
        "never a fabricated score",
        "the variance ladder",
    ):
        assert clause in src, f"the vitest lost its clause: {clause}"
    assert src.count("render(<UnsupervisedLearningLane") >= 7, (
        "the vitest no longer mounts the lane on every clause"
    )


def test_roadmap_row_11_status_records_this_round() -> None:
    """The roadmap's ML-4 row is marked Implemented for this round."""
    src = ROADMAP.read_text(encoding="utf-8")
    m = re.search(
        r"\| ID / Feature \| ML-4 — Extend `/api/v1/unsupervised` "
        r"beyond DBSCAN/mean-shift/spectral/IsolationForest \|\n"
        r"\| Current status \| ([^|]+)\|",
        src,
    )
    assert m, "the roadmap's ML-4 row header drifted"
    status = m.group(1)
    assert "Implemented" in status and "382" in status, (
        f"the ML-4 row status must record round 382 honestly, got: {status!r}"
    )
