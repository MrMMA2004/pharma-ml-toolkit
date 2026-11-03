"""Tests for :mod:`pharmaml.ml.umap` -- lmcinnes/umap absorbed offline
(round 186): smooth_knn_dist's binary search, the fuzzy simplicial set
and its union symmetrisation, find_ab_params' curve fit, the
negative-sampling SGD layout, transform's weighted-average init and the
Venna-Kaski trustworthiness (pinned against sklearn's own
implementation).
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.umap import (
    GRADIENT_CLIP,
    find_ab_params,
    fuzzy_simplicial_set,
    pairwise_distances,
    smooth_knn_dist,
    trustworthiness,
    umap_embed,
)


@pytest.fixture()
def circle() -> np.ndarray:
    """A noisy 1-D circle in 2-D -- the manifold to preserve."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2.0 * np.pi, 60, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(
        scale=0.02, size=(60, 2)
    )


@pytest.fixture()
def two_blobs() -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.vstack([rng.normal(size=(25, 4)), rng.normal(size=(25, 4)) + 10.0])


class TestPairwiseDistances:
    def test_euclidean_matches_sklearn(self) -> None:
        from sklearn.metrics import pairwise_distances as sk_pd

        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 5))
        assert np.allclose(pairwise_distances(X, "euclidean"), sk_pd(X))

    def test_cross_mode_shapes(self) -> None:
        rng = np.random.default_rng(1)
        X, Y = rng.normal(size=(4, 3)), rng.normal(size=(6, 3))
        assert pairwise_distances(X, "euclidean", Y=Y).shape == (4, 6)
        assert pairwise_distances(X, "cosine", Y=Y).shape == (4, 6)

    def test_manhattan_is_l1(self) -> None:
        X = np.asarray([[0.0, 0.0], [3.0, 4.0]])
        assert pairwise_distances(X, "manhattan")[0, 1] == pytest.approx(7.0)

    def test_cosine_bounds_and_identical_rows(self) -> None:
        X = np.asarray([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
        matrix = pairwise_distances(X, "cosine")
        assert matrix[0, 0] == pytest.approx(0.0)
        assert matrix[0, 1] == pytest.approx(0.0)  # parallel vectors
        assert matrix[0, 2] == pytest.approx(1.0)  # orthogonal

    def test_unknown_metric_refused(self) -> None:
        with pytest.raises(DataValidationError):
            pairwise_distances(np.eye(3), "minkowski")


class TestSmoothKnnDist:
    def test_sigma_search_hits_the_log2k_target(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(30, 4))
        distances = pairwise_distances(X, "euclidean")
        knn = np.argsort(distances, axis=1)[:, :11]
        knn_dists = np.take_along_axis(distances, knn, axis=1)
        sigmas, rhos = smooth_knn_dist(knn_dists, 10.0)
        assert (sigmas > 0).all()
        assert (rhos >= 0).all()
        # recompute the fuzzy cardinality: each row must sit near log2(10)
        target = np.log2(10.0)
        for i in range(3):
            psum = sum(
                np.exp(-(max(d - rhos[i], 0.0) / sigmas[i]))
                for d in knn_dists[i, 1:]
            )
            assert abs(psum - target) < 1e-3

    def test_rho_is_first_positive_distance(self) -> None:
        distances = np.asarray([[0.0, 0.5, 1.0, 2.0]])
        _sigmas, rhos = smooth_knn_dist(distances, 3.0)
        assert rhos[0] == pytest.approx(0.5)


class TestFuzzySimplicialSet:
    def test_symmetrised_edges_are_unique_and_bounded(self) -> None:
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40, 3))
        rows, cols, weights = fuzzy_simplicial_set(X, 8)
        assert (rows < cols).all()  # upper triangle only
        assert len(set(zip(rows.tolist(), cols.tolist()))) == len(rows)
        assert ((weights > 0.0) & (weights <= 1.0)).all()
        assert rows.max() < 40 and cols.max() < 40

    def test_cosine_metric_produces_a_graph(self) -> None:
        rng = np.random.default_rng(6)
        X = rng.normal(size=(30, 5))
        _rows, _cols, weights = fuzzy_simplicial_set(X, 6, metric="cosine")
        assert weights.size > 0

    def test_bad_n_neighbors_refused(self) -> None:
        with pytest.raises(DataValidationError):
            fuzzy_simplicial_set(np.eye(5), 1)


class TestFindAbParams:
    def test_umap_default_curve_pins(self) -> None:
        a, b = find_ab_params(1.0, 0.1)
        assert a == pytest.approx(1.577, rel=2e-3)
        assert b == pytest.approx(0.895, rel=2e-3)

    def test_curve_passes_one_at_zero_and_decays(self) -> None:
        a, b = find_ab_params(1.0, 0.1)
        curve = lambda x: 1.0 / (1.0 + a * x ** (2 * b))
        assert curve(0.0) == pytest.approx(1.0)
        assert curve(0.1) > 0.9  # the plateau min_dist buys
        assert curve(3.0) < 0.1  # and the tail it pays for

    def test_bad_params_refused(self) -> None:
        with pytest.raises(DataValidationError):
            find_ab_params(0.0, 0.1)
        with pytest.raises(DataValidationError):
            find_ab_params(1.0, -0.5)


class TestUmapEmbed:
    def test_shapes_and_record_fields(self, circle: np.ndarray) -> None:
        record = umap_embed(circle, n_neighbors=8, n_epochs=60, random_state=42)
        assert record.embedding_array.shape == (60, 2)
        assert record.embedding.shape == (60, 2)
        assert list(record.embedding.columns) == ["UMAP1", "UMAP2"]
        assert record.params["n_neighbors"] == 8
        assert record.n_edges > 0
        assert 0.0 < record.mean_weight <= 1.0
        assert record.init == "spectral"
        assert "UMAP" in str(record)

    def test_deterministic_per_seed(self, circle: np.ndarray) -> None:
        first = umap_embed(circle, n_neighbors=8, n_epochs=60, random_state=42)
        second = umap_embed(circle, n_neighbors=8, n_epochs=60, random_state=42)
        assert np.allclose(first.embedding_array, second.embedding_array)

    def test_different_seed_moves(self, circle: np.ndarray) -> None:
        first = umap_embed(circle, n_neighbors=8, n_epochs=60, random_state=42)
        second = umap_embed(circle, n_neighbors=8, n_epochs=60, random_state=7)
        assert not np.allclose(first.embedding_array, second.embedding_array)

    def test_circle_neighbourhoods_survive(self, circle: np.ndarray) -> None:
        record = umap_embed(circle, n_neighbors=8, n_epochs=100, random_state=42)
        score = trustworthiness(circle, record.embedding_array, n_neighbors=8)
        assert score > 0.9

    def test_two_blobs_separate(self, two_blobs: np.ndarray) -> None:
        record = umap_embed(two_blobs, n_neighbors=10, n_epochs=100, random_state=42)
        embedding = record.embedding_array
        gap = np.linalg.norm(
            embedding[:25].mean(axis=0) - embedding[25:].mean(axis=0)
        )
        spread = 0.5 * (
            np.linalg.norm(embedding[:25] - embedding[:25].mean(0), axis=1).mean()
            + np.linalg.norm(embedding[25:] - embedding[25:].mean(0), axis=1).mean()
        )
        assert gap > 2.0 * spread

    def test_random_init_is_used_when_asked(self, circle: np.ndarray) -> None:
        record = umap_embed(
            circle, n_neighbors=8, n_epochs=40, init="random", random_state=42
        )
        assert record.init == "random"

    def test_cosine_metric_runs(self, two_blobs: np.ndarray) -> None:
        record = umap_embed(
            two_blobs, n_neighbors=6, metric="cosine", n_epochs=50, random_state=1
        )
        assert record.embedding_array.shape == (50, 2)

    def test_gradient_clip_constant_is_umaps(self) -> None:
        assert GRADIENT_CLIP == 4.0

    def test_guards(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        with pytest.raises(DataValidationError):
            umap_embed(X, metric="minkowski")
        with pytest.raises(DataValidationError):
            umap_embed(X, init="pca")
        with pytest.raises(DataValidationError):
            umap_embed(X, n_neighbors=1)
        with pytest.raises(DataValidationError):
            umap_embed(X, n_epochs=0)
        with pytest.raises(DataValidationError):
            umap_embed(X, min_dist=1.5)  # above spread
        with pytest.raises(DataValidationError):
            umap_embed(np.asarray([[1.0, 2.0]]))  # one row
        with pytest.raises(DataValidationError):
            umap_embed(np.asarray([[1.0, float("nan")]] * 5))


class TestTransform:
    def test_new_blob_rows_land_near_their_blob(self, two_blobs: np.ndarray) -> None:
        record = umap_embed(two_blobs, n_neighbors=10, n_epochs=100, random_state=42)
        embedding = record.embedding_array
        rng = np.random.default_rng(11)
        fresh = rng.normal(size=(4, 4))  # blob-1-like rows
        placed = record.transform(fresh, n_epochs=50)
        assert placed.shape == (4, 2)
        near_first = np.linalg.norm(placed - embedding[:25].mean(0), axis=1).mean()
        near_second = np.linalg.norm(placed - embedding[25:].mean(0), axis=1).mean()
        assert near_first < near_second

    def test_feature_count_guard(self, two_blobs: np.ndarray) -> None:
        record = umap_embed(two_blobs, n_neighbors=10, n_epochs=40, random_state=42)
        with pytest.raises(DataValidationError):
            record.transform(np.zeros((3, 9)))


class TestTrustworthiness:
    def test_matches_sklearn_exactly(self) -> None:
        from sklearn.manifold import trustworthiness as sk_t

        rng = np.random.default_rng(7)
        X = rng.normal(size=(40, 5))
        embedding = rng.normal(size=(40, 2))
        for k in (1, 3, 5, 10):
            assert trustworthiness(X, embedding, n_neighbors=k) == pytest.approx(
                sk_t(X, embedding, n_neighbors=k)
            )

    def test_identity_embedding_scores_one(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 6))
        assert trustworthiness(X, X, n_neighbors=5) == pytest.approx(1.0)

    def test_guards(self) -> None:
        X = np.zeros((10, 2))
        with pytest.raises(DataValidationError):
            trustworthiness(X, np.zeros((9, 2)), n_neighbors=3)
        with pytest.raises(DataValidationError):
            trustworthiness(X, np.zeros((10, 2)), n_neighbors=6)  # >= n/2
