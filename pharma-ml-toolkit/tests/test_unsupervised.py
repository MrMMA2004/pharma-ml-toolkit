"""Tests for :mod:`pharmaml.ml.unsupervised` -- the PDSH handbook's
unsupervised chapters (05.09 PCA, 05.11 k-means, 05.12 GMM, 05.13 KDE)
absorbed in 100x-157.

Every record contract is pinned: the explained-variance ladder sums,
the auto-k scan reports its sweep, the BIC selection recovers a known
component count, the CV bandwidth is positive and finite, and every
guard refuses its bad input with the typed error naming the problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.unsupervised import (
    GMM_COVARIANCE_TYPES,
    KDE_KERNELS,
    GmmResult,
    KdeResult,
    KMeansResult,
    PcaResult,
    gmm_cluster,
    kde_fit,
    kmeans_cluster,
    pca_reduce,
)


@pytest.fixture()
def blobs() -> pd.DataFrame:
    """Three well-separated 2-D blobs (the scan tests' ground truth)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x": np.concatenate(
                [rng.normal(0, 0.3, 100), rng.normal(8, 0.3, 100), rng.normal(-8, 0.3, 100)]
            ),
            "y": np.concatenate(
                [rng.normal(0, 0.3, 100), rng.normal(8, 0.3, 100), rng.normal(8, 0.3, 100)]
            ),
        }
    )


# --------------------------------------------------------------------- #
# pca_reduce
# --------------------------------------------------------------------- #
class TestPcaReduce:
    def test_ratio_ladder_sums_to_one(self) -> None:
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
        result = pca_reduce(frame)
        assert isinstance(result, PcaResult)
        assert result.n_components == 5  # min(10, 60, 5)
        assert pytest.approx(sum(result.explained_variance_ratio), abs=1e-9) == 1.0
        assert result.cumulative_variance[-1] == pytest.approx(1.0, abs=1e-9)
        assert result.cumulative_variance == sorted(result.cumulative_variance)

    def test_dominant_axis_is_first(self) -> None:
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(80, 3)), columns=list("abc"))
        frame["d"] = 3.0 * frame["a"] - 2.0 * frame["b"]
        # raw (unscaled) budget: var(d) = 9 + 4 = 13 of 16 total -> PC1 ~ 81 %
        raw = pca_reduce(frame, n_components=2, scale=False)
        assert raw.explained_variance_ratio[0] > 0.75
        assert raw.transformed.shape == (80, 2)
        assert list(raw.transformed.columns) == ["PC1", "PC2"]
        # standardised budget: every column unit-variance, but d still
        # carries the a/b structure -> PC1 clearly above the 25 % floor
        scaled = pca_reduce(frame, n_components=2, scale=True)
        assert 0.4 < scaled.explained_variance_ratio[0] < 0.6

    def test_loadings_name_the_features(self) -> None:
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=["mw", "logp", "tpsa"])
        result = pca_reduce(frame)
        assert list(result.loadings.index) == ["mw", "logp", "tpsa"]
        assert result.loadings.shape == (3, 3)
        assert result.column_names == ["mw", "logp", "tpsa"]

    def test_deterministic_rerun(self) -> None:
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.normal(size=(50, 4)))
        first = pca_reduce(frame, n_components=3)
        second = pca_reduce(frame, n_components=3)
        pd.testing.assert_frame_equal(first.transformed, second.transformed)

    def test_scaling_changes_the_answer(self) -> None:
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(60, 2)), columns=["small", "big"])
        frame["big"] *= 50.0
        scaled = pca_reduce(frame, n_components=1, scale=True)
        raw = pca_reduce(frame, n_components=1, scale=False)
        assert scaled.scaled and not raw.scaled
        assert not np.allclose(scaled.transformed.to_numpy(), raw.transformed.to_numpy())

    def test_records_present_their_own_summary(self) -> None:
        rng = np.random.default_rng(42)
        result = pca_reduce(pd.DataFrame(rng.normal(size=(30, 2))), n_components=2)
        text = str(result)
        assert "PCA(2 components" in text
        assert "PC1 " in text and "cumulative" in text
        assert result.as_frame() is result.transformed

    def test_series_input_becomes_one_column(self) -> None:
        rng = np.random.default_rng(42)
        series = pd.Series(rng.normal(size=40), name="sol")
        result = pca_reduce(series, n_components=1)
        assert result.n_features == 1
        assert result.column_names == ["sol"]

    def test_plain_array_input(self) -> None:
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(30, 4))
        result = pca_reduce(matrix)
        assert result.transformed.shape == (30, 4)
        assert result.column_names == ["0", "1", "2", "3"]

    def test_n_components_out_of_range(self) -> None:
        frame = pd.DataFrame(np.ones((10, 3)))
        with pytest.raises(DataValidationError, match="n_components must be in"):
            pca_reduce(frame, n_components=4)

    def test_nan_refused_typed(self) -> None:
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
        with pytest.raises(DataValidationError, match="NaN"):
            pca_reduce(frame)

    def test_non_numeric_column_refused(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0], "smiles": ["CCO", "CCC"]})
        with pytest.raises(DataValidationError, match="'smiles' is not numeric"):
            pca_reduce(frame)


# --------------------------------------------------------------------- #
# kmeans_cluster
# --------------------------------------------------------------------- #
class TestKMeansCluster:
    def test_auto_k_recovers_three_blobs(self, blobs: pd.DataFrame) -> None:
        result = kmeans_cluster(blobs)
        assert isinstance(result, KMeansResult)
        assert result.k == 3
        assert result.silhouette is not None and result.silhouette > 0.6
        assert not result.scan.empty
        assert {"k", "inertia", "silhouette"} <= set(result.scan.columns)

    def test_manual_k(self, blobs: pd.DataFrame) -> None:
        result = kmeans_cluster(blobs, k=3)
        assert result.k == 3
        assert result.scan.empty  # no sweep for a pinned k
        assert result.labels.nunique() == 3
        assert result.centers.shape == (3, 2)
        assert int(result.sizes.sum()) == len(blobs)

    def test_labels_carry_the_frame_index(self, blobs: pd.DataFrame) -> None:
        renamed = blobs.set_index(pd.Index([f"row{i}" for i in range(len(blobs))]))
        result = kmeans_cluster(renamed, k=3)
        assert isinstance(result.labels, pd.Series)
        assert list(result.labels.index[:2]) == ["row0", "row1"]

    def test_deterministic_seed(self, blobs: pd.DataFrame) -> None:
        first = kmeans_cluster(blobs, k=3)
        second = kmeans_cluster(blobs, k=3)
        assert list(first.labels) == list(second.labels)
        assert first.inertia == pytest.approx(second.inertia)

    def test_mini_batch_variant(self, blobs: pd.DataFrame) -> None:
        result = kmeans_cluster(blobs, k=3, mini_batch=True)
        assert result.method == "mini_batch"
        assert result.labels.nunique() == 3

    def test_as_frame_joins_centroids(self, blobs: pd.DataFrame) -> None:
        result = kmeans_cluster(blobs, k=3)
        frame = result.as_frame()
        assert "cluster" in frame.columns
        assert "center_x" in frame.columns and "center_y" in frame.columns
        assert len(frame) == len(blobs)

    def test_summary_text_names_the_choice(self, blobs: pd.DataFrame) -> None:
        auto = kmeans_cluster(blobs)
        assert "best silhouette" in str(auto)
        manual = kmeans_cluster(blobs, k=2)
        assert "k=2 (manual)" in str(manual)

    def test_k_above_rows_refused(self) -> None:
        frame = pd.DataFrame(np.ones((5, 2)))
        with pytest.raises(DataValidationError, match="k must be in"):
            kmeans_cluster(frame, k=9)

    def test_too_few_rows_refused(self) -> None:
        with pytest.raises(DataValidationError, match="at least 4 rows"):
            kmeans_cluster(pd.DataFrame(np.ones((3, 2))), k=2)

    def test_inertia_decreases_across_the_scan(self, blobs: pd.DataFrame) -> None:
        result = kmeans_cluster(blobs)
        inertias = result.scan["inertia"].tolist()
        assert inertias == sorted(inertias, reverse=True)


# --------------------------------------------------------------------- #
# gmm_cluster
# --------------------------------------------------------------------- #
class TestGmmCluster:
    def test_bic_recovers_three_blobs(self, blobs: pd.DataFrame) -> None:
        result = gmm_cluster(blobs)
        assert isinstance(result, GmmResult)
        assert result.n_components == 3
        assert result.chosen_by == "bic"
        assert not result.scan.empty
        assert {"n_components", "aic", "bic"} <= set(result.scan.columns)

    def test_aic_criterion_is_selectable(self, blobs: pd.DataFrame) -> None:
        result = gmm_cluster(blobs, criterion="aic")
        assert result.chosen_by == "aic"
        bic_pick = gmm_cluster(blobs)
        assert result.scan.equals(bic_pick.scan)  # same sweep, different vote

    def test_manual_components(self, blobs: pd.DataFrame) -> None:
        result = gmm_cluster(blobs, n_components=3)
        assert result.n_components == 3
        assert result.chosen_by == "manual"
        assert result.scan.empty

    def test_responsibilities_sum_to_one(self, blobs: pd.DataFrame) -> None:
        result = gmm_cluster(blobs, n_components=3)
        totals = result.probabilities.sum(axis=1)
        assert np.allclose(totals.to_numpy(), 1.0, atol=1e-9)
        assert result.probabilities.shape == (len(blobs), 3)

    def test_covariance_types_all_run(self, blobs: pd.DataFrame) -> None:
        for covariance in GMM_COVARIANCE_TYPES:
            result = gmm_cluster(blobs, n_components=3, covariance_type=covariance)
            assert result.covariance_type == covariance
            assert result.converged

    def test_as_frame_carries_responsibilities(self, blobs: pd.DataFrame) -> None:
        result = gmm_cluster(blobs, n_components=2)
        frame = result.as_frame()
        assert "cluster" in frame.columns
        assert "comp_1" in frame.columns

    def test_summary_text(self, blobs: pd.DataFrame) -> None:
        text = str(gmm_cluster(blobs, n_components=2))
        assert "GaussianMixture(2 components" in text
        assert "converged" in text

    def test_bad_covariance_refused(self, blobs: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="covariance_type"):
            gmm_cluster(blobs, covariance_type="hexagonal")

    def test_bad_criterion_refused(self, blobs: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="criterion"):
            gmm_cluster(blobs, criterion="mdl")

    def test_components_above_rows_refused(self) -> None:
        with pytest.raises(DataValidationError, match="n_components must be in"):
            gmm_cluster(pd.DataFrame(np.ones((4, 2))), n_components=9)


# --------------------------------------------------------------------- #
# kde_fit
# --------------------------------------------------------------------- #
class TestKdeFit:
    def test_cv_bandwidth_positive_and_density_smooth(self) -> None:
        rng = np.random.default_rng(42)
        series = pd.Series(rng.normal(size=120), name="pic50")
        result = kde_fit(series)
        assert isinstance(result, KdeResult)
        assert result.bandwidth_source == "cv"
        assert result.bandwidth > 0
        assert result.column == "pic50"
        assert len(result.grid) == len(result.density) == 200
        assert np.isfinite(result.density).all()
        assert (result.density >= 0).all()

    def test_manual_bandwidth(self) -> None:
        rng = np.random.default_rng(42)
        result = kde_fit(rng.normal(size=60), bandwidth=0.25)
        assert result.bandwidth_source == "manual"
        assert result.bandwidth == pytest.approx(0.25)

    def test_small_sample_uses_silverman(self) -> None:
        rng = np.random.default_rng(42)
        result = kde_fit(pd.Series(rng.normal(size=8)))
        assert result.bandwidth_source == "silverman"
        assert result.bandwidth > 0

    def test_kernels_all_run(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(size=40)
        for kernel in KDE_KERNELS:
            result = kde_fit(values, kernel=kernel, bandwidth=0.4)
            assert result.kernel == kernel

    def test_grid_brackets_the_data(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(size=50)
        result = kde_fit(values)
        assert result.grid[0] < values.min()
        assert result.grid[-1] > values.max()

    def test_as_frame_and_summary(self) -> None:
        rng = np.random.default_rng(42)
        result = kde_fit(pd.Series(rng.normal(size=40), name="logp"))
        frame = result.as_frame()
        assert list(frame.columns) == ["x", "density"]
        text = str(result)
        assert "KDE('logp'" in text and "bandwidth" in text

    def test_two_columns_refused(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
        with pytest.raises(DataValidationError, match="ONE column"):
            kde_fit(frame)

    def test_constant_input_refused(self) -> None:
        with pytest.raises(DataValidationError, match="constant input"):
            kde_fit(pd.Series([5.0, 5.0, 5.0, 5.0]))

    def test_bad_kernel_refused(self) -> None:
        with pytest.raises(DataValidationError, match="kernel"):
            kde_fit(pd.Series([1.0, 2.0, 3.0]), kernel="triangular")

    def test_too_few_points_refused(self) -> None:
        with pytest.raises(DataValidationError, match="at least 3"):
            kde_fit(pd.Series([1.0, 2.0]))


# --------------------------------------------------------------------- #
# package surface
# --------------------------------------------------------------------- #
class TestPackageSurface:
    def test_exported_from_ml_package(self) -> None:
        from pharmaml import ml

        for name in (
            "pca_reduce",
            "kmeans_cluster",
            "gmm_cluster",
            "kde_fit",
            "PcaResult",
            "KMeansResult",
            "GmmResult",
            "KdeResult",
        ):
            assert name in ml.__all__, name
            assert getattr(ml, name) is not None
