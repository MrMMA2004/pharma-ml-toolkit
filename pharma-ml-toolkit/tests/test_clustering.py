"""Tests for :mod:`pharmaml.ml.clustering` -- the part18 §4.3 lane.

The contract pins (part18's own required design):

* DBSCAN: epsilon and min-samples controls, noise-point reporting,
  the honest all-noise / single-cluster degenerate answers.
* Mean Shift: bandwidth estimation when None, mode count over blobs.
* Spectral: affinity choices (rbf / nearest_neighbors), the
  embedding-vs-clustering distinction, parameter validation.
* The shared diagnostics: silhouette, Davies-Bouldin, Calinski-
  Harabasz, cluster sizes, noise ratio -- computed over non-noise
  samples only, ``None`` when honestly not computable.
* Reproducibility: the seed pin returns identical labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.clustering import (
    ClusterDiagnostics,
    DbscanResult,
    MeanShiftClusterResult,
    SpectralResult,
    dbscan_cluster,
    meanshift_cluster,
    spectral_cluster,
)


@pytest.fixture()
def moons() -> pd.DataFrame:
    """Two well-separated blobs (density clustering's ground truth)."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x": np.concatenate([rng.normal(0, 0.3, 80), rng.normal(6, 0.3, 80)]),
            "y": np.concatenate([rng.normal(0, 0.3, 80), rng.normal(6, 0.3, 80)]),
        }
    )


@pytest.fixture()
def noisy(moons: pd.DataFrame) -> pd.DataFrame:
    """The blobs plus three far-scattered noise rows."""
    rng = np.random.default_rng(7)
    far = pd.DataFrame(
        {
            "x": rng.uniform(25, 30, 3),
            "y": rng.uniform(25, 30, 3),
        }
    )
    return pd.concat([moons, far], ignore_index=True)


# --------------------------------------------------------------------- #
# dbscan_cluster
# --------------------------------------------------------------------- #
class TestDbscan:
    def test_two_blobs_two_clusters(self, moons: pd.DataFrame) -> None:
        result = dbscan_cluster(moons, eps=1.0, min_samples=4)
        assert isinstance(result, DbscanResult)
        assert result.n_clusters == 2
        assert result.noise_ratio <= 0.02
        assert result.eps_source == "manual"
        assert result.n_samples == 160

    def test_noise_points_reported_first_class(self, noisy: pd.DataFrame) -> None:
        result = dbscan_cluster(noisy, eps=1.0, min_samples=4)
        # the three planted far points must be noise (label -1)
        labels = np.asarray(result.labels)
        assert (labels[-3:] == -1).all()
        assert result.noise_count >= 3
        assert result.diagnostics.noise_count == result.noise_count
        # the diagnostics' noise ratio is over all rows
        assert result.diagnostics.noise_ratio == pytest.approx(result.noise_count / 163)

    def test_all_noise_degenerate_is_honest(self, noisy: pd.DataFrame) -> None:
        result = dbscan_cluster(noisy, eps=0.001, min_samples=5)
        assert result.n_clusters == 0
        assert result.diagnostics.silhouette is None
        assert result.diagnostics.davies_bouldin is None
        assert result.diagnostics.calinski_harabasz is None
        assert result.diagnostics.noise_ratio == 1.0
        # the sizes table degrades to the single noise row
        assert list(result.diagnostics.sizes.index) == ["noise"]
        assert int(result.diagnostics.sizes["noise"]) == 163

    def test_eps_estimation_is_labelled(self, moons: pd.DataFrame) -> None:
        result = dbscan_cluster(moons, min_samples=5)
        assert result.eps_source == "k-distance"
        assert result.eps > 0

    def test_reproducible_same_seed(self, noisy: pd.DataFrame) -> None:
        first = np.asarray(dbscan_cluster(noisy, eps=1.0, min_samples=4).labels)
        second = np.asarray(dbscan_cluster(noisy, eps=1.0, min_samples=4).labels)
        np.testing.assert_array_equal(first, second)

    def test_as_frame_carries_noise_flag(self, noisy: pd.DataFrame) -> None:
        frame = dbscan_cluster(noisy, eps=1.0, min_samples=4).as_frame()
        assert set(frame.columns) == {"cluster", "is_noise"}
        assert bool(frame["is_noise"].iloc[-1])

    @pytest.mark.parametrize(
        "bad",
        [
            {"eps": -1.0},
            {"eps": 0.0},
        ],
    )
    def test_bad_eps_refused(self, moons: pd.DataFrame, bad: dict) -> None:
        with pytest.raises(DataValidationError):
            dbscan_cluster(moons, **bad)

    def test_bad_min_samples_refused(self, moons: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError):
            dbscan_cluster(moons, eps=1.0, min_samples=0)

    def test_too_few_rows_refused(self) -> None:
        tiny = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(DataValidationError):
            dbscan_cluster(tiny, eps=0.5)


# --------------------------------------------------------------------- #
# meanshift_cluster
# --------------------------------------------------------------------- #
class TestMeanShift:
    def test_blobs_modes(self, moons: pd.DataFrame) -> None:
        result = meanshift_cluster(moons, bandwidth=1.5)
        assert isinstance(result, MeanShiftClusterResult)
        assert result.n_clusters >= 2
        assert result.bandwidth_source == "manual"
        assert result.centers.shape[1] == 2

    def test_bandwidth_estimation_is_labelled(self, moons: pd.DataFrame) -> None:
        result = meanshift_cluster(moons)
        assert result.bandwidth_source == "estimated"
        assert result.bandwidth > 0

    def test_reproducible(self, moons: pd.DataFrame) -> None:
        first = np.asarray(meanshift_cluster(moons, bandwidth=1.5).labels)
        second = np.asarray(meanshift_cluster(moons, bandwidth=1.5).labels)
        np.testing.assert_array_equal(first, second)

    def test_as_frame_maps_modes(self, moons: pd.DataFrame) -> None:
        frame = meanshift_cluster(moons, bandwidth=1.5).as_frame()
        assert "mode_x" in frame.columns and "mode_y" in frame.columns

    def test_bad_bandwidth_refused(self, moons: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError):
            meanshift_cluster(moons, bandwidth=0.0)

    def test_too_few_rows_refused(self) -> None:
        tiny = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(DataValidationError):
            meanshift_cluster(tiny)


# --------------------------------------------------------------------- #
# spectral_cluster
# --------------------------------------------------------------------- #
class TestSpectral:
    def test_blobs_rbf(self, moons: pd.DataFrame) -> None:
        result = spectral_cluster(moons, n_clusters=2, affinity="rbf", gamma=0.5)
        assert isinstance(result, SpectralResult)
        assert result.n_clusters == 2
        assert result.gamma == 0.5
        assert result.n_neighbors is None

    def test_blobs_nearest_neighbors(self, moons: pd.DataFrame) -> None:
        result = spectral_cluster(
            moons, n_clusters=2, affinity="nearest_neighbors", n_neighbors=10
        )
        assert result.n_clusters == 2
        assert result.n_neighbors == 10
        assert result.gamma is None
        # well-separated blobs -> both affinities agree on the partition
        rbf = spectral_cluster(moons, n_clusters=2, affinity="rbf", gamma=0.5)
        agree = float(
            np.mean(np.asarray(result.labels) == np.asarray(rbf.labels))
        )
        assert agree > 0.95

    def test_reproducible(self, moons: pd.DataFrame) -> None:
        first = np.asarray(spectral_cluster(moons, n_clusters=2).labels)
        second = np.asarray(spectral_cluster(moons, n_clusters=2).labels)
        np.testing.assert_array_equal(first, second)

    def test_bad_affinity_refused(self, moons: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError):
            spectral_cluster(moons, n_clusters=2, affinity="cosine")

    def test_bad_cluster_count_refused(self, moons: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError):
            spectral_cluster(moons, n_clusters=1)

    def test_bad_gamma_refused(self, moons: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError):
            spectral_cluster(moons, n_clusters=2, gamma=0.0)

    def test_neighbors_bounded_by_rows(self, moons: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError):
            spectral_cluster(
                moons, n_clusters=2, affinity="nearest_neighbors", n_neighbors=160
            )


# --------------------------------------------------------------------- #
# the shared diagnostics block
# --------------------------------------------------------------------- #
class TestDiagnostics:
    def test_blobs_silhouette_high(self, moons: pd.DataFrame) -> None:
        result = dbscan_cluster(moons, eps=1.0, min_samples=4)
        assert result.diagnostics.silhouette is not None
        assert result.diagnostics.silhouette > 0.5

    def test_davies_bouldin_low_for_blobs(self, moons: pd.DataFrame) -> None:
        result = dbscan_cluster(moons, eps=1.0, min_samples=4)
        assert result.diagnostics.davies_bouldin is not None
        assert result.diagnostics.davies_bouldin < 0.8

    def test_calinski_harabasz_reported(self, moons: pd.DataFrame) -> None:
        result = dbscan_cluster(moons, eps=1.0, min_samples=4)
        assert result.diagnostics.calinski_harabasz is not None
        assert result.diagnostics.calinski_harabasz > 100

    def test_single_cluster_metrics_none(self, moons: pd.DataFrame) -> None:
        # eps huge -> everything is one cluster
        result = dbscan_cluster(moons, eps=50.0, min_samples=2)
        assert result.n_clusters == 1
        assert result.diagnostics.silhouette is None
        assert result.diagnostics.davies_bouldin is None

    def test_sizes_series_shape(self, noisy: pd.DataFrame) -> None:
        result = dbscan_cluster(noisy, eps=1.0, min_samples=4)
        diag: ClusterDiagnostics = result.diagnostics
        total = int(diag.sizes.sum())
        assert total == 163
        assert "noise" in list(diag.sizes.index)

    def test_str_mentions_every_metric(self, moons: pd.DataFrame) -> None:
        text = str(dbscan_cluster(moons, eps=1.0, min_samples=4))
        for token in ("silhouette", "Davies-Bouldin", "Calinski-Harabasz", "noise"):
            assert token in text
