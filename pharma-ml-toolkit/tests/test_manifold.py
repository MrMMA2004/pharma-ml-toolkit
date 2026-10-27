"""Tests for :mod:`pharmaml.ml.manifold` -- the PDSH handbook's
manifold chapter (05.10) absorbed in 100x-157: Isomap, LLE, MDS,
Spectral Embedding and t-SNE behind one typed call with seeded
determinism and honest estimator-precondition guards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.manifold import (
    LLE_METHODS,
    MANIFOLD_METHODS,
    MAX_MDS_SAMPLES,
    MAX_TSNE_SAMPLES,
    EmbeddingRecord,
    manifold_embedding,
)


@pytest.fixture()
def three_blobs() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x": np.concatenate(
                [rng.normal(0, 0.4, 60), rng.normal(6, 0.4, 60), rng.normal(-6, 0.4, 60)]
            ),
            "y": np.concatenate(
                [rng.normal(0, 0.4, 60), rng.normal(6, 0.4, 60), rng.normal(6, 0.4, 60)]
            ),
            "z": np.concatenate(
                [rng.normal(0, 0.4, 60), rng.normal(-6, 0.4, 60), rng.normal(0, 0.4, 60)]
            ),
        }
    )


class TestManifoldEmbedding:
    @pytest.mark.parametrize("method", MANIFOLD_METHODS)
    def test_every_method_returns_two_d_coords(self, method: str, three_blobs: pd.DataFrame) -> None:
        kwargs: dict = {}
        if method in {"isomap", "lle", "spectral"}:
            kwargs["n_neighbors"] = 8
        if method == "tsne":
            kwargs["perplexity"] = 15
        record = manifold_embedding(three_blobs, method, **kwargs)
        assert isinstance(record, EmbeddingRecord)
        assert record.coords.shape == (180, 2)
        prefix = {"tsne": "TSNE"}.get(method, method.upper())
        assert list(record.coords.columns) == [f"{prefix}1", f"{prefix}2"]
        assert record.method == method
        assert record.n_samples == 180 and record.n_features == 3

    def test_lle_method_variants(self, three_blobs: pd.DataFrame) -> None:
        for variant in LLE_METHODS:
            record = manifold_embedding(
                three_blobs, "lle", n_neighbors=8, lle_method=variant
            )
            assert record.params["lle_method"] == variant

    def test_one_d_output_supported(self, three_blobs: pd.DataFrame) -> None:
        record = manifold_embedding(three_blobs, "isomap", n_components=1, n_neighbors=8)
        assert record.coords.shape == (180, 1)
        assert list(record.coords.columns) == ["ISOMAP1"]

    def test_seeded_methods_are_deterministic(self, three_blobs: pd.DataFrame) -> None:
        for method, kwargs in (
            ("tsne", {"perplexity": 15}),
            ("mds", {}),
            ("spectral", {"n_neighbors": 8}),
        ):
            first = manifold_embedding(three_blobs, method, **kwargs).coords
            second = manifold_embedding(three_blobs, method, **kwargs).coords
            pd.testing.assert_frame_equal(first, second, check_exact=False)

    def test_params_recorded_for_replay(self, three_blobs: pd.DataFrame) -> None:
        record = manifold_embedding(three_blobs, "isomap", n_neighbors=12)
        assert record.params == {"n_neighbors": 12}
        tsne = manifold_embedding(three_blobs, "tsne", perplexity=25)
        assert tsne.params == {"perplexity": 25}

    def test_summary_and_as_frame(self, three_blobs: pd.DataFrame) -> None:
        record = manifold_embedding(three_blobs, "mds")
        assert "mds embedding" in str(record)
        assert "n_neighbors" not in str(record)  # mds takes no neighbourhood knob
        assert record.as_frame() is record.coords

    def test_labels_index_preserved(self) -> None:
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(50, 3)), index=[f"c{i}" for i in range(50)])
        record = manifold_embedding(frame, "mds")
        assert list(record.coords.index[:3]) == ["c0", "c1", "c2"]

    def test_unknown_method_refused(self, three_blobs: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="method must be one of"):
            manifold_embedding(three_blobs, "umap")

    def test_tsne_perplexity_guard(self, three_blobs: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="perplexity strictly below"):
            manifold_embedding(three_blobs, "tsne", perplexity=500)

    def test_neighbours_guard(self) -> None:
        rng = np.random.default_rng(42)
        tiny = pd.DataFrame(rng.normal(size=(6, 3)))
        with pytest.raises(DataValidationError, match="n_neighbors < n_samples"):
            manifold_embedding(tiny, "isomap", n_neighbors=10)

    def test_bad_lle_method_refused(self, three_blobs: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="lle_method"):
            manifold_embedding(three_blobs, "lle", n_neighbors=8, lle_method="hessian")

    def test_tsne_sample_ceiling(self) -> None:
        rng = np.random.default_rng(42)
        big = pd.DataFrame(rng.normal(size=(MAX_TSNE_SAMPLES + 1, 3)))
        with pytest.raises(DataValidationError, match="t-SNE is O"):
            manifold_embedding(big, "tsne", perplexity=30)

    def test_mds_sample_ceiling(self) -> None:
        rng = np.random.default_rng(42)
        big = pd.DataFrame(rng.normal(size=(MAX_MDS_SAMPLES + 1, 3)))
        with pytest.raises(DataValidationError, match="MDS is O"):
            manifold_embedding(big, "mds")

    def test_nan_refused_typed(self, three_blobs: pd.DataFrame) -> None:
        dirty = three_blobs.copy()
        dirty.iloc[0, 0] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            manifold_embedding(dirty, "isomap", n_neighbors=8)

    def test_n_components_out_of_range(self, three_blobs: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError, match="n_components must be in"):
            manifold_embedding(three_blobs, "mds", n_components=4)

    def test_exported_from_ml_package(self) -> None:
        from pharmaml import ml

        for name in ("manifold_embedding", "EmbeddingRecord", "MANIFOLD_METHODS"):
            assert name in ml.__all__, name
