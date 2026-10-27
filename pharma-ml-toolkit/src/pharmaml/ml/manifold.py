"""Manifold-learning embeddings (PDSH 05.10).

The handbook's manifold chapter -- Isomap, Locally Linear Embedding,
MDS, Spectral Embedding, t-SNE -- is the classic answer to the
question PCA cannot honestly answer: when the interesting structure
lives on a curved, nonlinear sheet inside the feature space, linear
projection flattens it into overlap.  :func:`manifold_embedding`
wraps all five scikit-learn estimators behind one typed call with the
toolkit's standing disciplines:

* one entry point, one record type (:class:`EmbeddingRecord`) -- the
  coordinates land in a DataFrame whose columns are named after the
  method (``ISOMAP1``/``ISOMAP2``...), ready for
  :func:`~pharmaml.viz.ml_plots.plot_manifold_embedding`;
* ``random_state=42`` threaded through every estimator that accepts
  one, so the same table re-embeds to the same picture;
* typed guards instead of raw sklearn errors: the perplexity rule
  (``perplexity < n_samples``), the neighbour rule
  (``n_neighbors < n_samples``), and honest sample-size ceilings for
  the two O(n^2) citizens -- MDS refuses beyond 2,500 rows and t-SNE
  beyond 6,000 with the subsampling advice in the message, because a
  GUI that hangs for an hour is not a feature.

The honest scope, stated once (the chapter's own warning): t-SNE
distances between well-separated clusters are NOT meaningful, and no
embedding here is a preprocessing step for a supervised model -- the
supervised side keeps its own PCA-free pipelines.  These are
*visualisation* embeddings: the shape of the data, coloured by a
target, exactly the digits figures the chapter is famous for.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from pharmaml.ml.manifold import manifold_embedding
>>> rng = np.random.default_rng(42)
>>> swiss = pd.DataFrame({
...     "t": np.linspace(0, 1, 120),
... })
>>> swiss["x"] = np.cos(4 * np.pi * swiss["t"])
>>> swiss["y"] = np.sin(4 * np.pi * swiss["t"])
>>> swiss["z"] = 2 * swiss["t"]
>>> emb = manifold_embedding(swiss, "isomap", n_components=2, n_neighbors=8)
>>> emb.coords.shape
(120, 2)
>>> emb.method
'isomap'
>>> sorted(emb.coords.columns)
['ISOMAP1', 'ISOMAP2']

>>> tiny = pd.DataFrame(rng.normal(size=(5, 3)))
>>> manifold_embedding(tiny, "tsne")
Traceback (most recent call last):
    ...
pharmaml.exceptions.DataValidationError: manifold_embedding: t-SNE needs perplexity strictly below the sample count -- got perplexity 30 with n_samples=5 (lower the perplexity or subsample)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger
from pharmaml.ml.unsupervised import RANDOM_STATE, _as_numeric_matrix

_LOG = get_logger(__name__)

#: The five estimators this module wraps (PDSH 05.10's own roster).
MANIFOLD_METHODS: tuple[str, ...] = ("isomap", "lle", "mds", "spectral", "tsne")

#: LLE flavours (the 'hessian' variant demands n_neighbors >
#: n_features and stays deliberately unlisted).
LLE_METHODS: tuple[str, ...] = ("standard", "modified", "ltsa")

#: MDS refuses beyond this row count (O(n^2) memory + time).
MAX_MDS_SAMPLES = 2500

#: t-SNE refuses beyond this row count (O(n^2) pairwise affinities).
MAX_TSNE_SAMPLES = 6000


@dataclass
class EmbeddingRecord:
    """Typed report of :func:`manifold_embedding`.

    Attributes
    ----------
    method:
        One of :data:`MANIFOLD_METHODS`.
    coords:
        The embedding as a DataFrame (samples x dims), columns named
        ``<METHOD>1``, ``<METHOD>2``...
    n_components:
        Embedding dimensionality (2 by default -- the plot twin).
    n_samples / n_features:
        Input shape.
    params:
        The method's own knobs as passed (``n_neighbors``,
        ``perplexity``, ``lle_method``...), for the record's honest
        replay.
    """

    method: str
    coords: pd.DataFrame
    n_components: int
    n_samples: int
    n_features: int
    params: dict[str, Any]

    def as_frame(self) -> pd.DataFrame:
        """The embedding coordinates."""
        return self.coords

    def __str__(self) -> str:
        knobs = ", ".join(f"{key}={value}" for key, value in self.params.items())
        return (
            f"{self.method} embedding -- {self.n_samples} rows x "
            f"{self.n_features} features -> {self.n_components}-D"
            + (f" ({knobs})" if knobs else "")
        )


def manifold_embedding(
    X: Any,
    method: str,
    *,
    n_components: int = 2,
    n_neighbors: int = 10,
    perplexity: float = 30.0,
    lle_method: str = "standard",
    random_state: int = RANDOM_STATE,
) -> EmbeddingRecord:
    """Project a feature table onto 2-D with a manifold learner (PDSH 05.10).

    Parameters
    ----------
    X:
        Numeric DataFrame / 2-D array (complete rows -- the same
        up-front rule :mod:`pharmaml.ml.unsupervised` states).
    method:
        ``'isomap'`` | ``'lle'`` | ``'mds'`` | ``'spectral'`` |
        ``'tsne'``.
    n_components:
        Output dimensionality (default 2).
    n_neighbors:
        Neighbourhood size for isomap / LLE / spectral embedding.
    perplexity:
        t-SNE's effective-neighbourhood knob.
    lle_method:
        ``'standard'`` | ``'modified'`` | ``'ltsa'``.
    random_state:
        Seed (the toolkit's 42 by default) -- every estimator that
        accepts one gets it, so reruns reproduce the same picture.

    Returns
    -------
    EmbeddingRecord

    Raises
    ------
    DataValidationError
        Unknown method, non-numeric or incomplete input, or a violated
        estimator precondition (perplexity/neighbourhood vs sample
        count, the MDS/t-SNE ceilings).
    """
    key = str(method).lower().replace("-", "_")
    if key not in MANIFOLD_METHODS:
        raise DataValidationError(
            f"manifold_embedding: method must be one of {MANIFOLD_METHODS}, got {method!r}"
        )
    matrix, _names, index = _as_numeric_matrix(X, caller="manifold_embedding")
    n_samples, n_features = matrix.shape
    n_components = int(n_components)
    if not 1 <= n_components <= min(n_samples, n_features):
        raise DataValidationError(
            f"n_components must be in [1, min(n_samples, n_features)] = "
            f"[1, {min(n_samples, n_features)}], got {n_components}"
        )

    params: dict[str, Any] = {}
    if key in {"isomap", "lle", "spectral"}:
        n_neighbors = int(n_neighbors)
        if not 1 <= n_neighbors < n_samples:
            raise DataValidationError(
                f"manifold_embedding: {key} needs 1 <= n_neighbors < n_samples "
                f"({n_samples}), got {n_neighbors}"
            )
        params["n_neighbors"] = n_neighbors
    if key == "lle":
        lle_method = str(lle_method)
        if lle_method not in LLE_METHODS:
            raise DataValidationError(
                f"lle_method must be one of {LLE_METHODS}, got {lle_method!r}"
            )
        params["lle_method"] = lle_method
    if key == "tsne":
        perplexity = float(perplexity)
        if not 0 < perplexity < n_samples:
            raise DataValidationError(
                f"manifold_embedding: t-SNE needs perplexity strictly below the "
                f"sample count -- got perplexity {perplexity:g} with "
                f"n_samples={n_samples} (lower the perplexity or subsample)"
            )
        params["perplexity"] = perplexity
        if n_samples > MAX_TSNE_SAMPLES:
            raise DataValidationError(
                f"manifold_embedding: t-SNE is O(n^2) in samples and refuses "
                f"n={n_samples} > {MAX_TSNE_SAMPLES} -- subsample first (the "
                "handbook's own advice for large sets)"
            )
    if key == "mds" and n_samples > MAX_MDS_SAMPLES:
        raise DataValidationError(
            f"manifold_embedding: MDS is O(n^2) in memory and refuses "
            f"n={n_samples} > {MAX_MDS_SAMPLES} -- subsample first"
        )

    if key == "isomap":
        from sklearn.manifold import Isomap

        fitted = Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(matrix)
    elif key == "lle":
        from sklearn.manifold import LocallyLinearEmbedding

        fitted = LocallyLinearEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors,
            method=lle_method,
            random_state=random_state,
        ).fit_transform(matrix)
    elif key == "mds":
        from sklearn.manifold import MDS

        fitted = MDS(
            n_components=n_components, n_init=4, random_state=random_state
        ).fit_transform(matrix)
    elif key == "spectral":
        from sklearn.manifold import SpectralEmbedding

        fitted = SpectralEmbedding(
            n_components=n_components, n_neighbors=n_neighbors, random_state=random_state
        ).fit_transform(matrix)
    else:  # tsne
        from sklearn.manifold import TSNE

        fitted = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(matrix)

    upper = key.upper()
    columns = [f"{upper}{i + 1}" for i in range(n_components)]
    coords = pd.DataFrame(np.asarray(fitted), index=index, columns=columns)
    _LOG.info("%s embedding fitted (%d rows)", key, n_samples)
    return EmbeddingRecord(
        method=key,
        coords=coords,
        n_components=n_components,
        n_samples=n_samples,
        n_features=n_features,
        params=params,
    )


__all__ = [
    "LLE_METHODS",
    "MANIFOLD_METHODS",
    "MAX_MDS_SAMPLES",
    "MAX_TSNE_SAMPLES",
    "EmbeddingRecord",
    "manifold_embedding",
]
