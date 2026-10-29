"""Approximate nearest-neighbour search in plain NumPy: the LSH lesson.

The best-of-ml-python *Vector Similarity Search (ANN)* category (13
projects: Faiss, Milvus, Annoy, hnswlib, NMSLIB ...) was reviewed entry
by entry for this module.  Every industrial ANN engine ships the same
bargain -- trade exactness for speed by looking at *few* candidates --
and the smallest engine that teaches the bargain is Charikar's
random-hyperplane LSH (the scheme Annoy's author cites as the
ancestry): project every vector on ``n_planes`` random hyperplanes,
keep the sign bits as a key, and let near vectors collide in buckets.

What lands here is that engine, honest and complete:

* :class:`LSHIndex` -- the random-hyperplane table: seeded Gaussian
  planes, integer keys from the sign bits, bucket dictionary, and
  queries with **multiprobe** (the Lv et al. trick: also visit buckets
  whose keys sit at Hamming distance 1..p−1 from the query key -- the
  recall knob every engine exposes).
* :func:`brute_force_knn` -- the exact top-k (argpartition), the
  baseline every ANN benchmark reports against.
* :func:`ann_benchmark` -- the bargain measured: recall@k (what
  fraction of the true top-k the LSH query found), the candidate-set
  compression (what fraction of the collection the index actually
  touched) and how many queries came back empty -- the failure mode
  more planes buys back, at the price of the compression.

The pharma reading, stated once: the vectors are descriptor rows or
fingerprint bit-vectors of a compound library; the query is one
promising hit; the lesson is how cheaply "find me its neighbours"
can be answered -- and what that cheapness costs.

The honest scope: Faiss-class engines ship GPU kernels, product
quantisation and HNSW graphs -- none of which changes *what* the
bargain is, and all of which need the C++/CUDA tiers the toolkit
deliberately does not vendor.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.annsearch import (
...     LSHIndex, ann_benchmark, brute_force_knn,
... )
>>> rng = np.random.default_rng(42)
>>> vectors = np.vstack([
...     rng.normal(size=(120, 8)) + 2.0,
...     rng.normal(size=(120, 8)) - 2.0,
...     rng.normal(size=(120, 8)) * 0.5,
... ])
>>> neighbors = brute_force_knn(vectors, vectors[5], k=4)
>>> neighbors[0].item == 5                # the query finds itself first
True
>>> index = LSHCollection.build(vectors, n_planes=6, seed=42)
>>> result = index.query(vectors[5], k=4)
>>> len(result.neighbors) <= 4
True
>>> report = ann_benchmark(vectors, n_planes=6, probes=2, k=5, seed=42)
>>> 0.3 < report.recall_at_k <= 1.0
True
>>> 0.0 < report.candidate_ratio < 1.0    # that is the whole bargain
True
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through the planes and the benchmark queries
#: (the pin).
RANDOM_STATE = 42


def _check_vectors(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=float)
    if array.ndim != 2:
        raise DataValidationError(f"vectors must be 2-D (samples x dims), got {array.ndim}-D")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise DataValidationError(f"vectors must hold >= 2 samples, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise DataValidationError("vectors must all be finite (no NaN/inf)")
    return array


def _check_k(k: int) -> int:
    if k < 1:
        raise DataValidationError(f"k must be >= 1, got {k}")
    return k


@dataclass(frozen=True)
class Neighbor:
    """One retrieved neighbour: row index and its Euclidean distance."""

    item: int
    distance: float


def brute_force_knn(vectors: np.ndarray, query: np.ndarray, k: int = 10) -> list[Neighbor]:
    """The exact top-k by Euclidean distance -- the benchmark baseline.

    Raises
    ------
    DataValidationError
        On malformed vectors/query or non-positive k.
    """
    array = _check_vectors(vectors)
    _check_k(k)
    point = np.asarray(query, dtype=float).reshape(-1)
    if point.size != array.shape[1]:
        raise DataValidationError(
            f"query has {point.size} dims, the collection has {array.shape[1]}"
        )
    if not np.all(np.isfinite(point)):
        raise DataValidationError("query must be finite (no NaN/inf)")
    distances = np.linalg.norm(array - point, axis=1)
    top = np.argsort(distances, kind="stable")[: min(k, array.shape[0])]
    return [Neighbor(int(i), float(distances[i])) for i in top]


class LSHIndex:
    """The random-hyperplane LSH table (Charikar's SimHash engine).

    ``n_planes`` seeded Gaussian hyperplanes cut the space into
    ``2**n_planes`` cells; each vector's sign bits form its integer
    key and the buckets hold the row indices that share a key.  A
    query hashes itself, collects the candidates from its own bucket
    (plus Hamming-adjacent ones under ``probes``), and ranks *only
    those* by exact distance -- the engine's whole contract: wrong
    answers are possible, cheap answers are guaranteed.

    Attributes
    ----------
    planes:
        The ``(n_planes, dims)`` Gaussian normals (seeded).
    buckets:
        key -> list of row indices.
    n_planes:
        The number of hyperplanes (the precision/recall knob: more
        planes, finer cells, emptier buckets).
    """

    def __init__(self, planes: np.ndarray, buckets: dict[int, list[int]]) -> None:
        self.planes = planes
        self.buckets = buckets
        self.n_planes = int(planes.shape[0])

    @classmethod
    def build(cls, vectors: np.ndarray, n_planes: int = 8, seed: int = RANDOM_STATE) -> LSHIndex:
        """Hash the collection under ``n_planes`` seeded hyperplanes.

        Raises
        ------
        DataValidationError
            On malformed vectors or a non-positive plane count.
        """
        array = _check_vectors(vectors)
        if n_planes < 1 or n_planes > 31:
            raise DataValidationError(
                f"n_planes must be in [1, 31] (31 bits keeps keys int64-safe), got {n_planes}"
            )
        rng = np.random.default_rng(seed)
        planes = rng.normal(size=(n_planes, array.shape[1]))
        keys = cls._keys(array, planes)
        buckets: dict[int, list[int]] = defaultdict(list)
        for row, key in enumerate(keys):
            buckets[int(key)].append(int(row))
        _LOG.debug(
            "LSH index built: %d vectors, %d planes, %d non-empty buckets",
            array.shape[0],
            n_planes,
            len(buckets),
        )
        return cls(planes=planes, buckets=dict(buckets))

    @staticmethod
    def _keys(vectors: np.ndarray, planes: np.ndarray) -> np.ndarray:
        """Sign-bit integer keys: ``sum(bit_i << i)`` over the planes."""
        bits = (vectors @ planes.T) > 0.0
        weights = 1 << np.arange(planes.shape[0], dtype=np.int64)
        return bits.astype(np.int64) @ weights

    def _probe_keys(self, key: int, probes: int) -> list[int]:
        """The query key plus its Hamming-1 neighbours (``probes`` == 2).

        ``probes`` > 2 is refused: the Hamming ball grows
        combinatorially and the engine's lesson does not -- the two
        settings every demo needs are "own bucket" and "own bucket
        plus its flip neighbours".
        """
        if probes not in (1, 2):
            raise DataValidationError(f"probes must be 1 or 2, got {probes}")
        if probes == 1:
            return [key]
        return [key] + [key ^ (1 << flip) for flip in range(self.n_planes)]

    def query_plan(
        self, vector: np.ndarray, probes: int = 1
    ) -> tuple[int, tuple[int, ...], list[int]]:
        """The hash layer alone: (key, probed keys, candidate rows).

        This is the engine's bucket half without the re-ranking half --
        useful for auditing *who gets looked at* (the approximation's
        actual location) independent of the collection.
        """
        point = np.asarray(vector, dtype=float).reshape(-1)
        if point.size != self.planes.shape[1]:
            raise DataValidationError(
                f"query has {point.size} dims, the index was built on {self.planes.shape[1]}"
            )
        if not np.all(np.isfinite(point)):
            raise DataValidationError("query must be finite (no NaN/inf)")
        key = int(self._keys(point.reshape(1, -1), self.planes)[0])
        probe_keys = self._probe_keys(key, probes)
        candidates: list[int] = []
        for probe_key in probe_keys:
            candidates.extend(self.buckets.get(probe_key, ()))
        return key, tuple(probe_keys), candidates


@dataclass(frozen=True)
class LSHQueryResult:
    """One approximate query: the key, the candidates, the neighbours."""

    key: int
    probes: tuple[int, ...]
    n_candidates: int
    neighbors: tuple[Neighbor, ...]


class LSHCollection:
    """The usable engine: an :class:`LSHIndex` plus the stored vectors.

    :class:`LSHIndex` is the pure hash layer -- planes, sign-bit keys,
    buckets (and :meth:`LSHIndex.query_plan` audits who would get
    looked at); this class adds the stored collection so queries can
    re-rank their candidates by *exact* distance.  The approximation
    is entirely in who gets looked at, never in how they are ranked:
    Annoy's on-disk trees and Faiss' flat stores make the same split.
    """

    def __init__(self, vectors: np.ndarray, index: LSHIndex) -> None:
        self.vectors = vectors
        self.index = index

    @classmethod
    def build(
        cls, vectors: np.ndarray, n_planes: int = 8, seed: int = RANDOM_STATE
    ) -> LSHCollection:
        """Build the index over a stored copy of the collection."""
        array = _check_vectors(vectors)
        return cls(array, LSHIndex.build(array, n_planes=n_planes, seed=seed))

    def query(self, vector: np.ndarray, k: int = 10, probes: int = 1) -> LSHQueryResult:
        """Approximate top-k over the stored collection (exact ranking).

        Raises
        ------
        DataValidationError
            On a malformed vector, non-positive k, or probes outside
            {1, 2}.
        """
        _check_k(k)
        key, probe_keys, candidates = self.index.query_plan(vector, probes=probes)
        if not candidates:
            return LSHQueryResult(key=key, probes=probe_keys, n_candidates=0, neighbors=())
        point = np.asarray(vector, dtype=float).reshape(-1)
        stored = self.vectors[candidates]
        distances = np.linalg.norm(stored - point, axis=1)
        order = np.argsort(distances, kind="stable")[: min(k, len(candidates))]
        neighbors = tuple(Neighbor(int(candidates[i]), float(distances[i])) for i in order)
        _LOG.debug(
            "LSH collection query: key=%d candidates=%d returned=%d",
            key,
            len(candidates),
            len(neighbors),
        )
        return LSHQueryResult(
            key=key,
            probes=probe_keys,
            n_candidates=len(candidates),
            neighbors=neighbors,
        )


@dataclass(frozen=True)
class ANNReport:
    """The bargain measured: recall against the exact baseline.

    ``recall_at_k`` averages |approx ∩ exact| / k over the queries;
    ``candidate_ratio`` is the mean fraction of the collection the
    queries touched (the compression the engine bought);
    ``empty_queries`` counts queries whose bucket set was empty (the
    failure mode more planes or probes buys back -- at the cost of the
    compression, which is the lesson).
    """

    n_vectors: int
    n_queries: int
    k: int
    n_planes: int
    probes: int
    recall_at_k: float
    candidate_ratio: float
    empty_queries: int

    @property
    def compression(self) -> float:
        """1 / candidate ratio -- how many times cheaper than brute force."""
        return 1.0 / self.candidate_ratio if self.candidate_ratio > 0 else float("inf")


def ann_benchmark(
    vectors: np.ndarray,
    n_planes: int = 8,
    probes: int = 1,
    k: int = 10,
    n_queries: int = 40,
    seed: int = RANDOM_STATE,
) -> ANNReport:
    """Recall@k and candidate compression of LSH against brute force.

    Seeded query rows (the first ``n_queries`` collection rows are the
    queries -- the benchmark convention that keeps the query
    distribution honest), exact top-k by :func:`brute_force_knn`,
    approximate top-k by :class:`LSHCollection`, and the two numbers
    that define the engine: what fraction of the true neighbours
    survived, and what fraction of the collection was never touched.

    Raises
    ------
    DataValidationError
        On malformed vectors, non-positive parameters, probes outside
        {1, 2}, or fewer vectors than queries + k.
    """
    array = _check_vectors(vectors)
    _check_k(k)
    if probes not in (1, 2):
        raise DataValidationError(f"probes must be 1 or 2, got {probes}")
    if n_queries < 1:
        raise DataValidationError(f"n_queries must be >= 1, got {n_queries}")
    if array.shape[0] < n_queries + k:
        raise DataValidationError(
            f"need >= {n_queries + k} vectors for {n_queries} queries at k={k}, "
            f"got {array.shape[0]}"
        )
    rng = np.random.default_rng(seed)
    query_rows = rng.choice(array.shape[0], size=n_queries, replace=False)
    collection = LSHCollection.build(array, n_planes=n_planes, seed=seed)
    recalls: list[float] = []
    ratios: list[float] = []
    empty = 0
    for row in query_rows:
        exact = {n.item for n in brute_force_knn(array, array[row], k=k)}
        result = collection.query(array[row], k=k, probes=probes)
        if result.n_candidates == 0:
            empty += 1
            recalls.append(0.0)
        else:
            approx = {n.item for n in result.neighbors}
            recalls.append(len(exact & approx) / k)
        ratios.append(result.n_candidates / array.shape[0])
    report = ANNReport(
        n_vectors=int(array.shape[0]),
        n_queries=int(n_queries),
        k=int(k),
        n_planes=int(n_planes),
        probes=int(probes),
        recall_at_k=float(np.mean(recalls)),
        candidate_ratio=float(np.mean(ratios)),
        empty_queries=int(empty),
    )
    _LOG.debug(
        "ann_benchmark: planes=%d probes=%d recall@%d=%.3f touched=%.1f%%",
        n_planes,
        probes,
        k,
        report.recall_at_k,
        100.0 * report.candidate_ratio,
    )
    return report
