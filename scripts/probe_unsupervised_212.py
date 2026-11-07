"""Probe: print the real numbers the Unsupervised Workbench tests will assert.

Run: cd /home/z/my-project/pharma-ml-toolkit && /home/z/.venv/bin/python /home/z/my-project/scripts/probe_unsupervised_212.py

The bundled descriptor library (examples/data/library_with_descriptors.csv,
77 rows x 8 descriptor columns) is the page's default dataset; the probe
exercises every engine the page wires -- pca_reduce, kmeans_cluster (plain +
mini-batch + auto-scan), gmm_cluster (BIC sweep + manual), hierarchical_cluster
(ward + complete), kde_fit (auto + manual bandwidth), manifold_embedding
(isomap / lle / mds / spectral / tsne) -- plus the three DataValidationError
paths the page surfaces honestly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TK = Path(__file__).resolve().parents[1] / "pharma-ml-toolkit"
import sys

sys.path.insert(0, str(TK / "src"))

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.manifold import manifold_embedding
from pharmaml.ml.unsupervised import (
    gmm_cluster,
    hierarchical_cluster,
    kde_fit,
    kmeans_cluster,
    pca_reduce,
)
from pharmaml.utils.io import read_table

DESCRIPTOR_COLUMNS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "LabuteASA",
    "RingCount",
]


def main() -> int:
    frame = read_table(TK / "examples" / "data" / "library_with_descriptors.csv")
    print(f"library_with_descriptors.csv: {frame.shape[0]} rows x {frame.shape[1]} cols")
    print(f"descriptor columns: {DESCRIPTOR_COLUMNS}")
    matrix = frame[DESCRIPTOR_COLUMNS]
    print(f"matrix: {matrix.shape}, NaNs={int(matrix.isna().sum().sum())}")

    print("\n=== PCA (auto n_components, scaled) ===")
    pca = pca_reduce(matrix)
    print(f"n_components={pca.n_components} n_samples={pca.n_samples} n_features={pca.n_features} scaled={pca.scaled}")
    print(f"ratios={[round(r, 6) for r in pca.explained_variance_ratio]}")
    print(f"cumulative={[round(c, 6) for c in pca.cumulative_variance]}")
    print(f"total={pca.total_variance_explained:.6f}")
    pc1 = pca.loadings["PC1"].sort_values()
    print(f"PC1 loadings (sorted): {[(i, round(v, 4)) for i, v in pc1.items()]}")
    print(f"PC1 dominant feature: {pc1.abs().idxmax()} (|loading|={pc1.abs().max():.4f})")
    print(f"transformed shape={pca.transformed.shape}, columns={list(pca.transformed.columns[:3])}...")

    print("\n=== PCA (n_components=3) ===")
    pca3 = pca_reduce(matrix, 3)
    print(f"n_components={pca3.n_components}, total={pca3.total_variance_explained:.6f}")

    print("\n=== k-means k=3 (Lloyd) ===")
    km = kmeans_cluster(matrix, k=3)
    print(f"k={km.k} method={km.method} inertia={km.inertia:.4f} silhouette={km.silhouette:.6f}")
    print(f"sizes={list(km.sizes)} (sum={int(km.sizes.sum())})")
    print(f"centers columns={list(km.centers.columns)}")
    print(f"labels dtype={km.labels.dtype}, first 10={list(km.labels[:10])}")
    print(f"scan empty={km.scan.empty}")

    print("\n=== k-means auto (silhouette scan) ===")
    kma = kmeans_cluster(matrix)
    print(f"k={kma.k} method={kma.method} silhouette={kma.silhouette:.6f}")
    print(f"scan rows={len(kma.scan)}, scan k range={list(kma.scan['k'])}")

    print("\n=== mini-batch k-means k=3 ===")
    mb = kmeans_cluster(matrix, k=3, mini_batch=True)
    print(f"k={mb.k} method={mb.method} silhouette={mb.silhouette:.6f} sizes={list(mb.sizes)}")

    print("\n=== GMM (BIC sweep, full covariance) ===")
    gmm = gmm_cluster(matrix)
    print(
        f"n_components={gmm.n_components} covariance={gmm.covariance_type} "
        f"chosen_by={gmm.chosen_by} converged={gmm.converged}"
    )
    print(f"log_likelihood={gmm.log_likelihood:.4f}")
    print(f"weights={[round(w, 4) for w in gmm.weights]} (sum={float(gmm.weights.sum()):.4f})")
    print(f"scan rows={len(gmm.scan)}, n_components range={list(gmm.scan['n_components'])}")
    print(f"probabilities shape={gmm.probabilities.shape}, row sums ok={bool(np.allclose(gmm.probabilities.sum(axis=1), 1.0))}")
    print(f"labels value counts={dict(gmm.labels.value_counts().sort_index())}")

    print("\n=== GMM (manual n=3, diag) ===")
    gmm3 = gmm_cluster(matrix, n_components=3, covariance_type="diag")
    print(f"n_components={gmm3.n_components} chosen_by={gmm3.chosen_by} converged={gmm3.converged}")
    print(f"weights={[round(w, 4) for w in gmm3.weights]}")

    print("\n=== hierarchical (ward, k=3) ===")
    hier = hierarchical_cluster(matrix, k=3)
    print(f"k={hier.k} linkage={hier.linkage} inertia={hier.inertia:.4f} n={hier.n_samples}")
    counts = pd.Series(hier.labels).value_counts().sort_index()
    print(f"sizes={list(counts)} merge_heights={[round(h, 4) for h in hier.merge_heights]}")

    print("\n=== hierarchical (complete, k=3) ===")
    comp = hierarchical_cluster(matrix, k=3, linkage="complete")
    counts_c = pd.Series(comp.labels).value_counts().sort_index()
    print(f"k={comp.k} linkage={comp.linkage} inertia={comp.inertia:.4f} sizes={list(counts_c)}")

    print("\n=== KDE on MolWt (auto bandwidth) ===")
    kde = kde_fit(frame["MolWt"])
    print(
        f"column={kde.column} bandwidth={kde.bandwidth:.6f} source={kde.bandwidth_source} "
        f"kernel={kde.kernel} n={kde.n_samples}"
    )
    print(f"log_likelihood={kde.log_likelihood:.6f} grid size={len(kde.grid)}")
    print(f"grid range=[{kde.grid[0]:.4f}, {kde.grid[-1]:.4f}] density max={kde.density.max():.6f}")

    print("\n=== KDE on MolWt (manual bandwidth 20) ===")
    kde20 = kde_fit(frame["MolWt"], bandwidth=20.0)
    print(f"bandwidth={kde20.bandwidth} source={kde20.bandwidth_source}")

    print("\n=== manifolds ===")
    for method, kwargs in (
        ("isomap", {}),
        ("lle", {}),
        ("lle", {"lle_method": "modified"}),
        ("mds", {}),
        ("spectral", {}),
        ("tsne", {}),
    ):
        rec = manifold_embedding(matrix, method, **kwargs)
        c = rec.coords
        print(
            f"{method} (kwargs={kwargs}): shape={c.shape} cols={list(c.columns)} "
            f"params={rec.params} n_samples={rec.n_samples} n_features={rec.n_features}"
        )
        print(f"  coord1 range=[{c.iloc[:, 0].min():.4f}, {c.iloc[:, 0].max():.4f}]")

    print("\n=== error paths ===")
    try:
        manifold_embedding(matrix, "tsne", perplexity=100.0)
    except DataValidationError as exc:
        print(f"tsne perplexity=100 -> DataValidationError: {exc}")
    try:
        kmeans_cluster(matrix, k=200)
    except DataValidationError as exc:
        print(f"kmeans k=200 -> DataValidationError: {exc}")
    try:
        manifold_embedding(matrix, "isomap", n_neighbors=100)
    except DataValidationError as exc:
        print(f"isomap n_neighbors=100 -> DataValidationError: {exc}")

    print("\n=== second dataset: qsar_regression.csv ===")
    qsar = read_table(TK / "examples" / "data" / "qsar_regression.csv")
    qmatrix = qsar.select_dtypes(include=[np.number])
    print(f"qsar_regression: {qsar.shape[0]} rows, numeric cols={qmatrix.shape[1]}")
    qpca = pca_reduce(qmatrix)
    print(f"auto PCA n_components={qpca.n_components} total={qpca.total_variance_explained:.6f}")
    print(f"first ratio={qpca.explained_variance_ratio[0]:.6f}")

    print("\nprobe done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
