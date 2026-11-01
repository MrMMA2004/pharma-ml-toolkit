"""Probe: does hierarchical_cluster's merge_heights let me rebuild the full merge tree?

Hypothesis (from code reading): merge_heights at cut k = heights of the k-1
merges ABOVE the cut (ascending). Therefore height of merge P_{k+1} -> P_k
= min(merge_heights at cut k+1). Structure comes from successive partitions.
"""
import numpy as np
from pharmaml.ml.unsupervised import hierarchical_cluster

rng = np.random.default_rng(7)
blobs = np.vstack([
    rng.normal(size=(3, 2)) + 5.0,
    rng.normal(size=(3, 2)) - 5.0,
    rng.normal(size=(3, 2)),
])
n = blobs.shape[0]
print("n =", n)

for linkage in ("ward", "complete"):
    parts = {}
    heights = {}
    for k in range(n, 1, -1):
        res = hierarchical_cluster(blobs, k=k, linkage=linkage)
        parts[k] = [frozenset(np.flatnonzero(res.labels == c).tolist()) for c in range(res.k)]
        heights[k] = list(res.merge_heights)
    print(f"--- linkage={linkage}")
    for k in sorted(parts):
        print(f"cut {k}: parts={[sorted(s) for s in sorted(parts[k], key=min)]} heights={heights[k]}")
    # rebuild merges
    merges = []
    for k in range(n - 1, 0, -1):
        upper = parts[k + 1] if k + 1 in parts else [frozenset([i]) for i in range(n)]
        lower = parts[k] if k in parts else [frozenset(range(n))]
        found = None
        for a_idx in range(len(upper)):
            for b_idx in range(a_idx + 1, len(upper)):
                union = upper[a_idx] | upper[b_idx]
                if union in lower:
                    found = (upper[a_idx], upper[b_idx])
                    break
            if found:
                break
        h = min(heights[k + 1]) if k + 1 in heights else None
        merges.append((sorted(found[0]), sorted(found[1]), h))
    print("merges (A, B, height):")
    for m in merges:
        print("  ", m)
    # cross-check: ward heights should be nondecreasing
    hs = [m[2] for m in merges]
    print("nondecreasing heights:", all(hs[i] <= hs[i + 1] for i in range(len(hs) - 1)))
