"""Round 221 probe: the Image Lab page's engine-printed numbers.

The house rule (round 205+): every default-input number a Studio page
asserts in its tests was engine-printed FIRST by a committed probe.
This one covers the Part-2 image-visualization closure (directive
16.8.2 ImageView / 16.8.3 ROI / 16.8.4 histogram + colormap controls
/ 16.8.7 + 16.19 export), re-seeded per section so each tab is a
standalone seeded job (the bestofml probe's house pattern).

Sections:
  1. the similarity matrix   (seed_molecules.csv -> morgan 2048-bit ->
                               tanimoto_matrix; the demo's own numbers)
  2. the response matrix     (the bundled GDSC-style screen, the same
                               pipeline as the Biology Workbench's card)
  3. the explorer matrix     (a seeded 12-row demo frame -> z-scores;
                               the same arithmetic the tab applies to
                               the loaded table's numeric columns)
  4. the ROI statistics      (fixed region over the similarity matrix;
                               pure numpy, the widget's arithmetic)
  5. the large-matrix timing (4096 molecules x 2048 bits through
                               tanimoto_matrix -- 16.7M pairs, the
                               16.19 large-dataset performance number)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import default_rng

_TK = Path(__file__).resolve().parents[1]

SEED_MOLECULES = _TK / "examples" / "data" / "seed_molecules.csv"
SCREEN = _TK / "examples" / "data" / "cellline_screen.csv"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    from pharmaml.biology import (
        collapse_replicates,
        load_response_table,
        response_matrix,
    )
    from pharmaml.chem.descriptors import compute_fingerprints
    from pharmaml.chem.similarity import tanimoto_matrix

    # -- 1. the similarity matrix ------------------------------------------
    section("1. similarity matrix (seed_molecules.csv, morgan r=2, 2048 bits)")
    seeds = pd.read_csv(SEED_MOLECULES)
    names = [str(n) for n in seeds["name"].tolist()]
    smiles = [str(s) for s in seeds["smiles"].tolist()]
    matrix, valid = compute_fingerprints(smiles, kind="morgan", radius=2, n_bits=2048)
    print(f"molecules: {len(smiles)}, valid: {len(valid)} (indices {valid})")
    sim = tanimoto_matrix(matrix, matrix)
    print(f"matrix shape: {sim.shape}, dtype: {sim.dtype}")
    print(f"diagonal all-ones: {bool(np.allclose(np.diag(sim), 1.0))}")
    off = sim[~np.eye(len(sim), dtype=bool)]
    print(f"off-diagonal min: {off.min():.6f}")
    print(f"off-diagonal max: {off.max():.6f}")
    print(f"off-diagonal mean: {off.mean():.6f}")
    i, j = np.unravel_index(np.argmax(sim + np.eye(len(sim)) * -2.0), sim.shape)
    print(f"most-similar pair: {names[i]} ~ {names[j]} = {sim[i, j]:.6f}")
    k, l = np.unravel_index(np.argmin(sim + np.eye(len(sim))), sim.shape)
    print(f"least-similar pair: {names[k]} ~ {names[l]} = {sim[k, l]:.6f}")
    mean_per = (sim.sum(axis=1) - 1.0) / (len(sim) - 1)
    top = int(np.argmax(mean_per))
    low = int(np.argmin(mean_per))
    print(f"highest mean-similarity molecule: {names[top]} = {mean_per[top]:.6f}")
    print(f"lowest mean-similarity molecule: {names[low]} = {mean_per[low]:.6f}")

    # -- 2. the response matrix ---------------------------------------------
    section("2. response matrix (bundled GDSC-style screen, LN_IC50)")
    screen = load_response_table(SCREEN)
    collapsed = collapse_replicates(screen, add_cv=True)
    matrix_df = response_matrix(collapsed, "LN_IC50")
    values = matrix_df.to_numpy(dtype=float)
    print(f"frame shape: {len(matrix_df)} x {len(matrix_df.columns)} "
          f"(index={matrix_df.index.name}, columns={list(matrix_df.columns)[:3]}...)")
    print(f"n cells: {len(matrix_df)}, n compounds: {len(matrix_df.columns)}")
    print(f"NaNs in pivot: {int(np.isnan(values).sum())} (the audit's known holes)")
    finite = values[np.isfinite(values)]
    print(f"finite min: {finite.min():.6f} (most sensitive)")
    print(f"finite max: {finite.max():.6f} (most resistant)")
    print(f"finite mean: {finite.mean():.6f}")
    ii, jj = np.unravel_index(
        np.nanargmin(np.where(np.eye(1) * 0 + 1, values, np.nan)), values.shape
    )
    print(f"most sensitive pair: {matrix_df.index[ii]} / {matrix_df.columns[jj]} "
          f"= {values[ii, jj]:.6f}")
    kk, ll = np.unravel_index(np.nanargmax(values), values.shape)
    print(f"most resistant pair: {matrix_df.index[kk]} / {matrix_df.columns[ll]} "
          f"= {values[kk, ll]:.6f}")

    # -- 3. the explorer matrix ----------------------------------------------
    section("3. explorer matrix (seeded 12-row demo frame, z-scores)")
    rng = default_rng(42)
    demo = pd.DataFrame(
        {
            "compound": [f"cmp-{i:02d}" for i in range(12)],
            "potency": rng.normal(5.0, 1.2, 12).round(4),
            "logp": rng.normal(2.0, 0.8, 12).round(4),
            "mw": rng.normal(330.0, 45.0, 12).round(2),
        }
    )
    print("demo frame head (potency/logp/mw raw):")
    print(demo.head(3).to_string(index=False))
    numeric = ["potency", "logp", "mw"]
    block = demo[numeric].to_numpy(dtype=float)
    means = block.mean(axis=0)
    stds = block.std(axis=0, ddof=0)
    print(f"raw column means: {np.round(means, 6).tolist()}")
    print(f"raw column stds (ddof=0): {np.round(stds, 6).tolist()}")
    z = (block - means) / stds
    print(f"z-matrix shape: {z.shape}")
    print(f"z column means: {np.round(z.mean(axis=0), 6).tolist()}")
    print(f"z column stds: {np.round(z.std(axis=0, ddof=0), 6).tolist()}")
    print(f"z min: {z.min():.6f}, z max: {z.max():.6f}")
    print(f"z[0, :]: {np.round(z[0], 6).tolist()}")
    print(f"row of max |z|: {int(np.argmax(np.abs(z).max(axis=1)))} "
          f"(|z| = {np.abs(z).max():.6f})")

    # -- 4. the ROI statistics ------------------------------------------------
    section("4. ROI statistics (fixed region over the similarity matrix)")
    r0, r1, c0, c1 = 2, 6, 7, 11  # rows [2,6), cols [7,11)
    region = sim[r0:r1, c0:c1]
    print(f"region rows {r0}..{r1 - 1} ({names[r0]}..{names[r1 - 1]}), "
          f"cols {c0}..{c1 - 1} ({names[c0]}..{names[c1 - 1]})")
    print(f"region cells: {region.size}, mean: {region.mean():.6f}, "
          f"min: {region.min():.6f}, max: {region.max():.6f}")
    print(f"region per-row means: {np.round(region.mean(axis=1), 6).tolist()}")

    # -- 5. the large-matrix timing -------------------------------------------
    section("5. large-matrix timing (tanimoto_matrix over seeded bit matrices)")
    for n in (1024, 4096):
        rng2 = default_rng(42)
        big = (rng2.random((n, 2048)) < 0.15).astype(np.uint8)
        t0 = time.perf_counter()
        big_sim = tanimoto_matrix(big, big)
        elapsed = time.perf_counter() - t0
        off_mean = (big_sim.sum() - np.trace(big_sim)) / (n * (n - 1))
        print(f"{n}^2 = {n * n:,} pairs in {elapsed:.3f} s "
              f"({n * n / elapsed / 1e6:.1f} M pairs/s), "
              f"diag mean {np.diag(big_sim).mean():.6f}, "
              f"off-diagonal mean {off_mean:.6f}")


if __name__ == "__main__":
    main()
