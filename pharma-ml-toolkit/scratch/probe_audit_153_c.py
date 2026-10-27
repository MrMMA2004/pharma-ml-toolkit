"""Probe C: the three new viz plots (signed bars, corr heatmap, perm heatmap).

Round-36 audit (100x-153). Deterministic, self-contained, no network.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.evaluation import PermutationImportance
from pharmaml.viz.ml_plots import (
    plot_correlation_heatmap,
    plot_permutation_heatmap,
    plot_signed_correlations,
)

def rec(names, means):
    return PermutationImportance(
        names=list(names), means=np.asarray(means, dtype=float),
        stds=np.zeros(len(names)), scoring="r2", n_repeats=5, n_samples=30,
    )

# ---------------------------------------------------------------- C1: signed bars basics
corr = pd.Series({"logp": 0.9, "mw": -0.8, "tpsa": 0.3, "zero": 0.0})
fig = plot_signed_correlations(corr)
ax = fig.axes[0]
print("C1 bars:", len(ax.patches), "ticks:", [t.get_text() for t in ax.get_yticklabels()])
print("C1 title:", ax.get_title())
import matplotlib.pyplot as plt

plt.close(fig)

# C1b: top larger than length
fig = plot_signed_correlations(corr, top=99)
print("C1b top>len keeps all:", len(fig.axes[0].patches))
plt.close(fig)

# C1c: NaN dropped then head taken -- head must come from the SURVIVORS
corr_nan = pd.Series({"a": 0.9, "bad": float("nan"), "b": 0.5, "c": 0.3})
fig = plot_signed_correlations(corr_nan, top=2)
print("C1c NaN-drop head:", [t.get_text() for t in fig.axes[0].get_yticklabels()])
plt.close(fig)

# ---------------------------------------------------------------- C2: correlation heatmap
rng = np.random.default_rng(42)
frame = pd.DataFrame(rng.uniform(size=(40, 5)), columns=list("abcde"))
fig = plot_correlation_heatmap(frame)
ax = fig.axes[0]
print("C2 ticks:", [t.get_text() for t in ax.get_yticklabels()],
      [t.get_text() for t in ax.get_xticklabels()])
print("C2 texts (annotated cells):", len(ax.texts))
image = ax.images[0]
print("C2 clim:", image.get_clim())
plt.close(fig)

# C2b: constant column -> NaN row/col renders grey, no annotation there
fconst = pd.DataFrame({"a": rng.uniform(size=30), "const": [1.0] * 30})
fig = plot_correlation_heatmap(fconst)
ax = fig.axes[0]
print("C2b annotated cells (expect 2, diagonal a-a and const-const NaN):", len(ax.texts))
cmap = ax.images[0].get_cmap()
nan_rgba = cmap.set_bad  # noqa: F841  (set_bad already applied at build)
bad = cmap(np.ma.masked_invalid(np.array([np.nan]))[0])
print("C2b bad-colour rgba:", bad)
plt.close(fig)

# C2c: duplicate column labels in columns=
try:
    fig = plot_correlation_heatmap(frame, columns=["a", "a"])
    print("C2c duplicate columns accepted:", fig.axes[0].get_title())
    plt.close(fig)
except DataValidationError as exc:
    print("C2c duplicate columns typed:", str(exc)[:60])
except Exception as exc:  # noqa: BLE001
    print("C2c duplicate columns OTHER:", type(exc).__name__, str(exc)[:80])

# ---------------------------------------------------------------- C3: permutation heatmap
panel = {
    "ols": rec(["signal", "noise"], [0.5, 0.0]),
    "ridge": rec(["signal", "noise"], [0.4, -0.1]),
}
fig = plot_permutation_heatmap(panel)
ax = fig.axes[0]
print("C3 rows:", [t.get_text() for t in ax.get_yticklabels()],
      "cols:", [t.get_text() for t in ax.get_xticklabels()])
print("C3 annotated cells:", len(ax.texts))
print("C3 clim (symmetric):", ax.images[0].get_clim())
plt.close(fig)

# C3b: absent feature -> grey cell; check the rendered matrix + bad colour
panel2 = {
    "m1": rec(["f1", "f2"], [0.8, 0.2]),
    "m2": rec(["f1"], [0.4]),          # f2 never seen by m2
}
fig = plot_permutation_heatmap(panel2)
ax = fig.axes[0]
m = ax.images[0].get_array()
print("C3b matrix:\n", m)
cmap = ax.images[0].get_cmap()
print("C3b grey for NaN:", cmap(np.ma.masked_invalid(np.array([np.nan]))[0]))
print("C3b colorbar label:", fig.axes[1].get_ylabel())
plt.close(fig)

# C3c: a feature ONLY one model saw with big |imp| ranks above agreed features
panel3 = {
    "m1": rec(["solo"], [0.9]),
    "m2": rec(["shared"], [0.5]),
    "m3": rec(["shared"], [0.5]),
}
fig = plot_permutation_heatmap(panel3, top=1)
print("C3c top-1 row is solo:", [t.get_text() for t in fig.axes[0].get_yticklabels()])
plt.close(fig)

# C3d: top=None shows every feature
fig = plot_permutation_heatmap(panel3, top=None)
print("C3d top=None rows:", [t.get_text() for t in fig.axes[0].get_yticklabels()])
plt.close(fig)

# C3e: empty records panel
try:
    plot_permutation_heatmap({"m1": rec([], [])})
    print("C3e empty-record accepted")
except DataValidationError as exc:
    print("C3e empty-record typed:", str(exc)[:90])

# C3f: duplicate feature names inside ONE record
try:
    fig = plot_permutation_heatmap({"m1": rec(["dup", "dup"], [0.1, 0.7])})
    ax = fig.axes[0]
    print("C3f duplicate names accepted; rows:", [t.get_text() for t in ax.get_yticklabels()])
    print("C3f matrix:", ax.images[0].get_array())
    plt.close(fig)
except Exception as exc:  # noqa: BLE001
    print("C3f duplicate names OTHER:", type(exc).__name__, str(exc)[:80])

# C3g: determinism -- same input, two figures, identical data + ticks
f1 = plot_permutation_heatmap(panel)
d1 = f1.axes[0].images[0].get_array()
t1 = [t.get_text() for t in f1.axes[0].get_yticklabels()]
f2 = plot_permutation_heatmap(panel)
d2 = f2.axes[0].images[0].get_array()
t2 = [t.get_text() for t in f2.axes[0].get_yticklabels()]
print("C3g deterministic:", np.array_equal(np.ma.filled(d1, 1e9), np.ma.filled(d2, 1e9)), t1 == t2)
plt.close(f1)
plt.close(f2)

# C3h: all-zero importances -> bound floor, white cells, dark annotations
fig = plot_permutation_heatmap({"m": rec(["a", "b"], [0.0, 0.0])})
print("C3h clim:", fig.axes[0].images[0].get_clim(), "texts:", len(fig.axes[0].texts))
plt.close(fig)

# C3i: single model (one column) still aligned
fig = plot_permutation_heatmap({"only": rec(["x", "y"], [0.3, -0.2])})
ax = fig.axes[0]
print("C3i cols:", [t.get_text() for t in ax.get_xticklabels()],
      "rows:", [t.get_text() for t in ax.get_yticklabels()])
plt.close(fig)

# C3j: top=0 / negative guard
for bad in (0, -3, 2.5, "5"):
    try:
        plot_permutation_heatmap(panel, top=bad)  # type: ignore[arg-type]
        print("C3j FAIL accepted", bad)
    except DataValidationError as exc:
        print("C3j typed for", repr(bad), ":", str(exc)[:60])

print("PROBE_C_DONE")
