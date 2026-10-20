#!/usr/bin/env python3
"""105 verification, Python side: sklearn GPC on the EXACT dataset the
TypeScript fitter saw (the deterministic mod-based grid from
scratch/verify_gpc_parity.ts, replicated in numpy) — compares the
probabilities / Newton iterations / Laplace lml against the TS output.

Run:  /home/z/.venv/bin/python scratch/verify_gpc_parity_ts.py
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


def build_dataset():
    """Replicate scratch/verify_gpc_parity.ts's deterministic grid."""
    X, y = [], []
    for i in range(60):
        a = -2 + (((i * 37) % 41) / 41) * 4
        b = -1.5 + (((i * 23) % 53) / 53) * 3
        X.append([a, b])
        upper = b > 0.35 + 0.55 * np.sin(a)
        y.append(1 if upper else 0)
    grid = []
    for i in range(25):
        grid.append(
            [-2 + (((i * 29) % 37) / 37) * 4, -1.5 + (((i * 17) % 31) / 31) * 3]
        )
    return np.array(X), np.array(y), np.array(grid)


def main() -> None:
    with open("/tmp/gpc_ts.json") as fh:
        ts = json.load(fh)

    X, y, grid = build_dataset()
    # The web fitter standardizes features on TRAIN statistics (the
    # honest no-leakage contract every web fitter follows — kernel
    # hyperparameters live in standardized-feature units); replicate the
    # same transform (population sd, ddof=0) before fitting sklearn so
    # both sides solve the same latent GP.
    mu_x = X.mean(axis=0)
    sd_x = np.sqrt(((X - mu_x) ** 2).mean(axis=0))
    Z = (X - mu_x) / sd_x
    Zg = (grid - mu_x) / sd_x
    kernel = ConstantKernel(1.2**2) * RBF(length_scale=0.8)
    clf = GaussianProcessClassifier(kernel=kernel, optimizer=None, random_state=42)
    clf.fit(Z, y)
    sk_proba = clf.predict_proba(Zg)[:, 1]

    d = np.abs(sk_proba - np.array(ts["probas"])).max()
    lml_d = abs(clf.log_marginal_likelihood() - ts["logMarginalLikelihood"])
    print(f"sklearn lml:      {clf.log_marginal_likelihood():.10f}")
    print(f"TS       lml:     {ts['logMarginalLikelihood']:.10f}")
    print(f"lml delta:        {lml_d:.3e}")
    print(f"max |proba Δ|:    {d:.3e}")
    print(f"proba range:      [{min(ts['probas']):.4f}, {max(ts['probas']):.4f}]")

    ok = d < 1e-5 and lml_d < 1e-9
    print("GATE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
