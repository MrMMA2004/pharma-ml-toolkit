"""100x-114 flake hunt: GPC deterministic-refit under CPU load.

The 100x-114 baseline run failed TestGaussianProcessClassifierFamily::
test_deterministic_refit once (exact-equality assert_array_equal on
predict_proba) while ruff+mypy ran in PARALLEL; unloaded re-runs were
green 8/8.  Hypothesis: OpenBLAS thread scheduling under load changes
gemm reduction order by 1 ULP.  This probe fits GPC twice, N rounds,
half under artificial CPU load, and reports bit-exact equality rate.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _stopworkers(pool: list[mp.Process]) -> None:
    for p in pool:
        p.terminate()
        p.join()


def _busy() -> None:  # pragma: no cover - worker
    while True:
        sum(i * i for i in range(10000))


def main() -> None:
    rng = np.random.default_rng(7)
    n = 120
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.0, 1.0, n)
    y = (b + 0.8 * np.sin(a) > 0.0).astype(int)
    x = pd.DataFrame({"a": a, "b": b})

    def fit_predict() -> np.ndarray:
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "gpc",
                    GaussianProcessClassifier(
                        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
                    ),
                ),
            ]
        )
        pipe.fit(x, y)
        return pipe.predict_proba(x)

    rounds = 40
    mismatches_plain = 0
    for _ in range(rounds // 2):
        if not np.array_equal(fit_predict(), fit_predict()):
            mismatches_plain += 1

    # ONE busy worker on a 2-core box: enough to perturb scheduling
    # without starving the probe (3 workers starved it — 240 s timeout).
    workers = [mp.Process(target=_busy, daemon=True)]
    for w in workers:
        w.start()
    mismatches_load = 0
    try:
        for _ in range(rounds // 2):
            if not np.array_equal(fit_predict(), fit_predict()):
                mismatches_load += 1
    finally:
        _stopworkers(workers)

    print(
        f"plain: {mismatches_plain}/{rounds // 2} mismatched refits; "
        f"under-load: {mismatches_load}/{rounds // 2} mismatched refits"
    )
    print(
        "openblas threads env:",
        __import__("os").environ.get("OPENBLAS_NUM_THREADS", "<unset>"),
    )


if __name__ == "__main__":
    main()
