#!/usr/bin/env python3
"""Round-35 audit probes (100x-152) for the round-34 quantile additions.

READ-ONLY audit: imports pharmaml (allowed this round), writes nothing.
Run: .venv/bin/python scratch/probe_audit_152.py   (repo-relative)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml import quantile_prediction_intervals


def toy(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    x = pd.DataFrame({"a": rng.uniform(0.0, 10.0, n)})
    y = pd.Series(2.0 * x["a"] + 1.0 + rng.normal(0.0, 1.0, n))
    return x, y


def main() -> None:
    x, y = toy()

    print("== P1: knob-type edge cases (string / float n_estimators) ==")
    for label, kwargs in [
        ("n_estimators=2.5", {"n_estimators": 2.5}),
        ("n_estimators='10'", {"n_estimators": "10"}),
        ("alpha='0.1'", {"alpha": "0.1"}),
        ("subsample='0.5'", {"subsample": "0.5"}),
        ("learning_rate='0.1'", {"learning_rate": "0.1"}),
        ("max_depth=-1", {"max_depth": -1}),
        ("max_depth=0", {"max_depth": 0}),
    ]:
        try:
            quantile_prediction_intervals(x, y, x.iloc[:4], **kwargs)
            print(f"  {label:24s} -> NO ERROR (accepted)")
        except ModelError as exc:
            print(f"  {label:24s} -> ModelError: {str(exc)[:70]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:24s} -> {type(exc).__name__}: {str(exc)[:70]}")

    print("== P2: alpha extremes inside (0, 0.5) ==")
    for alpha in (1e-6, 0.49, 0.4999999):
        res = quantile_prediction_intervals(
            x, y, x.iloc[:4], alpha=alpha, n_estimators=15)
        print(f"  alpha={alpha!r:12s} width={res.mean_width:12.4g} "
              f"crossing={res.n_crossing} coverage={res.coverage}")

    print("== P3: coverage arithmetic on a hand-made bracket (toy) ==")
    truth = pd.Series([1.0, 2.0, 3.0, 4.0])
    # lower/upper crafted to straddle exactly 3 of the 4 truths
    lower = np.array([0.5, 2.5, 2.5, 5.0])
    upper = np.array([1.5, 3.5, 3.5, 6.0])
    inside = (truth.values >= lower) & (truth.values <= upper)
    print(f"  expected coverage 3/4 = {inside.mean()} "
          f"(inclusive bounds, truth==bound counts as covered)")
    # and the library's identical formula on a real run:
    res = quantile_prediction_intervals(x, y, x.iloc[:20], y.iloc[:20],
                                        n_estimators=25)
    recomputed = ((y.iloc[:20].values >= res.lower)
                  & (y.iloc[:20].values <= res.upper))
    assert res.coverage == float(recomputed.mean()), "coverage drift"
    print(f"  library coverage == recomputed: {res.coverage:.3f} OK")

    print("== P4: crossing repair keeps bounds ordered, counts raw ==")
    # force crossings with a tiny alpha (5e-3 -> 0.25%/99.75% quantiles)
    res = quantile_prediction_intervals(x, y, x.iloc[:20], alpha=1e-6,
                                        n_estimators=15)
    ok = bool((res.lower <= res.upper).all())
    print(f"  alpha=1e-6: ordered={ok} n_crossing={res.n_crossing} "
          f"width={res.mean_width:.4g}")

    print("== P5: as_frame with mismatched y_test length ==")
    res = quantile_prediction_intervals(x, y, x.iloc[:4], n_estimators=15)
    try:
        res.as_frame(y.iloc[:3])
        print("  mismatched as_frame -> NO ERROR (unexpected)")
    except Exception as exc:  # noqa: BLE001
        print(f"  mismatched as_frame -> {type(exc).__name__}: {str(exc)[:60]}")

    print("== P6: X_test empty / y_train list input ==")
    try:
        quantile_prediction_intervals(x, y, x.iloc[:0], n_estimators=10)
    except DataValidationError as exc:
        print(f"  empty X_test -> DataValidationError: {str(exc)[:50]}")
    res = quantile_prediction_intervals(
        x, list(y), x.iloc[:4], n_estimators=15)
    print(f"  list y_train accepted, n_test={res.n_test}")

    print("== P7: determinism with subsample<1 and same seed ==")
    a = quantile_prediction_intervals(x, y, x.iloc[:8], subsample=0.8,
                                      n_estimators=20, random_state=42)
    b = quantile_prediction_intervals(x, y, x.iloc[:8], subsample=0.8,
                                      n_estimators=20, random_state=42)
    same = bool(np.allclose(a.lower, b.lower))
    print(f"  same-seed stochastic runs identical: {same}")

    print("PROBE152_DONE")


if __name__ == "__main__":
    main()
