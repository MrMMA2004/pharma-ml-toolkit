#!/usr/bin/env python3
"""Round-34 audit probe (100x-151) — DISCLOSED, read-only, stdlib + numpy/sklearn.

Two questions, one run:
  A. sklearn 1.5.x quantile-regression semantics for the D2.py port target:
     GradientBoostingRegressor(loss="quantile", alpha=q) and
     HistGradientBoostingRegressor(loss="quantile", quantile=q) -- which
     spellings exist, and do 5%/95% models give a non-degenerate interval
     with sane empirical coverage on train?
  B. D2.py's generate_uncertainty_quantification as shipped: three
     RandomForestRegressor(n_estimators=150, max_depth=10, seed 42) on the
     SAME data with NO quantile loss -- identical forests? zero width?
     ~0% coverage?
  C. pic50_to_ic50's float(pic50) lexing edge: a huge python int
     (10**400) raises OverflowError INSIDE float() -- is that caught by
     the (TypeError, ValueError) net?  (The pow-side OverflowError IS
     caught; this probes the lex side only, no pharmaml import needed.)

Run:  .venv/bin/python scratch/probe_audit_151.py   (repo-relative)
"""
from __future__ import annotations

import numpy as np


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def probe_sklearn_version() -> None:
    import sklearn

    section("A0. sklearn version")
    print(f"sklearn {sklearn.__version__}")


def probe_quantile_spelling() -> None:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    section("A1. quantile spelling on GradientBoostingRegressor")
    try:
        GradientBoostingRegressor(loss="quantile", alpha=0.05)
        print("GBR(loss='quantile', alpha=0.05) accepted -> alpha= is the knob")
    except TypeError as exc:
        print(f"GBR alpha= rejected: {exc}")
    try:
        GradientBoostingRegressor(loss="quantile", quantile=0.05)
        print("GBR(loss='quantile', quantile=0.05) accepted -> quantile= is the knob")
    except TypeError as exc:
        print(f"GBR quantile= rejected: {exc}")

    try:
        GradientBoostingRegressor(loss="banana")
        print("WARNING: loss='banana' accepted??")
    except ValueError as exc:
        print(f"GBR loss='banana' rejected (as expected): {str(exc)[:80]}")

    section("A2. RandomForestRegressor has NO quantile mode")
    rf = RandomForestRegressor(n_estimators=5, random_state=0)
    params = rf.get_params()
    quantileish = [k for k in params if "quant" in k.lower() or k == "alpha"]
    print(f"RF params carrying any quantile/alpha knob: {quantileish or 'NONE'}")
    try:
        RandomForestRegressor(loss="quantile")
        print("WARNING: RF(loss='quantile') accepted??")
    except TypeError as exc:
        print(f"RF(loss=...) rejected: {exc}")


def probe_quantile_coverage() -> None:
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
    )

    section("A3. empirical 90% interval coverage (GBR quantile trio)")
    rng = np.random.RandomState(42)
    n = 400
    x = rng.uniform(0, 10, size=(n, 1))
    y = (2.0 * x[:, 0] + rng.normal(0, 1.0, size=n)).ravel()
    lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, random_state=42).fit(x, y)
    mid = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=42).fit(x, y)
    hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, random_state=42).fit(x, y)
    p_lo, p_mid, p_hi = lo.predict(x), mid.predict(x), hi.predict(x)
    width = p_hi - p_lo
    covered = np.mean((y >= p_lo) & (y <= p_hi))
    print(f"GBR trio on train: width min/med/max = "
          f"{width.min():.3f}/{np.median(width):.3f}/{width.max():.3f}")
    print(f"GBR trio on train: coverage = {covered:.1%}")
    print(f"ordering lower<=median<=upper: {bool(np.all(p_lo <= p_mid) and np.all(p_mid <= p_hi))}")

    section("A4. HistGradientBoostingRegressor quantile spelling")
    try:
        hlo = HistGradientBoostingRegressor(loss="quantile", quantile=0.05).fit(x, y)
        print(f"HGB(loss='quantile', quantile=0.05) accepted; "
              f"mean pred {hlo.predict(x).mean():.3f}")
    except TypeError as exc:
        print(f"HGB quantile= rejected: {exc}")
    try:
        HistGradientBoostingRegressor(loss="quantile", alpha=0.05)
        print("HGB alpha= accepted")
    except TypeError as exc:
        print(f"HGB alpha= rejected: {exc}")


def probe_d2_uncertainty_bug() -> None:
    from sklearn.ensemble import RandomForestRegressor

    section("B. D2.py generate_uncertainty_quantification as shipped")
    rng = np.random.RandomState(0)
    n = 300
    x_train = rng.uniform(0, 10, size=(n, 6))
    y_train = (2.0 * x_train[:, 0] + rng.normal(0, 1.0, size=n)).ravel()
    x_test = rng.uniform(0, 10, size=(100, 6))
    y_test = 2.0 * x_test[:, 0] + rng.normal(0, 1.0, size=100)

    def fresh() -> RandomForestRegressor:
        # EXACTLY the upload's constructor (D2.py line 539-541), N_JOBS=1
        # here (the probe box has fewer cores; n_jobs does not change the
        # seeded tree stream, verified below by the identity itself).
        return RandomForestRegressor(
            n_estimators=150, max_depth=10, n_jobs=1, random_state=42
        )

    lower, median, upper = fresh(), fresh(), fresh()
    lower.fit(x_train, y_train)
    median.fit(x_train, y_train)
    upper.fit(x_train, y_train)
    p_lo, p_mid, p_hi = lower.predict(x_test), median.predict(x_test), upper.predict(x_test)
    identical_lo_mid = bool(np.array_equal(p_lo, p_mid))
    identical_mid_hi = bool(np.array_equal(p_mid, p_hi))
    width = p_hi - p_lo
    coverage = float(np.mean((y_test >= p_lo) & (y_test <= p_hi)))
    print(f"lower == median predictions bit-for-bit: {identical_lo_mid}")
    print(f"median == upper predictions bit-for-bit: {identical_mid_hi}")
    print(f"interval width: min={width.min():.3e} max={width.max():.3e} (all zero?) "
          f"{bool(np.all(width == 0.0))}")
    print(f"empirical coverage of the '90%' interval: {coverage:.1%}")
    exact = int(np.sum(y_test == p_mid))
    print(f"points where y_test == the degenerate 'interval' exactly: {exact}/{len(y_test)}")


def probe_float_lexing_edge() -> None:
    section("C. float(pic50) lexing edge (pic50_to_ic50's first net)")
    try:
        value = float(10**400)  # a huge python int
        print(f"float(10**400) -> {value}")
    except OverflowError as exc:
        print(f"float(10**400) raises OverflowError: {exc}  "
              "-> NOT in (TypeError, ValueError); escapes the net")
    from decimal import Decimal

    try:
        value = float(Decimal("1e400"))
        print(f"float(Decimal('1e400')) -> {value} (no raise)")
    except OverflowError as exc:
        print(f"float(Decimal('1e400')) raises OverflowError: {exc}")
    # the reachable surfaces: pandas int64/float64 cells convert cleanly
    import pandas as pd

    s = pd.Series([6.0, -300.0, 10**18], dtype="int64")
    print(f"pandas int64 cell -> float(): {[float(v) for v in s.tolist()]}")


if __name__ == "__main__":
    probe_sklearn_version()
    probe_quantile_spelling()
    probe_quantile_coverage()
    probe_d2_uncertainty_bug()
    probe_float_lexing_edge()
    print("\nPROBE_151_DONE")
