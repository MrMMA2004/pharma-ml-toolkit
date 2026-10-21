"""100x-114 — EI/UCB acquisition-math parity, Python side.

The web GP band overlay (src/lib/bayes-acquisition.ts, extracted from
ml-playground.tsx this round) implements Φ via the Abramowitz–Stegun
26.2.17 polynomial and EI/UCB on top of it.  This probe:

1. emits the FIXED input grid + the scipy.stats.norm gold values to
   /tmp/acq_inputs_114.json + /tmp/acq_ref_114.json (for the TS twin),
2. mirrors the shipped TS formulas line-by-line in Python and checks
   the AS-26.2.17 error bound claim (|ε_Φ| < 7.5e-8) against scipy,
3. if /tmp/acq_ts_114.json exists (the TS twin already ran), prints
   the cross-language verdict with the SAME tolerances the TS side
   enforces: Φ ≤ 1.5e-7, φ ≤ 1e-12, EI ≤ 1e-6, UCB ≤ 1e-12 (abs).

Run: python3 scratch/verify_acquisition_parity_114.py   (then the TS twin)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import norm

INPUTS = Path("/tmp/acq_inputs_114.json")
REF = Path("/tmp/acq_ref_114.json")
TS_OUT = Path("/tmp/acq_ts_114.json")


def _grid() -> dict[str, list[float] | list[list[float]]]:
    z_coarse = [-8.0, -5.0, -3.0, -1.5, -0.75, -0.1, 0.0, 0.1, 0.75, 1.5, 3.0, 5.0, 8.0]
    z_fine = [round(-4.0 + 0.13 * i, 2) for i in range(62)]  # -4.00 .. 3.93
    z = sorted({*z_coarse, *z_fine})

    ei_cases: list[list[float]] = []
    for mu in (-3.0, -1.0, 0.0, 0.5, 2.0, 4.0):
        for sigma in (0.05, 0.3, 1.0, 2.5):
            for f_star in (-1.0, 0.0, 1.5):
                ei_cases.append([mu, sigma, f_star])
    # the guard branch: sigma at/under the 1e-12 collapse threshold
    ei_cases += [[2.0, 1e-12, 1.0], [2.0, 0.0, 3.0], [-1.0, 1e-13, -2.0]]

    ucb_cases: list[list[float]] = []
    for mu in (-2.0, 0.0, 1.7):
        for sigma in (0.01, 0.8, 3.0):
            for kappa in (0.25, 1.0, 2.5, 3.0):
                ucb_cases.append([mu, sigma, kappa])
    return {"z": z, "ei": ei_cases, "ucb": ucb_cases}


def _ts_mirror_cdf(z: float) -> float:
    """The shipped TS stdNormalCdf, mirrored line-by-line."""
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    p = 1.0 - (math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)) * poly
    return p if z >= 0 else 1.0 - p


def _ts_mirror_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _ts_mirror_ei(mu: float, sigma: float, f_star: float) -> float:
    if sigma <= 1e-12:
        return max(0.0, mu - f_star)
    z = (mu - f_star) / sigma
    return (mu - f_star) * _ts_mirror_cdf(z) + sigma * _ts_mirror_pdf(z)


def main() -> None:
    grid = _grid()
    INPUTS.write_text(json.dumps(grid))

    # -- scipy gold ------------------------------------------------------
    ref = {
        "cdf": [float(norm.cdf(z)) for z in grid["z"]],
        "pdf": [float(norm.pdf(z)) for z in grid["z"]],
        # the reference honours the SAME degenerate-sigma convention as
        # the shipped guard branch, so the guard is parity-checked too
        "ei": [
            (
                max(0.0, mu - f_star)
                if sigma <= 1e-12
                else float(
                    (mu - f_star) * norm.cdf((mu - f_star) / sigma)
                    + sigma * norm.pdf((mu - f_star) / sigma)
                )
            )
            for mu, sigma, f_star in grid["ei"]
        ],
        "ucb": [mu + kappa * sigma for mu, sigma, kappa in grid["ucb"]],
    }
    REF.write_text(json.dumps(ref))

    # -- python mirror of the shipped formulas vs scipy -------------------
    worst = {
        "cdf": max(
            abs(_ts_mirror_cdf(z) - c) for z, c in zip(grid["z"], ref["cdf"])
        ),
        "pdf": max(
            abs(_ts_mirror_pdf(z) - p) for z, p in zip(grid["z"], ref["pdf"])
        ),
        "ei": max(
            abs(_ts_mirror_ei(mu, sigma, f_star) - r)
            for (mu, sigma, f_star), r in zip(grid["ei"], ref["ei"])
        ),
    }
    print(f"python mirror vs scipy: max|dPhi|={worst['cdf']:.3e} (bound 7.5e-8 claim)")
    print(f"python mirror vs scipy: max|dPdf|={worst['pdf']:.3e}")
    print(f"python mirror vs scipy: max|dEI |={worst['ei']:.3e}")
    assert worst["cdf"] < 7.5e-8, "AS-26.2.17 error bound violated in the mirror"
    assert worst["pdf"] < 1e-12
    assert worst["ei"] < 1e-6

    # -- cross-language verdict when the TS twin has run ------------------
    if TS_OUT.exists():
        ts = json.loads(TS_OUT.read_text())
        diffs = {
            k: max(abs(a - b) for a, b in zip(ts[k], ref[k]))
            for k in ("cdf", "pdf", "ei", "ucb")
        }
        tols = {"cdf": 1.5e-7, "pdf": 1e-12, "ei": 1e-6, "ucb": 1e-12}
        print("TS vs scipy parity (same tolerances as the TS twin):")
        for k in ("cdf", "pdf", "ei", "ucb"):
            ok = diffs[k] <= tols[k]
            print(f"  {k:>3}: max|d|={diffs[k]:.3e} tol={tols[k]:.1e} {'OK' if ok else 'FAIL'}")
        assert all(diffs[k] <= tols[k] for k in diffs), "TS/scipy parity FAILED"
        print("ACQUISITION PARITY: PASS (TS twin vs scipy, all four quantities)")
    else:
        print("TS twin output not found yet — run the bun side for the cross verdict.")


if __name__ == "__main__":
    main()
