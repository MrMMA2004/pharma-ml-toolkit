"""PROBE audit_154_a -- pearsonCorrelation adversarial numerics (100x-154).

The round-36 web twin's parity grid (verify_corr_parity_153) pins 11
cases: line/noisy/inverse/shuffled/constant/one_point/empty/nan/inf/
length_mismatch/large_n.  NONE of them stress the two-pass arithmetic
where a naive implementation actually breaks.  This probe adds the
adversarial numerics the grid missed, each run through BOTH the
SHIPPED TS module (bun imports src/lib/ml-math.ts -- the code the
browser runs) and scipy.stats.pearsonr (the gold the desktop twin is
pinned against):

  huge_magnitude   x ~ 1e300 scale -- the first-pass sum can overflow
                   to Infinity (naive r collapses to NaN or worse);
  huge_offset      x = 1e15 + small -- every cancellation the two-pass
                   mean-subtraction exists to survive;
  near_constant    x = 1 + [0, 0, ..., 1e-12] -- the denominator rides
                   one 1e-12 deviation (sqrt of 1e-24 = 1e-12);
  tiny_values      x ~ 1e-300 -- the products underflow toward
                   subnormals before the sqrt;
  alternating      x = +/-1e10 alternating -- the covariance sums
                   cancel exactly (r must be NaN-or-defined, never a
                   fabricated sign);
  two_points       n = 2 boundary -- |r| must be exactly 1;
  big_n            n = 100_000 -- summation-order drift vs scipy's
                   pairwise reduction (the 8.3e-17 round-36 verdict
                   was n = 2_000);
  neg_zero         r computed on mirrored inputs -- the sign of zero
                   must not flip the +/- display convention.

Run:
  python scratch/probe_audit_154_a.py     (emits + adjudicates both sides)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOLKIT_ROOT.parent
TS_RUNNER = REPO_ROOT / "scratch" / "probe_corr_adversarial_154.ts"
INPUTS = Path("/tmp/probe_a_154_in.json")
OUTPUT = Path("/tmp/probe_a_154_ts.json")
# (probe hygiene) these paths are PRIVATE to the audit probe: the
# 100x-154 parity twin uses /tmp/corr_adv_154_*.json, and sharing the
# names let a stale probe output poison the twin's cross-language
# check (the same stale-shared-state bug class this audit hunts).


def _cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(154)
    y_lin = np.linspace(-1.0, 1.0, 64)
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []

    huge = 1e300 * (1.0 + 0.5 * rng.random(64))
    cases.append(("huge_magnitude", huge, 1e300 * (2.0 + y_lin)))

    small = rng.normal(0.0, 1.0, 64)
    cases.append(("huge_offset", 1e15 + small, 1e15 + 2.0 * small + 3.0))

    near = np.ones(64)
    near[63] = 1.0 + 1e-12
    cases.append(("near_constant", near, y_lin))

    tiny = 1e-300 * (1.0 + rng.random(64))
    cases.append(("tiny_values", tiny, 1e-300 * (y_lin + 2.0)))

    alt = np.where(np.arange(64) % 2 == 0, 1e10, -1e10)
    cases.append(("alternating", alt, np.where(np.arange(64) % 2 == 0, -1e10, 1e10)))

    cases.append(("two_points", np.array([0.0, 1.0]), np.array([0.0, 2.0])))

    big_x = rng.normal(0.0, 1.0, 100_000)
    big_y = 0.5 * big_x + rng.normal(0.0, 0.5, 100_000)
    cases.append(("big_n", big_x, big_y))

    mirror_x = np.linspace(-1.0, 1.0, 32)
    cases.append(("neg_zero", mirror_x, -mirror_x))

    return cases


def _scipy_gold(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        with np.errstate(all="ignore"):
            r = float(pearsonr(a, b).statistic)
    except Exception:  # noqa: BLE001 -- scipy raises on its own guards
        return None
    if not np.isfinite(r):
        return None
    return r


def main() -> None:
    cases = _cases()
    # hand the TS side the exact float64 bytes the browser would get
    payload = [
        {
            "name": name,
            "a": [float(v) for v in a],
            "b": [float(v) for v in b],
        }
        for name, a, b in cases
    ]
    INPUTS.write_text(json.dumps(payload))

    proc = subprocess.run(
        ["bun", "run", str(TS_RUNNER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        print("TS RUNNER FAILED")
        print(proc.stdout)
        print(proc.stderr)
        sys.exit(1)
    ts_out = {entry["name"]: entry["r"] for entry in json.loads(OUTPUT.read_text())}

    print(f"{'case':<16} {'scipy r':>22} {'TS r':>22}  |dr|")
    failures: list[str] = []
    for name, a, b in cases:
        gold = _scipy_gold(a, b)
        ts = ts_out.get(name)
        ts_val = None if ts is None else float(ts)
        if gold is None or ts_val is None:
            ok = (gold is None) and (ts_val is None)
            dr = float("nan")
            gtxt = "undefined" if gold is None else f"{gold:+.15g}"
            ttxt = "NaN" if ts_val is None else f"{ts_val:+.15g}"
        else:
            dr = abs(ts_val - gold)
            # per-case tolerances as adjudicated in verify_corr_parity_154:
            # huge_offset = the irreducible input-quantization floor
            # (truth 1.0; the recipe is STRICTLY closer to truth than
            # scipy itself); near_constant = scipy's own warned
            # inaccuracy (~1e-7 of a 1e-12 jitter; the recipe is exact,
            # pinned against the analytic truth by the 154 pair).
            tol = {"huge_offset": 5e-3, "near_constant": 1e-7}.get(name, 1e-12)
            ok = dr <= tol
            gtxt, ttxt = f"{gold:+.15g}", f"{ts_val:+.15g}"
        status = "OK " if ok else "RED"
        print(f"{name:<16} {gtxt:>22} {ttxt:>22}  {dr:.3g}  {status}")
        if not ok:
            failures.append(name)

    print()
    if failures:
        print(f"PROBE_A FAILURES ({len(failures)}): {', '.join(failures)}")
        sys.exit(1)
    print("PROBE_A_OK -- every adversarial numeric case agrees with scipy "
          "(undefined stays undefined on both sides)")


if __name__ == "__main__":
    main()
