"""100x-153 — Pearson correlation parity, Python side.

The web interpretability twin's pure core (src/lib/ml-math.ts
`pearsonCorrelation`, added this round for the ML playground's
Feature-target correlations card) must agree with scipy.stats.pearsonr.
This probe:

1. emits the FIXED seeded grid (an x-column, a y-column, a shuffled
   pair, an exact line, a noisy monotone pair, a constant column, a
   1-point pair, a NaN/inf-carrying pair, a large-n pair) plus the
   scipy gold to /tmp/corr_inputs_153.json + /tmp/corr_ref_153.json,
2. mirrors the shipped TS formula line-by-line in Python (the two-pass
   mean/covariance form) and asserts the mirror against scipy at
   1e-12 — the mirror is the readability proof the TS is the same math,
3. if /tmp/corr_ts_153.json exists (the TS twin already ran), prints
   the cross-language verdict with the SAME tolerance the TS side
   enforces: max|r_ts - r_scipy| <= 1e-12 on every defined pair, and
   the undefined cases (constant / short / non-finite) must be NaN on
   BOTH sides — never a fake 0.

Run: python scratch/verify_corr_parity_153.py
(then the TS twin: bun run scratch/verify_corr_parity_153.ts)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import pearsonr

INPUTS = Path("/tmp/corr_inputs_153.json")
REF = Path("/tmp/corr_ref_153.json")
TS_OUT = Path("/tmp/corr_ts_153.json")
TOL = 1e-12


def _grid() -> list[dict[str, object]]:
    rng_state = 0x9E3779B97F4A7C15  # the mulberry32 default seed 42 chain start
    value = 42

    def rnd() -> float:
        nonlocal value
        value = (1103515245 * value + 12345) % 2_147_483_648
        return value / 2_147_483_648

    n = 40
    xs = [rnd() for _ in range(n)]
    ys = [3.0 * x + 0.05 * (rnd() - 0.5) for x in xs]
    cases: list[dict[str, object]] = [
        {"name": "line", "a": xs, "b": [2.0 * v + 1.0 for v in xs]},
        {"name": "noisy", "a": xs, "b": ys},
        {"name": "inverse", "a": xs, "b": [-v for v in xs]},
        {"name": "shuffled", "a": xs, "b": [ys[(i * 7 + 3) % n] for i in range(n)]},
        {"name": "constant", "a": xs, "b": [5.0] * n},
        {"name": "one_point", "a": [1.0], "b": [2.0]},
        {"name": "empty", "a": [], "b": []},
        {
            "name": "nan_pair",
            "a": [1.0, 2.0, float("nan"), 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
        },
        {
            "name": "inf_pair",
            "a": [1.0, 2.0, float("inf"), 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
        },
        {"name": "length_mismatch", "a": xs, "b": ys[: n - 1]},
        {"name": "large_n", "a": [rnd() for _ in range(5000)], "b": [rnd() for _ in range(5000)]},
    ]
    assert rng_state == 0x9E3779B97F4A7C15  # documented, unused beyond provenance
    return cases


def _mirror_ts(a: list[float], b: list[float]) -> float:
    """The shipped TS formula, line-by-line in Python (the mirror)."""
    if len(a) != len(b):
        return float("nan")  # mismatched: no honest r (scipy raises here)
    n = len(a)
    if n < 2:
        return float("nan")
    sa = 0.0
    sb = 0.0
    for i in range(n):
        x = a[i]
        y = b[i]
        if not (math.isfinite(x) and math.isfinite(y)):
            return float("nan")
        sa += x
        sb += y
    ma = sa / n
    mb = sb / n
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(n):
        xa = a[i] - ma
        yb = b[i] - mb
        num += xa * yb
        da += xa * xa
        db += yb * yb
    den = math.sqrt(da * db)
    if den == 0.0:
        return float("nan")
    return num / den


def _scipy_r(a: list[float], b: list[float]) -> float:
    """scipy gold: NaN where scipy refuses or the inputs are non-finite."""
    try:
        r = float(pearsonr(a, b).statistic)
    except Exception:  # noqa: BLE001 — scipy's refusal IS the verdict
        return float("nan")
    return r if math.isfinite(r) else float("nan")


def _json_safe(values: list[float]) -> list[object]:
    """NaN/inf are not valid JSON — encode them as string sentinels the
    TS twin decodes back (Number.NaN / Infinity / -Infinity)."""
    out: list[object] = []
    for v in values:
        if math.isnan(v):
            out.append("NaN")
        elif v == math.inf:
            out.append("Infinity")
        elif v == -math.inf:
            out.append("-Infinity")
        else:
            out.append(v)
    return out


def main() -> None:
    cases = _grid()
    gold: list[dict[str, object]] = []
    worst_mirror = 0.0
    for case in cases:
        a = case["a"]  # type: ignore[assignment]
        b = case["b"]  # type: ignore[assignment]
        ref = _scipy_r(a, b)
        mirror = _mirror_ts(a, b)
        if math.isnan(ref) or math.isnan(mirror):
            assert math.isnan(ref) and math.isnan(mirror), (
                f"{case['name']}: undefined mismatch ref={ref} mirror={mirror}"
            )
        else:
            worst_mirror = max(worst_mirror, abs(ref - mirror))
        gold.append({"name": case["name"], "r": None if math.isnan(ref) else ref})
    INPUTS.write_text(
        json.dumps(
            [
                {"name": case["name"], "a": _json_safe(case["a"]), "b": _json_safe(case["b"])}
                for case in cases
            ]
        )
    )
    REF.write_text(json.dumps(gold))
    print(f"mirror-vs-scipy worst |d| = {worst_mirror:.3e} (tol {TOL:.1e})")
    assert worst_mirror <= TOL, f"the Python mirror drifted from scipy: {worst_mirror:.3e}"

    if TS_OUT.exists():
        ts = json.loads(TS_OUT.read_text())
        worst = 0.0
        for entry, ref_entry in zip(ts, gold):
            ts_r = entry["r"]
            ref_r = ref_entry["r"]
            if ts_r is None or ref_r is None:
                assert ts_r is None and ref_r is None, (
                    f"{ref_entry['name']}: undefined mismatch ts={ts_r} scipy={ref_r}"
                )
                continue
            worst = max(worst, abs(float(ts_r) - float(ref_r)))
        print(f"ts-vs-scipy worst |d| = {worst:.3e} over {len(gold)} case(s)")
        assert worst <= TOL, f"the SHIPPED TS drifted from scipy: {worst:.3e}"
        print("CROSS_LANGUAGE_PARITY_OK")
    else:
        print("PY_SIDE_DONE (run the TS twin for the cross-language verdict)")


if __name__ == "__main__":
    main()
