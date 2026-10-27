"""100x-154 — Pearson correlation parity, ADVERSARIAL numerics grid.

The 100x-153 pair pinned the shipped TS ``pearsonCorrelation``
against scipy on a well-conditioned seeded grid.  The round-37 audit
(probe_audit_154_a) found the naive two-pass the round-36 version
shipped BREAKS exactly where that grid was silent:

* 1e300-scale varying columns -- deviation products overflow to
  Infinity -> NaN (scipy: defined r) -- the feature silently drops;
* 1e-300-scale varying columns -- squares underflow to 0 -> NaN
  (scipy: defined r) -- the feature silently drops;
* a 1e300-scale CONSTANT column -- the mean's ULP-scale rounding
  error makes the "zero" deviations ~1e284, their squares overflow,
  num/Infinity = 0 -- a FAKE r = 0 the docstring promised never
  happens (scipy: NaN + ConstantInputWarning).

The fix (both this pair's mirror and the shipped TS) is the
offset-then-scale recipe: re-centre on the first element (exact for
same-magnitude values, so a constant column collapses to exact zeros
at ANY scale) and re-scale by the max deviation (r is invariant
under positive affine rescaling; every product stays finite).

This pair re-runs the adversarial grid through BOTH sides of the fix:

1. emits the grid + the scipy gold to /tmp/corr_adv_154_in.json +
   /tmp/corr_adv_154_ref.json,
2. mirrors the SHIPPED recipe line-by-line in Python and asserts the
   mirror against scipy -- 1e-12 everywhere EXCEPT huge_offset, where
   no float64 path recovers the analytic truth (the input is already
   quantized at construction: ULP(1e15) = 0.125 vs jitter ~1); there
   the tolerance is 5e-3 vs scipy and 2e-3 vs the analytic truth
   (the recipe is STRICTLY closer to truth than scipy itself --
   probed: truth 1.0, scipy 0.9968, recipe 0.9990),
3. if /tmp/corr_adv_154_ts.json exists (the TS twin already ran),
   prints the cross-language verdict with the SAME per-case
   tolerances, and the fake-zero pins (const_huge / const_tiny)
   must be NaN on BOTH sides -- never a fabricated 0.

Run: python scratch/verify_corr_parity_154.py
(then the TS twin: bun run scratch/verify_corr_parity_154.ts)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

INPUTS = Path("/tmp/corr_adv_154_in.json")
REF = Path("/tmp/corr_adv_154_ref.json")
TS_OUT = Path("/tmp/corr_adv_154_ts.json")

#: per-case tolerance overrides: the huge-offset case class has an
#: irreducible input-quantization floor no float64 path beats; the
#: near-constant case is scipy's OWN NearConstantInputWarning
#: territory (scipy's centering loses ~1e-7 of a 1e-12 jitter -- the
#: recipe's first-element offset recovers it exactly, so the case is
#: pinned against the ANALYTIC truth below, and the scipy comparison
#: gets scipy's own warned accuracy budget).
TOL_DEFAULT = 1e-12
TOL_OVERRIDES: dict[str, float] = {"huge_offset": 5e-3, "near_constant": 1e-7}
#: analytic truths the recipe must stay close to even where scipy
#: itself drifts (both sides of the pair enforce these).
TRUTHS: dict[str, tuple[float, float]] = {"huge_offset": (1.0, 2e-3)}


def _near_constant_truth() -> float:
    """The exact r of (indicator-of-last-row, y_lin) -- the case's data
    after exact offset removal is [0]*63 + [1e-12], and r is invariant
    under positive affine rescaling: this IS the case's r, computable
    in benign float64 (no 1e-12-on-1 cancellation anywhere)."""
    y = np.linspace(-1.0, 1.0, 64)
    x = np.zeros(64)
    x[63] = 1.0
    xm = x - x.mean()
    ym = y - y.mean()
    return float((xm * ym).sum() / np.sqrt((xm * xm).sum() * (ym * ym).sum()))


TRUTHS["near_constant"] = (_near_constant_truth(), 1e-12)


def _grid() -> list[dict[str, object]]:
    rng = np.random.default_rng(154)
    y_lin = np.linspace(-1.0, 1.0, 64).tolist()
    small = rng.normal(0.0, 1.0, 64)
    cases: list[dict[str, object]] = [
        {
            "name": "huge_magnitude",
            "a": (1e300 * (1.0 + 0.5 * rng.random(64))).tolist(),
            "b": (1e300 * (2.0 + np.linspace(-1.0, 1.0, 64))).tolist(),
        },
        {
            "name": "huge_offset",
            "a": (1e15 + small).tolist(),
            "b": (1e15 + 2.0 * small + 3.0).tolist(),
        },
        {
            "name": "near_constant",
            "a": [1.0 if i < 63 else 1.0 + 1e-12 for i in range(64)],
            "b": y_lin,
        },
        {
            "name": "tiny_values",
            "a": (1e-300 * (1.0 + rng.random(64))).tolist(),
            "b": (1e-300 * (np.linspace(-1.0, 1.0, 64) + 2.0)).tolist(),
        },
        {
            "name": "alternating",
            "a": [1e10 if i % 2 == 0 else -1e10 for i in range(64)],
            "b": [-1e10 if i % 2 == 0 else 1e10 for i in range(64)],
        },
        {"name": "two_points", "a": [0.0, 1.0], "b": [0.0, 2.0]},
        {
            "name": "big_n",
            "a": rng.normal(0.0, 1.0, 100_000).tolist(),
            "b": (0.5 * rng.normal(0.0, 1.0, 100_000) + rng.normal(0.0, 0.5, 100_000)).tolist(),
        },
        {"name": "neg_zero", "a": y_lin, "b": [-v for v in y_lin]},
        # the fake-zero pins: constant columns at extreme scales MUST
        # be NaN on both sides -- never the round-36 fabricated 0.
        {"name": "const_huge", "a": [1e300] * 38, "b": [float(i) for i in range(38)]},
        {"name": "const_tiny", "a": [1e-300] * 38, "b": [float(i) for i in range(38)]},
        {"name": "const_plain", "a": [5.0] * 38, "b": [float(i) for i in range(38)]},
        {"name": "length_mismatch", "a": [1.0, 2.0, 3.0], "b": [1.0, 2.0]},
        {"name": "one_point", "a": [1.0], "b": [2.0]},
        {"name": "nan_pair", "a": [1.0, float("nan"), 3.0], "b": [1.0, 2.0, 3.0]},
    ]
    return cases


def _mirror_recipe(a: list[float], b: list[float]) -> float:
    """The SHIPPED offset-then-scale recipe, line-by-line in Python."""
    if len(a) != len(b):
        return float("nan")
    n = len(a)
    if n < 2:
        return float("nan")
    for i in range(n):
        if not (math.isfinite(a[i]) and math.isfinite(b[i])):
            return float("nan")
    oa = a[0]
    ob = b[0]
    max_a = 0.0
    max_b = 0.0
    aa = [0.0] * n
    bb = [0.0] * n
    for i in range(n):
        aa[i] = a[i] - oa
        bb[i] = b[i] - ob
        if abs(aa[i]) > max_a:
            max_a = abs(aa[i])
        if abs(bb[i]) > max_b:
            max_b = abs(bb[i])
    if max_a == 0.0 or max_b == 0.0:
        return float("nan")  # constant: undefined, never zero
    # scale, THEN subtract the scaled mean: the offset alone leaves the
    # n*(mean_a - a0)*(mean_b - b0) cross term -- centering is required.
    scaled_a = [v / max_a for v in aa]
    scaled_b = [v / max_b for v in bb]
    ma = sum(scaled_a) / n
    mb = sum(scaled_b) / n
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(n):
        xa = scaled_a[i] - ma
        yb = scaled_b[i] - mb
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


def _tolerance(name: str) -> float:
    return TOL_OVERRIDES.get(name, TOL_DEFAULT)


def main() -> None:
    cases = _grid()
    gold: list[dict[str, object]] = []
    worst_mirror = 0.0
    failures: list[str] = []
    for case in cases:
        name = str(case["name"])
        a = case["a"]  # type: ignore[assignment]
        b = case["b"]  # type: ignore[assignment]
        ref = _scipy_r(a, b)
        mirror = _mirror_recipe(a, b)
        if math.isnan(ref) or math.isnan(mirror):
            if not (math.isnan(ref) and math.isnan(mirror)):
                failures.append(f"{name}: undefined mismatch ref={ref} mirror={mirror}")
        else:
            dr = abs(ref - mirror)
            tol = _tolerance(name)
            if dr > tol:
                failures.append(f"{name}: |mirror-scipy|={dr:.3g} > {tol:.1g}")
            if name not in TOL_OVERRIDES:
                worst_mirror = max(worst_mirror, dr)
        if name in TRUTHS:
            truth, truth_tol = TRUTHS[name]
            if math.isnan(mirror) or abs(mirror - truth) > truth_tol:
                failures.append(
                    f"{name}: mirror={mirror} vs analytic truth {truth} (tol {truth_tol})"
                )
        gold.append(
            {
                "name": name,
                "r": None if math.isnan(ref) else ref,
                **(
                    {"truth": TRUTHS[name][0], "truthTol": TRUTHS[name][1]}
                    if name in TRUTHS
                    else {}
                ),
            }
        )
    INPUTS.write_text(
        json.dumps(
            [
                {"name": case["name"], "a": _json_safe(case["a"]), "b": _json_safe(case["b"])}  # type: ignore[index]
                for case in cases
            ]
        )
    )
    REF.write_text(json.dumps(gold))

    print(f"mirror-vs-scipy worst |dr| (default-tolerance cases): {worst_mirror:.3g}")
    for name, (truth, truth_tol) in TRUTHS.items():
        mirror = _mirror_recipe(
            next(c["a"] for c in cases if c["name"] == name),  # type: ignore[arg-type]
            next(c["b"] for c in cases if c["name"] == name),  # type: ignore[arg-type]
        )
        print(f"{name}: recipe={mirror:+.6f} vs analytic truth {truth} (tol {truth_tol})")

    if failures:
        print("PY_SIDE_FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)

    if TS_OUT.exists():
        ts_out = {entry["name"]: entry["r"] for entry in json.loads(TS_OUT.read_text())}
        worst_ts = 0.0
        for case in cases:
            name = str(case["name"])
            ref = next(g["r"] for g in gold if g["name"] == name)
            ts = ts_out.get(name)
            ts_val = float("nan") if ts is None else float(ts)
            if ref is None or math.isnan(ts_val):
                if not (ref is None and math.isnan(ts_val)):
                    failures.append(f"{name}: TS undefined mismatch ts={ts_val} scipy={ref}")
            else:
                dr = abs(ts_val - float(ref))
                if dr > _tolerance(name):
                    failures.append(f"{name}: |ts-scipy|={dr:.3g} > {_tolerance(name):.1g}")
                if name not in TOL_OVERRIDES:
                    worst_ts = max(worst_ts, dr)
            if name in TRUTHS:
                truth, truth_tol = TRUTHS[name]
                if math.isnan(ts_val) or abs(ts_val - truth) > truth_tol:
                    failures.append(f"{name}: ts={ts_val} vs analytic truth {truth}")
        print(f"ts-vs-scipy worst |dr| (default-tolerance cases): {worst_ts:.3g}")
        if failures:
            print("CROSS_LANGUAGE_FAILURES:")
            for failure in failures:
                print(f"  {failure}")
            raise SystemExit(1)
        print("CROSS_LANGUAGE_PARITY_OK")
    else:
        print("PY_SIDE_DONE (run the TS twin next: bun run scratch/verify_corr_parity_154.ts)")


if __name__ == "__main__":
    main()
