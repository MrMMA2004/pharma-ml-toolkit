"""PROBE audit_154_b -- the live /api/ml/train correlation bundle (100x-154).

Round-36's web twin audited from the OUTSIDE (the route the browser
calls, not the source): five questions the parity grid never asked.

  classification_synth   does a synthetic classification run carry the
                         bundle?  (The desktop twin serves
                         classification runs -- point-biserial r --
                         via signed_feature_correlations with NO task
                         guard; a missing web bundle is a parity gap.)
  classification_csv     same question for the 2-feature CSV path.
  constant_column_csv    a constant feature column: dropped from the
                         bundle (undefined, never a fake 0) and the
                         matrix cell honest?
  eight_features_csv     8 features (the schema's max): the matrix
                         truncates to top-7 + target = 8 columns?
  huge_scale_csv         a 1e300-scale feature column: the current
                         pearsonCorrelation overflows internally and
                         DROPS the feature the desktop twin reports --
                         probe A's RED, live in the route.

Run (dev server on :3000):
  python scratch/probe_audit_154_b.py
"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://localhost:3000/api/ml/train"


def post(body: dict) -> dict:
    req = urllib.request.Request(
        BASE,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read())


def show(name: str, resp: dict) -> dict | None:
    bundle = resp.get("correlations")
    if bundle is None:
        print(f"{name:<24} NO BUNDLE")
        return None
    corrs = bundle.get("correlations", [])
    matrix = bundle.get("matrix")
    top = ", ".join(f"{c['name']}:{c['r']:+.4f}" for c in corrs[:8])
    n_matrix = len(matrix["columns"]) if matrix else 0
    print(f"{name:<24} bundle: nRows={bundle.get('nRows')} target={bundle.get('target')!r}")
    print(f"{'':<24}   correlations=[{top}]")
    print(f"{'':<24}   matrix columns={n_matrix}")
    return bundle


def main() -> None:
    print("== classification, synthetic moons ==")
    resp = post({
        "task": "classification", "model": "logistic", "samples": 200,
        "noise": 0.3, "testSize": 0.25, "seed": 42, "cv": 0,
    })
    show("classification_synth", resp)

    print("== classification, 2-feature CSV ==")
    csv_cls = "\n".join(
        f"{0.3 * i},{0.7 * (i % 7)},{i % 2}" for i in range(120)
    )
    resp = post({
        "task": "classification", "model": "logistic", "testSize": 0.25,
        "seed": 42, "cv": 0,
        "csv": csv_cls, "csvName": "probe_cls.csv",
        "featureColumn": "f1", "featureColumn2": "f2", "targetColumn": "label",
    })
    show("classification_csv", resp)

    print("== regression, 8-feature CSV, 'a' constant ON THE TEST SPLIT ==")
    # 'a' varies on exactly two rows (the parse-layer constant guard
    # passes); whether the held-out split keeps the variation is the
    # seed's call -- the probe tries seeds until the variation lands in
    # TRAIN, so the test rows see a constant column: the bundle must
    # DROP it (undefined on the held-out rows), never zero it.
    varied = None
    for seed in (42, 7, 11, 3, 5):
        # deterministic pseudo-random columns (an LCG in the probe, so
        # no two columns are collinear and the linear fit stays solvable)
        state = 12345
        cols = []
        for _ in range(8):
            state = (1103515245 * state + 12345) % (2 ** 31)
            cols.append([(state := (1103515245 * state + 12345) % (2 ** 31)) % 100 / 100.0
                         for _ in range(150)])
        # column 'a': constant everywhere except rows 3 and 17
        cols[0] = [1.0 if i not in (3, 17) else 2.0 for i in range(150)]
        rows = [",".join(f"{v:.4f}" for v in (cols[j][i] for j in range(8))) + f",{i / 10.0:.1f}"
                for i in range(150)]
        header = "a,b,c,d,e,f,g,h,y"
        csv8 = header + "\n" + "\n".join(rows)
        resp = post({
            "task": "regression", "model": "linear", "testSize": 0.25,
            "seed": seed, "cv": 0,
            "csv": csv8, "csvName": "probe8.csv", "targetColumn": "y",
            "featureColumns": ["a", "b", "c", "d", "e", "f", "g", "h"],
        })
        bundle = resp.get("correlations") or {}
        names = [c["name"] for c in bundle.get("correlations", [])]
        if "a" not in names:
            varied = seed
            break
    print(f"{'':<24} (test-split-constant 'a' dropped at seed {varied})")
    bundle = show("eight_features_csv", resp)
    if bundle is not None and varied is not None:
        names = [c["name"] for c in bundle["correlations"]]
        print(f"{'':<24}   dropped-on-held-out 'a' in correlations? {'a' in names} "
              f"(must be False -- undefined, never zeroed)")

    print("== regression, 1e300-scale feature CSV (probe A live) ==")
    rows = [f"1e300,{i}" for i in range(150)]
    csv_huge = "big,y\n" + "\n".join(rows)
    resp = post({
        "task": "regression", "model": "linear", "testSize": 0.25,
        "seed": 42, "cv": 0,
        "csv": csv_huge, "csvName": "probe_huge.csv",
        "featureColumn": "big", "targetColumn": "y",
    })
    bundle = show("huge_scale_csv", resp)
    if bundle is not None:
        names = [c["name"] for c in bundle["correlations"]]
        print(f"{'':<24}   'big' in correlations? {'big' in names} "
              f"(dropped => the overflow RED, live)")

    print("PROBE_B_DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 -- the probe reports, never raises
        print(f"PROBE_B_ERROR: {exc}")
        sys.exit(1)
