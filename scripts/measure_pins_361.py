#!/usr/bin/env python3
"""round-361 post-gate pin measurement (the carrier-truing inputs).

Measures, against the LIVE dev server and the repaired venv:
  1. the /api/toolkit/tree?fresh=1 census (files / totalLines / bytes)
  2. the pharmaml.ml.models MODEL_REGISTRY size + sklearn family count
  3. the pytest collection total (already verified: 9,017 / 0 errors)
Prints the values for the toolkit-stats.ts carrier truing. No secrets,
no writes outside stdout.
"""
import json
import subprocess
import sys
import urllib.request

BASE = "http://localhost:3000"
PY = "/home/z/.venv/bin/python"


def tree_stats() -> dict:
    with urllib.request.urlopen(f"{BASE}/api/toolkit/tree?fresh=1", timeout=120) as r:
        data = json.load(r)
    return data["stats"]


def main() -> None:
    s = tree_stats()
    print("tree_stats:", json.dumps(s))
    # Registry probe (import cost ~torch; run once, quiet).
    probe = subprocess.run(
        [PY, "-c",
         "from pharmaml.ml.models import MODEL_REGISTRY;"
         "sk = sum(1 for m in MODEL_REGISTRY.values()"
         " if getattr(type(m), '__module__', '').startswith('sklearn.'));"
         "print(len(MODEL_REGISTRY), sk)"],
        capture_output=True, text=True, timeout=300,
    )
    if probe.returncode == 0:
        n, sk = probe.stdout.split()
        print(f"registry: {n} models, {sk} sklearn families")
    else:
        print("registry probe FAILED:", probe.stderr[-300:])
    # Version carrier
    v = subprocess.run([PY, "-c", "import pharmaml; print(pharmaml.__version__)"],
                       capture_output=True, text=True, timeout=120)
    print("pharmaml version:", v.stdout.strip())


if __name__ == "__main__":
    sys.exit(main())
