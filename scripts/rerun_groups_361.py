#!/usr/bin/env python3
"""round-361 post-fix group re-runs (the house convention: a decoded
find's group re-runs rc=0 after the source fix).

Leg A: the exact chunk-11 group (tests/api/test_*.py).
Leg B: the chunk-3-equivalent group -- the runner's own balance()
algorithm re-derived over the current tests/ file set (minus api,
minus nine-cards), picking the group that contains
tests/test_leakage.py (the mid-gate byte deltas of the two fixes can
shift one boundary file; the group is the honest equivalent).
Leg C: the chunk-2 OOM casualty drill -- test_nine_card_first.py
split by CLASS, each class solo (the r353 six-class split precedent).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/scratch")
from run_suite_chunks_361 import CHUNKS, NINE_CARD_SUITE, PYTEST, balance, run  # noqa: E402

TOOLKIT = Path("/home/z/my-project/pharma-ml-toolkit")


def main() -> None:
    # Leg A: the api service suite (the exact chunk-11 group).
    api_files = sorted((TOOLKIT / "tests" / "api").glob("test_*.py"))
    rc, detail = run(PYTEST + list(map(str, api_files)))
    print(f"LEG-A tests/api: {detail.splitlines()[0]}")

    # Leg B: the group containing test_leakage.py (the runner's balance).
    files = sorted(p for p in (TOOLKIT / "tests").rglob("test_*.py")
                   if p != NINE_CARD_SUITE and "tests/api" not in str(p))
    groups = balance(files, CHUNKS)
    target = next(g for g in groups
                  if any(f.name == "test_leakage.py" for f in g))
    rc, detail = run(PYTEST + list(map(str, target)))
    print(f"LEG-B chunk3-equivalent ({len(target)} files, "
          f"leakage={any(f.name == 'test_leakage.py' for f in target)}): "
          f"{detail.splitlines()[0]}")

    # Leg C: the nine-card-first class split (the r353 drill).
    import re
    src = NINE_CARD_SUITE.read_text()
    classes = re.findall(r"^class (Test\w+)", src, re.M)
    print(f"LEG-C test_nine_card_first.py: {len(classes)} classes -> solo runs")
    bad = 0
    for cls in classes:
        rc, detail = run(PYTEST + [f"{NINE_CARD_SUITE}::{cls}"])
        first = detail.splitlines()[0]
        print(f"  {cls}: {first}")
        if rc != 0:
            bad += 1
    print(f"LEG-C verdict: {len(classes) - bad}/{len(classes)} classes green")


if __name__ == "__main__":
    main()
