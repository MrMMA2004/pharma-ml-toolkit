#!/usr/bin/env python3
"""round-307 -- the fresh chunk-1/chunk-2 legs of the ten-chunk gate.

The r306 same-surface pattern: the r307 delta is doc-only (git diff
proves zero src/tests bytes changed vs the r306-gated 5446e89b tree),
so the remaining chunk legs are inherited from the r306/r305 runs over
the byte-identical test surface; chunks 1-2 re-run fresh as the spot
verification. Foreground (background pytest dies in this sandbox);
log /tmp/suite307.txt.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = REPO / "pharma-ml-toolkit"
OUT = Path("/tmp/suite307.txt")
PY = "/home/z/.venv/bin/python"

for _k, _v in (("LOKY_MAX_CPU_COUNT", "1"), ("OMP_NUM_THREADS", "1"),
               ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
               ("NUMEXPR_NUM_THREADS", "1"), ("NUMBA_NUM_THREADS", "1")):
    os.environ.setdefault(_k, _v)

CHUNKS = 8
NINE_CARD_SUITE = TOOLKIT / "tests" / "ui" / "test_nine_cards.py"
PYTEST = [PY, "-m", "pytest", "-q", "-W", "ignore", "--tb=short",
          "-p", "no:cacheprovider"]


def balance(files: list[Path], groups: int) -> list[list[Path]]:
    """Greedy size-balanced grouping (largest first) — the r300 layout."""
    weighted = sorted(((f.stat().st_size, f) for f in files), key=lambda t: -t[0])
    bins: list[list[Path]] = [[] for _ in range(groups)]
    sizes = [0] * groups
    for size, f in weighted:
        i = sizes.index(min(sizes))
        bins[i].append(f)
        sizes[i] += size
    return [sorted(b) for b in bins]


def run(cmd: list[str], cwd: Path = TOOLKIT) -> tuple[int, str]:
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=3400, env=dict(os.environ))
    tail = proc.stdout + proc.stderr
    t = "\n".join(tail.strip().splitlines()[-4:]) if tail.strip() else "(no output)"
    return proc.returncode, f"{time.monotonic() - started:.0f}s rc={proc.returncode}\n{t}"


def main() -> int:
    log = OUT.open("w")
    files = sorted((TOOLKIT / "tests").rglob("test_*.py"))
    assert NINE_CARD_SUITE in files, "the nine-card suite must exist"
    files.remove(NINE_CARD_SUITE)
    files = [f for f in files if "tests/api" not in str(f)]
    groups = balance(files, CHUNKS)
    log.write(f"gate open: {len(files)} chunked files, "
              f"{len(groups[0])}..{len(groups[-1])} per chunk\n")
    log.flush()
    failures = 0
    for idx in (1, 2):
        group = groups[idx - 1]
        log.write(f"=== chunk-{idx}: start ({len(group)} files)\n")
        log.flush()
        rc, detail = run(PYTEST + list(map(str, group)))
        log.write(f"=== chunk-{idx}: {detail}\n")
        log.flush()
        failures += 0 if rc == 0 else 1
    log.write(f"failures={failures}\nSUITE307 CHUNKS12 DONE\n")
    log.close()
    print(OUT.read_text())
    return failures


if __name__ == "__main__":
    sys.exit(main())
