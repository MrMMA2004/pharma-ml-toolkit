#!/usr/bin/env python
"""round-263 -- the TEN-chunk gate runner (reconstruction of the lost
r260-r262 runner; scratch/ came back from the rollback-#31 restore at
the round-249 shape, so the ten-chunk structure is rebuilt here).

Design (the r262 shape): collect tests/**/test_*.py recursively, pull
the nine-card suite out as its own Qt process envelope (chunk 9 -- it
renders the full grid + molecule pixmaps), line-balance the remaining
111 files into EIGHT chunks, then run the src-doctest chunk.  Ten
chunks total, each a fresh venv pytest process, streamed to
/tmp/suite263.txt ending in the sentinel line ``SUITE263 DONE``.

The dev server must be DOWN before launching (the OOM ritual).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = REPO / "pharma-ml-toolkit"
OUT = Path("/tmp/suite263.txt")
PY = "/home/z/.venv/bin/python"
SENTINEL = "SUITE263 DONE"

CHUNKS = 8
NINE_CARD_SUITE = TOOLKIT / "tests" / "ui" / "test_nine_cards.py"


def balance(files: list[Path], groups: int) -> list[list[Path]]:
    """Greedy line-balanced grouping (largest file first)."""
    weighted = sorted(
        ((f.stat().st_size, f) for f in files), key=lambda t: -t[0]
    )
    bins: list[list[Path]] = [[] for _ in range(groups)]
    sizes = [0] * groups
    for size, f in weighted:
        i = sizes.index(min(sizes))
        bins[i].append(f)
        sizes[i] += size
    return [sorted(b) for b in bins]


def run(cmd: list[str]) -> tuple[int, str]:
    started = time.monotonic()
    proc = subprocess.run(
        cmd, cwd=TOOLKIT, capture_output=True, text=True, timeout=3000
    )
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-3:])
    return proc.returncode, f"{time.monotonic() - started:.0f}s rc={proc.returncode}\n{tail}"


def main() -> None:  # child after double-fork
    log = OUT.open("w")
    files = sorted((TOOLKIT / "tests").rglob("test_*.py"))
    assert NINE_CARD_SUITE in files, "the nine-card suite must exist"
    files.remove(NINE_CARD_SUITE)
    groups = balance(files, CHUNKS)
    total_passed = 0
    failures = 0
    for idx, group in enumerate(groups, 1):
        label = f"chunk-{idx} ({len(group)} files)"
        log.write(f"=== {label}: start\n")
        log.flush()
        rc, detail = run([PY, "-m", "pytest", "-q", "-W", "ignore", "--tb=short",
                          "-p", "no:cacheprovider", *map(str, group)])
        log.write(f"=== {label}: {detail}\n")
        log.flush()
        if rc != 0:
            failures += 1
    rc, detail = run([PY, "-m", "pytest", str(NINE_CARD_SUITE), "-q", "-W", "ignore",
                      "--tb=short", "-p", "no:cacheprovider"])
    log.write(f"=== chunk-9 (the nine-card suite, own Qt envelope): {detail}\n")
    if rc != 0:
        failures += 1
    rc, detail = run([PY, "-m", "pytest", "src/pharmaml", "--doctest-modules",
                      "-q", "-W", "ignore", "--tb=short", "-p", "no:cacheprovider"])
    log.write(f"=== chunk-10 (src doctests): {detail}\n")
    if rc != 0:
        failures += 1
    log.write(f"failures={failures}\n{SENTINEL}\n")
    log.close()


def daemonize() -> None:
    if os.fork():
        sys.exit(0)
    os.setsid()
    if os.fork():
        sys.exit(0)
    sys.stdout.close()
    sys.stderr.close()
    main()


if __name__ == "__main__":
    daemonize()
