#!/usr/bin/env python
"""round-272 -- the gate CONTINUATION (the sibling r272 runner died
mid-chunk-4 at 22:49; chunks 2-3 are green in /tmp/suite272.txt, this
script re-runs chunk-1 (its notebook-freshness failure is fixed: the 8
notebooks re-executed at 1.2.216 + the no_output twins stripped) and
completes chunks 4-8 + the nine-card leg 9 + the src-doctest leg 10,
each a fresh venv pytest process, streaming to /tmp/suite272-cont.txt
ending in ``SUITE272C DONE``.  The dev server stays DOWN (the OOM
ritual)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = REPO / "pharma-ml-toolkit"
OUT = Path("/tmp/suite272-cont.txt")
PY = "/home/z/.venv/bin/python"
SENTINEL = "SUITE272C DONE"

CHUNKS = 8
NINE_CARD_SUITE = TOOLKIT / "tests" / "ui" / "test_nine_cards.py"


def balance(files: list[Path], groups: int) -> list[list[Path]]:
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
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-2:])
    return proc.returncode, f"{time.monotonic() - started:.0f}s rc={proc.returncode}\n{tail}"


def main() -> None:
    log = OUT.open("w")
    files = sorted((TOOLKIT / "tests").rglob("test_*.py"))
    assert NINE_CARD_SUITE in files, "the nine-card suite must exist"
    files.remove(NINE_CARD_SUITE)
    groups = balance(files, CHUNKS)
    failures = 0
    for idx in (1, 4, 5, 6, 7, 8):
        group = groups[idx - 1]
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
    log.flush()
    if rc != 0:
        failures += 1
    rc, detail = run([PY, "-m", "pytest", "src/pharmaml", "--doctest-modules",
                      "-q", "-W", "ignore", "--tb=short", "-p", "no:cacheprovider"])
    log.write(f"=== chunk-10 (src doctests): {detail}\n")
    log.flush()
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
