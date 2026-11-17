#!/usr/bin/env python
"""round-273 -- the TEN-chunk gate runner (the r272 recipe verbatim,
plus the OOM-resilience the r271 narrowing earned): a chunk killed by
the container's memory limit (rc=-9, SIGKILL) is re-run as fresh
7-file pytest sub-batches, and any sub-batch still SIGKILLed is re-run
file-by-file.  Nothing is skipped; every leg that claims green is a
real pytest exit 0.  The split is recorded in the log so the ritual
stays honest about HOW a green chunk was produced.

Ten legs: 8 line-balanced chunks (greedy largest-first, the r263
balance()), the nine-card suite as its own Qt envelope (leg 9), the
src doctests (leg 10).  Streams to /tmp/suite273.txt ending in the
sentinel line ``SUITE273 DONE``.

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
OUT = Path("/tmp/suite273.txt")
PY = "/home/z/.venv/bin/python"
SENTINEL = "SUITE273 DONE"

CHUNKS = 8
NINE_CARD_SUITE = TOOLKIT / "tests" / "ui" / "test_nine_cards.py"
PYTEST = [PY, "-m", "pytest", "-q", "-W", "ignore", "--tb=short",
          "-p", "no:cacheprovider"]


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


def run_group(log, label: str, group: list[Path]) -> int:
    """Run a group; on SIGKILL (rc=-9) degrade to 7-file sub-batches,
    then singles.  Returns 0 iff every leg is green."""
    log.write(f"=== {label}: start\n")
    log.flush()
    rc, detail = run(PYTEST + list(map(str, group)))
    log.write(f"=== {label}: {detail}\n")
    log.flush()
    if rc == 0:
        return 0
    if rc != -9:
        return 1  # a real failure, not the container's OOM kill
    # the OOM path: fresh sub-batches of 7, then singles per dying batch
    log.write(f"=== {label}: SIGKILLed -- the OOM split begins\n")
    log.flush()
    ok = True
    for b in range(0, len(group), 7):
        batch = group[b:b + 7]
        blabel = f"{label} sub-batch {b // 7 + 1} ({len(batch)} files)"
        log.write(f"=== {blabel}: start\n")
        log.flush()
        rc, detail = run(PYTEST + list(map(str, batch)))
        log.write(f"=== {blabel}: {detail}\n")
        log.flush()
        if rc == 0:
            continue
        if rc != -9:
            ok = False
            continue
        for f in batch:
            flabel = f"{label} single {f.name}"
            log.write(f"=== {flabel}: start\n")
            log.flush()
            rc, detail = run(PYTEST + [str(f)])
            log.write(f"=== {flabel}: {detail}\n")
            log.flush()
            if rc != 0:
                ok = False
    return 0 if ok else 1


def main() -> None:  # child after double-fork
    log = OUT.open("w")
    files = sorted((TOOLKIT / "tests").rglob("test_*.py"))
    assert NINE_CARD_SUITE in files, "the nine-card suite must exist"
    files.remove(NINE_CARD_SUITE)
    groups = balance(files, CHUNKS)
    failures = 0
    for idx, group in enumerate(groups, 1):
        failures += run_group(log, f"chunk-{idx} ({len(group)} files)", group)
    rc, detail = run(PYTEST + [str(NINE_CARD_SUITE)])
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
