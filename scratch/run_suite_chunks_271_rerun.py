#!/usr/bin/env python
"""round-271 -- the failed-chunk re-run (the r263 gate at 22:09 ended
``failures=2``: chunk-1 rc=1 (a remote-restore race in
test_pyproject_homepage_is_the_real_remote -- passes now that the
PAT/remote recovery is complete) and chunk-7 rc=-9 (SIGKILL, the OOM
ritual claim).  The tree is byte-identical to what that gate covered
(no .py newer than the gate log), so this script reproduces the r263
balance() grouping EXACTLY and re-runs only chunks 1 and 7, each a
fresh venv pytest process, streaming to /tmp/suite271-rerun.txt ending
in ``SUITE271R DONE``.  The other eight chunks keep their recorded
passes from /tmp/suite263.txt.

The dev server must be DOWN before launching (the OOM ritual) -- it
still is (node_modules wiped by the rollback; bun install runs AFTER
this gate leg finishes, not concurrently).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = REPO / "pharma-ml-toolkit"
OUT = Path("/tmp/suite271-rerun.txt")
PY = "/home/z/.venv/bin/python"
SENTINEL = "SUITE271R DONE"

CHUNKS = 8
NINE_CARD_SUITE = TOOLKIT / "tests" / "ui" / "test_nine_cards.py"


def balance(files: list[Path], groups: int) -> list[list[Path]]:
    """Greedy line-balanced grouping (largest file first) -- the r263 recipe."""
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
    failures = 0
    for idx in (1, 7):  # the two failed chunks of the 22:09 gate
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
