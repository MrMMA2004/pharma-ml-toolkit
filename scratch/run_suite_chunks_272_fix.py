#!/usr/bin/env python
"""round-272 -- the chunk-6 diagnosis re-run + the chunk-7 sub-batch
re-run, post-conftest-fix, on the CURRENT tree.  Streams to
/tmp/suite272-fix.txt ending in ``SUITE272F DONE``.  Full failure
lines are preserved (the r272 runner's 2-line tail window truncated
chunk-6's FAILED line)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = REPO / "pharma-ml-toolkit"
OUT = Path("/tmp/suite272-fix.txt")
PY = "/home/z/.venv/bin/python"


def balance(files: list[Path], groups: int) -> list[list[Path]]:
    weighted = sorted(((f.stat().st_size, f) for f in files), key=lambda t: -t[0])
    bins: list[list[Path]] = [[] for _ in range(groups)]
    sizes = [0] * groups
    for size, f in weighted:
        i = sizes.index(min(sizes))
        bins[i].append(f)
        sizes[i] += size
    return [sorted(b) for b in bins]


def run(files: list[Path], label: str) -> int:
    log = OUT.open("a")
    log.write(f"=== {label}: start\n")
    log.flush()
    t0 = time.monotonic()
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "-W", "ignore", "--tb=line",
         "-p", "no:cacheprovider", *map(str, files)],
        cwd=TOOLKIT, capture_output=True, text=True, timeout=3000,
    )
    body = (proc.stdout + "\n" + proc.stderr)
    keep = [l for l in body.splitlines()
            if ("FAILED" in l or "ERROR" in l or "passed" in l
                or "failed" in l or "no tests ran" in l)]
    log.write(f"=== {label}: {time.monotonic()-t0:.0f}s rc={proc.returncode}\n")
    for l in keep[:25]:
        log.write("    " + l[:400] + "\n")
    log.flush()
    log.close()
    return proc.returncode


def main() -> None:
    OUT.write_text("")
    files = sorted((TOOLKIT / "tests").rglob("test_*.py"))
    files.remove(TOOLKIT / "tests" / "ui" / "test_nine_cards.py")
    groups = balance(files, 8)
    failures = 0
    # chunk-6, full-output diagnosis:
    if run(groups[5], "chunk-6 (29 files, full output)") != 0:
        failures += 1
    # chunk-7 as the four fresh sub-batches (post fixture fix):
    g7 = groups[6]
    for i in range(0, len(g7), 7):
        sub = g7[i:i + 7]
        if run(sub, f"chunk-7.{i // 7 + 1} ({len(sub)} files)") != 0:
            failures += 1
    log = OUT.open("a")
    log.write(f"failures={failures}\nSUITE272F DONE\n")
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
