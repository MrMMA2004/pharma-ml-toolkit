#!/usr/bin/env python
"""round-271 -- isolate the chunk-7 OOM: run its 28 files as FOUR fresh
pytest sub-batches (7 files each), logging rc + duration to
/tmp/suite271-isolate.txt.  The dying sub-batch gets narrowed next."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

TOOLKIT = Path("/home/z/my-project/pharma-ml-toolkit")
OUT = Path("/tmp/suite271-isolate.txt")
PY = "/home/z/.venv/bin/python"


def balance(files, groups):
    weighted = sorted(((f.stat().st_size, f) for f in files), key=lambda t: -t[0])
    bins = [[] for _ in range(groups)]
    sizes = [0] * groups
    for size, f in weighted:
        i = sizes.index(min(sizes))
        bins[i].append(f)
        sizes[i] += size
    return [sorted(b) for b in bins]


def main() -> None:
    files = sorted((TOOLKIT / "tests").rglob("test_*.py"))
    files.remove(TOOLKIT / "tests" / "ui" / "test_nine_cards.py")
    g7 = sorted(balance(files, 8)[6])
    log = OUT.open("w")
    for i in range(0, len(g7), 7):
        sub = g7[i:i + 7]
        label = f"7.{i // 7 + 1} ({len(sub)} files)"
        log.write(f"=== sub-{label}: start\n")
        log.flush()
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [PY, "-m", "pytest", "-q", "-W", "ignore", "--tb=short",
                 "-p", "no:cacheprovider", *map(str, sub)],
                cwd=TOOLKIT, capture_output=True, text=True, timeout=900,
            )
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-2:])
            log.write(f"=== sub-{label}: {time.monotonic()-t0:.0f}s rc={proc.returncode}\n{tail}\n")
        except subprocess.TimeoutExpired:
            log.write(f"=== sub-{label}: TIMEOUT after {time.monotonic()-t0:.0f}s\n")
        log.flush()
    log.write("ISOLATE DONE\n")
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
