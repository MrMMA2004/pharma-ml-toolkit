#!/usr/bin/env python
"""r304: chunks 5-6 re-run, daemonized (double fork) because chunk-5's
theme-screenshot suite exceeds the 600s tool window. Log
/tmp/suite304_rerun56.txt, sentinel RERUN56 DONE."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

sys.path.insert(0, "/home/z/my-project/scratch")
from run_suite_chunks_304 import PYTEST, balance

TOOLKIT = Path("/home/z/my-project/pharma-ml-toolkit")
OUT = Path("/tmp/suite304_rerun56.txt")
SENTINEL = "RERUN56 DONE"


def run_group(log, label: str, group, out_lines: list) -> int:
    log.write(f"=== {label}: start ({len(group)} files)\n")
    log.flush()
    t = time.monotonic()
    proc = subprocess.run(PYTEST + list(map(str, group)), cwd=TOOLKIT,
                          capture_output=True, text=True,
                          timeout=5400, env=dict(os.environ))
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-8:]
    log.write(f"=== {label}: {time.monotonic()-t:.0f}s rc={proc.returncode}\n")
    log.write("\n".join(tail) + "\n")
    log.flush()
    return proc.returncode


def main() -> None:
    log = OUT.open("w")
    files = sorted((TOOLKIT / "tests").rglob("test_*.py"))
    nine = TOOLKIT / "tests" / "ui" / "test_nine_cards.py"
    files.remove(nine)
    files = [f for f in files if "tests/api" not in str(f)]
    groups = balance(files, 8)
    fails = []
    for idx in (5, 6):
        rc = run_group(log, f"chunk-{idx} re-run", groups[idx - 1], [])
        if rc != 0:
            fails.append(idx)
            # real failure decode: singles
            for f in groups[idx - 1]:
                p = subprocess.run(PYTEST + [str(f)], cwd=TOOLKIT,
                                   capture_output=True, text=True,
                                   timeout=5400, env=dict(os.environ))
                t2 = (p.stdout + p.stderr).strip().splitlines()[-4:]
                log.write(f"  single {f.name}: rc={p.returncode}\n" + "\n".join(t2) + "\n")
                log.flush()
    log.write(f"rerun56 done: failed_groups={fails}\n{SENTINEL}\n")
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
