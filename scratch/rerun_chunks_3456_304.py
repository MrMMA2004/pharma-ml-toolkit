#!/usr/bin/env python
"""r304 gate follow-up: chunks 3-6 of the ten-chunk gate lost their log
lines to a NUL-hole corruption in /tmp/suite304.txt (the sentinel says
failures=1; every surviving leg is rc=0). Re-run the four groups in the
FOREGROUND with the same deterministic chunking and capture everything.
"""
from __future__ import annotations

import os
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
from run_suite_chunks_304 import PYTEST, balance  # the same deterministic split

TOOLKIT = Path("/home/z/my-project/pharma-ml-toolkit")
OUT = Path("/tmp/suite304_rerun.txt")


def main() -> int:
    log = OUT.open("w")
    wanted = [int(a) for a in sys.argv[1:]] or [3, 4, 5, 6]
    files = sorted((TOOLKIT / "tests").rglob("test_*.py"))
    nine = TOOLKIT / "tests" / "ui" / "test_nine_cards.py"
    files.remove(nine)
    files = [f for f in files if "tests/api" not in str(f)]
    groups = balance(files, 8)
    total_failed = 0
    for idx in wanted:
        group = groups[idx - 1]
        log.write(f"=== chunk-{idx} re-run: start ({len(group)} files)\n")
        log.flush()
        t = time.monotonic()
        import subprocess
        proc = subprocess.run(PYTEST + list(map(str, group)), cwd=TOOLKIT,
                              capture_output=True, text=True,
                              timeout=3600, env=dict(os.environ))
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
        log.write(f"=== chunk-{idx} re-run: {time.monotonic()-t:.0f}s rc={proc.returncode}\n{tail}\n")
        log.flush()
        if proc.returncode != 0:
            # the real failure: re-run every file of the group singly for the name
            for f in group:
                p2 = subprocess.run(PYTEST + [str(f)], cwd=TOOLKIT,
                                    capture_output=True, text=True,
                                    timeout=3600, env=dict(os.environ))
                t2 = "\n".join((p2.stdout + p2.stderr).strip().splitlines()[-4:])
                log.write(f"  single {f.name}: rc={p2.returncode}\n{t2}\n")
                log.flush()
                if p2.returncode != 0:
                    total_failed += 1
    log.write(f"rerun done: groups_with_real_failures={total_failed}\n")
    log.close()
    print(open(OUT).read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
