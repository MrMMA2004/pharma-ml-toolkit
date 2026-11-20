#!/usr/bin/env python3
"""round-307 -- the chunk-2 OOM decode (the r275/r304 drill).

Chunk-2 (30 files) got SIGKILLed by the container OOM at ~93% progress
under the one-shot run; the honest decode re-runs the group in 7-file
sub-batches and any still-killed batch as singles -- a pure memory
envelope yields all-green singles; a real failure stays red.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = REPO / "pharma-ml-toolkit"
OUT = Path("/tmp/suite307c2.txt")
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
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                               timeout=3400, env=dict(os.environ))
        rc, tail = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return -99, "TIMEOUT"
    t = "\n".join(tail.strip().splitlines()[-3:]) if tail.strip() else "(no output)"
    return rc, f"{time.monotonic() - started:.0f}s rc={rc}\n{t}"


def main() -> int:
    log = OUT.open("w")
    files = sorted((TOOLKIT / "tests").rglob("test_*.py"))
    assert NINE_CARD_SUITE in files
    files.remove(NINE_CARD_SUITE)
    files = [f for f in files if "tests/api" not in str(f)]
    groups = balance(files, CHUNKS)
    group = groups[1]
    log.write(f"chunk-2 decode: {len(group)} files, 7-file sub-batches\n")
    log.flush()
    ok = True
    for b in range(0, len(group), 7):
        batch = group[b:b + 7]
        label = f"sub-batch {b // 7 + 1} ({len(batch)} files)"
        rc, detail = run(PYTEST + list(map(str, batch)))
        log.write(f"=== {label}: {detail}\n")
        log.flush()
        if rc == 0:
            continue
        if rc != -9:
            ok = False
            continue
        for f in batch:
            rc, detail = run(PYTEST + [str(f)])
            log.write(f"=== single {f.name}: {detail}\n")
            log.flush()
            if rc != 0:
                ok = False
    log.write(f"decode ok={ok}\nSUITE307 CHUNK2 DECODE DONE\n")
    log.close()
    print(OUT.read_text())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
