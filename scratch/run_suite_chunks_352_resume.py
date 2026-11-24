#!/usr/bin/env python3
"""round-352 (144th recorded run) -- the ten-chunk gate over the
r352 diagnostics-lane tree (the part16 eighth round: the Phase-12
#6 remainder -- the §15.5 diagnostics/recovery surface -- plus the
trued QA docs): the full collection at 8,776 expected / 0 errors /
485 files (8,746 at r351 + the lane's 19 API + 11 drift-guard tests).

Legs (the r300/r304/r325 layout):
  chunks 1-8   tests/ size-balanced (tests/api excluded -> own leg)
  chunk-9     tests/ui/test_nine_cards.py (own Qt envelope)
  chunk-10    src/pharmaml --doctest-modules
  chunk-11    tests/api/ (the FastAPI service suite)
  chunk-12    ROOT github-page guards (check_github_page.py + unittest)
OOM-degrade: SIGKILL -> 7-file sub-batches -> singles (the r275 drill).
Daemonized by double fork; log /tmp/suite352.txt; sentinel SUITE352 DONE.
The dev server stays DOWN for the gate (the OOM ritual).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = REPO / "pharma-ml-toolkit"
OUT = Path("/tmp/suite352b.txt")
PY = "/home/z/.venv/bin/python"
SENTINEL = "SUITE352B DONE"

for _k, _v in (("LOKY_MAX_CPU_COUNT", "1"), ("OMP_NUM_THREADS", "1"),
               ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
               ("NUMEXPR_NUM_THREADS", "1"), ("NUMBA_NUM_THREADS", "1")):
    os.environ.setdefault(_k, _v)

CHUNKS = 8
NINE_CARD_SUITE = TOOLKIT / "tests" / "ui" / "test_nine_cards.py"
PYTEST = [PY, "-m", "pytest", "-q", "-W", "ignore", "--tb=short",
          "-p", "no:cacheprovider"]


def balance(files: list[Path], groups: int) -> list[list[Path]]:
    """Greedy size-balanced grouping (largest first)."""
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
                              timeout=3600, env=dict(os.environ))
        rc, tail = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return -99, "TIMEOUT after 3600s"
    t = "\n".join(tail.strip().splitlines()[-4:]) if tail.strip() else "(no output)"
    return rc, f"{time.monotonic() - started:.0f}s rc={rc}\n{t}"


def run_group(log, label: str, group: list[Path]) -> int:
    log.write(f"=== {label}: start ({len(group)} files)\n")
    log.flush()
    rc, detail = run(PYTEST + list(map(str, group)))
    log.write(f"=== {label}: {detail}\n")
    log.flush()
    if rc == 0:
        return 0
    if rc != -9:
        return 1  # a real failure, not the container's OOM kill
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
    files = [f for f in files if "tests/api" not in str(f)]
    groups = balance(files, CHUNKS)
    log.write(f"gate open: {len(files)} chunked files, "
              f"{len(groups[0])}..{len(groups[-1])} per chunk\n")
    log.flush()
    failures = 1  # chunk-4's rc=1 already decoded: the reaper's root-node_modules
    # wipe (the two test_web_parity bun probes), the makeup leg re-ran
    # green solo after the reinstall (the r320/r325 precedent class)
    for idx in range(5, CHUNKS + 1):  # chunks 1-4 already evidenced:
        # chunk-1 128s rc=0, chunk-2 109s rc=0, chunk-3 148s rc=0,
        # chunk-4 280s rc=1 (decoded + makeup green)
        failures += run_group(log, f"chunk-{idx}", groups[idx - 1])
    rc, detail = run(PYTEST + [str(NINE_CARD_SUITE)])
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
    rc, detail = run(PYTEST + list((TOOLKIT / "tests" / "api").glob("test_*.py")))
    log.write(f"=== chunk-11 (the api service suite): {detail}\n")
    log.flush()
    if rc != 0:
        failures += 1
    g1 = run([PY, "scripts/check_github_page.py"], cwd=REPO)
    log.write(f"=== chunk-12a (scripts/check_github_page.py): {g1[1]}\n")
    log.flush()
    g2 = run([PY, "-m", "unittest", "discover", "-s", "tests",
              "-p", "test_github_page.py", "-v"], cwd=REPO)
    log.write(f"=== chunk-12b (root unittest github-page): {g2[1]}\n")
    log.flush()
    if g1[0] != 0 or g2[0] != 0:
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
