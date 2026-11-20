#!/usr/bin/env python
"""round-301 (91st recorded run) -- the ten-chunk gate over the POST-REPAIR
state: the sandbox rolled the checkout back to 2eb0ede4 again (482 commits
behind origin/main) with the r301+ content riding untracked; the sync
completed as a verified ff-only to 35aab254 (the round-300 tree, byte-audit:
154/156 tracked identical, 598+441 untracked identical removed, the 229/230
zip ghosts + the stale toolkit-dir windows-release.yml swept, the two
locally-deleted upload zips finalized by the merge), and the venv was
repaired FOREGROUND from a wiped state (numpy 2.1.3 -> 1.26.4, rdkit/torch/
shap/PySide6/pyinstaller reinstalled, stub patch applied) -- so the gate
re-runs fresh on the repaired venv even though the tree is unchanged.
Legs (the r300 layout):
  chunks 1-8   tests/ size-balanced (tests/api excluded -> own leg)
  chunk-9     tests/ui/test_nine_cards.py (own Qt envelope)
  chunk-10    src/pharmaml --doctest-modules
  chunk-11    tests/api/ (the FastAPI service suite)
  chunk-12    ROOT github-page guard (check_github_page.py + unittest)
OOM-degrade: SIGKILL -> 7-file sub-batches -> singles (the r275 drill).
Daemonized by double fork; log /tmp/suite301.txt; sentinel SUITE301 DONE.
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
OUT = Path("/tmp/suite301.txt")
PY = "/home/z/.venv/bin/python"
SENTINEL = "SUITE301 DONE"

# the r300 chunk-1 lesson: test_playground.py alone reaches ~3 GB RSS
# on the 3 GB box when BLAS/loky spin their default thread pools -- the
# same thread caps the notebook re-exec uses keep the forked workers
# from doubling memory (they limit parallelism, not semantics).
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
    failures = 0
    for idx in range(1, CHUNKS + 1):
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
    # chunk-12: the ROOT-level github-page guard (arena merge territory)
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
