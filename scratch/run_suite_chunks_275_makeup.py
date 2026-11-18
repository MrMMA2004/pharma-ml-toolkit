#!/usr/bin/env python
"""round-275 -- the chunk-4 MAKEUP leg.  chunk-4 went red at 403s on
ONE test (tests/test_web_parity.py::...ligands_group) because the
container recycle had wiped node_modules — the probe could not import
its modules (environmental, the r274 eviction pattern repeating; bun
install restored 828 packages mid-gate).  This script reproduces the
cont-runner's exact greedy balance (same file set, same CHUNKS=8) to
recover chunk-4's file list, then re-runs that group as one fresh
pytest process — the r274 makeup discipline.  Run AFTER the gate's
sentinel (no parallel pytest)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = REPO / "pharma-ml-toolkit"
PY = "/home/z/.venv/bin/python"
NINE_CARD_SUITE = TOOLKIT / "tests" / "ui" / "test_nine_cards.py"
PYTEST = [PY, "-m", "pytest", "-q", "-W", "ignore", "--tb=short",
          "-p", "no:cacheprovider"]


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
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-4:])
    return proc.returncode, f"{time.monotonic() - started:.0f}s rc={proc.returncode}\n{tail}"


def run_degraded(group: list[Path], label: str) -> int:
    """The gate's own OOM discipline: full group -> 7-file sub-batches
    -> singles.  Returns 0 iff every leg is green."""
    rc, detail = run(PYTEST + list(map(str, group)))
    print(f"=== {label}: {detail}")
    if rc == 0:
        return 0
    if rc != -9:
        return 1  # real failure — the env fix did not cure it
    print(f"=== {label}: SIGKILLed -- the OOM split begins")
    ok = True
    for b in range(0, len(group), 7):
        batch = group[b:b + 7]
        blabel = f"{label} sub-batch {b // 7 + 1} ({len(batch)} files)"
        rc, detail = run(PYTEST + list(map(str, batch)))
        print(f"=== {blabel}: {detail}")
        if rc == 0:
            continue
        if rc != -9:
            ok = False
            continue
        for f in batch:
            flabel = f"{label} single {f.name}"
            rc, detail = run(PYTEST + [str(f)])
            print(f"=== {flabel}: {detail}")
            if rc != 0:
                ok = False
    return 0 if ok else 1


def main() -> int:
    rc_all = 0
    files = sorted((TOOLKIT / "tests").rglob("test_*.py"))
    files.remove(NINE_CARD_SUITE)
    files = [f for f in files if "tests/api" not in str(f)]
    groups = balance(files, 8)
    group = groups[3]  # chunk-4 == the fourth balanced group
    print(f"chunk-4 makeup ({len(group)} files)")
    rc_all |= run_degraded(group, "chunk-4 makeup")
    # Leg B -- sub-batch 2's PyInstaller casualty (the venv had lost
    # PyInstaller; pinned 6.22.3 restored mid-gate at 10:46).
    # Leg C -- the post-OOM single casualty.
    for label, one in (
        ("makeup leg B (pyinstaller restored): tests/test_freeze_equivalent.py",
         TOOLKIT / "tests" / "test_freeze_equivalent.py"),
        ("makeup leg C (post-OOM single): tests/ui/test_nine_card_first.py",
         TOOLKIT / "tests" / "ui" / "test_nine_card_first.py"),
    ):
        rc, detail = run(PYTEST + [str(one)])
        print(f"=== {label}: {detail}")
        rc_all |= rc
    print("MAKEUP275 DONE rc_all=%d" % rc_all)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
