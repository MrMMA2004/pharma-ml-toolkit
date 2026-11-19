"""round-300 -- the release archive builder for v1.2.231 (the r265/r299
recipe: the route's canonical exclusion list, the §4.3 hygiene, the
composition guard, the leak scan, the predecessor sweep) PLUS the
fifth offsite home rebuilt AGAIN (the sandbox rollback wiped
~/pharma-ml-toolkit-zips a second time; it is backfilled from the
in-repo homes' full history before the new release lands).

Sweep policy this round: v1.2.230 (the new predecessor, still the
GitHub-released artifact -- its bytes live in the Release assets and
in git history) leaves the four in-repo homes; the untracked v1.2.229
rollback ghosts (never committed to origin) leave the three homes
that carried them; the offsite home keeps the full history it was
backfilled with (227/228/230) plus the new 231.

VERSION is read dynamically; testzip integrity is checked per home;
the leak scan refuses absolute home paths and token shapes.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = REPO / "pharma-ml-toolkit"
VERSION = (TOOLKIT / "VERSION").read_text().strip()
NAME = f"pharma-ml-toolkit-v{VERSION}.zip"
PREDECESSOR = "pharma-ml-toolkit-v1.2.230.zip"
GHOST = "pharma-ml-toolkit-v1.2.229.zip"
OFFSITE = Path.home() / "pharma-ml-toolkit-zips"

EXCLUDES = [
    # the route's canonical list (toolkit-fs.ts ZIP_EXCLUDE_ARGS)
    ".pytest_cache/*", "*/.pytest_cache/*", "__pycache__/*", "*/__pycache__/*",
    "*.pyc", "catboost_info/*", "*/catboost_info/*",
    "*.log", "*/*.log",
    "artifacts/*", "*/artifacts/*",
    "*.egg-info", "*.egg-info/*", "*/src/*.egg-info/*",
    "*.zip",
    ".venv/*", "*/.venv/*", "venv/*", "*/venv/*", "env/*", "*/env/*",
    ".git/*", "*/.git/*",
    ".ruff_cache/*", "*/.ruff_cache/*", ".mypy_cache/*", "*/.mypy_cache/*",
    ".tox/*", "*/.tox/*", ".idea/*", "*/.idea/*", ".vscode/*", "*/.vscode/*",
    # §4.3 (round 260, v1.2.167): the archive ships the toolkit only --
    # audit trails, screenshots and the archive homes themselves stay out.
    "scratch/*", "*/scratch/*",
    "download/*", "*/download/*",
    "dist-archive/*", "*/dist-archive/*",
]

HOMES = [
    REPO / "download",
    TOOLKIT / "dist-archive",
    TOOLKIT / "download",
    REPO / "upload",
]


def main() -> int:
    target = HOMES[0] / NAME
    print(f"building {NAME} ...")
    proc = subprocess.run(
        ["zip", "-r", "-q", str(target), ".", "-x", *EXCLUDES],
        cwd=TOOLKIT,
    )
    if proc.returncode != 0:
        print("zip failed", proc.returncode)
        return proc.returncode
    data = target.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(target) as zf:
        names = zf.namelist()
        files = [n for n in names if not n.endswith("/")]
        bad = zf.testzip()
    print(f"entries={len(names)} files={len(files)} bytes={len(data)}")
    print(f"testzip first-bad-entry={bad!r}")
    print(f"sha-256={digest}")

    # byte-identical copies into every in-repo home
    for home in HOMES[1:]:
        copy = home / NAME
        copy.write_bytes(data)
        with zipfile.ZipFile(copy) as zf:
            assert zf.testzip() is None, f"testzip failed in {copy}"
        print(f"home (testzip OK): {copy}")

    # the fifth offsite home: recreate + backfill the history from the
    # in-repo homes, then add this release byte-identically
    OFFSITE.mkdir(exist_ok=True)
    seen: set[str] = set()
    for home in [HOMES[0], HOMES[1], HOMES[2], HOMES[3]]:
        for old in sorted(home.glob("pharma-ml-toolkit-v*.zip")):
            if old.name in seen:
                continue
            seen.add(old.name)
            dst = OFFSITE / old.name
            if not dst.exists():
                shutil.copy2(old, dst)
                print(f"offsite backfill: {old.name}")
    off = OFFSITE / NAME
    off.write_bytes(data)
    with zipfile.ZipFile(off) as zf:
        assert zf.testzip() is None, f"testzip failed in {off}"
    print(f"offsite (testzip OK): {off}")

    # the composition guard: the §4.3 exclusions must hold
    forbidden = ("scratch/", "download/", "dist-archive/")
    violations = sorted(n for n in names if n.split("/")[0] in forbidden)
    print(f"composition violations: {len(violations)}")
    for name in violations[:10]:
        print("  VIOLATION:", name)

    # the leak scan: no absolute paths, no tokens in TEXT members
    leaks = []
    with zipfile.ZipFile(target) as zf:
        for name in files:
            if name.endswith((".png", ".jpg", ".zip", ".db", ".pcap", ".parquet")):
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            for needle in ("/home/z/", "github_pat_", "ghp_", "AKIA"):
                if needle in text:
                    leaks.append((name, needle))
    print(f"real leaks: {len(leaks)}")
    for name, needle in leaks[:10]:
        print("  LEAK:", name, needle)

    # the sweep: predecessor v1.2.230 from the four in-repo homes, the
    # untracked v1.2.229 rollback ghosts from wherever they linger
    swept = 0
    for home in HOMES:
        old = home / PREDECESSOR
        if old.exists():
            old.unlink()
            swept += 1
            print(f"swept predecessor: {old}")
        ghost = home / GHOST
        if ghost.exists():
            ghost.unlink()
            swept += 1
            print(f"swept rollback ghost: {ghost}")
    print(f"swept {swept} file(s) from the in-repo homes")

    # VERSION stamp parity: the archive must carry 1.2.231
    with zipfile.ZipFile(target) as zf:
        stamp = zf.read("VERSION").decode().strip()
    print(f"archive VERSION stamp: {stamp}")
    if stamp != VERSION:
        print("VERSION STAMP MISMATCH")
        return 1
    if leaks or violations or bad is not None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
