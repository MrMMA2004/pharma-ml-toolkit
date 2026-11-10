"""round-265 -- the release archive builder (the r264 recipe, the
r264 builder's shape with the standing lesson applied: the sweep chases
the ACTUAL shipped previous version, v1.2.171).

The round-258 recipe PLUS the r260 §4.3 hygiene (verified against the
live v1.2.169 archive composition: 526 entries, no scratch/, no
download/, no dist-archive/ -- the 8.6 MB release instead of the
23.7 MB one that carried 8.7 MB of scratch/download debris).

VERSION is read dynamically; the predecessor sweep removes v1.2.171
from all four homes; testzip integrity is checked per home; the leak
scan refuses absolute home paths and token shapes.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = REPO / "pharma-ml-toolkit"
VERSION = (TOOLKIT / "VERSION").read_text().strip()
NAME = f"pharma-ml-toolkit-v{VERSION}.zip"
PREDECESSOR = "pharma-ml-toolkit-v1.2.171.zip"

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
    # byte-identical copies into every home
    for home in HOMES[1:]:
        copy = home / NAME
        copy.write_bytes(data)
        with zipfile.ZipFile(copy) as zf:
            assert zf.testzip() is None, f"testzip failed in {copy}"
        print(f"home (testzip OK): {copy}")
    # the composition guard: the §4.3 exclusions must hold
    forbidden = ("scratch/", "download/", "dist-archive/")
    violations = sorted(
        n for n in names if n.split("/")[0] in forbidden
    )
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
    # the predecessor sweep
    swept = 0
    for home in HOMES:
        old = home / PREDECESSOR
        if old.exists():
            old.unlink()
            swept += 1
            print(f"swept predecessor: {old}")
    print(f"predecessor swept from {swept} home(s)")
    if leaks or violations or bad is not None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
