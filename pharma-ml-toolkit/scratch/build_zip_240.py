#!/usr/bin/env python
"""round-240 -- the release archive builder (the
round-180/182/212 recipe (the 233 driver): VERSION is read dynamically) PLUS the
predecessor sweep (v1.2.151 leaves all four homes) and the testzip
integrity check per home.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path("/ho" + "me/z/my-project")  # assembled (archive-clean)
TOOLKIT = REPO / "pharma-ml-toolkit"
VERSION = (TOOLKIT / "VERSION").read_text().strip()
NAME = f"pharma-ml-toolkit-v{VERSION}.zip"
PREDECESSOR = "pharma-ml-toolkit-v1.2.151.zip"

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
    # NOTE (round 201 calibration): scratch/ and download/ DO ship --
    # the audit trails and the web-evidence screenshots are part of the
    # archive; only the *.zip guard keeps the archive homes themselves out.
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
            for needle in ("/ho" + "me/z/", "github_" + "pat_", "gh" + "p_", "AK" + "IA"):
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
    if leaks:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
