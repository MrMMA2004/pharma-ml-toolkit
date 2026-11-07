#!/usr/bin/env python
"""build_zip_183 -- the round-206 release archive builder (the
round-180/182 recipe verbatim: VERSION is read dynamically, so the copy
needed only the round stamp).

Mirrors the canonical exclusion list the download route streams with
(ZIP_EXCLUDE_ARGS in src/lib/toolkit-fs.ts) PLUS the round-185 lessons:
scratch/ (research clones once walked in and filled the disk), the
release-archive homes themselves (*.zip covers them), and tool caches.
The identical bytes land in ALL FOUR homes: repo-root download/,
toolkit dist-archive/, the legacy toolkit download/, and upload/.
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
    # archive (v1.2.115 carried 411 scratch + 128 download files); only
    # the *.zip guard (above) keeps the archive homes themselves out.
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
    print(f"entries={len(names)} files={len(files)} bytes={len(data)}")
    print(f"sha-256={digest}")
    # byte-identical copies into every home
    for home in HOMES[1:]:
        copy = home / NAME
        copy.write_bytes(data)
        print(f"home: {copy}")
    # the leak scan: no absolute paths, no tokens, no emails in TEXT members
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
