#!/usr/bin/env python
"""round-272 -- the v1.2.216 release archive builder (the
round-180/182/212/215 recipe verbatim: VERSION is read dynamically)
extended to ALL SIX homes: the four tracked homes (download/,
toolkit/dist-archive/, toolkit/download/, upload/), the root
dist-archive/ home (untracked, the task-text home), and the offsite
~/pharma-ml-toolkit-zips/ mirror.  testzip integrity per home, the
leak scan (no /home/z/, no token material), and NO predecessor sweep
(the homes keep their release history: 182..215 stay).

The root dist-archive/ home was missing v1.2.215 in the lost session
(215 shipped to the four tracked homes + offsite only); 216 restores
the six-home shape everywhere.
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
    # NOTE (round 201 calibration): scratch/ and download/ DO ship --
    # the audit trails and the web-evidence screenshots are part of the
    # archive; only the *.zip guard keeps the archive homes themselves out.
]

HOMES = [
    REPO / "download",          # build here (home 1)
    TOOLKIT / "dist-archive",   # home 2 (tracked)
    TOOLKIT / "download",       # home 3 (tracked)
    REPO / "upload",            # home 4 (tracked; FUSE - sha-verified copy)
    REPO / "dist-archive",      # home 5 (root, untracked, the task-text home)
    OFFSITE,                    # home 6 (offsite mirror)
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
    # byte-identical copies into every other home, testzip each
    for home in HOMES[1:]:
        home.mkdir(parents=True, exist_ok=True)
        copy = home / NAME
        shutil.copyfile(target, copy)
        if copy.read_bytes() != data:
            print(f"BYTE MISMATCH: {copy}")
            return 1
        with zipfile.ZipFile(copy) as zf:
            assert zf.testzip() is None, f"testzip failed in {copy}"
        print(f"home (sha+testzip OK): {copy}")
    # the leak scan: no absolute paths, no REAL token material in TEXT
    # members.  (r272: length-qualified regexes — the naive substring
    # needles flagged the repo's own detection signatures and incident
    # documentation: packaging_validation.py's scanner regexes,
    # tools/purge_secret_history.sh's purge patterns, and
    # docs/security/PAT_EXPOSURE_REMEDIATION.md's incident write-up all
    # legitimately SAY "github_pat_"/"ghp_" without carrying any token;
    # a real PAT carries a 60+ char body, a real ghp_ 36+.  Verified by
    # hand first: zero [A-Za-z0-9_]+ bodies follow any prefix in the
    # tree's text members.)
    import re
    TOKEN_PATTERNS = [
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"ghs_[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ]
    leaks = []
    with zipfile.ZipFile(target) as zf:
        for name in files:
            if name.endswith((".png", ".jpg", ".zip", ".db", ".pcap", ".parquet",
                              ".exe", ".ico", ".ttf", ".pdf")):
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            if "/home/z/" in text:
                leaks.append((name, "/home/z/"))
            for pat in TOKEN_PATTERNS:
                if pat.search(text):
                    leaks.append((name, pat.pattern))
    print(f"real leaks: {len(leaks)}")
    for name, needle in leaks[:10]:
        print("  LEAK:", name, needle)
    if leaks:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
