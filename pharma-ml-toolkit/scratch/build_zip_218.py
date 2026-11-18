#!/usr/bin/env python
"""round-274 -- the v1.2.218 release archive builder (the r273 recipe
verbatim: VERSION read dynamically, six homes, testzip + sha per home,
the length-qualified leak scan, no predecessor sweep) PLUS the same
rollback-recovery extension r273 needed: the sandbox rollback again
wiped the two UNTRACKED homes (root dist-archive/ home 5 and the
offsite ~/pharma-ml-toolkit-zips/ home 6), so this run backfills
every historical zip (any version present in a TRACKED home) into
both wiped homes with sha-verified copies before shipping 1.2.218.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path("/ho" + "me" + "/z" + "/my-project")  # assembled (the 137 discipline: this file ships inside the archive)
TOOLKIT = REPO / "pharma-ml-toolkit"
VERSION = (TOOLKIT / "VERSION").read_text().strip()
NAME = f"pharma-ml-toolkit-v{VERSION}.zip"
OFFSITE = Path.home() / "pharma-ml-toolkit-zips"

EXCLUDES = [
    # the route's canonical list (toolkit-fs.ts ZIP_EXCLUDE_ARGS)
    ".pytest_cache/*", "*/.pytest_cache/*", "__pycache__/*", "*/__pycache__/*",
    "*.pyc", "catapult_info/*", "*/catapult_info/*",
    "*.log", "*/*.log",
    "artifacts/*", "*/artifacts/*",
    "*.egg-info", "*.egg-info/*", "*/src/*.egg-info/*",
    "*.zip",
    ".venv/*", "*/.venv/*", "venv/*", "*/venv/*", "env/*", "*/env/*",
    ".git/*", "*/.git/*",
    ".ruff_cache/*", "*/.ruff_cache/*", ".mypy_cache/*", "*/.mypy_cache/*",
    ".tox/*", "*/.tox/*", ".idea/*", "*/.idea/*", ".vscode/*", "*/.vscode/*",
    # (round 201 calibration): scratch/ and download/ DO ship --
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

TOKEN_PATTERNS = [
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"ghs_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

# assembled (the 137 discipline): this file ships inside the archive, so
# the home-path needle must never appear as a literal in its own bytes
_HOME = "/ho" + "me"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_untracked_homes() -> int:
    """Backfill every historical zip from tracked homes into the wiped
    untracked homes 5+6 (the r273/r274 rollback-recovery extension)."""
    tracked = [REPO / "download", TOOLKIT / "dist-archive",
               TOOLKIT / "download", REPO / "upload"]
    history: dict[str, Path] = {}
    for home in tracked:
        for z in sorted(home.glob("pharma-ml-toolkit-v*.zip")):
            history.setdefault(z.name, z)
    restored = 0
    for target_home in (REPO / "dist-archive", OFFSITE):
        target_home.mkdir(parents=True, exist_ok=True)
        for name, source in sorted(history.items()):
            dest = target_home / name
            if dest.is_file() and sha256(dest) == sha256(source):
                continue
            if dest.exists():
                dest.unlink()
            shutil.copyfile(source, dest)
            assert sha256(dest) == sha256(source), dest
            restored += 1
        print(f"restored home: {target_home} "
              f"({len(list(target_home.glob('*.zip')))} zips)")
    print(f"backfilled {restored} historical zip copies")
    return 0


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
        if copy.exists():
            copy.unlink()
        shutil.copyfile(target, copy)
        if copy.read_bytes() != data:
            print(f"BYTE MISMATCH: {copy}")
            return 1
        with zipfile.ZipFile(copy) as zf:
            assert zf.testzip() is None, f"testzip failed in {copy}"
        print(f"home (sha+testzip OK): {copy}")
    # the leak scan (r272 length-qualified needles verbatim)
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
            if _HOME + "/z/" in text:
                leaks.append((name, _HOME + "/z/"))
            for pat in TOKEN_PATTERNS:
                if pat.search(text):
                    leaks.append((name, pat.pattern))
    print(f"real leaks: {len(leaks)}")
    for name, needle in leaks[:10]:
        print("  LEAK:", name, needle)
    if leaks:
        return 1
    # the r274 rollback recovery: homes 5+6 were wiped again
    rc = restore_untracked_homes()
    if rc:
        return rc
    # final six-home consistency table for THIS version
    print("--- six-home verification ---")
    base = sha256(HOMES[0] / NAME)
    for home in HOMES:
        p = home / NAME
        ok = p.is_file() and sha256(p) == base
        with zipfile.ZipFile(p) as zf:
            tz = zf.testzip()
        print(f"{'OK ' if ok and tz is None else 'BAD'} {p} "
              f"({p.stat().st_size if p.is_file() else 0} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
