#!/usr/bin/env python3
"""Comprehensive disk-vs-origin/main comparison for every file tracked in
origin/main: catches the reset blind spot (files with identical blobs in
HEAD and origin/main whose disk copies were modified by the r289 session
and skipped by reset --hard). Prints only paths + byte sizes, no content."""
import subprocess
from pathlib import Path

REPO = Path("/home/z/my-project")


def main() -> int:
    files = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "origin/main"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    import hashlib

    differ = []
    missing = []
    for path in files:
        p = REPO / path
        if not p.is_file():
            missing.append(path)
            continue
        blob = subprocess.run(
            ["git", "cat-file", "blob", f"origin/main:{path}"],
            cwd=REPO, capture_output=True, check=False,
        ).stdout
        if hashlib.md5(blob).hexdigest() != hashlib.md5(p.read_bytes()).hexdigest():
            differ.append(path)
    print(f"tracked-in-origin: {len(files)}")
    print(f"missing-on-disk: {len(missing)}")
    for m in missing[:10]:
        print(f"  MISSING: {m}")
    print(f"differ-from-origin: {len(differ)}")
    for d in differ:
        disk = (REPO / d).stat().st_size
        blob = subprocess.run(
            ["git", "cat-file", "-s", f"origin/main:{d}"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.strip()
        print(f"  DIFFER: {d} (disk {disk}B vs origin {blob}B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
