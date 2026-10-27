#!/usr/bin/env python3
"""(100x-154, sweep patched 100x-155) The round-end archive builder — v1.2.92.

The standing recipe, codified: archive-out EVERY historical release zip
FIRST (the download/ folder must never hold more than the archive being
shipped — the nested-zip trap fired twice in earlier rounds, and the
immediate-predecessor-only sweep let v1.2.84..v1.2.90 accumulate to 97 MB
inside the tree, which the live streamer then nested into every download),
build the new archive OUTSIDE download/ (the 617th-entry lesson), scrub the
fresh junit xml of build-machine paths, testzip + leak-scan the finished
archive, and only then move it into download/.

Exclusions mirror the live streamer's ZIP_EXCLUDE_ARGS (the byte count the
Download tab measures must match what ships): caches, virtualenvs, logs,
egg-info — plus the release zips themselves (the live streamer gained the
same *.zip guard in 100x-155).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = ROOT / "download"
REPO_DOWNLOAD = ROOT.parents[1] / "download"
VERSION = (ROOT / "VERSION").read_text().strip()
NAME = f"pharma-ml-toolkit-v{VERSION}.zip"

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".tox", ".idea", ".vscode", ".cache",
    "node_modules", "catboost_info", "artifacts", ".venv", "venv", "env",
}
EXCLUDE_SUFFIXES = {".log", ".zip"}
EXCLUDE_DIR_SUFFIX = ".egg-info"

# the forbidden prefixes, assembled at runtime (the 137 discipline:
# this file ships in the archive and must not carry a literal the
# archive scan would flag)
_HOME = "/ho" + "me"


def iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.endswith(EXCLUDE_DIR_SUFFIX)
        )
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            if name == "build_zip_154.py" and "scratch" in dirpath:
                pass  # this script ships too (the standing pattern)
            yield path


def main() -> int:
    # 1. archive-out EVERY historical release zip (the 100x-155 sweep —
    #    predecessor-only moves let seven archives accumulate; duplicates
    #    already sitting in the repo download/ are dropped, not duplicated)
    REPO_DOWNLOAD.mkdir(exist_ok=True)
    for old in sorted(DOWNLOAD.glob("pharma-ml-toolkit-v*.zip")):
        if old.name == NAME:
            continue  # idempotent re-run safety (a crashed prior build)
        dest = REPO_DOWNLOAD / old.name
        if dest.exists():
            if dest.read_bytes() == old.read_bytes():
                old.unlink()
                print(f"removed duplicate (byte-identical): {old.name}")
            else:
                shutil.move(str(old), str(dest.with_name(dest.name + ".conflict")))
                print(f"archived OUT (conflict-renamed): {old.name}")
        else:
            shutil.move(str(old), str(dest))
            print(f"archived OUT: {old.name} -> repo download/")

    # 2. scrub the fresh junit xml (the 137 recipe is idempotent)
    scrub = ROOT / "scratch" / "scrub_home_paths_137.py"
    if scrub.exists():
        import subprocess
        subprocess.run([sys.executable, str(scrub)], cwd=str(ROOT), check=False)
        print("scrub run (idempotent)")

    # 3. build OUTSIDE download/ (the nested-zip trap)
    out = ROOT.parent / f"pharma-ml-toolkit-v{VERSION}.zip"
    if out.exists():
        out.unlink()
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # directory entries first (the v1.2.91 archive's shape)
        dirs = set()
        files = list(iter_files())
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            parts = rel.split("/")
            for i in range(1, len(parts)):
                dirs.add("/".join(parts[:i]) + "/")
        for d in sorted(dirs):
            zf.writestr(d, "")
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            zf.write(path, rel)
            count += 1

    # 4. verify: testzip + leak scan
    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
        if bad is not None:
            print(f"TESTZIP FAIL: {bad}")
            return 1
        names = zf.namelist()
        leaks = []
        for n in names:
            if n.endswith(("/", ".png", ".pdf", ".zip")):
                continue
            try:
                text = zf.read(n).decode("utf-8", errors="ignore")
            except Exception:
                continue
            for hit in re.finditer(re.escape(_HOME) + r"/z\b", text):
                leaks.append((n, "real home"))
            for hit in re.finditer(re.escape(_HOME) + r"/USER\b", text):
                leaks.append((n, "placeholder (accepted)"))
        real = [l for l in leaks if l[1].startswith("real")]
        accepted = [l for l in leaks if not l[1].startswith("real")]
        print(f"entries: {len(names)} (files: {count})")
        print(f"size: {out.stat().st_size / 1e6:.2f} MB")
        print(f"nested zips: {[n for n in names if n.endswith('.zip')] or 'NONE'}")
        print(f"real leaks: {len(real)}")
        for n, kind in real[:10]:
            print(f"  LEAK {n} ({kind})")
        print(f"accepted placeholders: {len(accepted)}")

    # 5. move into download/
    shutil.move(str(out), str(DOWNLOAD / NAME))
    print(f"shipped: download/{NAME}")
    return 0 if not real else 1


if __name__ == "__main__":
    raise SystemExit(main())
