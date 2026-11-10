#!/usr/bin/env python3
"""round 210: the round-end archive builder -- the FOUR-HOME edition,
canonical in tools/ since round 260 (MASTER_PROMPT part4 §4.3: the
build utilities moved out of the research-history scratch/; the
historical ``scratch/build_zip_210.py`` invocation path lives on as a
forwarder shim).

The round-179 recipe (build OUTSIDE download/, testzip + leak scan,
then ship) evolved: the release zip now lands byte-identical in ALL
FOUR homes -- repo ``download/`` (the live streamer's first lookup),
``pharma-ml-toolkit/dist-archive/`` (the tracked archive), the legacy
``pharma-ml-toolkit/download/`` home, and ``upload/`` -- and the
IMMEDIATE predecessor's copies leave all four homes (the round-209
policy; deeper history stays -- the repo download/ sweep archive holds
v1.2.106's only surviving copy).  The round-209 ordering discipline
rides along: the home-path scrub runs BEFORE the build (the first
v1.2.123 build leaked the ML notebook's warning traceback;
scrub-then-build caught it same-round).

Round-260 §4.3 policy: the research-history homes (scratch/ probes,
download/ evidence shots, dist-archive/ zip copies) are PROVENANCE,
kept locally and in git, but they no longer ride the release zip --
they were 475 + 129 + 1 entries, 8.7 MB of evidence PNGs alone (37%
of the v1.2.166 archive).  Nothing is deleted; only the archive
diet changes.

This file is COMMITTED (the round-210 lesson: the round-209 builders
-- build_zip_183.py, census.py -- lived untracked and died with the
sandbox rollback; the recipe must ride the repo like everything
else).  It ships in the archive like its ancestors, and the
forbidden prefix is assembled at runtime so this text never carries
a literal the archive scan would flag.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text().strip()
NAME = f"pharma-ml-toolkit-v{VERSION}.zip"

# the four homes (the b5cc1b9 lookup order + the tracked archive pair)
REPO = ROOT.parent
REPO_DOWNLOAD = REPO / "download"
DIST_ARCHIVE = ROOT / "dist-archive"
LEGACY_DOWNLOAD = ROOT / "download"
UPLOAD = REPO / "upload"
HOMES = (REPO_DOWNLOAD, DIST_ARCHIVE, LEGACY_DOWNLOAD, UPLOAD)

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".tox", ".idea", ".vscode", ".cache",
    "node_modules", "catboost_info", "artifacts", ".venv", "venv", "env",
    # (round-260 §4.3) the research-history homes: provenance, not
    # product.  scratch/ keeps the probe archive (475 tracked files,
    # the one-off exploratory drivers), download/ the E2E evidence
    # shots (129 PNGs, 8.7 MB -- 37% of the v1.2.166 archive was
    # screenshots), dist-archive/ the zip safe-copies themselves.
    # Kept on disk and in git; excluded from RELEASE ZIPS only.
    "scratch", "download", "dist-archive",
}
EXCLUDE_SUFFIXES = {".log", ".zip"}
EXCLUDE_DIR_SUFFIX = ".egg-info"

_HOME = "/ho" + "me"  # assembled (the 137 discipline: this file ships)


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
            yield path


def predecessor_versions() -> list[str]:
    """The IMMEDIATE predecessor: the newest shipped version below ours
    found in any home (the round-209 policy — the predecessor's zips
    leave all four homes, deeper history stays where it is)."""
    def key(name: str) -> tuple[int, ...]:
        stem = name.removeprefix("pharma-ml-toolkit-v").removesuffix(".zip")
        try:
            return tuple(int(p) for p in stem.split("."))
        except ValueError:
            return (0,)
    ours = key(NAME)
    best: tuple[tuple[int, ...], str] | None = None
    for home in HOMES:
        for old in home.glob("pharma-ml-toolkit-v*.zip"):
            if old.name == NAME:
                continue
            k = key(old.name)
            if k < ours and (best is None or k > best[0]):
                best = (k, old.name)
    return [best[1]] if best else []


def main() -> int:
    # 1. the home-path scrub FIRST (the round-209 ordering lesson) --
    #    canonical in tools/ since round 260 (§4.3) alongside this
    #    builder; nothing references the scratch/ copy any more.
    scrub = ROOT / "tools" / "scrub_home_paths.py"
    if scrub.exists():
        subprocess.run([sys.executable, str(scrub)], cwd=str(ROOT), check=False)
        print("scrub run (idempotent, BEFORE the build)")

    # 2. build OUTSIDE every home (the nested-zip trap)
    out = ROOT.parent / NAME
    if out.exists():
        out.unlink()
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        dirs: set[str] = set()
        files = list(iter_files())
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            parts = rel.split("/")
            for i in range(1, len(parts)):
                dirs.add("/".join(parts[:i]) + "/")
        for d in sorted(dirs):
            zf.writestr(d, "")
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())
            count += 1

    # 3. verify: testzip + leak scan + nested-zip guard
    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
        if bad is not None:
            print(f"TESTZIP FAIL: {bad}")
            return 1
        names = zf.namelist()
        real: list[tuple[str, str]] = []
        accepted = 0
        for n in names:
            if n.endswith(("/", ".png", ".pdf", ".zip")):
                continue
            try:
                text = zf.read(n).decode("utf-8", errors="ignore")
            except (OSError, ValueError):
                # a binary member that survives errors="ignore" is not
                # text -- it simply cannot carry the leak pattern
                continue
            real += [(n, "real home") for _ in re.finditer(re.escape(_HOME) + r"/z\b", text)]
            accepted += len(re.findall(re.escape(_HOME) + r"/USER\b", text))
        nested = [n for n in names if n.endswith(".zip")]
        print(f"entries: {len(names)} (files: {count})")
        print(f"size: {out.stat().st_size:,} bytes")
        print(f"nested zips: {nested or 'NONE'}")
        print(f"real leaks: {len(real)}")
        for n, _kind in real[:10]:
            print(f"  LEAK {n}")
        print(f"accepted placeholders: {accepted}")
        if real or nested:
            return 1

    # 4. ship into ALL FOUR homes, byte-identical
    for home in HOMES:
        home.mkdir(exist_ok=True)
        dest = home / NAME
        if dest.exists():
            dest.unlink()
        shutil.copyfile(out, dest)
        print(f"shipped: {dest}")

    # 5. the immediate predecessor leaves ALL FOUR homes (the round-209
    #    policy; deeper history stays — v1.2.106's only copy lives in the
    #    repo download/ sweep archive and must survive)
    for name in predecessor_versions():
        for home in HOMES:
            old = home / name
            if old.exists():
                old.unlink()
                print(f"removed predecessor from {home.name}: {name}")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"sha-256: {digest[:8]}… ({digest})")
    out.unlink()  # the temp build copy outside every home
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
