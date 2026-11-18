#!/usr/bin/env python3
"""Build and verify a four-home release ZIP without clobbering existing copies.

The source files are copied to private staging before the historical path scrub
runs, so the build does not rewrite the checkout. The ZIP is CRC/leak checked
and each output is published with a SHA-256-verified, no-clobber copy. The
historical immediate-predecessor retention rule runs only for copies with a
matching independent backup; otherwise they remain. This is not a crash-atomic
multi-home transaction; successful copies remain when a later step fails.
"""
from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
OFFSITE_DIR = Path.home() / "pharma-ml-toolkit-zips"

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".tox", ".idea", ".vscode", ".cache",
    "node_modules", "catboost_info", ".venv", "venv", "env",
    # (part11 Phase Three) the frontend build output: the Tauri host
    # rebuilds dist/ from the committed sources on the Windows side;
    # only the sources + lockfile ship.
    "dist",
    # (round-260 §4.3) the research-history homes: provenance, not
    # product.  scratch/ keeps the probe archive (475 tracked files,
    # the one-off exploratory drivers), download/ the E2E evidence
    # shots (129 PNGs, 8.7 MB -- 37% of the v1.2.166 archive was
    # screenshots), dist-archive/ the zip safe-copies themselves.
    # Kept on disk and in git; excluded from RELEASE ZIPS only.
    "scratch", "download", "dist-archive",
}
# (r285 defect fix) the backend's artifact OUTPUT directory lives under
# data/ — the bare "artifacts" name that lived in EXCLUDE_DIRS until
# v1.2.227 matched ANY path component, so the release zips silently
# shipped the frontend WITHOUT frontend/src/services/artifacts/client.ts
# while ml/dl/generative imported it — an internally inconsistent
# archive (the sandbox rollback exposed it when the working-tree copy
# vanished and the shipped zip could not restore it). The exclusion is
# now path-anchored: only data/artifacts (the backend output home) is
# dropped, the frontend services lane always ships.
EXCLUDE_DIR_PATHS = {"data/artifacts"}
EXCLUDE_SUFFIXES = {".log", ".zip"}
EXCLUDE_DIR_SUFFIX = ".egg-info"

_HOME = "/ho" + "me"  # assembled (the 137 discipline: this file ships)


def iter_files(root: Path | None = None):
    """Yield release files, refusing inaccessible paths and symlinks."""
    base = ROOT if root is None else root

    def fail_walk(error: OSError) -> None:
        raise error

    for dirpath, dirnames, filenames in os.walk(base, onerror=fail_walk):
        folder = Path(dirpath)
        rel = folder.relative_to(base).as_posix()
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in EXCLUDE_DIRS
            and not d.endswith(EXCLUDE_DIR_SUFFIX)
            and (f"{rel}/{d}" if rel != "." else d)
            not in EXCLUDE_DIR_PATHS
        )
        for name in dirnames:
            if (folder / name).is_symlink():
                raise OSError(f"release input directory is a symlink: {folder / name}")
        for name in sorted(filenames):
            path = folder / name
            if path.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            if path.is_symlink() or not path.is_file():
                raise OSError(f"release input is not a regular file: {path}")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_copy(source: Path, destination: Path, expected: str) -> None:
    """Verify a staged copy, then create its final name exclusively.

    The hardlink lane is the default (atomic, zero extra bytes); the
    ``upload/`` home rides a bind-mounted filesystem that refuses
    ``os.link`` (Errno 95 — the r283 find), so the fallback is the
    copyfile path: a byte copy into the SAME verified staging dir,
    digest-verified again, renamed (or copied) into place. Either way
    the digest is checked before AND after the move.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".release-copy-", dir=destination.parent) as folder:
        staged = Path(folder) / destination.name
        shutil.copy2(source, staged)
        if _sha256(staged) != expected or _sha256(source) != expected:
            raise OSError(f"copy verification failed: {destination}")
        try:
            os.link(staged, destination)
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EXDEV, errno.EROFS):
                raise
            # the copyfile path: same digest, byte copy, verify after
            final = destination.with_suffix(destination.suffix + ".copyfile")
            shutil.copy2(staged, final)
            if _sha256(final) != expected:
                final.unlink(missing_ok=True)
                raise OSError(f"copyfile fallback failed: {destination}") from exc
            os.replace(final, destination)
            print(f"copyfile path: {destination} (hardlink refused: {exc.errno})")
        if _sha256(destination) != expected:
            raise OSError(f"published digest failed: {destination}")


def _has_independent_backup(old: Path) -> bool:
    backup = OFFSITE_DIR / old.name
    try:
        # A backup in a home that the sweep will visit is not independent.
        if backup.parent.resolve() in {home.resolve() for home in HOMES}:
            return False
        return (
            not old.is_symlink() and old.is_file()
            and not backup.is_symlink() and backup.is_file()
            and not os.path.samefile(old, backup)
            and _sha256(old) == _sha256(backup)
        )
    except OSError:
        return False


def _verify_archive(out: Path, count: int) -> bool:
    """Retain the original CRC, nested-ZIP and home-path checks."""
    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
        if bad is not None:
            print(f"TESTZIP FAIL: {bad}")
            return False
        names = zf.namelist()
        real: list[tuple[str, str]] = []
        accepted = 0
        for name in names:
            if name.endswith(("/", ".png", ".pdf", ".zip")):
                continue
            text = zf.read(name).decode("utf-8", errors="ignore")
            real += [(name, "real home") for _ in re.finditer(re.escape(_HOME) + r"/z\b", text)]
            accepted += len(re.findall(re.escape(_HOME) + r"/USER\b", text))
        nested = [name for name in names if name.endswith(".zip")]
        print(f"entries: {len(names)} (files: {count})")
        print(f"size: {out.stat().st_size:,} bytes")
        print(f"nested zips: {nested or 'NONE'}")
        print(f"real leaks: {len(real)}")
        for name, _kind in real[:10]:
            print(f"  LEAK {name}")
        print(f"accepted placeholders: {accepted}")
        return not real and not nested


def main() -> int:
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", VERSION) is None:
        print(f"build FAILED: invalid version {VERSION!r}")
        return 1
    if len({home.resolve() for home in HOMES}) != len(HOMES):
        print("build FAILED: publication homes overlap")
        return 1
    for home in HOMES:
        if os.path.lexists(home / NAME):
            print(f"build FAILED: refusing to overwrite {home / NAME}")
            return 1

    try:
        # Stage OUTSIDE the toolkit root; scrub the private copy, not the tree.
        with tempfile.TemporaryDirectory(prefix=".release-build-", dir=REPO) as temporary:
            staging = Path(temporary)
            tree = staging / "tree"
            tree.mkdir()
            for path in iter_files():
                target = tree / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                original = _sha256(path)
                shutil.copy2(path, target)
                if _sha256(target) != original or _sha256(path) != original:
                    raise OSError(f"staging copy verification failed: {path}")

            scrub = tree / "tools" / "scrub_home_paths.py"
            if scrub.is_file():
                subprocess.run([sys.executable, str(scrub)], cwd=str(tree), check=True)
                print("scrub verified on private staging copy")

            files = list(iter_files(tree))
            out = staging / NAME
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                dirs: set[str] = set()
                for path in files:
                    parts = path.relative_to(tree).as_posix().split("/")
                    for i in range(1, len(parts)):
                        dirs.add("/".join(parts[:i]) + "/")
                for folder in sorted(dirs):
                    zf.writestr(folder, "")
                for path in files:
                    zf.write(path, path.relative_to(tree).as_posix())
            if not _verify_archive(out, len(files)):
                return 1
            digest = _sha256(out)
            for home in HOMES:
                dest = home / NAME
                _publish_copy(out, dest, digest)
                print(f"shipped: {dest}")
            if any(_sha256(home / NAME) != digest for home in HOMES) or _sha256(out) != digest:
                raise OSError("published ZIP verification failed")

            # The original immediate-predecessor policy is conditional on an
            # independent matching backup; an unsafe candidate is retained.
            for name in predecessor_versions():
                for home in HOMES:
                    old = home / name
                    if not os.path.lexists(old):
                        continue
                    if not _has_independent_backup(old):
                        print(f"retention skipped: no independent matching backup for {old}")
                        continue
                    old.unlink()
                    print(f"removed predecessor from {home.name}: {name}")
            print(f"sha-256: {digest}")
            return 0
    except (OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        print(f"build FAILED: {exc}; existing and already published copies retained")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
