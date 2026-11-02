#!/usr/bin/env python3
"""Anti-rollback checkpoint ritual for the pharma-ml-toolkit tree.

Rebuilt 2026-11-02 after the sandbox rollback that ate rounds 180-182
(the original tool, ``docs/RESUME.md`` and three rounds of uncommitted
work vanished together; only git history and TRACKED files survived).

Modes
-----
--doctor        read-only health probe: git, version consistency, venv
                packages, zip state.  Prints ROLLBACK DETECTED when the
                working tree's version sits BELOW the newest surviving
                archive -- follow docs/RESUME.md recovery protocol.
--light         doctor + commit any work-in-progress (UUID message).
                This is what the 30-minute cron guard runs.
--commit MSG    the SHIP ritual: gates (doctests, ruff, mypy, GUI
                self-test) + version consistency + zip build + tracked
                retention sweep + git commit.  Full pytest is NOT a
                gate -- it takes 10-15 min on this box and once timed
                out a 590 s ritual silently (the round-180 lesson);
                run it in chunks before shipping instead.
--restore       unpack the newest surviving zip over the tree, then
                light-commit the result.

Design lessons baked in (see docs/RESUME.md for the full ledger):

* commit the shipped zip itself -- tracked artifacts survived the
  rollback, untracked ones (v1.2.98-v1.2.103) did not;
* keep the newest TWO zips tracked, ``git rm`` the older ones (the
  download/ folder must stay lean; history keeps the blobs);
* build the archive OUTSIDE download/, testzip it, then move it in
  (the nested-zip trap, round-179's recipe);
* the venv is part of the sandbox state: doctor probes the optional
  packages and names the exact install command when they vanish.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[0]  # the git root (~/my-project)
DOWNLOAD = REPO / "download"
VERSION_FILE = ROOT / "VERSION"
PYPROJECT = ROOT / "pyproject.toml"
INIT_PY = ROOT / "src" / "pharmaml" / "__init__.py"

#: optional packages the venv probe expects (the guard's install list).
VENV_PACKAGES = (
    "rdkit", "selfies", "shap", "torch", "meeko", "gemmi", "ruff", "mypy",
)

#: how many shipped zips stay tracked + on disk.
KEEP_ZIPS = 2

ZIP_EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".tox", ".idea", ".vscode", ".cache", "node_modules", "catboost_info",
    "artifacts", ".venv", "venv", "env",
}
ZIP_EXCLUDE_SUFFIXES = {".log", ".zip"}
ZIP_EXCLUDE_DIR_SUFFIX = ".egg-info"


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, check=check,
    )


def _version() -> str:
    return VERSION_FILE.read_text().strip() if VERSION_FILE.is_file() else ""


def _zip_version(path: Path) -> tuple[int, int, int] | None:
    match = re.search(r"v(\d+)\.(\d+)\.(\d+)\.zip$", path.name)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _newest_zip(on_disk: bool = True) -> Path | None:
    """Newest shipped zip: on disk first, then anything git still tracks."""
    candidates: list[tuple[tuple[int, int, int], Path]] = []
    if on_disk and DOWNLOAD.is_dir():
        for path in DOWNLOAD.glob("pharma-ml-toolkit-v*.zip"):
            parsed = _zip_version(path)
            if parsed:
                candidates.append((parsed, path))
    if not candidates:
        listed = _git("ls-files", "download", check=False).stdout.split()
        for name in listed:
            if name.endswith(".zip"):
                parsed = _zip_version(Path(name))
                if parsed:
                    candidates.append((parsed, REPO / name))
    return max(candidates, key=lambda pair: pair[0])[1] if candidates else None


def doctor(verbose: bool = True) -> int:
    """Read-only probe.  0 = healthy, 2 = rollback, 3 = venv packages."""
    problems: list[str] = []
    infos: list[str] = []

    if not (REPO / ".git").is_dir():
        print("FATAL: git repository missing at", REPO)
        return 2

    version = _version()
    if not version:
        print("FATAL: VERSION file missing/empty at", VERSION_FILE)
        return 2

    pyproject = PYPROJECT.read_text() if PYPROJECT.is_file() else ""
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    py_version = match.group(1) if match else ""
    init = INIT_PY.read_text() if INIT_PY.is_file() else ""
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    init_version = match.group(1) if match else ""
    if py_version and py_version != version:
        problems.append(f"pyproject.toml version {py_version} != VERSION {version}")
    if init_version and init_version != version:
        problems.append(f"__init__.__version__ {init_version} != VERSION {version}")
    if py_version and init_version and py_version != version:
        infos.append(
            "note: pyproject lags VERSION -- the ship ritual re-pins it"
        )

    # rollback rule: an archive NEWER than the working tree's version
    # means the tree went backwards.
    newest = _newest_zip()
    if newest is not None:
        parsed = _zip_version(newest)
        assert parsed is not None
        current = tuple(int(part) for part in version.split("."))
        if parsed > current:
            print("ROLLBACK DETECTED")
            print(f"  newest surviving archive: {newest.name}")
            print(f"  working tree VERSION:     {version}")
            print("  recovery: tools/checkpoint.py --restore, then --doctor")
            return 2
        if parsed < current:
            infos.append(f"zip pending: newest archive v{'.'.join(map(str, parsed))} < VERSION {version}")

    dirty = _git("status", "--porcelain", "--", str(ROOT), check=False).stdout.strip()
    if dirty:
        infos.append(f"{len(dirty.splitlines())} uncommitted path(s) under the toolkit (run --light)")

    missing: list[str] = []
    probe = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util, sys, json\n"
         "print(json.dumps([m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]))",
         *VENV_PACKAGES],
        capture_output=True, text=True,
    )
    if probe.returncode == 0 and probe.stdout.strip().startswith("["):
        missing = [name.strip(' "') for name in probe.stdout.strip()[1:-1].split(",") if name.strip()]
    if missing:
        print("MISSING packages:", ", ".join(missing))
        print(f"  install: {sys.executable} -m pip install " + " ".join(missing))
    else:
        infos.append("venv probe: all optional packages present")

    if verbose:
        print(f"VERSION {version} · HEAD {_git('rev-parse', '--short', 'HEAD').stdout.strip()}")
        for line in infos:
            print("  -", line)
    if problems:
        print("PROBLEMS:")
        for line in problems:
            print("  !", line)
        return 4
    return 3 if missing else 0


def light() -> int:
    """Doctor + commit work-in-progress under the toolkit tree."""
    status = doctor(verbose=True)
    if status == 2:
        return 2
    dirty = _git("status", "--porcelain", "--", str(ROOT), check=False).stdout.strip()
    if not dirty:
        print("light: tree clean, nothing to commit")
        return status
    _git("add", "-A", "--", str(ROOT))
    _git("commit", "-m", str(uuid.uuid4()), "--", str(ROOT))
    print(f"light: committed {len(dirty.splitlines())} path(s) as WIP")
    return 0


def _gate(label: str, *args: str, env: dict[str, str] | None = None) -> bool:
    print(f"gate: {label}", flush=True)
    run = subprocess.run(
        list(args), cwd=str(ROOT), capture_output=True, text=True,
        env={**os.environ, **(env or {})},
    )
    if run.returncode != 0:
        tail = "\n".join((run.stdout + run.stderr).splitlines()[-25:])
        print(f"gate FAILED: {label}\n{tail}")
        return False
    return True


def _build_zip(version: str) -> Path | None:
    """Round-179's archive recipe, self-contained: build OUTSIDE download/."""
    name = f"pharma-ml-toolkit-v{version}.zip"
    staging = REPO / f".zip-staging-{version}"
    target = staging / name
    staging.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            rel = Path(dirpath).relative_to(ROOT)
            parts = set(rel.parts)
            if parts & ZIP_EXCLUDE_DIRS or any(
                part.endswith(ZIP_EXCLUDE_DIR_SUFFIX) for part in parts
            ):
                dirnames[:] = []
                continue
            dirnames[:] = sorted(dirnames)
            for filename in sorted(filenames):
                if Path(filename).suffix in ZIP_EXCLUDE_SUFFIXES:
                    continue
                source = Path(dirpath) / filename
                arcname = Path("pharma-ml-toolkit") / rel / filename
                archive.write(source, arcname)
                count += 1
    with zipfile.ZipFile(target) as probe:
        bad = probe.testzip()
        if bad is not None:
            print(f"zip FAILED integrity: {bad}")
            shutil.rmtree(staging, ignore_errors=True)
            return None
    DOWNLOAD.mkdir(parents=True, exist_ok=True)
    final = DOWNLOAD / name
    shutil.move(str(target), str(final))
    shutil.rmtree(staging, ignore_errors=True)
    print(f"zip: {final.name} ({count} files, {final.stat().st_size:,} bytes)")
    return final


def ship(message: str) -> int:
    """The full ritual: gates + zip + retention + commit."""
    status = doctor(verbose=True)
    if status == 2:
        return 2
    version = _version()

    ok = _gate(
        "doctests", sys.executable, "-m", "pytest", "--doctest-modules",
        "-q", "-x", "src/pharmaml", "--no-header",
    )
    ok = ok and _gate(
        "ruff", sys.executable, "-m", "ruff", "check", "src", "playground.py", "tests",
    )
    ok = ok and _gate("mypy", sys.executable, "-m", "mypy", "src")
    self_env = {"DISPLAY": os.environ.get("DISPLAY", ":99")}
    ok = ok and _gate(
        "gui self-test", sys.executable, "playground.py", "--self-test",
        env=self_env,
    )
    if not ok:
        print("ship aborted: a gate failed")
        return 4

    # version consistency is a ship gate (doctor only warns)
    pyproject = PYPROJECT.read_text()
    PYPROJECT.write_text(
        re.sub(r'^version\s*=\s*"[^"]+"', f'version = "{version}"', pyproject, count=1, flags=re.MULTILINE)
    )
    init = INIT_PY.read_text()
    INIT_PY.write_text(
        re.sub(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"', init, count=1)
    )

    archive = _build_zip(version)
    if archive is None:
        return 4

    # retention: newest KEEP_ZIPS stay tracked; older ones leave index+disk
    zips = sorted(
        (path for path in DOWNLOAD.glob("pharma-ml-toolkit-v*.zip")),
        key=lambda path: _zip_version(path) or (0, 0, 0),
    )
    for stale in zips[:-KEEP_ZIPS]:
        print(f"retire: {stale.name}")
        _git("rm", "-q", "--cached", str(stale), check=False)
        stale.unlink(missing_ok=True)

    _git("add", "-A", "--", str(ROOT))
    _git("commit", "-m", message, "--", str(ROOT))
    print(f"ship: committed · {message}")
    return 0


def restore() -> int:
    """Unpack the newest surviving zip over the tree, then light-commit."""
    newest = _newest_zip()
    if newest is None:
        print("restore: no surviving archive found")
        return 4
    if not newest.is_file():
        # tracked in git but gone from disk -- materialise it first
        _git("checkout", "HEAD", "--", str(newest.relative_to(REPO)), check=False)
    if not newest.is_file():
        print(f"restore: {newest} is not recoverable")
        return 4
    print(f"restore: unpacking {newest.name} over {ROOT}")
    with zipfile.ZipFile(newest) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            target = ROOT / Path(member).relative_to("pharma-ml-toolkit")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink)
    return light()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doctor", action="store_true", help="read-only health probe")
    group.add_argument("--light", action="store_true", help="commit work-in-progress")
    group.add_argument("--commit", metavar="MSG", help="the full ship ritual")
    group.add_argument("--restore", action="store_true", help="recover from rollback")
    args = parser.parse_args()
    if args.doctor:
        return doctor()
    if args.light:
        return light()
    if args.restore:
        return restore()
    return ship(args.commit)


if __name__ == "__main__":
    sys.exit(main())
