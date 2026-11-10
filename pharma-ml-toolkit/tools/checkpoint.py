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
--commit MSG    the SHIP ritual: gates (doctests, ruff, mypy, backend
                tooling smoke, the offscreen PySide6 suite, notebook
                version consistency) + version consistency + zip build
                + tracked retention sweep + git commit.  Full pytest is
                NOT a gate -- it takes 10-15 min on this box and once
                timed out a 590 s ritual silently (the round-180
                lesson); run it in chunks before shipping instead
                (tools/run_suite_chunks.py, the NINE-chunk gate).
--restore       unpack the newest surviving zip over the tree, then
                light-commit the result.  Handles BOTH archive layouts
                (root-level, the round-210+ ritual, and the historical
                ``pharma-ml-toolkit/`` prefix), refuses absolute /
                traversal / backslash members outright, stages the
                extraction and verifies the version before a single
                file lands, and never deletes anything the archive
                does not carry (user data survives).
--restore --dry-run
                preview the member -> target plan; writes nothing.

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
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath

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

#: the (round-260 §4.3 policy) research-history homes: scratch/ keeps
#: the probe archive and download/ the E2E evidence shots -- both are
#: PROVENANCE, kept locally and in git, but they stopped riding the
#: release zip in round 260 (they were 37% of the archive).  dist-
#: archive/ holds the zip safe-copies themselves (the .zip suffix
#: already keeps the blobs out; the dir entry goes too).
ZIP_EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".tox", ".idea", ".vscode", ".cache", "node_modules", "catboost_info",
    "artifacts", ".venv", "venv", "env", "scratch", "download", "dist-archive",
}
ZIP_EXCLUDE_SUFFIXES = {".log", ".zip"}
ZIP_EXCLUDE_DIR_SUFFIX = ".egg-info"

#: (round-184 lesson, 2026-11-03) the SAFE COPY home: inside the toolkit
#: tree, so the ship commit's ``-- ROOT`` pathspec carries it in the SAME
#: commit as the code.  A rollback's snapshot either contains that commit
#: (both code and zip survive) or predates it (both are lost together)
#: -- the zip can never again slip out of sync with the round that
#: built it.  Retention keeps the newest four here.
ARCHIVE_DIR = ROOT / "dist-archive"

#: the off-repo copy: outside the repository entirely, immune to any
#: git accident (a whole-filesystem snapshot rollback still reverts
#: it -- nothing on this box survives that -- but no amend, retention
#: sweep or reset ever touches it).
OFFSITE_DIR = Path.home() / "pharma-ml-toolkit-zips"  # off-repo, runtime-derived (no shipped literal)

#: how many the SAFE COPY home keeps (safety beats leanness here).
KEEP_ARCHIVE = 4


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
    """Newest shipped zip: the three homes, then anything git tracks.

    The off-repo copy first (immune to amend/retention/reset), then the
    in-tree safe copy, then download/, then git's index (a tracked zip
    whose disk copy a rollback ate).
    """
    candidates: list[tuple[tuple[int, int, int], Path]] = []
    for home in (OFFSITE_DIR, ARCHIVE_DIR, DOWNLOAD):
        if home.is_dir():
            for path in sorted(home.glob("pharma-ml-toolkit-v*.zip")):
                parsed = _zip_version(path)
                if parsed:
                    candidates.append((parsed, path))
    if candidates:
        return max(candidates, key=lambda pair: pair[0])[1]
    listed = _git("ls-files", "download", "pharma-ml-toolkit/dist-archive", check=False).stdout.split()
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
         ("import importlib.util, sys, json\n"
          "print(json.dumps([m for m in sys.argv[1:] "
          "if importlib.util.find_spec(m) is None]))"),
         *VENV_PACKAGES],
        capture_output=True, text=True, check=False,
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
        env={**os.environ, **(env or {})}, check=False,
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
                # root-level arcname: the layout the round-210+
                # four-home ritual ships (restore handles both, but
                # the two builders must not drift apart again)
                arcname = rel / filename
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


def _ship_gate_commands() -> list[tuple[str, list[str]]]:
    """The ship ritual's gate commands (round-260 §4.1: real, supported
    test commands -- the retired playground.py's ruff slot and its
    ``--self-test`` are gone; studio.py implements no CLI, so the GUI
    gate is the offscreen PySide6 suite the repo actually supports).

    Pinned by tests/test_checkpoint_tooling.py: every referenced path
    must exist on the tree, and no retired entry point may reappear.
    """
    quiet = ["-q", "-p", "no:warnings", "--no-header"]
    return [
        ("doctests", [sys.executable, "-m", "pytest", "--doctest-modules",
                      "-q", "-x", "src/pharmaml", "--no-header"]),
        ("ruff", [sys.executable, "-m", "ruff", "check",
                  "src", "tests", "tools", "ui", "studio.py"]),
        ("mypy", [sys.executable, "-m", "mypy", "src"]),
        ("backend tooling smoke", [sys.executable, "-m", "pytest",
                                   "tests/test_checkpoint_tooling.py", *quiet]),
        ("gui offscreen suite", [sys.executable, "-m", "pytest",
                                 "tests/ui/test_pages_smoke.py", *quiet]),
    ]


def _notebook_version_hits(path: Path) -> list[str]:
    """Every ``1.2.x`` string a notebook's committed outputs print."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    hits: list[str] = []
    pattern = re.compile(r"1\.2\.\d+")
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            chunks: list[str] = []
            text = output.get("text", "")
            if isinstance(text, list):
                chunks.append("".join(text))
            elif isinstance(text, str):
                chunks.append(text)
            data = output.get("data") or {}
            plain = data.get("text/plain", "")
            if isinstance(plain, list):
                chunks.append("".join(plain))
            elif isinstance(plain, str):
                chunks.append(plain)
            hits.extend(pattern.findall("\n".join(chunks)))
    return hits


def _notebook_freshness_problems() -> list[str]:
    """Release checklist item (round-260 §4.1): notebook version
    consistency -- every version a committed notebook prints must be
    the shipped VERSION, and the eight executed workflows must all
    carry the print (the no-output twins ship review-only and carry
    nothing to check)."""
    version = _version()
    if not version:
        return ["VERSION file missing/empty -- cannot check notebooks"]
    problems: list[str] = []
    carrying = 0
    for path in sorted((ROOT / "examples").glob("*.ipynb")):
        hits = _notebook_version_hits(path)
        if not hits:
            continue
        carrying += 1
        stale = sorted({hit for hit in hits if hit != version})
        if stale:
            problems.append(f"{path.name}: prints {', '.join(stale)} != VERSION {version}")
    if carrying < 8:
        problems.append(
            f"only {carrying} notebook(s) print the version -- the eight "
            "executed workflows must (tests/test_notebook_variants.py twin)"
        )
    return problems


def ship(message: str) -> int:
    """The full ritual: gates + zip + retention + commit."""
    status = doctor(verbose=True)
    if status == 2:
        return 2
    version = _version()

    # notebook version consistency gates FIRST (fast, in-process --
    # the round-260 §4.1 release checklist item)
    problems = _notebook_freshness_problems()
    if problems:
        print("ship aborted: notebook version drift")
        for line in problems[:10]:
            print("  !", line)
        return 4

    ok = True
    for label, command in _ship_gate_commands():
        env = {"DISPLAY": os.environ.get("DISPLAY", ":99")} if "gui" in label else None
        ok = ok and _gate(label, *command, env=env)
        if not ok:
            break
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

    # (the user's standing rule, 2026-11-03) SAVE THE ZIP IN A SAFE
    # PLACE -- before anything else can go wrong: the toolkit's own
    # dist-archive/ (rides this round's commit) and the off-repo copy
    # (beyond any git accident).  A failure in the steps below can
    # never cost the fresh zip.
    try:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, ARCHIVE_DIR / archive.name)
        OFFSITE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, OFFSITE_DIR / archive.name)
        print(
            f"safe-copy: {archive.name} -> {ARCHIVE_DIR.relative_to(REPO)} "
            f"+ {OFFSITE_DIR}"
        )
    except OSError as exc:
        print(f"safe-copy FAILED (the zip itself is safe in download/): {exc}")

    # retention: newest KEEP_ZIPS stay tracked; older ones leave index+disk
    zips = sorted(
        (path for path in DOWNLOAD.glob("pharma-ml-toolkit-v*.zip")),
        key=lambda path: _zip_version(path) or (0, 0, 0),
    )
    for stale in zips[:-KEEP_ZIPS]:
        print(f"retire: {stale.name}")
        _git("rm", "-q", "--cached", str(stale), check=False)
        stale.unlink(missing_ok=True)

    # the safe home keeps a longer memory (newest KEEP_ARCHIVE), the
    # same tracked-then-deleted retirement as download/.
    archive_zips = sorted(
        (path for path in ARCHIVE_DIR.glob("pharma-ml-toolkit-v*.zip")),
        key=lambda path: _zip_version(path) or (0, 0, 0),
    )
    for stale in archive_zips[:-KEEP_ARCHIVE]:
        print(f"retire (archive): {stale.name}")
        _git("rm", "-q", "--cached", str(stale), check=False)
        stale.unlink(missing_ok=True)

    _git("add", "-A", "--", str(ROOT))
    _git("commit", "-m", message, "--", str(ROOT))
    print(f"ship: committed · {message}")
    return 0


def _safe_member_path(member: str, prefix: str | None) -> Path | None:
    """Normalize one archive member to a safe relative path.

    Returns None for anything that could escape the target root:
    absolute paths, ``..`` segments, and Windows-style backslash
    separators (which PurePosixPath would not split -- the member
    would extract as a literal filename on POSIX but traverse on
    win32, so it is refused on every platform).
    """
    rel = member[len(prefix):] if prefix else member
    if "\\" in member:
        return None
    path = PurePosixPath(rel)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    return Path(*path.parts)


def _detect_prefix(names: list[str]) -> str | None:
    """Which archive layout is this?  All members under
    ``pharma-ml-toolkit/`` -> the prefixed layout; none -> root-level.
    SOME but not all -> malformed, refuses via the empty-file check in
    _restore_archive (the caller distinguishes)."""
    files = [name for name in names if not name.endswith("/")]
    prefixed = [name for name in files if name.startswith("pharma-ml-toolkit/")]
    if files and len(prefixed) == len(files):
        return "pharma-ml-toolkit/"
    if prefixed:
        return ""  # mixed: malformed sentinel (empty prefix, not None)
    return None


def _restore_archive(archive_path: Path, root: Path, dry_run: bool = False) -> int:
    """Unpack one release zip over ``root`` -- safely (round-260 §4.2).

    * both layouts: root-level (the shipped one) and the historical
      ``pharma-ml-toolkit/`` prefix (auto-detected over FILE members);
    * refuses absolute / traversal / backslash members outright;
    * ``--dry-run`` previews the member -> target plan, writes nothing;
    * stages the extraction beside the target and verifies the staged
      VERSION against the archive's own filename version before a
      single file lands on the tree;
    * never deletes anything the archive does not carry -- user data
      survives a restore.

    Returns 0 on success (or a clean dry run), 4 on any refusal.
    """
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"restore: {archive_path.name} is not a readable zip: {exc}")
        return 4
    with archive:
        bad = archive.testzip()
        if bad is not None:
            print(f"restore: CRC failure on {bad} -- archive corrupt")
            return 4
        names = archive.namelist()
        prefix = _detect_prefix(names)
        if prefix == "":
            prefixed = sum(1 for n in names if n.startswith("pharma-ml-toolkit/"))
            print(
                f"restore: REFUSED -- mixed layout ({prefixed} of {len(names)} "
                "members carry the prefix); the archive is malformed"
            )
            return 4
        plan: list[tuple[str, Path]] = []
        rejected: list[str] = []
        for member in names:
            if member.endswith("/"):
                continue
            rel = _safe_member_path(member, prefix)
            if rel is None:
                rejected.append(member)
                continue
            target = root / rel
            # the containment guarantee, checked rather than assumed
            if root.resolve() not in target.resolve().parents:
                rejected.append(member)
                continue
            plan.append((member, target))
        if rejected:
            print(f"restore: REFUSED -- {len(rejected)} unsafe member(s):")
            for member in rejected[:5]:
                print(f"  ! {member}")
            return 4
        if not plan:
            print("restore: REFUSED -- the archive carries no files")
            return 4
        layout = prefix or "root-level"
        if dry_run:
            print(f"restore (dry run): {len(plan)} member(s), layout: {layout}")
            for member, target in plan[:10]:
                print(f"  {member} -> {target}")
            if len(plan) > 10:
                print(f"  ... and {len(plan) - 10} more")
            return 0
        # staging + validation before replacing a single file
        expected = re.search(r"v(\d+\.\d+\.\d+)\.zip$", archive_path.name)
        staged_version: str | None = None
        staging = Path(tempfile.mkdtemp(prefix=".restore-staging-", dir=str(root.parent)))
        try:
            for member, target in plan:
                staged = staging / target.relative_to(root)
                staged.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, open(staged, "wb") as sink:
                    shutil.copyfileobj(source, sink)
            staged_version_file = staging / "VERSION"
            if staged_version_file.is_file():
                staged_version = staged_version_file.read_text().strip()
            if (
                expected is not None
                and staged_version is not None
                and staged_version != expected.group(1)
            ):
                print(
                    f"restore: REFUSED -- archive name says v{expected.group(1)} "
                    f"but its VERSION member says {staged_version}"
                )
                return 4
            for member, target in plan:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staging / target.relative_to(root), target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    print(
        f"restore: {len(plan)} file(s) landed (layout: {layout}"
        + (f", version {staged_version}" if staged_version else "")
        + ")"
    )
    return 0


def restore(dry_run: bool = False) -> int:
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
    mode = " (dry run)" if dry_run else ""
    print(f"restore: unpacking {newest.name} over {ROOT}{mode}")
    rc = _restore_archive(newest, ROOT, dry_run=dry_run)
    if rc != 0 or dry_run:
        return rc
    return light()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doctor", action="store_true", help="read-only health probe")
    group.add_argument("--light", action="store_true", help="commit work-in-progress")
    group.add_argument("--commit", metavar="MSG", help="the full ship ritual")
    group.add_argument("--restore", action="store_true", help="recover from rollback")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="with --restore: preview the member -> target plan, write nothing",
    )
    args = parser.parse_args()
    if args.dry_run and not args.restore:
        parser.error("--dry-run only makes sense together with --restore")
    if args.doctor:
        return doctor()
    if args.light:
        return light()
    if args.restore:
        return restore(dry_run=args.dry_run)
    return ship(args.commit)


if __name__ == "__main__":
    sys.exit(main())
