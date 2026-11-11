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
                extraction and verifies metadata, backs up overwritten
                files and rolls back failed installs. Files absent from
                the archive are preserved. A crash may require manual
                recovery from the retained sibling journal/backups.
--restore --dry-run
                preview the member -> target plan; writes nothing.

Design lessons baked in (see docs/RESUME.md for the full ledger):

* commit the shipped zip itself -- tracked artifacts survived the
  rollback, untracked ones (v1.2.98-v1.2.103) did not;
* keep the newest TWO zips tracked, ``git rm`` the older ones (the
  download/ folder must stay lean; history keeps the blobs);
* build in private staging, CRC-check, then publish without replacing an
  existing path; archive staging and research-history directories are excluded;
* the venv is part of the sandbox state: doctor probes the optional
  packages and names the exact install command when they vanish.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
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


def validate(target: str) -> int:
    """Part-5 §13 checkpoint conformance check (read-only).

    Exit codes: 0 = valid v2 envelope; 1 = invalid or legacy (a legacy
    checkpoint is storage-valid but not §13-resumable — the report says
    which); 4 = the pharmaml package could not be imported.
    """
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    try:
        from pharmaml.utils.checkpoint_schema import (
            describe_report,
            validate_checkpoint,
        )
    except ImportError as exc:
        print(f"cannot import the toolkit schema module: {exc}", file=sys.stderr)
        return 4
    report = validate_checkpoint(target)
    print(describe_report(report))
    return 0 if report.ok else 1


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
    """Build and CRC-check a new root-level ZIP, without replacing any file.

    Staging uses a unique private directory in the output filesystem. An
    exclusive hard link publishes the verified ZIP atomically without clobbering
    an existing path, including a dangling symlink or a concurrent creator.
    Filesystems without hard-link support fail closed. This protects only this
    build step; the separate ship ritual's safe-copy and retention steps have
    additional side effects and are not certified by this function.
    """
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        print(f"zip FAILED: invalid release version {version!r}")
        return None
    name = f"pharma-ml-toolkit-v{version}.zip"
    final = DOWNLOAD / name
    if os.path.lexists(final):
        print(f"zip FAILED: refusing to overwrite {final}")
        return None
    count = 0
    try:
        DOWNLOAD.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".checkpoint-zip-", dir=DOWNLOAD) as staging:
            stage = Path(staging)
            target = stage / name
            def fail_walk(error: OSError) -> None:
                raise error

            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for dirpath, dirnames, filenames in os.walk(ROOT, onerror=fail_walk):
                    folder = Path(dirpath)
                    rel = folder.relative_to(ROOT)
                    parts = set(rel.parts)
                    if (folder == stage or parts & ZIP_EXCLUDE_DIRS or any(
                        part.endswith(ZIP_EXCLUDE_DIR_SUFFIX) for part in parts
                    )):
                        dirnames[:] = []
                        continue
                    dirnames[:] = sorted(dirnames)
                    for filename in sorted(filenames):
                        if Path(filename).suffix in ZIP_EXCLUDE_SUFFIXES:
                            continue
                        source = folder / filename
                        archive.write(source, rel / filename)
                        count += 1
            with zipfile.ZipFile(target) as probe:
                bad = probe.testzip()
                if bad is not None:
                    print(f"zip FAILED integrity: {bad}")
                    return None
            # Source and destination are on the same filesystem. Unlike
            # rename/replace, link refuses an existing name atomically.
            os.link(target, final)
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"zip FAILED: {exc}")
        return None
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


# Bounds apply before testzip/decompression. Callers can lower or raise them
# deliberately for an independently trusted archive.
RESTORE_LIMITS = {
    "max_members": 10_000,
    "max_file_size": 256 * 1024 * 1024,
    "max_total_size": 2 * 1024 * 1024 * 1024,
    "max_ratio": 1000.0,
}
_REQUIRED_RESTORE_FILES = ("VERSION", "pyproject.toml", "src/pharmaml/__init__.py")


def _safe_member_path(member: str, prefix: str | None) -> Path | None:
    """Reject ambiguous, platform-dependent or traversal paths, not normalize them."""
    if "\\" in member or member.startswith("/") or "\x00" in member:
        return None
    if prefix and not member.startswith(prefix):
        return None
    rel = member[len(prefix):] if prefix else member
    rel = rel.removesuffix("/")
    parts = rel.split("/")
    if (not rel or any(part in ("", ".", "..") for part in parts)
            or any(":" in part or part.endswith((" ", ".")) for part in parts)
            or any(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", part)
                   for part in parts)):
        return None
    path = PurePosixPath(rel)
    return Path(*path.parts)


def _detect_prefix(names: list[str]) -> str | None:
    """Detect root-level or historical prefixed layout; reject mixed files."""
    files = [name for name in names if not name.endswith("/")]
    prefixed = [name for name in files if name.startswith("pharma-ml-toolkit/")]
    if files and len(prefixed) == len(files):
        return "pharma-ml-toolkit/"
    if prefixed:
        return ""
    return None


def _restore_plan(archive: zipfile.ZipFile, limits: dict[str, int | float]):
    """Validate *all* entries and size bounds before any decompression."""
    infos = archive.infolist()
    if len(infos) > limits["max_members"]:
        raise ValueError("archive member-count limit exceeded")
    prefix = _detect_prefix([info.filename for info in infos])
    if prefix == "":
        raise ValueError("mixed archive layouts")
    nodes: dict[str, tuple[str, str]] = {}
    explicit: set[str] = set()
    raw: set[str] = set()
    plan: list[tuple[zipfile.ZipInfo, Path]] = []
    total = 0
    for info in infos:
        name = info.filename
        if name in raw:
            raise ValueError(f"duplicate raw member name: {name!r}")
        raw.add(name)
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"special-file or symlink archive member: {name!r}")
        if kind == stat.S_IFDIR and not info.is_dir() or kind == stat.S_IFREG and info.is_dir():
            raise ValueError(f"inconsistent archive member type: {name!r}")
        if info.flag_bits & 1:
            raise ValueError(f"encrypted archive member: {name!r}")
        if info.is_dir() and info.file_size:
            raise ValueError(f"directory contains unexpected data: {name!r}")
        metadata_name = name[len(prefix):] if prefix and name.startswith(prefix) else name
        if metadata_name in _REQUIRED_RESTORE_FILES and info.file_size > 1024 * 1024:
            raise ValueError(f"version metadata is unexpectedly large: {name!r}")
        if info.file_size > limits["max_file_size"]:
            raise ValueError(f"per-file size limit exceeded: {name!r}")
        total += info.file_size
        if total > limits["max_total_size"]:
            raise ValueError("archive total-size limit exceeded")
        if info.file_size and (not info.compress_size or info.file_size / info.compress_size > limits["max_ratio"]):
            raise ValueError(f"compression-ratio limit exceeded: {name!r}")
        if prefix and name == prefix and info.is_dir():
            continue
        rel = _safe_member_path(name, prefix)
        if rel is None:
            raise ValueError(f"unsafe archive path: {name!r}")
        parts = rel.parts
        for i in range(1, len(parts) + 1):
            path = "/".join(parts[:i])
            normalized = unicodedata.normalize("NFC", path)
            key = normalized.casefold()
            node_kind = "dir" if i < len(parts) or info.is_dir() else "file"
            if key in nodes and nodes[key] != (path, node_kind):
                raise ValueError(f"normalized, case-fold, or file/directory collision: {name!r}")
            nodes[key] = (path, node_kind)
            if i == len(parts):
                if key in explicit:
                    raise ValueError(f"duplicate normalized archive member: {name!r}")
                explicit.add(key)
        if not info.is_dir():
            plan.append((info, rel))
    if not plan:
        raise ValueError("archive carries no files")
    if not set(_REQUIRED_RESTORE_FILES).issubset({rel.as_posix() for _, rel in plan}):
        raise ValueError("archive is missing VERSION or required package metadata")
    return prefix, plan


def _validate_restore_versions(values: dict[str, bytes], archive_path: Path) -> str:
    try:
        version = values["VERSION"].decode("utf-8").strip()
        project = values["pyproject.toml"].decode("utf-8")
        init = values["src/pharmaml/__init__.py"].decode("utf-8")
    except (KeyError, UnicodeError) as exc:
        raise ValueError("missing or invalid UTF-8 version metadata") from exc
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("invalid VERSION in archive")
    project_section = re.search(r"(?ms)^\[project\]\s*$([\s\S]*?)(?=^\[|\Z)", project)
    py_versions = re.findall(r'''(?m)^version\s*=\s*["']([^"']+)["']''', project_section.group(1)) if project_section else []
    init_versions = re.findall(r'''(?m)^__version__\s*=\s*["']([^"']+)["']''', init)
    if py_versions != [version] or init_versions != [version]:
        raise ValueError("archive package version carriers disagree, are missing, or are duplicated")
    expected = _zip_version(archive_path)
    if expected and tuple(map(int, version.split("."))) != expected:
        raise ValueError(f"archive filename version disagrees with VERSION {version}")
    return version


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_restore_target(root: Path, rel: Path) -> None:
    """Refuse symlink ancestors or non-regular destination files."""
    current = root
    for part in rel.parts[:-1]:
        current = current / part
        if current.is_symlink() or current.exists() and not current.is_dir():
            raise ValueError(f"unsafe target parent: {current}")
        if current.exists() and current.stat().st_dev != root.stat().st_dev:
            raise ValueError(f"target parent is on a different filesystem: {current}")
    target = root / rel
    if target.is_symlink() or target.exists() and not target.is_file():
        raise ValueError(f"unsafe target file: {target}")


def _backup_restore_file(source: Path, backup: Path) -> None:
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, backup)
    if (_sha256(source) != _sha256(backup)
            or stat.S_IMODE(source.stat().st_mode) != stat.S_IMODE(backup.stat().st_mode)
            or source.stat().st_mtime_ns != backup.stat().st_mtime_ns):
        raise OSError(f"backup verification failed: {source}")


def _restore_mkdir(path: Path) -> None:
    path.mkdir()


def _restore_install(staged: Path, target: Path) -> None:
    os.replace(staged, target)


def _verify_restore_tree(root: Path, expected_hashes: dict[str, str], archive_path: Path) -> None:
    for rel, digest in expected_hashes.items():
        target = root / rel
        _check_restore_target(root, Path(rel))
        if not target.is_file() or _sha256(target) != digest:
            raise OSError(f"installed file verification failed: {rel}")
    values = {name: (root / name).read_bytes() for name in _REQUIRED_RESTORE_FILES}
    _validate_restore_versions(values, archive_path)


def _write_restore_journal(path: Path, journal: dict) -> None:
    # Replaced atomically; recovery files are retained when rollback fails.
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(journal, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _restore_archive(
    archive_path: Path, root: Path, dry_run: bool = False, *,
    max_members: int = 10_000, max_file_size: int = 256 * 1024 * 1024,
    max_total_size: int = 2 * 1024 * 1024 * 1024, max_ratio: float = 1000.0,
) -> int:
    """Validate, stage, back up and install an archive with exception rollback.

    Existing files outside the archive are untouched. A failed rollback keeps
    the sibling recovery directory and its journal for manual recovery. This
    is not crash-atomic: power loss, concurrent writers, and an unavailable
    filesystem can require manual recovery; do not run concurrent restores.
    """
    limits = {"max_members": max_members, "max_file_size": max_file_size,
              "max_total_size": max_total_size, "max_ratio": max_ratio}
    if any(not isinstance(v, (int, float)) or isinstance(v, bool)
           or not math.isfinite(v) or v <= 0 for v in limits.values()):
        print("restore: REFUSED -- limits must be positive")
        return 4
    if not root.is_dir() or root.is_symlink():
        print("restore: REFUSED -- target root must be an existing non-symlink directory")
        return 4
    root = root.absolute()
    try:
        if root.stat().st_dev != root.parent.stat().st_dev:
            print("restore: REFUSED -- root and recovery directory must share a filesystem")
            return 4
    except OSError as exc:
        print(f"restore: REFUSED -- cannot inspect target filesystem: {exc}")
        return 4
    recovery: Path | None = None
    keep_recovery = False
    attempted: list[tuple[Path, bool]] = []
    created_dirs: list[Path] = []
    backup_hashes: dict[str, str] = {}
    backup_metadata: dict[str, tuple[int, int]] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            prefix, plan = _restore_plan(archive, limits)
            for _, rel in plan:
                _check_restore_target(root, rel)
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"CRC failure on {bad}")
            required = {rel.as_posix(): archive.read(info) for info, rel in plan
                        if rel.as_posix() in _REQUIRED_RESTORE_FILES}
            version = _validate_restore_versions(required, archive_path)
            if dry_run:
                print(f"restore (dry run): {len(plan)} member(s), layout: {prefix or 'root-level'}, version {version}")
                for info, rel in plan[:10]:
                    print(f"  {info.filename} -> {root / rel}")
                if len(plan) > 10:
                    print(f"  ... and {len(plan) - 10} more")
                return 0
            recovery = Path(tempfile.mkdtemp(prefix=".restore-recovery-", dir=str(root.parent)))
            stage, backups = recovery / "staged", recovery / "backups"
            hashes: dict[str, str] = {}
            existing: dict[str, bool] = {}
            for info, rel in plan:
                staged = stage / rel
                staged.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, staged.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                if staged.stat().st_size != info.file_size:
                    raise OSError(f"staged size mismatch: {rel}")
                hashes[rel.as_posix()] = _sha256(staged)
            _validate_restore_versions({name: (stage / name).read_bytes() for name in _REQUIRED_RESTORE_FILES}, archive_path)
            for _, rel in plan:
                _check_restore_target(root, rel)
                target = root / rel
                existed = target.is_file()
                existing[rel.as_posix()] = existed
                if existed:
                    _backup_restore_file(target, backups / rel)
                    backup_hashes[rel.as_posix()] = _sha256(backups / rel)
                    file_stat = (backups / rel).stat()
                    backup_metadata[rel.as_posix()] = (stat.S_IMODE(file_stat.st_mode), file_stat.st_mtime_ns)
            journal: dict = {"root": str(root), "archive": str(archive_path.absolute()),
                             "version": version, "existing": existing,
                             "backup_sha256": backup_hashes, "backup_metadata": backup_metadata,
                             "attempted": [], "created_dirs": []}
            journal_path = recovery / "journal.json"
            _write_restore_journal(journal_path, journal)
            for _, rel in plan:
                _check_restore_target(root, rel)
                target = root / rel
                parents: list[Path] = []
                parent = target.parent
                while parent != root and not parent.exists():
                    parents.append(parent)
                    parent = parent.parent
                for directory in reversed(parents):
                    created_dirs.append(directory)
                    journal["created_dirs"].append(str(directory.relative_to(root)))
                    _write_restore_journal(journal_path, journal)
                    _restore_mkdir(directory)
                had_file = existing[rel.as_posix()]
                if had_file and (_sha256(target) != _sha256(backups / rel)):
                    raise OSError(f"target changed during restore: {rel}")
                if not had_file and target.exists():
                    raise OSError(f"target appeared during restore: {rel}")
                attempted.append((rel, had_file))
                journal["attempted"].append(rel.as_posix())
                _write_restore_journal(journal_path, journal)
                _restore_install(stage / rel, target)
            _verify_restore_tree(root, hashes, archive_path)
        print(f"restore: {len(plan)} file(s) landed (layout: {prefix or 'root-level'}, version {version})")
        return 0
    except BaseException as exc:
        errors = []
        if recovery is not None:
            for rel, existed in reversed(attempted):
                try:
                    if existed:
                        rollback_copy = recovery / "rollback" / rel
                        rollback_copy.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(recovery / "backups" / rel, rollback_copy)
                        os.replace(rollback_copy, root / rel)
                        restored_stat = (root / rel).stat()
                        if (_sha256(root / rel) != backup_hashes[rel.as_posix()]
                                or (stat.S_IMODE(restored_stat.st_mode), restored_stat.st_mtime_ns)
                                != backup_metadata[rel.as_posix()]):
                            raise OSError("restored backup verification failed")
                    else:
                        (root / rel).unlink(missing_ok=True)
                except OSError as rollback_exc:
                    errors.append(f"{rel}: {rollback_exc}")
            for directory in reversed(created_dirs):
                try:
                    if directory.exists():
                        directory.rmdir()
                except OSError as rollback_exc:
                    errors.append(f"{directory}: {rollback_exc}")
        if errors:
            keep_recovery = True
            print(f"restore: ROLLBACK INCOMPLETE -- retain recovery directory {recovery}")
            for error in errors:
                print(f"  ! {error}")
        print(f"restore: REFUSED or failed -- {exc}")
        if not isinstance(exc, Exception):
            raise
        return 4
    finally:
        if recovery is not None and not keep_recovery:
            shutil.rmtree(recovery, ignore_errors=True)


def restore(dry_run: bool = False, **limits: float) -> int:
    """Unpack the newest surviving zip over the tree, then light-commit."""
    newest = _newest_zip()
    if newest is None:
        print("restore: no surviving archive found")
        return 4
    if not newest.is_file() and dry_run:
        print(f"restore: {newest} is not on disk; dry run will not materialise it")
        return 4
    if not newest.is_file():
        # tracked in git but gone from disk -- materialise it first
        _git("checkout", "HEAD", "--", str(newest.relative_to(REPO)), check=False)
    if not newest.is_file():
        print(f"restore: {newest} is not recoverable")
        return 4
    mode = " (dry run)" if dry_run else ""
    print(f"restore: unpacking {newest.name} over {ROOT}{mode}")
    rc = _restore_archive(newest, ROOT, dry_run=dry_run, **limits)
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
    group.add_argument(
        "--validate", metavar="CHECKPOINT_JSON",
        help="validate one checkpoint file against the Part-5 §13 v2 envelope "
        "(read-only; exit 0 only when fully valid)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="with --restore: preview the member -> target plan, write nothing",
    )
    parser.add_argument("--max-members", type=int, default=None, help="restore archive member limit")
    parser.add_argument("--max-file-size", type=int, default=None, help="restore per-file uncompressed bytes")
    parser.add_argument("--max-total-size", type=int, default=None, help="restore total uncompressed bytes")
    parser.add_argument("--max-ratio", type=float, default=None, help="restore maximum compression ratio")
    args = parser.parse_args()
    limits = {name: getattr(args, name) for name in RESTORE_LIMITS if getattr(args, name) is not None}
    if limits and not args.restore:
        parser.error("restore limits only make sense together with --restore")
    if args.dry_run and not args.restore:
        parser.error("--dry-run only makes sense together with --restore")
    if args.doctor:
        return doctor()
    if args.light:
        return light()
    if args.restore:
        return restore(dry_run=args.dry_run, **limits)
    if args.validate is not None:
        return validate(args.validate)
    return ship(args.commit)


if __name__ == "__main__":
    sys.exit(main())
