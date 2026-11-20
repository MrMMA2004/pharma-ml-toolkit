"""round-303 -- the part12 Phase-2 archive RE-SHIP for v1.2.231.

Same recipe as r302 (the route's canonical exclusion list, the §4.3
hygiene, the composition guard, the leak scan) with the round-303 gates:

  * the must-carry set extends to the round's own proofs: the Williams
    route (applicability_domain.py), the chirality route (chemistry.py),
    the /evaluate job-mode fix (semi_supervised.py), and the frontend
    consumers (MLPage.tsx + the ml client) ride the archive -- plus the
    standing B-1/B-2/audit-doc pins from r302;
  * no version bump -- the new bytes REPLACE the stale 231 in all five
    homes (the r300-standard same-version re-ship; the v1.2.231 release
    EXE/MSI stay THE release, only the source archive re-ships);
  * the offsite home re-seeded from the 221-231 history (the rollback
    wiped it again): 221-228 from the in-repo dist-archive copies, 229
    from git c7989dc9, 230 from git 62fac28e, 231 from this build.

VERSION is read dynamically; testzip integrity is checked per home; the
leak scan refuses absolute home paths and token shapes.
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
OFFSITE = Path.home() / "pharma-ml-toolkit-zips"

EXCLUDES = [
    # the route's canonical list (toolkit-fs.ts ZIP_EXCLUDE_ARGS), round 302:
    # the artifacts pair is now SCOPED (B-1) and frontend/dist rides (B-1b)
    ".pytest_cache/*", "*/.pytest_cache/*", "__pycache__/*", "*/__pycache__/*",
    "*.pyc", "catboost_info/*", "*/catboost_info/*",
    "*.log", "*/*.log",
    "artifacts/*", "examples/artifacts/*", "frontend/dist/*",
    "build/theme-shots/*",
    "node_modules/*", "*/node_modules/*",
    "*.egg-info", "*.egg-info/*", "*/src/*.egg-info/*",
    "*.zip",
    ".venv/*", "*/.venv/*", "venv/*", "*/venv/*", "env/*", "*/env/*",
    ".git/*", "*/.git/*",
    ".ruff_cache/*", "*/.ruff_cache/*", ".mypy_cache/*", "*/.mypy_cache/*",
    ".tox/*", "*/.tox/*", ".idea/*", "*/.idea/*", ".vscode/*", "*/.vscode/*",
    # §4.3 (round 260, v1.2.167): the archive ships the toolkit only --
    # audit trails, screenshots and the archive homes themselves stay out.
    "scratch/*", "*/scratch/*",
    "download/*", "*/download/*",
    "dist-archive/*", "*/dist-archive/*",
]

HOMES = [
    REPO / "download",
    TOOLKIT / "dist-archive",
    TOOLKIT / "download",
    REPO / "upload",
]

# the offsite history: 229/230 only exist in git
OFFSITE_FROM_GIT = {
    "v1.2.229": ("c7989dc9", "download/pharma-ml-toolkit-v1.2.229.zip"),
    "v1.2.230": ("62fac28e", "download/pharma-ml-toolkit-v1.2.230.zip"),
}


def main() -> int:
    target = HOMES[0] / NAME
    # (r303 lesson) zip -r UPDATES an existing archive without dropping
    # entries that no longer exist on disk -- the first r303 build's
    # theme-shot PNGs survived the sweep + exclusion as stale members.
    # The ship must always build FRESH.
    if target.exists():
        target.unlink()
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

    # the round-303 must-carry gate: the standing r302 pins PLUS the
    # round's own proof files (the Williams route, the chirality route,
    # the /evaluate fix, and both frontend consumers)
    must_carry = [
        "frontend/src/services/artifacts/client.ts",
        "frontend/src/services/database/client.ts",
        "docs/part12_audit/COMPREHENSIVE_PROJECT_AUDIT.md",
        "src/pharmaml/api/routes/applicability_domain.py",
        "src/pharmaml/api/routes/chemistry.py",
        "src/pharmaml/api/routes/semi_supervised.py",
        "src/pharmaml/ml/applicability.py",
        "src/pharmaml/viz/ml_plots.py",
        "frontend/src/services/ml/client.ts",
        "frontend/src/features/ml/MLPage.tsx",
        "frontend/src/features/datasets/DatasetsPage.tsx",
        "contracts/openapi.json",
    ]
    missing = [n for n in must_carry if n not in names]
    print(f"must-carry gate: {missing!r}")
    if missing:
        print("MUST-CARRY FAIL -- refusing to ship")
        return 1
    if any(n.startswith("frontend/dist/") for n in names):
        print("FRONTEND/DIST LEAKED INTO ARCHIVE -- refusing to ship")
        return 1

    # the version_info quadruple gate (born from the r301 fire-1 lesson)
    with zipfile.ZipFile(target) as zf:
        vi = zf.read("packaging/windows/version_info.txt").decode("utf-8")
    quad = "(" + VERSION.replace(".", ",") + ",0)"
    vi_flat = vi.replace(" ", "")
    vi_ok = (
        f"filevers={quad}" in vi_flat
        and f"prodvers={quad}" in vi_flat
        and f"'FileVersion','{VERSION}.0'" in vi_flat
        and f"'ProductVersion','{VERSION}'" in vi_flat
    )
    print(f"version_info.txt in-archive carries {VERSION}: {vi_ok}")
    if not vi_ok:
        print("VERSION_INFO STILL STALE IN ARCHIVE -- refusing to ship")
        return 1

    # byte-identical copies into every in-repo home (replacing the stale
    # r302 zips of the same name)
    for home in HOMES[1:]:
        copy = home / NAME
        copy.write_bytes(data)
        with zipfile.ZipFile(copy) as zf:
            assert zf.testzip() is None, f"testzip failed in {copy}"
        print(f"home (testzip OK): {copy}")

    # the composition guard: the §4.3 exclusions must hold
    forbidden = ("scratch/", "download/", "dist-archive/")
    violations = sorted(n for n in names if n.split("/")[0] in forbidden)
    print(f"composition violations: {len(violations)}")
    for name in violations[:10]:
        print("  VIOLATION:", name)

    # the leak scan: no absolute home paths, no REAL token shapes in
    # TEXT members. The intentional security files name the token
    # prefixes they scan for -- those bare pattern literals are the
    # r301-documented six hits, counted and printed, never blocking;
    # the GATE needs token SHAPES (a real 20+ char secret), not the
    # prefix strings. node_modules is excluded wholesale, so
    # third-party minified AKIA fragments cannot ride either.
    import re
    TOKEN_PATTERNS = [
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"/home/z/"),
    ]
    leaks = []
    pattern_literals = 0
    with zipfile.ZipFile(target) as zf:
        for name in files:
            if name.endswith((".png", ".jpg", ".zip", ".db", ".pcap", ".parquet")):
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            for pat in TOKEN_PATTERNS:
                if pat.search(text):
                    leaks.append((name, pat.pattern))
            for needle in ("github_pat_", "ghp_"):
                if needle in text:
                    pattern_literals += 1
    print(f"real leaks: {len(leaks)}")
    for name, needle in leaks[:10]:
        print("  LEAK:", name, needle)
    print(f"pattern-literal mentions (documented, non-blocking): {pattern_literals}")

    # VERSION stamp parity: the archive must carry the live VERSION
    with zipfile.ZipFile(target) as zf:
        stamp = zf.read("VERSION").decode().strip()
    print(f"archive VERSION stamp: {stamp}")
    if stamp != VERSION:
        print("VERSION STAMP MISMATCH")
        return 1
    if leaks or violations or bad is not None:
        return 1

    # the fifth offsite home: re-seed the 221-231 history (the rollback
    # wiped it), then place this round's zip on top
    OFFSITE.mkdir(exist_ok=True)
    seeded = 0
    for ver in [f"v1.2.{n}" for n in range(221, 229)]:
        src = TOOLKIT / "dist-archive" / f"pharma-ml-toolkit-{ver}.zip"
        dst = OFFSITE / f"pharma-ml-toolkit-{ver}.zip"
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())
            seeded += 1
    for ver, (ref, path) in OFFSITE_FROM_GIT.items():
        dst = OFFSITE / f"pharma-ml-toolkit-{ver}.zip"
        if not dst.exists():
            blob = subprocess.run(
                ["git", "show", f"{ref}:{path}"],
                cwd=REPO, capture_output=True, check=True,
            ).stdout
            dst.write_bytes(blob)
            seeded += 1
    print(f"offsite history re-seeded: {seeded} zips from dist-archive/git")
    off = OFFSITE / NAME
    off.write_bytes(data)
    with zipfile.ZipFile(off) as zf:
        assert zf.testzip() is None, f"testzip failed in {off}"
    print(f"offsite (testzip OK): {off}")
    print(f"offsite inventory: {len(list(OFFSITE.glob('*.zip')))} zips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
