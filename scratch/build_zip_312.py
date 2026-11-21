#!/usr/bin/env python3
"""round-312 -- the part13 archive SHIP for v1.2.234.

Same recipe as r303..r311 (the canonical exclusion list, the §4.3
hygiene, the composition guard, the leak scan, the fresh-build guard)
with the round-312 gates:

  * a NEW version ship: v1.2.234 carries the part13 Phase-4 API
    connection (the /api/v1/model-evaluation conformal + coverage
    family) + the Phase-5 GNN/MPNN API lane (dl/gnn-train +
    dl/gnn-predict + the guards);
  * the must-carry gate adds this round's own proof files (the new
    route family + its 30-test file);
  * the standing gates re-asserted: the NEGATIVE retired-pair gate,
    the version_info quadruple, the composition + leak scans;
  * the offsite history top-up generalized (221-228 + 231-233 all
    live tracked in dist-archive now — plus the 229/230 git-blob
    re-derivation if ever needed), then this round's zip on top.
    The sixth rollback left the offsite INTACT (13 zips) — the
    top-up should seed nothing; it is the honest no-op check.
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

OFFSITE_FROM_GIT = {
    "v1.2.229": ("c7989dc9", "download/pharma-ml-toolkit-v1.2.229.zip"),
    "v1.2.230": ("62fac28e", "download/pharma-ml-toolkit-v1.2.230.zip"),
}


def main() -> int:
    target = HOMES[0] / NAME
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
        "tests/test_architecture_hygiene.py",
        "packaging/sbom/sbom.cdx.json",
        "src/pharmaml/biology/association.py",
        "src/pharmaml/ml/pharmacogenomics.py",
        "src/pharmaml/pharma/fih.py",
        "src/pharmaml/pharma/pbpk.py",
        "src/pharmaml/pkpd/pbpk.py",
        "src/pharmaml/ml/dataset.py",
        "src/pharmaml/ml/benchmark.py",
        # the r307 proof:
        "docs/PROJECT_STATUS.md",
        # the r309 proofs:
        "docs/security/INSTALLER_SIGNING_ASSESSMENT.md",
        "docs/part12_audit/WINDOWS_CLEAN_INSTALL_REPORT.md",
        # the r310 proofs (the component-test lane):
        "frontend/vitest.config.ts",
        "frontend/src/test/setup.ts",
        "frontend/src/app/__tests__/App.test.tsx",
        "frontend/src/app/__tests__/StartupGate.test.tsx",
        "frontend/src/app/__tests__/workspaces.test.ts",
        "frontend/src/components/ui/__tests__/atoms.test.tsx",
        "frontend/src/components/charts/__tests__/PlotlyFigure.test.tsx",
        "frontend/src/features/datasets/__tests__/DatasetsPage.test.tsx",
        "frontend/src/services/websocket/__tests__/client.test.ts",
        # the r311 proofs (the §9.3 scaffold-aware coverage bridge
        # + the Phase-3 browser-E2E evidence):
        "src/pharmaml/ml/scaffold_coverage.py",
        "tests/test_scaffold_coverage.py",
        # the r312 proofs (the Phase-4 API connection + the Phase-5
        # GNN/MPNN API lane):
        "src/pharmaml/api/routes/model_evaluation.py",
        "tests/api/test_model_evaluation.py",
        "docs/gui/screenshots/r312-pins-desktop.png",
        "docs/gui/screenshots/r312-pins-mobile.png",
        "docs/gui/screenshots/r311-boot-introduction-desktop.png",
        "docs/gui/screenshots/r311-generative-verdict-desktop.png",
        "docs/gui/screenshots/r311-generative-error-422-desktop.png",
        "docs/gui/screenshots/r311-visualization-stat-inspect-desktop.png",
        "docs/gui/screenshots/r311-visualization-registry-desktop.png",
    ]
    missing = [n for n in must_carry if n not in names]
    print(f"must-carry gate: {missing!r}")
    if missing:
        print("MUST-CARRY FAIL -- refusing to ship")
        return 1

    retired = [n for n in names if n.startswith("playground.py")
               or n.startswith("tests/test_playground.py")]
    print(f"retired-pair gate (must be empty): {retired!r}")
    if retired:
        print("RETIRED-PAIR STILL IN ARCHIVE -- refusing to ship")
        return 1
    if any(n.startswith("frontend/dist/") for n in names):
        print("FRONTEND/DIST LEAKED INTO ARCHIVE -- refusing to ship")
        return 1

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

    for home in HOMES[1:]:
        copy = home / NAME
        copy.write_bytes(data)
        with zipfile.ZipFile(copy) as zf:
            assert zf.testzip() is None, f"testzip failed in {copy}"
        print(f"home (testzip OK): {copy}")

    forbidden = ("scratch/", "download/", "dist-archive/")
    violations = sorted(n for n in names if n.split("/")[0] in forbidden)
    print(f"composition violations: {len(violations)}")
    for name in violations[:10]:
        print("  VIOLATION:", name)

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

    with zipfile.ZipFile(target) as zf:
        stamp = zf.read("VERSION").decode().strip()
    print(f"archive VERSION stamp: {stamp}")
    if stamp != VERSION:
        print("VERSION STAMP MISMATCH")
        return 1
    if leaks or violations or bad is not None:
        return 1

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
    for ver in ("v1.2.231", "v1.2.232", "v1.2.233"):
        dst = OFFSITE / f"pharma-ml-toolkit-{ver}.zip"
        src = TOOLKIT / "dist-archive" / f"pharma-ml-toolkit-{ver}.zip"
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())
            seeded += 1
    print(f"offsite history topped up: {seeded} zips")
    off = OFFSITE / NAME
    off.write_bytes(data)
    with zipfile.ZipFile(off) as zf:
        assert zf.testzip() is None, f"testzip failed in {off}"
    print(f"offsite (testzip OK): {off}")
    print(f"offsite inventory: {len(list(OFFSITE.glob('*.zip')))} zips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
