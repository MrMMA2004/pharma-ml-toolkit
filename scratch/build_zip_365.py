#!/usr/bin/env python3
"""round-365 (the 157th recorded run, the part18 campaign, trace
1a218363342face5) -- the v1.2.242 ship: the clustering/anomaly/RL
lanes with FULL backend->API->frontend integration, the MANDATORY
studio.py retirement (the quarantined legacy/ area), docs/DESIGN.md,
the notebooks re-executed at 1.2.242, and THIS script ships the new
zip to the six homes + the offsite mirror.

Same recipe as r303..r329 (the canonical exclusion list, the §4.3
hygiene, the composition guard, the leak scan, the fresh-build guard,
the version_info quadruple) with the six-home parity (download /
toolkit download / toolkit dist-archive / upload / root dist-archive /
offsite) + the offsite inventory.
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
    REPO / "dist-archive",
]


def main() -> int:
    target = HOMES[0] / NAME
    if target.exists():
        target.unlink()
    print(f"building {NAME} (the r365 part18 ship) ...")
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
        # the standing core (the r308..r321 proofs, unchanged):
        "frontend/src/services/artifacts/client.ts",
        "frontend/src/services/database/client.ts",
        "docs/PROJECT_STATUS.md",
        "contracts/openapi.json",
        "tests/test_architecture_hygiene.py",
        "packaging/sbom/sbom.cdx.json",
        "src/pharmaml/api/routes/admet.py",
        "tests/api/test_admet.py",
        "src/pharmaml/api/routes/dta.py",
        "tests/api/test_dta.py",
        "src/pharmaml/api/routes/repurposing.py",
        "tests/api/test_repurposing.py",
        "src/pharmaml/api/routes/drugresponse.py",
        "tests/api/test_drugresponse.py",
        "src/pharmaml/api/routes/pharmacogenomics.py",
        "tests/api/test_pharmacogenomics.py",
        "src/pharmaml/api/routes/multiobjective.py",
        "tests/api/test_multiobjective.py",
        "src/pharmaml/workflow/engine.py",
        "tests/test_workflow_engine.py",
        "src/pharmaml/api/routes/system.py",
        "tests/api/test_system.py",
        "tests/test_security_supply_chain.py",
        "tests/test_documentation_consistency.py",
        "docs/gui/screenshots/r325-pins-desktop.png",
        "docs/gui/screenshots/r325-pins-mobile.png",
        # the r327/r328 proofs (the D-1..D-5 incident fixes + the
        # 34-leg regression matrix + the QA Phase-1 audit + register):
        "src/pharmaml/api/security.py",
        "src/pharmaml/api/__init__.py",
        "frontend/src/services/api/startup.ts",
        "frontend/src/services/api/__tests__/startup.test.ts",
        "src-tauri/src/backend_manager.rs",
        "tests/api/test_trusted_origins.py",
        "docs/QA/PART14_PHASE1_AUDIT.md",
        "docs/QA/DEFECT_REGISTER.md",
        "docs/gui/screenshots/p14r326-pins-desktop.png",
        "docs/gui/screenshots/p14r328-pins-desktop.png",
        "docs/gui/screenshots/p14r328-pins-mobile.png",
        # the r329 proofs (the push-success round: the QA deliverable
        # set's remaining eight documents + the r329 pins pair):
        "docs/QA/MASTER_TEST_PLAN.md",
        "docs/QA/MANUAL_TEST_CHECKLIST.md",
        "docs/QA/AUTOMATED_TEST_MATRIX.md",
        "docs/QA/WINDOWS_TEST_MATRIX.md",
        "docs/QA/PERFORMANCE_BASELINES.md",
        "docs/QA/SECURITY_TEST_REPORT.md",
        "docs/QA/BLOCKED_TESTS_AND_SOLUTIONS.md",
        "docs/QA/RELEASE_ACCEPTANCE_REPORT.md",
        "docs/gui/screenshots/p14r329-pins-desktop.png",
        "docs/gui/screenshots/p14r329-pins-mobile.png",
        # the r365 proofs (part18: the three new lanes + the retirement):
        "src/pharmaml/ml/clustering.py",
        "src/pharmaml/ml/anomaly.py",
        "src/pharmaml/ml/reinforce.py",
        "src/pharmaml/api/routes/unsupervised.py",
        "src/pharmaml/api/routes/reinforce.py",
        "tests/test_clustering.py",
        "tests/test_anomaly.py",
        "tests/test_reinforce.py",
        "tests/api/test_unsupervised_reinforce.py",
        "tests/test_legacy_retirement.py",
        "frontend/src/features/ml/UnsupervisedLearningLane.tsx",
        "frontend/src/features/ml/__tests__/UnsupervisedReinforceLane.test.tsx",
        "frontend/src/services/ml/unsupervised.ts",
        "legacy/studio.py",
        "legacy/README.md",
        "docs/DESIGN.md",
        "docs/CAPABILITY_INTEGRATION_MATRIX.md",
    ]
    missing = [n for n in must_carry if n not in names]
    print(f"must-carry gate: {missing!r}")
    if missing:
        print("MUST-CARRY FAIL -- refusing to ship")
        return 1

    retired = [n for n in names if n.startswith("playground.py")
               or n.startswith("tests/test_playground.py")]
    # the r365 retirement gate: no root-level studio.py, the quarantined
    # legacy copy present, the quarantine README present
    if "studio.py" in names or any(n == "studio.py" for n in names):
        print("ROOT studio.py LEAKED INTO ARCHIVE -- refusing to ship")
        return 1
    if "legacy/studio.py" not in names or "legacy/README.md" not in names:
        print("LEGACY QUARANTINE MISSING FROM ARCHIVE -- refusing to ship")
        return 1
    print(f"retired-pair gate (must be empty): {retired!r}")
    if retired:
        print("RETIRED-PAIR STILL IN ARCHIVE -- refusing to ship")
        return 1
    if any(n.startswith("frontend/dist/") for n in names):
        print("FRONTEND/DIST LEAKED INTO ARCHIVE -- refusing to ship")
        return 1
    if any(n.startswith(".github/") for n in names):
        print("STALE .GITHUB DUPLICATE LEAKED INTO ARCHIVE -- refusing")
        return 1
    if any(n.startswith("build/theme-shots/") for n in names):
        print("THEME-SHOT STRAYS LEAKED INTO ARCHIVE -- refusing")
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
        home.mkdir(parents=True, exist_ok=True)
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

    # the offsite mirror + this round's zip
    OFFSITE.mkdir(exist_ok=True)
    off = OFFSITE / NAME
    off.write_bytes(data)
    with zipfile.ZipFile(off) as zf:
        assert zf.testzip() is None, f"testzip failed in {off}"
    print(f"offsite (testzip OK): {off}")
    print(f"offsite inventory: {len(list(OFFSITE.glob('*.zip')))} zips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
