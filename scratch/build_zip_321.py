#!/usr/bin/env python3
"""round-321 -- the part13 SAME-VERSION re-ship for v1.2.240 (the
r303/r304/r317/r319 convention): the Phase-5 multi-objective
optimization GUI leg (the "Multi-objective Optimization" stage, stage
16, the later stages renumbered 17/18/19 + the 8-test MultiobjectiveLane
suite + the r321 evidence pair) rides the rebuilt archive at the
converged fixed point.

Same recipe as r303..r319 (the canonical exclusion list, the §4.3
hygiene, the composition guard, the leak scan, the fresh-build guard,
the version_info quadruple) with the six-home parity (download /
toolkit download / toolkit dist-archive / upload / root dist-archive /
offsite) + the 18-zip offsite inventory.

The pins pair already exists on disk (the placeholders closed the
census fixed point before the capture), so the single build here is
the TERMINAL one -- the must-carry requires all four r321 screenshots.
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
    print(f"building {NAME} (the r321 terminal same-version re-ship) ...")
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
        "src/pharmaml/api/routes/model_evaluation.py",
        "src/pharmaml/api/routes/dl.py",
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
        "tests/api/test_model_evaluation.py",
        "docs/gui/screenshots/r312-pins-desktop.png",
        "docs/gui/screenshots/r312-pins-mobile.png",
        # the r313 proofs (the Phase-5 GUI lane over the GNN API):
        "frontend/src/services/dl/client.ts",
        "frontend/src/features/dl/DLPage.tsx",
        "frontend/src/features/dl/__tests__/GnnLane.test.tsx",
        "docs/gui/screenshots/r313-gnn-train-card-desktop.png",
        "docs/gui/screenshots/r313-gnn-predict-table-desktop.png",
        "docs/gui/screenshots/r313-pins-desktop.png",
        "docs/gui/screenshots/r313-pins-mobile.png",
        # the r314 proofs (the Phase-5 ADMET lane -- both legs):
        "src/pharmaml/api/routes/admet.py",
        "src/pharmaml/ml/admet.py",
        "tests/api/test_admet.py",
        "frontend/src/features/ml/__tests__/AdmetLane.test.tsx",
        "docs/gui/screenshots/r314-admet-card-desktop.png",
        "docs/gui/screenshots/r314-admet-invalid-desktop.png",
        "docs/gui/screenshots/r314-pins-desktop.png",
        "docs/gui/screenshots/r314-pins-mobile.png",
        # the r315 proofs (the Phase-5 DTA lane -- both legs):
        "src/pharmaml/api/routes/dta.py",
        "src/pharmaml/ml/dta.py",
        "tests/api/test_dta.py",
        "frontend/src/features/ml/__tests__/DtaLane.test.tsx",
        "docs/gui/screenshots/r315-dta-card-desktop.png",
        "docs/gui/screenshots/r315-dta-invalid-desktop.png",
        # the r316 proofs (the Phase-5 Drug Repurposing lane -- both legs):
        "src/pharmaml/api/routes/repurposing.py",
        "src/pharmaml/ml/response_model.py",
        "tests/api/test_repurposing.py",
        "frontend/src/features/ml/__tests__/RepurposingLane.test.tsx",
        "docs/gui/screenshots/r316-repurposing-card-desktop.png",
        "docs/gui/screenshots/r316-repurposing-gate-desktop.png",
        # the r317 proofs (the Phase-5 Cancer Cell-Line Drug Response
        # lane -- both legs):
        "src/pharmaml/api/routes/drugresponse.py",
        "src/pharmaml/ml/drugresponse.py",
        "tests/api/test_drugresponse.py",
        "frontend/src/features/ml/__tests__/DrugResponseLane.test.tsx",
        "docs/gui/screenshots/r317-drugresponse-card-desktop.png",
        "docs/gui/screenshots/r317-drugresponse-gate-desktop.png",
        # the r318 proofs (the Phase-5 Pharmacogenomics lane -- both legs):
        "src/pharmaml/api/routes/pharmacogenomics.py",
        "tests/api/test_pharmacogenomics.py",
        "frontend/src/features/ml/__tests__/PharmacogenomicsLane.test.tsx",
        "docs/gui/screenshots/r318-pharmacogenomics-card-desktop.png",
        "docs/gui/screenshots/r318-pharmacogenomics-gate-desktop.png",
        "docs/gui/screenshots/r318-pins-desktop.png",
        "docs/gui/screenshots/r318-pins-mobile.png",
        # the r319 proofs (the Phase-5 multiobjective API family the
        # interrupted continuation left, absorbed and gated):
        "src/pharmaml/api/routes/multiobjective.py",
        "src/pharmaml/ml/multiobjective.py",
        "src/pharmaml/viz/multiobjective_plots.py",
        "tests/api/test_multiobjective.py",
        "tests/test_multiobjective_hpo.py",
        # the r321 proofs (the Phase-5 multiobjective GUI leg -- the
        # domain's CLOSE; the pins pair rides the terminal build):
        "frontend/src/features/ml/__tests__/MultiobjectiveLane.test.tsx",
        "docs/gui/screenshots/r321-multiobjective-card-desktop.png",
        "docs/gui/screenshots/r321-multiobjective-gate-desktop.png",
        "docs/gui/screenshots/r321-pins-desktop.png",
        "docs/gui/screenshots/r321-pins-mobile.png",
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

    # the offsite mirror (the 18-zip inventory) + this round's zip
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
