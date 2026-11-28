#!/usr/bin/env python3
"""round-384 (the 177th recorded run, the WORKFLOW_ENHANCEMENT_ROADMAP
continuation, trace 1a2228bc4dbfa515) -- the v1.2.256 ship: the fifth
3-item set executed end-to-end IN ONE ROUND ('3 rows one after another')
-- DL-3 (the efficiency lab: POST /api/v1/dl/efficiency over the real
quantization + pruning engines, the pruned-COPY metric delta, the table
artifact), DL-4 (the D-MPNN molecular lane: POST /api/v1/dl/mpnn-train +
mpnn-predict over the real numpy chemprop-style engine, the seeded
held-out carve with the pre-registered tolerance, the honest
rejected-rows accounting), GEN-3 (the REINVENT-style RL optimizer: POST
/api/v1/generative/rl-optimize over the real run_rl_optimization engine
with the engine's own new on_iteration cancellation seam, the third
method card + the step curve). The contract re-exported at 152 paths /
107 schemas, all four count pins trued.
NO release: no tag, no installer; v1.2.244 stands published.

Same recipe as r303..r379 (the canonical exclusion list, the §4.3
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
    print(f"building {NAME} (the r384 fifth-3-item-set ship) ...")
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
        # the r367 proofs (part18 §5: the five DL families + the §9
        # chain; part18 §10: the ui-quarantine completion):
        "src/pharmaml/ml/transformer.py",
        "src/pharmaml/ml/cnn.py",
        "src/pharmaml/ml/autoencoder.py",
        "src/pharmaml/generative/gan.py",
        "src/pharmaml/generative/diffusion.py",
        "src/pharmaml/api/routes/dl.py",
        "src/pharmaml/api/routes/generative.py",
        "tests/test_transformer.py",
        "tests/test_cnn.py",
        "tests/test_autoencoder.py",
        "tests/test_generative_gan_diffusion.py",
        "tests/api/test_dl_architectures.py",
        "frontend/src/services/dl/client.ts",
        "frontend/src/features/dl/DLPage.tsx",
        "frontend/src/features/dl/__tests__/CnnAutoencoderLane.test.tsx",
        "frontend/src/services/generative/client.ts",
        "frontend/src/features/generative/GenerativePage.tsx",
        "frontend/src/features/generative/__tests__/FeatureGenerativeLane.test.tsx",
        # the quarantined GUI tree rides the archive (archival
        # reproducibility; unsupported status pinned by legacy/README):
        "legacy/ui/__init__.py",
        "legacy/ui/headless.py",
        "legacy/ui/native/libEGL.so.1",
        "legacy/tests_ui/conftest.py",
        # the r368 proofs (the 13-phase audit: the five reports, the
        # clean-machine suite, the F-1/F-3 UI fixes riding the
        # installers for the first time):
        "docs/QA/ROUND368_BASELINE_REPORT.md",
        "docs/QA/ROUND368_PR_RECONCILIATION.md",
        "docs/QA/ROUND368_API_TRACEABILITY_MATRIX.md",
        "docs/QA/ROUND368_SCIENTIFIC_VALIDATION_REPORT.md",
        "docs/QA/ROUND368_FINAL_REPORT.md",
        "scripts/windows/validate-clean-machine.ps1",
        "docs/QA/WINDOWS_CLEAN_INSTALL_REPORT.md",
        "tests/test_ml_catalog.py",
        # the r370 proofs (part20: the six QA deliverables + the
        # D-9 chemistry-lane fix + the D-10 Cargo.toml alignment):
        "docs/QA/FULL_REPOSITORY_AUDIT.md",
        "docs/QA/BACKEND_FRONTEND_TRACEABILITY.md",
        "docs/QA/WINDOWS_EXE_MSI_AUDIT.md",
        "docs/QA/TEST_EXECUTION_REPORT.md",
        "docs/RELEASE_READINESS.md",
        "src-tauri/Cargo.toml",
        "tests/test_docking_prep.py",
        # the r371 carrier proofs (the release bump itself):
        "VERSION",
        "CHANGELOG.md",
        "packaging/windows/version_info.txt",
        # the r372/r373 proofs (part21: the shared ML/DL evaluation
        # layer + the revalidation + the roadmap):
        "src/pharmaml/api/evaluation_curves.py",
        "tests/api/test_ml_predict_evaluation_curves.py",
        "frontend/src/components/evaluation/EvaluationCharts.tsx",
        "frontend/src/components/evaluation/ParityChart.tsx",
        "frontend/src/components/evaluation/ConfusionChart.tsx",
        "frontend/src/components/evaluation/EvalRowInspector.tsx",
        "frontend/src/components/evaluation/ResidualHistogramChart.tsx",
        "frontend/src/components/evaluation/QqPlotChart.tsx",
        "frontend/src/components/evaluation/RocCurveChart.tsx",
        "frontend/src/components/evaluation/PrCurveChart.tsx",
        "frontend/src/components/evaluation/CalibrationChart.tsx",
        "frontend/src/components/evaluation/ClassMetricsTable.tsx",
        "frontend/src/components/evaluation/exportRows.ts",
        "frontend/src/components/evaluation/formatValue.ts",
        "frontend/src/components/training/LossCurveChart.tsx",
        "frontend/src/components/evaluation/__tests__/EvaluationCharts.test.tsx",
        "frontend/src/features/dl/__tests__/EvaluateStageCharts.test.tsx",
        "docs/QA/ROUND372_PREVIOUS_TASK_REVALIDATION.md",
        "docs/QA/ML_DL_VISUALIZATION_UNIFICATION.md",
        "docs/QA/ML_DL_VISUALIZATION_TEST_REPORT.md",
        "docs/QA/WORKFLOW_ENHANCEMENT_ROADMAP.md",
        # the r374 proofs (DS-1, the roadmap's row 1 — the strategy
        # lane at the Datasets stage):
        "src/pharmaml/api/routes/datasets.py",
        "frontend/src/services/datasets/client.ts",
        "frontend/src/features/datasets/DatasetsPage.tsx",
        "frontend/src/features/datasets/__tests__/DatasetsPage.test.tsx",
        "tests/api/test_datasets_ml_artifacts.py",
        # the r375 proofs (ML-3, the roadmap's row 2 — the split
        # section inside the Train lane):
        "src/pharmaml/api/routes/ml.py",
        "src/pharmaml/api/runs.py",
        "src/pharmaml/ml/splits.py",
        "frontend/src/services/ml/client.ts",
        "frontend/src/services/ml/__tests__/train-split.test.ts",
        "frontend/src/features/ml/MLPage.tsx",
        "frontend/src/features/ml/__tests__/TrainSplitLane.test.tsx",
        "tests/test_frontend_split_strategy.py",
        "tests/test_frontend_multiresponse.py",
        "tests/test_frontend_rsm.py",
        "tests/api/test_contracts.py",
        # the r376 proofs (ML-1, the roadmap's row 3 -- the
        # model-evaluation family's frontend consumer):
        "src/pharmaml/api/routes/model_evaluation.py",
        "tests/api/test_model_evaluation.py",
        "frontend/src/services/ml/__tests__/model-evaluation.test.ts",
        "frontend/src/features/ml/__tests__/ConformalLane.test.tsx",
        "tests/test_frontend_model_evaluation.py",
        "tests/test_ml_workspace.py",
        "frontend/src/lib/version.ts",
        "tests/test_frontend_settings.py",
        # the r377 proofs (GEN-1, the roadmap's row 4 -- the GA's
        # operator modes reachable from the optimize route):
        "src/pharmaml/generative/genetic.py",
        "src/pharmaml/generative/scaffold_hop.py",
        "src/pharmaml/api/routes/generative.py",
        "tests/api/test_dl_semi_generative.py",
        "frontend/src/services/generative/client.ts",
        "frontend/src/features/generative/GenerativePage.tsx",
        "frontend/src/features/generative/__tests__/OperatorModesLane.test.tsx",
        "tests/test_generative_workspace.py",
        # the r378 proofs (GEN-2, the roadmap's row 5 -- the MOSES
        # generation-metrics route + the Validate stage's metrics card):
        "src/pharmaml/generative/moses_metrics.py",
        "frontend/src/features/generative/__tests__/MetricsLane.test.tsx",
        # the r379 proofs (DS-2, the roadmap's row 6 -- the Clean
        # stage's standardization verb + the derived-dataset lane):
        "src/pharmaml/chem/standardize.py",
        "src/pharmaml/api/routes/chemistry.py",
        "src/pharmaml/api/stores.py",
        "tests/test_frontend_standardize_action.py",
        "tests/test_frontend_registry.py",
        "tests/test_frontend_runs.py",
        # the r383 proofs (the fourth 3-item set: ML-5 + DL-1 + DL-2):
        "tests/api/test_ml_drift.py",
        "tests/api/test_dl_tune.py",
        "tests/api/test_dl_gnn_explain.py",
        "tests/test_frontend_ml_drift.py",
        "tests/test_frontend_dl_tune.py",
        "tests/test_frontend_dl_gnn_explain.py",
        "frontend/src/features/ml/__tests__/DriftLane.test.tsx",
        "frontend/src/features/dl/__tests__/TuneLane.test.tsx",
        "frontend/src/features/dl/__tests__/GnnExplainLane.test.tsx",
        "frontend/src/services/ml/__tests__/drift.test.ts",
        "frontend/src/services/dl/__tests__/dl-tune.test.ts",
        "frontend/src/services/dl/__tests__/gnn-explain.test.ts",
        # the r384 proofs (the fifth 3-item set: DL-3 + DL-4 + GEN-3):
        "src/pharmaml/ml/quantization.py",
        "src/pharmaml/ml/pruning.py",
        "src/pharmaml/ml/mpnn.py",
        "src/pharmaml/generative/rl.py",
        "tests/api/test_dl_efficiency.py",
        "tests/api/test_dl_mpnn.py",
        "tests/api/test_generative_rl.py",
        "tests/test_frontend_dl_efficiency.py",
        "tests/test_frontend_dl_mpnn.py",
        "tests/test_frontend_generative_rl.py",
        "frontend/src/features/dl/__tests__/EfficiencyLane.test.tsx",
        "frontend/src/features/dl/__tests__/MpnnLane.test.tsx",
        "frontend/src/features/generative/__tests__/RlLane.test.tsx",
        "frontend/src/services/dl/__tests__/dl-efficiency.test.ts",
        "frontend/src/services/dl/__tests__/mpnn-client.test.ts",
        "frontend/src/services/generative/__tests__/rl-client.test.ts",
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
        cargo = zf.read("src-tauri/Cargo.toml").decode("utf-8")
        sbom = zf.read("packaging/sbom/sbom.cdx.json").decode("utf-8")
        pyproject = zf.read("pyproject.toml").decode("utf-8")
        dunder = zf.read("src/pharmaml/__init__.py").decode("utf-8")
    quad = "(" + VERSION.replace(".", ",") + ",0)"
    vi_flat = vi.replace(" ", "")
    vi_ok = (
        f"filevers={quad}" in vi_flat
        and f"prodvers={quad}" in vi_flat
        and f"'FileVersion','{VERSION}.0'" in vi_flat
        and f"'ProductVersion','{VERSION}'" in vi_flat
    )
    print(f"version_info.txt in-archive carries {VERSION}: {vi_ok}")
    carriers_ok = (
        f'version = "{VERSION}"' in pyproject
        and f'__version__ = "{VERSION}"' in dunder
        and f'version = "{VERSION}"' in cargo
        and f'"version": "{VERSION}"' in sbom
    )
    print(f"in-archive version carriers at {VERSION}: {carriers_ok}")
    if not vi_ok or not carriers_ok:
        print("A VERSION CARRIER IS STALE IN THE ARCHIVE -- refusing to ship")
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
