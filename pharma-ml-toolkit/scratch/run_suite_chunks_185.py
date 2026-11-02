"""Double-fork chunked full-suite runner (round 185 re-ship edition).

The box OOMs a whole-tree run at ~62% (the standing workaround:
per-chunk processes reset the GUI-widget accumulation).  This script
runs every chunk SEQUENTIALLY in its own pytest process and writes one
summary line per chunk, so the honest pass/skip tally is the sum.

Home-directory literals are forbidden in the shipped tree (the leakage
guard): paths derive from this file's location instead.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
PY = str(Path.home() / ".venv" / "bin" / "python")
OUT = "/tmp/suite185.txt"

A1 = [
    "test_applicability", "test_approvals", "test_biology", "test_campaign",
    "test_cellline_examples", "test_chem_standardize", "test_chem_tables",
    "test_cli_features", "test_converters", "test_data_loading",
    "test_data_provenance", "test_database_audit", "test_descriptors",
    "test_diversity", "test_docking", "test_docking_prep",
]
A2 = [
    "test_docs_index", "test_dosage", "test_explain", "test_features",
    "test_format_converters", "test_generative", "test_hierarchical",
    "test_interpret", "test_io", "test_leakage", "test_manifold",
    "test_mcts", "test_ml_pipeline", "test_multiclass", "test_neural",
    "test_notebook_variants",
]
A3 = [
    "test_pd", "test_pk", "test_pose_rmsd", "test_requirements",
    "test_rism", "test_scoring_transforms", "test_selfies_codec",
    "test_similarity", "test_split_categorical", "test_stacking",
    "test_tuner", "test_units", "test_unsupervised", "test_viz",
    "test_web_parity",
]
#: the Deep Learning tab's classes, definition order thirds
PG_CLASSES = [
    # B1
    "TestModuleSurface", "TestToolkitVersion", "TestCellLineSurface",
    "TestDockingTab", "TestPureHelpers", "TestManualHyperparameterHelpers",
    "TestDockingHelpers", "TestSessionHelpers", "TestPharmamlIntegration",
    "TestCellLinePipeline", "TestDockingPipeline", "TestCommandLine",
    "TestGuiUnderXvfb", "TestDockingGuiUnderXvfb", "TestDockingRescoreReport",
    "TestSessionGuiUnderXvfb", "TestFeatureFactorySurface",
    "TestFeatureFactoryGuiUnderXvfb", "TestSessionShareAndReset",
    "TestPerTabResetButtons", "TestGenerativeParetoCard", "TestCopyTabInputs",
    "TestShortcutHints", "TestDeepLearningTab", "TestDefaultTargetColumn",
    "TestNumericFeatureGuard", "TestTunerCard", "TestGenerativeRLCard",
    "TestMLReportBundle", "TestMLCheckCSV", "TestDLCheckCSV",
    "TestUtilitiesCoverageGui", "TestDockingCoverageGui",
    "TestApplicabilityDomainGui", "TestCompletenessReviewGui",
    # B2
    "TestReview120Gui", "TestReview121Gui", "TestReview122Gui",
    "TestReview123Gui", "TestReview124Gui", "TestReview125Gui",
    "TestReview126Gui", "TestReview127Gui", "TestReview128Gui",
    "TestReview129Gui", "TestReview130Gui", "TestReview131Gui",
    "TestReview132Gui", "TestReview133Gui", "TestReview134Gui",
    "TestReview135Gui", "TestReview136Gui", "TestReview137Gui",
    "TestReview138Gui", "TestReview139Gui", "TestReview140Gui",
    "TestReview141Gui", "TestReview142Gui", "TestReview143Gui",
    "TestReview144Gui", "TestReview145Gui", "TestReview146Gui",
    "TestReview148Gui", "TestReview149Gui", "TestReview150Gui",
    "TestReview151Gui", "TestPredictionIntervalsGui", "TestOverfitGapGui",
    # B3 -- the interpretability tail incl. the round-185 XAI classes
    "TestInterpretabilityGui", "TestCrossModelImportanceGui",
    "TestVarianceFilterGui", "TestUnsupervisedTab",
    "TestDearPyGuiAffordances", "TestDearPyGuiRoundTwo",
    "TestShapOptunaAbsorptionGui", "TestReinvent4DiffDockAbsorptionGui",
    "TestXaiHelpers", "TestXaiAbsorptionGui",
]
CHUNKS: list[tuple[str, list[str]]] = [
    ("A1", [f"tests/{name}.py" for name in A1]),
    ("A2", [f"tests/{name}.py" for name in A2]),
    ("A3", [f"tests/{name}.py" for name in A3]),
    ("B1", [f"tests/test_playground.py::{name}" for name in PG_CLASSES[:36]]),
    ("B2", [f"tests/test_playground.py::{name}" for name in PG_CLASSES[36:68]]),
    ("B3", [f"tests/test_playground.py::{name}" for name in PG_CLASSES[68:]]),
    ("C", ["src"]),
]


def main() -> int:
    if os.fork() > 0:
        return 0
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    out = open(OUT, "wb", buffering=0)  # noqa: SIM115 - dup2'd, must stay open
    os.dup2(out.fileno(), 1)
    os.dup2(out.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.chdir(TOOLKIT)
    env = dict(os.environ)
    env["DISPLAY"] = ":99"
    env["PYTHONPATH"] = str(TOOLKIT / "src")
    total_start = time.perf_counter()
    for label, targets in CHUNKS:
        started = time.perf_counter()
        result = subprocess.run(
            [
                PY, "-m", "pytest", "-p", "no:warnings", "--tb=line", "-q",
                f"--junitxml=scratch/junit185_{label}.xml", *targets,
            ],
            capture_output=True, text=True,
        )
        tail = result.stdout.strip().splitlines()[-2:] if result.stdout.strip() else ["NO OUTPUT"]
        print(
            f"CHUNK {label}: rc={result.returncode} "
            f"({time.perf_counter() - started:.0f} s) :: {' | '.join(tail)}",
            flush=True,
        )
    print(f"SUITE185 TOTAL {time.perf_counter() - total_start:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
