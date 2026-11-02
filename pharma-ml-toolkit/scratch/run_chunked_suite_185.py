"""Double-fork chunked-suite runner (round 185, the OOM workaround).

The 2-CPU/4-GB box OOM-kills whole-file runs of test_playground.py at
~61% and eats 590 s on the big test_[m-z] chunks, so every chunk runs
in its OWN fresh pytest process (memory freed between chunks), all
sequential under one detached daemon, each with a clear chunk marker
in /tmp/suite185.txt.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
PY = str(Path.home() / ".venv" / "bin" / "python")
OUT = "/tmp/suite185.txt"

BASE = [PY, "-m", "pytest", "-q", "-p", "no:warnings", "--tb=line"]

CHUNKS: list[list[str]] = [
    # 1. the alphabetical a-l leg (leakage + provenance fixes included)
    ["tests/test_a*.py", "tests/test_b*.py", "tests/test_c*.py",
     "tests/test_d*.py", "tests/test_e*.py", "tests/test_f*.py",
     "tests/test_g*.py", "tests/test_h*.py", "tests/test_i*.py",
     "tests/test_l*.py"],
    # 2. m-p (playground split out; notebook variants AFTER nbexec)
    ["tests/test_m*.py", "tests/test_notebook_variants.py",
     "tests/test_p*.py", "--ignore=tests/test_playground.py"],
    # 3. r-z
    ["tests/test_r*.py", "tests/test_s*.py", "tests/test_t*.py",
     "tests/test_u*.py", "tests/test_v*.py", "tests/test_w*.py"],
    # 4. src doctests
    ["src/pharmaml", "--doctest-modules"],
]

PLAYGROUND_CLASS_CHUNKS: list[list[str]] = [
    ["TestModuleSurface", "TestToolkitVersion", "TestCellLineSurface",
     "TestDockingTab", "TestPureHelpers", "TestManualHyperparameterHelpers",
     "TestDockingHelpers", "TestSessionHelpers", "TestPharmamlIntegration"],
    ["TestCellLinePipeline", "TestDockingPipeline", "TestCommandLine",
     "TestGuiUnderXvfb", "TestDockingGuiUnderXvfb", "TestDockingRescoreReport",
     "TestSessionGuiUnderXvfb", "TestFeatureFactorySurface",
     "TestFeatureFactoryGuiUnderXvfb"],
    ["TestSessionShareAndReset", "TestPerTabResetButtons",
     "TestGenerativeParetoCard", "TestCopyTabInputs", "TestShortcutHints",
     "TestDeepLearningTab", "TestDefaultTargetColumn",
     "TestNumericFeatureGuard", "TestTunerCard", "TestGenerativeRLCard"],
    ["TestMLReportBundle", "TestMLCheckCSV", "TestDLCheckCSV",
     "TestUtilitiesCoverageGui", "TestDockingCoverageGui",
     "TestApplicabilityDomainGui", "TestCompletenessReviewGui"],
    ["TestReview120Gui", "TestReview121Gui", "TestReview122Gui",
     "TestReview123Gui", "TestReview124Gui", "TestReview125Gui",
     "TestReview126Gui", "TestReview127Gui"],
    ["TestReview128Gui", "TestReview129Gui", "TestReview130Gui",
     "TestReview131Gui", "TestReview132Gui", "TestReview133Gui",
     "TestReview134Gui", "TestReview135Gui"],
    ["TestReview136Gui", "TestReview137Gui", "TestReview138Gui",
     "TestReview139Gui", "TestReview140Gui", "TestReview141Gui",
     "TestReview142Gui", "TestReview143Gui"],
    ["TestReview144Gui", "TestReview145Gui", "TestReview146Gui",
     "TestReview148Gui", "TestReview149Gui", "TestReview150Gui",
     "TestReview151Gui", "TestPredictionIntervalsGui"],
    ["TestOverfitGapGui", "TestInterpretabilityGui",
     "TestCrossModelImportanceGui", "TestVarianceFilterGui",
     "TestUnsupervisedTab", "TestDearPyGuiAffordances",
     "TestDearPyGuiRoundTwo"],
    ["TestShapOptunaAbsorptionGui", "TestReinvent4DiffDockAbsorptionGui",
     "TestXaiHelpers", "TestXaiAbsorptionGui"],
]


def run_chunk(label: str, args: list[str]) -> tuple[int, str]:
    command = BASE + args
    done = subprocess.run(command, cwd=str(TOOLKIT), capture_output=True, text=True)
    tail = "\n".join(done.stdout.strip().splitlines()[-2:])
    return done.returncode, tail


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
    os.environ["DISPLAY"] = ":99"
    os.environ["PYTHONPATH"] = str(TOOLKIT / "src")

    failures = 0
    for index, args in enumerate(CHUNKS, start=1):
        code, tail = run_chunk(f"file-chunk {index}", args)
        failures += code != 0
        print(f"CHUNK {index} (files) exit={code}: {tail}", flush=True)
    for index, classes in enumerate(PLAYGROUND_CLASS_CHUNKS, start=1):
        selectors = [f"tests/test_playground.py::{name}" for name in classes]
        code, tail = run_chunk(f"playground-chunk {index}", selectors)
        failures += code != 0
        print(f"CHUNK P{index} exit={code}: {tail}", flush=True)
    print(f"SUITE185 DONE: {failures} failing chunk(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
