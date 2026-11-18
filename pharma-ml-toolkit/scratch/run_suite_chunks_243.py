"""Round-243 chunked suite runner (the 242 pattern; gains
test_plot_legend_placement_243.py in chunk 4c -- the round-243 SS23.9
legend placement suite).

Chunks the suite into eight batches so no
single pytest process has to hold the whole tree (the OOM lesson).
Chunk 4 (tests/ui) and chunk 5 (playground, Tk under Xvfb) need
DISPLAY=:99 -- the runner exports it.  Writes /tmp/suite240.txt.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
PY = str(Path.home() / ".venv" / "bin" / "python")
OUT = "/tmp/suite243.txt"

BASE = [PY, "-m", "pytest", "-q", "-p", "no:warnings", "--tb=line"]


def expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            out.extend(sorted(glob.glob(str(TOOLKIT / pattern))) or [pattern])
        else:
            out.append(pattern)
    return out


CHUNKS: list[tuple[str, list[str]]] = [
    ("1 (a-l)", ["tests/test_a*.py", "tests/test_b*.py", "tests/test_c*.py",
                 "tests/test_d*.py", "tests/test_e*.py", "tests/test_f*.py",
                 "tests/test_g*.py", "tests/test_h*.py", "tests/test_i*.py",
                 "tests/test_l*.py"]),
    # (round 228) test_playground_workflows.py retired WITH playground.py
    # (the 980-test Tk-coupled suite: test_playground.py, workflows,
    # job_safety, excel_io_and_samples, feature_recalculation) -- the
    # Studio's own page tests carry the migration's evidence now.
    ("2 (m-p)", ["tests/test_m*.py", "tests/test_notebook_variants.py",
                 "tests/test_pd.py", "tests/test_pk.py",
                 "tests/test_pkpd_engines.py", "tests/test_pose_rmsd.py",
                 "tests/test_pruning.py"]),
    ("3 (r-z)", ["tests/test_r*.py", "tests/test_s*.py", "tests/test_t*.py",
                 "tests/test_u*.py", "tests/test_v*.py", "tests/test_w*.py"]),
    # (round 225) chunk 4 used to be the whole tests/ui directory in ONE
    # pytest process -- the kernel OOM-killed it at ~163/241 tests (4 GB
    # sandbox, next-server down, nothing else running: the ui pages'
    # PySide6 footprints simply exceed one process's share).  Split into
    # two processes, a-m / n-z, so each holds half the Qt pages.
    ("4a (ui bestof..docking)", ["tests/ui/test_bestofml_page.py", "tests/ui/test_biology_page.py",
                     "tests/ui/test_chemistry_page.py", "tests/ui/test_data_table.py",
                     "tests/ui/test_deep_learning_page.py", "tests/ui/test_docking_page.py"]),
    # (round 232) chunk 4b split AGAIN (the round-225 OOM lesson, second
    # verse): six page-suites in one pytest process crossed 1.66 GB rss at
    # teardown and the kernel OOM-killed it AFTER the last test passed --
    # three suites per process now (4b + the new 4d).
    ("4b (ui export..generative)", ["tests/ui/test_export_gui.py", "tests/ui/test_foundations_page.py",
                     "tests/ui/test_generative_page.py"]),
    # (round 235) 4d carries the SS23 responsive-policy suite at 16
    # tests (the scroll-policy section added three): two MainWindow
    # constructions in the process and the chunk still fits -- 50/50
    # green in the round-235 pre-gate.
    ("4d (ui imagelab..responsive)", ["tests/ui/test_image_lab_page.py",
                     "tests/ui/test_job_worker.py", "tests/ui/test_main_window.py",
                     "tests/ui/test_responsive_policy.py"]),
    # (round 240) 4c gains the SS21.4 image-hover suite (20 tests, pure
    # ImagePanel + three Image Lab page constructions -- the same page
    # suite 4d already runs, so the memory envelope is known).
    ("4c (ui n-z)", ["tests/ui/test_chart_inspector.py",
                     "tests/ui/test_headless_platform.py",
                     "tests/ui/test_image_hover_probe.py",
                     "tests/ui/test_legend_policy.py",
                     "tests/ui/test_plot_tick_density.py",
                     "tests/ui/test_plot_title_budget_242.py",
                     "tests/ui/test_plot_legend_placement_243.py",
                     "tests/ui/test_ml_workbench_page.py",
                     "tests/ui/test_model_analysis_page.py",
                     "tests/ui/test_pages_smoke.py", "tests/ui/test_pkpd_page.py",
                     "tests/ui/test_qt_doctor.py",
                     "tests/ui/test_qt_noise_filter.py",
                     "tests/ui/test_responseai_page.py",
                     "tests/ui/test_scatter_meta_241.py",
                     "tests/ui/test_smiles_preview.py",
                     "tests/ui/test_theme_manager.py",
                     "tests/ui/test_unsupervised_page.py",
                     "tests/ui/test_utilities_page.py"]),
    # (round 228) chunk 5 ("playground Tk", tests/test_playground.py) is
    # GONE -- playground.py retired; the Studio pages are chunks 4a-4c.
    ("6 (src doctests)", ["src/pharmaml", "--doctest-modules"]),
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
    os.environ["DISPLAY"] = ":99"
    os.environ["PYTHONPATH"] = str(TOOLKIT / "src")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    failures = 0
    for label, patterns in CHUNKS:
        command = BASE + expand(patterns)
        done = subprocess.run(command, cwd=str(TOOLKIT), capture_output=True, text=True)
        failures += done.returncode != 0
        tail = "\n".join(done.stdout.strip().splitlines()[-2:])
        print(f"CHUNK {label} exit={done.returncode}: {tail}", flush=True)
    print(f"SUITE243 DONE: {failures} failing chunk(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
