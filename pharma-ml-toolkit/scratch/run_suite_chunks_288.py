"""Double-fork suite runner (round 288, the 284 pattern — the arena-
absorption gate: the r288 merge landed the 2h pipeline's audit catches
(.gitignore examples/artifacts anchoring, audit_release.py path-anchored
excluded_paths check, the browser_verify_pkpd.sh $HOME scrub) plus the
regenerated RELEASE_READINESS/FEATURES carriers — the eleven legs run on
the MERGED tree before the superseding v1.2.228 rebuild. No new test
families rode the merge, so the leg scope is identical to r287's.

The eleven legs of the r280 gate: chunks 1-3 (the a-l / m-p / r-z
alphabet splits), the ui splits 4a/4b/4c/4d/4e/4f (the OOM lessons),
the src doctests (6), and the api leg (7: tests/api + the two
frontend guard suites).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
PY = str(Path.home() / ".venv" / "bin" / "python")
OUT = "/tmp/suite288.txt"

CHUNKS: list[tuple[str, list[str]]] = [
    ("1 (a-l)", ["tests/test_a*.py", "tests/test_b*.py", "tests/test_c*.py",
                 "tests/test_d*.py", "tests/test_e*.py", "tests/test_f*.py",
                 "tests/test_g*.py", "tests/test_h*.py", "tests/test_i*.py",
                 "tests/test_l*.py"]),
    ("2 (m-p)", ["tests/test_m*.py", "tests/test_notebook_variants.py",
                 "tests/test_pd.py", "tests/test_pk.py",
                 "tests/test_pkpd_engines.py", "tests/test_pkpd_workspace.py",
                 "tests/test_pose_rmsd.py", "tests/test_pruning.py"]),
    ("3 (r-z)", ["tests/test_r*.py", "tests/test_s*.py", "tests/test_t*.py",
                 "tests/test_u*.py", "tests/test_v*.py", "tests/test_w*.py"]),
    ("4a (ui bestof..docking)", ["tests/ui/test_bestofml_page.py", "tests/ui/test_biology_page.py",
                    "tests/ui/test_chemistry_page.py", "tests/ui/test_data_table.py",
                    "tests/ui/test_deep_learning_page.py", "tests/ui/test_docking_page.py"]),
    ("4b (ui export..generative)", ["tests/ui/test_export_gui.py", "tests/ui/test_foundations_page.py",
                    "tests/ui/test_generative_page.py"]),
    ("4d (ui imagelab..responsive)", ["tests/ui/test_image_lab_page.py",
                    "tests/ui/test_job_worker.py", "tests/ui/test_main_window.py",
                    "tests/ui/test_responsive_policy.py"]),
    ("4c (ui chart..legend)", ["tests/ui/test_chart_inspector.py",
                    "tests/ui/test_headless_platform.py",
                    "tests/ui/test_image_hover_probe.py",
                   "tests/ui/test_image_subpixel_245.py",
                   "tests/ui/test_image_1to1_246.py",
                    "tests/ui/test_legend_policy.py",
                    "tests/ui/test_plot_tick_density.py",
                    "tests/ui/test_plot_title_budget_242.py",
                    "tests/ui/test_plot_legend_placement_243.py"]),
    ("4e (ui mlworkbench..utilities)", ["tests/ui/test_ml_workbench_page.py",
                    "tests/ui/test_molecular_dynamics_page.py",
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
    ("4f (ui nine-cards)", ["tests/ui/test_nine_cards.py"]),
    ("6 (src doctests)", ["src/pharmaml", "--doctest-modules"]),
    ("7 (api + frontend guards)", ["tests/api", "tests/test_frontend_services.py",
                    "tests/test_frontend_foundation.py"]),
]

BASE = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts="]


def expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        if "*" in pattern:
            out.extend(sorted(str(p) for p in TOOLKIT.glob(pattern)))
        else:
            out.append(pattern)
    return out


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
        done = subprocess.run(command, cwd=str(TOOLKIT), capture_output=True, text=True, check=False)
        failures += done.returncode != 0
        tail = "\n".join(done.stdout.strip().splitlines()[-2:])
        print(f"CHUNK {label} exit={done.returncode}: {tail}", flush=True)
    print(f"SUITE288 DONE: {failures} failing chunk(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
