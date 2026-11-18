"""Round 232 -- re-run chunks 4a/4b/4c after the lazy-marker fix."""
import os, subprocess, sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
PY = str(Path.home() / ".venv" / "bin" / "python")
OUT = "/tmp/suite232b.txt"
CHUNKS = [
    ("4a (ui bestof..docking)", ["tests/ui/test_bestofml_page.py", "tests/ui/test_biology_page.py",
                    "tests/ui/test_chemistry_page.py", "tests/ui/test_data_table.py",
                    "tests/ui/test_deep_learning_page.py", "tests/ui/test_docking_page.py"]),
    ("4b (ui export..generative)", ["tests/ui/test_export_gui.py", "tests/ui/test_foundations_page.py",
                    "tests/ui/test_generative_page.py"]),
    ("4d (ui imagelab..mainwindow)", ["tests/ui/test_image_lab_page.py",
                    "tests/ui/test_job_worker.py", "tests/ui/test_main_window.py"]),
    ("4c (ui n-z)", ["tests/ui/test_chart_inspector.py",
                    "tests/ui/test_headless_platform.py",
                    "tests/ui/test_ml_workbench_page.py",
                    "tests/ui/test_model_analysis_page.py",
                    "tests/ui/test_pages_smoke.py", "tests/ui/test_pkpd_page.py",
                    "tests/ui/test_qt_noise_filter.py",
                    "tests/ui/test_responseai_page.py",
                    "tests/ui/test_smiles_preview.py",
                    "tests/ui/test_theme_manager.py",
                    "tests/ui/test_unsupervised_page.py",
                    "tests/ui/test_utilities_page.py"]),
]
if os.fork() > 0:
    sys.exit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)
out = open(OUT, "wb", buffering=0)
os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2)
dn = os.open(os.devnull, os.O_RDONLY); os.dup2(dn, 0)
os.chdir(TOOLKIT)
os.environ["DISPLAY"] = ":99"
os.environ["PYTHONPATH"] = str(TOOLKIT / "src")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
failures = 0
for label, patterns in CHUNKS:
    done = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:warnings", "--tb=line"] + patterns,
                          cwd=str(TOOLKIT), capture_output=True, text=True)
    failures += done.returncode != 0
    tail = "\n".join(done.stdout.strip().splitlines()[-2:])
    print(f"CHUNK {label} exit={done.returncode}: {tail}", flush=True)
print(f"RERUN232 DONE: {failures} failing chunk(s)", flush=True)
