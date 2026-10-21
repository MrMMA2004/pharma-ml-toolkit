#!/bin/bash
# venv repair (9th reset) — the proven recipe from the worklog (100x-109)
set -x
PIP="/home/z/.venv/bin/python -m pip"
$PIP install "rdkit==2026.09.1" "meeko==0.8.0" "gemmi==0.7.5"
$PIP install "numpy==1.26.4" "mordred==1.2.0"
$PIP install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
$PIP install "ruff==0.17.1" "mypy==2.4.0"
cd /home/z/my-project/pharma-ml-toolkit && $PIP install -e . --no-deps
# pytest already present; ensure joblib etc. are fine
$PIP install "joblib" "nbformat" "nbclient"
# Xvfb :99 respawn
pkill Xvfb 2>/dev/null
sleep 1
Xvfb :99 -screen 0 1600x1200x24 >/dev/null 2>&1 &
sleep 2
echo "=== VERIFY ==="
/home/z/.venv/bin/python - <<'PY'
import rdkit, meeko, gemmi, mordred, numpy, torch, pharmaml
print("rdkit", rdkit.__version__)
print("meeko", meeko.__version__)
print("gemmi", gemmi.__version__)
print("mordred ok; numpy", numpy.__version__)
print("torch", torch.__version__)
print("pharmaml", pharmaml.__version__)
PY
echo "REPAIR_DONE rc=$?"
