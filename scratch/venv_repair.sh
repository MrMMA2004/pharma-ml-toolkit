#!/usr/bin/env bash
# venv repair after 5th rollback — restore pinned stack per requirements.txt
set -x
P=/home/z/.venv/bin/pip
printf 'numpy==1.26.4\n' > /tmp/constraint.txt
$P install "numpy==1.26.4" 2>&1 | tail -2
$P install rdkit==2026.9.1 meeko==0.8.0 gemmi==0.7.5 selfies==2.1.1 shap==0.54.1 ruff==0.17.1 mypy==2.4.0 pytest==9.0.2 pytest-cov==7.0.0 -c /tmp/constraint.txt 2>&1 | tail -3
$P install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2
/home/z/.venv/bin/python -c "import numpy,rdkit,selfies,shap,meeko,gemmi,ruff,mypy,matplotlib,torch; print('VENV-REPAIR-DONE numpy',numpy.__version__,'torch',torch.__version__)"
# editable install of the toolkit itself (lost with the venv in rollbacks)
cd /home/z/my-project/pharma-ml-toolkit && /home/z/.venv/bin/pip install --no-deps -e . 2>&1 | tail -1
