#!/usr/bin/env bash
# venv repair after sandbox rollbacks — restore the pinned stack of requirements.txt.
#
# Round 196 rewrite (the 10th-rollback drill): the original one-shot
# install line dies at ResolutionImpossible on today's resolver — shap
# 0.54.1's metadata asks numpy>=2 while the stack pins numpy 1.26.4
# (requirements.txt documents the warning as "not a runtime break"; the
# honest install is shap via --no-deps plus its real runtime deps).
# Idempotent: every step tolerates an already-satisfied state.
#
# Note for the firing session: background pip DIES in this sandbox
# (three losses across rounds 194-196, setsid or not) — run this script
# FOREGROUND inside one tool-call window.
set -x
P=/home/z/.venv/bin/pip
printf 'numpy==1.26.4\n' > /tmp/constraint.txt

# 1. the numpy pin first (the constraint anchor; downgrades 2.x survivors)
$P install "numpy==1.26.4" 2>&1 | tail -1

# 2. the chem + QA batch (shap NOT here — see step 3)
$P install rdkit==2026.9.1 meeko==0.8.0 gemmi==0.7.5 selfies==2.1.1 \
    ruff==0.17.1 mypy==2.4.0 pytest==9.0.2 pytest-cov==7.0.0 \
    -c /tmp/constraint.txt 2>&1 | tail -1

# 3. shap 0.54.1 WITHOUT deps (the numpy>=2 metadata lie), then its real
#    runtime deps under the numpy constraint
$P install --no-deps shap==0.54.1 2>&1 | tail -1
$P install slicer numba cloudpickle packaging -c /tmp/constraint.txt 2>&1 | tail -1

# 4. the GUI/report/ML layer (matplotlib 3.10.9 = the live-TkAgg pin)
$P install matplotlib==3.10.9 seaborn==0.13.2 fpdf2==2.8.5 python-docx==1.2.0 \
    mordred==1.2.0 xgboost==2.1.3 lightgbm==4.5.0 catboost==1.2.8 \
    -c /tmp/constraint.txt 2>&1 | tail -1

# 5. torch CPU (the heavy wheel — foreground, ~2-5 min)
$P install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1

# 6. pytest-xdist (the round-195 parallel chunk runner; optional but pinned now)
$P install pytest-xdist 2>&1 | tail -1

# 7. the editable toolkit itself (its version tracks the checkout)
cd /home/z/my-project/pharma-ml-toolkit && $P install --no-deps -e . 2>&1 | tail -1

# 8. verify the full import surface
/home/z/.venv/bin/python -c "import numpy, rdkit, selfies, shap, meeko, gemmi, matplotlib, torch, mordred, xgboost, lightgbm, catboost, seaborn, fpdf, docx, pharmaml; print('VENV-REPAIR-DONE numpy', numpy.__version__, 'torch', torch.__version__, 'matplotlib', matplotlib.__version__, 'pharmaml', pharmaml.__version__)"
