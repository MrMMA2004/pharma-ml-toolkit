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
#    openpyxl 3.1.5 rides here since v1.2.116: the playground's Excel
#    input/output guarantee (xlsx read + write) depends on it explicitly —
#    pre-116 repairs did not carry it and a future repair must not drop it.
$P install matplotlib==3.10.9 seaborn==0.13.2 fpdf2==2.8.5 python-docx==1.2.0 \
    openpyxl==3.1.5 \
    mordred==1.2.0 xgboost==2.1.3 lightgbm==4.5.0 catboost==1.2.8 \
    -c /tmp/constraint.txt 2>&1 | tail -1

# 5. torch CPU (the heavy wheel — foreground, ~2-5 min)
$P install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1

# 6. pytest-xdist (the round-195 parallel chunk runner; optional but pinned now)
$P install pytest-xdist 2>&1 | tail -1

# 6b. the Studio GUI stack (round 206, directive #16 — the PySide6
#     redesign). pyqtgraph MUST stay 0.13.7: 0.14 requires numpy>=2.2
#     and this stack pins 1.26.4 (mordred). PySide6 6.8.3 = the Qt 6.8
#     LTS wheel verified under Python 3.12. The EGL stub under
#     pharma-ml-toolkit/ui/native/ carries the headless fallback: this
#     box has no system libEGL, so the stub is ALSO copied into the
#     wheel's Qt/lib dir (libQt6Gui.so.6 carries RUNPATH $ORIGIN, so
#     the dependency resolves with zero env vars — plain pytest runs
#     import pytest-qt's binding probe, which loads QtGui eagerly).
$P install PySide6==6.8.3 pyqtgraph==0.13.7 qt-material==2.17 \
    qdarkstyle==3.2.3 pytest-qt==4.5.0 \
    -c /tmp/constraint.txt 2>&1 | tail -1
cp /home/z/my-project/pharma-ml-toolkit/ui/native/libEGL.so.1 \
   /home/z/.venv/lib/python3.12/site-packages/PySide6/Qt/lib/libEGL.so.1

# 6c. the part10 FastAPI service stack (r275: src/pharmaml/api needs
#     it — without these pins a repair would drop the api package's
#     deps and tests/api would stop importing). Versions = the live
#     venv at r275; pydantic wheels are numpy-neutral (1.26.4 safe).
$P install fastapi==0.128.0 uvicorn==0.44.0 pydantic==2.12.5 \
    httpx==0.28.1 \
    -c /tmp/constraint.txt 2>&1 | tail -1

# 6d. the freeze stack (r274/r275: the part9 frozen-equivalent tier runs
#     PyInstaller 6.22.3 per the captured Windows-spec policy — the
#     r275 round-open repair dropped it and the freeze tests went red
#     on the import; the pin keeps it across future rebuilds).
$P install pyinstaller==6.22.3 2>&1 | tail -1

# 7. the editable toolkit itself (its version tracks the checkout)
cd /home/z/my-project/pharma-ml-toolkit && $P install --no-deps -e . 2>&1 | tail -1

# 8. verify the full import surface
/home/z/.venv/bin/python -c "import numpy, rdkit, selfies, shap, meeko, gemmi, matplotlib, torch, mordred, xgboost, lightgbm, catboost, seaborn, fpdf, docx, openpyxl, pandas, pharmaml, fastapi, uvicorn, pydantic; print('VENV-REPAIR-DONE numpy', numpy.__version__, 'torch', torch.__version__, 'matplotlib', matplotlib.__version__, 'openpyxl', openpyxl.__version__, 'pharmaml', pharmaml.__version__, 'fastapi', fastapi.__version__, 'uvicorn', uvicorn.__version__)"

# 8b. verify the Studio GUI stack offscreen (the EGL stub + offscreen
#     platform ride inside ui.headless — real EGL, when present, wins)
cd /home/z/my-project/pharma-ml-toolkit && /home/z/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from ui.headless import ensure_headless
ensure_headless()
from PySide6.QtWidgets import QApplication
import pyqtgraph, qt_material, qdarkstyle
app = QApplication([])
print('STUDIO-REPAIR-DONE PySide6 OK pyqtgraph', pyqtgraph.__version__)"
