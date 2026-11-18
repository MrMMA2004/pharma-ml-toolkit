# -*- mode: python ; coding: utf-8 -*-
"""
PharmaML Studio — the BACKEND SIDECAR PyInstaller "onedir" spec
(part11/part11_2 §5, §6 and §31.1).

The part11 architecture splits the deployment into:

* the React + TypeScript frontend (static assets, built by Vite — no
  Python, no Node on the end-user machine);
* the Tauri 2 host (Rust + WebView2 — the Windows runner builds it);
* THIS spec's artifact: the standalone Python backend sidecar —
  FastAPI + Uvicorn + the complete scientific stack, frozen as a
  onedir bundle the Tauri installer ships under ``resources/backend``.

The entry is ``backend_entry.py`` (NOT ``studio.py``): part11_2 §5 is
explicit — the production installer must not bundle the retired-era
PySide6 production GUI.  Until the part11 §23 feature-parity gate
retires PySide6, ``studio.py`` remains the legacy entry for the
frozen-equivalent tier only; this sidecar excludes the whole Qt
stack while preserving every Qt-independent scientific dependency
(matplotlib figure generation keeps its Agg/SVG/PDF backends — part11
§23's "Preserve Qt-independent scientific libraries" rule).

Paths resolve from this spec's directory (packaging/windows inside
the toolkit root) — reproducible from any checkout, no
developer-specific absolute paths.
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

SPEC_DIR = Path(SPECPATH).resolve()
TOOLKIT = SPEC_DIR.parents[1]  # packaging/windows -> toolkit root

APP_NAME = "pharmaml-backend"
ICON_FILE = str(SPEC_DIR / "assets" / "pharmaml.ico")

# The dynamic-import inventory (part9 §3.1's rule, preserved for the
# backend sidecar — "do not omit modules simply because they are
# loaded dynamically"):
hiddenimports = (
    # mordred's descriptor registry is discovered dynamically.
    *collect_submodules("mordred"),
    # matplotlib figure-export backends the backend figure routes
    # serve (PNG/JPEG/TIFF via Agg+Pillow, SVG, PDF).  The Qt canvas
    # backend is deliberately NOT here — no Qt in the sidecar.
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_pdf",
    # seaborn's style/matplotlib registration is import-side-effect.
    "seaborn",
    # SHAP's numba/llvmlite JIT layer needs the explicit entry.
    "shap",
    "numba",
    "llvmlite",
    # the boosting estimators' sklearn-facing shims.
    "xgboost",
    "lightgbm",
    "catboost",
    # SELFIES codec (generative routes).
    "selfies",
    # uvicorn's standard worker/loop implementations load lazily.
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
)

datas = [
    # the engines' pinned AutoDock trio (docking routes locate them
    # through Path(__file__).parents[1] / "engines" — the datas
    # target keeps that math aligned in the frozen tree).
    (str(TOOLKIT / "engines"), "pharmaml/engines"),
    (str(TOOLKIT / "LICENSE"), "."),
    (str(TOOLKIT / "VERSION"), "."),
    # meeko's forcefield/verification data (docking preparation).
    *collect_data_files("meeko"),
]

binaries = [
    # llvmlite ships its compiled core as a package DLL.
    *collect_dynamic_libs("llvmlite"),
]

excludes = [
    # ---- the Qt stack (part11_2 §5: no PySide6 GUI in the sidecar) ----
    "PySide6",
    "shiboken6",
    "pyqtgraph",
    "qt_material",
    "qdarkstyle",
    "matplotlib.backends.backend_qtagg",
    # ---- dev/test-only deps ----
    "pytest",
    "_pytest",
    "pytestqt",
    "pytest_cov",
    "pytest_xdist",
    "mypy",
    "ruff",
    # tkinter is not imported anywhere in the backend.
    "tkinter",
    # keep dev tooling out of the frozen bundle:
    "pip",
    "setuptools",
    "wheel",
    "IPython",
    "jupyter",
]

a = Analysis(
    [str(SPEC_DIR / "backend_entry.py")],
    pathex=[str(TOOLKIT), str(TOOLKIT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # never strip scientific DLLs (part9 §12)
    upx=False,   # UPX on scientific DLLs is a known AV/compat hazard
    console=True,  # the sidecar IS a console subprocess: stdout carries
    #               the PHARMAML-READY handshake line the Tauri host
    #               parses (part11 §14.1 — machine-readable stdout).
    icon=ICON_FILE if Path(ICON_FILE).is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
