# -*- mode: python ; coding: utf-8 -*-
"""
PharmaML Studio — the PyInstaller "onedir" spec (part9 §2/§3).

Deployment architecture (the documented technical evaluation):
PyInstaller onedir + Inno Setup.  Chosen over Nuitka/cx_Freeze because
the app is a large scientific PySide6 stack (torch, rdkit, catboost,
numba) where onedir keeps the bundle inspectable, start-up fast, and
AV-false-positive surface small; over onefile because §2 explicitly
forbids trading start-up time for a single-file executable.

Paths: everything is resolved from this spec's directory
(packaging/windows inside the toolkit root), so the build is
reproducible from any checkout — no developer-specific absolute
paths (part9 §10 rule 9).

Layout contract with the application code (the part9 §3/§4 audit
findings, verified against the real source):

* ``ui/font_loader.py`` reads ``<parent-of-ui>/fonts`` — frozen, the
  ui package lands under ``_internal/ui``, so the datas target
  ``fonts`` puts the TTFs exactly where ``Path(__file__)`` math looks.
* ``pharmaml/docking/workflow.py``'s ``_bundled_engine_dirs()`` uses
  ``Path(__file__).parents[1] / "engines"`` — frozen, pharmaml lands
  under ``_internal/pharmaml``, so the engines' datas target
  ``pharmaml/engines`` aligns with the same parents[1] math.  The
  pinned SHA-256 + PE-machine validation in workflow.py then guards
  the bundled trio exactly as it does in a source checkout.
* ``ui/headless.py`` is a no-op on Windows desktops (offscreen is only
  forced on headless Linux-class platforms), so the frozen app runs
  the real windows platform plugin.
* ``studio.py`` now calls ``multiprocessing.freeze_support()`` first
  (part9 §13, added after inspecting the actual architecture: the only
  spawn-context pool lives in ``docking/ligands.py`` and is launched
  from worker threads; the GUI itself is guarded by the
  ``if __name__ == "__main__"`` entry guard, so children never re-launch
  the GUI).  ``--doctor`` keeps working frozen: it is argv-level
  dispatch inside ``ui.run()``.
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

SPEC_DIR = Path(SPECPATH).resolve()
TOOLKIT = SPEC_DIR.parents[1]  # packaging/windows -> toolkit root

APP_NAME = "PharmaMLStudio"
VERSION_FILE = str(SPEC_DIR / "version_info.txt")
ICON_FILE = str(SPEC_DIR / "assets" / "pharmaml.ico")

# The dynamic-import inventory (part9 §3.1: "Do not omit modules simply
# because they are loaded dynamically"):
hiddenimports = (
    # mordred's descriptor registry is discovered dynamically at import
    # time — the classic freeze trap; collect everything.
    *collect_submodules("mordred"),
    # pyqtgraph resolves backends/exports lazily.
    *collect_submodules("pyqtgraph"),
    # matplotlib figure-export backends the Studio offers (PNG/JPEG/TIFF
    # via Agg+Pillow, SVG, PDF, and the interactive Qt canvas).
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_pdf",
    # Qt SVG support (pyqtgraph's SVG export path).
    "PySide6.QtSvg",
    # seaborn's style/matplotlib registration is import-side-effect only.
    "seaborn",
    # SHAP's TreeExplainer surface (function-level import in
    # pharmaml.ml.explain — modulegraph finds it, but the numba/llvmlite
    # JIT layer behind it needs the explicit entry).
    "shap",
    "numba",
    "llvmlite",
    # the boosting estimators' sklearn-facing shims.
    "xgboost.sklearn",
    "lightgbm.sklearn",
    "catboost.utils",
    # the report/document writers' template-bearing modules.
    "docx",
    # SELFIES codec (pure Python, plain import).
    "selfies",
)

datas = [
    # the app's own immutable resources (part9 §4):
    (str(TOOLKIT / "fonts"), "fonts"),  # IBM Plex Sans + JetBrains Mono + OFL
    (str(TOOLKIT / "engines"), "pharmaml/engines"),  # the AutoDock trio (sha-pinned)
    (str(TOOLKIT / "LICENSE"), "."),
    (str(TOOLKIT / "VERSION"), "."),
    # stylesheet/theme engines ship their resource files (QSS/JSON/MS):
    *collect_data_files("qt_material"),
    *collect_data_files("qdarkstyle"),
    # meeko's forcefield/verification data (docking preparation):
    *collect_data_files("meeko"),
    # python-docx's built-in default template:
    *collect_data_files("docx"),
]

binaries = [
    # llvmlite ships its compiled core as a package DLL — belt and
    # braces on top of the stock hook (part9 §3.3: audit native deps).
    *collect_dynamic_libs("llvmlite"),
]

excludes = [
    # part9 §12: dev/test-only deps never enter the installer.  Verified
    # against the tree: no app module under src/ or ui/ imports pytest.
    "pytest",
    "_pytest",
    "pytestqt",
    "pytest_cov",
    "mypy",
    "ruff",
    # tkinter is not imported anywhere in the app.
    "tkinter",
    # keep dev tooling out of the frozen bundle:
    "pip",
    "setuptools",
    "wheel",
    "IPython",
    "jupyter",
]

a = Analysis(
    [str(TOOLKIT / "legacy" / "studio.py")],  # part18 §10: quarantined legacy entry
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
    upx=False,   # UPX on Qt/scientific DLLs is a known AV/compat hazard
    console=False,  # the Studio is a GUI application; --doctor writes to stderr on demand
    icon=ICON_FILE if Path(ICON_FILE).is_file() else None,
    version=VERSION_FILE if Path(VERSION_FILE).is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
