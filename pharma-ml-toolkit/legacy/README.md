# legacy/ — quarantined unsupported source (part18 §10)

This directory is the explicit quarantine area required by part18 §10
("Mandatory Retirement of studio.py") for the legacy PySide6 GUI:

* `legacy/studio.py` — the retired application launcher.  It is NOT an
  active supported entry point: the banner at its head says so, and
  no product path starts it.
* `ui/` and `src/pharmaml/studio/workspace.py` remain at their
  historical in-tree locations this round (their import graph is wide:
  the FastAPI backend legitimately uses the Qt-free
  `pharmaml.studio` persistence/evidence/registry modules, and the
  74-test `tests/ui` regression suite keeps the archived GUI
  reproducible); they are on the migration path to this quarantine in
  a later part18 increment.

Status: **UNSUPPORTED** — not part of the supported product, not part
of production packaging (the Windows sidecar spec excludes the whole
Qt stack: PySide6, shiboken6, pyqtgraph, qt_material, qdarkstyle), and
not covered by any release gate that matters for the product.  The
files exist for archival and reproducibility reasons only.

The supported application surfaces are:

1. **the React/Tauri 2 desktop application** — `frontend/` +
   `src-tauri/`, shipped as the NSIS EXE / WiX MSI with the frozen
   FastAPI sidecar (`packaging/windows/backend_entry.py`);
2. **the Python scientific toolkit + CLI** — `src/pharmaml/`,
   `python -m pharmaml ...`;
3. **the FastAPI backend** — `src/pharmaml/api/` (also the sidecar).

User data, trained models and artifacts are untouched by the
retirement: the persistence formats (`pharmaml.studio.runs` records,
job artifacts, model bundles) are consumed by the current backend and
remain compatible.
