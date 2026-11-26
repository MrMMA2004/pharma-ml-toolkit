# legacy/ — quarantined unsupported source (part18 §10)

This directory is the explicit quarantine area required by part18 §10
("Mandatory Retirement of studio.py") for the legacy PySide6 GUI:

* `legacy/studio.py` — the retired application launcher.  It is NOT an
  active supported entry point: the banner at its head says so, and
  no product path starts it.
* `legacy/ui/` — the complete legacy PySide6 GUI package (pages,
  widgets, the design tokens, the headless EGL bootstrap), moved here
  from the toolkit root as the part18 §10 quarantine completion.  Its
  import graph is fully contained: the FastAPI backend uses only the
  Qt-free `pharmaml.studio` persistence/evidence/registry modules,
  which stayed in `src/` where they legitimately serve the backend.
* `legacy/tests_ui/` — the 74-file archival regression suite of the
  quarantined GUI (moved from `tests/ui/`).  It is NOT part of the
  default pytest gate (`testpaths = ["tests", "src"]`); run it
  explicitly when the optional PySide6 stack is installed::

      pytest legacy/tests_ui

  The root `conftest.py` still preloads the EGL stub for every
  invocation, so the offscreen runs work in bare containers too.

Status: **UNSUPPORTED** — not part of the supported product, not part
of production packaging (the Windows sidecar spec excludes the whole
Qt stack: PySide6, shiboken6, pyqtgraph, qt_material, qdarkstyle; the
legacy GUI freeze spec resolves the package through `legacy/` on its
pathex), and not covered by any release gate that matters for the
product.  The files exist for archival and reproducibility reasons
only.

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
