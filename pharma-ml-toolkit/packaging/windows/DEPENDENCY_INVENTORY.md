# PharmaML Studio — dependency inventory (part9 §1.1)

Generated 2026-11-26 21:52 by `tools/packaging_audit.py` from the ACTUAL
source tree (studio.py + ui/ + src/ + scripts/, 3330 import statements parsed) and the live venv's dist-info. Versions are
importlib.metadata's truth, not a hand-maintained table.

Toolkit version audited: **1.2.243**. Python: 3.12.16.

| Package | Version | Purpose | Required or Optional | Included in Installer | Verification |
| --- | --- | --- | --- | --- | --- |
| numpy | 1.26.4 | array math everywhere | Required (unconditional) | YES | src/pharmaml/api/routes/applicability_domain.py; src/pharmaml/api/routes/chemistry.py; src/pharmaml/api/routes/dl.py; ... |
| pandas | 2.2.3 | tabular datasets, drift tables | Required (unconditional) | YES | src/pharmaml/api/routes/applicability_domain.py; src/pharmaml/api/routes/artifacts.py; src/pharmaml/api/routes/chemistry.py; ... |
| scikit-learn | 1.5.2 | the ml.* estimator families | Required (unconditional) | YES | src/pharmaml/chem/features.py; src/pharmaml/cli.py; src/pharmaml/data.py; ... |
| scipy | 1.14.1 | stats + optimization | Required (unconditional) | YES | src/pharmaml/biology/association.py; src/pharmaml/biology/biomarkers.py; src/pharmaml/biology/omics.py; ... |
| joblib | 1.5.3 | worker utilities (utils/compute absorbs its API) | Required (unconditional) | YES | src/pharmaml/chem/features.py; src/pharmaml/ml/persistence.py; src/pharmaml/ml/training.py |
| openpyxl | 3.1.5 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| tqdm | 4.67.1 | progress bars for long jobs | Required (unconditional) | YES | src/pharmaml/chem/descriptors.py; src/pharmaml/chem/rism.py |
| rdkit | 2026.9.1 | chem.* + docking preparation (Chem/Draw/DataStructs) | Required (feature-gated import) | YES | src/pharmaml/api/routes/docking.py; src/pharmaml/api/routes/health.py; src/pharmaml/api/routes/system.py; ... |
| mordred | 1.2.0 | the mordred descriptor set (dynamic registry) | Required (feature-gated import) | YES | src/pharmaml/chem/descriptors.py |
| meeko | 0.8.0 | SMILES -> PDBQT ligand preparation | Required (feature-gated import) | YES | src/pharmaml/docking/preparation.py; src/pharmaml/docking/receptor.py |
| gemmi | 0.7.5 | receptor .pdb parsing/preparation | Required (feature-gated import) | YES | src/pharmaml/docking/receptor.py |
| xgboost | 2.1.3 | gradient-boosted estimators | Required (feature-gated import) | YES | src/pharmaml/ml/models.py |
| lightgbm | 4.5.0 | gradient-boosted estimators | Required (feature-gated import) | YES | src/pharmaml/chem/features.py; src/pharmaml/ml/models.py |
| catboost | 1.2.8 | gradient-boosted estimators | Required (feature-gated import) | YES | src/pharmaml/ml/models.py |
| shap | 0.54.1 | TreeExplainer values + interaction cube (ml/explain) | Required (feature-gated import) | YES | src/pharmaml/ml/explain.py |
| numba | 0.61.0 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| llvmlite | 0.44.0 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| torch | 2.14.1+cpu | deep-learning estimators, generative VAEs, drug-response model | Required (feature-gated import) | YES | src/pharmaml/api/routes/system.py; src/pharmaml/generative/diffusion.py; src/pharmaml/generative/gan.py; ... |
| selfies | 2.1.1 | robust molecular string codec (chem/selfies_codec) | Required (feature-gated import) | YES | src/pharmaml/api/routes/system.py; src/pharmaml/chem/selfies_codec.py |
| matplotlib | 3.10.9 | all figure rendering + export backends | Required (feature-gated import) | YES | src/pharmaml/viz/_compat.py; src/pharmaml/viz/docking_plots.py; src/pharmaml/viz/export.py; ... |
| seaborn | 0.13.2 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| fpdf2 | 2.8.5 | PDF report writer (reports.py) | Required (feature-gated import) | YES | src/pharmaml/viz/report.py |
| python-docx | 1.2.0 | DOCX report writer (reports.py) | Required (feature-gated import) | YES | src/pharmaml/viz/report.py |
| Pillow | 11.3.0 | image IO behind PNG/JPEG/TIFF figure export | Required (feature-gated import) | YES | src/pharmaml/viz/export.py; src/pharmaml/viz/report.py |
| PySide6 | 6.8.3 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| pyqtgraph | 0.13.7 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| qt-material | 2.17 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| qdarkstyle | 3.2.3 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| shiboken6 | 6.8.3 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| statsmodels | 0.14.5 | multiple-testing FDR correction (biology/association) | Required (feature-gated import) | YES | src/pharmaml/biology/association.py |
| fastapi | 0.128.0 | the local REST API layer (api/routes/*, the app-level guard) | Required (feature-gated import) | YES | src/pharmaml/api/__init__.py; src/pharmaml/api/routes/__init__.py; src/pharmaml/api/routes/admet.py; ... |
| uvicorn | 0.44.0 | the loopback ASGI server the sidecar runs (port-0 bind) | Required (feature-gated import) | YES | src/pharmaml/api/server.py |
| pydantic | 2.12.5 | the typed request/response schemas behind the OpenAPI contract | Required (feature-gated import) | YES | src/pharmaml/api/events.py; src/pharmaml/api/routes/admet.py; src/pharmaml/api/routes/applicability_domain.py; ... |
| openbabel | NOT INSTALLED | optional mol2 writer lane (docking/ligands; NOT bundled — external) | Optional (feature-gated import) | NO (external, optional) | src/pharmaml/docking/ligands.py |

## Not used (the anti-padding rule)

part9 §1: *"Do not add packages that the project does not use merely
to increase the package size."* The following candidates from the
prompt's list are confirmed NOT imported anywhere in the app tree:

- **tensorflow**: NOT USED
- **cv2**: NOT USED
- **polars**: NOT USED
- **lime**: NOT USED
- **PyQt5**: NOT USED
- **PyQt6**: NOT USED
- **Gromacs wrapper**: NOT USED

## External executables (part9 §5)

The AutoDock trio ships in `engines/` and is bundled into the
installer at `_internal/pharmaml/engines` (the app's own
`_bundled_engine_dirs()` discovery, SHA-256- and PE-machine-validated):

| Executable | Build | Redistribution | Bundled |
| --- | --- | --- | --- |
| vina.exe | PE x86-64 | AutoDock Vina (Apache-2.0 family) | YES |
| autogrid4.exe | PE x86 | AutoDock (GPL-family; engine NOTICE included) | YES |
| autodock4.exe | PE x86 | AutoDock (GPL-family; engine NOTICE included) | YES |

GROMACS: no import, no subprocess call site — not invoked
anywhere in the tree; honestly NOT bundled.

Open Babel — the honest §5 note: the MOL2 leg of ligand
generation (``docking/ligands.py``) uses the ``openbabel``
PyPI wrapper ONLY when the user installs it, inside a guarded
``try/except ImportError`` whose failure the GUI reports as
`MOL2 skipped: OpenBabel not installed` in the result rows —
the exact pattern §5 demands (the capability reports the
missing tool clearly; every other workflow is unaffected).
The wrapper (GPL + native library) is deliberately NOT
bundled with the installer.

## GUI/Qt runtime

PySide6 6.8.3 (Essentials + Addons) is the complete Qt runtime for
the Studio; platform/imageformat/style plugins ride along via
PyInstaller's PySide6 hooks and are asserted post-build by
`packaging_validation.py --dist` (qwindows.dll etc.).

## Frontend and desktop dependencies (part11 §29, Phase Twenty-Four)

Derived from the actual `frontend/package.json` (name: pharmaml-studio-frontend, version 1.2.243), `src-tauri/Cargo.toml`
and `src-tauri/tauri.conf.json` — the same files the Windows
release workflow's step-2 version lockstep gate enforces(`tests/test_windows_pipeline.py`).

| Package/Component | Version | Purpose | Required or Optional | Frontend/Host/Backend | Included in Installer | Verification |
| --- | --- | --- | --- | --- | --- | --- |
| @radix-ui/react-tabs | ^1.1.13 | the shadcn/ui Tab head | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| @tauri-apps/api | ^2.12.5 | the startup bridge's invoke (backend_endpoint IPC; offline-bundled) | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| class-variance-authority | ^0.7.1 | shadcn/ui component variants | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| clsx | ^2.1.1 | shadcn/ui class merge | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| lucide-react | ^0.548.0 | the icon set | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| motion | ^12.23.24 | Motion for React — restrained shell animations (§8.5) | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| plotly.js-dist-min | ^3.1.1 | the mandated scientific charting stack (declared + chunked; src adoption is the §3.2 campaign row) | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| react | ^19.0.0 | the UI framework (the SPA's components/hooks) | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| react-dom | ^19.0.0 | React's DOM renderer | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| tailwind-merge | ^3.3.1 | shadcn/ui class merge | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| three | ^0.180.0 | the Three.js 3D molecular viewer + trajectory playback (§23) | Required (runtime) | Frontend | YES (bundled into the Vite static dist/) | package.json + bun.lock; tsc/vite build gates
| @tailwindcss/vite | ^4 | the Tailwind 4 Vite integration (build-time) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| @testing-library/jest-dom | ^8.0.0 | build-time tooling (devDependency) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| @testing-library/react | ^16.3.3 | build-time tooling (devDependency) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| @testing-library/user-event | ^14.6.13 | build-time tooling (devDependency) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| @types/node | ^26.6.6 | type declarations (build-time) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| @types/react | ^19.0.0 | type declarations (build-time) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| @types/react-dom | ^19.0.0 | type declarations (build-time) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| @types/three | ^0.180.0 | type declarations (build-time) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| @vitejs/plugin-react | ^5.1.1 | the React transform (build-time) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| jsdom | ^31.0.1 | build-time tooling (devDependency) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| openapi-typescript | ^7.9.1 | generates src/types/api.d.ts from contracts/openapi.json (codegen-time — §12.2's single DTO source) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| tailwindcss | ^4 | the styling engine (build-time CSS generation) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| typescript | ^5 | strict typechecking (build-time) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| vite | ^7.1.10 | the static-asset build tool (build-time; dist/ ships) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| vitest | ^5.1.1 | build-time tooling (devDependency) | Build-time only | Frontend (toolchain) | NO (never ships) | package.json + bun.lock
| description (Rust) | The PharmaML Studio desktop host (Tauri 2): the offline WebView2 shell, the Python backend sidecar supervisor, and nothing else — no scientific logic lives in Rust (part11 §3). | host dependency (Cargo.toml) | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| edition (Rust) | 2021 | host dependency (Cargo.toml) | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| license (Rust) | MIT | host dependency (Cargo.toml) | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| name (Rust) | pharmaml-studio | host dependency (Cargo.toml) | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| opt-level (Rust) | s | host dependency (Cargo.toml) | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| panic (Rust) | abort | host dependency (Cargo.toml) | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| repository (Rust) | https://github.com/MrMMA2004/pharma-ml-toolkit | host dependency (Cargo.toml) | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| rust-version (Rust) | 1.90 | host dependency (Cargo.toml) | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| serde (Rust) | 1 | the BackendEndpoint IPC serialization | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| serde_json (Rust) | 1 | the handshake/IPC JSON layer | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| tauri (Rust) | 2 | the Tauri 2 host core (window, IPC, sidecar supervision) | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| tauri-build (Rust) | 2 | the Tauri build script (build-time) | Build-time only | Host | NO (never ships) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)
| version (Rust) | 1.2.229 | host dependency (Cargo.toml) | Required (runtime) | Host | YES (compiled into pharmaml-studio.exe) | src-tauri/Cargo.toml (cargo build --release on the Windows runner)

### Runtimes and redistribution notes (§29's runtime-dependency audit)

- **Microsoft WebView2 Runtime**: `{}` in
  `tauri.conf.json` — the offline Evergreen installer rides the
  NSIS/MSI bundle (part11_2 §7 / §32); the workflow's step 19 installs
  it offline; the release manifest records the selected mode.
- **Node.js / bun / Vite / TypeScript / Tailwind**: build-time only —
  the end user needs none of them (`vite build` emits the static
  `frontendDist` assets the Tauri host loads locally).
- **Rust toolchain (rustc/cargo 1.77+)**: build-time only.
- **Python runtime + the table above's wheels**: bundled via the
  PyInstaller onedir sidecar (`packaging/windows/pharmaml_backend.spec`)
  — the end user needs no Python, conda, or pip (part11_2 §5).
- **AutoDock trio (vina/autogrid4/autodock4)**: the engines/ table above.
- **No CDN, no remote URL**: the offline CSP pins the frontend to
  `self` + the loopback API (`frontend/index.html`, drift-guarded).

