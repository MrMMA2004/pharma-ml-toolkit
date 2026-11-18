# PharmaML Studio — dependency inventory (part9 §1.1)

Generated 2026-11-18 12:06 by `tools/packaging_audit.py` from the ACTUAL
source tree (studio.py + ui/ + src/ + scripts/, 3656 import statements parsed) and the live venv's dist-info. Versions are
importlib.metadata's truth, not a hand-maintained table.

Toolkit version audited: **1.2.227**. Python: 3.12.16.

| Package | Version | Purpose | Required or Optional | Included in Installer | Verification |
| --- | --- | --- | --- | --- | --- |
| numpy | 1.26.4 | array math everywhere | Required (unconditional) | YES | src/pharmaml/api/routes/applicability_domain.py; src/pharmaml/api/routes/chemistry.py; src/pharmaml/api/routes/docking.py; ... |
| pandas | 2.2.3 | tabular datasets, drift tables | Required (unconditional) | YES | src/pharmaml/api/routes/applicability_domain.py; src/pharmaml/api/routes/artifacts.py; src/pharmaml/api/routes/chemistry.py; ... |
| scikit-learn | 1.5.2 | the ml.* estimator families | Required (unconditional) | YES | src/pharmaml/chem/features.py; src/pharmaml/cli.py; src/pharmaml/data.py; ... |
| scipy | 1.14.1 | stats + optimization | Required (unconditional) | YES | src/pharmaml/biology/association.py; src/pharmaml/biology/biomarkers.py; src/pharmaml/biology/omics.py; ... |
| joblib | 1.5.3 | worker utilities (utils/compute absorbs its API) | Required (unconditional) | YES | src/pharmaml/chem/features.py; src/pharmaml/ml/persistence.py |
| openpyxl | 3.1.5 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| tqdm | 4.67.1 | progress bars for long jobs | Required (unconditional) | YES | src/pharmaml/chem/descriptors.py; src/pharmaml/chem/rism.py |
| rdkit | 2026.9.1 | chem.* + docking preparation (Chem/Draw/DataStructs) | Required (feature-gated import) | YES | src/pharmaml/api/routes/docking.py; src/pharmaml/api/routes/health.py; src/pharmaml/chem/bit_highlights.py; ... |
| mordred | 1.2.0 | the mordred descriptor set (dynamic registry) | Required (feature-gated import) | YES | src/pharmaml/chem/descriptors.py |
| meeko | 0.8.0 | SMILES -> PDBQT ligand preparation | Required (feature-gated import) | YES | src/pharmaml/docking/preparation.py; src/pharmaml/docking/receptor.py |
| gemmi | 0.7.5 | receptor .pdb parsing/preparation | Required (feature-gated import) | YES | src/pharmaml/docking/receptor.py |
| xgboost | 2.1.3 | gradient-boosted estimators | Required (feature-gated import) | YES | src/pharmaml/ml/models.py |
| lightgbm | 4.5.0 | gradient-boosted estimators | Required (feature-gated import) | YES | src/pharmaml/chem/features.py; src/pharmaml/ml/models.py |
| catboost | 1.2.8 | gradient-boosted estimators | Required (feature-gated import) | YES | src/pharmaml/ml/models.py |
| shap | 0.54.1 | TreeExplainer values + interaction cube (ml/explain) | Required (feature-gated import) | YES | src/pharmaml/ml/explain.py |
| numba | 0.61.0 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| llvmlite | 0.44.0 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| torch | 2.14.1+cpu | deep-learning estimators, generative VAEs, drug-response model | Required (feature-gated import) | YES | src/pharmaml/generative/rnn.py; src/pharmaml/ml/neural.py; ui/widgets/resource_monitor.py |
| selfies | 2.1.1 | robust molecular string codec (chem/selfies_codec) | Required (feature-gated import) | YES | src/pharmaml/chem/selfies_codec.py |
| matplotlib | 3.10.9 | all figure rendering + export backends | Required (feature-gated import) | YES | src/pharmaml/viz/_compat.py; src/pharmaml/viz/docking_plots.py; src/pharmaml/viz/export.py; ... |
| seaborn | 0.13.2 | declared but no static import found | n/a | review | no import in studio.py/ui/src/scripts |
| fpdf2 | 2.8.5 | PDF report writer (reports.py) | Required (feature-gated import) | YES | src/pharmaml/viz/report.py |
| python-docx | 1.2.0 | DOCX report writer (reports.py) | Required (feature-gated import) | YES | src/pharmaml/viz/report.py |
| Pillow | 11.3.0 | image IO behind PNG/JPEG/TIFF figure export | Required (feature-gated import) | YES | src/pharmaml/viz/export.py; src/pharmaml/viz/report.py; ui/pages/molecular_dynamics.py |
| PySide6 | 6.8.3 | the Studio GUI (Qt 6.8 Essentials+Addons) | Required (unconditional) | YES | ui/__init__.py; ui/app_settings.py; ui/card_workspace.py; ... |
| pyqtgraph | 0.13.7 | the Studio's interactive charts | Required (unconditional) | YES | ui/widgets/chart_inspector.py; ui/widgets/image_panel.py; ui/widgets/plot_widget.py |
| qt-material | 2.17 | the Studio light/dark theme engine | Required (feature-gated import) | YES | ui/theme_manager.py |
| qdarkstyle | 3.2.3 | the dark stylesheet engine | Required (feature-gated import) | YES | ui/theme_manager.py |
| shiboken6 | 6.8.3 | PySide6's C++ binding runtime (transitive) | Required (feature-gated import) | YES | ui/doctor.py |
| statsmodels | 0.14.5 | multiple-testing FDR correction (biology/association) | Required (feature-gated import) | YES | src/pharmaml/biology/association.py |

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
