# Release readiness — the Phase-6 evidence report (part6 §16)

Generated 2026-11-13 by `tools/release_readiness.py`.
Every number below comes from a live probe or a recorded test
run; sections without gathered evidence say so and name the
command that gathers them. Nothing here converts absence of
evidence into a pass.

## 1. Dependency matrix (§12)

| Dependency | Status (this environment) | Consumed by |
| --- | --- | --- |
| numpy | installed | everything |
| pandas | installed | tables, QSAR, drift |
| scikit-learn | installed | the ml.* modules |
| scipy | installed | stats, optimization |
| joblib | installed | workers, Memory |
| matplotlib | installed | viz, reports, figures |
| rdkit | installed | chem.*, docking prep |
| mordred | installed | descriptors (mordred set) |
| meeko | installed | SMILES -> PDBQT |
| gemmi | installed | receptor prep |
| selfies | installed | chem.selfies_codec |
| xgboost | installed | ml boosting estimators |
| lightgbm | installed | ml boosting estimators |
| catboost | installed | ml boosting estimators |
| shap | installed | ml.explain (TreeExplainer) |
| torch | installed | dnn estimators, generative |
| PySide6 | installed | the Studio GUI |
| pyqtgraph | installed | Studio charts |

18 installed / 18 probed. Optional rows that are not installed are honestly reported
as unavailable — never as broken, and never silently.

All probed dependencies import cleanly in this environment (the Qt rows are probed under the committed EGL-stub bootstrap, so their status matches what the test suites actually exercise).

## 2. Capability coverage

- Registry rows: **60**
- integrated: **60**
- Rows without test references: **0**

## 3. Test evidence

- Suites discovered: **186** (46 UI, 140 library)
- Tests collected: **5880**

| Suite chunk (as actually run) | Result |
| --- | --- |
| `library chunks 1-10 (§14.2 sequential, 139 suites)` | 5234 passed, 13 skipped, 0 failed |
| `UI battery (46 suites, class-split for nine_card_first)` | 589 passed, 0 failed |
| `gates (registry/inventory/carriers/docs-index)` | 81 passed |
| `phase6 readiness pins` | 6 passed |

Chunkening follows §14.2 (low-memory constraints): suites
run sequentially in small groups; nothing was skipped in
bulk and no skipped test is claimed as passed.

## 4. Version carriers (§15)

- VERSION: `1.2.193`
- pyproject.toml: `1.2.193`
- src/pharmaml/__init__.py: `1.2.193`
- CHANGELOG.md leads with: `1.2.193`
- Agreement: **yes**

## 5. Release-archive verification (§14)

- Status: **passed** (8 checks)

