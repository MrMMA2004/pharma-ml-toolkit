# Release readiness — the Phase-6 evidence report (part6 §16)

Generated 2026-11-19 by `tools/release_readiness.py`.
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

- Registry rows: **62**
- integrated: **62**
- Rows without test references: **0**

## 3. Test evidence

- Suites discovered: **243** (72 UI, 171 library)
- Tests collected: **7401**

**Not gathered this run** — run the §14.2-chunkened suites
and record results, then re-generate with
`--record <chunk>=<result>` (repeatable).

## 4. Version carriers (§15)

- VERSION: `1.2.231`
- pyproject.toml: `1.2.231`
- src/pharmaml/__init__.py: `1.2.231`
- CHANGELOG.md leads with: `1.2.231`
- Agreement: **yes**

## 5. Release-archive verification (§14)

- Status: **not gathered**
- How to gather: `tools/audit_release.py <zip> after tools/build_zip.py`

