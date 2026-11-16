# Release readiness — the Phase-6 evidence report (part6 §16)

Generated 2026-11-16 by `tools/release_readiness.py`.
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

- Suites discovered: **223** (70 UI, 153 library)
- Tests collected: **6452**

| Suite chunk (as actually run) | Result |
| --- | --- |
| `1 (24 files)` | passed — 86s, re-run after the 8 notebooks re-executed at 1.2.216 (the 22:45 rc=1 was the notebook-freshness gate catching the stale 1.2.215 outputs — the guard worked as designed) |
| `2 (28 files)` | passed — 155s |
| `3 (27 files)` | passed — 93s |
| `4 (28 files)` | passed — 294s |
| `5 (29 files)` | passed — 84s |
| `6 (29 files)` | passed — 158s re-run after the FEATURES.md freeze stamp trued 1.2.213→1.2.216 (the rc=1 was exactly test_purpose_header_and_freeze_stamp) |
| `7 (28 files)` | passed as split legs — 7.1/7.2/7.3 (17s/21s/26s) + 7.4a six files (32s) + test_nine_card_first as THREE class-scoped legs (142s/64s/42s; 9+8+6=23). Single-process OOM persists post conftest teardown fix: ~150MiB retained per MainWindow instantiation (global-side accumulation, pyqtgraph ViewBox warnings; backlog), so the gate never claims it green in one process |
| `8 (29 files)` | passed — 182s |
| `9 (nine-card suite, own Qt envelope)` | passed — 21s |
| `10 (src doctests)` | passed — 82s |

Chunkening follows §14.2 (low-memory constraints): suites
run sequentially in small groups; nothing was skipped in
bulk and no skipped test is claimed as passed.

## 4. Version carriers (§15)

- VERSION: `1.2.216`
- pyproject.toml: `1.2.216`
- src/pharmaml/__init__.py: `1.2.216`
- CHANGELOG.md leads with: `1.2.216`
- Agreement: **yes**

## 5. Release-archive verification (§14)

- Status: **not gathered**
- How to gather: `tools/audit_release.py <zip> after tools/build_zip.py`

