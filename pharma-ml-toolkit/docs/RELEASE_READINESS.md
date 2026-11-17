# Release readiness — the Phase-6 evidence report (part6 §16)

Generated 2026-11-17 by `tools/release_readiness.py`.
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

- Suites discovered: **224** (70 UI, 154 library)
- Tests collected: **6459**

| Suite chunk (as actually run) | Result |
| --- | --- |
| `chunk-1 (24 files)` | 99s rc=0 green |
| `chunk-2 (28 files)` | 140s rc=0 green |
| `chunk-3 (28 files)` | 141s rc=0 green |
| `chunk-4 (28 files)` | 403s rc=1 ENV (node_modules eviction; makeup leg A green: 27 files + 3 web-parity probes with node_modules restored) |
| `chunk-5 (29 files)` | 96s rc=0 green |
| `chunk-6 (29 files)` | 152s rc=0 green |
| `chunk-7 (29 files)` | OOM-split; sub-batches 1,3,5 green; sub-batch-2 rc=1 ENV (pyinstaller gap; makeup leg B green); nine_card_first single rc=-9 (makeup leg C green in two halves) |
| `chunk-8 (28 files)` | 145s rc=0 green |
| `chunk-9 nine-card suite` | 21s rc=0 green |
| `chunk-10 src doctests` | 86s rc=0 green |
| `chunk-11 api suite (42 tests)` | 12s rc=0 green |
| `makeup-A web_parity probes` | 3/3 green (fresh node_modules) |
| `makeup-B freeze_equivalent` | 7 tests rc=0 green (pyinstaller 6.22.3 restored) |
| `makeup-C nine_card_first` | 23 tests in two halves rc=0 green (post-OOM) |

Chunkening follows §14.2 (low-memory constraints): suites
run sequentially in small groups; nothing was skipped in
bulk and no skipped test is claimed as passed.

## 4. Version carriers (§15)

- VERSION: `1.2.219`
- pyproject.toml: `1.2.219`
- src/pharmaml/__init__.py: `1.2.219`
- CHANGELOG.md leads with: `1.2.219`
- Agreement: **yes**

## 5. Release-archive verification (§14)

- Status: **not gathered**
- How to gather: `tools/audit_release.py <zip> after tools/build_zip.py`

