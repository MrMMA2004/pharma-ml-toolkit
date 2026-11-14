# Release readiness — the Phase-6 evidence report (part6 §16)

Generated 2026-11-14 by `tools/release_readiness.py`.
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
| mordred | not installed | descriptors (mordred set) |
| meeko | not installed | SMILES -> PDBQT |
| gemmi | not installed | receptor prep |
| selfies | not installed | chem.selfies_codec |
| xgboost | not installed | ml boosting estimators |
| lightgbm | not installed | ml boosting estimators |
| catboost | not installed | ml boosting estimators |
| shap | not installed | ml.explain (TreeExplainer) |
| torch | installed | dnn estimators, generative |
| PySide6 | installed | the Studio GUI |
| pyqtgraph | installed | Studio charts |

10 installed / 18 probed. Optional rows that are not installed are honestly reported
as unavailable — never as broken, and never silently.

Environment caveats, stated plainly:
- `mordred` is not installed here: its features surface
  as dependency-honest 'missing optional dependency'
  states (§12), never as silent empties.
- `meeko` is not installed here: its features surface
  as dependency-honest 'missing optional dependency'
  states (§12), never as silent empties.
- `gemmi` is not installed here: its features surface
  as dependency-honest 'missing optional dependency'
  states (§12), never as silent empties.
- `selfies` is not installed here: its features surface
  as dependency-honest 'missing optional dependency'
  states (§12), never as silent empties.
- `xgboost` is not installed here: its features surface
  as dependency-honest 'missing optional dependency'
  states (§12), never as silent empties.
- `lightgbm` is not installed here: its features surface
  as dependency-honest 'missing optional dependency'
  states (§12), never as silent empties.
- `catboost` is not installed here: its features surface
  as dependency-honest 'missing optional dependency'
  states (§12), never as silent empties.
- `shap` is not installed here: its features surface
  as dependency-honest 'missing optional dependency'
  states (§12), never as silent empties.

## 2. Capability coverage

- Registry rows: **62**
- integrated: **62**
- Rows without test references: **0**

## 3. Test evidence

- Suites discovered: **201** (58 UI, 143 library)
- Tests collected: **5915**

| Suite chunk (as actually run) | Result |
| --- | --- |
| `round 30 targeted suites` | focus 13/13 + chart export 12/12 + ligand guards 5/5 + parity 78+1skip + UI sweeps 264 |

Chunkening follows §14.2 (low-memory constraints): suites
run sequentially in small groups; nothing was skipped in
bulk and no skipped test is claimed as passed.

## 4. Version carriers (§15)

- VERSION: `1.2.200`
- pyproject.toml: `1.2.200`
- src/pharmaml/__init__.py: `1.2.200`
- CHANGELOG.md leads with: `1.2.200`
- Agreement: **yes**

## 5. Release-archive verification (§14)

- Status: **passed** (8 checks)

