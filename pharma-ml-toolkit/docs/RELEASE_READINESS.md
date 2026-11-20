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


---

## part12 addendum — Phase-1 re-probe (2026-11-20, round 302)

Re-probed against the audited baseline v1.2.231 / `7bbc41ed` (live
session; full detail in `docs/part12_audit/`):

- **Dependencies**: 18/18 installed on the repaired venv (numpy 1.26.4 ·
  torch 2.14.1+cpu · rdkit 2026.9.1 · shap 0.54.1 · PySide6 6.8.3 ·
  FastAPI 0.128.0 · PyInstaller 6.22.3 · pharmaml 1.2.231 editable) —
  import-surface + offscreen-QApplication probes green.
- **Version carriers**: 8/8 at 1.2.231 (VERSION, pyproject,
  `__version__`, toolkit-version.ts, frontend package.json,
  tauri.conf.json, lockfile, version_info.txt quadruple) — read directly
  this session; the two named guard tests enforce them.
- **Test collection (r303 re-true)**: 8,643 collected / 0 errors
  (46 collected-file tally, 456 files) — matches the shipped pin; the
  r303 additions over the r302 8,590: the Williams lane (23 unit +
  3 figure + 7 route), the chirality toggle (8 descriptor + 8 factory +
  3 route), and the /evaluate job-mode COMPLETION pin (1).
- **Census (r303 re-true)**: the walk = 1,591 files / 597,662 lines —
  the r302 tree (the same tracked set: the round changed no file's
  existence, only content) plus the round's +1,654 viewable lines (the +9: the docs/INDEX.md regeneration for the WilliamsResult/leverage_williams entries; the +15: the build/theme-shots anchored exclusion across the canonical builders);
  the five census-riding ghosts swept (the stale toolkit-dir
  windows-release.yml + the 229/230 zip recovery residue in three
  local homes; the offsite home keeps the full 221-231 history).
- **Live release**: v1.2.231 published (not draft) with EXE
  557,655,583 B + MSI 710,122,853 B + manifest + checksums; CI green on
  the audited commit; Windows Release run 44693818307 all-26-green.
- **Archive note (r302, closed)**: the shipped v1.2.231 source zip was
  missing exactly one real source file (`frontend/src/services/artifacts/
  client.ts` — the B-1 exclusion defect); the r302 re-ship carries it
  complete, and the r303 archive re-ships over that fixed base.
- **Honest gaps carried**: SBOM (open, CycloneDX lane documented),
  signing (certificate-gated), clean-machine ladder (BLOCKED — see
  part12_audit/WINDOWS_CLEAN_INSTALL_REPORT.md).
