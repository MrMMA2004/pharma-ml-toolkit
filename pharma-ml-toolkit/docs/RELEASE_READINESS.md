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
- **Census (r303 re-true)**: the walk = 1,591 files / 597,658 lines —
  the r302 tree (the same tracked set: the round changed no file's
  existence, only content) plus the round's +1,650 viewable lines (the +9: the docs/INDEX.md regeneration for the WilliamsResult/leverage_williams entries; the +15: the build/theme-shots anchored exclusion across the canonical builders);
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
- **Honest gaps carried**: signing (certificate-gated),
  clean-machine ladder (BLOCKED — see
  part12_audit/WINDOWS_CLEAN_INSTALL_REPORT.md); SBOM was open here
  and CLOSED by the r304 Phase-3 addendum below.

## part12 addendum — Phase-3 (2026-11-20, round 304: the architecture-hygiene round)

- **B-6/A-7 (the Tk playground true retirement)**: `playground.py`
  (30,047 lines) and `tests/test_playground.py` (16,370 lines, 753
  collected) left the shipped tree — the v1.2.141 declaration
  completed, the final version preserved in git history. The
  collection re-trued 8,643 → 8,362 (−753 retired, +472 the new
  architecture-hygiene guard). README/COVERAGE/requirements trued to
  the honest wording (declared v1.2.141, truly removed r304).
- **B-7/A-6 (duplicate symbols + the import-graph guard)**: the
  gdsc_anova ×2 / mabel_dose + noael_hed ×3 / scaffold_split ×2
  families carry docstring cross-reference maps at all ten definition
  sites, and `tests/test_architecture_hygiene.py` (472 tests) pins
  the definition-site sets, the nine-module library-surface
  zero-importer census, the mabel mg-chain NUMERIC parity (fih ↔
  pbpk over a 144-point grid), the unitless pkpd relationship
  (mg = raw · MW/1e6), the noael trio's default parity, and the
  registry references — drift now fires a test, not an audit.
- **B-8 (SBOM — CLOSED this round)**:
  `packaging/sbom/sbom.cdx.json` — the CycloneDX 1.6 SBOM of the
  release stack (installer-requirements + the step-7 extras: torch,
  shap, the exact pyinstaller pin, pytest; 40 components, validated,
  reproducible) ships in the source archive, and the Windows
  workflow's new step 22b regenerates it on the runner with
  `--validate`, writes it into the release dir (riding the artifact
  upload and the release attach untouched) and self-registers its
  hash into SHA256SUMS.txt; the step is order-pinned by
  `test_windows_pipeline.py`'s walk. Signing remains certificate-
  gated; the clean-machine ladder remains BLOCKED (no machine).
- **Test collection (r304 re-true)**: 8,362 collected / 0 errors —
  8,643 − 753 (the retired Tk suite) + 472 (the hygiene guard).
- **Census (r304 re-true)**: the live `/api/toolkit/tree` walk
  re-trued after the ship (the −2 files / −46,417 lines of the
  retirement plus the round's additions); pins synced in
  `toolkit-stats.ts`.

## part17 upkeep addendum — the r361 firing (2026-11-26, the fourteenth-class rollback recovery + the gate's three finds fixed at source)

- **Recovery**: the durable `origin/main` (`4c86969d`, the r360
  terminal) re-synced ff-only after the sandbox recycle regressed the
  local `.git` to the round-186 checkpoint; the venv rebuilt at the
  exact pins; the five zip homes re-seeded byte-identical
  (`0fadb6e5…`, exactly 23,489,390 B); the release-archive
  verification status above is unchanged (still not gathered for a
  NEW version — v1.2.240 remains the shipped archive).
- **Gate finds fixed at source**: the QA report's own r353 row
  carried the raw `/home/<user>/` pattern (the leakage guard's
  blind-spot find — scrubbed pattern-safe); the r356-lane
  SQL-payload test's async-queue race fixed to the siblings'
  `mode="sync"` convention; the nine-card-first whole-file OOM
  decoded 5/5 classes green. The FULL tests/api leg re-run rc=0.
- **Pins (r361 re-true)**: collection 9,017 / 0 errors / 495 files
  (measured); vitest 337/337 over 38 files; the census carriers
  9,017 / 337 / 1,842 / 589,728 (pin = live walk = page render
  three-way); registry 33 / 24 unchanged.
- **Version**: unchanged at 1.2.240 — the deltas are
  doc/test/carrier-only; NO re-ship, NO silent bump. The standing
  BLOCKED lanes (clean-machine smoke, Stage-5 PySide6 retirement,
  installer signing) carry unchanged.

## part17 upkeep addendum — the r362 firing (2026-11-26, the fifteenth-class rollback recovery — the partial-restore signature)

- **Recovery**: the durable `origin/main` (`af15580c`, the r361
  terminal) re-synced ff-only after the sandbox recycle regressed the
  local `.git` to the round-186 checkpoint; the residue signature
  changed to a PARTIAL disk restore (1,405 tracked files missing,
  zero untracked residue, the survivors hash-identical to the durable
  blobs — the lossless-reset proof preceded the reset); the venv
  rebuilt at the exact pins; the six zip homes re-seeded byte-identical
  (`0fadb6e5…`, exactly 23,489,390 B — the root dist-archive + the
  offsite mirror wiped again); the release-archive verification status
  above is unchanged (still not gathered for a NEW version — v1.2.240
  remains the shipped archive).
- **The census-carrier debt settled**: the walk's +132-line drift over
  the r361-committed pin decoded (r361's own ledger close-out landed
  after its walk measurement); re-trued to the post-ledger measured
  fixed point, pin = live walk = page render three-way re-verified.
- **Pins (r362 re-true)**: collection 9,017 / 0 errors / 495 files
  (measured); vitest 337/337 over 38 files; registry 33 / 24
  unchanged; the census carriers re-measured at the close-out (the
  exact line count lives in the carrier + the worklog R362 entry).
- **Version**: unchanged at 1.2.240 — the deltas are doc/carrier-only;
  NO re-ship, NO silent bump. The standing BLOCKED lanes (clean-machine
  smoke, Stage-5 PySide6 retirement, installer signing) carry
  unchanged.

### r363 addendum (the 04:30 2h-sync upkeep firing — the sixteenth-class rollback recovered; the r362 gate debt decoded and fixed at source)

- **The rollback recovery (sixteenth-class)**: the recycle regressed the
  local `.git` to the round-186 machinery checkpoint `2eb0ede` (590
  commits behind) and restored a PARTIAL disk again (1,723 tracked
  files missing, 625,356 lines; the survivors hash-identical to the
  durable blobs; the only content deltas the same transient pid +
  corrupted nested `ci.yml` ghost pair as r362); the PAT re-spliced
  (93 chars, 600-mode, never printed), the ff-only reset onto the
  durable r362 terminal `d8308db9` restored 1,900 files, the venv
  rebuilt at the exact pins, the six zip homes re-seeded byte-identical
  (`0fadb6e5…`, exactly 23,489,390 B — the root dist-archive + the
  offsite mirror wiped again); the release-archive verification status
  above is unchanged (still not gathered for a NEW version — v1.2.240
  remains the shipped archive).
- **The r362 `failures=1` decoded and fixed at source**: the
  `studio_app` fixture's per-test no-op `setStyleSheet("")` re-polish
  deadlock (faulthandler-stacked to tests/ui/conftest.py:33) — the
  reset now applies once per process; the known r352 whole-file
  nine-card-first OOM fragility made structural in the r363 runner
  (the six-class chunk-9f1..f6 legs).
- **The gate**: `scratch/run_suite_chunks_363.py` — `failures=0`, all
  17 legs rc=0 (the first full-clean gate verdict since the r353
  lineage).
- **Pins (r363 re-true)**: collection 9,017 / 0 errors / 495 files
  (measured); vitest 337/337 over 38 files; registry 33 / 24
  unchanged; the census walk 1,842 files / 166 dirs / 589,961 lines /
  528,220,486 B (the canonical replica; the carrier stood).
- **Version**: unchanged at 1.2.240 — doc/carrier/test-infra-only
  deltas; NO re-ship, NO silent bump. The standing BLOCKED lanes
  (clean-machine smoke, Stage-5 PySide6 retirement, installer signing)
  carry unchanged.

### r364 addendum (the owner-directed release round — the seventeenth-class rollback recovered + the v1.2.241 NEW-version ship)

- **The recovery**: the seventeenth-class playbook execution against
  the same partial-restore signature (the `git diff origin/main`
  proof: only the known transient pair, 52 insertions); the PAT
  re-spliced, the ff-only reset onto the durable r363 terminal
  `81e36045`, the venv rebuilt at the exact pins, the frontend
  `npm ci`, the dev daemon restarted.
- **The v1.2.241 ship — the version-carrier section above is
  re-trued by this round**: the carriers now read 1.2.241 (VERSION,
  pyproject, `__init__`, CHANGELOG lead — plus the
  `version_info.txt` quadruple, `frontend/package.json`,
  `src-tauri/tauri.conf.json`, `frontend/src/lib/version.ts`, and
  the web `src/lib/toolkit-version.ts`), every carrier guard green
  at close-out.
- **The eight-notebook re-execution**: 8/8 in 235s,
  home-path-survivors 0 — every executed workflow prints
  `pharmaml 1.2.241`; the freshness + portability + twins guards
  re-run rc=0.
- **The gate**: the ten-chunk runner re-run — `failures=4`, all four
  decoded as the new-version-ship legs caught by design while the
  ship legs were in flight (two mid-bump carrier transients + two
  notebook-freshness legs + the github-page zip-missing leg), each
  re-run green at close-out; every other leg rc=0.
- **The release-archive verification section above is superseded by
  this round**: the shipped archive is now
  `pharma-ml-toolkit-v1.2.241.zip` (the SHA-256 + byte size
  recorded in the worklog R364 entry), seeded byte-identical across
  the six homes + the offsite mirror; the EXE/MSI released via the
  tag-triggered Windows workflow with the run verdict + asset sizes
  recorded in the worklog R364 entry; the penetration + install-smoke
  lanes dispatched and verified green.
- **Version**: NEW — 1.2.241 (the owner-directed release round); the
  counter 155 → 156 rides the commit. The standing BLOCKED lanes
  (clean-machine smoke, Stage-5 PySide6 retirement, installer
  signing) carry unchanged.
