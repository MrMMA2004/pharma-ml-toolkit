# Feature Gap and GitHub References (part12 PART 20)

Every gap below states: what problem it solves, whether an equivalent
already exists, dependencies, integration strategy, testing requirement,
and performance implications — per part12's citation rules. Items are
ordered by value density. Status legend: OPEN (owner decision), PARKED
(behind a gate), DONE (this round).

## Fixed this round (carried for the record)

1. **Database GUI wiring** — DONE (B-2). Problem: a complete, tested
   persistence API (backup/restore/preferences/job history) with no
   product consumer and an over-claiming Introduction card. Equivalent
   existed: backend only. Integration: typed client + live card.
   Testing: backend suite (67 tests) + round E2E. Performance: one
   bounded fetch on Introduction mount.
2. **Scoped explorer/ZIP exclusions** — DONE (B-1/B-1b). Problem: real
   source dropped from shipped archives; build output leaking into
   census. No external references needed — a path-anchoring fix.

## Scientific gaps (OPEN, each small and evidence-justified)

3. **Leverage / Williams AD method** (B-4). Why: the standard regression-AD
   companion to the existing kNN-distance method; part11/12 name it.
   Exists: hat matrix only inside Cook's distance — no standalone method.
   References: `scikit-learn/scikit-learn` (LinearRegression hat-matrix
   idiom), `rdkit/rdkit`. Deps: none new. Strategy: one function in
   `ml/applicability.py` (h\* = 3(p+1)/n, standardized residuals) + a
   Williams scatter in the ML AD stage + catalog row. Tests: numerical
   edge cases (p+1>n guard, perfect-fit divide-by-zero) + GUI contract
   guard. Performance: O(n·p) once — negligible.
4. **Fingerprint chirality toggle** (B-5). Why: `includeChirality=False`
   is pinned while nBits/radius/counts are configurable. Exists: the
   parameter is native RDKit (`GetMorganGenerator(includeChirality=…)`).
   References: `rdkit/rdkit`. Strategy: expose the toggle for morgan/fcfp
   families only (algorithms that use it), persist it in dataset
   metadata like the other fingerprint params. Tests: parity vs RDKit
   direct call; a chirality-sensitive SMILES pair regression. Performance:
   negligible. **Caution**: changes feature semantics — a data-compat
   decision (default stays off).
5. **Semi-supervised `/evaluate` GUI lane**. Why: the route exists and is
   tested; the GUI runs evaluation through the train payload. Strategy:
   one button in the ML semi-supervised stage posting the saved run id.
   Tests: contract guard. Small, independent.

## Architecture hygiene (OPEN)

6. **Tk playground true retirement** (B-6) — `playground.py` 30,047
   lines + `test_playground.py` 16,370 lines, declared retired, still
   shipped. Why: dead weight in archive/installer and test budget.
   Equivalent: the PySide6 Studio + React app cover the surfaces
   (FEATURES.md grades them A). Strategy: delete module + test file +
   requirements refs + docstring refs; re-true census/test pins;
   re-ship zip. Gate: same evidence rule as PySide6 Stage-5 — this is a
   Phase-8 owner decision.
7. **Doctest-only modules + duplicate symbols** (A-6/B-7): 9 modules
   imported nowhere; `gdsc_anova`/`mabel_dose`/`scaffold_split` each
   implemented in 2–3 places. Strategy: pick canonical modules, demote
   duplicates to thin re-exports or delete, add import-graph guard.
   References: internal only. Performance: compile/import time only.

## Packaging/distribution gaps (PARKED or owner-gated)

8. **SBOM generation** (part12 PART 16). Why: provenance for the
   installer deps. References: `CycloneDX/cyclonedx-python`. Strategy: run
   against the release venv lockfile in the Windows workflow after step
   22; attach `sbom.cdx.json` to the release. Deps: one tool addition to
   the runner. Testing: schema validation step. Performance: seconds.
   Status: OPEN (owner sign-off on workflow change).
9. **Digital signing** — BLOCKED on certificate availability (owner).
10. **Clean-machine verification** — BLOCKED (no machine; see
    WINDOWS_CLEAN_INSTALL_REPORT.md).
11. **Installer size tiering** — OPEN decision (torch-in-bundle vs
    on-demand), explicitly NOT taken silently.

## Referenced repositories (part12 PART 20 citations, dispositioned)

- `pola-rs/polars` — evaluated, not adopted (no measured benefit; decision
  recorded in PERFORMANCE_REPORT.md).
- `scikit-learn/scikit-learn` — semi-supervised + AD references stand;
  leverage/Williams gap parked as item 3.
- `rdkit/rdkit` — descriptor/fingerprint core in production; chirality
  gap item 4.
- `joblib/joblib`, `giampaolo/psutil` — joblib in the pinned stack; psutil
  deliberately NOT a dependency (resources probe is /proc-based).
- `plotly/plotly.js`, `3dmol/3Dmol.js`, `molstar/molstar` — plotly +
  three.js in production; molstar/3dmol not adopted (three.js lane
  covers the shipped 3D needs — recorded decision).
- `optuna/optuna`, `catboost/catboost`, `microsoft/LightGBM`,
  `dmlc/xgboost` — boosters in production; Optuna not adopted (existing
  tuner covers; recorded).
- `mlflow/mlflow`, `iterative/dvc` — not adopted: local-first SQLite +
  FileVault experiment tracking covers the requirement with no external
  server (part12's own "do not add an external tracking server unless
  demonstrated").
- `CycloneDX/cyclonedx-python` — SBOM gap item 8.
- `pyinstaller/pyinstaller`, `tauri-apps/tauri` — in production (release
  ladder green).
- `chemprop/chemprop`, `deepchem/deepchem` — message-passing/GNN space:
  `ml/gnn.py`/`ml/mpnn.py`/`ml/geometric.py` exist as library-surface
  modules; GUI wiring NOT claimed (matrix records the tier honestly).
- `pytest-dev/pluggy` — plugin architecture NOT added (part12 PART 15:
  no framework added solely for decoration — the registry/capability
  pattern covers current extensibility; recorded decision).
- `unionai-oss/pandera`, `deepchem/moleculenet` — data-quality layer
  implemented with the native profiling suite; pandera not adopted
  (equivalent coverage recorded).

## Verdict

No fabricated coverage: every "exists" above names the file that holds it;
every "gap" names the lane where it is honest. The three highest-value
open items are #3 (Williams plot), #6 (Tk retirement), #8 (SBOM) — each a
small, isolated, evidence-justified increment for the next campaign.
