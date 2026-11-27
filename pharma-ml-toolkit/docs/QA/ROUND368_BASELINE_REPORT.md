# ROUND 368 BASELINE REPORT (part19, Phase 0)

Date: 2026-11-27 (Asia/Tehran session)
Baseline target: commit `53f2c92d033bd7fd82481c12943147e2c0c5825e`, version **1.2.243**, branch `main`.
Working branch for Round 368: `round368` (created from the verified `origin/main` = 53f2c92d; no direct changes to `main`).

## 1. Baseline reconciliation (verified locally)

| Item | Evidence | Status |
|---|---|---|
| Local git HEAD | `git fetch origin` → `origin/main` = 53f2c92d; `round368` created at that SHA; worktree **clean** (`git status --porcelain` = 0 entries) | Verified locally |
| Tag `v1.2.243` | `git ls-remote --tags origin v1.2.243` → tag object `fcd3b79a…`, resolves to commit `902dd89b` (the fixed tree the release was built from) | Verified locally + remote |
| Version carriers | `VERSION` = 1.2.243; `pyproject.toml` `version = "1.2.243"`; `src/pharmaml/__init__.py` `__version__ = "1.2.243"`; root web console `src/lib/toolkit-version.ts` `SHIPPED_TOOLKIT_VERSION = "1.2.243"` | Verified locally (all four) |
| Python test collection | `pytest --collect-only -q` → **8,274 tests / 441 files** (matches the r367 claim exactly) | Verified locally |
| Vitest | `bun run test` (frontend) → **357 passed / 357** (52.5 s) | Verified locally (executed, not just collected) |
| TypeScript | `tsc --noEmit` → exit 0 | Verified locally |
| mypy | `mypy src/pharmaml` → **Success: no issues found in 314 source files** | Verified locally |
| ruff | `ruff check src/pharmaml` → 70 × EXE002 **local-only residue** (see §3) | Verified locally (documented sandbox artifact) |
| API penetration-security tests | `pytest --collect-only tests/api/test_penetration_security.py` → **67 collected** | Verified locally |
| Legacy-retirement guards | `pytest --collect-only tests/test_legacy_retirement.py` → 21 collected (13 legacy-retirement + 4 quarantine guards + drift guards; full run in Phase 12) | Verified locally (collection) |
| Committed OpenAPI contract | `contracts/openapi.json` → **142 paths** (the r367 commit message said 144; docs said 142 — the committed contract is 142; runtime derivation happens in Phase 3) | Verified locally (static) |
| Repository size | `git count-objects -vH` → size-pack **942.12 MiB** ≈ 987 MB (the inspected GitHub-metadata claim, binary→decimal) | Verified locally (approximation of the API value) |

## 2. Remote evidence (verified via the GitHub API, authenticated session tools)

| Item | Evidence | Status |
|---|---|---|
| Release `v1.2.243` | id 460002401, published 2026-11-26T23:39:57Z, `draft:false`, `prerelease:false`, target `main` | Verified by GitHub API |
| Release assets | `PharmaML_Studio_Setup_v1.2.243.exe` **558,600,414 B** (digest `sha256:58e8e144…b2d6f`); `PharmaML_Studio_v1.2.243.msi` **711,110,134 B** (digest `sha256:39631928…fcc6`); `RELEASE_MANIFEST.json` 510 B; `sbom.cdx.json` 23,634 B; `SHA256SUMS.txt` 368 B — all uploaded by `github-actions[bot]` (the bot upload is itself independent proof the Windows Release workflow ran to completion) | Verified by GitHub API |
| Open pull requests | exactly one open PR: **#15 "Clarify toolkit parity and verification claims"** (open, `mergeable_state: dirty`, head `arena/3eb4a9a6-pharma-ml-toolkit` @ `c1e3b1cd`, base `main` @ `902dd89b`, 1 commit, +1,872/−2,015 over 138 files) | Verified by GitHub API |
| Repo metadata | default branch `main`; private repository (anonymous API cannot read it; all remote evidence above used the authenticated session tools) | Verified by API behavior |

## 3. Reported by historical documentation only (not independently re-verified this round)

- Windows Release workflow run **45907478532** and Windows Install Smoke run **45912713199** (SUCCESS, 20 steps) — run IDs from the r367 close-out ledger; no workflow-run listing tool was available in this session to re-read the runs themselves. The published release assets (§2) independently confirm the release lane completed; the smoke run's own evidence bundle remains CI-side.
- Push CI runs 45907471751 (CI) + 45907472523 (Push) "GREEN over 902dd89b" — same class: reported, not re-read.
- The 8,274/357 test counts were **re-measured this round** (§1) and match; they are not stale copies.

## 4. Local lint residue decoded (the 70 × EXE002)

The r367 register recorded "local-EXE002-only residue". Round 368 decoded the mechanism with direct evidence: the git tree records mode `100644` for `src/pharmaml/__main__.py` (and the other 69 flagged files) while the sandbox filesystem carries mode `755` — ruff 0.17.1 fires EXE002 (executable bit set, no shebang) only on the local filesystem view. A CI checkout materializes modes from git (`100644`) and therefore does not flag them, which is exactly why the CI lint lane is green while the local lane shows residue. **Conclusion: not a repository defect; sandbox filesystem artifact; no action required** (the CI lane carries the gate, per the standing register).

One real ruff finding exists **outside** the CI lint scope: `scripts/build_capability_map.py:34` FURB188 (prefer `str.removeprefix` over a conditional slice) — registered as defect D-368-02 (LOW) and fixed this round in Phase 12 gates.

## 5. Session-start reconciliation work (sandbox drift recovery, honest record)

The session opened with the sandbox git state drifted from the pushed history: local `HEAD` sat at a stale sandbox snapshot lineage commit (`2eb0ede`, the v1.2.106 era) with **1,228 uncommitted path entries**, while the working tree content was the r367 v1.2.243 state. Reconciliation steps (all verified by post-state checks):

1. PAT restored silently from the owner's token upload to the 600-mode credential file (never printed, never committed; `.credentials/` is gitignored line 104 of the root `.gitignore`, verified untracked).
2. `git fetch origin` (ff-only bookkeeping; `origin/main` = 53f2c92d = the prompt's baseline).
3. Index reset to `origin/main`; branch `round368` created from it; local `main` re-pointed to match the remote (no commits made to `main`).
4. **Stale local-only residue removed** (166 files that exist on the sandbox disk but not in the pushed tree, i.e. pre-quarantine copies that the r367 quarantine had already retired at the remote): `pharma-ml-toolkit/ui/` (78 files), `pharma-ml-toolkit/tests/ui/` (74 files), `pharma-ml-toolkit/studio.py`, `playground.py`, `tests/test_playground.py`, the stale duplicate `pharma-ml-toolkit/.github/workflows/ci.yml` (the live workflows live at the repo root), four superseded zip copies in stale homes (`download/v1.2.105–106`, toolkit `dist-archive/v1.2.105–106–238`, toolkit `download/v1.2.103`), `scratch/__pycache__` pyc files, `.zscripts/dev.pid`. `.env` (SQLite URL only, no secrets) kept on disk untracked for the web console runtime.
5. `git checkout -- .` restored exact tracked modes; final `git status --porcelain` = **0 entries** (clean baseline reproduced).

venv drift was found and repaired with the standing `scratch/venv_repair.sh FOREGROUND` (numpy 1.26.4, rdkit 2026.9.1, torch 2.14.1+cpu, matplotlib 3.10.9, pharmaml 1.2.243 editable, PySide6 stack for the explicitly-run legacy lane). The dev server (web console) is up via `scratch/dev_daemon.py` (GET / 200).

## 6. Verdict

The Round 367 baseline is **reproducible locally** and matches the pushed remote state at `53f2c92d` / tag `v1.2.243` / release assets byte-for-byte on sizes and digests. No stale test counts were used (all key figures re-measured this round). Round 368 work proceeds on `round368`.
