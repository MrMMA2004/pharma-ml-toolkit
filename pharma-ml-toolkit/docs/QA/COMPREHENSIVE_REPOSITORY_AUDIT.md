# COMPREHENSIVE REPOSITORY AUDIT — part15 §13 deliverable #1

The part15 Phase-One verification matrix and reconciliation record.
Audited commit: **`0132d409`** (main == origin/main, the r339 evidence
terminal), audited round **r340** (the 132nd recorded run); the state
rows re-established live and the matrix evidence re-dated through
**r343** (the 135th recorded run — the sixth-rollback recovery round,
over the recovered `84b24f5c` tree). Statuses use
the part15 §3.2 vocabulary; a source-only implementation is never marked
VERIFIED COMPLETE without executed evidence.

## 1. Repository state (§3.1, re-established live this round — r343)

| Item | Value | Evidence |
| --- | --- | --- |
| Default branch | `main` (public repo) | live API poll, authenticated |
| HEAD | `84b24f5c` == origin/main (the r342 evidence terminal; the r343 payload rides this round) | fetch via the askpass credential lane |
| Recent commits | r342 ledger `c196849f` → r342 evidence `84b24f5c` | git log |
| CI at HEAD | 19/19 check-runs all `completed/success` | authenticated API, check-runs endpoint (closed on arrival at r343) |
| Release tag | `v1.2.240`, published (draft:false), five assets intact | r337 live verification + standing ruling |
| Version declarations | 1.2.240 across all nine carriers | `test_documentation_consistency.py` 12/12 green (the r341 record correction: 12 test methods since r324; re-run 12/12 at r343) |
| Uncommitted changes | none (tree 0/0 clean at the audited SHA, post-recovery post-sweep) | `git status` post-recovery |
| Rollback recovered | the SIXTH wave of the part15 era, the deepest class again (history reset to the r186 checkpoint `2eb0ede` — a strict ancestor of origin/main, nothing local-only to sweep; remote + PAT + venv + both node_modules + ECC + the offsite home wiped, the working tree survived at the r342 terminal content); the r322 playbook re-run in full + the 1,498-file permission-bit realignment + the walk re-trued (+55) | worklog R343 |

## 2. Previous-work verification matrix (§3.2)

| Task | Previously reported | Current source status | Test evidence | Actual status |
| --- | --- | --- | --- | --- |
| Backend implementation (FastAPI sidecar) | complete | 27 route modules, 114 path operations, factory + lifecycle + middleware present | `tests/api` full suite exit 0 (r342: 461+1 skip of 462, 117.495 s); quick chunk 108/108 + collection 8,602/472 re-pinned fresh (r343); the backend perf lanes measured through the app's own instruments (r342 — see PERFORMANCE_BASELINES) | VERIFIED COMPLETE |
| Frontend implementation (React/Tauri) | complete | React 19 + TS + Vite + Tailwind + Radix stack per §2 constraints | tsc clean, vitest 126/126 (re-run fresh at r343 — the node_modules rebuilt after the rollback wipe); browser-E2E lanes r337–r343 | VERIFIED COMPLETE (source+test surface; packaged-app legs below) |
| Tauri integration | complete (source) | `src-tauri/` conf + backend_manager.rs | D-4/D-5 unit tests written; cargo compile/test BLOCKED in sandbox | SOURCE IMPLEMENTED — EXECUTION BLOCKED |
| REST API | complete | contract 109 paths / 71 schemas (r324 export, CI-guarded) | contract drift guards + full `tests/api` exit 0 | VERIFIED COMPLETE |
| WebSocket (events route) | token-auth only (D-6 open) | pre-accept Origin/Referer guard (close 4403 untrusted, valid token included); the client lane now terminal-codes 4401/4403 (D-8, r341) | `tests/api/test_ws_origin.py` 12/12 (r339, FAIL-FIRST) + the client D-6 lane 6 vitest legs + the banner surfacing (r341, FAIL-FIRST: 6 red pre-fix) | VERIFIED COMPLETE (backend + client lanes) |
| SQLite | complete | schema + migrations + WAL handling in `persistence/` | persistence suites green in the api lane (isolated-temp-DB discipline) | VERIFIED COMPLETE |
| FileVault | complete | artifact registry + integrity validation | persistence/api lanes | VERIFIED COMPLETE |
| Scientific workspaces (9) | complete | the 20-stage §11.2 chain across the nine workspaces | 8,602-test collection; scientific suites in the ten-chunk gate (r325 pin, 8,555 at ship) | VERIFIED COMPLETE |
| Rust handshake | D-4/D-5 fixed (source) | `validate_handshake` + `redacted_parse_failure` with 11 `#[cfg(test)]` cases | cargo fmt/clippy/test BLOCKED (no Rust toolchain); `scripts/windows/test-rust.ps1` delivered | SOURCE IMPLEMENTED — EXECUTION BLOCKED |
| Security (origin/token/CORS/FS) | D-1..D-3 fixed | trusted-origin policy + scoped CORS + startup classification + WS origin guard | `test_trusted_origins.py` 34 legs (r335 pin), `test_ws_origin.py` 12/12, security suites in CI | VERIFIED COMPLETE |
| Automated tests | 8,555 shipped | live collection 8,602 / 0 errors / 472 files (r343, fresh) | collection + quick chunk re-pinned this round | VERIFIED COMPLETE (the shipped pin 8,555 = the frozen v1.2.240 release; live tree honestly exceeds it by 47 — see §4) |
| Browser E2E | web-console lanes green | agent-browser passes r337–r343 | zero page/console errors, pins + footer geometry (3967 desktop / 7882 mobile, zero 390×844 overflow; the live walk re-trued 1,752/570,433/8,555 at r343) | VERIFIED COMPLETE (console surface; desktop-app E2E rides Windows) |
| Windows installers | v1.2.240 shipped | EXE 558,022,042 B + MSI 710,553,078 B + SHA256SUMS + manifest + SBOM | ninth smoke dispatch 45086526069 green end-to-end (16 steps, r336/r337 verified) | VERIFIED COMPLETE (published artifacts; the E/F clean-machine gates below) |
| Offline operation | local-first | no network deps in the shipped app; the scientific stack offline | CI offline lanes; local-first audit (part13 §8) | VERIFIED COMPLETE |
| Documentation | §21 twelve-deliverable set | docs/QA set now completed by the SS13 additions + the r341–r342 truthing | `test_documentation_consistency.py` **12/12** (the r341 record correction applied here: the r340-era "13/13" was a ledger-sentence overcount, not a lost test — the suite has carried 12 test methods since its r324 creation) | VERIFIED COMPLETE |

## 3. Stale/contradictory documentation reconciliation (§3.3)

- **Version numbers agree**: the nine-carrier lockstep (VERSION, pyproject,
  `pharmaml.__version__`, frontend package.json, tauri.conf.json, windows
  version_info, CHANGELOG lead, FEATURES freeze stamp, web TS carrier) is
  machine-checked green at 1.2.240.
- **Test counts agree with collection**: the live collection is
  **8,602 / 0 errors / 472 files**; the shipped footer pin 8,555 describes
  the frozen v1.2.240 release archive, not the live tree — the two-lane
  convention is documented at `src/lib/toolkit-stats.ts` (SHIPPED_* pins)
  and this audit. No contradiction: the live number is stated here.
- **Workspace counts agree**: nine workspaces; the §11.2 stage chain is 20
  stages (the stage count is a within-workspace property, not a
  workspace-count drift).
- **API route counts agree**: 114 path operations across 27 route modules;
  the exported OpenAPI contract carries 109 unique paths / 71 schemas
  (multiple methods share paths) — CI-guarded against drift.
- **Release documentation reflects the actual artifacts**: v1.2.240's five
  assets verified byte-level in r337 (digests in SHA256SUMS + manifest);
  the byte-frozen ruling (r334) stands — no re-ship without approval.
- **Known-defect statuses accurate**: DEFECT_REGISTER rows D-1..D-8 all
  FIXED with named evidence (D-4/D-5 carry their compile-blocked class
  honestly; D-7 found r340, D-8 found r341 — see the register).
- **Historical vs current reports distinguished**: PART14_PHASE1_AUDIT.md
  and the part14-era checklist sections stand as historical records; this
  part15 set is the current-state layer.

## 4. Findings this audit

1. **D-7 (found + fixed r340)**: the ledger-agreement guard's entry regex
   was ordinal-suffix-blind (`\d+th`), silently skipping the R339 entry
   ("the 131**st** recorded run") and comparing stale R338 against the
   counter. Minimal fix: the `(?:st|nd|rd|th)` suffix alternation; the
   failing guard was its own reproduction; suite 12/12 green post-fix
   (the "13/13" in the r340-era sentence was an overcount — the r341
   record correction).
2. The sandbox reaper now wipes `node_modules` between tool-call windows
   (root web + nested frontend) — a documented environmental class, not a
   repo defect; the per-call restore drill is recorded in the worklog.
3. No other new defects: the reconciliation read PROJECT_STATUS,
  DEFECT_REGISTER, SECURITY_TEST_REPORT, PART14_PHASE1_AUDIT, the worklog
  R340 entry and the counter before any work; every r340 claim re-verified
  true (CI 19/19 at `8a14c06b`, the gates re-run live).
4. **The r342 full-tree lint sweep (found + fixed, no functional bug
   class)**: `ruff check .` beyond the CI scope (`src/pharmaml` was clean
   throughout) surfaced the r340 permission-bit class recurring (15 files,
   realigned to the git index modes) + 7 pre-existing code findings in
   two out-of-scope dev scripts (`packaging/windows/packaging_validation.py`,
   `scripts/build_capability_map.py`) — all fixed minimally with no
   semantic change; py_compile both; ruff clean over the full tree AND
   the CI scope post-fix. The root console's own lint gate caught the
   round's first CJS cut (`no-require-imports`) — re-landed ESM, clean.
5. **The r342 perf-baseline refresh** (roadmap #4): the release-era rows
   re-measured + the new backend lanes (startup/health-latency/idle-RSS/
   graceful-shutdown) measured through the app's own instruments — see
   PERFORMANCE_BASELINES.md; wall-clock drift vs the r331 pins honestly
   load-labeled.

— audited round 342 (the 134th recorded run; trued r341);
the SS13 companion docs: BACKEND_AUDIT, FRONTEND_AUDIT, SECURITY_AUDIT,
TEST_EXECUTION_REPORT, MANUAL_TEST_CHECKLIST (part15 extension),
WINDOWS_VALIDATION_GUIDE, RELEASE_BLOCKERS, IMPROVEMENT_ROADMAP,
PERFORMANCE_BASELINES (the r342 refresh).
