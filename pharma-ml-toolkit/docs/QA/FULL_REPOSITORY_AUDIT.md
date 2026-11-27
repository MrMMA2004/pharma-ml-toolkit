# FULL REPOSITORY AUDIT — MASTER_PROMPT part20 (round 370)

**Inspected revision:** `bd53f7b35092e774aac752d3a564a863f18a1989` (main HEAD at
session open, worktree clean, synced with origin/main); audit work executed on
the dedicated branch `audit/part20-repository-audit` cut from that revision.
**Release under audit:** v1.2.243 — tag `v1.2.243@902dd89b`, published
(EXE 558,600,414 B + MSI 711,110,134 B + RELEASE_MANIFEST 510 B + SBOM
23,634 B + SHA256SUMS 368 B, re-polled live via the authenticated API in r369).

This audit rides on, and re-verifies, the r368 13-phase comprehensive audit
(ROUND368_FINAL_REPORT.md) and the r369 absorption close-out. Part20's mandate
— "inspect the actual current repository state before making any assumptions" —
was honored: every claim below was re-derived live this round or cites the
round that executed it.

## 1. Repository and CI state at session open

| Item | Evidence | State |
| --- | --- | --- |
| Branch/HEAD | `git status`/`git log` | main @ bd53f7b3, clean, = origin/main |
| Branches | `git branch -a` | main + origin/main only (arena + round368 refs absorbed and deleted in r369 per the owner directive; nothing left to absorb) |
| Tags | `git tag -l` | latest v1.2.243 (also v1.2.231..v1.2.243 present) |
| Release | authenticated API poll (r369 ledger) | v1.2.243 published, not draft, assets intact |
| Open PRs | GitHub API (this round) | none open (round368 + PR #15 superseded/absorbed in r368/r369) |
| CI at HEAD bd53f7b3 | check-runs API, 19 checks | **18 green + 1 FAILED: "Chemistry and docking preparation"** (run 46016289515, job 140564287542: `test_mol2_graceful_without_openbabel` assert `'OpenBabel' in 'MOL2 skipped: Open Babel Python bindings are unavailable…'` — 1 failed, 498 passed) → **FIXED this round (D-9)**; the whole CI job's test set re-run locally: **499 passed, exit 0** |
| Push CI | run 46016288919 | success |

## 2. Architecture overview (verified unchanged, per part20 §2)

* **Scientific Python backend** — `pharma-ml-toolkit/src/pharmaml/` (314 files,
  32 route files under `api/routes/` mounted at `/api/v1`; FastAPI +
  WebSocket events lane; the scientific stack: ml/dl/docking/generative/
  pkpd/md/chem/biology/viz/workflow).
* **Desktop application** — `frontend/` (React + TS + Vite, nine workspaces,
  115 src files) + `src-tauri/` (Tauri 2 Rust host, 5 files, backend_manager
  460 lines) with the Python backend frozen by PyInstaller
  (`packaging/windows/pharmaml_backend.spec`, Qt-free) bundled as
  `resources/backend` (NOT an externalBin sidecar).
* **Web console** — repo root `src/` + `prisma/` + Next.js app (the sandbox's
  port-3000 render lane; download panel + zip-size API probe verified this
  round: GET / 200, zip-size exact 28,052,601 B @1.2.243).
* **Persistence** — SQLite + FileVault, backend-owned (persistence family:
  66 passed + 1 skipped this round).
* **Distribution** — Windows EXE (NSIS) + MSI (WiX) via
  `.github/workflows/windows-release.yml` on tag push; install-smoke on
  release publish. The Inno-Setup/`build_windows.ps1` lane is RETIRED
  (banner-marked, guard-pinned 26/26 green).

The part20 §2 architectural constraints hold: no C#/WinUI, no second desktop
architecture, no Tauri replacement, frontend holds no scientific authority
(typed client only), no parallel API, `studio.py` stays retired
(legacy quarantine verified: retirement guards 26/26 this round).

## 3. Audit coverage matrix (part20 §3)

| Area | Files identified | Files reviewed | Tests executed | Issues found | Status |
| --- | --- | --- | --- | --- | --- |
| Python backend | 314 (src/pharmaml) | module-level inventory + route prefixes + focused reads (docking/ligands.py, model_evaluation.py, api/security.py et al.); r368 phase5 line-level scientific audit carried + spot re-verified | mypy 314 clean; ruff canonical clean; chemistry-docking CI lane 499 green post-fix; persistence 66+1s; penetration 67 | D-9 (test pin, fixed) | Reviewed + tested |
| FastAPI | 32 route files, server/events/jobs/persistence/security | r368 phase3 contract reconciliation (runtime OpenAPI == contracts/openapi.json, 142 paths / 149 ops incl. 1 HEAD) re-derived; this round: conformal + AD + workflow-detail consumers re-verified | tests/api family (r368: 753 green; this round: penetration 67 + persistence 66) | D-13/D-14 (P3 documentation) | Reviewed + tested |
| React frontend | 115 src files, 9 workspace pages, 20 service dirs | services/client.ts + startup.ts + StartupGate.tsx full-read (this round); every-workspace wiring re-verified (r368 phase4 + this round's dead-method sweep: zero placeholder buttons, zero TODO/FIXME/alert in production) | vitest 357/357 (41 files); tsc --noEmit 0; production build green 24.13s | D-13 (4 dead service methods, P3 documented) | Reviewed + tested |
| Frontend services | 20 service dirs incl. api/, websocket/ | client.ts (token header, memory-only), startup.ts (handshake/health), ws client (4401/4403 terminal policy) — full read this round (task 20-d) | vitest + the 16-file drift-guard family 202/202 | none new (D-4..D-8 rides earlier rounds) | Reviewed + tested |
| Tauri/Rust | 5 files (backend_manager 460L, commands 20L, lib 58L, security.rs, main.rs) + tauri.conf.json + Cargo.toml + capabilities/default.json | all lifecycle files full-read this round (task 20-d) | runner evidence r367 install-smoke (all 20 steps: handshake/health/guard/shutdown probe over the INSTALLED sidecar) | D-11 orphan-on-host-crash (P2 open), D-12 retry-no-respawn (P2 open), D-10 version skew (fixed) | Reviewed; runtime = runner evidence |
| Persistence | SQLite + FileVault + model bundles | persistence family both files executed; backup-restore round-trip + registry covered (r368 phase8 legs carried) | 66 passed + 1 skipped (documented skip) | none new | Tested |
| Python packaging | pharmaml_backend.spec (ACTIVE), build_windows.ps1 + installer.iss + pharmaml.spec (RETIRED), packaging_validation.py, verify_sidecar.py, backend_entry.py | all inventoried; active/legacy split confirmed by workflow cross-refs + 26/26 retirement guards | Qt-free exclusion guard green (in the 26) | none new | Reviewed + guard-tested |
| Windows EXE | NSIS lane (windows-release.yml steps 16-19) | workflow read line-level this round; r367 build ladder + r367 install-smoke run 45912713199 (20 steps) evidence carried; asset integrity re-polled | runner-verified (not this sandbox — no Windows infra) | D-15 source lag vs main (16 commits/172 files post-tag) | Runner-verified |
| Windows MSI | WiX lane (same workflow) | same as EXE | same install-smoke (silent install + lifecycle + uninstall + upgrade 1.2.242→1.2.243) | D-15 (same) | Runner-verified |
| CI/CD | 5 workflows (ci, security-scan, penetration-test, windows-release, windows-install-smoke) | all read this round; the failing chemistry lane root-caused and fixed | 19 check-runs polled at HEAD: 18 green + the 1 failing lane now fixed | D-9 (fixed) | Reviewed + live-polled |
| Web Console | root src/, prisma/ | render lane verified this round (GET / 200; zip-size API exact; dev.log clean) | render + API probe | none | Verified |

Generated files, lockfiles, binaries, and vendored deps were verified through
their generation/integrity process (openapi.json == runtime; api.d.ts
regenerates byte-identically; zip testzip + sha-256 parity across six homes),
not by meaningless line-by-line review, per part20 §3.

## 4. Backend–frontend integration (part20 §4 — summary; full matrix in BACKEND_FRONTEND_TRACEABILITY.md)

* **149 operations / 142 paths** (148 + the HEAD artifact-download; the
  r368 matrix's "148" undercounts by excluding HEAD — corrected in this
  round's traceability doc; runtime == contracts/openapi.json byte-identical,
  carried from the r368 phase3 byte-level reconciliation).
* Every operation has a frontend consumer reference and a backend test lane
  (r368 matrix), with **10 routes documented API-only** (capabilities x2, AD
  catalog + knn + domain-box, model-eval catalog, the conformal family x3,
  workflow run-detail) — each with its documented reason; the conformal
  family remains the real GUI-coverage gap (registered, recommendation
  carried).
* Dead-code sweep this round: exactly **4 dead frontend service methods**
  (unsupervisedApi.methods, reinforceApi.methods, reinforceApi.evaluate,
  generativeApi.priorView) — D-13, P3, documented (typed-surface parity
  kept; no runtime consumer, no test consumer).
* Zero placeholder buttons / fake data / hardcoded results / missing handlers
  in production frontend (greps: TODO/FIXME/coming soon/alert( — zero
  production hits; the r368 phase4 audit's conclusion re-verified).

## 5. Tauri and backend runtime (part20 §6 — task 20-d full-read; summary)

Verified green: direct-exec spawn (no shell → spaces/non-ASCII safe),
CREATE_NO_WINDOW, backend-side port-0 bind (no check-then-bind race),
in-backend token generation (never argv/env; stdout-only; shape-validated
16..=128 urlsafe; redaction on parse failure — D-4/D-5 fixed lanes), the
single-IPC `backend_endpoint` command, PHARMAML-READY one-line JSON
handshake, 180s bounded startup timeout with kill+wait, RunEvent::Exit
graceful terminate (10s try_wait ladder then force kill), loopback-only CSP
connect-src (no remote origins), `core:default`-only capabilities, memory-only
token in the WebView (never persisted), WS reconnect ladder with 4401/4403
terminal policy closes (D-8 fixed lane).

Registered defects (both P2, OPEN — no cargo/rustc in this sandbox to
compile-verify the fixes; the D-4/D-5 precedent: source-only changes without
compile verification are never marked fixed):

* **D-11 — orphan on host crash**: termination runs only on RunEvent::Exit;
  no Windows Job Object KILL_ON_JOB_CLOSE / Unix PDEATHSIG — a force-killed
  host leaves the sidecar running (port + process leak). Recommendation:
  wrap the child in a Job Object on Windows.
* **D-12 — Retry cannot recover a dead backend**: StartupGate's retry
  re-acquires the stale endpoint over IPC; the Rust child is never respawned
  (no restart command) — recovery requires an app relaunch. Recommendation:
  a respawn-aware retry (restart IPC or respawn-on-EPIPE).

## 6. Windows EXE/MSI (part20 §7 — full report in WINDOWS_EXE_MSI_AUDIT.md)

Active lane confirmed: tag push → windows-release.yml (PyInstaller freeze via
pharmaml_backend.spec → staged into src-tauri/resources/backend → Vite build →
cargo tauri build → NSIS EXE + WiX MSI + manifest + SHA256SUMS + SBOM →
release assets). v1.2.243: built from tag 902dd89b (run 45907478532 SUCCESS),
install-smoked (run 45912713199, 20 steps, both installers + upgrade
1.2.242→1.2.243 + user-data retention), assets intact. **Source-to-binary
consistency (finding B revalidated): main is 16 commits / 172 files past the
release commit** (38 backend, 3 frontend — the r368 F-1/F-3 frontend fixes —
plus this round's test/metadata fixes), i.e. the installers do not carry the
post-tag fixes; the next release cycle must ship them (D-15). Windows build
infrastructure is unavailable in this sandbox (Linux, no cargo) — the fresh
build leg stays BLOCKED-with-reason, never fabricated.

## 7. Persistence, data integrity, model compatibility (part20 §8)

Persistence family: 66 passed + 1 skipped (the documented ephemeral-mode
BLOCKED skip). SQLite backup+restore round-trip, model registry, dataset
metadata, FileVault path safety — the r368 phase8 legs carried; no
destructive migration anywhere; no user data touched by this round's changes
(test/metadata/docs only).

## 8. Scientific backend correctness (part20 §9)

Carried from the r368 phase5 line-level scientific validation (zero
registry task-compat mismatches; DL honesty constraints executed — CNN 2-D
refusal, transformer no-pretrained-claims, autoencoder supervised-lane
refusal; feature-space GAN/DDPM molecular:false pinned; RL registry-separation
test-enforced) + this round's chemistry-docking lane 499 green. The five
r367 DL modules (transformer/cnn/autoencoder/gan/diffusion) verified in-tree
in r369 (sizes recorded in the ledger).

## 9. CI/CD and regression coverage (part20 §10 — full report in TEST_AND_CI_REPORT.md r370 section)

Executed this round: chemistry-docking CI lane 499 passed (post-fix, was
1 failed/498 on remote); vitest 357/357 (41 files); tsc 0; the 16-file
drift-guard family 202/202; retirement+quarantine guards 26/26; penetration
lane 67 (exit 0); root eslint 0; mypy Success 314 files; ruff canonical
(src/pharmaml) All checks passed; persistence 66+1s; frontend production
build green 24.13s; zip testzip + six-home sha parity dcaf51fa…f9317e;
web-console render GET / 200 + zip-size API exact.

## 10. Overall readiness assessment

| Area | State | Note |
| --- | --- | --- |
| Backend | PASS | gates green; science carried from r368 + re-runs |
| Frontend | PASS | vitest/tsc/build green; zero placeholder controls |
| Backend–Frontend integration | PASS | 149 ops reconciled; conformal API-only documented (D-gap registered) |
| Tauri runtime | PASS | lifecycle verified + runner evidence; D-11/D-12 registered (P2, compile-blocked here) |
| EXE | PASS (v1.2.243 as shipped) | runner build+smoke evidence; D-15 staleness registered |
| MSI | PASS (v1.2.243 as shipped) | same |
| Windows 10/11 compatibility | PASS on runner evidence; clean-machine lane NOT_RUN (owner); physical-machine untested — honestly bounded |
| Offline functionality | PASS (runner evidence: WebView2 offline embedded; loopback-only; no remote origins) |
| Signing | BLOCKED — certificate-gated (no fake, never claimed) |

No P0/P1 defects open. Remaining P2s: D-11, D-12, D-15 (all with root
cause + recommendation + the exact blocking resource). P3s: D-13, D-14
(documented/handled). The full per-defect ledger with regression evidence
lives in DEFECT_REGISTER.md (part20 rows) and the r370 worklog entry.
