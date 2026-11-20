# PROJECT_AUDIT_PROGRESS.md

Persistent execution checkpoint for the MASTER_PROMPT audit
(*Comprehensive Project Audit, Database Reconciliation, Playground Integration,
and GitHub Backup*). This file is the session-handoff source of truth — read it
first when resuming, then verify the recorded state is still accurate.

Status labels: NOT STARTED · IN PROGRESS · COMPLETED · PARTIALLY COMPLETED ·
BLOCKED · VERIFICATION REQUIRED

---

## Current session: 2026-11-05 (later) — 7th rollback recovered mid-session; TWO deliverables shipped: the GitHub Link Registry (9e4dbc5) and the Prompt Registry (27b14f3). All ten phases remain COMPLETED.

## Deliverables record — 2026-11-05 (the two registries)

1. **`GitHub_Links_Registry.pdf`** (9e4dbc5, 11 pages): every GitHub link the
   owner gave — 33 user-given repositories + the private backup, 34 URLs
   re-verified HTTP 200 against the GitHub API on 2026-11-05, 13 link
   instructions quoted, HTML cover source in `scratch/github_links_registry/`.
2. **`Prompt_Registry.pdf`** (27b14f3, 11 pages): every prompt the owner gave —
   56 register entries in 8 eras, recurring standing prompts quantified, the
   MASTER_PROMPT structure documented, method notes on paraphrases and the
   transliterated Persian filenames, HTML cover source in
   `scratch/prompt_registry/`.
3. **A 7th rollback hit mid-build** (HEAD → 2eb0ede, token/remote/venv/PDFs
   wiped). Recovered via the standard drill; both deliverables pushed to
   origin as they landed. venv re-pinned after ship (import verification +
   385 green engine tests).

## Recovery record — 2026-11-05 (6th rollback, restored from GitHub)

A 6th rollback hit after the close-out: local HEAD was reset `2f7c5d9` →
`2eb0ede`, `/home/z/.venv` reverted to numpy 2.1.3 (torch/rdkit/shap/meeko/
gemmi/selfies/ruff missing), `~/.github_token` + `~/.git_askpass_gh.sh`
destroyed, and the cron scheduler emptied (guard job 495807 gone).

**Recovery executed and verified this session (evidence):**

1. Token re-extracted from the re-uploaded `upload/Command.txt` →
   `~/.github_token` (93 bytes, chmod 600, outside the repo). API auth
   re-verified: login `MrMMA2004`, HTTP 200, 5000 req/hr.
2. Remote set as `origin` (clean URL); fetch via GIT_ASKPASS — token never
   in config/URLs/ps.
3. `git merge --ff-only origin/main`: pure fast-forward `2eb0ede` →
   `2f7c5d9` (merge-base = old HEAD, zero divergence, no local loss).
   Working tree byte-identical to the shipped v1.2.107 (engines restored:
   hpo 666L / pruning 433L / quantization 441L / harness 498L;
   playground.py 31,105L; clean status, token files still gitignored).
4. venv re-pinned to the contract: numpy 1.26.4 · torch 2.14.1+cpu ·
   rdkit 2026.9.1 · shap 0.54.1 (`--no-deps`, resolver-vs-runtime note in
   requirements.txt) + slicer · meeko 0.8.0 · gemmi 0.7.5 · selfies 2.1.1
   · ruff 0.17.1 · mypy 2.4.0 · pytest 9.0.2 · pharmaml 1.2.107 editable.
   All imports verified in one interpreter pass.
5. Tests re-run green (exit 0): engines+graph+features **385 passed
   +1 skip**; explain/interpret/geometric **250 passed**; round-189 GUI
   cards **14 passed +1 skip**. Full-suite 4,496 accounting stands from
   the close-out (identical tree SHA); no re-ship needed.
6. Web verified: dev server HTTP 200, hero pins render 802 files /
   277,550 lines / 4,483 tests under the v1.2.107 badge; agent-browser
   confirmed navigation and the Code Explorer tab.
7. Cron scheduler re-populated: webDevReview (15 min) + the guard
   minimal-action protocol (30 min) recreated.

**Conclusion:** the GitHub backup fulfilled its purpose — one catastrophic
local loss, fully recovered from origin with SHA-verified parity. The
phase table below is unchanged (all ten phases COMPLETED at the close-out).

## Phase status

| Phase | Name | Status | Notes |
|---|---|---|---|
| P1 | Resource discovery & inventory | **COMPLETED** | every upload item + architecture enumerated with evidence |
| P2 | Detailed resource inspection | **COMPLETED** | inventory table in this file + the audit matrix |
| P3 | Database audit | **COMPLETED** | Prisma/SQLite = empty scaffold (User/Post, 0 rows); data layer is file-based; no migration (§3.4) |
| P4 | Cross-reference & traceability matrix | **COMPLETED** | `scratch/master_prompt_audit.md` — 23 evidence-checked rows |
| P5 | playground.py audit | **COMPLETED** | 113-card map, 13 tabs, module-import cross-check, orphan/zero-reference analysis |
| P6 | Gap analysis | **COMPLETED** | G-1 critical (4 lost engines), G-2 high (orphan trio + card gaps), G-3 deferred |
| P7 | Implementation | **COMPLETED** | 4 engines rebuilt (225 tests) + orphan trio adopted (55 tests, 2 shipped bugs fixed) + 6 GUI cards (14 card tests, pin 37→58) |
| P8 | Testing & verification | **COMPLETED** | 4496 collected = 4483 passed + 13 skipped, 0 failures; ruff/mypy/doctests/gui self-test green; v1.2.107 shipped; web pins synced; E2E agent-browser green |
| P9 | GitHub backup verification | **COMPLETED** | HEAD `c4034c0` == remote == clone; 1,701 files; zip sha-256 parity; zero tokens in clone |
| P10 | Final report (deliverables A–I) | **COMPLETED** | delivered in the session close-out (this file + the audit matrix + the worklog) |

## Verified facts so far (with evidence)

1. **GitHub authentication RESOLVED** (was BLOCKED in the prior session).
   Token extracted from re-uploaded `upload/MASTER_PROMPT.docx`, stored at
   `~/.github_token` (chmod 600, outside the repo). Verified via API:
   login `MrMMA2004`, 5000 req/hr. Token-bearing upload files are now
   `.gitignore`-guarded and confirmed absent from the backup (clone grep clean).
2. **Target repo existed empty & private:** `MrMMA2004/pharma-ml-toolkit`
   (no branches, no commits before our push). Remote set as `origin`, clean URL,
   auth via GIT_ASKPASS helper `.zscripts/gh_backup.sh` (token never in
   config/URLs/logs).
3. **Initial backup COMPLETE & VERIFIED** (2026-11-05): full history pushed
   (1,683 tracked files, ~102 MiB pack). Remote HEAD `17e7609` == local HEAD.
   Test clone to `/tmp/backup_verify_clone`: clean, zero token matches,
   upload/Command* + MASTER_PROMPT.docx correctly excluded.
   Details: `GITHUB_BACKUP_MANIFEST.md`.
4. **A 5th rollback occurred** before this session: HEAD was reset from the
   guard/cron commits back to `2eb0ede`, `scratch/round189_state_after_rollback.md`
   was destroyed (recreated this session), and `/home/z/.venv` reverted to
   numpy 2.1.3 with 8 packages missing. **venv repair COMPLETED and VERIFIED
   this session**: numpy 1.26.4 · torch 2.14.1+cpu · rdkit 2026.9.1 · meeko 0.8.0
   · gemmi 0.7.5 · shap 0.54.1 (+slicer, numba) · selfies 2.1.1 · matplotlib
   3.10.9 · ruff 0.17.1 · mypy 2.4.0 · pytest 9.0.2. Sanity pytest runs
   GREEN (exit 0): explain+interpret+geometric (250 tests) and
   tuner+neural+ml_pipeline+applicability+umap+runboard subsets.
5. **Four engine modules confirmed lost** (never committed): `ml/hpo.py`,
   `ml/pruning.py`, `ml/quantization.py`, `ml/harness.py` + their 305 tests.
   Rebuild specs preserved in `scratch/round189_state_after_rollback.md` §3.
6. **Survivor modules verified in tree** at `2eb0ede`: explain (2285L),
   interpret (1795L), geometric (1424L), gnn (780L), gnnxai (480L),
   autograd (1054L), xai (1929L), tuner (2542L), chem/featurize (174L).
7. **Dev server alive** on :3000 (next-server v16.1.3, HTTP 200).
8. Single git repo at `/home/z/my-project` (Next.js app + `pharma-ml-toolkit/`
   subdir, no nested .git). Branch: `main`. Guard cron job 495807 (30 min,
   minimal-action protocol) still scheduled.

## P2 resource inventory (in progress)

| Resource | Type | Purpose | Status |
|---|---|---|---|
| upload/Command.txt + Command.docx + MASTER_PROMPT.docx | prompt docs | the MASTER_PROMPT itself (token-bearing, gitignored) | INSPECTED (full text → scratch/master_prompt_full.txt, sanitized) |
| upload/Applicability_domain_analysis.py | py script | kNN/leverage ADA on catboost pIC50 | INSPECTED — absorbed as `ml/applicability.py` (module verified) |
| upload/fak_ml.ipynb (1.35 MB) | notebook | 183-cell ML workflow | INSPECTED (structure); absorbed via scratch/ml-codes-extract earlier rounds |
| upload/look.ipynb | notebook | 298-line pandas exploration | INSPECTED (structure) |
| upload/xxx.ipynb | notebook | 3-class molecular classification, 35 cells | INSPECTED (structure) |
| upload/new folder.zip → extracted_newfolder/ | archive | Vina + Vina_contol + ligand generation docking scripts | INSPECTED (22 items) — docking lineage absorbed into chem/docking |
| upload/کد علی.zip → extracted_ali_code/ | archive | pdbqt generation, SMILES→PDB, mismatch check, vina multi | INSPECTED (10 items) — absorbed into chem/docking + converters |
| upload/کدهای ماشین لرنینگ.rar → extracted/ | archive | Classification/Regression/converters/Tanimoto scripts | INSPECTED (12 items) — absorbed into chem/ (converters, descriptors, similarity) |
| upload/vina.exe, autodock4.exe, autogrid4.exe | win binaries | docking executables | inventory-only (binary assets) |
| upload/*.png (2) | screenshots | UI evidence | inventory-only |
| repo-clones/pymol.tgz, pymol2.tgz | source tarballs | PyMOL sources | inventory-only (graphics3d lineage) |
| scripts/*_bridge.py (6) | py bridges | web↔toolkit API bridges | wiring check pending (P5) |

Toolkit module map (verified): ml/ = 41 modules (applicability, attention,
autograd, bayesopt, curation, explain, geometric, gnn, gnnxai, interpret,
runboard, tuner, umap, visiblenet, xai, …); chem/ = 17; viz/ = 11; plus
biology/, pharma/, docking/, generative/. **Missing: ml/hpo.py, ml/pruning.py,
ml/quantization.py, ml/harness.py (lost in rollbacks).**

## Resources not yet inspected

- notebooks' cell-level content vs toolkit absorption (P2 remainder).
- scripts/ bridge wiring vs web routes (P5).
- db/custom.db tables + Prisma schema (P3).
- playground.py card/module wiring (P5).

## Unresolved issues / risks

- Rollback risk remains for UNCOMMITTED work only — mitigation: commit+push at
  every milestone (§12.5). Remote history is now safe.
- `db/custom.db` is tracked in git (24 KB scaffold) — P3 decides whether that
  is appropriate to keep backing up.

## Session close-out (P10)

- **Deliverable A (resource inventory):** the P2 table above + the audit
  matrix's 23 rows; every accessible upload/repo item has an inspection
  status.
- **Deliverable B (database reconciliation):** the DB is an unused scaffold
  (0 rows); no external data was expected to live there — no missing/
  duplicated/conflicting records exist by construction; documented, no
  migration without authorization.
- **Deliverable C (traceability matrix):** `scratch/master_prompt_audit.md`.
- **Deliverable D (coverage gap report):** same file, D section — G-1/G-2
  RESOLVED (rebuilt + adopted + wired); G-3 (torchlens/alibi/Quantus
  fold-ins) DEFERRED with reasons recorded; G-4/5/6 documented.
- **Deliverable E (updated playground.py):** v1.2.107 — six new cards,
  every engine has an execution path; existing behavior preserved
  (4483-passing suite).
- **Deliverable F (verification report):** the chunked full-suite accounting
  (4496 = 4483+13+0), all four ship gates green, the notebook-version catch
  and re-ship, E2E agent-browser (pins/explorer/download/live-training).
- **Deliverable G (coverage summary):** fully covered & verified — all
  absorbed lineages; excluded — PyMOL tarballs (inventory-only), the
  Windows exes (binary assets), unverified — none outstanding.
- **Deliverable H (GitHub backup report):** `GITHUB_BACKUP_MANIFEST.md`
  (final state: HEAD c4034c0, 1,701 files, clone-verified, sha-256 parity).
- **Deliverable I (continuity report):** this file + the worklog entry;
  next session resumes from "all phases CLOSED" — only G-3 and new
  commissions remain.

## Exact next steps

None for the MASTER_PROMPT — closed. Standing opportunities: the G-3
torchlens/alibi/Quantus compact fold-ins (spec recorded), and the guard
cron (495807) keeps the minimal-action protocol.

---

## part13 Phase 1 — state reconciliation (2026-11-21, round 307, the 98th recorded run)

The MASTER_PROMPT continued past the closure recorded above through the
part2–part12 installments (ledger: repo-root worklog.md rounds 108–306,
CHANGELOG.md, docs/part12_audit/). This round executed the part13
Phase-1 mandate (verify the actual repository state; reconcile audit
documents; update outdated status statements; create a single
authoritative project-status document):

- **Authoritative status doc created**: `docs/PROJECT_STATUS.md` (the
  single current-state source; it wins status disagreements; carries
  the part13 §22 labels and the evidence table).
- **Stale statements trued in-place**: the six previously-open items
  (B-4 leverage/Williams, B-5 chirality, the /evaluate GUI lane + its
  job-mode fix, B-6 Tk retirement, B-8 SBOM) marked CLOSED with the
  landing round + evidence across COMPREHENSIVE_PROJECT_AUDIT.md,
  BUG_REPORT.md, SCIENTIFIC_VALIDATION_REPORT.md,
  FEATURE_GAP_AND_GITHUB_REFERENCES.md, SECURITY_AND_PACKAGING_AUDIT.md,
  CAPABILITY_INTEGRATION_MATRIX.md, BACKEND_GUI_INTEGRATION_MATRIX.md.
- **Verified this round**: repo public / CI green at HEAD 5446e89b /
  v1.2.231 release assets intact / carriers 1.2.231 / collection 8,362 /
  census 1,591 files / five-home + offsite zip parity d3673da7.
- **Sandbox rollback recovered** (the second rollback; the r306
  precedent re-executed): PAT re-spliced (93-char, 600-mode,
  API-verified, never printed), origin re-added, reset --hard
  origin/main, ghosts re-swept (6 zips + the nested .github), offsite
  rebuilt (11 zips, testzip 11/11), venv re-repaired foreground.
- **BLOCKED (unchanged)**: clean-machine smoke (C-3/C-5), Stage-5
  PySide6 retirement, installer signing.

Next: part13 Phases 2–7 per `docs/PROJECT_STATUS.md` §6.

## part13 Phase 2 — round 309 (the 100th recorded run: the interactive continuation)

- **Phase 2 executed** (the remaining-release-blockers assessment; base
  19effec8, the r308 close-out): the six items each got their in-tree
  leg — the offline-runtime config audit (tauri.conf
  `webviewInstallMode: offlineInstaller`; the PyInstaller onedir
  sidecar carrying its own CPython; frontendDist compiled in; the
  compiled Rust host; CSP connect-src loopback-only), the
  upgrade/uninstall + user-data design proof (NSIS currentUser
  per-user install; `%LOCALAPPDATA%\PharmaML Studio` data home
  outside the install dir; the 67-test backup/restore suite), the
  PySide6 retirement decision record (stays gated — parity ledger
  65/6/3 + the machine gate both stand; now explicit in
  PROJECT_STATUS §4), and the digital-signing assessment deliverable
  (`pharma-ml-toolkit/docs/security/INSTALLER_SIGNING_ASSESSMENT.md`:
  the exact certificate requirement, RFC 3161 timestamping,
  Actions-secrets custody, signtool verification, provenance
  chaining, the staged conditional lane).
- **Machine/certificate legs honestly stay BLOCKED** — no clean
  Windows machine, no certificate; full evidence table in
  WINDOWS_CLEAN_INSTALL_REPORT.md § "part13 Phase 2".
- **Same-version re-ship** (the r301 three-sha convention:
  2c4de09d → the round's final sha): 929 files / 1,042 entries /
  testzip-OK across five homes + offsite; must-carry (incl. the
  signing assessment + the clean-install report), retired-pair
  negative, version_info quadruple, composition 0, real leaks 0 (6
  documented pattern-literals).
- **Pins re-trued**: census doc-only delta (zero src/tests bytes —
  git-diff verified); tests 8,362/0 fresh collection this round.
- **BLOCKED (unchanged)**: clean-machine smoke (C-3/C-5), Stage-5
  PySide6 retirement, installer signing (certificate-gated, the
  exact requirement now documented).

Next: part13 Phase 3 (frontend quality) per `docs/PROJECT_STATUS.md` §6.

## Round 310 — 2026-11-21 — part13 Phase 3 (frontend quality: the component-test lane), over the fourth sandbox rollback

- **The fourth rollback arrived with the firing** (trace
  1a1fb0a0b935472d): HEAD reverted to the ancient 2eb0ede, the origin
  remote gone, the 600-mode credential file gone, the venv broken —
  but the working tree mostly preserved at the r309/r310 frontier
  (the root worklog/counter/scripts survived; the git objects for
  recent commits wiped). Full recovery per the r306–r308 playbook:
  PAT re-spliced (93-char, 600-mode, API-validated 200), origin
  re-added, fetch, pre-reset check (1,092 shared-path untracked files
  byte-identical, 0 conflicts), reset --hard to `ca1accc9` (the r309
  close-out, fetched), venv repaired foreground (torch 2.14.1+cpu,
  PySide6 6.8.3, pharmaml 1.2.231→232 pins verified).
- **part13 Phase 3 EXECUTED — the component-test lane is real**:
  Vitest 5.1.1 + jsdom 31.0.1 + React Testing Library 16.3.3 +
  user-event 14.6.13 + jest-dom 8.0.0 as pinned devDependencies
  (npm install — package-lock.json in sync for the CI's npm ci;
  esbuild postinstall approved), `vitest.config.ts` (jsdom, the `@`
  alias, setup file), and 7 test files / 48 tests over the REAL
  component tree: the §10 shell navigation + nine-entry registry
  order, the §14.1 startup-gate states (connecting/failed/ready,
  typed stage verbatim, §14.2 controlled restart, the WS degradation
  banner), the §13 event-stream reconnection ENGINE (hello handshake,
  replay cursor + duplicate protection, doubling backoff with cap +
  max-attempt give-up, graceful shutdown vs user close, 4008
  dead-peer liveness, ping probe, listener isolation, malformed
  drops), the UI atoms (Button/Card/Tabs a11y + keyboard roving), the
  §22 chart surface (empty state, chunk loading, hovermode closest
  pin, §22.5 customdata point identity, §23 purge), and the Datasets
  workflow first stage end-to-end (file-selection gating, the real
  File upload, typed ApiError verbatim, JobRunCard progress/stage/
  Cancel, the COMPLETED relay auto-selecting the registered dataset,
  the registry table + row limit + failure banner).
- **CI wired**: the studio-frontend job runs `npm --prefix frontend
  test` after the build (the registered root .github/workflows/ci.yml
  — the ONLY registered path).
- **THE FIND (the ghost)**: the rollback resurrected the pre-r293
  `pharma-ml-toolkit/.github/workflows/windows-release.yml` ghost
  (rust 1.85.0, tauri-cli 2.0.0-beta.17 — the pre-registration era).
  GitHub never registered it and no archive ever carried it (verified:
  the r309 zip has zero .github entries) — caught via the census
  delta and REMOVED. The offsite zip home was wiped by the same
  rollback: history re-seeded (221-228 dist-archive, 229/230 git
  blobs, 231 on-disk) — 12 zips incl. this round's ship.
- **Version bump 1.2.231 → 1.2.232, all eight carriers in lockstep**
  (VERSION, pyproject, pharmaml.__version__, frontend/package.json,
  tauri.conf.json, version_info.txt quadruple+strings,
  SHIPPED_TOOLKIT_VERSION) — the lockstep guard 9/9 green.
- **Gates**: vitest 48/48 fresh, tsc --noEmit clean, production build
  green (25.95s), frontend contract guards 66/66, collection
  8,362/0 fresh (zero Python src/tests bytes — the inheritance
  argument: diff ca1accc9 -- tests src = 0 files).
- **Ship v1.2.232** (terminal build, self-reference-free claims
  inside the archive per the r301 convention): 938 files /
  15,439,877 B, sha-256
  1b6284cea6fc4f7a9c4bb2282cbf3c6ce326594df89c8c0fe30f594a971130fc,
  testzip-OK five homes + offsite (12 zips), must-carry includes the
  lane files, retired-pair negative, version_info quadruple 1.2.232,
  composition 0, real leaks 0; the zip-size API serves 15,439,877
  exact:true.
- **Pins re-trued three-way** (pin = live /api/toolkit/tree walk =
  page render): files 1,593→1,608, lines 553,185→552,641, the NEW
  component-test pin 48 painted on the overview card beside the
  pytest pin; Python tests 8,362 unchanged.
- **BLOCKED (unchanged, honest)**: clean-machine smoke (C-3/C-5),
  Stage-5 PySide6 retirement, installer signing (certificate-gated).
  Playwright E2E + the remaining workspace page lanes + visual
  regression honestly queued for the Phase-3 continuation.

Next: part13 Phase 4 (scientific workflow improvements) per
`docs/PROJECT_STATUS.md` §6 — Active Learning, Chemical Space
Explorer, Conformal Prediction audit, Automated Benchmarking, Report
Builder, Model Cards.

- **The first push's two real CI catches — the gates did their job**:
  the Web console job failed on `@next/next/
  no-assign-module-variable` (the plotly mock factory's `module`
  variable — renamed `plotlyModule`) and the Generative SA job on
  the notebook version-freshness gate (the eight executed notebooks
  still printed 1.2.231 against VERSION 1.2.232). Fixes: the venv's
  pharmaml re-installed EDITABLE against the source tree (1.2.232),
  the eight notebooks re-executed (231 s, 8/8 OK, zero error
  outputs, the 7 no-output twins regenerated), the round-250
  builder-path scrubber applied to the executed outputs (the
  ConvergenceWarning traces carried the live venv prefix — «venv»
  form restored, the zip leak scan back to real leaks 0), the
  freshness gate green locally (72/72 over the failing jobs' test
  files), vitest 48/48 re-green, and the terminal ship rebuilt:
  938 files / 15,439,877 B / sha
  1b6284cea6fc4f7a9c4bb2282cbf3c6ce326594df89c8c0fe30f594a971130fc.
  Census fixed point re-trued three-way after the notebook
  re-execution: files 1,608 (unchanged), lines 553,185→552,641 (the
  re-executed outputs are 1,198 lines leaner than the 1.2.231-era
  ones).
