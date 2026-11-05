# PROJECT_AUDIT_PROGRESS.md

Persistent execution checkpoint for the MASTER_PROMPT audit
(*Comprehensive Project Audit, Database Reconciliation, Playground Integration,
and GitHub Backup*). This file is the session-handoff source of truth — read it
first when resuming, then verify the recorded state is still accurate.

Status labels: NOT STARTED · IN PROGRESS · COMPLETED · PARTIALLY COMPLETED ·
BLOCKED · VERIFICATION REQUIRED

---

## Current session: 2026-11-05 — RECOVERY from the 6th rollback (see Recovery record below); all ten phases remain COMPLETED as closed out 2026-11-04/05

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
