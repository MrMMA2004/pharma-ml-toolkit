# PROJECT_AUDIT_PROGRESS.md

Persistent execution checkpoint for the MASTER_PROMPT audit
(*Comprehensive Project Audit, Database Reconciliation, Playground Integration,
and GitHub Backup*). This file is the session-handoff source of truth — read it
first when resuming, then verify the recorded state is still accurate.

Status labels: NOT STARTED · IN PROGRESS · COMPLETED · PARTIALLY COMPLETED ·
BLOCKED · VERIFICATION REQUIRED

---

## Current session: 2026-11-05 (Asia/Tehran)

## Phase status

| Phase | Name | Status | Notes |
|---|---|---|---|
| P1 | Resource discovery & inventory | **COMPLETED** | all top-level resources enumerated & verified (see Verified facts) |
| P2 | Detailed resource inspection | **IN PROGRESS** (~70%) | upload items inspected at inventory level; notebooks vs toolkit absorption cross-check pending |
| P3 | Database audit | NOT STARTED | Prisma schema = scaffold default (User/Post); `db/custom.db` 24 KB basically unused — to verify |
| P4 | Cross-reference & traceability matrix | NOT STARTED | |
| P5 | playground.py audit | NOT STARTED | 30,026 lines, v1.2.106, 53 test files |
| P6 | Gap analysis | NOT STARTED | 4 lost engines confirmed (hpo/pruning/quantization/harness) |
| P7 | Implementation | NOT STARTED | rebuild specs preserved in `scratch/round189_state_after_rollback.md` §3 |
| P8 | Testing & verification | NOT STARTED | gates: doctests / ruff 0.17.1 / mypy 2.4.0 / pytest suite |
| P9 | GitHub backup verification | **PARTIALLY COMPLETED** | initial full-history push DONE & verified; must re-push final state at end |
| P10 | Final report (deliverables A–I) | NOT STARTED | |

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

## Exact next steps

1. P3: audit Prisma schema + db/custom.db contents (read-only).
2. P5: map playground.py tab/card structure; verify every ml/chem/viz module
   has an execution path; check scripts/ bridges vs web routes.
3. P4: build the traceability matrix from the module map + card map.
4. P6: gap report — four lost engines + torchlens/alibi/Quantus fold-in + GUI
   coverage gaps found in P5.
5. P7: rebuild engines + GUI cards; P8 full gates; ship v1.2.107;
   P9 final push + verification; P10 report.
