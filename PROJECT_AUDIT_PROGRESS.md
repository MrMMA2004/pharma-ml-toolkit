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
| P1 | Resource discovery & inventory | **IN PROGRESS** (~80%) | git/env/token/remote/playground/module facts gathered; upload-file deep inspection pending |
| P2 | Detailed resource inspection | NOT STARTED | 16 upload items, extracted/, repo-clones/, mini-services/ |
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
   numpy 2.1.3 with 8 packages missing. venv repair relaunched
   (`scratch/venv_repair.sh` → `scratch/venv_repair.log`).
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

## Resources not yet inspected

- upload/ contents in depth: Applicability_domain_analysis.py, fak_ml.ipynb
  (1.35 MB), look.ipynb, xxx.ipynb, new folder.zip, کد علی.zip,
  کدهای ماشین لرنینگ.rar, 2 PNGs, vina.exe/autogrid4.exe/autodock4.exe,
  extracted_ali_code/, extracted_newfolder/ — P2.
- repo-clones/, extracted/, mini-services/, scripts/ — P2.
- db/custom.db tables + Prisma schema — P3.
- playground.py card/module wiring — P5.

## Unresolved issues / risks

- venv repair result must be re-verified (numpy 1.26.4, all 8 packages, then a
  pytest sanity run) once `scratch/venv_repair.log` shows VENV-REPAIR-DONE.
- Rollback risk remains for UNCOMMITTED work only — mitigation: commit+push at
  every milestone (§12.5). Remote history is now safe.
- `db/custom.db` is tracked in git (24 KB scaffold) — P3 decides whether that
  is appropriate to keep backing up.

## Exact next steps

1. Verify venv repair completed; run pytest sanity (subset) on the toolkit.
2. P2: inspect the 16 upload items + extracted/ + repo-clones/ one by one;
   write the resource inventory (deliverable A scaffold).
3. P3: audit Prisma schema + db/custom.db contents.
4. P4–P6: build the traceability matrix and gap report.
5. P7: rebuild the four engines (+ torchlens/alibi/Quantus fold-in) with GUI
   cards; P8 full gates; ship v1.2.107; P9 final push + verification; P10 report.
