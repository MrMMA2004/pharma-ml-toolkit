# PROJECT_AUDIT_PROGRESS.md

Persistent execution checkpoint for the MASTER_PROMPT audit
(*Comprehensive Project Audit, Database Reconciliation, Playground Integration,
and GitHub Backup*). This file is the session-handoff source of truth — read it
first when resuming, then verify the recorded state is still accurate.

Status labels: NOT STARTED · IN PROGRESS · COMPLETED · PARTIALLY COMPLETED ·
BLOCKED · VERIFICATION REQUIRED

---

## Current re-verification: 2026-11-05 — COMPLETED WITH DOCUMENTED LIMITS

Arena checkout: `/home/user/pharma-ml-toolkit`; active branch:
`arena/c44913ab-pharma-ml-toolkit`; baseline: `2f7c5d9`. The attached
43-page `MASTER_PROMPT.pdf` was extracted and read in full. It remains outside
Git because it contains credential material; no credential from the prompt was
used or copied. GitHub operations use Arena's configured authentication.

Completed in this pass:

- Verified the empty SQLite scaffold read-only (`integrity_check=ok`, User=0,
  Post=0); documented the absent `Post.authorId` foreign key without migrating.
- Confirmed all G-1/G-2 engine files, GUI execution cards, and collected test
  evidence; corrected the stale traceability rows.
- Reproduced and fixed the no-`tkinter` `--probe`/`--version` crash using the
  two existing CLI tests (red: 2 failed; green: 2 passed).
- Added the missing `nbformat` development dependency after a clean collection
  failure; its eight requirements guards now pass.
- Restored `tools/checkpoint.py`'s executable bit; project lint and `mypy src`
  now pass. An optional whole-playground mypy probe is **not** clean (88
  pre-existing errors) and is not represented as the project's mypy gate.
- Removed the tracked `.env` from the index while preserving it locally; added
  a safe `.env.example` template and its ignore-rule exception. The file holds
  only a local SQLite URL, but this aligns actual backup contents with policy.
- Implementation commit `1718302a25efc94aba2c8056e74a599c43e5d6c5` was pushed
  to the active branch and recovered in a fresh clone: local == remote == clone,
  1,701 files, zero credential-pattern files, and release-ZIP SHA parity.
  `GITHUB_BACKUP_MANIFEST.md` records the evidence and final limitations.
- Installed the core, QA, chemistry, docking, reporting, boosting, SHAP, and
  PyTorch dependencies in an isolated `/tmp` environment. SHAP is 0.48.0
  because the historical 0.54.1 pin requires Python ≥3.12 and this sandbox has
  Python 3.11.2. Editable installation and wheel/sdist builds succeeded
  (setuptools emitted license-metadata deprecation warnings only).
- Full-suite diagnosis: the initial run collected 4,496 items and produced
  3,840 passes, 22 failures, 276 skips, and 358 setup errors. All 358 errors
  were missing-display fixture launches; the fixture now skips explicitly.
  All 22 failures were traced to environment setup: one needed the documented
  editable install (now passing), and 21 need RDKit drawing's absent native
  `libXrender.so.1`. The final run used exact deselections and retained every
  other collected test.
- Final applicable suite: **3,840 passed + 634 skipped + 22 deselected, 0
  failures/errors** in 160.98 s. The 22 deselected items are the 21 previous
  native-drawing failures plus the mixed module-doctest container; its
  independent `signed_feature_correlations` doctest was run separately and
  passed. All 91 non-native-drawing `test_interpret.py` cases pass (8 exact
  native-drawing cases deselected), and 11 focused CLI/dependency/bridge cases
  pass. The 634 skips comprise the former 358 display setup errors plus the
  suite's 276 existing optional/display skips.

Current limitations: system `tkinter`, Xvfb, and `libXrender.so.1` are
unavailable, and the sandbox cannot reach Debian package mirrors. Display GUI
and 21 native-drawing cases therefore remain unverified here; they are not
represented as product passes. G-3 remains the documented, deferred
torchlens/alibi/Quantus fold-in; no speculative integration was started.

### Current verification report

| Check | Procedure | Actual result |
|---|---|---|
| Headless CLI regression | `pytest tests/test_playground.py::TestCommandLine` | RED before fix: 2 failed; GREEN after: 2 passed; `--probe` and `--version` both executed |
| Full applicable Python suite | `python -m pytest` with the 21 native-drawing failures and their mixed module-doctest node deselected by exact ID | **PASS:** 3,840 passed, 634 skipped, 22 deselected, 0 failed/error |
| Interpret regression | `test_interpret.py` minus its 8 native-drawing cases; correlation doctest separately | **PASS:** 91 + 1; computational `with_drawing=False` probe also passed |
| Python lint / types | `ruff check .`; `mypy src` | **PASS:** zero lint findings; 117 source files typed successfully |
| Package build | `python -m build --outdir /tmp/pharmaml-dist` | **PASS:** wheel + sdist; license-metadata deprecation warnings only |
| SQLite audit | SQLite URI `mode=ro`, `PRAGMA query_only=ON`, schema/index/row-count/integrity queries | **PASS:** integrity ok, User/Post empty; no writes or migrations |
| Web dependency installation | `npm install --no-package-lock` | Completed; npm reported 12 moderate + 5 high vulnerabilities |
| Web lint | `npm run lint` | **FAIL (pre-existing):** 12 errors + 1 warning in React hooks/refs; not suppressed |
| Web production build | `npm run build` | **BLOCKED by network:** Turbopack cannot fetch Google Fonts; filesystem-tracing warnings also reported |
| Offline web compile | `next build --webpack` with Next.js's offline font-response hook | **PASS:** all application routes compiled and static pages generated; this does not verify real font downloads |
| Web application types | `tsc` with a temporary config including only `src/` + `next-env.d.ts` | **PASS**; whole-repository `tsc` still reports pre-existing scratch/example/skill errors |
| Dependency security audit | temporary lockfile + `npm audit --json` | **ATTENTION REQUIRED:** 12 moderate + 5 high; direct highs include `prisma` and `sharp`; no broad auto-fix applied |

The normal web build script, whole-repository TypeScript check, web lint,
and native-display suites must not be described as green in this environment.
The project's declared Python gates and all applicable regression tests are
green as itemized above.

## Historical close-out: 2026-11-04/05

The phase table and facts below preserve the prior session's checkpoint. Where
it conflicts with the re-verification above (paths, branch, credentials, test
counts), the current section is authoritative.

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

## P2 resource inventory

| Resource | Type | Purpose | Status |
|---|---|---|---|
| `/tmp/arena-coding-uploads/MASTER_PROMPT.pdf` (current attachment) | 43-page PDF | audit, integration, testing, and backup directive | INSPECTED in full; credential-bearing and intentionally outside Git |
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

- None from the original inventory; the items formerly listed here were closed
  in the prior P2/P3/P5 work and their outcomes are recorded below.
- Current runtime verification gap: display-backed GUI behavior is unverified
  in this sandbox because system Tk/Xvfb are unavailable (not a resource
  inspection gap). CLI and non-display playground tests remain applicable.

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

No mandatory step remains for this targeted re-verification and backup. Resume
from the current section and `GITHUB_BACKUP_MANIFEST.md`, not the historical
`main` checkpoint below. Separately commissioned follow-ups, in priority order:

1. Review and remediate G-12 dependency advisories, then rerun web and bridge
   regression tests; avoid an unreviewed `npm audit fix --force`.
2. Correct G-11's 12 web lint errors with focused component tests.
3. On a host with Tk, Xvfb, and RDKit's native drawing prerequisites, run the
   complete suite without display exclusions and the normal online font build.
4. Consider the deferred G-3 torchlens/alibi/Quantus compact fold-ins only as a
   new feature commission, not as completed coverage.

The historical cron/job identifiers below are not assumed to exist here.

---

## GitHub Links Registry commission — 2026-11-05 (new follow-up)

This section supersedes the earlier G-3 deferral, not the historical evidence
above. The attached registry yielded 34 unique URLs. Every URL received an
individual live metadata and complete recursive-tree check; all external
READMEs were read. Entries 1–30 map to existing local implementations/cards;
entry 34 is the verified private backup. Entries 31–33 were shallow-cloned and
their relevant algorithms/tests inspected before coding.

**Implemented:** torchlens-style `model_activation_trace`, alibi-style numeric
`tabular_anchor`, and Quantus-style `attribution_complexity`,
`faithfulness_estimate`, `max_sensitivity`, plus the combined metric report.
The existing Deep Learning attribution card now has **Model graph**, **Anchor
rule**, and **Quantus metrics** actions. The APIs are offline and compact;
their explicit non-parity limits and the one-by-one register are in
`GITHUB_LINKS_REGISTRY.md`.

**Verification:** the focused run passed 24 tests with 10 display-only skips
across the new 11-test numerical module, existing XAI helpers, two
display-independent handler tests, and the native XAI GUI class. The counted
full applicable suite passed **3,853 tests, skipped 637, and deselected 22**;
the independent correlation doctest passed separately. This is 13 new passing
tests (11 numerical + two headless handlers) and three new display-only skips
over the prior baseline. The deselections are the exact native RDKit-drawing
failures and their mixed doctest container, not XAI tests.

Ruff, `git diff --check`, mypy over 117 package files, and wheel/sdist builds
passed. The headless harness executes graph/anchor/metric callbacks and schema
restoration; native Tk layout and RDKit drawing remain unverified in this
environment. Normal branch push/SHA/recovery checks are the remaining steps
for this commission.
