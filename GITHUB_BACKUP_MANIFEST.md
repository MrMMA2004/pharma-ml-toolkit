# GITHUB_BACKUP_MANIFEST.md

No credentials or sensitive values appear in this file.

## Repository

- **URL:** https://github.com/MrMMA2004/pharma-ml-toolkit
- **Visibility:** private (verified through GitHub on 2026-11-05)
- **Default branch:** `main` (unchanged)
- **Active Arena branch:** `arena/c44913ab-pharma-ml-toolkit`
- **Remote:** `origin`, clean HTTPS URL without embedded credentials
- **Authentication:** Arena's configured `GH_TOKEN` session; no prompt-supplied
  credential was used, copied, or stored

## Latest verified implementation backup

| Field | Value |
|---|---|
| Backup date/time | 2026-11-05 09:25:02 UTC |
| Milestone | v1.2.107 source + headless CLI, QA dependency, display-fixture, lint, and audit fixes |
| Local implementation HEAD | `1718302a25efc94aba2c8056e74a599c43e5d6c5` |
| Remote implementation HEAD | `1718302a25efc94aba2c8056e74a599c43e5d6c5` |
| Fresh-clone HEAD | `1718302a25efc94aba2c8056e74a599c43e5d6c5` |
| Tracked files, original and clone | 1,701 / 1,701 |
| Recovery verification | **PASS** — single-branch test clone at `/tmp/pharmaml-backup-verify-1718302` |

The manifest itself is committed as the following documentation-only milestone.
The branch's final remote HEAD is verified after that push and reported in the
session close-out; an embedded manifest cannot contain its own future commit
SHA.

## Backed-up components

- Complete Next.js application, API routes, TypeScript utilities, UI components,
  scripts, Prisma schema, configuration, and dependency specifications.
- `pharma-ml-toolkit/playground.py` (31,111 lines) with the no-`tkinter` CLI fix.
- Python package sources (117 files checked by mypy), 58 test modules, examples,
  executed notebooks, documentation, requirements, and reproducible build files.
- `PROJECT_AUDIT_PROGRESS.md` and `scratch/master_prompt_audit.md`, including
  actual test results, exclusions, unresolved gaps, and resume instructions.
- Historical release archives in `download/`, `download_archive/`, and
  `pharma-ml-toolkit/dist-archive/`; no file requires Git LFS.
- `db/custom.db` (empty SQLite scaffold), `prisma/schema.prisma`, and the safe
  local-only `.env.example` recovery template.

**Release-scope note:** this milestone updates source and audit records, not the
v1.2.107 version number or release ZIPs. Existing ZIPs are preserved byte for
byte and do not contain the new post-v1.2.107 source fixes.

## Intentionally excluded

- `.env` and other `.env*` files except `.env.example`: runtime configuration.
  The former tracked `.env` contained only a local SQLite URL; it remains on
  disk but was removed from the Git index. The recovery template is included.
- Current attachment `/tmp/arena-coding-uploads/MASTER_PROMPT.pdf`: inspected in
  full but contains credential material and is intentionally outside Git.
- `upload/Command.txt`, `upload/Command.docx`, and `upload/MASTER_PROMPT.docx`:
  credential-bearing prompt files guarded by `.gitignore`.
- `node_modules/`, `.next/`, Python virtualenvs, caches, build output, test
  output, logs, and editor state: rebuildable or machine-local artifacts.
- Credential-manager files and tokens: never included in the repository, clone,
  logs, remote URL, or this manifest.

## Database backup status

- **Class:** source + schema scaffold only, not a production data backup.
- SQLite 3.40.1 read-only inspection: `integrity_check=ok`; `User` and `Post`
  each contain zero rows; the unique User email index exists; no views/triggers
  or foreign keys exist. `Post.authorId` is therefore only a scaffold column.
- The 24 KB `db/custom.db` and Prisma schema are tracked and recoverable. No
  migration, record overwrite, or destructive operation was performed.
- No sensitive production database records were identified or uploaded.

## Git LFS and archive integrity

- Git LFS is not used; the largest tracked files are approximately 17 MB.
- Local and cloned `pharma-ml-toolkit-v1.2.107.zip` SHA-256 values match:
  `152a51a533f54b6445e7dd3ed64fe93dbbefe83dec83d044f77db3bca2e71759`.

## Verification results

1. Remote repository accessible and private: **YES**.
2. Active branch exists remotely: **YES**.
3. Local == remote == fresh-clone implementation SHA: **YES** (`1718302`).
4. Complete tracked-file set recovered: **YES** (1,701 files).
5. Updated `playground.py` and `nbformat` declaration present in clone: **YES**.
6. `.env` absent and `.env.example` present in clone: **YES**.
7. Credential-pattern scan across the clone: **0 matching files**.
8. Release archive local/clone byte comparison and SHA-256 parity: **PASS**.
9. Applicable Python regression suite: **3,840 passed, 634 skipped, 22
   deselected, 0 failures/errors**. Exact limits are in the progress report.
10. Python lint, package mypy, CLI probes, editable install, wheel/sdist build,
    offline webpack compilation, and source-only TypeScript check: **PASS**.

## Outstanding issues and limits

- Display-backed GUI and 21 native RDKit drawing behaviors are unverified in
  this sandbox because Tk, Xvfb, and `libXrender.so.1` are unavailable.
- Normal Turbopack `npm run build` cannot reach Google Fonts here. Offline
  webpack compilation passed; real font retrieval was not verified.
- G-3 torchlens/alibi/Quantus fold-ins remain deferred and documented.
- G-11 web lint (12 errors + 1 warning) and G-12 dependency audit (12 moderate
  + 5 high findings) remain open for targeted follow-up; no rules were disabled
  and no broad force-upgrade was applied.
- The normal whole-repository TypeScript check includes pre-existing scratch,
  example, and skill errors; the application `src/` check passes.
