# GITHUB_BACKUP_MANIFEST.md

No credentials or sensitive values appear in this file (by design, §12.8).

## Repository

- **URL:** https://github.com/MrMMA2004/pharma-ml-toolkit
- **Visibility:** private (default security policy, §12.2)
- **Active branch:** `main`
- **Remote name:** `origin` (clean HTTPS URL; token NEVER embedded)
- **Authentication:** fine-grained PAT, stored at `~/.github_token`
  (chmod 600, outside the repo). Used only at runtime via GIT_ASKPASS by
  `.zscripts/gh_backup.sh`. Not in `.git/config`, URLs, logs, or history.

## Latest verified backup

| Field | Value |
|---|---|
| Backup date/time | 2026-11-04/05 — final close-out of the MASTER_PROMPT audit |
| Milestone / version | **v1.2.107 shipped** (round 189: four engines rebuilt, orphan trio adopted, six GUI cards) |
| Local HEAD == remote HEAD == clone HEAD | `c4034c0d066dab6c4b882b8e861b626bd55cad11` (VERIFIED) |
| Tracked files uploaded | 1,701 (clone `git ls-files`, VERIFIED) |
| Pack size | ~102 MiB (initial) + incremental pushes at every milestone |

## Backed-up components

- Full git history of `/home/z/my-project` (Next.js app + `pharma-ml-toolkit/`
  inside one repo — the entire authorized project tree).
- Python sources, playground.py (30,026 lines, v1.2.106), tests (53 files),
  docs, notebooks, `requirements.txt` (pinned stack), `pyproject.toml`.
- Release artifacts intentionally INCLUDED as milestone recoveries:
  `dist-archive/` v1.2.105 + v1.2.106 zips, `download/` + `download_archive/`
  + `pharma-ml-toolkit/download/` zips (~85 MB total; each ≤ 17 MB — no
  Git LFS needed).
- Prisma schema + tracked `db/custom.db` (24 KB scaffold — see P3 audit).
- `PROJECT_AUDIT_PROGRESS.md` and this manifest.

## Intentionally excluded (with reasons)

- `upload/Command.txt`, `upload/Command.docx`, `upload/MASTER_PROMPT.docx` —
  **contain GitHub PATs**; `.gitignore`-guarded; verified ABSENT from clone.
- `.env*` (DATABASE_URL secret) — gitignored by the scaffold.
- `node_modules/`, `.next/`, venvs, caches, `dev.log`, tool-results —
  rebuildable environment state, gitignored.
- `~/.github_token`, `~/.git_askpass_gh.sh` — credential material, lives
  outside the repo.

## Database backup status

- `db/custom.db` (SQLite, 24 KB) is tracked and pushed — effectively a schema
  scaffold with negligible data. Full database audit (P3) will decide whether
  any real data backup is required; no sensitive production data is present
  so far. Backup class today: source + schema scaffold only.

## Git LFS

- Not used (no tracked file exceeds GitHub limits; largest single file ≈ 17 MB).

## Verification results (§12.7)

1. Remote accessible: YES (API + git).
2. Branch `main` exists on remote: YES.
3. Remote SHA == local SHA == clone SHA: YES (`c4034c0`).
4. All tracked files uploaded: YES (1,701 recoverable from a fresh clone).
5. No essential files accidentally excluded: token files excluded BY DESIGN;
   nothing else missing from the tracked set.
6. Large files accessible: YES (zips present in clone).
7. Test clone performed: YES → `/tmp/backup_verify_clone`, clean checkout.
8. Token-leak scan on clone: CLEAN (`git grep github_pat_` → no matches).
9. Project reconstructability: requirements.txt + pyproject.toml + docs
   pushed; venv rebuild scripted in `scratch/venv_repair.sh`.

## Restore-from-backup verification (2026-11-05 — the 6th rollback)

The backup system was exercised for real: a catastrophic local rollback
(HEAD reset to a day-old commit, venv reverted, credentials and cron wiped)
was fully recovered from this remote.

| Field | Value |
|---|---|
| Local HEAD before recovery | `2eb0ede` (round-187 state — engines lost again locally) |
| Recovery method | `git fetch origin main` (GIT_ASKPASS) → `git merge --ff-only` |
| Restored HEAD | `2f7c5d9` == remote HEAD (VERIFIED, pure fast-forward, zero divergence) |
| Tree | byte-identical to shipped v1.2.107 (engines + cards + docs all back) |
| Post-restore tests | 385 + 250 + 14 green chunks (exit 0); full-suite accounting stands |
| Post-restore push | this recovery-record commit (incremental discipline, §12.5) |

**Conclusion:** off-site history + milestone pushes = full reconstructability,
proven under actual loss, not just test-clone drills.

## Outstanding backup issues

- None. The incremental-push discipline held at every milestone (P1
  checkpoint, each engine, the GUI half, both ship rituals, the finalize);
  the final verification re-ran SHA parity + a fresh test clone + the
  v1.2.107 zip's sha-256 comparison local-vs-clone (equal).
- The token lives only in `~/.github_token` (600) + the docx uploads
  (gitignored); the clone grep is clean.
