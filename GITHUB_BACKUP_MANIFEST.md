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

## Restore-from-backup verification (2026-11-06 — the 8th rollback)

Exercised for real a second time, mid-round: the sandbox filesystem
snapped back to the November-3 snapshot (HEAD `2eb0ede`, venv without
rdkit/pharmaml, token/remote/global-config/tmp all wiped) — minutes
AFTER the three stranded commits had been pushed. The remote was the
only thing standing; GitHub held every round since November 3.

| Field | Value |
|---|---|
| Local HEAD before recovery | `2eb0ede` (the November-3 snapshot) |
| Remote HEAD at rollback | `08390e3` (rounds 186-192 + the cron's walk-filter fix) |
| Recovery method | token re-extract → remote re-add → `git fetch` → `git merge --ff-only` |
| upload/ untracked blockers | moved aside, merged, restored — every file md5-identical to the remote blobs |
| Post-restore venv | numpy 1.26.4 · rdkit 2026.09.1 · pharmaml 1.2.111 editable (re-pinned) |
| Post-restore tests | 86 green (fold-in + picklist + csv standardize + docs guards) |
| Post-restore pushes | `77b871a` (the round-193 fold-in, pushed within the hour) → `5435f8d` |

**Conclusion (the owner's own directive, proven by the hour):** "Update
the GitHub repository more frequently" — the fast-forward push minutes
before the rollback was the only reason rounds 186-192 survived. The
push cadence is now: every landed unit is committed AND pushed before
the next unit starts.

## v1.2.112 release-zip homes (2026-11-06)

`pharma-ml-toolkit-v1.2.112.zip` — 19,155,891 bytes, sha-256
`b2bec19c4c11db336cd8a047a465e72a95afd9d2f7a76c2d0ff36b8b4d6c8e13`,
built with the route's own exclusion list (ZIP_EXCLUDE_ARGS, `*.zip`
nesting guard included), placed in all three homes:

- `download/pharma-ml-toolkit-v1.2.112.zip` (repo root)
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.112.zip`
- `upload/pharma-ml-toolkit-v1.2.112.zip`

The download route flipped to `exact: true, source: "release"` with a
strong ETag and the exact content-length; the zip-size route answers
`19155891` with zero spawns.

## v1.2.113 release-zip homes (2026-11-06, run 1/100 of the 2-hour sync mandate)

`pharma-ml-toolkit-v1.2.113.zip` — 19,066,185 bytes, sha-256
`b12719264cf99ae1ae82c411179f545a74ffdc1103c35d996dca1cc9f13a3efb`,
built with the standing builder (`scratch/build_zip_179.py`, the
ZIP_EXCLUDE_ARGS mirror + testzip + leak scan: 912 entries / 875
files / nested zips NONE / real leaks 0), placed in all three homes:

- `download/pharma-ml-toolkit-v1.2.113.zip` (repo root)
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.113.zip`
- `upload/pharma-ml-toolkit-v1.2.113.zip`

**Why the rebuild:** the run's full-suite net caught all eight executed
notebooks still carrying `1.2.110`-era committed outputs — the v1.2.112
zip shipped a time-travel artifact inside the artifact. All eight were
re-executed at 1.2.113 (driver `scratch/reexecute_all_notebooks_64.py`,
201 s) and the twins regenerated; every notebook now prints
`pharmaml 1.2.113`. Plus the session-count pin fix (61 → 63 `_dl_`
variables: round 193's anchor-τ / noise-n XAI eval spins) and the fresh
doctest recount (773 items with the fully repaired torch-inclusive
stack; the 765 baseline under-collected by exactly those eight
torch-dependent items). Total honest count: **4,372 + 773 = 5,145
tests, all green**.

## v1.2.114 release-zip homes (2026-11-06, round 199 — the catalog absorption)

`pharma-ml-toolkit-v1.2.114.zip` — 19,073,364 bytes, sha-256
`ea2cf57d97e14e3392cd7817611c9432b9f2f18e81eb0db30c7a464e0bba5c96`,
built with the standing builder (`scratch/build_zip_179.py`, the
ZIP_EXCLUDE_ARGS mirror + testzip + leak scan: 912 entries / 875
files / nested zips NONE / real leaks 0 / accepted placeholders 12 —
the two flagged scratch junit XMLs were scrubbed by the idempotent
137 recipe before acceptance), placed in all three homes:

- `download/pharma-ml-toolkit-v1.2.114.zip` (repo root)
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.114.zip`
- `upload/pharma-ml-toolkit-v1.2.114.zip`

**Why the rebuild:** the owner's directive — "check the GitHub links in
GitHub_Research_Repository_Catalog.pdf one by one and add the remaining
features to playground.py." All 140 catalog links verified live (report:
`pharma-ml-toolkit/docs/GITHUB_CATALOG_LINKCHECK.md`); the gap analysis
found nine tested backend modules with no playground card, and this
release adds those nine cards (maxsmi augmentation, ECFP bit highlighter,
GuacaMol scoring oracles, pingouin-style inferential statistics, SciPy
interpolation lab, PBPK & FIH chain, OSP model library, RxODE-style
dosing events, apriori association rules) plus ten GUI tests
(`TestCatalogAbsorptionGui`). Honest count: **4,382 + 773 = 5,155
tests**; fresh walk pins **882 files / 333,040 lines** at v1.2.114.
