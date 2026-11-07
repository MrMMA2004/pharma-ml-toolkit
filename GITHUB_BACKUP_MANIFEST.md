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

## v1.2.120 release-zip homes (2026-11-07, round 206 — PharmaML Studio, the PySide6 GUI increment 1)

`pharma-ml-toolkit-v1.2.120.zip` — 19,755,438 bytes, sha-256
`9cf93f9015c7c7be8c1d543f01bab51ce97c5d59c024025328a413395f136429`,
built with the standing builder (`scratch/build_zip_183.py`: 967 entries /
921 files / nested zips NONE / real leaks 0 — the first build leaked one
`/home/z/` path my own `docs/gui/GUI_REDESIGN.md` carried in its test
command; generalized to `python -m pytest` and rebuilt), placed in all
four homes:

- `download/pharma-ml-toolkit-v1.2.120.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.120.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.120.zip`
- `upload/pharma-ml-toolkit-v1.2.120.zip`

**What shipped:** the owner's re-uploaded, expanded directive #16
(`upload/MASTER_PROMPT part2.pdf`, the 44-page four-technology edition —
the 17-page PySide6-only v1 had been absorbed in round 205). Increment 1
of the GUI redesign: **PharmaML Studio** (`studio.py` + `ui/` package) —
PySide6 6.8.3 shell (sidebar nav, QStackedWidget, status bar, toasts,
Ctrl+1..9), PyQtGraph 0.13.7 themed plots, Qt-Material 2.17 primary
theming with QDarkStyleSheet 3.2.3 as the verified alternative and
native as fallback (one provider active at a time; every switch
re-colours the plots through ChartPalette), QSettings persistence, and
the QThreadPool JobWorker riding the v1.2.119 `JobCancelled`
cooperative-stop contract. Dashboard census computed live from the
installed package; Data Explorer reads real CSV/TSV/XLSX through
`read_table`; Model Analysis trains real registry models with real
held-out metrics. 36 new offscreen tests (`tests/ui/`, pytest-qt 4.5.0)
— honest count: **4,551 + 778 = 5,329 tests**; fresh walk pins **937
files / 342,548 lines** at v1.2.120 (`scratch/census.py`, the
toolkit-fs.ts-exact walk). The recovery layer carries the stack:
`scratch/venv_repair.sh` step 6b installs the five pinned GUI packages
under the numpy constraint and re-copies the EGL stub
(`ui/native/libEGL.so.1`, SONAME-correct, committed) into the wheel's
Qt/lib — RUNPATH $ORIGIN makes plain pytest runs survive pytest-qt's
eager binding probe on this EGL-less box. The 13th rollback (repo to
`2eb0ede`, venv rdkit-less, token/remote/webDevReview lost) was
absorbed by the standing drill before the round started.

## v1.2.119 release-zip homes (2026-11-07, round 205 — the heavy-work safety kit)

`pharma-ml-toolkit-v1.2.119.zip` — 19,275,451 bytes, sha-256
`0c78934edd1b1327ec8d6fd0212bb7cec2b7895dae8d52b6b2fb4d0cc3851f44`,
built with the standing builder (`scratch/build_zip_182.py`: 925 entries /
887 files / nested zips NONE / real leaks 0 — the first build leaked one
`/home/z/` path a peer session's notebook re-execution had embedded in
`examples/complete_ml_workflow.ipynb`'s sklearn warning outputs; sanitized
and rebuilt), placed in all four homes:

- `download/pharma-ml-toolkit-v1.2.119.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.119.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.119.zip`
- `upload/pharma-ml-toolkit-v1.2.119.zip`

**Why the rebuild:** the owner's collapse report — *"playground.py
collapses during heavy work"* — diagnosed as the OOM story (dmesg's two
recorded python OOM-kills; the tune path's 7 loky workers for one
n_jobs=4 call; 4 GB / 2 cores with ~1.6 GB available). The fix:
`pharmaml.exceptions.JobCancelled` — the cooperative-stop channel every
checkpoint loop honours when its progress callback answers `False`
(feature factory batches, the recalculation replay, both progressive
table readers, both CV leaderboards) — plus the playground's Stop
button, live memory watchdog (launch RSS + 75% of launch-time
MemAvailable ceiling, 256 MB system floor), pre-flight batch guard
(molecules × columns × 3 copies × 8 bytes vs half the available RAM,
refusing with its own arithmetic) and ML n_jobs memory cap (768 MB per
loky worker, the reduction riding the job label). 26 new tests
(`tests/test_job_safety.py`) — honest count: **4,515 + 778 = 5,293
tests**; fresh walk pins **901 files / 339,748 lines** at v1.2.119
(`scratch/census.py`, the toolkit-fs.ts-exact walk). The owner also
uploaded `upload/MASTER_PROMPT part2.pdf` — directive #16, the complete
PySide6 (Qt for Python) GUI redesign, the next multi-round mandate.

## v1.2.118 release-zip homes (2026-11-07, round 204 — the feature-recalculation roundtrip)

`pharma-ml-toolkit-v1.2.118.zip` — 19,255,469 bytes, sha-256
`5e7e07b0f3377acc3b20f60fbeec285f8d075cc263666427864c85794c12d7e1`,
built with the standing builder (`scratch/build_zip_181.py`: 923 entries /
885 files / nested zips NONE / real leaks 0), placed in all four homes:

- `download/pharma-ml-toolkit-v1.2.118.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.118.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.118.zip`
- `upload/pharma-ml-toolkit-v1.2.118.zip`

**Why the rebuild:** the owner's catalog continuation directive — use
the metadata stored by the feature factory to recalculate the selected
features. `pharmaml.chem.features.recalculate_selected_features`
replays a saved selection on NEW molecules (only the needed families,
the recorded columns in the recorded rank order, stale metadata
failing loudly: fingerprint-size contradictions, families_computed
inconsistencies, vanished features); the playground gained the
*Feature recalculation (from metadata)* card (Chemistry row 13, the
Descriptors & features workflow's 5th step, three adoption paths:
the bundled `examples/data/selection_metadata.json` example, any JSON
file, the factory's in-session selection; the molecule-x-feature value
matrix render; CSV/Excel save). 30 new tests
(`tests/test_feature_recalculation.py`) + 1 new doctest item —
honest count: **4,489 + 778 = 5,267 tests**; fresh walk pins
**897 files / 338,379 lines** at v1.2.118 (all eight notebooks
re-executed at 1.2.118, 204 s).

## v1.2.117 release-zip homes (2026-11-06, round 202 — every card described, complementary cards grouped)

`pharma-ml-toolkit-v1.2.117.zip` — 19,251,057 bytes, sha-256
`6b5c93e4b5d2677f994230e522f7123511f614cea01778944d32393557df1785`,
built with the standing builder (`scratch/build_zip_180.py`: 921 entries /
883 files / nested zips NONE / real leaks 0), placed in all four homes:

- `download/pharma-ml-toolkit-v1.2.117.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.117.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.117.zip`
- `upload/pharma-ml-toolkit-v1.2.117.zip`

**Why the rebuild:** the owner's continuation directive — every card a
short description, complementary cards grouped into processes. All 137
cards now carry subtitles (nine had shipped bare); a declarative
`WORKFLOWS` registry (11 pipelines / 57 steps, addressed by tab title +
card title) chains the complementary cards; the Home tab gained the
*"Complementary cards — the workflows"* map (numbered step chips that
jump straight to each step's card, a missing card a TRUE no-op — the
test caught the first draft teleporting to the tab anyway); every
participating card wears a `◇ pipeline · step i/n` badge
(`_WORKFLOW_BADGES`, `_CARD_INDEX` untouched as a 3-tuple); the Ctrl+F
finder's haystack now matches workflow names. 17 new tests
(`tests/test_playground_workflows.py`), the DearPyGui registry pin
gained `home-workflows`. Honest count: **4,459 + 777 = 5,236 tests**;
fresh walk pins **893 files / 337,231 lines** at v1.2.117.

## v1.2.116 release-zip homes (2026-11-06, round 201 — Excel everywhere + sample-data-ready tabs)

`pharma-ml-toolkit-v1.2.116.zip` — 19,233,206 bytes, sha-256
`cc4eabb50c72d5cc57fb244c4d20385dc87cdc209919628901fe53f27efaa6d8`,
built with the standing builder (`scratch/build_zip_180.py`: 920 entries /
882 files / nested zips NONE / real leaks 0), placed in all four homes:

- `download/pharma-ml-toolkit-v1.2.116.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.116.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.116.zip`
- `upload/pharma-ml-toolkit-v1.2.116.zip`

**Why the rebuild:** the owner's continuation directive — every
playground tab sample-data ready, Excel input AND output beside CSV.
Excel rides all 14 dataset pickers (`DATA_FILE_TYPES`, read through the
Excel-aware `read_table`) and every table card's save dialog
(`export_table`'s sixth format `xlsx`, plus the DTA/UMAP/
canonical-SMILES exporters via `write_table`/`to_excel`). The four
missing tabs got Load-example buttons (Unsupervised, Best-of ML,
Utilities, and the Home "Try it now — sample data" quick-start card).
The round's full 674-test class-aligned GUI battery also caught five
latent stale pins (three home jump chips selected wrong tabs — now
title-based; the finder/session-report/registry pins predated the
100x-174 census table and the round-200 arena cards) — all repaired.
Honest count: **4,442 + 777 = 5,219 tests** (the 773-doctest pin was
itself stale post-arena-merge); fresh census pins **890 files /
336,329 lines** at v1.2.116.

## v1.2.115 release-zip homes (2026-11-06, round 200 — the arena merge)

`pharma-ml-toolkit-v1.2.115.zip` — 19,112,553 bytes, sha-256
`d04238d1b1fde31d1468b6b7f4384f01e39f508fd4475c95cb1b8ce7f4d0ac27`,
built with the standing builder (`scratch/build_zip_179.py`: 917 entries /
880 files / nested zips NONE / real leaks 0 / accepted placeholders 12),
placed in all four homes:

- `download/pharma-ml-toolkit-v1.2.115.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.115.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.115.zip`
- `upload/pharma-ml-toolkit-v1.2.115.zip`

**Why the rebuild:** the owner's "check other branches" directive surfaced
`arena/2697c755-pharma-ml-toolkit` — a parallel catalog-absorption stream
whose single completed commit folded in the catalog items main had never
absorbed as modules (`pharmaml.docking.dockstring`, `pharmaml.ml.dta`
(DeepDTA), `pharmaml.pkpd.pregnancy`), a 140-repository disposition audit
(`docs/GITHUB_RESEARCH_CATALOG_AUDIT.md`), three playground cards and 34
tests. Merged with main's round-199 nine cards (no overlap: 128 cards
total), all 8 notebooks re-executed at 1.2.115, honest count
**4,416 + 773 = 5,189 tests**; fresh walk pins **886 files / 335,487
lines**. The second arena branch (`arena/c44913ab`) was verified
byte-absorbed into main (stale, no merge needed).
