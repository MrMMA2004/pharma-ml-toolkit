
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

## v1.2.121 release-zip homes (2026-11-07, round 207 — Chemistry Workbench, the Studio's increment 2)

`pharma-ml-toolkit-v1.2.121.zip` — 19,764,621 bytes, sha-256
`0986dc2c4078b5768b0cb6156dd4062af962252dd98c26aa7ad37b43e9bb12d8`,
built with the standing builder (`scratch/build_zip_183.py`: 969 entries /
923 files / real leaks 0), placed in all four homes:

- `download/pharma-ml-toolkit-v1.2.121.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.121.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.121.zip`
- `upload/pharma-ml-toolkit-v1.2.121.zip`

**What shipped:** directive #16's 16.18 migration step 2 — the Studio's
Chemistry Workbench (`ui/pages/chemistry.py`): the playground's own
`standardize_csv` batch front door + `compute_feature_frame` multi-family
factory with the v1.2.119 batch guard carried over (five-molecule probe →
exact column count → dense-matrix estimate vs half of MemAvailable, refuse
with arithmetic before the heavy compute), the factory's
`(done, total, note)` progress convention adapted to the Studio's
`(fraction, label)` with the cooperative-stop channel propagating, and
13 new offscreen tests. Honest count **4,564 + 778 = 5,342 tests**; fresh
walk pins **939 files / 343,253 lines**.

## v1.2.124 release-zip homes (2026-11-07, round 210 — the Biology Workbench, the Studio's increment 4)

`pharma-ml-toolkit-v1.2.124.zip` — 19,694,253 bytes, sha-256
`ca6b5a0b9e45733b2f25abbdf51a9f608f6507d54b3c2882d8a4de35e16144e7`
(the first build at 19,692,660 / `64e81fcd…` is SUPERSEDED — its
builder had a home-path bug, `ROOT.parents[1]` resolving to `/home/z`,
that shipped the zip into two directories OUTSIDE the repo and carried
the broken recipe inside the archive; the fixed builder rebuilt the
same round), built with the standing builder
(`scratch/build_zip_210.py` — the FOUR-HOME edition, COMMITTED this
round after the 13th rollback ate the untracked round-209 builders:
975 entries / 930 files / real leaks 0; the scrub-before-build
discipline caught the re-executed ML notebook's fresh
ConvergenceWarning traceback — the version-bump re-execution
re-leaks it every round), placed in all four homes:

- `download/pharma-ml-toolkit-v1.2.124.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.124.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.124.zip`
- `upload/pharma-ml-toolkit-v1.2.124.zip`

The immediate predecessor (v1.2.123) left all four homes — tracked
removals in dist-archive/ + upload/ + the legacy download/, the
untracked repo download/ copy unlinked; deeper history stays where it
is (v1.2.106's only surviving copy lives in the repo download/ sweep
archive and survived this round on purpose).

**What shipped:** directive #16's 16.18 migration step 4 — the
Studio's Biology Workbench (`ui/pages/biology.py`, audit row 4, the
playground Cell Lines tab): four engine tabs wired to the same public
backend APIs the cards call — the screen audit
(`load_response_table` → `audit_response_table` →
`replicate_variability` → `collapse_replicates` →
`summarize_compounds` → `response_matrix`, one job: the bundled
185-row GDSC-style screen loads, audits its missing metric values and
the MCF-7/crizotinib duplicate pair, collapses the replicates and
pivots the 23×8 matrix the other tabs share), the sensitivity matrix
(`response_matrix` + `call_sensitive_resistant` quartile calls,
lower LN_IC50 = more sensitive), the MAD gene ranking
(`most_variable_genes` with the per-gene MAD recomputed honestly),
and biomarker discovery (`normalize_expression` →
`impute_expression` → `sensitivity_associations` BH-corrected →
`top_biomarkers` + `direction_of_association`). The sidebar grows to
seven sections; cross-tab state is written only from the GUI thread
(the round-208 lost-update lesson applied from birth). All 8
notebooks re-executed at 1.2.124 (201 s, zero error outputs, 7 twins
regenerated); 13 new offscreen tests, every asserted number
engine-printed first. Honest totals: collect-only **4,599** + 778
doctests = **5,377**; web pins 5,364 → 5,377 (seven sites) + census
946/344,494 → **950 / 345,766** (`scratch/census_210.py`, committed,
the toolkit-fs.ts-exact walk, run on the post-ship tree). The 13th
rollback was absorbed first
(local HEAD at `2eb0ede`/round-186-era, counter recovered by the
ff-pull — the counter-rides-commits design proving itself again;
venv rdkit-less → the foreground repair; webDevReview job recreated
as 511610).

## v1.2.140 release-zip homes (2026-11-08, round 227 — the Utilities Workbench joins the Studio + the JobRunner GC-race fix, the owner's "everything must be in studio.py" directive increment 2)

`pharma-ml-toolkit-v1.2.140.zip` — 20,387,313 bytes, sha-256
`a911956dbd34ecedea2b30192957283cdfce724da58a63f757c2f44532228694`,
1015 entries / 970 files, built with the standing four-home builder
(`scratch/build_zip_210.py`, the committed version-agnostic recipe)
— testzip OK per home, **real leaks 0**, nested zips NONE, predecessor
v1.2.139 swept from all four homes; placed byte-identical in
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`.

**What shipped:** the retirement backlog's missing tab.  (1) **The
Utilities Workbench** (`ui/pages/utilities.py`, increment 13) — the
legacy Tk playground's Utilities tab ported as five Studio tabs: table
merge & split (`merge_tables`/`merge_table_folder`/
`split_table_into_chunks`, the round-201 example loader riding along),
compare & annotate (`compare_tables`/`annotate_with_smiles` with the
reconciliation counts drawn as NATIVE PyQtGraph bars per §16.8 — no
matplotlib anywhere on the page), the approved-drugs report
(`filter_approved_drugs` + the CSV and bordered-Word `export_docx`
arms), the campaign file tools (`collect_files`/`list_file_names`/
`distribute_files`/`delete_named_files`/`cleanup_tree`, dry-run by
default, one confirmation before any real destructive run, the shipped
14-ligand example tree loadable in one click), and reindex by number
(`reindex_by_number` with the dense-gap honesty).  Honest pre-flight
refusals everywhere — nothing picks a dialog mid-job; the split parts
folder is picked BEFORE the run.  18 new page tests
(`tests/ui/test_utilities_page.py`), every asserted number the real
engine's.  (2) **The JobRunner GC-race engine fix**
(`ui/workers/job_worker.py`): the runner's finisher dropping the last
worker reference mid-emission let CPython's immediate destruction
tear down the worker's JobSignals BEFORE the page slots connected
after it delivered — the Utilities campaign jobs (1.35 s docking
import) hit it deterministically (report stuck on "Running …" while
busy already flipped false); the retired-workers hold keeps signals
alive past the emission, plus the forced-gc regression test.
Notebooks re-executed 8/8 at `pharmaml 1.2.140` (198 s), no-output
twins regenerated; census 990 files / 367,201 lines; pins
5,725 = 4,930 collect + 795 doctest; `docs/gui/GUI_AUDIT.md`'s
Part-2 assessment advances to ~93% (twelve of thirteen surfaces; the
Utilities blocker resolved, the remaining blocker is the Model-Analysis
depth alone); the section-count tests trued to the 16-page window.

## v1.2.139 release-zip homes (2026-11-08, round 226 — the §19 CV interface joins the Studio, the owner's "everything must be in studio.py" directive)

`pharma-ml-toolkit-v1.2.139.zip` — 20,367,978 bytes, sha-256
`580f30e1328c73e2f7f22ac4c9a69a0ff158ac3e6cf1c8bdea124413709eeb6e`,
1013 entries / 968 files, built with the standing four-home builder
(`scratch/build_zip_210.py`, the committed version-agnostic recipe)
— testzip OK per home, **real leaks 0**, nested zips NONE, predecessor
v1.2.138 swept from all four homes; placed byte-identical in
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`.

**What shipped:** the owner's two-part directive answered.  (1) **The
Part-2 verdict** — `docs/gui/GUI_AUDIT.md` now carries the honest §16
completion assessment: architecture, theming, visualization, workers
and eleven of thirteen content surfaces verified, §16.22's "all
relevant features accessible" not yet literally true (the remaining
Model-Analysis depth + the Utilities workflow cards), so §16.18 step
15 correctly refuses to retire `playground.py` today — its module
docstring now states the legacy status.  (2) **The §19 studio port** —
`ui/pages/model_analysis.py` grows the cross-validation interface
round 225 gave the playground only: the **Compare models (CV)** board
(the whole available registry under ONE shared splitter via
`compare_models_cv`), the cv-folds spinner (2–10, default 5) + the
cv-strategy combo (auto/kfold/stratified/group/stratified_group/
scaffold), the honest pre-flight refusals (scaffold CV without a
SMILES column, grouped strategies without a group-like column —
refused BEFORE any job), the formatted `mean ± std (k-fold CV)`
leaderboard, and the double-click **CV report dialog** (fold table
with train-side mirrors, per-metric aggregates with honest
directions, notes, and the PyQtGraph fold chart with mean line + ± 1
std band edges — no matplotlib backend roulette).  The group-source
detectors are studio-owned (`detect_smiles_column` /
`detect_group_column`, the playground's honesty contracts) so the
page stays independent of the legacy file.  7 new page tests
(`tests/ui/test_model_analysis_page.py`); the suite driver chunk 4c
extended.  18th sandbox rollback recovered pre-round (PAT via
newline-join from MASTER_PROMPT.pdf, 19 upload blockers byte-identical,
ff-pull to 086902c, venv repaired).  Suite: the nine-chunk runner
**all green** (chunk 4c carried the new tests).  Notebooks re-executed
8/8 at 1.2.139 (**196 s**, 7 twins regenerated).  Honest totals:
collect-only **4,911 + 795 doctests = 5,706 tests**; census **988
files / 365,533 lines**; web pins 5,699 → 5,706 (download-panel +
overview + page FALLBACK_STATS 987/363,303 → 988/365,533) +
SHIPPED_TOOLKIT_VERSION 1.2.139 across its four homes.

## v1.2.138 release-zip homes (2026-11-08, round 225 — the Part-3 §19 cross-validation results layer)

`pharma-ml-toolkit-v1.2.138.zip` — 20,349,987 bytes, sha-256
`ce3fd87cbdb21b036e432b801222e15abc2c0ab4ecc89b17f09a9e83de057e76`,
1012 entries / 967 files, built with the standing four-home builder
(`scratch/build_zip_210.py`, the committed version-agnostic recipe)
— testzip OK per home, **real leaks 0**, nested zips NONE, predecessor
v1.2.137 swept from all four homes; placed byte-identical in
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`.

**What shipped:** the Part-3 §19 directive — cross-validation promoted
from a hidden training detail to a first-class, reportable result.
The engine (`src/pharmaml/ml/cv_results.py`): the §19.3 splitter core
(kfold / stratified / group / stratified_group / **scaffold** —
Bemis–Murcko grouping over `splits.scaffold_groups`, honest guards for
labels / groups / scaffold counts, limitations notes travelling with
every result); the explicit fold loop **cloning the whole pipeline per
fold** so preprocessing refits per fold (§19.7's leakage rule) and deep
models get a fresh module + optimizer + scheduler every fold (§19.6,
proven by a counted-objects test on the torch `dnn`); the §19.2
bookkeeping (strategy / folds / seed, individual fold scores,
train-side mirrors, per-fold timings, mean/std aggregates, and the
`0.923 ± 0.018 (5-fold CV)` display contract); the honest §19.4/§19.5
metric panels (binary-only rows dropped for multiclass, proba rows
dropped for no-predict_proba estimators, balanced_accuracy primary
fallback — every drop leaving a note); §19.9 persistence
(`dataset_fingerprint` + feature snapshot + schema-checked JSON
round-trip); `compare_models_cv` sharing ONE splitter across the panel
with the cooperative JobCancelled stop channel. The playground §19.8
interface: the cv-strategy combo + configurable folds on the Compare
button, the board's formatted `cv_score` column, and the double-click
CV report window (fold table, aggregate table with honest directions,
notes, fold-by-fold chart with mean ± std band — the matplotlib import
routed through `pharmaml.viz.get_pyplot` because a bare pyplot import
let backend resolution pick PySide6/Qt which **aborts the process**
under a cursor-less Xvfb, caught by the round-225 probe and pinned by
a regression test). Standing-mandate side quest: the §18 layer's 15
latent mypy errors trued to zero via the `_bounds` contract helper —
**src/pharmaml is now mypy-clean end to end** (191 files, 0 errors).
41 engine tests + 7 GUI tests + doctests; notebooks re-executed 8/8 at
1.2.138 (**223 s**, 7 twins regenerated with the variant-note cell, 4
warning-stderr leaks scrubbed — the standing round-212 rule); the
nine-chunk suite green (chunks 1/2's three reds — the leakage pin and
two twin note-cell pins — went green immediately after the scrub).
Honest totals: collect-only **4,904 + 795 doctests = 5,699 tests**;
census **987 files / 363,303 lines**; web pins 5,649 → 5,699
(download-panel + overview + page FALLBACK_STATS 984/362,491 →
986/363,183) + SHIPPED_TOOLKIT_VERSION 1.2.138 across its four homes.
Scheduler self-heal during the round: master cron 514037 found
"Disabled by API" again → recreated as **514067**; the missing
webDevReview 15-min job recreated as **514068**.

## v1.2.137 release-zip homes (2026-11-08, round 224 — the Part-3 §18.2/§18.3 hyperparameter recommendations layer)

`pharma-ml-toolkit-v1.2.137.zip` — 20,330,910 bytes, sha-256
`d00db97929064108b5b60e22656da9d54c9131cc5d53892146d7199b94576308`,
1009 entries / 964 files, built with the standing four-home builder
(`scratch/build_zip_210.py`, the committed version-agnostic recipe)
— testzip OK per home, **real leaks 0** (the pre-build home-path
scrub caught two stale scratch junit XMLs first, exactly the
round-209 ordering discipline), predecessor v1.2.136 swept from all
four homes.  Byte-identical copies ship to all four homes (repo-root
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, `upload/`).  The increment:
`pharmaml.ml.hpo_presets` — the §18.2/§18.3 layer the search engines
lacked: hand-authored `ParamAdvice` tables for the directive's eight
families (logistic/RF/SVM-SVR/XGBoost/LightGBM/CatBoost/KNN plus the
shipped ann/dnn neural families) with per-parameter ranges, log
priors, essential/key/extended priorities, one-line purposes and
dependency structure; the registry-grid fallback for every other
grid-bearing family; the three deterministic presets
(conservative/balanced/exploratory, step-snapped middle-half
narrowing); `recommend_search`'s dataset/budget adaptation (small-n
caps, imbalance keeps `balanced` weighting searchable —
xgboost/catboost get their honest `scale_pos_weight` /
`auto_class_weights` notes — low resources cap budgets and drop
sub-32 batches, every adaptation leaving a visible note);
`validate_space`/`validate_values` enforcing the solver-penalty
matrix (solver list ordered by compatibility so preset trims stay
valid outside exploratory), kernel/task-conditioned parameters and
range sanity, with the per-trial guard failing incompatible draws
loudly; `hypermodel_from_advice` compiling a recommendation into a
tuner-searchable HyperModel (proven with real RandomSearch/TPESearch
runs over advised spaces); lossless schema-checked JSON round-trip
and `estimate_evaluations` budget answers.  Verification: honest
totals **5,649 = 4,856 collect + 793 doctest** (66 new collected +
8 new doctests), the nine-chunk suite green (chunk 2's only reds
were the two notebook version pins, green again after the re-exec),
notebooks re-executed **199 s (8/8 at `pharmaml 1.2.137`)**, twins
regenerated, ruff clean, census **984 files / 362,491 lines**, zip
endpoint `exact:true`.  The 17th sandbox rollback recovered before
the round (PAT re-extracted from the PDF with newline-join — the
space-join token was a 76-char truncation; 19 upload blockers moved
aside for the ff-pull and all byte-identical; venv foreground
repair; the 0777 mode smear swept again — 383 files back to 0644).

## v1.2.136 release-zip homes (2026-11-08, round 223 — the playground SIGILL root-caused and fixed)

`pharma-ml-toolkit-v1.2.136.zip` — 20,406,407 bytes, sha-256
`2ff1fb348772de53e70428ad7eeaeda9095fbc95d46da4474d1bcf85183ce0a7`,
1008 entries / 962 files, built with the standing four-home builder
(`scratch/build_zip_223.py`, the round-217 recipe verbatim) — testzip
OK per home, **real leaks 0** (fifth consecutive zero-drop round;
twin scrub 0 drops), predecessor v1.2.135 swept from all four homes.
Byte-identical copies ship to all four homes (repo-root `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`,
`upload/`).  The increment: round 222 had called the playground's
`Illegal instruction` crash "environmental on the pristine tree" —
round 223 root-caused it to a **latent worker-thread GC race in
`_start_job` itself** (the automatic GC pass firing on the
`threading.Thread` worker during native-heavy phases: matplotlib
transform/marker construction in the rescore-report path, the `torch`
import inside `neural.fit`; crash position drifts run-to-run, which
is a timing race, not a poisoned binary).  The fix suspends automatic
GC for the bounded job body in `runner()` and drains the deferred
cycles once on the main thread in `_poll_jobs`; the report bundle's
`work()` carries the same guard.  Proof: the minimal repro
(`TestDockingRescoreReport`) went 3/3 crash → **3/3 green**, the full
869-test playground chunk completed for the first time on this
sandbox, and the restructured nine-chunk suite is fully green.  Also:
the stale review-144 source-scan pin trued 3 → 4 (the round-215 DL
file-load job legitimately adopted the streaming reader), and the
OOM-killed ui chunk split across three processes (4a/4b/4c) with
`test_playground.py` exclusively in chunk 5.  Verification: honest
totals **5,575 = 4,790 collect + 785 doctest** (unchanged — the fix
adds behaviour, not tests; the review-144 pin edit re-points an
existing test), notebooks re-executed **200 s (8/8 at `pharmaml
1.2.136`)**, twin scrub 0 drops, ruff clean, mypy 0 on touched files,
census **982 files / 358,388 lines**, zip endpoint `exact:true`.

## v1.2.135 release-zip homes (2026-11-08, round 222 — the Part-3 §17 image-export engine, TIFF/JPEG first-class)

`pharma-ml-toolkit-v1.2.135.zip` — 20,408,163 bytes, sha-256
`d150ff7d5451770b856bb63743650a2858c40c9fe69e04f6c92d302e56e4e136`,
1008 entries / 962 files, built with the standing four-home builder
(`scratch/build_zip_222.py`, the round-217 recipe verbatim) — testzip
OK per home, **real leaks 0** (fifth consecutive zero-drop round),
predecessor v1.2.134 swept from all four homes. Byte-identical copies
ship to all four homes (repo-root `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`,
`upload/`). The increment: the Part-3 directive's §17 —
`src/pharmaml/viz/export.py` (the engine: TIFF as the scientific path
with genuine 16-bit `I;16` preservation, lossless LZW/DEFLATE/none,
multi-page stacks, dpi tags; JPEG as the presentation path with
quality 1-95 and the 16-bit/alpha/float integrity refusals;
validation as part of every export — magic-byte sniff + dimension
check + extension/content honesty; the `ExportManager` batch with a
JSON manifest; matplotlib figures and Qt widget grabs through the
same rules; Pillow optional with the actionable install hint),
`ui/widgets/image_export.py` (the grab bridge),
`ImagePanel.export_image` + the new `PlotPanel.export_image`, and the
Image Lab's §17 export row (format combo with TIFF default, JPEG
quality spin, TIFF compression combo, the extension-stamping rule,
the honest result-card status line). 36 new collected items + 7 new
doctests. Verification: the six-chunk suite green (chunk 1/3 fixed
in-flight — docs INDEX rows for the new viz surface, the Pillow pin
block; chunk 2 green after the notebook re-execution; ui +
job_safety 272 dots exit 0 with the OOM ritual respected;
playground's Tk suite crashes SIGILL on BOTH the round-221 pristine
tree and this tree — environmental, the freshly-repaired venv after
the 16th rollback, non-regression proven by stash-and-rerun),
collect-only 4,790, doctests 785, ruff clean, mypy 0 on the touched
files, notebooks re-executed 197 s (8/8, all print
`pharmaml 1.2.135`) + 7 twins regenerated, census **982 files /
359,737 lines** (census runs LAST), web pins synced (982/359,737,
5,575 tests), four version carriers at 1.2.135. The 16th rollback
recovered before all of it: token re-extracted from the PDF (93
chars), remote re-added, pure ff to `234a34b`, venv repaired
foreground.

## v1.2.134 release-zip homes (2026-11-08, round 221 — the Image Lab, the Part-2 §16.8.2-16.8.4 closure)

`pharma-ml-toolkit-v1.2.134.zip` — 20,370,596 bytes, sha-256
`7e817d3611b1bd8f983f2fc14484630d275141d0696cf29ed0bd62d4a6bc4776`,
1002 entries / 956 files, built with the standing four-home builder
(`scratch/build_zip_221.py`, the round-217 recipe verbatim) — testzip
OK per home, **real leaks 0** (fourth consecutive zero-drop round),
predecessor v1.2.133 swept from all four homes. Byte-identical copies
ship to all four homes (repo-root `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`,
`upload/`). The increment: the Image Lab (Studio increment 12,
sidebar section 15) — `ui/widgets/image_panel.py` (ImageItem +
HistogramLUT colormap control with seven gradient presets + the
shared RectROI with live sub-matrix statistics + PNG export + theme
sync) and `ui/pages/image_lab.py` (five tabs on real project data:
the 12-drug Tanimoto similarity heatmap, the 23x8 LN_IC50 response
matrix with its masked NaN hole, the explorer's z-scored numeric
columns, the fixed-region ROI demo reproducing the probe's
statistics, and the tanimoto timing surface). 18 offscreen tests;
the sidebar growth re-pinned five older tests (15 pages).
Verification: ui suite + job_safety 265 dots exit 0 (the OOM ritual
respected — next-server down for the run), collect-only 4,754,
doctests 775 + 3 (pytest-9 accounting), ruff clean, mypy 0 on the
three touched ui files, notebooks re-executed 204 s (8/8, all print
`pharmaml 1.2.134`) + 7 twins regenerated, scrub dropped 0, census
**976 files / 356,316 lines** (census runs LAST), web pins synced
(976/356,316, 5,532 tests), four version carriers at 1.2.134.

## v1.2.133 release-zip homes (2026-11-08, round 220, cron run 15/100 — the lint-restore round)

`pharma-ml-toolkit-v1.2.133.zip` — 20,348,595 bytes, sha-256
`eac9c60af3f76662cb7d17b7915441587d531539569e65575cdcd4cc219e9f7f`,
998 entries / 952 files, built with the standing four-home builder
(`scratch/build_zip_220.py`, the round-217 recipe verbatim) — testzip
OK per home, **real leaks 0** (third consecutive zero-drop round),
predecessor v1.2.132 swept from all four homes. Byte-identical copies
ship to all four homes (repo-root `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`,
`upload/`). The round: the 15th sandbox rollback recovered first
(local wiped to 2eb0ede/round-186 v1.2.106; remote survived at
27633fc/round-219; pure fast-forward, every `upload/` artefact
re-synced byte-identical), then the standing mandate's engineering
review: the bare-`ruff check .` green convention restored (the
rollback had smeared the upload mount's 777 bits over 153 no-shebang
`.py` files -> 0644; `studio.py` stays the one 0755) and the five
round-205 cooperative-stop guards collapsed (SIM102, identical
short-circuit semantics): `utils/io.py` x2, `ml/training.py` x2,
`chem/features.py` x1. Verification: focused batches 213 + 409 dots,
ui suite + job_safety re-run AFTER the edits 247 dots exit 0,
doctests 775 passed + 3 skipped under the repaired venv's pytest 9.0.2
(the same 778 collected items; the three all-`+SKIP` docstring items
now itemized as skips — honest grand total 5,514 unchanged),
collect-only 4,736, ruff clean, mypy 0 on the three touched files,
notebooks re-executed 204 s (8/8, zero errors, all print
`pharmaml 1.2.133`) + 7 twins regenerated, scrub dropped 0, census
**972 files / 354,599 lines** (census runs LAST), web pins synced
(972/354,599, 5,514 tests), four version carriers at 1.2.133.

## v1.2.132 release-zip homes (2026-11-08, round 219 — the Response AI Workbench, the Studio's increment 11)

`pharma-ml-toolkit-v1.2.132.zip` — 20,288,273 bytes, sha-256
`8ae2c2487be43b96c8619dc08168400536604a0bac9b48389aa70f96907dcf27`,
997 entries / 951 files, built with the standing four-home builder
(`scratch/build_zip_219.py`, the round-217 recipe verbatim) — testzip
OK per home, **real leaks 0** (a zero-drop scrub round: the leak
sources fixed in rounds 216/217 hold), predecessor v1.2.131 swept
from all four homes. Byte-identical copies ship to all four homes
(repo-root `download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, `upload/`). The increment:
`ui/pages/responseai.py` (the Studio's eleventh feature page, audit
row 12) gives `pharmaml.ml.drugresponse` — absorbed from
CaDRReS/Precily/DeepResponse with NO prior GUI surface — its first
one: eight tabs (Hill curve / response metrics / rank correlations /
the CaDRReS response model / cell-line-grouped cross-validation /
informative regression / the ResponseVAE / multi-omics fusion), 17
offscreen tests (`tests/ui/test_responseai_page.py`), every asserted
number engine-printed first by the re-seeded committed probe
(`pharma-ml-toolkit/scratch/probe_response_218.py` — round 219
re-seeded it per section, fresh `default_rng(42)` at every section
head, so each tab is a standalone seeded job; `block_report`'s audit
numbers added). Notebooks re-executed at 1.2.132 (201 s, all eight
print `pharmaml 1.2.132`, zero stale); 7 twins regenerated; the twin
scrub dropped 0. Honest totals: collect-only **4,736** + 778
doctests = **5,514**; web pins 5,497 → 5,514 (page.tsx hero +
FALLBACK_STATS 971/354,525 + download-panel ×3 + overview ×2) +
census 968/352,830 → **971 / 354,525**. The sidebar grows to
**14 sections** (test_main_window + bestofml/foundations wiring pins
re-pinned). No rollback this round — the tree opened clean and synced
at `5a50f90` (round 217's close-out); this is an IM continuation, no
cron fire — the canonical counter stays **14/100**; the webDevReview
15-min job had vanished from the scheduler again and was recreated as
job 512508.

## v1.2.131 release-zip homes (2026-11-07, round 217 — the Best-of ML Workbench, the Studio's increment 10)

`pharma-ml-toolkit-v1.2.131.zip` — 20,078,596 bytes, sha-256
`6243f110c8e719a3fd7ff9561d5b752ad9d6c92b6b9a6403cc3e86bfcc069eac`,
994 entries / 948 files, built with the standing four-home builder
(`scratch/build_zip_217.py`, the round-216 recipe plus the predecessor
sweep and per-home testzip verification); the first build's leak scan
caught the round-216 venv-repair wrapper carrying a hardcoded
`/home/z/` path (the first zip to include it — the scan doing exactly
its job): the wrapper now resolves via `$HOME` and the dynamic
`BASH_SOURCE` walk, verified from a relocated clone, rebuilt clean
with **real leaks 0**. Byte-identical copies ship to all four homes
(repo-root `download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, `upload/`); predecessor v1.2.130 swept
from all four. The 14th rollback was absorbed before the increment
(local wiped to `2eb0ede` again; remote survived at `ba60af0`; PAT
re-extracted from `upload/MASTER_PROMPT.pdf`'s 76+17 line-wrap;
ff-pull clean; venv repaired foreground; the concurrent agent's
cron run 13 recovered in parallel and — seeing this round's
work-in-flight — verified it instead of duplicating it: every
overlapping probe number recomputed identical, 19/19 bestofml tests
green under it, one increment one owner).

`pharma-ml-toolkit-v1.2.130.zip` — 19,862,379 bytes, sha-256
`0d8571ce98ea8b244e16787433754efc317423ca609787265ebedf130f5e28f6`,
built with the standing four-home builder (`scratch/build_zip_216.py`,
the round-180/212/213/214/215 recipe plus the predecessor sweep and
per-home testzip verification); the first build's leak scan caught the
round's own probe docstring carrying an absolute `/home/z/` path —
fixed to the toolkit-root relative form and rebuilt before anything
shipped (the scan doing exactly its job); the re-executed notebooks
scrubbed clean (0 dropped); 988 entries / 942 files / nested zips NONE
/ real leaks 0, placed byte-identical in all four homes:

- `download/pharma-ml-toolkit-v1.2.130.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.130.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.130.zip`
- `upload/pharma-ml-toolkit-v1.2.130.zip`

The immediate predecessor (v1.2.129) was removed from all four homes
(the deletion rides this commit).

**What shipped:** audit row 10 (the playground Foundations tab) — the
Foundations Workbench (`ui/pages/foundations.py`, Studio increment 9):
compare groups (`compare_groups` with the branch reported and both
histograms on a shared axis; the synthetic demo mirrors the card's
rng-42 groups, the columns mode reads the Data Explorer's frame), the
five-distribution sampler (`sample_distribution` with the theory drawn
in the same value space via the new `PlotPanel.bars(x=…)` /
`histogram(range=…)` extension), the inferential tests (`welch_anova`
plus Tukey HSD in the table and Games-Howell in the report), and the
table profiler (`profile_dataset` with the type census and alert
rules). Sidebar 11 → 12 sections; 13 offscreen tests with the
committed probe; notebooks re-executed (205 s, all eight print
1.2.130); honest totals 4,700 + 778 = **5,478**; census 962 files /
350,687 lines; web pins 5,478 + census.

## v1.2.129 release-zip homes (2026-11-07, round 215 — the file-load jobs stream too)

`pharma-ml-toolkit-v1.2.129.zip` — 19,848,910 bytes, sha-256
`3f7da583f7716040facd1817289751d4df211571f28c711b88156ef021220d51`,
built with the standing four-home builder (`scratch/build_zip_215.py`,
the round-180/212/213/214 recipe plus the predecessor sweep and
per-home testzip verification); the re-executed notebooks scrubbed
clean on the first pass this round (0 dropped — the fifth round of the
leak-streak watch, 210 / 212 / 213 / 214 / 215); 985 entries / 939
files / nested zips NONE / real leaks 0, placed byte-identical in all
four homes:

- `download/pharma-ml-toolkit-v1.2.129.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.129.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.129.zip`
- `upload/pharma-ml-toolkit-v1.2.129.zip`

The immediate predecessor (v1.2.128) was removed from all four homes
(the deletion rides this commit).

**What shipped:** the round-215 extension of the heavy-work safety kit
— the three user-file loads that still called the plain `read_table`
inside a ZERO-argument `work` (the pIC50 append, the library
similarity-scan pick, the docking rescore summary CSV) now stream
through `read_table_with_progress` with the job reporter wired: a
determinate rows bar for a multi-hundred-megabyte table, a Stop the
watchdog can flip, and the cooperative stop honoured at every chunk
boundary — the "huge CSV load" story the round-205 header names, now
closed. Small files keep the plain reader byte-for-byte. Three new
regression tests in `tests/test_job_safety.py` (the real jobs started
against temp CSVs, each pinned cancellable mid-flight with its render
landing). Honest totals 4,687 + 778 = **5,465**; census 959 files /
349,557 lines; web pins 5,465 + census; version carriers bumped to
1.2.129 in all four places; notebooks re-executed (202 s, all eight
print 1.2.129) + the 7 twins regenerated.

## v1.2.128 release-zip homes (2026-11-07, round 214 — the Generative Workbench, the Studio's increment 8)

`pharma-ml-toolkit-v1.2.128.zip` — 19,845,385 bytes, sha-256
`93cb100b69359ba734950fd30f6f8f003c819f2ba87279c0c4929942fae4a7a6`
(the canonical final build; per the round-213 lesson the BYTE SIZE is
the rebuild-stable pin — the endpoint serves it exactly — while zip's
directory walk order varies the digest between otherwise-identical
builds), built with the standing four-home builder
(`scratch/build_zip_214.py`, the round-180/212/213 recipe plus the
predecessor sweep and per-home testzip verification);
scrub-before-build caught the re-executed ML notebook's warning stderr
AGAIN (the fourth-round streak: 210 / 212 / 213 / 214 — four sklearn
ConvergenceWarning/FutureWarning stderr streams carrying the venv path
dropped whole by `scripts/scrub_twins_214.py` before the archive was
cut); 985 entries / 939 files / nested zips NONE / real leaks 0,
placed byte-identical in all four homes:

- `download/pharma-ml-toolkit-v1.2.128.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.128.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.128.zip`
- `upload/pharma-ml-toolkit-v1.2.128.zip`

The immediate predecessor (v1.2.127) was removed from all four homes
(the deletion rides this commit); deeper history stays (v1.2.106's
only surviving copy lives in the repo download/ sweep archive).

**What shipped:** directive #16's 16.18 migration step 8 — the
Studio's Generative Workbench (`ui/pages/generative.py`, audit row 9,
the playground Generative tab): the audit row's clusters wired to the
same public backend APIs the cards call — the evolution run
(`build_objective`-built objectives over GeneticOptimizer /
ReinventOptimizer / MctsOptimizer with the CLI corpus idiom mirrored
exactly and the ranked leaderboard + per-generation best/mean curves:
GA best 0.927 / 271 uniques, multi-objective seeded 0.962 / 276, RL
0.582 / 58, MCTS 0.593 / 17 with 276 tree nodes / 828 rollouts), the
Pareto front (`pareto_front` over the multi-objective breakdown: 9 of
276 non-dominated, the strongest member dominates 262, the
deepest-front-member rank 59 — a single-objective run refuses
honestly), molecule surgery (`ring_topology_moves` — piperidine's 3
mutants, aromatic rings never re-sized — and `scaffold_hop_crossover`
breeding, every product scored by `QedObjective` for the bars) and the
GuacaMol scoring oracles (`qed_score` / `sa_score` / `penalized_logp`
/ `rediscovery_score` + the property filter: aspirin vs salicylic
acid QED 0.550 · SA 8.61 · logP −7.30 · rediscovery 0.448 · PASS).
Sidebar 10 → 11 sections; 21 new offscreen tests
(`tests/ui/test_generative_page.py`) with every number
engine-printed first by the committed `scripts/probe_generative_214.py`;
**the round-213 notebook driver's copy-back bug caught and fixed** —
`shutil.copy2(work_nb, src)` restored the UNEXECUTED work file
(nbclient mutates the in-memory notebook; the work FILE is never
written), round 213's pass was masked because the tree already carried
1.2.127 outputs from the concurrent agent's run, and the fixed driver
writes with `nbformat.write(nb, src)` (both the 213 and 214 copies
carry the fix — the first 214 run reported all-green while silently
restoring 1.2.127 outputs; the version-pin test caught it, the second
run committed real 1.2.128 outputs); notebooks re-executed at 1.2.128
(227 s, zero error outputs) + the 7 no-output twins regenerated with
the variant-note id; the OOM ritual respected (dev server down for the
re-execution, restarted via the double-fork daemon after, GET / 200);
honest totals 5,462 (4,684 collect-only + 778 doctests); web pins
5,441 → 5,462 (page.tsx hero + FALLBACK_STATS 957/348,207 →
959/349,352, download-panel 3 lines, overview 3 lines) + census
957/348,207 → 959/349,352 (`scratch/census.py`, the toolkit-fs.ts-
exact walk, run on the post-ship tree).

## v1.2.127 release-zip homes (2026-11-07, round 213 — the Deep Learning Workbench, the Studio's increment 7)

`pharma-ml-toolkit-v1.2.127.zip` — 19,833,253 bytes, sha-256
`9ccabfd3a66953947070891e1d1d2923ac5b08b5ac54c9e3b7aba72ed87cdcd3`,
built with the standing four-home builder (`scratch/build_zip_213.py`,
the round-180/212 recipe plus the predecessor sweep and per-home
testzip verification; this round had a two-agent race — while the
IM continuation session was mid-flight, the 2-hour cron fired its
run 11 as a concurrent agent that found the WIP page, incremented
the counter to 11, improved the builder, re-executed the notebooks
and rebuilt, then stalled; three consecutive builds of the same
post-scrub tree all measured exactly 19,833,253 bytes — the byte
pin the web endpoint serves is rebuild-stable, while zip's directory
walk order varies the digest between otherwise-identical builds, so
the sha-256 above is the FINAL canonical build's, superseding both
the cut-off session's 19,840,058 / `910685b4…` unscrubbed-tree
snapshot and the concurrent agent's `2bccb408…` intermediate);
scrub-before-build caught ONE real leak this round — the re-executed
ML notebook's warning stderr again (the round-212 rule holding: a
version-bump re-execution can leak ANY warning stream carrying an
absolute path; four stderr streams over sklearn's
ConvergenceWarning/FutureWarning tracebacks with the venv path were
dropped whole before the archive was cut); 983 entries / 937 files /
nested zips NONE / real leaks 0, placed byte-identical in all four
homes:

- `download/pharma-ml-toolkit-v1.2.127.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.127.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.127.zip`
- `upload/pharma-ml-toolkit-v1.2.127.zip`

The immediate predecessor (v1.2.126) was removed from all four homes
(this round's deletion rides the commit); deeper history stays
(v1.2.106's only surviving copy lives in the repo download/ sweep
archive).

**What shipped:** directive #16's 16.18 migration step 7 — the
Studio's Deep Learning Workbench (`ui/pages/deep_learning.py`, audit
row 8, the playground Deep Learning tab): the audit row's four
clusters wired to the same public backend APIs the cards call — the
training run (`make_train_test_split` over the numeric-feature rule,
then `DeepMLP*`/`ResidualMLP*` (torch) or `NumpyMLP*` (the numpy
autograd twin when torch is absent) with per-epoch callbacks and
early stopping, then `evaluate_model`: the parity scatter + overfit
line for regression — R² 0.891 / gap 0.061 on the bundled 260-row
table — or the per-class F1 bars + confusion table for
classification — accuracy 0.952 over active/inactive/weak), the
runboard (`RunTrace` + `smooth_series` EMA loss curves;
`numpy_layer_stack` + `histogram_event` weight histograms with the
honest torch refusal), the UMAP map (`umap_embed` + Venna-Kaski
`trustworthiness` 0.770 over the run's 52 test rows), and the
fiftyone-style curation (`SampleSet`/`DataView` stage algebra +
`find_duplicates`/`find_outliers`/`uniqueness`/`sort_by_similarity`).
Sidebar 9 → 10 sections; 27 new offscreen tests
(`tests/ui/test_deep_learning_page.py`) with every number
engine-printed first by the committed `scripts/probe_dl_213.py`
(repo-root scripts/, outside the archive); the probe's dropout
mismatch caught in-round (the numpy branch had defaulted to 0.0
while the page passes 0.1 to both engines — the probe was aligned to
the page's honest shared-knob contract and re-run before any test
assertion was written); the GUI_REDESIGN.md migration status caught
up (it had lagged one increment — v1.2.126's entry recorded
alongside this one); notebooks re-executed at 1.2.127 (397 s, zero
error outputs, foreground — the second execution after the cut-off
session's 247 s run, same all-green result, the committed outputs
are the second run's) + the 7 no-output twins regenerated (the
variant-note id pin reproduced); the OOM ritual respected (dev
server down for the re-execution, restarted via the double-fork
daemon after, GET / 200); honest totals 5,441 (4,663 collect-only +
778 doctests); web pins 5,414 → 5,441 (page.tsx hero chip +
FALLBACK_STATS census mirror + download-panel 3 lines + overview 3
lines — the comma-less `5441` literals; the census fix rode
FALLBACK_STATS: 349,558 → 348,207) + census
955/347,988 → 957/348,207 (`scratch/census.py`, the toolkit-fs.ts-
exact walk, run on the post-ship tree).

## v1.2.126 release-zip homes (2026-11-07, round 212 — the Unsupervised Workbench, the Studio's increment 6)

`pharma-ml-toolkit-v1.2.126.zip` — 19,823,465 bytes, sha-256
`0a7eeea27571950d5846e2f1d0bb691f26e43df91f02379b82dfa17df076061e`,
built with the standing four-home builder (`scratch/build_zip_212.py`,
the round-180/183 recipe with VERSION read dynamically);
scrub-before-build caught TWO real leaks this round — the re-executed
ML notebook's warning stderr (a FutureWarning traceback carrying the
venv path, next to the usual ConvergenceWarning family) AND the
round-211 probe's absolute run-hint (created after v1.2.125 was cut,
never scrubbed until now; its paths are file-relative and its TK
constant parents-based); 981 entries / 935 files / nested zips NONE /
real leaks 0 (the /home/jake and /home/USER hits in two old junit
xmls are historical guard-test quotations, byte-identical to the
v1.2.125 archive), placed byte-identical in all four homes:

- `download/pharma-ml-toolkit-v1.2.126.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.126.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.126.zip`
- `upload/pharma-ml-toolkit-v1.2.126.zip`

The immediate predecessor (v1.2.125) left all four homes; deeper
history stays (v1.2.106's only surviving copy lives in the repo
download/ sweep archive).

## v1.2.125 release-zip homes (2026-11-07, round 211 — the Docking Workbench, the Studio's increment 5)

`pharma-ml-toolkit-v1.2.125.zip` — 19,705,454 bytes, sha-256
`877a5ecf76b29c4ad23fbfda94e89b8a6ccce5f0e23890ebf039bdd45fa724be`,
built with the standing four-home builder (`scratch/build_zip_210.py`,
scrub-before-build: the re-executed ML notebook's ConvergenceWarning
traceback caught and scrubbed again before the archive was cut;
977 entries / 932 files / nested zips NONE / real leaks 0 / 12
accepted placeholders), placed byte-identical in all four homes:

- `download/pharma-ml-toolkit-v1.2.125.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.125.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.125.zip`
- `upload/pharma-ml-toolkit-v1.2.125.zip`

The immediate predecessor (v1.2.124) left all four homes; deeper
history stays (v1.2.106's only surviving copy lives in the repo
download/ sweep archive).

**What shipped:** directive #16's 16.18 migration step 5 — the
Studio's Docking Workbench (`ui/pages/docking.py`, audit row 5, the
playground Docking tab): four engine tabs wired to the same public
backend APIs the cards call, none needing the external engine
binaries — the campaign summary (`check_docking_completion` +
`scan_docking_directory` over the bundled synthetic AutoDock4
campaign: 14 folders, 12 OK / 1 missing log / 1 incomplete, then the
rescore card's arithmetic on the OK rows — RDKit heavy atoms from the
campaign's ligands.csv, `ligand_efficiency_series`, `consensus_rescore`
over `best_lowest_energy` + `best_num_in_cluster` +
`ligand_efficiency`, metformin 0.782 the engine-printed top), the
cluster analysis (`parse_dlg_cluster_table` — the CLUSTERING
HISTOGRAM as a real table, the folder combo fed by the scan's
cross-tab state), the ligand library audit (`load_ligand_csv` — 8
valid + 2 deliberately invalid rows whose reasons ride the table —
plus RDKit heavy atoms / rotatable bonds), and the DOCKSTRING lab
(`target_frame`'s 58 upstream Vina search boxes, `get_target` config
previews, `parse_search_box` validation). The sidebar grows to eight
sections (docking between biology and models); pages.count() pin
7 → 8. All 8 notebooks re-executed at 1.2.125 (201 s, zero error
outputs, 7 twins regenerated); 14 new offscreen tests, every asserted
number engine-printed first. Honest totals: collect-only **4,613** +
778 doctests = **5,391**; web pins 5,377 → 5,391 (seven sites) +
census 950/345,766 → **952 / 346,738** (`scratch/census_210.py` on
the post-ship tree). Counter 9 → 10 (this run's cron trigger, the
first since round 210 recorded "no new cron trigger").

## v1.2.123 release-zip homes (2026-11-07, round 209 — the second playground collapse mode fixed) — the second playground collapse mode fixed)

`pharma-ml-toolkit-v1.2.123.zip` — 19,781,667 bytes, sha-256
`d1627ecd3f9fdc778dfd862ec1aa1e766c995e618265cf13edc19c1f4c1d74c2`
(the first build at 19,781,694 leaked one `/home/z/` path into the
re-executed ML notebook's ConvergenceWarning traceback — scrubbed by
the standing `scratch/scrub_home_paths_137.py` pass, rebuild
supersedes it), built with the standing builder
(`scratch/build_zip_183.py`: 972 entries / 926 files / real leaks 0),
placed in all four homes:

- `download/pharma-ml-toolkit-v1.2.123.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.123.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.123.zip`
- `upload/pharma-ml-toolkit-v1.2.123.zip`

**What shipped:** the collapse forensics round — the "Playground.py
collapses during heavy work" family's second mode, root-caused and
fixed.  The mechanism: `PlaygroundApp.destroy()` left the module-level
card registries, Tcl's pending `after` scripts and every
`_register`ed callback command (bind handlers, variable traces — the
trace commands tkinter's `Variable` never deletes) holding the whole
app graph past the app's death; the Tcl interpreter and its X display
connection therefore survived `destroy()` + `del` + `gc.collect()`,
and when the X server went away (a per-class/per-module Xvfb fixture
teardown) the next Tk event pump touched the broken connection — Tk
9's X error handler terminated the whole process (`X connection to :N
broken` → `exit(1)`, rc=1, zero output).  Reproduced deterministically
outside pytest (`scratch/probe_kill_between.py`; the `exit()` call
site captured via an LD_PRELOAD shim, `scratch/exit-shim.c`:
`Tk_RestackWindow → TkWmRestackToplevel → TkWmMapWindow →
TkUnixDoOneXEvent → exit(1)`).  The fix: a real three-arm teardown
(registries cleared at build+destroy, afters cancelled, 2,147
digit-leading callback commands swept post-destroy, variables
silenced) plus one session-scoped Xvfb (:96) replacing the four
per-file servers — a mid-session server death is structurally
impossible now.  Honest totals: collect-only **4,586** (+2 regression
tests) + doctests 775 passed / 3 skipped = **5,364**; the combined
job_safety+workflows battery — the exact pre-fix killing shape —
passes twice from a clean room (45 tests, rc=0, zero exit noise);
self-test green; web pins 5,362 → 5,364 + census 946/344,354 → 946 /
344,494.

## v1.2.122 release-zip homes (2026-11-07, round 208 — PK/PD Workbench, the Studio's increment 3)

`pharma-ml-toolkit-v1.2.122.zip` — 19,784,995 bytes, sha-256
`702a83dc876885b5ee3334d4d94e84ceee8425feeac22e36503a405b2350037e`
(the Run 8 addendum rebuild after the lost-update race fix — the
pre-addendum build at 19,777,011 / `7cc737a1…` is superseded),
built with the standing builder (`scratch/build_zip_183.py`: 972 entries /
926 files / real leaks 0), placed in all four homes:

- `download/pharma-ml-toolkit-v1.2.122.zip` (repo root)
- `pharma-ml-toolkit/download/pharma-ml-toolkit-v1.2.122.zip`
- `pharma-ml-toolkit/dist-archive/pharma-ml-toolkit-v1.2.122.zip`
- `upload/pharma-ml-toolkit-v1.2.122.zip`

**What shipped:** directive #16's 16.18 migration step 3 — the Studio's
PK/PD Workbench (`ui/pages/pkpd.py`, audit row 3): three of the
playground PK/PD tab's engines wired to the same public backend APIs
the cards call — the exact two-compartment regimen simulator
(`TwoCompartment` matrix-exponential linCmt with explicit repeated
`DosingEvent` records; concentration curve with dotted dose markers,
Cmax/Tmax, trapezoid AUC, the realized dose-event table), the
non-compartmental analysis (`validate_concentration_series` →
`nca_summary` + the extended `fit_terminal_phase`/`auc_linear_log`:
linear-up/log-down AUC, terminal λz window with adjusted R² and span,
Vd/CL with an IV dose, the observed profile with its AUC shaded), and
the NONMEM/mrgsolve event grammar (`ev`/`addl_until`/`realize_addl`/
`ev_days` — the full realized dose-event table with weekday schedules).
`ui/widgets/plot_widget.py` gained two reusable palette-aware primitives
(`line(fill=True)` area shading + `vlines` dose markers); all 8 notebooks
re-executed at 1.2.122 (200 s, 7 twins regenerated); 18 new offscreen
tests. Honest count **4,582 + 778 = 5,360 tests**; fresh walk pins
**945 files / 344,253 lines**.

## v1.2.141 release-zip homes (2026-11-08, round 228 — THE RETIREMENT ROUND: the ML Workbench joins the Studio, Part-2 declared complete, playground.py retired — the owner's "everything must be in studio.py" directive, increment 3)

`pharma-ml-toolkit-v1.2.141.zip` — 19,834,074 bytes, sha-256
`a50ae20d6050a4cf83f5a52db5f72a21f08f1130caed42f88adf74496d3626e7`,
1010 entries / 965 files, built with the standing four-home builder
(`scratch/build_zip_210.py`, the committed version-agnostic recipe)
— testzip OK per home, **real leaks 0**, nested zips NONE, predecessor
v1.2.140 swept from all four homes; placed byte-identical in
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`. Census **985 files /
311,333 lines** (the retirement's −55,868 lines: playground.py
38,317 + its 980-test Tk-coupled suite ~21,600, plus the ML Workbench
+1,798 and the Model Analysis round-228 additions). Honest totals
**4,769 = 3,974 collect + 795 doctest** (5,725 − 980 retired + 24 new).

## v1.2.142 release-zip homes (2026-11-08, round 229 — the Part-3 §20 checkpoint engine: long jobs pause/resume/stop without losing work)

`pharma-ml-toolkit-v1.2.142.zip` — 19,982,355 bytes, sha-256
`5579a34af1c854b6478549a7b58aeb3c4f3d4fd69385194765bff89e3b0c5732`,
1013 entries / 968 files, built with the standing four-home builder
(`scratch/build_zip_210.py`, the committed version-agnostic recipe)
— testzip OK per home, **real leaks 0**, nested zips NONE, predecessor
v1.2.141 swept from all four homes; placed byte-identical in
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`. Census **988 files /
312,512 lines** (the checkpoint engine +1 module and its 36-test
file, the six benchmark integration tests, the GUI resume-loop test,
the README §20 section and the round-229 docs rows). Honest totals
**4,815 = 4,019 collect + 796 doctest** (4,769 + 43 new: 36 engine +
6 benchmark integration + 1 GUI two-Start loop; +2 straggler tests
the round-228 count predates). Notebooks re-executed **8/8 at
pharmaml 1.2.142 in 201 s** (the foreground recipe).

## v1.2.143 release-zip homes (2026-11-08, round 230 — the Part-3 §24 SA engine: Ertl synthetic accessibility end-to-end, completing the interrupted round's ship)

`pharma-ml-toolkit-v1.2.143.zip` — 20,002,126 bytes, sha-256
`a99c67ed11d95c0883580df15f95113bc79b865bbf57ab88759e40acf455ce7e`,
1016 entries / 971 files, built with the standing four-home builder
(`scratch/build_zip_210.py`, the committed version-agnostic recipe)
— testzip OK per home, **real leaks 0**, nested zips NONE, predecessor
v1.2.142 swept from all four homes; placed byte-identical in
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`. Census **991 files /
313,916 lines** (the sa_scoring engine + the generative scoring
objective module + the root EGL-safe conftest, the 37-test engine
file, the six SA-tab GUI tests, the README §24 section, the
round-230 docs rows and the regenerated INDEX). Honest totals
**4,860 = 4,062 collect + 798 doctest** (4,815 + 45: 37 engine + 6
GUI + 2 doctests). The notebooks re-executed **8/8 at pharmaml
1.2.143 in 198 s** (foreground, error-outputs 0). Round 230 also
carried the interrupted session's completion: the generative.py
docstring syntax fix, the true-value pin corrections (easiest-first
ordering, the 2/2 filter split, `_obj_lo`/`_obj_hi`), the
`penalized_logp` doctest range, and the docs-index regeneration the
new symbols required.

## v1.2.144 release-zip homes (2026-11-09, round 231 — the user-reported Windows console-noise fix + the Part-3 §22 SMILES structure previews)

`pharma-ml-toolkit-v1.2.144.zip` — 20,015,981 bytes, sha-256
`c7321d18114c2c2d5bbce2d2a740d2c40f9707c080e3ffad5c0468c53fbd65dc`,
1020 entries / 975 files, built with the standing four-home builder
(`scratch/build_zip_210.py`, the committed version-agnostic recipe)
— testzip OK per home, **real leaks 0**, nested zips NONE, predecessor
v1.2.143 swept from all four homes; placed byte-identical in
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`. Census **995 files /
314,875 lines** (the ui.qt_noise_filter module + the §22
smiles_preview widget module + their two test files, the README §22
section, the round-231 wiring and the chunk-runner update). Honest
totals **4,877 = 4,079 collect + 798 doctest** (4,860 + 17: 5
noise-filter tests incl. a live qInstallMessageHandler round-trip +
12 preview tests over the real MolDraw2DCairo renderer, the hover
popup, and the detail dialog). The notebooks re-executed **8/8 at
pharmaml 1.2.144 in 194 s** (foreground, error-outputs 0). Round 231
also completed the interrupted session's diagnosis for the
user-reported v1.2.142 Windows console noise: both families are
benign (the PySide6 wheel ships no lib/fonts on Windows, where Qt
renders through GDI/DirectWrite; the QStyleHints connect lines are
Qt 6.8 internal plumbing, not Studio code) and `ui.run()` now
installs a message-handler filter BEFORE the QApplication exists so
the noise never reaches the console; real warnings still pass and
`PHARMAML_STUDIO_QT_VERBOSE=1` disables the filter.

## v1.2.145 release-zip homes (2026-11-09, round 232 — the user-reported Windows black-window fix + the Part-3 §21 slice 1 hover inspection)

`pharma-ml-toolkit-v1.2.145.zip` — 20,140,128 bytes, sha-256
`d3c666272d52aba253eebe0d6f36de44fd943920486efdfda72fdc8dbc6fa0bc`,
1028 entries / 982 files, built with the standing four-home builder
recipe (`pharma-ml-toolkit/scratch/build_zip_232.py`, the needles and
the repo path assembled at runtime — the archive-ships-clean
discipline the round-137 scrubber taught) — testzip OK per home,
**real leaks 0** (after the scrub: the re-executed ML notebook's
sklearn-warning venv paths neutralised, the two stale round-148/158
junit quotations neutralised, the builder's own literals assembled),
nested zips NONE, predecessor v1.2.144 swept from all four homes;
placed byte-identical in `download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`. Census **1001 files /
316,019 lines** (the ui/widgets/chart_inspector module + the two test
files + the three round-232 scratch drivers). Honest totals
**4,899 = 4,101 collect + 798 doctest** (4,877 + 22: 7 platform
decision-matrix tests + 15 ChartInspector tests). The notebooks
re-executed **8/8 at pharmaml 1.2.145 in 195 s** (foreground,
error-outputs 0). Round 232's headline is the SECOND user-reported
Windows fix in a row, and this one was a real bug: `studio.py`
launched with console noise and NO window on the user's Windows
box because `ui/headless.py`'s displayless probe tested only X11's
`DISPLAY` — a variable that never exists on Windows — silently
forcing the offscreen platform there (the app ran with an invisible
window; `app.exec()` blocked forever). The probe is now
platform-aware: offscreen ONLY on Linux-class platforms with neither
`DISPLAY` nor `WAYLAND_DISPLAY`; Windows/macOS keep native; an
explicit `QT_QPA_PLATFORM` still wins. The round also shipped the
Part-3 §21 slice 1: every PyQtGraph chart (all fourteen pages plot
through PlotPanel) gained a ChartInspector — crosshair,
nearest-real-point snapping (bar bodies are whole hover targets),
the readout line of EXACT values with cached per-point metadata,
per-series visibility toggles + reset-view in the context menu, and
click-to-select with the `pointSelected` drill-down signal; the
decorations are fully LAZY (first hover) so the pages'
`listDataItems()[0]` and InfiniteLine-census pins stay true, and the
readout floats as an overlay so the §17 export pins keep the panel's
exact geometry. Suite: the seven-chunk gate grew an eighth chunk
(4b split into 4b+4d — the round-225 OOM lesson repeating at 1.66 GB
rss); 8/8 chunks green. Round 232 also repaired the interrupted
round 231's missing docs (the CHANGELOG 1.2.144 entry, the COVERAGE
row, the GUI_AUDIT progress) and recreated the missing 15-minute
webDevReview cron (job 515897).

## v1.2.146 release-zip homes (2026-11-09, round 233 — the Part-3 §23 slice 1: the responsive pass)

`pharma-ml-toolkit-v1.2.146.zip` — 20,244,070 bytes, sha-256
`18bdfbcffe3ef6f72a3febec1a307453581425db8c1b375a127c78a5b806a33d`,
1035 entries / 988 files, built with the standing four-home builder
recipe (`pharma-ml-toolkit/scratch/build_zip_233.py`, predecessor
v1.2.145 swept from all four homes) — testzip OK per home, **real
leaks 0** (the first build caught the re-executed ML notebook's
sklearn-warning venv paths; the round-137 scrubber neutralised them),
nested zips NONE.  Built from the REBASED tree: the user's PR #3
(repo trust signals + CI + the alphabetical `__all__` re-sort in
`chem/__init__.py` and the `_ring_bonds` condition inline in
`ml/mpnn.py`) landed mid-round and is absorbed — 149 focused tests
over the touched modules (chem tables / SA engine / docs-index / gnn
doctests) re-run green before the rebuild. Placed byte-identical in
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`. Census **1008 files /
317,082 lines** (the round's own additions — the
`ui/widgets/responsive.py` policy module, the 13-test
`tests/ui/test_responsive_policy.py`, the 233 chunk runner — plus the
user's PR #2 landing-page contribution absorbed from origin: the
repo-root README, `examples/chart_readability.py` +
`docs/CHART_READABILITY.md` + the chart-readability PNG). Honest
totals **4,912 = 4,114 collect + 798 doctest** (4,899 + 13 responsive
tests). The notebooks re-executed **8/8 at pharmaml 1.2.146 in 212 s**
(foreground, error-outputs 0). Round 233's headline is the Part-3 §23
responsive pass, slice 1, landed as WIRED POLICY rather than per-page
heroics: `ui/widgets/responsive.py::configure_splitter` puts the
§23.2 discipline on **all 20 splitters** in the app (every page
splitter, the Image Lab's four tabs, the ML Workbench's two, the main
window's nav/content splitter — whose nav side was previously
collapsible: the whole navigation could be dragged out of reach): no
collapsible children, hard per-child minimum extents (320 px default,
nav 200), stretch factors routing window growth to the chart side.
`LEGIBILITY_FLOOR` (640×480) answers §23.3's "plots remain legible
when exported": a squeezed QGraphicsView re-renders from its VECTOR
scene at floor scale — crisp text, preserved aspect, the pyqtgraph
chrome re-filled via `backgroundBrush()` — with §23.5 pinned by test
(data arrays and view ranges byte-identical across an export). The
ImagePanel locks aspect (§23.4: matrix pixels stay square, never
stretched). Suite: 8/8 chunks green — validated TWICE this round (the
session's own gate run and the concurrent cron session's re-run of
the same tree). A concurrent-session note: the master cron job fired
mid-round (pulled the user's PR #2 merge `240c082`, re-ran the 233
chunk runner) — no working-tree collision, both suite runs green on
identical content, and this release commit is the single ship.

## v1.2.147 release-zip homes (2026-11-09, round 235 — the Part-3 §23 slice 2: the scroll policy)

`pharma-ml-toolkit-v1.2.147.zip` — 20,301,559 bytes, sha-256
`2698ab08e94e187393021ca8e14579950622b1a37bc6369dbb4ff37011eb35a4`,
1038 entries / 991 files, built with the standing four-home builder
recipe (`pharma-ml-toolkit/scratch/build_zip_235.py`, predecessor
v1.2.146 swept from all four homes) — testzip OK per home, **real
leaks 0** (the first build caught the re-executed ML notebook's
sandbox venv paths again; the round-137 scrubber neutralised them),
nested zips NONE.  Placed byte-identical in `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`, and
`upload/`.  Census **1011 files / 317,447 lines** (the round's
additions: the `wrap_in_scroll` half of the §23 responsive policy,
the ML Workbench's five tab bodies wired through it, the three
scroll-policy tests, and the 235 runner + builder scripts).  Honest
totals **4,915 = 4,117 collect + 798 doctest** (+3).  The round also
recovered the 22nd sandbox rollback (local wiped to the round-186
commit; ff-only pull to origin) and verified the v1.2.145 Windows
GUI fix lineage end-to-end.

## v1.2.148 release-zip homes (2026-11-09, round 236 — the Part-3 §21.5 molecule readouts)

`pharma-ml-toolkit-v1.2.148.zip` — 20,311,284 bytes, sha-256
`7afc0aecf3d16a83b9b3283e522a33151b08e79beac442459516e4b63ce1f68b`,
1040 entries / 993 files, built with the standing four-home builder
recipe (`pharma-ml-toolkit/scratch/build_zip_236.py`, predecessor
v1.2.147 swept from all four homes) — testzip OK per home, **real
leaks 0** (the re-executed ML notebook's sandbox venv paths scrubbed
by the round-137 cleaner before the rebuild), nested zips NONE.
Placed byte-identical in `download/`, `pharma-ml-toolkit/
dist-archive/`, `pharma-ml-toolkit/download/`, and `upload/`.  Census
**1013 files / 317,964 lines** (the round's additions: the
`PlotPanel.point_selected` forwarding, the SA rank-vs-score scatter
with lazy molecule metadata, the Pareto-front hover metadata, the 4
tests, the 236 runner + builder, and the retroactive round-235 docs —
CHANGELOG 1.2.147, GUI_REDESIGN increment 18, GUI_AUDIT's §23
slice-2 progress).  Honest totals **4,919 = 4,121 collect + 798
doctest** (+4).

## v1.2.149 release-zip homes (2026-11-09, round 237 — the Qt import guard + `studio.py --doctor`)

`pharma-ml-toolkit-v1.2.149.zip` — 20,325,624 bytes, sha-256
`654da9e9a099a50701483c236ae568077408cedda8a817953b89941333d7d493`,
1044 entries / 997 files, built with the standing four-home builder
recipe (`scratch/build_zip_237.py`, predecessor v1.2.148 swept from
all four homes) — testzip OK per home, **real leaks 0** (the
re-executed ML notebook's sandbox venv paths scrubbed by the
round-137 cleaner before the rebuild), nested zips NONE.  *(Rebuilt
once intra-round: the first build's tree carried a stale inner
version chain — `pyproject.toml` and `pharmaml.__version__` still
said 1.2.148 while `VERSION` said 1.2.149, which the zip-size route
surfaced as an `exact:false` fallback answer; the chain is now
three-way synced inside the archive and the endpoint answers
`exact:true, 20325624`.)*  Placed
byte-identical in `download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`.  Census **1017 files /
318,789 lines** (the round's additions: `ui/qt_import_guard.py` and
`ui/doctor.py` behind the guarded `QApplication` import + the
`--doctor` dispatch, the 7-test `tests/ui/test_qt_doctor.py`, the
237 chunk runner, and the ui/-tree lint cleanup — 19 ruff findings +
7 mypy errors from rounds 233–236 cleared; studio.py now carries its
executable bit).  Honest totals **4,926 = 4,128 collect + 798
doctest** (+7).

## v1.2.150 release-zip homes (2026-11-09, round 238 — the Part-3 §23.6 adaptive tick-label density)

`pharma-ml-toolkit-v1.2.150.zip` — 20,332,751 bytes, sha-256
`0489ea9bb24e1b4fe7bae57ced13b774eee794a2426ce4033e15cc94b11cfdb7`
(1046 entries / 999 files, testzip OK per home, real leaks 0 — the
re-executed ML notebook's sandbox venv paths scrubbed by the
round-137 cleaner before the build, nested zips NONE).  Built with
the standing four-home builder recipe (`scratch/build_zip_238.py`,
predecessor v1.2.149 swept from all four homes).  *(Rebuilt once
intra-round: the first build predated the `[1.2.150]` CHANGELOG
entry, so the archive's CHANGELOG head still read 1.2.149 — the
rebuild carries the shipped release note.)*  Placed
byte-identical in `download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`.  Census **1019 files /
319,164 lines** (the round's addition: the 10-test
`tests/ui/test_plot_tick_density.py` behind the §23.6 policy in
`ui/widgets/plot_widget.py`).  Honest totals **4,936 = 4,138 collect
+ 798 doctest** (+10).

## v1.2.151 release-zip homes (2026-11-09, round 239 — the Part-3 §23.7 legend hygiene)

`pharma-ml-toolkit-v1.2.151.zip` — 20,339,302 bytes, sha-256
`fe003349b5bfc884a2201dbb9e6f8f63ae05709a7c40a73f21384090d10fdad0`
(1048 entries / 1001 files, testzip OK per home, real leaks 0 — the
re-executed ML notebook's sandbox venv paths scrubbed by the
round-137 cleaner before the build, nested zips NONE).  Built with
the standing four-home builder recipe (`scratch/build_zip_239.py`,
predecessor v1.2.150 swept from all four homes; the CHANGELOG
`[1.2.151]` entry predated the build — no intra-round rebuild).
Placed byte-identical in `download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`.  Census **1021 files /
319,620 lines** (the round's additions: the 11-test
`tests/ui/test_legend_policy.py` behind the §23.7 policy in
`ui/widgets/plot_widget.py` + the 239 chunk runner).  Honest totals
**4,947 = 4,149 collect + 798 doctest** (+11).

## v1.2.157 release-zip homes (2026-11-09, round 245 — the Part-3 §23.10 ImagePanel sub-pixel mapping disclosure)

`pharma-ml-toolkit-v1.2.157.zip` — 22,222,879 bytes, sha-256
`1ffd642900fb37f34aab1d170da54fcc4ae8194a10cb38ba203e3d880e1886ee`
(1073 entries / 1026 files, testzip OK per home, real leaks 0 — the
size grows by the re-executed notebooks, the round-244 QA pair
riding `scratch/` by the audit-trail design, and the new
§23.10 suite + probe; this round's `r245_web_verify.png` rides the
NEXT build; the sha is the byte-identical four-home state of THIS
run).  Built with the standing four-home builder recipe
(`scratch/build_zip_245.py`, predecessor v1.2.156 swept from all
four homes; the CHANGELOG `[1.2.157]` entry predated the build).
The builder's leak gate caught the notebook re-execution's
`/home/z/` embedding on the FIRST build (the known ritual: scrub
with `scrub_home_paths_137.py`, verification zero absolute-home
matches, second build clean) — AND the suite's own leakage guard
(`test_leakage.py`) caught the round's probe docstring carrying an
absolute interpreter path before any build; both fixed at the
source.  Placed byte-identical in `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`,
and `upload/`.  Census **1046 files / 323,489 lines** per the
canonical walk.  Honest totals **5,028 = 4,230 collect + 798
doctest** (+16).  Web pins trued 5028 ×6 + census fallbacks
1046/323,489 + `SHIPPED_TOOLKIT_VERSION` 1.2.157 + the round-245
clause in the Testing after-string.

## v1.2.158 release-zip homes (2026-11-09, round 246 — the Part-3 §21.4 original-resolution 1:1 view)

`pharma-ml-toolkit-v1.2.158.zip` — 22,394,779 bytes, sha-256
`c29b17c9845df4b35eea9d7a42873a3fae5cc5cb6a2fec9b68dd1fd5d46444b8`
(1079 entries / 1032 files, testzip OK per home, real leaks 0 — the
size grows by the re-executed notebooks, the round-245 QA pair
riding `scratch/` by the audit-trail design, and the new §21.4
1:1 suite + probe + the split 4c/4e runner; this round's
`r246_web_verify.png` rides the NEXT build; the sha is the
byte-identical four-home state of THIS run).  Built with the
standing four-home builder recipe (`scratch/build_zip_246.py`,
predecessor v1.2.157 swept from all four homes — the PREDECESSOR
constant trued BEFORE the first build this time, the round-244
lesson applied; the CHANGELOG `[1.2.158]` entry predated the
build).  The round-137 scrub caught the notebook re-execution's
`/home/z/` embedding (`complete_ml_workflow.ipynb`, the known
ritual: scrubbed, verification zero absolute-home matches) BEFORE
the build — the builder's leak gate then confirmed real leaks 0 on
the first build.  The gate itself grew to NINE chunks: the round's
+16 tests tipped the single 4c process past the Qt teardown edge
(SIGSEGV −11 at 74% with every test passing in two halves — the
round-232 lesson, third verse), so 4c split into 4c (widget suites)
+ 4e (page suites), both halves measured green before the full
re-run.  Placed byte-identical in `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`,
and `upload/`.  Census **1052 files / 324,397 lines** per the
canonical walk.  Honest totals **5,044 = 4,246 collect + 798
doctest** (+16).  Web pins trued 5044 ×6 + census fallbacks
1052/324,397 + `SHIPPED_TOOLKIT_VERSION` 1.2.158 + the round-246
clause in the Testing after-string (single quotes inside, the
round-245 Turbopack lesson applied proactively — the clause's quote
guard ran before the write).

## v1.2.161 release-zip homes (2026-11-11, round 249 — the `pharmaml.md` package: molecular-dynamics analysis)

`pharma-ml-toolkit-v1.2.161.zip` — 23,420,835 bytes, sha-256
`45c36a6f5899f54d9bb666a4a5c08f9cdf4ee627129a06a1f9ab7c346ca0a697`
(1103 entries / 1055 files, testzip OK per home, real leaks 0 — the
growth over v1.2.160's 22,750,159 B is the new `pharmaml.md` package
(10 modules, ~2,820 lines: xvg, trajectory, align, metrics, hbond,
dimred, enm, plif, study, `__init__`), the 55-test
`tests/test_md_analysis.py`, the docs INDEX `pharmaml.md` section
(+112 lines), the CHANGELOG round-249 entry, and the eight notebooks
re-executed at 1.2.161).  Built with the standing four-home builder
recipe (`scratch/build_zip_249.py`, predecessor v1.2.160 swept from
all four homes).  The round-249 feature productionizes the user's own
GROMACS post-processing script (`upload/MD_analysis.py`) against the
reference stack the user linked (MDAnalysis/mdtraj, GROMACS,
cpptraj/pytraj, PLUMED, ProLIF, getcontacts, gmx_MMPBSA, ProDy,
deeptime, nglview, pymol-open-source, vmd-python, MDAnalysisData);
the web twin is the 10th playground sub-tab ("Molecular Dynamics":
the GROMACS export parser, the trajectory analysis lab, the ecosystem
card) over `/api/md/xvg` + `/api/md/trajectory` bridge routes.  The
gate: NINE chunks — chunk-4's three honest fights, all fixed and
re-verified green in isolation: (1) the card-directory parity census
27→30 cards (three MD entries) + the probe's `md` tab mapping;
(2)+(3) `test_format_converters`' DOUBLE logging leak into
`test_numpy_neural_checkpoints` — the module-level
`logging.disable(CRITICAL)` (scoped by the round-249 fixture) AND the
CLI tests' `configure_logging()` flipping `pharmaml`.propagate to
False with a console handler bound to that test's captured stderr
(now the at-import propagate/handlers/level are restored after every
test; the two resume-contract caplog assertions had been silently
empty, with "--- Logging error ---" noise on a closed stream).
Placed byte-identical in `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`, and
`upload/`.  Census **1075 files / 329,046 lines** per the canonical
walk.  Honest totals **5,207 = 4,354 collect + 853 doctest** (+57
collect: the 55 MD tests plus two docs-index parametrization legs;
+44 doctest across the package).  Web pins trued 5207 ×7 (the flow's
first 5,205 projection missed the two parametrization legs — the
collect-only recount caught it) + census fallbacks 1075/329,046 +
`SHIPPED_TOOLKIT_VERSION` 1.2.161.

## v1.2.160 release-zip homes (2026-11-10, round 248 — the §20.4/§20.5 NumPy-trainer checkpoint depth)

`pharma-ml-toolkit-v1.2.160.zip` — 22,750,159 bytes, sha-256
`c6b4d7404af4dc57313692e2b6c5210b6c77b49b2d448f3338f935e65d58b6a6`
(1088 entries / 1041 files, testzip OK per home, real leaks 0 — the
growth over v1.2.159's 22,654,468 B is the checkpoint-depth wiring
(`ml/numpy_neural.py` +~180 lines of §20.4/§20.5 integration, the
`Optimizer.state_dict()`/`load_state_dict()` pair in `ml/autograd.py`,
the 24-test `tests/test_numpy_neural_checkpoints.py`) plus the eight
notebooks re-executed at 1.2.160).  Built with the standing four-home
builder recipe (`scratch/build_zip_248.py`, predecessor v1.2.159
swept from all four homes).  The round-137 scrub caught the notebook
re-execution's `/home/z/` embedding (`complete_ml_workflow.ipynb`,
the known ritual: scrubbed, twins re-synced, verification zero
absolute-home matches) BEFORE the build.  The gate: NINE chunks, all
green, first full pass (chunk 2 carries the checkpoint suite by
name).  Placed byte-identical in `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`, and
`upload/`.  Census **1061 files / 326,899 lines** per the canonical
walk.  Honest totals **5,106 = 4,297 collect + 809 doctest** (+24
collect, +1 doctest — the `fit()` bit-exactness example riding chunk
6).  Web pins trued 5106 x7 + census fallbacks 1061/326,899 +
`SHIPPED_TOOLKIT_VERSION` 1.2.160.

## v1.2.159 release-zip homes (2026-11-10, round 247 — multi-objective HPO, the Pareto study)

`pharma-ml-toolkit-v1.2.159.zip` — 22,654,468 bytes, sha-256
`36bfcb02a1ef1b88e54b7c4c32d1487725c4c2ab4c4cbe09770c367158c5870c`
(1085 entries / 1038 files, testzip OK per home, real leaks 0 — the
growth over v1.2.158's 22,394,779 B is the re-executed notebooks at
1.2.159 (the tuning notebook gains section 7, the trade-off demo over
the shipped qsar_regression.csv + a recap row), the new
`ml/multiobjective.py` (755 lines) + its 27-test suite, the INDEX.md
nine rows, and the round-247 runner/builder pair riding `scratch/`).
Built with the standing four-home builder recipe
(`scratch/build_zip_247.py`, predecessor v1.2.158 swept from all four
homes BEFORE the first build — the round-244 lesson held).  The
round-137 scrub caught the notebook re-execution's `/home/z/`
embedding (`complete_ml_workflow.ipynb` again, the known ritual:
scrubbed, verification zero absolute-home matches) BEFORE the build;
the builder's leak gate confirmed real leaks 0 first build.  The gate:
NINE chunks, all green — chunk 1's only fight was that round-137 leak
re-introduced by the re-execution (fixed by the standing scrubber +
isolated green re-run), the fabrication guard caught the demo cell's
first draft using `np.random.default_rng` inline (rewritten onto the
shipped `ds.X`/`ds.y` — the owner's round-24 rule, kept), and the
docs-index guard pinned the nine new `pharmaml.ml` exports into
`docs/INDEX.md`.  Placed byte-identical in `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`, and
`upload/`.  Census **1058 files / 326,049 lines** per the canonical
walk.  Honest totals **5,081 = 4,273 collect + 808 doctest** (+27
collect, +10 doctests — the module's own doctests riding chunk 6).
Web pins trued 5081 x7 + census fallbacks 1057/325,926 +
`SHIPPED_TOOLKIT_VERSION` 1.2.159 + the round-247 clause in the
Testing after-string (no new quotes inside — the round-245 guard held).

## v1.2.156 release-zip homes (2026-11-09, round 244 — the Part-3 lint-backlog close)

`pharma-ml-toolkit-v1.2.156.zip` — 22,068,281 bytes, sha-256
`3c9280837fe260c4df3e74218e599ddefdfa37fcae72824d54c08924f821752e`
(1066 entries / 1019 files, testzip OK per home, real leaks 0 — the
size grows by the re-executed notebooks and the round-243 QA
screenshot riding `scratch/` by the audit-trail design; this round's
`r244_web_verify.png` rides the NEXT build; note the zip embeds per-
entry mtimes so the sha is build-time dependent — the recorded sha is
the byte-identical four-home state of THIS run).  Built with the
standing four-home builder recipe (`scratch/build_zip_244.py`,
predecessor v1.2.155 swept from all four homes — after a copied
PREDECESSOR constant was caught pointing one round back and fixed;
the CHANGELOG `[1.2.156]` entry predated the build).  One home-path
leak (`examples/complete_ml_workflow.ipynb` carrying `/home/z/`) was
caught by the builder's leak gate on the FIRST build and scrubbed
(`scrub_home_paths_137.py`, verification: zero absolute-home
matches) before the recorded build.  Placed byte-identical in
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`.  Census **1039 files /
322,599 lines** per the canonical walk.  Honest totals
**5,012 = 4,214 collect + 798 doctest** (no test count change —
the round is lint-only).  Web pins trued: 5012 ×6 unchanged,
census fallbacks 1039/322,599, `SHIPPED_TOOLKIT_VERSION` 1.2.156,
the round-244 clause appended to the Testing after-string.

## v1.2.155 release-zip homes (2026-11-09, round 243 — the Part-3 §23.9 legend placement vs. data)

`pharma-ml-toolkit-v1.2.155.zip` — 21,851,407 bytes, sha-256
`a42f328593472be44758a091526a2da55182174a47c086fa5a66e1f88b1bb3df`
(1064 entries / 1017 files, testzip OK per home, real leaks 0 — the
size grows by the re-executed notebooks and the round-242 QA
screenshot riding `scratch/` by the audit-trail design; this round's
`r243_web_verify.png` rides the NEXT build).  Built with the standing
four-home builder recipe (`scratch/build_zip_243.py`, predecessor
v1.2.154 swept from all four homes; the CHANGELOG `[1.2.155]` entry
predated the build — no intra-round rebuild).  Placed byte-identical in
`download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`.  Census **1037 files /
322,457 lines** per the canonical walk.  Honest totals
**5,012 = 4,214 collect + 798 doctest**; the eight-chunk gate green
(8/8, 0 failing chunks) after the 8/8 foreground notebook re-execution
at 1.2.155.  Web pins trued 5012 ×6 (page badge, download-panel
describe/list/chip, overview chip + the round-243 clause in the
Testing after-string) + census fallbacks 1037/322,457 +
`SHIPPED_TOOLKIT_VERSION` 1.2.155.

## v1.2.154 release-zip homes (2026-11-09, round 242 — the Part-3 §23.8 title pixel budget)

`pharma-ml-toolkit-v1.2.154.zip` — 21,154,106 bytes, sha-256
`5fd0c6d3abaede61afd4d5ff85e365dfc0c63afbbaec77e9eb523b96d5aec441`
(1057 entries / 1010 files, testzip OK per home, real leaks 0 — the
size grows by the round-241 QA screenshot riding `scratch/` by the
audit-trail design).  Built with the standing four-home builder
recipe (`scratch/build_zip_242.py`, predecessor v1.2.153 swept from
all four homes; the CHANGELOG `[1.2.154]` entry predated the build —
no intra-round rebuild).  Placed byte-identical in `download/`,
`pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/`, and
`upload/`.  Census **1033 files / 321,738 lines** per the canonical
walk.  Honest totals **4,995 = 4,197 collect + 798 doctest**; the
eight-chunk gate green (8/8, 0 failing chunks) after the 8/8
foreground notebook re-execution at 1.2.154.  Web pins trued 4995 ×6
(page badge, download-panel describe/list/chip, overview chip + the
round-242 clause in the Testing after-string) + census fallbacks
1033/321,738 + `SHIPPED_TOOLKIT_VERSION` 1.2.154.

## v1.2.153 release-zip homes (2026-11-09, round 241 — the Part-3 §21 per-point meta adoption)

`pharma-ml-toolkit-v1.2.153.zip` — 20,449,306 bytes, sha-256
`83a85f11dc3c092f68fdebcb44d1a6c327fc9ec31e9a4bd998292e55a4e97249`
(1056 entries / 1009 files, testzip OK per home, real leaks 0 — the
re-executed ML notebook's sandbox venv paths scrubbed by the
round-137 cleaner before the build, nested zips NONE).  Built with
the standing four-home builder recipe (`scratch/build_zip_241.py`,
predecessor v1.2.152 swept from all four homes; the CHANGELOG
`[1.2.153]` entry predated the build — no intra-round rebuild).
Placed byte-identical in `download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`.  Census **1029 files /
321,233 lines** per the canonical walk (grows by the round-241 suite
file + the 241 driver).  Honest totals **4,983 = 4,185 collect + 798
doctest**; the eight-chunk gate green (8/8, 0 failing chunks) after
the 8/8 foreground notebook re-execution at 1.2.153.  Web pins trued
4983 ×6 (page badge, download-panel describe/list/chip, overview chip
+ the round-241 clause in the Testing after-string) + census fallbacks
1029/321,233 + `SHIPPED_TOOLKIT_VERSION` 1.2.153.

## v1.2.152 release-zip homes (2026-11-09, round 240 — the Part-3 §21.4 image pixel inspection)

`pharma-ml-toolkit-v1.2.152.zip` — 20,354,109 bytes, sha-256
`570724f860dd36d67dc52046f4c5475a02c8d463987b27d4c3e49d27a111c62e`
(1052 entries / 1005 files, testzip OK per home, real leaks 0 — the
re-executed ML notebook's sandbox venv paths scrubbed by the
round-137 cleaner before the build, nested zips NONE).  Built with
the standing four-home builder recipe (`scratch/build_zip_240.py`,
predecessor v1.2.151 swept from all four homes; the CHANGELOG
`[1.2.152]` entry predated the build — no intra-round rebuild).
SHA CORRECTED POST-PUSH: the first figure printed here and in the
`60cab75` commit message (`31ef34b3…`) was a mid-write read taken
while the four-home copies were in flight; the committed blobs and
all four homes verify `570724f8…` (this round's cron agent re-hashed
the shipped bytes and the git blobs — they agree).
Placed byte-identical in `download/`, `pharma-ml-toolkit/dist-archive/`,
`pharma-ml-toolkit/download/`, and `upload/`.  Census **1026 files /
320,658 lines** per the canonical walk (re-trued post-push: the
first-recorded 1024/320,559 was taken mid-flight before the
`r240_web_verify.png` QA screenshot landed; the fallback pins were
corrected to the verified walk in the same commit).  The round's
additions: the 20-test
`tests/ui/test_image_hover_probe.py` behind the §21.4 pixel
inspection in `ui/widgets/image_panel.py` + the
`scratch/probe_image_hover_240.py` feasibility probe + the 240 chunk
runner and build driver).  Honest totals **4,967 = 4,169 collect +
798 doctest** (+20).

## v1.2.162 release-zip homes (2026-11-10, round 250 — the MD viz twins + the unioned Studio Molecular-Dynamics page + the arena Pareto/SA PRs)

`pharma-ml-toolkit-v1.2.162.zip` — 23,669,404 bytes, sha-256
`47942c1a6a44b8519998d9bf3350f8f551de43724f6b974789a5b3ef7f2af748`
(1117 entries / 1069 files, testzip OK per home, real leaks 0 — the
growth over v1.2.161's 23,420,835 B is the new `viz/md_plots.py`
module, the two MD test suites (`tests/test_viz_md_plots.py` 29 tests
+ `tests/test_md_plots.py` 10), the unioned Studio page
(`ui/pages/molecular_dynamics.py` + `tests/ui/
test_molecular_dynamics_page.py` + navigation), `docs/
MD_VISUALIZATION.md`, the docs INDEX regeneration (13 modules / 887
symbols — now generated WITH `pharmaml.md` after the round-250
generator fix), the CHANGELOG round-250 entries, and the eight
notebooks re-executed at 1.2.162).  Built with the standing four-home
builder recipe (`scratch/build_zip_250.py`, predecessor v1.2.161
swept from all four homes).  Round 250 was a TWO-half round: the
morning shipped the viz twins (`plot_md_series` /
`plot_md_correlation` as stats-dict figures on the lazy `_compat`
contract); mid-round the arena flow's PR #6 landed a competing
smaller `md_plots.py` plus a Studio MD page and broke `main` for a
moment (its `MDStudy.metric_series` edit sat fine, but its toolkit
`ci.yml` was byte-broken `branches: ain]` and its page test rode no
gate chunk; the utilities-page index pin 15→16 was missed too).  The
union resolved it honestly: the released stats-dict API stayed
canonical and absorbed the arena deltas (mapping comparison input,
LigandResult acceptance, explicit label overrides, `close=False`
figure ownership, `N/A`-masked NaN cells, the "axes must match"
misaligned-abscissa refusal, the ±inf refusal); the Studio page was
adapted to it; their test file was rewritten to the union contract;
the workflow and the missed pins were fixed.  Round 250 also closed
the arena flow's PRs #7/#8 (the Pareto-study shortlist figure `viz/multiobjective_plots.py` + the SA-aware generation CLI objective) were merged clean on top and rode the third full gate; the notebook leak class for good: the round-137 no-absolute-path
ritual is now code — `reexec_notebooks_209.py` scrubs the live venv
prefix out of every text payload (str AND list — nbformat reads
`text` as `str`) and FAILS the run if any absolute home path
survives, so the release scanner's `real leaks: 0` is enforced at the
source.  Honest totals **5,277 = 4,423 collect + 854 doctest** (+70
over the 5,207 round-249 baseline: 29 viz-twin tests + 10 union
md-plot tests + the md_plots doctest module + the Studio page suite
+ the arena PRs' 22 tests and index-pin edits); the gate is NINE
chunks, all green THREE times (morning release, the PR #6 union,
and the PR #7/#8 union — chunk 1 re-run after the docs-index fix,
chunk 4e re-run after the utilities index pin).  Ruff zero, mypy
clean across 206 source files.  Census **1089/331,356**.  The round's version-carrier lesson, recorded: the version lives in THREE
plus files — `VERSION`, `pyproject.toml` [project], and
`src/pharmaml/__init__.py.__version__` — and round 250's first build missed
the latter two (the zip-size endpoint answered 1.2.161 exact:false from the
live-stream fallback until both were trued and the notebooks re-executed at
1.2.162; the freshness guard is green).  Web pins
trued 5,277 across overview/download-panel/page + census fallbacks 1089/331,356 +
`SHIPPED_TOOLKIT_VERSION` 1.2.162.

## v1.2.163 release-zip homes (2026-11-10, round 252 — the H-bond timeline: `gmx hbond -num` on a designed demo)

`pharma-ml-toolkit-v1.2.163.zip` — 23,673,301 bytes, sha-256
`89ccdad414e3ca4d73524ac77ca6845cffa6348b2ad66ccae37acbc5e1aa15ef`
(1118 entries / 1070 files, testzip OK per home, real leaks 0 — the
growth over v1.2.162's 23,669,404 B is `synthetic_hbond_trajectory`
in `md/trajectory.py` (the designed two-bond H-bond demo: two
phase-shifted D–H···A triads oscillating across the 3.5 Å distance
cutoff, hydrogens riding the donor–acceptor axis so the demo isolates
the distance gate on purpose), the bridge's `"hbond"` op, the web
H-bond timeline card over the new `/api/md/hbond` route, the
card-directory census 31 with the probe's `md-hbond` source mapping,
the CHANGELOG round-252 entry, three new MD tests, the docs INDEX
regeneration (890 symbols) and the eight notebooks re-executed at
1.2.163).  Built with the standing four-home builder recipe
(`scratch/build_zip_252.py`, predecessor v1.2.162 swept from all four
homes).  Round 252 picked the first round-250 MD follow-on candidate
(the H-bond timeline wiring) as a web+toolkit twin while the arena
conformal branch stood parked (its head unmoved for 30+ minutes, its
19-file footprint confined to ml/ + docs/ — zero overlap with this
round's md/ + scripts/ + web files).  Honest totals **5,281 = 4,426
collect + 855 doctest** (+4 over the 5,277 round-250 baseline); the
nine-chunk gate green (chunks 1+2 re-run after the ritual's INDEX
regeneration and the notebook re-execution — the freshness guard
caught the 1.2.162 prints exactly as designed); ruff zero (the
bridge's pre-existing F821 `LigandResult` annotation and the
round-250-sweep-stripped exec bit fixed en route); mypy clean across
the 10 md files.  Census **1089/331,474**.  Web pins trued 5,281 ×7 +
census fallbacks 1089/331,474 + `SHIPPED_TOOLKIT_VERSION` 1.2.163.

## v1.2.164 release-zip homes (2026-11-10, round 255 — the PLUMED-style collective variables)

`pharma-ml-toolkit-v1.2.164.zip` — 23,582,863 bytes, sha-256
`edc25e8624a7f50c1d97f868e06b32658f64f5579e31031190f222b50d530905`
(four byte-identical homes, testzip OK per home, real leaks 0 — the
growth over v1.2.163's 23,673,301 B is `md/cv.py`, the PLUMED-style
collective-variable module: the six CV actions (`cv_distance`,
`cv_angle`, `cv_dihedral`, `cv_coordination` with the honest hard-switch
subset, `cv_rmsd`, `cv_gyration`), the `CVTable` COLVAR equivalent with
its axes-must-match and finite-only contracts, and the `plumed.dat`
mini-parser with PLUMED's real 1-based semantics and honest refusals;
`tests/test_md_cv.py` at 65 tests; `docs/MD_COLLECTIVE_VARIABLES.md`;
INDEX regenerated at 899 symbols).  Notebooks re-executed at 1.2.164
(8/8, home-path-survivors=0).  Census **1093/332,850**.  Web pins trued
5,358 = 4,491 collect + 867 doctest (×7) + census fallbacks
1093/332,850 + `SHIPPED_TOOLKIT_VERSION` 1.2.164; all three version
carriers trued in one pass.
