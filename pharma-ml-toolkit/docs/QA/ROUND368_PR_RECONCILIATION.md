# ROUND 368 PR RECONCILIATION — PR #15 "Clarify toolkit parity and verification claims" (part19, Phase 1)

## 1. Verified PR state (GitHub API, this session)

| Field | Value |
|---|---|
| State | **open**, `mergeable_state: dirty`, not draft |
| Author | MrMMA2004 |
| Created / updated | 2026-11-26T23:29:21Z / 23:31:38Z (minutes before the r367 ship completion) |
| Base | `main` @ `902dd89b` (pre-53f2c92d) |
| Head | `arena/3eb4a9a6-pharma-ml-toolkit` @ `c1e3b1cd` |
| Commits | 1 — "docs: clarify toolkit boundaries and verification claims" |
| Diff vs main | 138 files, +1,872 / −2,015 |
| Fork point | merge-base with main = `762f60a5` (r365 penetration close) — the PR predates r366/r367 entirely |

## 2. Complete diff reconciliation (every file classified)

The single commit was classified hunk-by-hunk against main's current tree:

| Class | Files | Disposition |
|---|---|---|
| Already in main | 4 (`feature-factory.tsx`, `ligand-prep.tsx`, `vina-folder-scan.tsx`, `batch-configs.tsx` — r366/r367 already made the behavioral corrections; only comment cosmetics remained) | Not ported (took main's side in the 3-way conflicts) |
| Unique + useful | 117 | **Ported** (see §3) |
| Outdated / conflicting documentation | 17 (root README, inner README wholesale form, 15 example notebooks) | Not ported wholesale; the still-valuable pieces were applied as targeted edits to main's current content |
| Harmful / regressive | 0 as diffs stand (wholesale-swap of the two READMEs or `download-panel.tsx` would have regressed main's r366/r367 content — avoided) | n/a |

PR's own validation claims vs current evidence: its targeted suite claims (249 passed / 17 skipped / 18 github-page tests) were environment-scoped; the port re-verified everything against main's live tree (§4). The PR body's caveat "portable build blocked by Google Fonts fetch" was not re-tested (not needed for a docs port).

## 3. What was ported (mechanism and content)

Mechanism: `git cherry-pick c1e3b1cd` onto `round368` (the working branch), followed by honest conflict resolution — no wholesale swaps; main's newer content preserved wherever it was more current.

Ported content classes (117 files + 17 partial):
1. **Stricter GitHub-page regime**: `scripts/check_github_page.py` (version-carrier pins, no-PyPI claim, React+Tauri interface claims, retired-launcher command guard, fenced-code-aware link checking, inner-manual constrained-extras checks, download-panel boundary checks, RESEARCH_WORKFLOWS archival-label checks) + `tests/test_github_page.py` (18 tests).
2. **DependencyError/guidance message fixes across 34 source files** (chem ×11, cli, docking ×4, generative ×4, ml ×7, studio/pipelines, viz ×5): the broken `pip install 'pharmaml[extra]'`-style guidance (the package is not on PyPI) now says `python -m pip install -e '.[extra]'` from the package root; 5 coupled test files re-pinned to the new messages; `test_workflow_pipelines.py` gained 2 new tests (known-dep guidance + unknown-dep must not guess a pip command).
3. **Web-console honesty sweep** (55 files): de-twinning of ~120 "web twin of the desktop Studio" claims, bounded parity language, GP/LML-divergence and fold-leakage disclosures in the API routes, `overview.tsx` Testing row now says **collected at the shipped revision (collection count, not a live run)**, `ml-playground.tsx` "largest fold train size" metric label fix, `download-panel.tsx` boundary honesty (drops the retired `.[studio]` chip, constrained chip commands, "recorded cases", links to Releases + the frontend guide).
4. **Docs banners/labels** (17 files): archival/historical banners on the retired-UI documentation, `CAPABILITY_MAP.md`'s "studio.py → ui.run()" claim relabeled historical, `PROJECT_STATUS.md` self-deprecation of its "single authoritative" title, `FEATURES.md`/`TEST_PROFILES.md` retired-interface relabels (merged with main's re-aimed `legacy/` paths and 1.2.243 stamps), `RESEARCH_WORKFLOWS.md` archival labeling of the PySide6 walkthroughs.
5. **Tooling/packaging honesty**: `eslint.config.mjs` ignores `**/.venv/**` + `**/venv/**`; `Dockerfile` pinned-core `requirements.txt` install with `--no-deps` package and `-c`-constrained extras; `pyproject.toml` description corrected to "Research-oriented Python toolkit…" (main's `version = "1.2.243"` kept — the PR's 1.2.242 stamp was stale); hero SVG alt-text honesty; `build_social_preview.py` ILLUSTRATIVE label; CONTRIBUTING Studio-dev section fixed (the old `QT_QPA_PLATFORM=offscreen python studio.py` workflow removed); social preview regenerated with the ILLUSTRATIVE label (1280×640).
6. **Root README**: NOT swapped wholesale (the PR's copy was a v1.2.242/r311-era rewrite that would have deleted main's quickstart/interface table); main's current README already satisfied nearly the whole new checker regime — only its two stale version stamps (`1.2.242`) needed truing to `1.2.243`.

Conflict resolutions (9 files): pyproject (keep 1.2.243 + new description), FEATURES.md and TEST_PROFILES.md (main's re-aimed `legacy/` paths + pr15's archival labels), download-panel.tsx (main's r367 structure + pr15's boundary honesty + a factual root-files note verified against the actual v1.2.243 zip root), overview.tsx (main's detailed Interface row; pr15's Testing-row fix auto-merged), and the four already-in-main playground components (main's side).

Three port-induced contract breaks found and fixed at source during verification:
- `src/lib/session-report.ts` — the PR's comment rewrite deleted the pinned ORDER-confession contract comment ("nothing remembered: stay quiet"); restored the documented contract block (kept the PR's header rewrite).
- `scratch/probe_session_report_r30a.ts` — the ligand-table confession row string re-pinned to the new "web console's Ligand Sets tab" wording (the probe pins the source contract; the source's new wording is the accurate one for the web console).
- `src/components/explorer/playground/responseai-card.tsx` — restored the "deeper diagnostics … stay desktop-side" disclosure the ported test still pins (the PR's own tree was self-inconsistent here: its test asserted a phrase its rewritten card no longer contained).

## 4. Port verification (all executed this round on the resolved tree)

| Gate | Result |
|---|---|
| `pytest tests/test_web_parity.py` | **100 collected — 99 passed, 1 skipped, 0 failed** (probe lanes re-run green after the fixes; the one skip is the standing web-checkout-scoped marker) |
| Coupled toolkit suites (`test_chem_standardize`, `test_explain`, `test_format_converters`, `test_selfies_codec`, `test_workflow_pipelines`) | all green with the ported message guidance |
| `pytest tests/test_github_page.py` | **18 passed** |
| `python scripts/check_github_page.py` | **"GitHub page validation passed"** — release carriers 1.2.243, capture 1440×900 present, social preview 1280×640, 57 local links across 7 docs, no retired-launcher command anywhere in the console cards |
| `tsc --noEmit` (root console) | exit 0 |
| `bun run lint` (root) | clean |
| Frontend vitest (toolkit frontend, unchanged by the port) | 357/357 (Phase 0 measurement) |

## 5. Not ported (deferred register)

1. **15 example notebooks' guidance-cell wording** (`pip install -e ".[reports]"` → `python -m pip install -e '.[reports]'` + "from the package root"): main's notebooks were re-executed at 1.2.243 (r367); re-applying wording to re-executed copies belongs to the next ship cycle's notebook re-execution step. Deferred deliberately — not a correctness debt in this round.
2. **7 legacy-tree honesty ports** (`legacy/ui/workers/job_worker.py`'s wrong `_PIP_EXTRAS` table — `meeko→chem` should be `docking`, `torch→deep-learning` nonexistent extra; `legacy/ui/qt_import_guard.py`; `legacy/ui/pages/ml_workbench.py` + `utilities.py` broken guidance lines; coupled `legacy/tests_ui/test_v204_review_fixes.py` + `test_qt_doctor.py`): carried into **Phase 2** of this round (legacy retirement verification) — the quarantine tree is where those fixes belong.
3. The PR's wholesale root/inner README rewrites — superseded by main's newer structure plus the targeted truing described above.

## 6. Recommendation (no GitHub action taken — owner authorization required)

**Recommendation: close as superseded after `round368` lands on `main`.** Rationale:
- The branch forked from r365 and its head still carries the pre-quarantine `ui/` tree at dead paths; a rebase would replay the whole quarantine conflict surface, and its single commit's content is (after this round's port) fully present on `round368` in trued form.
- The PR's unique value was the honesty/wording audit — preserved and verified — not its branch topology.
- Per the round's work-safety rules, no GitHub administration (merge/close/rebase of PR #15) was performed in this session; the recommendation stands for the owner to action.
