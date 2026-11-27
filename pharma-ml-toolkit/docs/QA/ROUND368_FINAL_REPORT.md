# ROUND 368 FINAL REPORT (MASTER_PROMPT part19 — 2026-11-27)

Working branch: `round368` (from the verified `main` @ `53f2c92d`, v1.2.243). No direct changes
to `main`; no PR merged; no release published or tag moved; no force push.

## 1. Verified baseline (Phase 0 — `ROUND368_BASELINE_REPORT.md`)

`origin/main` = 53f2c92d reproduced locally with a **clean worktree**; all four version carriers
at 1.2.243; tag `v1.2.243` → 902dd89b (the built tree); release v1.2.243 assets byte-verified via
the authenticated API (EXE 558,600,414 B / MSI 711,110,134 B / manifest 510 / SBOM 23,634 /
SHA256SUMS 368); **re-measured** (not copied): pytest collection 8,274/441, vitest 357/357,
mypy Success 314 files, tsc 0, ruff canonical clean (the 70× EXE002 decoded as the sandbox-fs
mode artifact — git tree says 644, disk says 755), penetration collection 67, repo pack 942 MiB
(≈ the 987 MB claim). Session-start sandbox drift (stale git lineage + 166 local-only residue
files) recovered to the clean baseline before any work.

## 2. PR #15 reconciliation (Phase 1 — `ROUND368_PR_RECONCILIATION.md`)

The PR (open, dirty, 1 commit, 138 files, forked at r365) was classified hunk-by-hunk:
4 already-in-main / **117 unique-useful ported** / 17 outdated-doc-conflicts (ported as targeted
edits) / 0 harmful. Ported via cherry-pick with 9 honest conflict resolutions; three
port-induced contract breaks caught by the local suites and fixed at source; the stricter
`check_github_page.py` regime landed **GREEN end-to-end** (the two stale README version stamps
trued 1.2.242→1.2.243). Recommendation recorded (close as superseded once round368 lands on main);
no GitHub action taken. Gates re-verified post-port: web-parity 100 (99 pass/1 skip), the
coupled chem/explain/selfies/format/workflow suites, github-page 18/18, tsc 0, lint clean.

## 3. Legacy PySide6 retirement verification (Phase 2)

Repo-wide search classified every occurrence (65 launcher references: retirement disclosure /
historical evidence / guard tooling; zero active product launch paths). The quarantine tree
verified correct at source (the "7 deferred legacy ports" from the initial analysis dissolved
under direct verification — `_PIP_EXTRAS` and guidance already the fixed source-local form).
**The generator defect found and fixed**: `build_capability_map.py` itself still emitted the
stale "what the launch architecture actually is (verified): studio.py → ui.run()" header — now
the honest historical banner, regeneration-stable (+ the FURB188 fix). The toolkit README's
tree-diagram `ui/` entry re-aimed at `legacy/ui/`; MIGRATION_TAURI2's Phase-Sixteen row framed;
the retired Inno-Setup lane banner-marked; **production installer verified Qt-free** (the
release workflow freezes `pharmaml_backend.spec` which excludes the whole Qt stack; `gui-test`
preserved for the explicit CI legacy lane). Retirement guards **21→26** (production-workflow,
retired-lane, CI-lane, testpaths, freeze-entry pins; the README launcher guard refined to scan
EVERY occurrence with prohibition-context awareness — closing a real first-match-only gap).

## 4. API/OpenAPI/TypeScript contracts (Phase 3 — `ROUND368_API_TRACEABILITY_MATRIX.md`)

**The r367 discrepancy resolved honestly**: the runtime app re-derived live (142 paths / 148
operations) is **byte-identical** to `contracts/openapi.json`; 148/148 unique operation IDs;
`api.d.ts` regenerates identically; the "144 paths" existed only in the r367 commit-message
prose. The 148-row traceability matrix written (route file / request DTO / response DTO /
async-job-vs-sync / frontend consumers / backend tests — every operation has a consumer
reference and a test lane). Nine routes with no Studio GUI card documented with reasons
(the conformal family is the real GUI-coverage gap). Gates: contracts 6/6, frontend
drift-guard family 202 green, tests/api 753 collected.

## 5. Nine-workspace functional audit (Phase 4)

Every workspace's controls verified wired end-to-end (typed clients, real jobs, live WS
progress, working cancellation, digest-verified exports, honest sync-lane labelling); **no fake
progress, hardcoded results, placeholder buttons, or TODO markers anywhere in production
source**. JobManager/event architecture verified (cooperative cancellation incl. the honest
`cancellation:"unsupported"` record, SQLite persistence, bounded replay + duplicate protection +
since-cursor + policy close codes; REST-authoritative tracker reconciliation). **F-1 fixed**
(three Datasets chemistry buttons now carry in-flight guards); **F-3 fixed** (the StageHeader
dead icon prop now renders); F-2/F-4/F-5/F-6 classified (by-design/guard/test-only/defaults).
Gates after the fixes: tsc 0, vitest 357/357.

## 6. Scientific validation (Phase 5 — `ROUND368_SCIENTIFIC_VALIDATION_REPORT.md`)

Task-compatibility **zero mismatches** (33 registry families, cross-task builds refused live);
SpectralEmbedding never conflated with SpectralClustering (cross-documented + constructor-disjoint);
the DL honesty constraints intact (transformer trained-from-scratch, CNN 2-D refusal
live-verified, autoencoder heuristic-anomaly + lane refusal, GNN classifier-only, zero
GNN-regression claims); feature-space GAN/DDPM `molecular:false` separation store+route+test
pinned; RL real NumPy implementations with the §6 registry separation test-enforced;
multiresponse 12 strategies + per-target train-only scalers + RSM ANOVA/LoF/desirability all
executed green (194-test focused set). **New: `test_ml_catalog.py`** (6 pins — the 924-project
catalogue previously doctest-only). The honest-limitation register carried in full.

## 7. Resource management (Phase 6)

`n_jobs` never 0 at three layers (pydantic `ge=-1,le=64` default 1; route `==0→1` coercion;
`compute.py` hard refusal); the memory advisory clamps **only parallelism** (training unchanged)
with 4× safety factor, passes verbatim when memory is plenty, best-effort without psutil —
clamp/no-clamp/two-sequential/two-concurrent all test-pinned; worker-death translation typed +
actionable at library/sync-422/job-lane levels; **no pause/resume claims anywhere**; DL trains
in-process loops (no DataLoader workers); JobManager ThreadPool(4)+semaphore(2)+graceful drain.
219-test reliability set green.

## 8. Security / penetration (Phase 7 — `PENETRATION_TEST_REPORT.md` §6)

The 67-test API security suite **executed green locally**. The ZAP campaign honestly NOT
complete: the weekly Monday 04:30 UTC slot did not fire during this Friday round (next:
2026-11-30); tooling pins stand ready; the web-console scan limitation documented;
security-scan lane recorded as historical last-green.

## 9. Windows clean-machine validation + installer size (Phases 8–9 —
`WINDOWS_CLEAN_INSTALL_REPORT.md`)

The **reproducible PowerShell suite delivered** (`scripts/windows/validate-clean-machine.ps1`):
census / installer integrity (honest Authenticode BLOCKED-while-certificate-gated) / the live
sidecar handshake-health-guard-shutdown leg / the nine-workspace API matrix over synthetic data /
artifact export + backup-restore / the upgrade leg. Runner verification is never conflated with
clean-machine verification: the lane stays **NOT_RUN until the owner executes the suite on a
clean box**. Installer sizes investigated and **justified** (torch-CPU + full offline stack by
the no-silent-omission rule; Qt EXCLUDED from the payload by spec+guard). Signing stays
**BLOCKED — certificate-gated** (nothing fabricated or committed; SHA256SUMS/manifest/SBOM
preserved).

## 10. Repository hygiene + branch protection (Phase 10 — recommendations only)

The tracked tree is 1.31 GiB / 3,472 files, of which **~1.19 GiB is versioned ZIP archives
duplicated across the four in-tree homes** (download 343 MiB / 270 files; toolkit dist-archive
321 MiB / 20; upload 264 MiB / 47; toolkit download 240 MiB / 149; download_archive 22 MiB;
each release's zip × 3–4 copies). Per-item register (path/size/purpose/runtime/test/provenance/
stay-in-git/recommended home) in §10 of the hygiene table below. **Recommendation (owner
decision — no action taken):** move future release zips to GitHub release assets only, keep the
latest one per home locally, and prune pre-N versions from the tracked homes — with explicit
authorization and a full backup, never a history rewrite. INFORMATIONAL: build-venv carries the
Qt stack (build-time only); the SBOM describes the build stack rather than the payload;
`pharmaml.egg-info` is untracked build residue (clean). **Branch-protection recommendation
(no settings changed, per the owner-authorization rule):** required CI (the ci.yml checks) on
main; forbid force pushes; restrict tag-push rights to the release workflow's identity; one
required review for non-owner branches; `windows-release.yml` restricted to tags only;
Actions permissions least-privileged + secret scanning + dependency review already enforced by
the security-scan lane.

## 11. Documentation consistency (Phase 11)

The r367→r368 doc set reconciled by the port + the Phase 2 fixes: root README (checker-verified),
inner README (constrained extras + retirement wording — checker-verified), FEATURES/TEST_PROFILES
(archival relabels merged with the re-aimed paths + 1.2.243 stamps), CAPABILITY_MAP
(generator-backed historical banner), PROJECT_STATUS (archival banner), MIGRATION_TAURI2
(historical framing), RELEASE_READINESS (the r368 section), the penetration report (§6), the
QA ledger + worklog entries below. Status vocabulary honest throughout (the shipped-revision
census pins remain 8,274/357/1,873/601,763 — the working-tree collection is 8,286/443 after
this round's +12 tests; the pins describe the shipped v1.2.243 revision, and the ship ritual
re-trues them at the next version).

## 12. Tests actually executed this round (Phase 12)

| Gate | Result |
|---|---|
| pytest chunk 1 (a–l) | green (3m13s; the 2 generative failures found were stale pins, fixed + re-verified) |
| pytest chunk 2 (m–p) | green (35s) |
| pytest chunk 3 (r–z + q + x) | green (4m59s; the checkpoint-tooling guard re-aimed; release-portability sub-run green) |
| pytest chunk 6 (src doctests) | green (1m33s) |
| tests/api family (753) | green (3m) |
| q/x glob orphans (61) | green |
| legacy GUI sample lanes (62, offscreen) | green — pages_smoke + phase4 + qt_doctor + nine_cards + phase5 |
| vitest (toolkit frontend) | **357/357 over 41 files** |
| tsc (root console + toolkit frontend) | exit 0 both |
| mypy src/pharmaml | Success, 314 files |
| ruff check src/pharmaml (canonical CI lane) | clean (EXE002 = documented sandbox-fs artifact) |
| tests/test_legacy_retirement.py | **26/26** (21 + 5 new r368 guards) |
| tests/test_ml_catalog.py | **6/6 new** |
| penetration suite | 67 green |
| check_github_page.py | **validation passed** |
| bun run lint (root) | clean |
| browser E2E (web console) | **GREEN — the r366 harness wedge has healed**: nav + Download panel + Playground rendered, the seven constrained extra chips live, overview pins painting, zero console errors, desktop + 390×844 mobile screenshots saved (`scratch/r368-console-*.png`) |
| render probe | GET / 200 (0.23 s warm); zip-size API exact 28,052,601 B @ 1.2.243 |

**Tests not executed and why:** the full 972-test legacy GUI lane (the r367 precedent — a
62-test sample ran green; the CI's explicit offscreen job carries the full lane); the ZAP scan
(Monday slot); Windows-only legs (Linux sandbox — the PowerShell suite is the reproducible
path); the full local lint lane over tools/scripts (node_modules janitor; the CI lint lane
carries the gate — the standing register).

## 13. Defect register (Round 368)

| ID | Description | Severity | Status |
|---|---|---|---|
| D-368-01 | FURB188 in `scripts/build_capability_map.py` (conditional slice vs removeprefix) | LOW | **FIXED** |
| D-368-02 | Sandbox git drift: stale snapshot lineage + 166 local-only residue files (pre-quarantine copies) | MEDIUM | **FIXED** (baseline recovered; worktree clean) |
| D-368-03 | Root README's two stale version stamps (1.2.242 vs the 1.2.243 carriers) | LOW | **FIXED** |
| D-368-04 | Toolkit README tree diagram listed a root `ui/` entry (the quarantine home is `legacy/ui/`) | LOW | **FIXED** |
| D-368-05 | MIGRATION_TAURI2's Phase-Sixteen row read as present-tense "PySide6 remains the production Studio" | LOW | **FIXED** (recorded-baseline framing + executed note) |
| D-368-06 | The best-of-ML catalogue had no non-doctest regression coverage | MEDIUM | **FIXED** (test_ml_catalog.py, 6 pins) |
| D-368-07 | The capability-map GENERATOR emitted the stale pre-retirement "verified" header (regeneration stomps honest banners) | MEDIUM | **FIXED** (generator banner; doc regenerated) |
| D-368-08 | DatasetsPage: three chemistry buttons without in-flight guards (double-click could duplicate job submissions) | MEDIUM | **FIXED** (busy guards + labels, the established pattern) |
| D-368-09 | StageHeader's `icon` prop silently discarded (`{icon ? null : null}`) in two pages | LOW | **FIXED** (rendered in muted foreground) |
| D-368-10 | Port-induced contract breaks ×3 (session-report's pinned ORDER-confession comment deleted; probe r30a's confession string; responseai-card's "stay desktop-side" disclosure) | MEDIUM | **FIXED** (restored/re-pinned; the web-parity + probe suites green) |
| D-368-11 | Six stale test pins vs the ported source-local guidance (test_generative ×2 + RDKit case; test_neural ×2; the qt_doctor guard contract) | MEDIUM | **FIXED** (re-aimed to the retirement-aware contracts; all suites green) |
| D-368-12 | The five r367 DL modules' `_TORCH_HINT` still carried the broken PyPI-style guidance (the port predated them) | MEDIUM | **FIXED** (source-local form) |
| D-368-13 | The §4.4 drift guard pinned the retired UI's "eighteen pages" enumeration | LOW | **FIXED** (re-aimed to the nine-workspace scope + stale-count prohibition) |
| D-368-14 | The ten-shard runner missed the entire tests/api family (753) + test_q* + test_x* (61) | HIGH (gate coverage) | **FIXED** (chunk 3 widened; all green) |
| D-368-15 | ~1.19 GiB of duplicated versioned zips across the four tracked homes | INFORMATIONAL | **REGISTERED** (owner-authorization recommendation; no destructive action) |
| D-368-16 | Build-venv carries the Qt stack (payload excludes it; build-time only) | INFORMATIONAL | **REGISTERED** |
| D-368-17 | The SBOM describes the build stack rather than the shipped payload composition | LOW | **REGISTERED** (re-trued at ship time by the standing ritual) |

## 14. Remaining blockers (unchanged or clarified)

1. **Installer signing — BLOCKED (certificate-gated)** — unchanged, honestly recorded.
2. **Clean-machine validation — NOT_RUN** — the reproducible suite now exists; the owner
   executes it on a clean Windows box.
3. **ZAP campaign — scheduled** (Monday 04:30 UTC slot; not fired during this round).
4. **The browser-E2E constraint has HEALED** (this round's green run above) — the standing
   r366 note is retired.

## 15. Engineering handoff

* Branch `round368` (10 commits ahead of `53f2c92d`); files changed: ~370 (the PR15 port's 134 +
  the six r368 reports + guards + fixes detailed above).
* Fixes implemented: D-368-01…D-368-14 (see the register); D-368-15…17 registered with
  recommendations.
* Windows-specific steps requiring the owner: execute
  `scripts/windows/validate-clean-machine.ps1` on a clean box; action the PR #15
  close-as-superseded decision; action the branch-protection + zip-home recommendations;
  provide the signing certificate through an authorized channel when available.
* **The work is ready for review.**
