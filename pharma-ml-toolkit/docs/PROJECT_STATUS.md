# PROJECT_STATUS — the single authoritative status document

Part13 §2 deliverable ("Create a single authoritative project-status
document with the actual current state"). Replaces no audit; it
summarises them and wins any status disagreement.

- **Audit date**: 2026-11-21 (round 307, the 98th recorded run;
  Phase-2 addendum round 309 — the interactive part13 continuation).
- **Audited base**: `origin/main` = `5446e89b` (the r306 close-out);
  Phase-2 addendum on `19effec8` (the r308 close-out) — this document
  lands with the round-309 commit.
- **Scope**: whole repository — source, tests, docs, release
  artifacts, web pins, archive ritual.
- **Status labels** (part13 §22): PASS / PARTIAL / FAIL / BLOCKED /
  NOT APPLICABLE. Nothing here converts absence of evidence into a
  pass; every row names its evidence.

## 1. Where to look for what

- **This file** = the authoritative current-state summary. Where a
  per-audit doc (`docs/part12_audit/*`) disagrees on status, this file
  wins (their stale rows were trued at the r307 reconciliation; the
  audits' original findings are preserved as history in git).
- Release gates: `docs/RELEASE_READINESS.md` (generated + addenda).
- Integration tiers: `docs/CAPABILITY_INTEGRATION_MATRIX.md` +
  `docs/part12_audit/BACKEND_GUI_INTEGRATION_MATRIX.md`.
- Per-round engineering ledger: `CHANGELOG.md` + repo-root `worklog.md`
  + `PROJECT_AUDIT_PROGRESS.md`.

## 2. Verified current state (evidence gathered this round)

| Item | Value | Evidence (this session) |
| --- | --- | --- |
| Repository | public, default branch `main` | GitHub API, authenticated |
| HEAD | r307 audited at `5446e89b` = `origin/main`; the Phase-2 addendum verified `19effec8` (the r308 close-out) clean, and this document's addendum lands with the round-309 commit | fetch + reset --hard + ghost sweep (r307/r308) |
| CI at HEAD | green — 19/19 check-runs completed/success at `19effec8` (the r308 push); r307's evidence: CI + Push both success at `5446e89b` | `check-runs?head_sha=19effec8` (r308 poll, final) |
| Published release | v1.2.231 (not draft): EXE 557,655,583 B · MSI 710,122,853 B · RELEASE_MANIFEST.json · SHA256SUMS.txt | `releases/tags/v1.2.231` |
| Version carriers | VERSION · pyproject · `pharmaml.__version__` · TS carrier · CHANGELOG · tauri.conf.json — all 1.2.231 | read directly this session |
| Test surface | 8,362 collected / 0 errors / 456 files (`tests/` + `src/pharmaml` `--doctest-modules`) | this round's fresh collection on the repaired venv |
| Census | 1,593 files / 553,185 lines / v1.2.231 (the round-309 final walk with the empty-file true-up — equals the live /api/toolkit/tree walk; doc-only delta since r307 — zero src/tests bytes) | this round's fresh census walk |
| Source archive | v1.2.231 zip same-version re-ship cadence: the r307 close stood at 15,413,809 B sha `2c4de09d`; the round-309 re-ship supersedes it (same version — §5; the per-round sha/bytes live in the CHANGELOG/worklog per §7, never here — the archive cannot carry its own hash) | sha256 across five homes + offsite |
| venv pins | numpy 1.26.4 · torch 2.14.1+cpu · rdkit 2026.09.1 · sklearn 1.5.2 · pandas 2.2.3 · pharmaml 1.2.231 — zero drift vs `requirements.txt` | import probes post-repair |

## 3. The six previously-open audit items — CLOSED (re-verified in-tree this round)

1. **Standalone Leverage/Williams AD** — CLOSED r303:
   `ml/applicability.py::leverage_williams` (hat-matrix leverage,
   h\* = 3(p+1)/n, standardized-residual band, regression-only guards)
   + `plot_williams` + `/api/v1/applicability-domain/leverage-williams`
   (job/sync + figure artifact + the four-entry catalog) + the ML
   workspace's Williams lane; 23 unit + 3 figure + 7 route tests.
2. **Fingerprint chirality configuration** — CLOSED r303:
   `include_chirality` through `compute_fingerprints` /
   `FeatureConfig` / `/api/v1/chemistry/fingerprints` + the Datasets
   card (morgan-only, default OFF — byte-identical history);
   8 descriptor + 8 factory + 3 route tests.
3. **Semi-supervised `/evaluate` GUI integration** — CLOSED r303: the
   GUI consumer of `POST /semi-supervised/evaluate` landed.
4. **Semi-supervised job-mode argument correction** — CLOSED r303: the
   route's job mode passed a spurious `manager` kwarg (every job
   submission raised TypeError while sync masked it); fixed and pinned
   (the COMPLETION pin rides RELEASE_READINESS).
5. **Tk playground retirement** — CLOSED r304: `playground.py`
   (30,047 lines) + `tests/test_playground.py` (16,370 lines) left the
   tree; the v1.2.141 declaration executed; the zip builder's NEGATIVE
   gate pins the pair's absence; collection re-trued 8,643 → 8,362.
6. **SBOM generation + release attachment** — CLOSED r304:
   `packaging/sbom/sbom.cdx.json` (CycloneDX 1.6, 40 components,
   validated + reproducible) ships in the archive; the Windows
   workflow's step 22b regenerates it on the runner and self-registers
   its hash into SHA256SUMS.txt.

## 4. The three honest BLOCKED lanes (unchanged gates; Phase 2 executed around them)

- **C-3/C-5 clean-machine smoke test** — a genuine Windows 10/11 x64
  machine (physical or fresh VM, no dev tools) is required; a GitHub
  Windows runner is not a clean machine. BLOCKED.
- **Stage-5 PySide6 retirement** — gated behind the clean-machine pass
  + demonstrated feature parity (the rule the campaign set). BLOCKED.
  Phase-2 decision record (round 309): retirement does NOT proceed this
  phase — the parity ledger (§3.5: 65/73 PASS React rows, 6 PARTIAL)
  plus the machine gate both stand; the decision is now explicit, not
  implicit.
- **Installer digital signing** — certificate-gated (Authenticode +
  timestamping; secrets in GitHub Actions, never in the repository).
  BLOCKED. Phase-2 deliverable: the exact requirement + staged lane now
  documented in `docs/security/INSTALLER_SIGNING_ASSESSMENT.md` (round
  309) — the gate opens the day the certificate exists in Actions
  secrets; nothing about the current unsigned state was embellished.

## 5. Release posture (part13 §5)

- The published v1.2.231 binaries were built from an earlier commit
  than `main` — do not assume they include later source changes.
- Source-tree changes re-ship the archive at the SAME version (the
  r303/r304 same-version convention; superseding shas recorded in the
  CHANGELOG/worklog per the r301 three-sha convention). The published
  Windows assets are never overwritten by source re-ships.
- A NEW version (v1.2.232+) ships only after the applicable release
  gates pass (part13 Phase 7) — which requires the BLOCKED lanes to
  clear first. No silent version bumps, ever.

## 6. Next-campaign priorities (part13 Phases 2–7, in order)

- **Phase 2** — EXECUTED (round 309, the interactive continuation):
  the six items each got their in-tree leg (the offline-runtime config
  audit — offlineInstaller WebView2 + frozen-own-Python sidecar +
  compiled host; the upgrade/uninstall + user-data design proof —
  `%LOCALAPPDATA%` data home outside the install dir + 67-test
  backup/restore surface; the PySide6 decision record; the signing
  assessment deliverable). The machine/certificate legs stay honestly
  BLOCKED — full evidence table in
  `docs/part12_audit/WINDOWS_CLEAN_INSTALL_REPORT.md` § "part13
  Phase 2". **Phase 3 is next.**
- **Phase 3** — frontend quality: component tests, E2E workflows,
  visual regression (Vitest / React Testing Library / Playwright
  lanes), error/empty/loading state coverage.
- **Phase 4** — scientific workflow improvements: Active Learning,
  Chemical Space Explorer, Conformal Prediction audit, Automated
  Benchmarking, Report Builder, Model Cards.
- **Phase 5** — advanced pharmaceutical AI: GNN/MPNN GUI completion,
  ADMET, Drug–Target Interaction, Drug Repurposing, Cancer Cell-Line
  Drug Response, Pharmacogenomics, multi-objective molecular
  optimization.
- **Phase 6** — architecture hardening: the lightweight workflow
  engine evaluation (JobManager/capability-registry first),
  resource monitoring, security scanning lanes, documentation
  automation.
- **Phase 7** — the next production release, from the exact release
  commit, with all gates recorded.

## 7. Known limitations of this document

- Census/test numbers are as-of this round's runs; each round's ledger
  entries re-true them (the pins live in the web carrier
  `toolkit-stats.ts` + the live `/api/toolkit/tree` walk).
- The r307 same-version re-ship's sha/bytes are recorded in the
  CHANGELOG + worklog, not here — those move with each re-ship; this
  file records lanes and their statuses, not per-round bytes.
