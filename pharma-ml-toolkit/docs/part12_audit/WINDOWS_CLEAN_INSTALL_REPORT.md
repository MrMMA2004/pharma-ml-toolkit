# Windows Clean-Install Report (part12 PART 16 mandatory gate)

Audited: 2026-11-20, this environment. **Verdict: BLOCKED** — honestly and
for the exact reason part12 anticipates: "If a genuine clean Windows
environment is unavailable, report the gate as BLOCKED. Do not claim it
passed based on the GitHub build runner alone."

## Why BLOCKED

This session runs on a Linux sandbox with no Windows host, no clean
Windows 10/11 x64 VM, and no physical machine. The GitHub Actions
windows-2022 runner is NOT a clean machine: it carries preinstalled
toolchains, cached package managers, warm MSVC/WebView2 states and a
runner-maintained filesystem — the same closing note the release run
itself makes.

## What IS verified (runner evidence, live this session)

- The v1.2.231 release exists on the GitHub API (published, not draft)
  with real assets: `PharmaML_Studio_Setup_v1.2.231.exe` (557,655,583 B),
  `PharmaML_Studio_v1.2.231.msi` (710,122,853 B, offline WebView2),
  `RELEASE_MANIFEST.json`, `SHA256SUMS.txt`.
- Build-time runtime probes (release run 44693818307, all 26 steps green):
  frozen sidecar `PHARMAML-READY` handshake, authenticated health probe,
  guard refusing unauthenticated probes, graceful CTRL_BREAK shutdown
  (exit 0) — executed on the runner, not a clean machine.
- Static packaging validation: installer artifacts + contents, 26/26.

## The 24-step clean-machine ladder (status per row)

| # | Step | Status |
| --- | --- | --- |
| 1 | EXE installation | BLOCKED |
| 2 | MSI installation | BLOCKED |
| 3 | Offline WebView2 setup | BLOCKED |
| 4 | Startup without separately installed Python | BLOCKED (sidecar carries its own Python; unverified on clean HW) |
| 5 | Startup without Node.js or Rust | BLOCKED |
| 6 | Backend health | BLOCKED (runner-verified only) |
| 7 | REST communication | BLOCKED (runner-verified only) |
| 8 | WebSocket communication | BLOCKED (runner-verified only) |
| 9 | SQLite persistence | BLOCKED (67 tests green in CI; clean HW unverified) |
| 10 | Dataset import | BLOCKED |
| 11 | Chemistry calculations | BLOCKED |
| 12 | ML/DL execution | BLOCKED |
| 13 | Semi-supervised learning | BLOCKED |
| 14 | Applicability Domain | BLOCKED |
| 15 | Plotly and Three.js | BLOCKED |
| 16 | Docking | BLOCKED |
| 17 | PK/PD | BLOCKED |
| 18 | MD analysis | BLOCKED |
| 19 | Model save/load | BLOCKED |
| 20 | Figure export | BLOCKED |
| 21 | Backup/restore | BLOCKED |
| 22 | Upgrade | BLOCKED |
| 23 | User-data preservation | BLOCKED |
| 24 | Uninstallation | BLOCKED |

All 24 rows share one root cause: **no clean Windows machine exists in
this environment.** None is individually failed — the artifacts and the
in-app logic are runner-verified; the clean-machine execution lane is the
missing evidence.

## What unblocks it (the standing protocol)

A genuine Windows 10/11 x64 machine (physical or fresh VM, no dev tools)
with the two installers downloaded from the release: install EXE → run
the 24-row smoke (the app's own doctor + the nine-workspace one-action
pass) → uninstall → reinstall via MSI → upgrade from v1.2.230 → verify
user-data preservation → record evidence here and in the Blocker Register
(C-3/C-5 dissolve; §47 rows 38–48 fill; the §28 Stage-5 PySide6
retirement gate opens behind it).

Until then, every round's honest close repeats this register's line:
**clean-machine smoke test — BLOCKED; the runner is not a clean machine.**
