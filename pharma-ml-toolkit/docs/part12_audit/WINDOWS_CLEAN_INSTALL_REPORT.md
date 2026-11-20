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

---

## part13 Phase 2 — the in-tree static verification (round 309)

Part13 Phase 2's six items each got their executable-in-tree leg this
round. Nothing below converts absence of evidence into a pass: every
row either names its in-tree evidence or stays BLOCKED on the machine.

| Phase-2 item | Status | In-tree evidence (this round, static/config-level) |
| --- | --- | --- |
| Clean Windows installation | **BLOCKED** | Unchanged — no genuine Windows 10/11 x64 clean machine exists in this environment (part13 §4 Blocker A's own rule). The 28-row ladder's machine legs remain the missing evidence. |
| Offline runtime | **PARTIAL — config-verified, machine-unverified** | `src-tauri/tauri.conf.json` `webviewInstallMode: offlineInstaller` (the NSIS package embeds the WebView2 Evergreen OFFLINE installer — workflow step 16-19's own comment + the MSI's 710 MB offline-WebView2 size confirm); the backend sidecar is a PyInstaller **onedir** freeze (`packaging/windows/pharmaml_backend.spec`) carrying its own CPython (runner-verified `PHARMAML-READY` handshake — startup without separately installed Python by construction); `frontendDist` compiles the SPA into the host (no Node.js at runtime); the host is a compiled Rust binary (no Rust toolchain at runtime); the CSP (`connect-src ipc: http://ipc.localhost http://127.0.0.1:* ws://127.0.0.1:*`) pins the runtime network surface to the local loopback. The clean-HW execution rows stay BLOCKED. |
| Upgrade / uninstall | **PARTIAL — design-verified, machine-unverified** | NSIS `installMode: currentUser` (per-user install, no admin elevation requirement); the WiX MSI is the per-machine complement; upgrade = install-over semantics with Tauri's standard GUID-identity upgrade handling; the workflow's own honest closing note declares "BLOCKED: upgrade + uninstall behavior (user data retention)" — this round adds the design evidence, not a machine pass. |
| User-data preservation | **PARTIAL — design-verified, machine-unverified** | `src/pharmaml/api/persistence.py::default_data_dir()` resolves to `%LOCALAPPDATA%\PharmaML Studio` on Windows — the data home is OUTSIDE the install directory, so uninstall (which removes only the install dir) and upgrade-by-install-over preserve user data by design; the backup/restore surface (`/database` routes, §16/§35) carries 67 tests green in CI. The machine leg (install → use → uninstall → reinstall → data survives) stays BLOCKED. |
| PySide6 retirement decision | **DECISION: stays gated — documented, not executed** | Blocker B's own rule: retire only after feature parity + clean-machine validation pass. The parity evidence today: CAPABILITY_INTEGRATION_MATRIX 73 React rows (65 PASS / 6 PARTIAL / 0 FAIL / 3 N/A — §3.5: the legacy registry's 62 rows + 873 tests remain the preserved reference frontend). Retirement does NOT proceed this phase; the decision itself is now recorded (this row + PROJECT_STATUS §4) instead of implicit. |
| Digital signing assessment | **DELIVERED — gate BLOCKED** | The Blocker C evaluation now exists as `docs/security/INSTALLER_SIGNING_ASSESSMENT.md`: the exact certificate requirement (Authenticode OV/EV code-signing cert for the publishing identity), RFC 3161 timestamping (SHA-256 dual digest), GitHub Actions secrets custody (PFX + passphrase as separate secrets, never the repo), `signtool verify /pa /all` verification, provenance chaining into the existing SHA256SUMS/RELEASE_MANIFEST/SBOM spine, and the staged conditional workflow lane. No certificate exists — the gate stays BLOCKED with the requirement specified to the day it arrives. |

**Phase-2 verdict: no lane opened, no lane silently closed.** The
machine-gated rows keep their single root cause (no clean Windows
machine); the certificate-gated lane keeps its own (no certificate).
What this round added is the executable evidence that was missing
around them: the offline-runtime configuration audit, the
data-directory design proof, the PySide6 decision record, and the
signing assessment deliverable.
