# BLOCKED TESTS AND SOLUTIONS — part14 §13 deliverable #7

The consolidated blocked-test ledger: every blocked leg, why, the
required environment, and the lowest-cost unlock. Nothing blocked is
ever reported as PASS; nothing unblocks silently.

Seeded: round 329 (2026-11-22).

## Blocker classes

| ID | Blocker | Blocks | Required environment | Lowest-cost solution |
| --- | --- | --- | --- | --- |
| BLK-1 | No Windows in the dev sandbox | the §6.1 desktop checklist + the clean-machine smoke's DESKTOP half (the GUI nine-workspace click-through, the offline-network WebView2 bootstrap, sleep/resume, DPI/multi-monitor, the 45–60-min desktop-stability lane, GPU benchmarks, consumer-Win10/11 fidelity) — the smoke's INSTALLER-MECHANICS half (install/upgrade/uninstall, the packaged-app process lifecycle, the installed sidecar's live contract, antivirus) is CI-EXECUTED since r336 (see the section below) | physical Windows 10/11 (VM acceptable for most legs per §9.5) | the r336 CI lane for the installer mechanics; the §9.5 clean-VM procedure for the remaining desktop legs |
| BLK-2 | No Rust toolchain (cargo/rustc) in the sandbox | the D-4/D-5 `#[cfg(test)]` suites (11 malformed-handshake cases); `tauri build` | any cargo-enabled environment | install the Rust toolchain anywhere — the unit legs do NOT need Windows |
| BLK-3 | Signing certificate | signed-EXE/MSI validation | the certificate lane (owner-side) | r325 standing lane #3; rides the owner's cert decision |
| BLK-4 | Stage-5 PySide6 retirement approval | the legacy-app retirement gate | owner decision + the box | the r325 standing lane #2 |
| BLK-5 | ECC harness absent | the part14 §4 ECC dry-run/guided install | a harness decision (Codex on the Windows box, or an explicit alternative) | `npx ecc-universal@2.2.3 install --guided --dry-run` per the §G.3 plan; metadata/license/shasum already captured (r326) |
| BLK-6 | **The rollback can wipe SYSTEM packages** (observed r331: `libEGL.so.1` gone — every PySide6.QtGui import fails, so the whole tests/ui collection errors) — no root apt in the sandbox | the GUI test census legs, any Qt import | the system package OR the user-mode workaround standing | the r331 mitigation: `apt-get download libegl1` + `dpkg -x` to `~/.local/lib/egl` + `LD_LIBRARY_PATH=~/.local/lib/egl` on the pytest invocations — the full Qt chain verified importable offscreen; re-applies after any recurrence |

## UNBLOCKED / MITIGATED this round (r331)

| Former blocker / queued item | How it cleared |
| --- | --- |
| The venv gutted by the twenty-first (deepest-class) rollback — 9+ feature modules missing AND the surviving core rolled back to stale pins (numpy 2.1.3, sklearn 1.9.1) | rebuilt pin-exact in three legs: the core + torch-cpu + feature set, then the numpy-1.x wheel class (pyarrow 16.1.0 / umap-learn 0.5.12 / mdtraj 1.10.3 — the latest builds are numpy-2-only), then the sklearn/scipy re-pin — the full stack verified importable, the API suite green over it (449+1s) |
| The mode wall: the rollback restored 101 source files with +x bits (invisible to git via `core.fileMode=false`, fatal to ruff EXE002) | the exec bits stripped per-file from the ruff evidence list — `ruff check src/ tests/ ui/` "All checks passed!" again |
| The r326 §D queued row: "vitest/tsc — node_modules wiped; re-install + re-run queued (§G)" | the one-window npm ci + vitest 119/119 (16 files) + tsc clean re-run this round |
| The CI verdict at the r330 HEAD (queued "closes asynchronously, checked at round close") | closed GREEN this round: `a6cc830b` **19/19 check-runs success** via the authenticated API |
| The libEGL system package (BLK-6's first strike) | the user-mode deb-extract workaround (above) — the census re-trued to **8,589 / 0 errors** with it |

## UNBLOCKED this round (r329)

| Former blocker | How it cleared |
| --- | --- |
| Push lane (PAT credential gone, 22+ rounds) | the owner re-supplied the token via the plain-text channel (the PDF channel had truncated at 74/93 chars — diagnosed, evidence `scratch/p14_r329_token_analysis.txt`); token API-validated as MrMMA2004 with push permission; the r328 payload pushed `ad5f00a7..a75f98eb` + the r329 hardening commit |
| CI verification at HEAD | the authenticated API now reachable — 19 check-runs live at both pushed commits (verdict closes asynchronously) |
| The machinery token-sweep hazard | first push attempt was REFUSED by GitHub push protection (a machinery UUID commit had swept `upload/GitHub_token.txt` into history); the ghost was reset away per convention and `upload/GitHub_token.*` + `.credentials/` gitignored disk-only — recurrence prevented |

## UNBLOCKED this round (r336) — BLK-1's installer-mechanics half, CI-executed

The owner's directive ("check if you can do the blocked items
through GitHub Actions"): a GitHub windows-2022 runner is a FRESH
Windows x64 machine on every run (immutable image, zero toolkit
residue), so the installer-mechanics half of BLK-1 now executes in
CI against the PUBLISHED release assets —
`.github/workflows/windows-install-smoke.yml` (registered at the
repository root, the round-293 rule; `contents: read` least
privilege; runs on every published release + manual dispatch):

* asset integrity — SHA256SUMS.txt verified against every
  downloaded asset + the RELEASE_MANIFEST cross-checked (version,
  the build commit == the tag's checkout, artifact byte counts);
* the Windows Defender scan of both installers (the antivirus leg,
  genuinely clean-machine; a detection fails the lane loudly, an
  inconclusive scan is recorded — never a silent pass);
* both silent installs (WiX MSI `/qn` per-machine; NSIS EXE `/S`
  currentUser) + the installed tree verified (host exe + version
  resource + the frozen sidecar + the Start-Menu shortcuts);
* the INSTALLED sidecar's live verification — the same
  handshake/health/guard/shutdown contract the release pipeline's
  step 13 pins on the build host, now on the bundle an end user
  receives (the §10 offline-launch leg's backend half);
* the app host launch — the sidecar child spawn + a bounded
  runner-scope resource sample (honestly NOT the desktop baseline)
  + the graceful close + the §10 leg-5 no-orphan check;
* both uninstalls + removal verification + the user-data retention
  state recorded;
* the v1.2.231 → v1.2.240 in-place upgrade with a user-data
  retention marker (the §10 leg-6 rule, file-marker form).

The first dispatch's verdict RODE the round's addendum and landed
there (the r334 convention — nine dispatches total, each early
failure a REAL catch fixed at the source, never a weakened gate):
**run 45086526069 (the ninth dispatch) GREEN END-TO-END — all 16
steps against the published v1.2.240 release assets on a fresh
windows-2022 runner**: SHA256SUMS verified (4 assets, zero mismatch,
zero unlisted) + the manifest cross-check (the build commit
`661e3f65` proven == the tag checkout) + Defender clean on BOTH
installers + the MSI silent install/tree/version + the INSTALLED
sidecar's live contract (handshake + authenticated health + the
refused guard + the graceful shutdown, `SIDECAR-VERIFIED`) + the
app host launch with the sidecar child spawn + the resource samples
+ the graceful closes + the zero-orphan checks on BOTH lanes + both
uninstalls + the v1.2.231 → v1.2.240 upgrade with the user-data
marker retained. The lane's own catches along the way (all real,
all fixed at source): the sumfile-coverage sweep, the GITHUB_SHA-
vs-checkout comparison, the pwsh parse hazards, the Cargo-name host
binary discovery, the NSIS async-uninstaller mechanics, and — the
REAL INSTALLER FINDINGS: (1) the v1.2.231-era MSI installs PER-USER
(`%LOCALAPPDATA%\PharmaML Studio`) while the v1.2.240 MSI installs
PER-MACHINE (`Program Files\PharmaML Studio`), so the upgrade is a
CROSS-SCOPE major upgrade that lands in the PREVIOUS per-user
location (the empirical verdict from the green run's evidence); (2)
the NSIS currentUser install home `%LOCALAPPDATA%\PharmaML Studio`
is ALSO the backend's default data dir (`APP_DIR_NAME`) — the two
COLLIDE, so the launched app legitimately writes `studio.db` +
`files/` into the install tree (falsifying the §10 leg-4
"user-data dir is never the install dir" claim for the NSIS lane —
the doc trued this round) and the §10 leg-6 retention accidentally
holds precisely because the data files are not in the uninstaller's
delete list (the honest re-derived semantics: the verdict checks the
APP FILES' removal; the user-data residue recorded, never counted as
app behavior). Still honestly BLOCKED even with the lane (recorded
in the lane's own closing note, so nothing converts to a pass by
omission): the GUI nine-workspace click-through (§10 leg 3 — no
human on the runner); the offline-network WebView2 bootstrap (the
runner is online + carries the WebView2 runtime, so the embedded
offline installer never fires); sleep/resume, DPI/multi-monitor,
GPU benchmarks, the 45–60-min desktop-stability lane at
owner-hardware fidelity; consumer Windows 10/11 edition fidelity
(the runner is Windows Server 2022); BLK-3 signing (the owner's
certificate) and BLK-4 (the owner's decision) untouched.

## Standing honest-status rules

- A blocked leg re-rides each round until its environment exists
  (the r317/r325 convention); the worklog records each re-ride.
- An unblocked lane converts to test rows in the matrices with named
  evidence — never retroactive status flips.
- The sandbox reaper class (root node_modules wiped between tool
  calls, observed 5× this round) is an environmental reliability
  constraint, NOT an app defect; it affects evidence collection
  (recovered per-occurrence via the one-window makeup) and is
  recorded here so its absences are never misread.

— seeded round 329.
