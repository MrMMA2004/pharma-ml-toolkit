# BLOCKED TESTS AND SOLUTIONS — part14 §13 deliverable #7

The consolidated blocked-test ledger: every blocked leg, why, the
required environment, and the lowest-cost unlock. Nothing blocked is
ever reported as PASS; nothing unblocks silently.

Seeded: round 329 (2026-11-22).

## Blocker classes

| ID | Blocker | Blocks | Required environment | Lowest-cost solution |
| --- | --- | --- | --- | --- |
| BLK-1 | No Windows in the dev sandbox | the §6.1 desktop checklist; clean-machine smoke; installer install/upgrade/uninstall; offline WebView2; sleep/resume; DPI/multi-monitor; 45–60-min stability; GPU benchmarks; antivirus | physical Windows 10/11 (VM acceptable for most legs per §9.5) | the §9.5 clean-VM procedure (snapshot → install → probe → assert → collect) |
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
