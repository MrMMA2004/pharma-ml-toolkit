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
