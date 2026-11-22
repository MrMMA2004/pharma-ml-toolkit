# AUTOMATED TEST MATRIX — part14 §13 deliverable #3

The part14 §13.1 master matrix instantiated with the honest current
fill: A = Automated (executed this round or standing with named
evidence), I = Integration, E = E2E, M = Manual, W = Windows,
P = Performance, S = Security. `✓` = has coverage (evidence named);
`B` = BLOCKED (environment); `—` = not yet covered (queued, not
fabricated).

Re-pinned: round 329 (2026-11-22) — the trusted-origin/API legs at
`scratch/p14_r329_*`, vitest/tsc at the same prefix, census
8,589/471/0 (r328 byte-parity), CI live at a75f98eb/013b3ff4.

| Component | A | I | E | M | W | P | S |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tauri desktop | — | B | B | B | B | B | B |
| Rust backend manager | B* | B | B | B | B | B | ✓¹ |
| FastAPI | ✓² | ✓ | — | — | B | — | ✓³ |
| REST client | ✓ | ✓ | — | — | B | — | ✓ |
| WebSocket | ✓⁴ | — | B | B | B | — | ✓⁴ |
| SQLite | ✓ | ✓ | — | — | B | — | ✓ |
| FileVault | ✓ | ✓ | — | — | B | — | ✓ |
| Datasets | ✓ | ✓ | — | — | B | B | — |
| Machine Learning | ✓ | ✓ | — | — | B | B | — |
| Deep Learning | ✓ | ✓ | — | — | B | B | — |
| Generative AI | ✓ | ✓ | — | — | B | B | — |
| Molecular Docking | ✓ | ✓ | — | — | B | B | — |
| PK/PD | ✓ | ✓ | — | — | B | B | — |
| Molecular Dynamics | ✓ | ✓ | — | — | B | B | — |
| Plotly.js | ✓ | — | ✓⁵ | — | B | B | — |
| Three.js | ✓ | — | — | — | B | B | — |

¹ Rust `#[cfg(test)]` suites WRITTEN (D-4/D-5, 11 cases) — `cargo
test` BLOCKED in this sandbox; never marked passed (§16.14).
² `tests/api/` 449 passed + 1 skipped, 0 failed (r329, 107.05s).
³ The 34-leg trusted-origin + CORS + Host + token matrix
(`test_trusted_origins.py`, 34/34 r329) + the r327 probe set.
⁴ WS token handshake source-verified (close 4401; §3.1-item-8 audit);
D-6 origin policy OPEN in the register.
⁵ Web-console E2E via agent-browser (r329: render/interaction/footer
legs, zero page/console errors) — the studio-frontend E2E rides the
desktop box.

**Standing totals**: collection 8,589 tests / 471 files / 0 errors
(r329, byte-parity with r328); vitest 119/119 (16 files); tsc clean;
ruff/mypy clean on the fix surface; CI 19 jobs live at the pushed
HEAD.

**Known gaps (honest)**: desktop E2E + Windows legs all BLOCKED (the
box); performance numbers only exist as NOT-MEASURED placeholders
(§16.15 forbids fabrication — see `PERFORMANCE_BASELINES.md`).

— seeded round 329.
