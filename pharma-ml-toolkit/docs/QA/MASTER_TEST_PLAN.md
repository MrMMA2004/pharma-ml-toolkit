# MASTER TEST PLAN — part14 §13 deliverable #1

The part14 §5 four-level testing program, instantiated against the
actual PharmaML Studio surfaces (A: Python toolkit + FastAPI + legacy
PySide6 `studio.py`; B: React/Tauri 2 desktop; C: the root web console).
Supersedes nothing — this is the plan layer; status lives in the
matrixes below it.

- **Seeded**: 2026-11-22 (round 329, the 121st recorded run).
- **Program rule** (part14 §5): every test carries ID / category /
  description / preconditions / procedure / expected / actual /
  PASS-FAIL-BLOCKED-NOT-APPLICABLE / evidence / defect-ref /
  environment / date / version. Unexecuted manual tests are NEVER
  marked PASS (§16.14).
- **Version of record**: v1.2.240 (the six carriers reconciled, r325).

## Level 1 — Automated tests (green at r329, evidence re-pinned)

| Leg | Scope | Status (r329) | Evidence |
| --- | --- | --- | --- |
| API regression | `tests/api/` — the FastAPI surface incl. the D-1/D-2 trusted-origin + CORS matrix | **449 passed + 1 skipped, 0 failed** (107.05s) | `scratch/p14_r329_api_suite_run.txt` |
| Focused security regression | `tests/api/test_trusted_origins.py` (34 legs: P1–P8 + lookalikes + preflight-then-401) | **34 passed** (5.67s) | `scratch/p14_r329_trusted_origins_run.txt` |
| Collection census | full tree | **8,589 tests / 471 files / 0 collection errors** (r328 byte-parity: per-file counts diff-clean) | `scratch/p14_r329_collection_census.txt` |
| Studio frontend unit/component | `frontend/` vitest (incl. the 9-test D-3 startup-classification suite + the 6 StartupGate tests) | **119 passed / 16 files** | `scratch/p14_r329_frontend_vitest_run.txt` |
| Type checking | `frontend/` tsc --noEmit | **clean (0 output, exit 0)** | `scratch/p14_r329_frontend_tsc_run.txt` |
| Lint + types (Python) | ruff 0.17.1 / mypy 2.4.0 on the changed fix surface | **clean** | `scratch/p14_r329_{ruff,mypy}_run.txt` |
| CI (the 19-job matrix) | on every push | **live at a75f98eb + 013b3ff4** (verdict closes asynchronously; checked at round close) | GitHub check-runs API, authenticated |
| Rust host unit | `src-tauri` `#[cfg(test)]` (D-4/D-5: 11 malformed-handshake cases) | **BLOCKED — no cargo/rustc in this sandbox** (written, never marked passed; §16.14) | `DEFECT_REGISTER.md` rows D-4/D-5 |

## Level 2 — Desktop and packaging tests

`tauri dev` / `tauri build` / packaged app / process lifecycle /
installers / clean-install legs: **BLOCKED** — the sandbox is Linux,
the app targets Windows WebView2. See `WINDOWS_TEST_MATRIX.md` and
`BLOCKED_TESTS_AND_SOLUTIONS.md`. The compile-verifiable source legs
(handshake validation + redaction) are written at source and ride the
first cargo-enabled environment.

## Level 3 — Manual expert testing

The §6 checklist instantiated in `MANUAL_TEST_CHECKLIST.md`. The
browser-executable subset (web console render/interaction/footer) is
verified per round (r329: zero page/console errors, ML-card
interaction live, footer at content-flow bottom — geometry measured:
footer bottom == doc height exactly). The desktop legs ride the
Windows box.

## Level 4 — Long-duration and environmental testing

45–60-min stability, memory/CPU under heavy scientific workloads,
offline behavior, Windows-version spread, antivirus compatibility:
**BLOCKED** on the physical Windows environment (unchanged from the
r325/r326 lanes). No fabricated numbers exist here by rule (§16.15).

## The honest-status convention

- **PASS** requires this-round or standing evidence named per row.
- **BLOCKED** names the missing environment and the lowest-cost
  unlock (§9.5 clean-VM procedure where applicable).
- Standing evidence (`PART14_PHASE1_AUDIT.md` §D, the r326–r328
  files) is cited by path; nothing converts citation into execution.

— seeded round 329; maintained per round alongside the register.
