# RELEASE ACCEPTANCE REPORT — part14 §13 deliverable #8

The part14 §14 acceptance gates, honestly stated at round 329
(2026-11-22); the r331 addendum re-pins G/D on this round's
evidence. Supersedes the audit's §H snapshot (r326): Gate B has
moved since (the incident fixed at source + regression-pinned; the
packaged-app legs still ride the box).

## Gate-by-gate

| Gate | Status | Basis |
| --- | --- | --- |
| A — Repository integrity | **PASS (this round)** | HEAD verified; the r328 payload + r329 hardening pushed to origin/main (`ad5f00a7..a75f98eb..013b3ff4`); the token-sweep ghost reset away (reflog-resident, never pushed — push protection proved it); tree understood (8 intended modifications + documented untracked residue); version carriers consistent at 1.2.240; the QA deliverable set now reconciled with the implementation |
| B — Backend and security | **PARTIAL (upgraded from FAIL)** | "Tauri origins handled correctly" — TRUE at source + regression level now (D-1/D-2 fixed, 34/34 pinned, r329 re-pin; D-3 honesty fixed, 9 tests; D-4/D-5 source-fixed, compile-blocked). Authentication/loopback-bind verified. The PACKAGED-app legs (real WebView2 round-trip, process lifecycle) remain BLOCKED on the box — the gate does not close without them |
| C — Frontend and scientific | **PARTIAL** | vitest 119/119 + tsc clean re-pinned; web console E2E green (render/title/interaction/footer, zero page/console errors, pins pair r329); the nine-workspace desktop legs ride the box |
| D — Automated tests | **PASS (the sandbox scope)** | tests/api 449+1s 0 failed (r331 re-pin: 108.88s, `p14_r331_api_suite_run2.txt`); census 8,589/0 errors (r331, `--doctest-modules` re-trued); trusted-origins 34/34; ruff clean + mypy clean (r331, after the mode-wall restoration); vitest 119/119 + tsc clean; CI 19 jobs live at the pushed HEAD (verdict closed GREEN at r331: a6cc830b 19/19); the Rust unit legs stay compile-blocked (§16.14) |
| E — Performance and stability | **BLOCKED** | unchanged — the 45–60-min + resource lanes need the physical box; no numbers fabricated (PERFORMANCE_BASELINES.md) |
| F — Windows release | **BLOCKED** | the three standing lanes (clean-machine smoke, Stage-5 PySide6 retirement, installer signing) |
| G — Release integrity | **PARTIAL (basis re-pinned r331)** | release homes at the CURRENT terminal bytes (v1.2.240 / 23,489,390 B / sha-256 `0fadb6e5…`, the r329 same-version re-ship, six-home parity + offsite exact) + live exact:true zip-size API re-verified r330 (×3) and r331; the CI verdict queue CLOSED — a6cc830b 19/19 check-runs success (checked r331), the r329 pair closed GREEN at r330; SBOM/pins standing from r324–r325 |

## The honest verdict

No production-ready declaration: gates E/F remain BLOCKED on
owner-side environments; B/G close only with the box legs; the Rust
suites compile-blocked. The release lanes ride each continuation
until the owner-side gates clear (the standing convention — no
premature publish, no silent version bump; the shipped fixed point
stays v1.2.231 with its assets intact).

## What changed this round (the delta log)

- The credential lane restored → **the first successful pushes of
  the part14 era** (the r328 payload + the r329 hardening).
- The QA deliverable set seeded (§13's ten documents: this set +
  PART14_PHASE1_AUDIT + DEFECT_REGISTER + the consolidated execution
  report embedded in the worklog round entry).
- Every sandbox-executable gate re-pinned at the arrival; the census
  re-proven at r328 byte-parity.

## Round 331 addendum — the queued items closed

- The CI verdict queue **closed**: the r330 terminal `a6cc830b` —
  **19/19 check-runs success** (checked this round via the
  authenticated API); the r329 pair had closed GREEN at r330.
- The **twenty-first arrival (deepest-class rollback: history reset
  to the r186 checkpoint, origin detached, PAT + venv gutted, the
  Studio frontend wiped, system `libEGL.so.1` gone)** recovered in
  full before any gate work: origin re-attached, reset onto
  `a6cc830b`, the ghost pair + the old-layout `.github` duplicate +
  the stale counter copy swept, the venv rebuilt pin-exact (the
  numpy-1.x wheel class documented), the mode wall (101 files'
  rollback +x bits) restored — ruff clean again.
- Every gate re-pinned at r329/r330 parity on the rebuilt stack
  (trusted-origins 34/34, tests/api 449+1s, census 8,589/0, vitest
  119/119, tsc clean, ruff/mypy clean, the console + zip-size API
  exact at 23,489,390 B, the browser E2E zero errors + footer
  geometry exact + the r331 pins pair).

— recorded round 329; addendum round 331; nothing above converts
absence of evidence into a pass (§16.14/§16.15).
